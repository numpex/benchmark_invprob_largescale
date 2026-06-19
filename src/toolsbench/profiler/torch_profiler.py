"""Profiler backed by torch.profiler.

Background — the raw torch.profiler data this module consumes
============================================================
After a profiling cycle, torch.profiler exposes the captured data two ways,
and everything below is built on top of these:

1. ``prof.key_averages()`` -> flat list of aggregated events, one+ per label.
   Each entry has:
     - ``key``                    : the op/section name (e.g. "aten::mm",
                                     "gradient", "ProfilerStep*")
     - ``cpu_time_total``         : total CPU time (us) for this key
     - ``device_time_total``      : total GPU time (us) for this key
     - ``count``                  : number of calls
     - ``is_user_annotation``     : True if the key is one of OUR
                                    record_function() labels (a "section"),
                                    False for raw aten/cuda ops.
   NOTE: every record_function label appears TWICE here — once as a CPU-view
   entry and once as a CUDA-view entry. _group_by_key() merges the pair.

2. ``prof.events()`` -> the raw event TREE (not aggregated). Each event has:
     - the same fields as above, plus
     - ``cpu_children``           : list of child events nested under it
     - ``self_device_time_total`` : GPU time excluding children
     - ``device_memory_usage`` / ``self_device_memory_usage`` (bytes)
   Our record_function sections are the ``is_user_annotation=True`` nodes; the
   ops they ran are their ``cpu_children``. We walk this tree to attribute ops
   and communication time to the section that contains them.

All times from torch are in MICROSECONDS; we divide by 1e6 for seconds and
memory by 1024**2 for MB.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import torch

from toolsbench.profiler.base import BenchProfiler


# Substrings that identify distributed communication ops (NCCL, Gloo, c10d).
_COMM_SUBSTRINGS = (
    "c10d::", "nccl",
    "all_gather", "all_reduce", "allgather", "allreduce",
    "gloo::", "broadcast",
)


def _is_comm_op(key: str) -> bool:
    """Return True if *key* is a distributed communication op, False otherwise."""
    k = key.lower()
    if k.startswith("aten::"):
        return False
    return any(s in k for s in _COMM_SUBSTRINGS)


def _group_by_key(avgs):
    """Merge CPU-view/CUDA-view duplicates from key_averages() into one entry per key.

    torch.profiler emits two entries per record_function label: a CPU-view and a
    CUDA-view. Merging rules (proven empirically):
      - cpu_time : max() — CUDA-view always has cpu_time=0, CPU-view has the real value.
      - dev_time : min() — CPU-view may sum all descendant kernel durations (inflated);
                           CUDA-view measures actual GPU wall time (closer to ground truth).
      - count    : min() — CPU-view sometimes over-counts internal bookkeeping calls.
    """
    grouped = defaultdict(
        lambda: {"cpu_time": 0.0, "dev_time": float("inf"),
                 "count": 10**9, "is_user": False}
    )
    for evt in avgs:
        g = grouped[evt.key]
        g["cpu_time"] = max(g["cpu_time"], evt.cpu_time_total)
        g["dev_time"] = min(g["dev_time"], evt.device_time_total)
        g["count"]   = min(g["count"],   evt.count)
        g["is_user"] = g["is_user"] or evt.is_user_annotation
    # replace sentinels with 0 for keys that only appeared in one view
    for g in grouped.values():
        if g["dev_time"] == float("inf"):
            g["dev_time"] = 0.0
        if g["count"] == 10**9:
            g["count"] = 0
    return grouped


def _comm_sec_in_subtree(event) -> float:
    """Recursively sum device time (sec) of communication ops under *event*."""
    total = 0.0
    for child in getattr(event, "cpu_children", []):
        if _is_comm_op(child.key):
            total += child.device_time_total / 1e6
        else:
            total += _comm_sec_in_subtree(child)
    return total


def _per_step_comm(events, section_names) -> dict:
    """Total comm time (sec) within each user section, summed over the event tree."""
    comm: dict = {name: 0.0 for name in section_names}
    for e in events:
        if e.is_user_annotation and e.key in comm:
            comm[e.key] += _comm_sec_in_subtree(e)
    return comm


def _section_summary(avgs, events=None) -> dict:
    """Return per-iteration per-section benchopt metrics: ``{sec}_cuda_sec``,
    ``{sec}_cpu_sec``, ``{sec}_comm_sec``, ``comm_cuda_sec``.

    Called once per profiler cycle (one iteration), so values are already per-iteration.
    """
    grouped = _group_by_key(avgs)
    out: dict = {}
    section_names = []
    comm_cuda_sec = 0.0
    for key, g in grouped.items():
        if g["count"] <= 0 or key.startswith("ProfilerStep"):
            continue
        if g["is_user"] and not _is_comm_op(key):
            # user-annotated section (gradient, denoise, ...): emit cuda/cpu time
            out[f"{key}_cuda_sec"] = round(g["dev_time"] / 1e6, 6)
            out[f"{key}_cpu_sec"]  = round(g["cpu_time"] / 1e6, 6)
            section_names.append(key)
        elif _is_comm_op(key):
            # accumulate iteration-level communication total
            comm_cuda_sec += g["dev_time"] / 1e6
    if events is not None and section_names:
        # per-section comm time: walk the raw event tree to attribute comm to its section
        for name, comm_sec in _per_step_comm(events, section_names).items():
            out[f"{name}_comm_sec"] = round(comm_sec, 6)
    out["comm_cuda_sec"] = round(comm_cuda_sec, 6)
    return out


def _collect_op_rows(events) -> dict:
    """Accumulate direct child ops of each user-annotated section from the event tree.
    Result: {(section, op): {cpu_sec, cuda_sec, self_cuda_sec, mem_mb, self_mem_mb, count}}
    """
    agg: dict = {}
    for e in events:
        if not getattr(e, "is_user_annotation", False):
            continue
        if e.key.startswith("ProfilerStep") or _is_comm_op(e.key):
            continue
        children = getattr(e, "cpu_children", None)
        if not children:  # skip CUDA-view duplicate (carries no children)
            continue
        for ch in children:
            slot = agg.setdefault(
                (e.key, ch.key),
                {"cpu_sec": 0.0, "cuda_sec": 0.0, "self_cuda_sec": 0.0,
                 "mem_mb": 0.0, "self_mem_mb": 0.0, "count": 0},
            )
            slot["cpu_sec"]       += ch.cpu_time_total / 1e6
            slot["cuda_sec"]      += ch.device_time_total / 1e6
            slot["self_cuda_sec"] += ch.self_device_time_total / 1e6
            slot["mem_mb"]        += ch.device_memory_usage / 1024**2
            slot["self_mem_mb"]   += ch.self_device_memory_usage / 1024**2
            slot["count"]         += ch.count
    return agg


def _op_rows_to_records(agg: dict, iter_label) -> list[dict]:
    """Convert _collect_op_rows output to tidy CSV rows tagged with iter_label."""
    return [
        {
            "iter": iter_label, "section": section, "op": op,
            "cpu_sec": round(g["cpu_sec"], 6), "cuda_sec": round(g["cuda_sec"], 6),
            "self_cuda_sec": round(g["self_cuda_sec"], 6),
            "mem_mb": round(g["mem_mb"], 4), "self_mem_mb": round(g["self_mem_mb"], 4),
            "count": g["count"],
        }
        for (section, op), g in agg.items()
    ]


class TorchProfiler(BenchProfiler):
    """Profiler backed by torch.profiler.profile.

    Wrapper lifecycle
    -----------------
    Use as a context manager around the iterative loop::

        with TorchProfiler(device, name, warmup=2, active=5) as prof:
            for i in range(n_iter):
                with prof.track_step("gradient"):  # annotate a named sub-step
                    ...
                with prof.track_step("denoise"):
                    ...
                prof.end_iteration()               # advance profiler, record timing
                metrics = prof.get_current_metrics()  # consumed by benchopt get_result()
        prof.finalize(ctx)                         # write CSV / Chrome trace

    ``track_step(name)`` wraps the block in ``torch.profiler.record_function(name)``,
    stamping a user annotation on both CPU and CUDA timelines. Each named block becomes
    a "section" in the output metrics and CSV.

    ``end_iteration()`` calls ``prof.step()`` to advance the internal torch schedule,
    syncs the GPU for accurate wall-clock timing, and snapshots peak GPU memory. It
    populates ``_current_metrics`` only for iterations inside [warmup, warmup+active).

    Operating modes
    ---------------
    per_step=True (default)
        One torch.profiler cycle per iteration — the profiler is reset and re-read at
        every ``end_iteration()`` via an ``on_trace_ready`` callback. Gives real
        per-iteration values in both CSV and benchopt metrics. trace_dir is unsupported.

    per_step=False
        torch handles the warmup/active schedule natively; the profiler runs once across
        the whole active window. CSV rows use ``iter="agg"`` (window totals). Benchopt
        metrics (total_time_sec, max_gpu_mb) remain per-iteration. Supports trace_dir.

    Outputs
    -------
    CSV  (``outputs/<name>_gpu_metrics.csv``)
        Tidy rows keyed by (iter, section, op) with cpu_sec, cuda_sec, self_cuda_sec,
        mem_mb, self_mem_mb, count — one row per (section, child-op) pair per iteration.

    Benchopt metrics (``get_current_metrics()``)
        ``{section}_cuda_sec``, ``{section}_cpu_sec``, ``{section}_comm_sec``,
        ``comm_cuda_sec``, ``total_time_sec``, ``max_gpu_mb``.

    warmup/active bound the profiling window; outside it the run executes at full speed.
    trace_dir raises ValueError with per_step=True (profiler is reset each iteration,
    so a full Chrome trace cannot be captured in that mode).

    Parameters
    ----------
    device, name, warmup, active, trace_dir, per_step, repeat : see create_profiler().
    """

    def __init__(self, device, name: str, warmup: int = 0, active: int = 0,
                 trace_dir: str | None = None, per_step: bool = True, repeat: int = 1):
        self._device = torch.device(device) if isinstance(device, str) else device
        self._name = name
        self._warmup = warmup
        self._active = active
        self._trace_dir = None if (trace_dir is None or trace_dir == "None") else trace_dir
        self._per_step = per_step
        self._repeat = repeat
        if self._trace_dir is not None and self._per_step:
            # per_step=True resets the profiler every iteration, so only the final
            # cycle survives — a full Chrome trace can't be captured in that mode.
            raise ValueError(
                "trace_dir is only supported with per_step=False; "
                "per_step=True resets the profiler each iteration, so a full "
                "trace cannot be captured."
            )
        self._has_cuda = torch.cuda.is_available() and self._device.type == "cuda"
        self._reset()

    def _reset(self):
        self._iter_count = 0
        self._all_op_rows: list[dict] = []
        self._current_metrics: dict = {}
        # temporary buffers: on_trace_ready stores data here; end_iteration reads it.
        # needed because on_trace_ready is a callback (can't return values) and torch
        # clears events immediately after the callback returns.
        self._pending_summary: dict = {}
        self._pending_ops: dict = {}
        self._iter_t0: float = 0.0
        self._prof = None
        self._started = False
        self._stopped = False

    def _on_trace_ready(self, prof):
        """per_step callback: capture this cycle's section summary + per-op detail."""
        events = prof.events()
        self._pending_summary = _section_summary(prof.key_averages(), events)
        self._pending_ops = _collect_op_rows(events)

    def _start_profiler(self):
        if self._started:
            return
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self._has_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if self._per_step:
            # one cycle per iteration; warmup/active handled manually via lazy start/stop
            sched = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=0)
            on_ready = self._on_trace_ready
        else:
            # torch schedule owns warmup/active; active=0 in our API means "all remaining"
            active = self._active if self._active > 0 else 10**9
            sched = torch.profiler.schedule(wait=0, warmup=self._warmup, active=active, repeat=self._repeat)
            on_ready = None
        self._prof = torch.profiler.profile(
            activities=activities, schedule=sched, on_trace_ready=on_ready,
            profile_memory=True, record_shapes=False, with_stack=False, with_flops=False,
        )
        if self._has_cuda:
            torch.cuda.reset_peak_memory_stats(self._device)
        self._prof.__enter__()
        self._started = True

    def _stop_profiler(self):
        if self._started and not self._stopped:
            self._prof.__exit__(None, None, None)
            self._stopped = True

    def __enter__(self):
        self._reset()
        # per_step=False: start immediately — torch schedule owns warmup/active
        # per_step=True:  start lazily after warmup iterations (manual control)
        if not self._per_step or self._warmup == 0:
            self._start_profiler()
        self._iter_t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._stop_profiler()

    @contextmanager
    def track_step(self, name: str):
        """Annotate a named sub-step; torch.profiler captures the true device timeline."""
        with torch.profiler.record_function(name):
            yield

    def end_iteration(self):
        # sync GPU so wall-clock time includes GPU execution, then measure peak memory
        if self._has_cuda:
            torch.cuda.synchronize(self._device)
        total_time_sec = time.perf_counter() - self._iter_t0
        max_gpu = torch.cuda.max_memory_allocated(self._device) / 1024**2 if self._has_cuda else 0.0

        # step() closes the current profiler cycle; for per_step=True this fires
        # on_trace_ready, which populates _pending_summary and _pending_ops
        if self._started and not self._stopped:
            self._prof.step()

        if self._warmup <= self._iter_count and \
                (self._active == 0 or self._iter_count < self._warmup + self._active):
            base = {"total_time_sec": round(total_time_sec, 6), "max_gpu_mb": round(max_gpu, 1)}
            if self._per_step:
                base.update(self._pending_summary)
                self._all_op_rows.extend(_op_rows_to_records(self._pending_ops, self._iter_count))
            self._current_metrics = base

        self._pending_summary = {}
        self._pending_ops = {}
        # reset peak so next iteration's max_gpu_mb reflects only that iteration
        if self._has_cuda:
            torch.cuda.reset_peak_memory_stats(self._device)
        self._iter_count += 1

        if self._per_step:
            # lazy start: profiler created only once warmup iters have elapsed
            if not self._started and self._iter_count >= self._warmup:
                self._start_profiler()
            # stop after the active window so remaining iters run unprofiled
            elif self._active > 0 and self._started and not self._stopped \
                    and self._iter_count >= self._warmup + self._active:
                self._stop_profiler()

        self._iter_t0 = time.perf_counter()

    def get_current_metrics(self) -> dict:
        return self._current_metrics

    def finalize(self, ctx) -> None:
        self._stop_profiler()
        if self._prof is None:
            return

        if not self._per_step:
            # read once after the cycle closes — key_averages() contains only the
            # active window (torch schedule discards warmup data internally)
            agg = _collect_op_rows(self._prof.events())
            if agg:
                self._all_op_rows = _op_rows_to_records(agg, "agg")

        # trace export is independent of the per-op CSV rows (per_step=False only)
        if self._trace_dir is not None:
            rank = ctx.rank if ctx is not None else 0
            os.makedirs(self._trace_dir, exist_ok=True)
            trace_path = os.path.join(self._trace_dir, f"rank_{rank}.pt.trace.json")
            self._prof.export_chrome_trace(trace_path)
            print(f"[profiler] Chrome trace saved to {trace_path}")

        if not self._all_op_rows:
            return

        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self._name}_gpu_metrics.csv"
        pd.DataFrame(self._all_op_rows).to_csv(path, index=False)
        mode = "per_step" if self._per_step else "aggregate"
        print(f"[profiler] ({mode}) saved {len(self._all_op_rows)} op rows to {path}")
