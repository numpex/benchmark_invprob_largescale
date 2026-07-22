from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import torch

from toolsbench.profiler.base import BenchProfiler


class NvidiaProfiler(BenchProfiler):
    """Labels sections of code with NVTX markers — ``track_step(name)`` for
    any named section the caller chooses (e.g. ``"forward"``, ``"backward"``,
    ``"denoise"``, ...), plus one ``"iter_N"`` marker per iteration. It does
    not measure or record anything itself; the labels are inert unless
    NVIDIA's ``nsys`` tool is wrapped around the process, in which case they
    become a real timeline (exact start/end of each named section, each
    iteration, plus GPU busy/idle detail from ``nsys`` itself) instead of a
    single aggregate number. Without ``nsys`` attached, this reports the same
    ``total_time_sec``/``max_gpu_mb`` to benchopt as ``CustomProfiler``, just
    with unused, free labels attached.

    Set ``profiler_mode: [nvidia]`` in a benchopt solver config to enable it.

    Wrapping the outer ``benchopt run`` command in ``nsys profile`` (the
    obvious first thing to try) captures nothing useful with the
    ``submitit`` backend: that command only submits a job via ``sbatch`` and
    polls for completion — the actual GPU training runs later, in a separate
    process on a different node, invisible to ``nsys``. Use
    ``--parallel-config benchmark_training/configs/config_parallel_nsys.yml``
    instead — its ``slurm_python`` setting inserts ``nsys`` directly into the
    real worker process, no manual wrapping needed.

    For a local, non-SLURM run, wrap the command directly, e.g.::

        nsys profile -o my_run --force-overwrite=true --trace=cuda,nvtx \\
            --trace-fork-before-exec=true --cuda-memory-usage=true --sample=none \\
            <your normal run command>

    Either way, inspect the resulting ``my_run.nsys-rep`` afterwards —
    either in the Nsight Systems GUI, or via
    ``nsys stats --report nvtx_sum my_run.nsys-rep`` /
    ``--report cuda_gpu_kern_sum`` for a text summary.

    Parameters
    ----------
    warmup : int
        Number of iterations to skip before recording starts (no NVTX
        markers or metrics are produced during warmup).
    active : int
        Number of iterations to record after warmup (0 = all remaining).
    """

    def __init__(
        self,
        device,
        name: str,
        warmup: int = 0,
        active: int = 0,
        save_file: bool = False,
    ):
        self._device = torch.device(device) if isinstance(device, str) else device
        self._name = name
        self._has_cuda = torch.cuda.is_available() and self._device.type == "cuda"
        self._warmup = warmup
        self._active = active
        self._save_file = save_file
        self._iter_count: int = 0
        self._all_results: list[dict] = []
        self._current_metrics: dict = {}
        self._iter_t0: float = 0.0
        self._cuda_profiler_started = False
        self._iter_range_open = False

    def _is_recording(self) -> bool:
        if self._iter_count < self._warmup:
            return False
        if self._active > 0 and self._iter_count >= self._warmup + self._active:
            return False
        return True

    def _start_cuda_profiler(self):
        if self._has_cuda and not self._cuda_profiler_started:
            torch.cuda.cudart().cudaProfilerStart()
            self._cuda_profiler_started = True

    def _stop_cuda_profiler(self):
        if self._has_cuda and self._cuda_profiler_started:
            torch.cuda.cudart().cudaProfilerStop()
            self._cuda_profiler_started = False

    def _push_iter_range(self):
        if self._has_cuda and self._is_recording():
            torch.cuda.nvtx.range_push(f"iter_{self._iter_count}")
            self._iter_range_open = True

    def _pop_iter_range(self):
        if self._has_cuda and self._iter_range_open:
            torch.cuda.nvtx.range_pop()
            self._iter_range_open = False

    def __enter__(self):
        self._all_results = []
        self._current_metrics = {}
        self._iter_count = 0
        if self._warmup == 0:
            self._start_cuda_profiler()
        self._push_iter_range()
        self._iter_t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._pop_iter_range()
        self._stop_cuda_profiler()

    @contextmanager
    def track_step(self, name: str):
        recording = self._has_cuda and self._is_recording()
        if recording:
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if recording:
                torch.cuda.nvtx.range_pop()

    def end_iteration(self, ctx=None):
        if self._has_cuda:
            torch.cuda.synchronize(self._device)
        total_time = time.perf_counter() - self._iter_t0
        max_gpu = (
            torch.cuda.max_memory_allocated(self._device) / 1024**2
            if self._has_cuda
            else 0.0
        )

        self._pop_iter_range()

        if self._is_recording():
            captured = {
                "total_time_sec": round(total_time, 6),
                "max_gpu_mb": round(max_gpu, 1),
            }
            self._all_results.append(captured)
            self._current_metrics = captured

        self._iter_count += 1

        if self._iter_count == self._warmup:
            self._start_cuda_profiler()
        elif self._active > 0 and self._iter_count == self._warmup + self._active:
            self._stop_cuda_profiler()

        if self._has_cuda:
            torch.cuda.reset_peak_memory_stats(self._device)

        if ctx is not None and getattr(ctx, "use_dist", False):
            ctx.barrier()

        self._push_iter_range()
        self._iter_t0 = time.perf_counter()

    def get_current_metrics(self) -> dict:
        return self._current_metrics

    def finalize(self, ctx) -> None:
        self._pop_iter_range()
        self._stop_cuda_profiler()
        if not self._all_results or not self._save_file:
            return
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self._name}_gpu_metrics.csv"
        pd.DataFrame(self._all_results).to_csv(path, index=False)
        print(f"[profiler] Saved {len(self._all_results)} records to {path}")
