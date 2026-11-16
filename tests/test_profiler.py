from types import SimpleNamespace

import pytest
import torch

from toolsbench.profiler import (
    create_profiler,
    NullProfiler,
    CustomProfiler,
    TorchProfiler,
    NvidiaProfiler,
)
from toolsbench.profiler.torch_profiler import _group_by_key

# CPU always; CUDA too when present. GitHub CI (ubuntu-latest) is CPU-only.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _run_torch(p, n_iters, device="cpu"):
    """Drive a TorchProfiler over n_iters, mimicking the PnP solver loop."""
    with p:
        for _ in range(n_iters):
            with p.track_step("gradient"):
                a = torch.randn(64, 64, device=device)
                _ = a @ a
            p.end_iteration()


# ---------------------------------------------------------------------------
# NullProfiler
# ---------------------------------------------------------------------------


class TestNullProfiler:
    def test_full_interface(self):
        """Entire interface runs without error and returns empty metrics."""
        p = NullProfiler()
        with p:
            with p.track_step("step1"):
                pass
            p.end_iteration()
        assert p.get_current_metrics() == {}
        p.finalize(None)


# ---------------------------------------------------------------------------
# CustomProfiler — recording window logic
# ---------------------------------------------------------------------------


class TestCustomProfilerRecordingWindow:
    def test_enter_resets_state(self):
        p = CustomProfiler(device="cpu", name="test")
        p._all_results = [{"x": 1}]
        p._current_metrics = {"x": 1}
        p._iter_count = 5
        with p:
            pass
        assert p._all_results == []
        assert p._current_metrics == {}
        assert p._iter_count == 0

    def test_warmup_skips_first_n_iterations(self):
        p = CustomProfiler(device="cpu", name="test", warmup=2)
        with p:
            for _ in range(4):
                with p.track_step("grad"):
                    pass
                p.end_iteration()
        assert len(p._all_results) == 2

    def test_active_stops_after_n_iterations(self):
        p = CustomProfiler(device="cpu", name="test", warmup=0, active=2)
        with p:
            for _ in range(5):
                with p.track_step("grad"):
                    pass
                p.end_iteration()
        assert len(p._all_results) == 2

    def test_track_step_when_not_recording_skips_metrics(self):
        """track_step during warmup must yield without populating _step_metrics."""
        p = CustomProfiler(device="cpu", name="test", warmup=1)
        with p:
            with p.track_step("grad"):
                pass
            assert p._step_metrics == {}
            p.end_iteration()
        assert p._all_results == []


# ---------------------------------------------------------------------------
# CustomProfiler — metric content and CSV output
# ---------------------------------------------------------------------------


class TestCustomProfilerMetrics:
    def test_track_step_records_time(self):
        p = CustomProfiler(device="cpu", name="test")
        with p:
            with p.track_step("gradient"):
                pass
            p.end_iteration()
        metrics = p.get_current_metrics()
        assert "gradient_time_sec" in metrics
        assert metrics["gradient_time_sec"] >= 0.0

    def test_end_iteration_stores_total_and_gpu(self):
        p = CustomProfiler(device="cpu", name="test")
        with p:
            with p.track_step("step"):
                pass
            p.end_iteration()
        metrics = p.get_current_metrics()
        assert "total_time_sec" in metrics
        assert "max_gpu_mb" in metrics

    def test_finalize_writes_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = CustomProfiler(device="cpu", name="myrun", save_file=True)
        with p:
            with p.track_step("grad"):
                pass
            p.end_iteration()
        p.finalize(None)
        csv_path = tmp_path / "outputs" / "myrun_gpu_metrics.csv"
        assert csv_path.exists()
        import pandas as pd

        df = pd.read_csv(csv_path)
        assert "total_time_sec" in df.columns
        assert len(df) == 1

    def test_finalize_no_op_when_no_results(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = CustomProfiler(device="cpu", name="myrun")
        with p:
            pass
        p.finalize(None)
        assert not (tmp_path / "outputs" / "myrun_gpu_metrics.csv").exists()

    def test_memory_snapshot_is_logged_and_added_to_metrics(self, monkeypatch, capsys):
        mib = 1024**2
        p = CustomProfiler(device="cpu", name="memory-test")
        p._has_cuda = True
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 10 * mib)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 12 * mib)
        monkeypatch.setattr(
            torch.cuda, "max_memory_allocated", lambda _device: 11 * mib
        )
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 13 * mib)
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda _device: (70 * mib, 80 * mib)
        )

        with p:
            snapshot = p.snapshot_memory("before_backward")
            p.end_iteration()

        assert snapshot["allocated_gpu_mb"] == 10.0
        assert p.get_current_metrics()["before_backward_device_free_gpu_mb"] == 70.0
        assert "[profiler][cuda-memory][memory-test] before_backward:" in (
            capsys.readouterr().out
        )


# ---------------------------------------------------------------------------
# create_profiler factory
# ---------------------------------------------------------------------------


class TestCreateProfiler:
    def test_none_mode_returns_null_profiler(self):
        assert isinstance(create_profiler(None, "cpu", "run"), NullProfiler)

    def test_custom_mode_returns_custom_profiler(self):
        assert isinstance(create_profiler("custom", "cpu", "run"), CustomProfiler)

    def test_unknown_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown profiler mode"):
            create_profiler("torch_profiler", "cpu", "run")

    def test_passes_warmup_and_active(self):
        p = create_profiler("custom", "cpu", "run", warmup=3, active=5)
        assert p._warmup == 3
        assert p._active == 5

    def test_nvidia_mode_returns_nvidia_profiler(self):
        assert isinstance(create_profiler("nvidia", "cpu", "run"), NvidiaProfiler)


# ---------------------------------------------------------------------------
# TorchProfiler
# ---------------------------------------------------------------------------


def _avg(key, cpu=0.0, dev=0.0, count=1, is_user=False):
    return SimpleNamespace(
        key=key,
        cpu_time_total=cpu,
        device_time_total=dev,
        count=count,
        is_user_annotation=is_user,
    )


class _StubCtx:
    """DistributedContext stand-in for a 2-rank world; ``peer`` is rank 1's
    summary and ``peer_collectives`` its per-section collective lists. Packs the
    peer's vectors in the profiler's layout and reduces against them, so the
    real collective semantics are exercised without a process group.
    """

    def __init__(self, peer, peer_collectives=None, use_dist=True):
        self.peer = peer
        self.peer_collectives = peer_collectives or {}
        self.use_dist = use_dist
        self.rank, self.world_size = 0, 2
        self.keys: list[str] | None = None
        self.counts: dict | None = None
        self.barriers = 0
        self.broadcasts = 0

    def broadcast_object_list(self, object_list, src=0, device=None):
        keys, counts = object_list[0]  # rank 0 is the source: unchanged
        self.keys, self.counts = list(keys), dict(counts)
        self.broadcasts += 1
        return object_list

    def _schema(self):
        """The profiler appends comm_sync_sec to the broadcast schema."""
        return self.keys + ["comm_sync_sec"]

    def _peer_collectives_and_flags(self):
        """The peer's per-collective slice plus its per-section aligned flags,
        padded with +inf where its counts differ from the latched layout."""
        vector, flags = [], []
        for name in sorted(self.counts):
            secs = self.peer_collectives.get(name, [])
            ok = len(secs) == self.counts[name]
            flags.append(1.0 if ok else 0.0)
            vector.extend(secs if ok else [float("inf")] * self.counts[name])
        return vector, flags

    def all_reduce(self, x, op=None, group=None):
        import torch.distributed as dist

        keys = self._schema()
        vector, flags = self._peer_collectives_and_flags()
        if op is dist.ReduceOp.SUM:
            other = (
                [self.peer.get(k, 0.0) for k in keys]
                + [1.0 if k in self.peer else 0.0 for k in keys]
                + flags
            )
            return x + torch.tensor(other, dtype=x.dtype)
        if op is dist.ReduceOp.MAX:
            # the max vector carries the per-key block only: compute_max is
            # reduced per section, not per collective
            other = [self.peer.get(k, float("-inf")) for k in keys]
            return torch.maximum(x, torch.tensor(other, dtype=x.dtype))
        other = [self.peer.get(k, float("inf")) for k in keys] + vector
        return torch.minimum(x, torch.tensor(other, dtype=x.dtype))

    def barrier(self):
        self.barriers += 1


class TestTorchProfiler:
    def test_factory_forwards_params(self):
        p = create_profiler("torch", "cpu", "run", per_step=False, repeat=3)
        assert isinstance(p, TorchProfiler)
        assert p._per_step is False
        assert p._repeat == 3

    def test_trace_dir_with_per_step_true_raises(self):
        with pytest.raises(ValueError, match="per_step=False"):
            TorchProfiler(device="cpu", name="x", trace_dir="/tmp/tr", per_step=True)

    def test_group_by_key_merges_two_views(self):
        # CPU-view carries cpu_time; CUDA-view carries the accurate dev_time.
        avgs = [
            _avg("denoise", cpu=100.0, dev=0.0, count=5, is_user=True),
            _avg("denoise", cpu=0.0, dev=200.0, count=5),
        ]
        g = _group_by_key(avgs)["denoise"]
        assert g["cpu_time"] == 100.0  # from CPU-view
        assert g["dev_time"] == 200.0  # CUDA-view wins over CPU-view fallback
        assert g["count"] == 5
        assert g["is_user"] is True

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("per_step,expected", [(True, {0, 1, 2}), (False, {"agg"})])
    def test_csv_iter_labels(self, tmp_path, monkeypatch, device, per_step, expected):
        monkeypatch.chdir(tmp_path)
        p = TorchProfiler(device=device, name="run", per_step=per_step, save_file=True)
        _run_torch(p, n_iters=3, device=device)
        p.finalize(None)
        import pandas as pd

        df = pd.read_csv(tmp_path / "outputs" / "run_gpu_metrics.csv")
        assert set(df["iter"]) == expected

    @pytest.mark.parametrize("device", DEVICES)
    def test_window_records_only_active_iters(self, tmp_path, monkeypatch, device):
        # warmup=1 skips iter 0; active=2 stops after iters 1,2 => iters 3,4 unrecorded.
        monkeypatch.chdir(tmp_path)
        p = TorchProfiler(
            device=device, name="run", warmup=1, active=2, per_step=True, save_file=True
        )
        _run_torch(p, n_iters=5, device=device)
        p.finalize(None)
        import pandas as pd

        df = pd.read_csv(tmp_path / "outputs" / "run_gpu_metrics.csv")
        assert set(df["iter"]) == {1, 2}

    def test_trace_written_without_op_rows(self, tmp_path, monkeypatch):
        # warmup beyond the run => torch discards everything => no op rows, but the
        # Chrome trace must still be exported (finalize reorder fix).
        monkeypatch.chdir(tmp_path)
        trace_dir = tmp_path / "traces"
        p = TorchProfiler(
            device="cpu",
            name="run",
            warmup=100,
            per_step=False,
            trace_dir=str(trace_dir),
        )
        _run_torch(p, n_iters=2, device="cpu")
        p.finalize(None)
        assert not p._all_op_rows
        assert (trace_dir / "rank_0.pt.trace.json").exists()

    @pytest.mark.parametrize("device", DEVICES)
    def test_metrics_mode_split(self, device):
        p_true = TorchProfiler(device=device, name="t", per_step=True)
        _run_torch(p_true, n_iters=2, device=device)
        assert "gradient_cpu_sec" in p_true.get_current_metrics()

        p_false = TorchProfiler(device=device, name="t", per_step=False)
        _run_torch(p_false, n_iters=2, device=device)
        assert set(p_false.get_current_metrics()) == {"total_time_sec", "max_gpu_mb"}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_dev_time_views_equal_gpu(self):
        """Codifies the investigation: key_averages() emits a CPU-view and a
        CUDA-view per user section; cpu_time lives on one, self_device on the
        other, and the two device_time_total values agree (min() is a safe tie-break).
        """
        x = torch.randn(1024, 1024, device="cuda")
        sched = torch.profiler.schedule(wait=0, warmup=2, active=3, repeat=1)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=sched,
        ) as prof:
            for _ in range(5):
                with torch.profiler.record_function("denoise"):
                    y = x
                    for _ in range(4):
                        y = y @ x
                prof.step()

        views = [
            e
            for e in prof.key_averages()
            if e.is_user_annotation and e.key == "denoise"
        ]
        assert len(views) == 2
        cpu_view = max(views, key=lambda e: e.cpu_time_total)
        cuda_view = max(views, key=lambda e: e.self_device_time_total)
        assert cpu_view is not cuda_view
        assert cpu_view.cpu_time_total > 0 and cuda_view.cpu_time_total == 0
        a, b = cpu_view.device_time_total, cuda_view.device_time_total
        assert abs(a - b) <= 0.05 * max(a, b)  # equal within 5% (jitter)


# ---------------------------------------------------------------------------
# TorchProfiler — cross-rank comm reduction
# ---------------------------------------------------------------------------


class TestCommAcrossRanks:
    """``{sec}_transfer_sec`` must become the min over ranks (pure transfer) and
    ``{sec}_wait_sec`` the MEAN gap, so that per section
    ``cuda_sec == mean(compute) + transfer_sec + wait_sec`` holds exactly.

    Fixture is the 2-GPU worked example. denoise: both GPUs start aligned,
    GPU-1 (rank 0) computes 10.0s and GPU-2 (rank 1) computes 11.0s; transfer
    is 0.05s, so both exit at 11.05s. Rank 0 idles 1.00s inside the collective
    (its raw value 1.05), rank 1 waits for nobody (0.05) and so wins the min.
    Expected: transfer 0.05, wait mean(1.00, 0)=0.50, mean(compute)=10.50,
    max(compute)=11.00, and 10.50+0.05+0.50 == 11.05.

    gradient reverses which rank wins: rank 0 computes 0.10 and rank 1 0.09,
    transfer 0.014, both exit at 0.114 -- so rank 0 now holds the min.
    """

    RANK0 = {
        "denoise_cuda_sec": 11.05,
        "denoise_transfer_sec": 1.05,  # computed 10.0, idled 1.00 -> 1.05 raw
        "gradient_cuda_sec": 0.114,
        "gradient_transfer_sec": 0.014,  # computed 0.10, arrived last -> pure
        "comm_cuda_sec": 1.064,
        "comm_sync_sec": 0.10,
        "total_time_sec": 90.0,  # control: no _transfer_sec/_cuda_sec suffix
    }
    RANK1 = {
        "denoise_cuda_sec": 11.05,
        "denoise_transfer_sec": 0.05,  # computed 11.0, arrived last -> pure
        "gradient_cuda_sec": 0.114,
        "gradient_transfer_sec": 0.024,  # computed 0.09, idled 0.01
        "comm_cuda_sec": 0.074,
        "comm_sync_sec": 0.04,
        "total_time_sec": 91.0,
    }

    @staticmethod
    def _with_compute(summary):
        """Mirror the profiler's local compute injection, for the peer's dict."""
        out = dict(summary)
        for key in list(out):
            if key.endswith("_transfer_sec"):
                section = key[: -len("_transfer_sec")]
                cuda = out.get(f"{section}_cuda_sec")
                if cuda is not None:
                    out[f"{section}_compute_sec"] = round(cuda - out[key], 6)
        return out

    @staticmethod
    def _one_collective_each(summary):
        """Sections holding a single collective, so its time is the section's."""
        return {
            key[: -len("_transfer_sec")]: [summary[key]]
            for key in summary
            if key.endswith("_transfer_sec")
        }

    def _sync(self, summary, peer, collectives=None, peer_collectives=None, **kw):
        p = TorchProfiler(device="cpu", name="run")
        p._pending_summary = dict(summary)
        p._pending_collectives = (
            self._one_collective_each(summary) if collectives is None else collectives
        )
        ctx = _StubCtx(
            self._with_compute(peer),
            (
                self._one_collective_each(peer)
                if peer_collectives is None
                else peer_collectives
            ),
            **kw,
        )
        p._sync_comm_metrics(ctx)
        return p._pending_summary, ctx, p

    def test_aligned_fixture_end_to_end(self):
        """One _sync(RANK0, RANK1) call, checked from every angle: transfer is
        the min (not rank 0's raw value); wait is the mean gap (not the max
        gap, which would give 12.05 instead of 11.05); compute is published as
        both mean and max, and LB is a direct ratio of the two; the three parts
        are additive back to cuda_sec per section; comm_cuda_sec is rebuilt
        from the reduced transfers (not rank 0's stale value); comm_sync_sec
        uses the same min basis and gains no wait column of its own; a
        non-suffix-matching column (total_time_sec) is untouched; and denoise's
        min comes from the peer while gradient's comes from this rank's own
        value, i.e. a different rank "wins" per section in the same call.
        """
        out, _, _ = self._sync(self.RANK0, self.RANK1)

        assert out["denoise_transfer_sec"] == 0.05  # min, not rank 0's 1.05
        assert out["denoise_wait_sec"] == pytest.approx(0.50)  # mean gap, not max
        assert out["gradient_transfer_sec"] == 0.014
        assert out["gradient_wait_sec"] == pytest.approx(0.005)
        assert out["denoise_transfer_sec"] == self.RANK1["denoise_transfer_sec"]
        assert out["gradient_transfer_sec"] == self.RANK0["gradient_transfer_sec"]

        assert out["denoise_compute_sec"] == pytest.approx(10.50)  # mean(10, 11)
        assert out["denoise_compute_max_sec"] == pytest.approx(11.00)  # critical path
        lb = out["denoise_compute_sec"] / out["denoise_compute_max_sec"]
        assert lb == pytest.approx(10.5 / 11.0)

        for section, span in (("denoise", 11.05), ("gradient", 0.114)):
            parts = (
                out[f"{section}_compute_sec"]
                + out[f"{section}_transfer_sec"]
                + out[f"{section}_wait_sec"]
            )
            assert parts == pytest.approx(span)
            assert parts == pytest.approx(out[f"{section}_cuda_sec"])

        sections = out["denoise_transfer_sec"] + out["gradient_transfer_sec"]
        assert out["comm_cuda_sec"] == pytest.approx(sections)
        assert out["comm_cuda_sec"] != self.RANK0["comm_cuda_sec"]
        assert out["comm_sync_sec"] == 0.04  # min over ranks
        assert "comm_sync_wait_sec" not in out
        assert out["total_time_sec"] == 90.0  # no _transfer_sec/_cuda_sec suffix

    def test_alternating_leader_across_two_collectives(self):
        """A section with 2 collectives where the faster rank ALTERNATES: rank 0
        idles 1.00s at the first, rank 1 at the second. Both totals are 1.10, so
        reducing the section total gives transfer 1.10 / wait 0.00 -- backwards.
        Reducing per collective gives 0.05 + 0.05 = 0.10 transfer, wait 1.00.
        """
        rank0 = {"denoise_cuda_sec": 10.10, "denoise_transfer_sec": 1.10}
        rank1 = {"denoise_cuda_sec": 10.10, "denoise_transfer_sec": 1.10}
        out, _, _ = self._sync(
            rank0,
            rank1,
            collectives={"denoise": [1.05, 0.05]},
            peer_collectives={"denoise": [0.05, 1.05]},
        )
        assert out["denoise_transfer_sec"] == pytest.approx(0.10)
        assert out["denoise_wait_sec"] == pytest.approx(1.00)
        assert out["denoise_compute_sec"] == pytest.approx(9.00)
        parts = (
            out["denoise_compute_sec"]
            + out["denoise_transfer_sec"]
            + out["denoise_wait_sec"]
        )
        assert parts == pytest.approx(10.10)

        # compute_max stays a SECTION-level reduction — max over each rank's
        # own (cuda - collectives) — while transfer is reduced per collective.
        # The two granularities differ, so compute_max + transfer falls short of
        # cuda_sec by whatever part of the critical path no single rank owned;
        # here both ranks compute 9.00, so the shortfall is the full 1.00s the
        # alternating leader spent idle. Reducing compute per compute-segment
        # would close that gap in theory, but measured on 2-64 GPUs it left the
        # gap unchanged (0.3-4.7%, driven by rank skew, not by granularity) --
        # see docs/compute-max-per-collective-proposal.md.
        assert out["denoise_compute_max_sec"] == pytest.approx(9.00)
        critical = out["denoise_compute_max_sec"] + out["denoise_transfer_sec"]
        assert critical == pytest.approx(9.10)
        assert critical < out["denoise_cuda_sec"]

    def test_collective_count_drift_falls_back_instead_of_misaligning(self):
        """If a rank's collective count changes after the schema latched (a
        different backward graph, say), that section must fall back to the
        section-level min on every rank rather than pairing up misaligned
        entries. Here the peer reports 3 collectives against a latched 2.
        """
        rank0 = {"denoise_cuda_sec": 10.10, "denoise_transfer_sec": 1.10}
        rank1 = {"denoise_cuda_sec": 10.10, "denoise_transfer_sec": 1.10}
        out, _, _ = self._sync(
            rank0,
            rank1,
            collectives={"denoise": [1.05, 0.05]},
            peer_collectives={"denoise": [0.05, 1.00, 0.05]},
        )
        # section-level min of the two 1.10 totals, not a misaligned pairing
        assert out["denoise_transfer_sec"] == pytest.approx(1.10)
        assert out["denoise_wait_sec"] == pytest.approx(0.0)

    def test_staggered_entry_still_gives_exact_compute(self):
        """Both ranks compute 10.0s but rank 1 enters the section 0.3s late, so
        the section spans differ (10.35 vs 10.05). Deriving compute downstream
        as cuda - transfer would give 9.85 (min-based) or 10.15 (max), and the
        old min(cuda) reduction lost the stagger entirely. Reducing each rank's
        own cuda - collective instead recovers 10.0 exactly, both mean and max.
        """
        rank0 = {"denoise_cuda_sec": 10.35, "denoise_transfer_sec": 0.35}
        rank1 = {"denoise_cuda_sec": 10.05, "denoise_transfer_sec": 0.05}
        out, _, _ = self._sync(rank0, rank1)
        assert out["denoise_compute_sec"] == pytest.approx(10.00)
        assert out["denoise_compute_max_sec"] == pytest.approx(10.00)
        assert out["denoise_transfer_sec"] == 0.05
        assert out["denoise_wait_sec"] == pytest.approx(0.15)
        assert out["denoise_cuda_sec"] == pytest.approx(10.20)  # mean, not min

    def test_schema_is_rank0s_and_a_missing_key_skews_neither_min_nor_mean(self):
        """Rank 1 lacks the gradient keys entirely: the schema still comes from
        rank 0, the absent value cannot win the min, and the presence mask makes
        the mean divide by 1 rather than 2 -- so wait stays 0 instead of being
        halved into a phantom 0.007.
        """
        peer = {
            "denoise_transfer_sec": 0.05,
            "denoise_cuda_sec": 11.05,
            "comm_sync_sec": 0.04,
        }
        out, ctx, _ = self._sync(self.RANK0, peer)
        assert ctx.keys == [
            "denoise_compute_sec",
            "denoise_cuda_sec",
            "denoise_transfer_sec",
            "gradient_compute_sec",
            "gradient_cuda_sec",
            "gradient_transfer_sec",
        ]
        assert out["gradient_transfer_sec"] == 0.014
        assert out["gradient_wait_sec"] == 0.0
        # only rank 0 reported gradient, so its compute is rank 0's alone
        assert out["gradient_compute_sec"] == pytest.approx(0.10)

    def test_single_gpu_run_is_untouched(self):
        out, ctx, p = self._sync(self.RANK0, self.RANK1, use_dist=False)
        assert out == self.RANK0
        assert not any(k.endswith("_wait_sec") for k in out)
        assert p._metric_keys is None

    def test_schema_not_latched_while_nothing_is_profiled(self):
        """During warmup the summary is empty; the schema must stay unlatched so
        a later iteration can still establish it.
        """
        _, _, p = self._sync({}, self.RANK1)
        assert p._metric_keys is None

    def test_schema_latched_once_then_reused(self):
        p = TorchProfiler(device="cpu", name="run")
        ctx = _StubCtx(self.RANK1)
        p._pending_summary = dict(self.RANK0)
        p._sync_comm_metrics(ctx)
        latched = p._metric_keys
        assert latched == [
            "denoise_compute_sec",
            "denoise_cuda_sec",
            "denoise_transfer_sec",
            "gradient_compute_sec",
            "gradient_cuda_sec",
            "gradient_transfer_sec",
            "comm_sync_sec",
        ]

        p._pending_summary = dict(self.RANK0)
        p._sync_comm_metrics(ctx)
        assert ctx.broadcasts == 1  # schema negotiated once per run, not per iter
        assert p._metric_keys is latched

    def test_end_iteration_reduces_and_still_barriers(self):
        """Integration through the real loop: the reduction runs inside
        end_iteration() and does not displace the end-of-iteration barrier.
        The trace callback is stubbed because a CPU run emits no NCCL ops.
        """
        p = TorchProfiler(device="cpu", name="run")
        ctx = _StubCtx(self.RANK1)
        p._on_trace_ready = lambda prof: p._pending_summary.update(self.RANK0)
        with p:
            with p.track_step("gradient"):
                a = torch.randn(32, 32)
                _ = a @ a
            p.end_iteration(ctx)
        assert ctx.barriers == 1
        m = p.get_current_metrics()
        assert m["denoise_transfer_sec"] == 0.05
        assert m["denoise_wait_sec"] == pytest.approx(0.50)
        assert m["denoise_cuda_sec"] == 11.05
        assert "total_time_sec" in m


# ---------------------------------------------------------------------------
# NvidiaProfiler
# ---------------------------------------------------------------------------


class TestNvidiaProfiler:
    def test_warmup_skips_first_n_iterations(self):
        p = NvidiaProfiler(device="cpu", name="test", warmup=2)
        with p:
            for _ in range(4):
                with p.track_step("grad"):
                    pass
                p.end_iteration()
        assert len(p._all_results) == 2

    def test_active_stops_after_n_iterations(self):
        p = NvidiaProfiler(device="cpu", name="test", warmup=0, active=2)
        with p:
            for _ in range(5):
                with p.track_step("grad"):
                    pass
                p.end_iteration()
        assert len(p._all_results) == 2

    def test_end_iteration_stores_total_and_gpu(self):
        p = NvidiaProfiler(device="cpu", name="test")
        with p:
            with p.track_step("step"):
                pass
            p.end_iteration()
        metrics = p.get_current_metrics()
        assert "total_time_sec" in metrics
        assert "max_gpu_mb" in metrics

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_no_nvtx_range_during_warmup(self):
        """On CPU, _has_cuda=False already skips NVTX regardless of warmup,
        so this needs a real GPU to actually exercise the _is_recording()
        guard added to _push_iter_range/track_step."""
        p = NvidiaProfiler(device="cuda", name="test", warmup=1)
        with p:
            assert p._iter_range_open is False  # iter 0 = warmup: no range opened
            with p.track_step("grad"):
                pass
            p.end_iteration()  # crosses the warmup boundary
            assert (
                p._iter_range_open is True
            )  # iter 1 = first recorded iter: range open
