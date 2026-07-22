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
