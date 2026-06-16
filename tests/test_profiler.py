import pytest

from toolsbench.profiler import create_profiler, NullProfiler, CustomProfiler


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
        p = CustomProfiler(device="cpu", name="myrun")
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
