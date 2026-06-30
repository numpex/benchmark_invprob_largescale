from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
import pandas as pd

import torch

from toolsbench.profiler.base import BenchProfiler


class CustomProfiler(BenchProfiler):
    """Tracks wall-clock time and max GPU per iteration, plus per-step time and max GPU.

    Parameters
    ----------
    warmup : int
        Number of iterations to skip before recording starts.
    active : int
        Number of iterations to record after warmup (0 = all remaining).
    """

    def __init__(self, device, name: str, warmup: int = 0, active: int = 0, save_file: bool = False):
        self._device = torch.device(device) if isinstance(device, str) else device
        self._name = name
        self._has_cuda = torch.cuda.is_available() and self._device.type == "cuda"
        self._warmup = warmup
        self._active = active
        self._save_file = save_file
        self._iter_count: int = 0
        self._step_metrics: dict[str, dict] = {}
        self._all_results: list[dict] = []
        self._current_metrics: dict = {}
        self._iter_t0: float = 0.0

    def _is_recording(self) -> bool:
        if self._iter_count < self._warmup:
            return False
        if self._active > 0 and self._iter_count >= self._warmup + self._active:
            return False
        return True

    def __enter__(self):
        self._all_results = []
        self._current_metrics = {}
        self._step_metrics = {}
        self._iter_count = 0
        self._iter_t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        pass

    @contextmanager
    def track_step(self, name: str):
        if not self._is_recording():
            yield
            return
        if self._has_cuda:
            torch.cuda.reset_peak_memory_stats(self._device)
            torch.cuda.synchronize(self._device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._has_cuda:
                torch.cuda.synchronize(self._device)
            self._step_metrics[name] = {
                "time_sec": time.perf_counter() - t0,
                "max_gpu_mb": (
                    torch.cuda.max_memory_allocated(self._device) / 1024**2
                    if self._has_cuda else 0.0
                ),
            }

    def end_iteration(self):
        if self._is_recording():
            total_time = time.perf_counter() - self._iter_t0
            max_gpu = max((m["max_gpu_mb"] for m in self._step_metrics.values()), default=0.0)
            captured = {"total_time_sec": total_time, "max_gpu_mb": max_gpu}
            for name, m in self._step_metrics.items():
                captured[f"{name}_time_sec"] = m["time_sec"]
                captured[f"{name}_max_gpu_mb"] = m["max_gpu_mb"]
            self._all_results.append(captured)
            self._current_metrics = captured
        self._step_metrics = {}
        self._iter_count += 1
        self._iter_t0 = time.perf_counter()

    def get_current_metrics(self) -> dict:
        return self._current_metrics

    def finalize(self, ctx) -> None:
        if not self._all_results or not self._save_file:
            return
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self._name}_gpu_metrics.csv"
        pd.DataFrame(self._all_results).to_csv(path, index=False)
        print(f"[profiler] Saved {len(self._all_results)} records to {path}")
