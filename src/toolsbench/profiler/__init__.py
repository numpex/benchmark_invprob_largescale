from __future__ import annotations

from toolsbench.profiler.base import (
    BenchProfiler as BenchProfiler,
    NullProfiler as NullProfiler,
)
from toolsbench.profiler.custom import CustomProfiler as CustomProfiler
from toolsbench.profiler.torch_profiler import TorchProfiler as TorchProfiler
from toolsbench.profiler.nvidia_profiler import NvidiaProfiler as NvidiaProfiler


def create_profiler(
    mode,
    device,
    name: str = "",
    warmup: int = 0,
    active: int = 0,
    trace_dir: str | None = None,
    per_step: bool = True,
    repeat: int = 1,
    save_file: bool = False,
) -> BenchProfiler:
    """Factory: return the right profiler for *mode*.

    Parameters
    ----------
    mode : None | "custom" | "torch" | "nvidia"
        ``None`` → NullProfiler (zero overhead).
        ``"custom"`` → CustomProfiler (wall-clock + GPU memory → CSV).
        ``"torch"`` → TorchProfiler (per-iteration CUDA/CPU/comm ms via torch.profiler).
        ``"nvidia"`` → NvidiaProfiler (NVTX ranges + cudaProfilerStart/Stop for
        external capture by ``nsys profile --capture-range=cudaProfilerApi``).
    device : torch.device or str
        Target device.
    name : str
        Run name used as the CSV filename stem.
    warmup : int
        Iterations to skip before recording (default 0).
    active : int
        Iterations to record after warmup, 0 = all remaining (default 0).
    trace_dir : str or None
        ``"torch"`` mode with ``per_step=False`` only — directory for Chrome trace.
        ``None`` = not saved. Passing a directory with ``per_step=True`` raises ValueError
        (the profiler resets each iteration, so a full trace cannot be captured).
    per_step : bool
        ``"torch"`` mode only. True (default): per-iteration section summary returned
        to benchopt + per-op CSV rows tagged with the iteration index. False: CSV only,
        with per-op rows aggregated over the whole window (``iter == "agg"``).
    repeat : int
        ``"torch"`` mode, ``per_step=False`` only. Number of (warmup+active) cycles
        passed to torch schedule. 1 (default) = one recording window; 0 = repeat forever.
    """
    if mode is None:
        return NullProfiler()
    if mode == "custom":
        return CustomProfiler(
            device=device, name=name, warmup=warmup, active=active, save_file=save_file
        )
    if mode == "torch":
        return TorchProfiler(
            device=device,
            name=name,
            warmup=warmup,
            active=active,
            trace_dir=trace_dir,
            per_step=per_step,
            repeat=repeat,
            save_file=save_file,
        )
    if mode == "nvidia":
        return NvidiaProfiler(
            device=device, name=name, warmup=warmup, active=active, save_file=save_file
        )
    raise ValueError(
        f"Unknown profiler mode {mode!r}. Choose None, 'custom', 'torch', or 'nvidia'."
    )
