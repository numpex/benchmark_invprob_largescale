from __future__ import annotations

from toolsbench.profiler.base import BenchProfiler as BenchProfiler, NullProfiler as NullProfiler
from toolsbench.profiler.custom import CustomProfiler as CustomProfiler


def create_profiler(
    mode,
    device,
    name: str = "",
    warmup: int = 0,
    active: int = 0,
) -> BenchProfiler:
    """Factory: return the right profiler for *mode*.

    Parameters
    ----------
    mode : None | "custom"
        ``None`` → NullProfiler (zero overhead).
        ``"custom"`` → CustomProfiler (GPU timing + memory → CSV).
    device : torch.device or str
        Target device (used by CustomProfiler).
    name : str
        Run name used as the CSV filename stem.
    warmup : int
        Iterations to skip before recording (default 0).
    active : int
        Iterations to record after warmup, 0 = all remaining (default 0).
    """
    if mode is None:
        return NullProfiler()
    if mode == "custom":
        return CustomProfiler(device=device, name=name, warmup=warmup, active=active)
    raise ValueError(f"Unknown profiler mode {mode!r}. Choose None or 'custom'.")
