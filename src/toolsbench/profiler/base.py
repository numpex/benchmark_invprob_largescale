from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager


class BenchProfiler(ABC):
    """Common interface for all profiler backends."""

    @abstractmethod
    def __enter__(self) -> "BenchProfiler": ...

    @abstractmethod
    def __exit__(self, *args): ...

    @abstractmethod
    def track_step(self, name: str):
        """Context manager: annotate a named sub-step within an iteration."""

    @abstractmethod
    def end_iteration(self):
        """Signal end of one iteration."""

    @abstractmethod
    def get_current_metrics(self) -> dict:
        """Return the latest per-iteration metrics — called from get_result() mid-run."""

    def snapshot_memory(self, name: str, log: bool = True) -> dict:
        """Capture an optional named accelerator-memory snapshot.

        Profilers without memory telemetry keep this as a no-op so solvers can
        instrument critical boundaries independently of the selected backend.
        """
        return {}

    @abstractmethod
    def finalize(self, ctx) -> None:
        """Write all outputs (CSV, JSON, …) at end of run."""


class NullProfiler(BenchProfiler):
    """No-op profiler — zero overhead."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @contextmanager
    def track_step(self, name: str):
        yield

    def end_iteration(self, ctx=None):
        pass

    def get_current_metrics(self) -> dict:
        return {}

    def finalize(self, ctx) -> None:
        pass
