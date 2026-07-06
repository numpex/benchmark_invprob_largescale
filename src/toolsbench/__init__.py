"""toolsbench package."""

import builtins
import sys
import typing

# SimAI-Bench may eagerly import Dragon symbols in worker contexts.
# Provide harmless fallbacks so non-Dragon runs keep working.
if not hasattr(builtins, "Task"):
    builtins.Task = object
if not hasattr(builtins, "Any"):
    builtins.Any = typing.Any
if not hasattr(builtins, "Sequence"):
    builtins.Sequence = typing.Sequence


def main(argv: list[str] | None = None) -> int:
    """Console entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["vizinference"]:
        from toolsbench.visualization.cli import main as visualization_main

        return visualization_main("vizinference", argv[1:])
    if argv[:1] == ["viztraining"]:
        from toolsbench.visualization.cli import main as visualization_main

        return visualization_main("viztraining", argv[1:])

    print(
        "toolsbench installs shared benchmark utilities. "
        "Run benchmarks with `benchopt run <benchmark_path>` or create "
        "visualizations with `toolsbench vizinference --help` or "
        "`toolsbench viztraining --help`."
    )
    return 0
