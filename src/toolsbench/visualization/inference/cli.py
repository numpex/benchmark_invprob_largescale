"""Compatibility CLI for inference visualizations."""

from __future__ import annotations

from toolsbench.visualization.cli import build_parser as _build_parser
from toolsbench.visualization.cli import main as _main


def build_parser():
    """Build the inference visualization parser."""
    return _build_parser("vizinference")


def main(argv: list[str] | None = None) -> int:
    """Run the inference visualization CLI."""
    return _main("vizinference", argv)
