"""Visualizations for training weak-scaling experiments."""

from __future__ import annotations

from pathlib import Path

from .common import (
    DEFAULT_TRAINING_OUTPUT_DIR,
    clear_png_outputs,
    configure_matplotlib,
    load_training_summary,
    make_output_path,
    plot_training_weak_scaling,
)


def create_weak_scaling_visualizations(
    results: str | Path,
    output_dir: str | Path = DEFAULT_TRAINING_OUTPUT_DIR,
) -> Path:
    """Create visualizations from a training weak-scaling parquet."""
    configure_matplotlib()
    summary, results_path = load_training_summary(results)
    output_path = make_output_path(output_dir, "weak_scaling", results_path)
    clear_png_outputs(output_path)

    plot_training_weak_scaling(summary, output_path)
    return output_path
