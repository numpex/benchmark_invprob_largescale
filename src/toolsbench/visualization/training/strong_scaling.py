"""Visualizations for training strong-scaling experiments."""

from __future__ import annotations

from pathlib import Path

from .common import (
    DEFAULT_TRAINING_OUTPUT_DIR,
    clear_png_outputs,
    configure_matplotlib,
    load_training_summary,
    make_output_path,
    problem_size_title,
    plot_training_strong_scaling,
)


def create_strong_scaling_visualizations(
    results: str | Path,
    output_dir: str | Path = DEFAULT_TRAINING_OUTPUT_DIR,
) -> Path:
    """Create visualizations from a training strong-scaling parquet."""
    configure_matplotlib()
    summary, results_path = load_training_summary(results)
    output_path = make_output_path(output_dir, "strong_scaling", results_path)
    clear_png_outputs(output_path)

    batch_sizes = sorted(summary["p_solver_max_batch_size"].dropna().unique())
    if len(batch_sizes) == 1:
        batch_label = f"max batch size {int(batch_sizes[0])}"
    else:
        batch_label = "max batch sizes " + ", ".join(
            str(int(batch_size)) for batch_size in batch_sizes
        )
    title = f"Training Strong Scaling Efficiency - {batch_label}"
    plot_training_strong_scaling(
        summary,
        output_path,
        title=f"{title}\n{problem_size_title(summary)}",
    )
    return output_path
