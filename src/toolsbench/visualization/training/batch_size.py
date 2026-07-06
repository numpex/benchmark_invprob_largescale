"""Visualizations for training batch-size experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from toolsbench.visualization.common import COLORWAY, set_gpu_axis, style_axes

from .common import (
    DEFAULT_TRAINING_OUTPUT_DIR,
    clear_png_outputs,
    configure_matplotlib,
    load_training_summary,
    make_output_path,
    problem_size_title,
    write_figure,
)


def create_batch_size_visualizations(
    results: str | Path,
    output_dir: str | Path = DEFAULT_TRAINING_OUTPUT_DIR,
) -> Path:
    """Create visualizations from a training batch-size parquet."""
    configure_matplotlib()
    summary, results_path = load_training_summary(results)
    output_path = make_output_path(output_dir, "batch_size", results_path)
    clear_png_outputs(output_path)

    _plot_batch_size_overview(summary, output_path)
    return output_path


def _plot_batch_size_overview(summary, output_path: Path) -> str:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.8, 6.2),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    _plot_strong_scaling_panel(summary, axes[0])
    _plot_memory_histogram(summary, axes[1])
    fig.suptitle(
        f"Batch Size Scaling - {problem_size_title(summary)}",
        x=0.02,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return write_figure(fig, output_path, "batch_size_overview.png")


def _plot_strong_scaling_panel(summary, ax) -> None:
    max_efficiency = 100.0
    for idx, (batch_size, group) in enumerate(
        summary.groupby("p_solver_max_batch_size", dropna=False)
    ):
        rows = (
            group.sort_values(["n_gpus", "avg_total_time_sec"], na_position="last")
            .groupby("n_gpus", as_index=False, dropna=False)
            .first()
            .sort_values("n_gpus")
        )
        baseline = rows.loc[rows["n_gpus"].idxmin()]
        baseline_time = float(baseline["avg_total_time_sec"])
        baseline_gpus = int(baseline["n_gpus"])
        rows = rows.assign(
            efficiency_pct=(
                baseline_time
                * baseline_gpus
                / (rows["n_gpus"] * rows["avg_total_time_sec"])
                * 100
            )
        )
        max_efficiency = max(max_efficiency, float(rows["efficiency_pct"].max()))
        ax.plot(
            rows["n_gpus"],
            rows["efficiency_pct"],
            marker="o",
            markersize=7,
            linewidth=2.8,
            color=COLORWAY[idx % len(COLORWAY)],
            label=f"max batch {int(batch_size)}",
        )

    set_gpu_axis(ax, summary["n_gpus"])
    ax.axhline(100, color="#111827", linestyle="--", linewidth=1.2, alpha=0.55)
    ax.set_ylim(0, max(110, max_efficiency * 1.12))
    style_axes(
        ax,
        "Strong Scaling by Max Batch Size",
        "Number of GPUs",
        "Parallel efficiency (%)",
    )
    ax.legend(loc="best", title="Configuration")


def _plot_memory_histogram(summary, ax) -> None:
    batches = sorted(summary["p_solver_max_batch_size"].dropna().unique())
    preferred_gpus = [1, 4, 16]
    available_gpus = sorted(summary["n_gpus"].dropna().unique())
    gpus = [gpu for gpu in preferred_gpus if gpu in available_gpus]
    if not gpus:
        gpus = available_gpus

    x = np.arange(len(batches))
    width = min(0.22, 0.72 / max(len(gpus), 1))
    offsets = (np.arange(len(gpus)) - (len(gpus) - 1) / 2) * width
    max_height = 0.0

    for idx, gpu in enumerate(gpus):
        values = []
        for batch_size in batches:
            rows = summary[
                (summary["p_solver_max_batch_size"] == batch_size)
                & (summary["n_gpus"] == gpu)
            ]
            if rows.empty:
                values.append(np.nan)
            else:
                values.append(float((rows["max_gpu_mb"] / 1024).mean()))
        bars = ax.bar(
            x + offsets[idx],
            values,
            width=width,
            color=COLORWAY[idx % len(COLORWAY)],
            label=f"{int(gpu)} GPUs" if gpu > 1 else "1 GPU",
            alpha=0.9,
        )
        finite_values = [value for value in values if np.isfinite(value)]
        if finite_values:
            max_height = max(max_height, max(finite_values))
        ax.bar_label(
            bars,
            labels=[
                f"{value:.1f}" if np.isfinite(value) else ""
                for value in values
            ],
            padding=3,
            fontsize=8.5,
            color="#374151",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(batch_size)) for batch_size in batches])
    if max_height:
        ax.set_ylim(0, max_height * 1.18)
    style_axes(
        ax,
        "GPU Memory by Batch Size",
        "Max batch size",
        "Max GPU memory / GPU (GB)",
    )
    ax.legend(loc="best", title="Hardware")
