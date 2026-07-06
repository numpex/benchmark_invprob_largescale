"""Visualizations for training communication-time experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolsbench.visualization.common import (
    COLORWAY,
    format_image_size,
    set_gpu_axis,
    style_axes,
)

from .common import (
    DEFAULT_TRAINING_OUTPUT_DIR,
    clear_png_outputs,
    configure_matplotlib,
    load_training_summary,
    make_output_path,
    problem_size_title,
    write_figure,
)


def create_comm_time_visualizations(
    results: str | Path,
    output_dir: str | Path = DEFAULT_TRAINING_OUTPUT_DIR,
) -> Path:
    """Create visualizations from a training communication-time parquet."""
    configure_matplotlib()
    summary, results_path = load_training_summary(results)
    summary = _with_comm_columns(summary)
    output_path = make_output_path(output_dir, "comm_time", results_path)
    clear_png_outputs(output_path)

    _plot_communication_breakdown(summary, output_path)
    _plot_communication_overhead(summary, output_path)
    return output_path


def _with_comm_columns(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["comm_cuda_sec"] = _optional_column(summary, "objective_comm_cuda_sec")
    summary["comm_total_sec"] = summary["comm_cuda_sec"]
    summary["compute_other_sec"] = np.clip(
        summary["avg_total_time_sec"] - summary["comm_total_sec"],
        a_min=0,
        a_max=None,
    )
    summary["comm_share_pct"] = (
        summary["comm_total_sec"] / summary["avg_total_time_sec"] * 100
    )
    return summary


def _optional_column(summary: pd.DataFrame, column: str) -> pd.Series:
    if column in summary.columns:
        return summary[column].fillna(0)
    return pd.Series(0.0, index=summary.index)


def _plot_communication_breakdown(summary: pd.DataFrame, output_path: Path) -> str:
    sizes = sorted(summary["training_image_size"].dropna().unique())
    fig, axes = plt.subplots(
        1,
        len(sizes),
        figsize=(5.2 * len(sizes), 6.0),
        squeeze=False,
    )
    components = [
        ("compute + other", "compute_other_sec", "#2563eb"),
        ("CUDA communication", "comm_cuda_sec", "#f97316"),
    ]

    for col_idx, size in enumerate(sizes):
        ax = axes[0, col_idx]
        rows = summary[summary["training_image_size"] == size].sort_values("n_gpus")
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        for label, column, color in components:
            values = rows[column].to_numpy(dtype=float)
            ax.bar(
                x,
                values,
                bottom=bottom,
                width=0.66,
                label=label,
                color=color,
                alpha=0.88,
            )
            bottom += values

        ax.plot(
            x,
            rows["avg_total_time_sec"],
            color="#111827",
            marker="o",
            linewidth=2.2,
            label="total",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(gpu)) for gpu in rows["n_gpus"]])
        style_axes(
            ax,
            format_image_size(size),
            "Number of GPUs",
            "Average time / training step (s)" if col_idx == 0 else "",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=3,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Training Communication Breakdown",
        x=0.02,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    return write_figure(fig, output_path, "communication_breakdown.png")


def _plot_communication_overhead(summary: pd.DataFrame, output_path: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8))

    for idx, (size, rows) in enumerate(summary.groupby("training_image_size")):
        rows = rows.sort_values("n_gpus")
        color = COLORWAY[idx % len(COLORWAY)]
        label = format_image_size(size)
        axes[0].plot(
            rows["n_gpus"],
            rows["comm_total_sec"],
            marker="o",
            linewidth=2.6,
            color=color,
            label=label,
        )
        axes[1].plot(
            rows["n_gpus"],
            rows["comm_share_pct"],
            marker="o",
            linewidth=2.6,
            color=color,
            label=label,
        )

    for ax in axes:
        set_gpu_axis(ax, summary["n_gpus"])
    style_axes(
        axes[0],
        "Communication Time",
        "Number of GPUs",
        "Average communication time / step (s)",
    )
    style_axes(
        axes[1],
        "Communication Share",
        "Number of GPUs",
        "Communication / total step time (%)",
    )
    axes[1].legend(loc="best", title="Problem size")
    fig.suptitle(
        f"Communication Overhead - {problem_size_title(summary)}",
        x=0.02,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return write_figure(fig, output_path, "communication_overhead.png")
