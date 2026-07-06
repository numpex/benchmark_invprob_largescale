"""Shared helpers for training benchmark visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolsbench.visualization.common import (
    COLORWAY,
    DEFAULT_TRAINING_OUTPUT_DIR,
    TRAINING_REQUIRED_COLUMNS,
    configure_matplotlib,
    format_image_size,
    load_results,
    set_gpu_axis,
    style_axes,
    summarize_training_configs,
    write_figure,
)


def load_training_summary(results: str | Path) -> tuple[pd.DataFrame, Path]:
    """Load training results and return per-configuration summaries."""
    df, results_path = load_results(results, required=TRAINING_REQUIRED_COLUMNS)
    normalize_training_timing_columns(df)
    summary = summarize_training_configs(df)
    return summary, results_path


def normalize_training_timing_columns(df: pd.DataFrame) -> None:
    """Add canonical timing columns for profiler schema variants."""
    if "objective_forward_time_sec" not in df.columns:
        if "objective_forward_cuda_sec" in df.columns:
            df["objective_forward_time_sec"] = df["objective_forward_cuda_sec"]
        elif "objective_forward_cpu_sec" in df.columns:
            df["objective_forward_time_sec"] = df["objective_forward_cpu_sec"]

    if "objective_backward_time_sec" not in df.columns:
        if "objective_backward_cuda_sec" in df.columns:
            df["objective_backward_time_sec"] = df["objective_backward_cuda_sec"]
        elif "objective_backward_cpu_sec" in df.columns:
            df["objective_backward_time_sec"] = df["objective_backward_cpu_sec"]

    if "objective_comm_time_sec" not in df.columns:
        comm_parts = [
            col
            for col in ["objective_comm_cuda_sec", "objective_comm_sync_sec"]
            if col in df.columns
        ]
        if comm_parts:
            df["objective_comm_time_sec"] = df[comm_parts].fillna(0).sum(axis=1)


def make_output_path(
    output_dir: str | Path,
    experiment: str,
    results_path: Path,
) -> Path:
    """Build the output directory for one training experiment result."""
    output_path = Path(output_dir) / f"{experiment}_{results_path.stem}"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def clear_png_outputs(output_path: Path) -> None:
    """Remove stale generated PNGs from an experiment output directory."""
    for path in output_path.glob("*.png"):
        path.unlink()


def problem_size_title(summary: pd.DataFrame) -> str:
    """Format the problem size portion of a figure title."""
    sizes = sorted(summary["training_image_size"].dropna().unique())
    if len(sizes) == 1:
        return f"Problem size {format_image_size(sizes[0])}"
    return "Problem sizes " + ", ".join(format_image_size(size) for size in sizes)


def plot_training_strong_scaling(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Training Strong Scaling Efficiency",
    filename: str = "strong_scaling.png",
    group_col: str = "training_image_size",
    group_label: str = "Image size",
) -> str:
    """Plot strong-scaling efficiency for training summaries."""
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    max_efficiency = 100.0
    all_gpus = sorted(summary["n_gpus"].unique())

    for idx, (group_value, group) in enumerate(summary.groupby(group_col, dropna=False)):
        rows = (
            group.sort_values(["n_gpus", "avg_total_time_sec"], na_position="last")
            .groupby("n_gpus", as_index=False, dropna=False)
            .first()
            .sort_values("n_gpus")
        )
        if rows.empty:
            continue
        baseline = rows.loc[rows["n_gpus"].idxmin()]
        baseline_gpus = int(baseline["n_gpus"])
        baseline_time = float(baseline["avg_total_time_sec"])
        rows = rows.assign(
            efficiency_pct=(
                baseline_time
                * baseline_gpus
                / (rows["n_gpus"] * rows["avg_total_time_sec"])
                * 100
            )
        )
        max_efficiency = max(max_efficiency, float(rows["efficiency_pct"].max()))
        color = COLORWAY[idx % len(COLORWAY)]
        if group_col == "training_image_size":
            label = f"{format_image_size(group_value)} ({baseline_gpus}-GPU baseline)"
        else:
            label = f"{group_label} {group_value:g} ({baseline_gpus}-GPU baseline)"
        ax.plot(
            rows["n_gpus"],
            rows["efficiency_pct"],
            marker="o",
            markersize=7,
            linewidth=2.8,
            color=color,
            label=label,
        )

    set_gpu_axis(ax, all_gpus)
    ax.axhline(100, color="#111827", linestyle="--", linewidth=1.3, alpha=0.55)
    ax.set_ylim(0, max(110, max_efficiency * 1.12))
    style_axes(
        ax,
        title,
        "Number of GPUs",
        "Parallel efficiency (%)",
    )
    ax.legend(loc="lower left", ncols=1)
    fig.tight_layout()
    return write_figure(fig, output_path, filename)


def plot_training_weak_scaling(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Training Weak Scaling Runtime Ratio",
    filename: str = "weak_scaling.png",
) -> str:
    """Plot weak-scaling runtime ratio for training summaries."""
    weak = summary.sort_values("n_gpus").copy()
    if weak.empty:
        raise ValueError("No rows available for weak-scaling plot.")
    baseline_time = float(weak.iloc[0]["avg_total_time_sec"])
    weak["time_ratio"] = weak["avg_total_time_sec"] / baseline_time

    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    ax.plot(
        weak["n_gpus"],
        weak["time_ratio"],
        marker="o",
        markersize=7,
        linewidth=2.8,
        color=COLORWAY[0],
        label="measured",
    )
    ax.axhline(1, color="#111827", linestyle="--", linewidth=1.3, alpha=0.55)
    for _, row in weak.iterrows():
        ax.annotate(
            format_image_size(row["training_image_size"]),
            (row["n_gpus"], row["time_ratio"]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#374151",
        )

    set_gpu_axis(ax, weak["n_gpus"])
    style_axes(
        ax,
        title,
        "Number of GPUs",
        "Average step time / 1-GPU time",
    )
    fig.tight_layout()
    return write_figure(fig, output_path, filename)


def plot_timing_breakdown_by_gpu(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    filename: str,
    group_col: str | None = None,
) -> str:
    """Plot forward/backward/residual step-time decomposition."""
    if group_col:
        groups = list(summary.groupby(group_col, dropna=False))
    else:
        groups = [("all", summary)]

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(5.4 * len(groups), 5.7),
        squeeze=False,
    )
    colors = ["#2563eb", "#f97316", "#16a34a"]

    for col, (group_value, group) in enumerate(groups):
        ax = axes[0, col]
        rows = (
            group.sort_values(["n_gpus", "avg_total_time_sec"], na_position="last")
            .groupby("n_gpus", as_index=False, dropna=False)
            .first()
            .sort_values("n_gpus")
        )
        x = rows["n_gpus"].to_numpy(dtype=float)
        forward = rows["avg_forward_time_sec"].fillna(0).to_numpy(dtype=float)
        backward = rows["avg_backward_time_sec"].fillna(0).to_numpy(dtype=float)
        other = rows["avg_other_time_sec"].fillna(0).to_numpy(dtype=float)
        stacked = np.vstack([forward, backward, other])

        ax.stackplot(
            x,
            stacked,
            labels=["forward", "backward", "other"],
            colors=colors,
            alpha=0.34,
            linewidth=0,
        )
        ax.plot(
            x,
            rows["avg_total_time_sec"],
            marker="o",
            markersize=5.5,
            linewidth=3.0,
            color="#111827",
            label="total",
        )
        set_gpu_axis(ax, x)
        if group_col == "training_image_size":
            subplot_title = format_image_size(group_value)
        elif group_col == "p_solver_max_batch_size":
            subplot_title = f"max batch {int(group_value)}"
        else:
            subplot_title = "All configurations"
        style_axes(
            ax,
            subplot_title,
            "Number of GPUs",
            "Average time / training step (s)" if col == 0 else "",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=4,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(title, x=0.02, ha="left", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    return write_figure(fig, output_path, filename)


def plot_memory_by_gpu(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    filename: str,
    group_col: str,
    ylabel: str = "Max GPU memory / GPU (GB)",
) -> str:
    """Plot max GPU memory against GPU count grouped by a summary column."""
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    for idx, (group_value, group) in enumerate(summary.groupby(group_col, dropna=False)):
        rows = group.sort_values("n_gpus")
        color = COLORWAY[idx % len(COLORWAY)]
        if group_col == "training_image_size":
            label = format_image_size(group_value)
        elif group_col == "p_solver_max_batch_size":
            label = f"max batch {int(group_value)}"
        else:
            label = str(group_value)
        ax.plot(
            rows["n_gpus"],
            rows["max_gpu_mb"] / 1024,
            marker="o",
            markersize=6.5,
            linewidth=2.5,
            color=color,
            label=label,
        )

    set_gpu_axis(ax, summary["n_gpus"])
    style_axes(ax, title, "Number of GPUs", ylabel)
    ax.legend(loc="best")
    fig.tight_layout()
    return write_figure(fig, output_path, filename)


__all__ = [
    "clear_png_outputs",
    "DEFAULT_TRAINING_OUTPUT_DIR",
    "configure_matplotlib",
    "normalize_training_timing_columns",
    "load_training_summary",
    "make_output_path",
    "problem_size_title",
    "plot_memory_by_gpu",
    "plot_timing_breakdown_by_gpu",
    "plot_training_strong_scaling",
    "plot_training_weak_scaling",
    "write_figure",
]
