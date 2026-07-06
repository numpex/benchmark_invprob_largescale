"""Visualizations for training checkpointing experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from toolsbench.visualization.common import COLORWAY, style_axes

from .common import (
    DEFAULT_TRAINING_OUTPUT_DIR,
    clear_png_outputs,
    configure_matplotlib,
    load_training_summary,
    make_output_path,
    problem_size_title,
    write_figure,
)


def create_checkpointing_visualizations(
    results: str | Path,
    output_dir: str | Path = DEFAULT_TRAINING_OUTPUT_DIR,
) -> Path:
    """Create visualizations from a training checkpointing parquet."""
    configure_matplotlib()
    summary, results_path = load_training_summary(results)
    output_path = make_output_path(output_dir, "checkpointing", results_path)
    clear_png_outputs(output_path)

    _plot_checkpoint_tradeoff(summary, output_path)
    return output_path


def _plot_checkpoint_tradeoff(summary, output_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(10.2, 6.3))
    summary = summary.copy()
    summary["worst_time_same_gpu"] = summary.groupby("n_gpus")[
        "avg_total_time_sec"
    ].transform("max")
    summary["relative_efficiency_pct"] = (
        summary["worst_time_same_gpu"] / summary["avg_total_time_sec"] * 100
    )
    markers = ["o", "s", "^", "D", "P", "X"]
    modes = sorted(summary["p_solver_checkpoint_batches"].dropna().unique())
    gpu_counts = sorted(summary["n_gpus"].dropna().unique())
    gpu_marker = {
        gpu: markers[idx % len(markers)] for idx, gpu in enumerate(gpu_counts)
    }

    for mode_idx, mode in enumerate(modes):
        rows = summary[summary["p_solver_checkpoint_batches"] == mode].copy()
        color = COLORWAY[mode_idx % len(COLORWAY)]
        for gpu in gpu_counts:
            gpu_rows = rows[rows["n_gpus"] == gpu].sort_values(
                "p_solver_max_batch_size"
            )
            if gpu_rows.empty:
                continue
            ax.scatter(
                gpu_rows["relative_efficiency_pct"],
                gpu_rows["max_gpu_mb"] / 1024,
                s=115,
                marker=gpu_marker[gpu],
                color=color,
                edgecolor="#111827",
                linewidth=0.85,
                alpha=0.9,
                label=f"{mode}, {_format_gpu_count(gpu)}",
            )
            for _, row in gpu_rows.iterrows():
                ax.annotate(
                    f"b{int(row['p_solver_max_batch_size'])}",
                    (
                        row["relative_efficiency_pct"],
                        row["max_gpu_mb"] / 1024,
                    ),
                    xytext=(6, 5),
                    textcoords="offset points",
                    fontsize=9,
                    color="#374151",
                )

    style_axes(
        ax,
        "Relative Runtime/Memory Tradeoff",
        "Relative step-time efficiency (%; higher is faster)",
        "Max GPU memory / GPU (GB)",
    )
    ax.axvline(100, color="#111827", linestyle="--", linewidth=1.2, alpha=0.45)
    ax.set_xlim(
        95,
        max(105, float(summary["relative_efficiency_pct"].max()) * 1.06),
    )
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Checkpoint mode, hardware",
    )
    fig.suptitle(
        f"Checkpointing - {problem_size_title(summary)}",
        x=0.02,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.78, 0.91))
    return write_figure(fig, output_path, "checkpoint_tradeoff.png")


def _format_gpu_count(gpu: int) -> str:
    count = int(gpu)
    return "1 GPU" if count == 1 else f"{count} GPUs"
