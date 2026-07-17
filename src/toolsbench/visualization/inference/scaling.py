"""Visualizations for distributed inference scaling experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolsbench.visualization.common import (
    COLORWAY,
    DEFAULT_OUTPUT_DIR,
    TIMING_WARMUP_ITERATIONS,
    best_per_gpu,
    configure_matplotlib,
    format_image_size,
    load_results,
    style_axes,
    summarize_configs,
    write_figure,
)


def create_scaling_visualizations(
    results: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Create scaling visualizations from a benchopt inference result parquet."""
    configure_matplotlib()
    df, results_path = load_results(results)
    summary = summarize_configs(df)

    output_path = Path(output_dir) / f"scaling_{results_path.stem}"
    output_path.mkdir(parents=True, exist_ok=True)
    _remove_stale_outputs(
        output_path,
        [
            "step_time_breakdown.png",
            "memory_usage.png",
            "runtime_by_size.png",
        ],
    )

    _plot_strong_scaling(summary, output_path)
    _plot_weak_scaling(summary, output_path)
    _plot_timing_breakdown(summary, output_path)
    return output_path


def _remove_stale_outputs(output_path: Path, filenames: list[str]) -> None:
    """Remove plots that are no longer produced by the current CLI."""
    for filename in filenames:
        path = output_path / filename
        if path.exists():
            path.unlink()


def _plot_strong_scaling(summary: pd.DataFrame, output_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    max_efficiency = 100.0
    all_gpus = sorted(summary["n_gpus"].unique())
    for idx, (image_size, group) in enumerate(
        summary.groupby("p_dataset_image_size", dropna=False)
    ):
        rows = best_per_gpu(group.copy())
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

        label = f"{format_image_size(image_size)} ({baseline_gpus}-GPU baseline)"
        color = COLORWAY[idx % len(COLORWAY)]
        ax.plot(
            rows["n_gpus"],
            rows["efficiency_pct"],
            marker="o",
            markersize=7,
            linewidth=2.8,
            color=color,
            label=label,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(all_gpus)
    ax.set_xticklabels([str(int(gpu)) for gpu in all_gpus])
    ax.axhline(100, color="#111827", linestyle="--", linewidth=1.3, alpha=0.55)
    ax.set_ylim(0, max(110, max_efficiency * 1.12))
    style_axes(
        ax,
        "Strong Scaling Efficiency",
        "Number of GPUs",
        "Parallel efficiency vs smallest feasible GPU count (%)",
    )
    ax.legend(loc="upper right", ncols=1)
    ax.text(
        0.99,
        0.03,
        f"100% means ideal scaling from each curve's baseline. Timings skip "
        f"first {TIMING_WARMUP_ITERATIONS} iterations.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#6b7280",
        fontsize=10,
    )
    fig.tight_layout()
    return write_figure(fig, output_path, "strong_scaling.png")


def _plot_weak_scaling(summary: pd.DataFrame, output_path: Path) -> str:
    weak = summary.copy()
    weak["local_workload_mpix"] = weak["image_mpix"] / weak["n_gpus"]
    weak["local_workload_label"] = weak["local_workload_mpix"].round(2)

    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    workloads = (
        weak.groupby("local_workload_label")
        .filter(lambda group: len(group) >= 2)["local_workload_label"]
        .drop_duplicates()
        .sort_values(ascending=False)
    )
    for idx, workload in enumerate(workloads):
        rows = (
            weak[weak["local_workload_label"] == workload]
            .sort_values("n_gpus")
            .copy()
        )
        baseline_time = float(rows.iloc[0]["avg_total_time_sec"])
        rows["time_ratio"] = rows["avg_total_time_sec"] / baseline_time
        color = COLORWAY[idx % len(COLORWAY)]
        ax.plot(
            rows["n_gpus"],
            rows["time_ratio"],
            marker="o",
            markersize=7,
            linewidth=2.6,
            color=color,
            label=f"{workload:g} Mpix/GPU",
        )

    ax.set_xscale("log", base=2)
    all_gpus = sorted(weak["n_gpus"].unique())
    ax.set_xticks(all_gpus)
    ax.set_xticklabels([str(int(gpu)) for gpu in all_gpus])
    ax.axhline(1, color="#111827", linestyle="--", linewidth=1.3, alpha=0.55)
    style_axes(
        ax,
        "Weak Scaling Runtime Ratio",
        "Number of GPUs",
        "Average iteration time / smallest-GPU time",
    )
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Local workload",
        ncols=1,
    )
    fig.tight_layout()
    return write_figure(fig, output_path, "weak_scaling.png")


def _plot_timing_breakdown(summary: pd.DataFrame, output_path: Path) -> str:
    sizes = sorted(summary["p_dataset_image_size"].unique())
    fig, axes = plt.subplots(
        1,
        len(sizes),
        figsize=(5.4 * len(sizes), 5.7),
        sharey=False,
        squeeze=False,
    )
    band_specs = [
        ("physics", "#2563eb"),
        ("denoising", "#f97316"),
        ("residual overhead", "#16a34a"),
    ]

    for col, image_size in enumerate(sizes):
        ax = axes[0, col]
        rows = best_per_gpu(
            summary[summary["p_dataset_image_size"] == image_size].copy()
        ).copy()
        x = rows["n_gpus"].to_numpy(dtype=float)
        gradient = rows["avg_gradient_time_sec"].fillna(0).to_numpy(dtype=float)
        denoising = rows["avg_denoise_time_sec"].fillna(0).to_numpy(dtype=float)
        total = rows["avg_total_time_sec"].fillna(0).to_numpy(dtype=float)
        measured = gradient + denoising
        residual = np.clip(total - measured, a_min=0, a_max=None)
        stacked = np.vstack([gradient, denoising, residual])

        ax.stackplot(
            x,
            stacked,
            labels=[label for label, _ in band_specs],
            colors=[color for _, color in band_specs],
            alpha=0.34,
            linewidth=0,
        )
        ax.plot(
            x,
            gradient,
            marker="o",
            markersize=4.8,
            linewidth=2.0,
            color="#2563eb",
            label="physics boundary",
        )
        ax.plot(
            x,
            measured,
            marker="o",
            markersize=4.8,
            linewidth=2.4,
            color="#f97316",
            label="physics + denoising",
        )
        ax.plot(
            x,
            total,
            marker="o",
            markersize=5.5,
            linewidth=3.0,
            color="#16a34a",
            label="total iteration",
        )

        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(gpu)) for gpu in x])
        style_axes(
            ax,
            format_image_size(image_size),
            "Number of GPUs",
            "Average time / iteration (s)" if col == 0 else "",
        )
        ax.text(
            0.96,
            0.92,
            (
                f"iters {int(rows['timing_start_iter'].min())}-"
                f"{int(rows['timing_end_iter'].max())}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="#6b7280",
            fontsize=10,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_items = [
        (handle, label)
        for handle, label in zip(handles, labels, strict=False)
        if label in {"physics", "denoising", "residual overhead", "total iteration"}
    ]
    handles, labels = zip(*legend_items)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=3,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Inference Time Decomposition by Problem Size",
        x=0.02,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.99,
        0.02,
        "Green band is total iteration time not explained by physics + denoising.",
        ha="right",
        va="bottom",
        color="#6b7280",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    return write_figure(fig, output_path, "timing_breakdown_by_size.png")
