"""Visualizations for distributed reconstruction-quality experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from toolsbench.visualization.common import (
    COLORWAY,
    DEFAULT_OUTPUT_DIR,
    TIMING_WARMUP_ITERATIONS,
    configure_matplotlib,
    format_image_size,
    load_results,
    style_axes,
    summarize_configs,
    write_figure,
)


def create_quality_visualizations(
    results: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Create quality visualizations from a benchopt inference result parquet."""
    configure_matplotlib()
    df, results_path = load_results(results)
    quality = _with_reference_delta(df)
    summary = _quality_summary(df, quality)

    output_path = Path(output_dir) / f"quality_{results_path.stem}"
    output_path.mkdir(parents=True, exist_ok=True)
    _remove_stale_outputs(
        output_path,
        [
            "delta_psnr_iterations.png",
            "final_delta_vs_overlap.png",
        ],
    )

    _plot_overlap_tradeoff(summary, output_path)
    return output_path


def _remove_stale_outputs(output_path: Path, filenames: list[str]) -> None:
    """Remove plots that are no longer produced by the current CLI."""
    for filename in filenames:
        path = output_path / filename
        if path.exists():
            path.unlink()


def _reference_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["n_gpus"] == 1)
        & (df["p_solver_patch_size"] == 0)
        & (~df["p_solver_distribute_physics"])
        & (~df["p_solver_distribute_denoiser"])
    )


def _with_reference_delta(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[_reference_mask(df)].copy()
    if baseline.empty:
        raise ValueError("No non-distributed 1-GPU reference run found.")

    baseline_cols = [
        "p_dataset_image_size",
        "stop_val",
        "objective_psnr",
        "objective_ssim",
        "objective_mse",
    ]
    baseline = (
        baseline[baseline_cols]
        .groupby(["p_dataset_image_size", "stop_val"], as_index=False)
        .mean(numeric_only=True)
        .rename(
            columns={
                "objective_psnr": "reference_psnr",
                "objective_ssim": "reference_ssim",
                "objective_mse": "reference_mse",
            }
        )
    )
    merged = df.merge(baseline, on=["p_dataset_image_size", "stop_val"], how="inner")
    merged = merged[~_reference_mask(merged)].copy()
    merged["delta_psnr_db"] = merged["objective_psnr"] - merged["reference_psnr"]
    if "objective_ssim" in merged.columns:
        merged["delta_ssim"] = merged["objective_ssim"] - merged["reference_ssim"]
    if "objective_mse" in merged.columns:
        merged["relative_mse_pct"] = (
            (merged["objective_mse"] - merged["reference_mse"])
            / merged["reference_mse"]
            * 100
        )
    return merged


def _quality_summary(df: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    config_summary = summarize_configs(df)
    max_iter = (
        quality.groupby(
            ["p_dataset_image_size", "p_solver_overlap", "p_solver_patch_size"],
            dropna=False,
        )["stop_val"]
        .max()
        .rename("max_iter")
        .reset_index()
    )
    final_quality = quality.merge(
        max_iter,
        on=["p_dataset_image_size", "p_solver_overlap", "p_solver_patch_size"],
        how="inner",
    )
    final_quality = final_quality[
        final_quality["stop_val"] == final_quality["max_iter"]
    ]
    final_quality = (
        final_quality.groupby(
            ["p_dataset_image_size", "p_solver_overlap", "p_solver_patch_size"],
            as_index=False,
            dropna=False,
        )
        .agg(
            delta_psnr_db=("delta_psnr_db", "mean"),
            objective_psnr=("objective_psnr", "mean"),
            reference_psnr=("reference_psnr", "mean"),
        )
    )
    summary = config_summary.merge(
        final_quality,
        on=["p_dataset_image_size", "p_solver_overlap", "p_solver_patch_size"],
        how="inner",
    )
    return summary.sort_values(["p_dataset_image_size", "p_solver_overlap"])


def _plot_overlap_tradeoff(summary: pd.DataFrame, output_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for idx, (image_size, group) in enumerate(
        summary.groupby("p_dataset_image_size", dropna=False)
    ):
        group = group.sort_values("p_solver_overlap")
        frontier = _pareto_frontier(group)
        dominated = group.loc[~group.index.isin(frontier.index)]
        color = COLORWAY[idx % len(COLORWAY)]

        if not dominated.empty:
            ax.scatter(
                dominated["avg_total_time_sec"],
                dominated["delta_psnr_db"],
                s=58,
                color=color,
                alpha=0.32,
                edgecolor="none",
            )
        ax.scatter(
            frontier["avg_total_time_sec"],
            frontier["delta_psnr_db"],
            s=82,
            color=color,
            alpha=0.92,
            edgecolor="#111827",
            linewidth=0.9,
            label=format_image_size(image_size),
        )
        ax.plot(
            frontier["avg_total_time_sec"],
            frontier["delta_psnr_db"],
            color=color,
            linewidth=2.8,
            alpha=0.9,
        )
        ax.fill_between(
            frontier["avg_total_time_sec"],
            frontier["delta_psnr_db"],
            group["delta_psnr_db"].min(),
            color=color,
            alpha=0.08,
            linewidth=0,
        )
        for _, row in group.iterrows():
            ax.annotate(
                str(int(row["p_solver_overlap"])),
                (row["avg_total_time_sec"], row["delta_psnr_db"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#374151",
            )
    ax.axhline(0, color="#111827", linestyle="--", linewidth=1.2, alpha=0.65)
    style_axes(
        ax,
        "Overlap Quality/Runtime Tradeoff",
        (
            f"Average time / iteration (s), "
            f"after {TIMING_WARMUP_ITERATIONS} warmup iterations"
        ),
        "Final delta PSNR vs reference (dB)",
    )
    ax.legend(loc="lower right", title="Highlighted Pareto frontier")
    fig.tight_layout()
    return write_figure(fig, output_path, "overlap_tradeoff.png")


def _pareto_frontier(group: pd.DataFrame) -> pd.DataFrame:
    """Return non-dominated points minimizing time and maximizing PSNR delta."""
    rows = group.sort_values(["avg_total_time_sec", "delta_psnr_db"]).copy()
    best_delta = -float("inf")
    frontier_indices = []
    for index, row in rows.iterrows():
        if row["delta_psnr_db"] >= best_delta:
            frontier_indices.append(index)
            best_delta = row["delta_psnr_db"]
    return rows.loc[frontier_indices]
