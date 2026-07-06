"""Shared helpers for benchmark visualization modules."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INFERENCE_OUTPUT_DIR = Path("visualizations") / "inference"
DEFAULT_TRAINING_OUTPUT_DIR = Path("visualizations") / "training"
DEFAULT_OUTPUT_DIR = DEFAULT_INFERENCE_OUTPUT_DIR
TIMING_WARMUP_ITERATIONS = 2

COLORWAY = [
    "#2563eb",
    "#f97316",
    "#16a34a",
    "#dc2626",
    "#7c3aed",
    "#0891b2",
    "#ca8a04",
    "#db2777",
]

INFERENCE_REQUIRED_COLUMNS = {
    "p_dataset_image_size",
    "p_solver_slurm_nodes",
    "p_solver_slurm_ntasks_per_node",
    "p_solver_slurm_gres",
    "p_solver_patch_size",
    "p_solver_overlap",
    "stop_val",
    "time",
    "objective_psnr",
}

TRAINING_REQUIRED_COLUMNS = {
    "p_dataset_image_size",
    "p_solver_slurm_nodes",
    "p_solver_slurm_ntasks_per_node",
    "p_solver_slurm_gres",
    "p_solver_patch_size",
    "p_solver_overlap",
    "p_solver_max_batch_size",
    "p_solver_checkpoint_batches",
    "stop_val",
    "time",
    "objective_psnr",
}

CONFIG_COLUMNS = [
    "p_dataset_image_size",
    "p_solver_patch_size",
    "p_solver_overlap",
    "p_solver_max_batch_size",
    "p_solver_distribute_physics",
    "p_solver_distribute_denoiser",
    "p_solver_slurm_nodes",
    "p_solver_slurm_ntasks_per_node",
    "p_solver_slurm_gres",
    "n_gpus",
]

TRAINING_CONFIG_COLUMNS = [
    "p_solver_image_size",
    "p_dataset_image_size",
    "p_dataset_batch_size",
    "p_solver_patch_size",
    "p_solver_overlap",
    "p_solver_max_batch_size",
    "p_solver_checkpoint_batches",
    "p_solver_distribute_model",
    "p_solver_slurm_nodes",
    "p_solver_slurm_ntasks_per_node",
    "p_solver_slurm_gres",
    "n_gpus",
]


def load_results(
    results: str | Path,
    *,
    required: set[str] | None = INFERENCE_REQUIRED_COLUMNS,
) -> tuple[pd.DataFrame, Path]:
    """Load a benchopt parquet file or the newest parquet inside a directory."""
    path = Path(results)
    if path.is_dir():
        parquet_files = sorted(path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        path = parquet_files[-1]
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    if required:
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Missing expected result columns: {', '.join(missing)}")

    df = df.copy()
    add_hardware_columns(df)
    if "p_dataset_image_size" in df.columns:
        df["image_mpix"] = (df["p_dataset_image_size"].astype(float) ** 2) / 1_000_000
    df["config_label"] = df.apply(config_label, axis=1)
    df["method_label"] = df.apply(method_label, axis=1)
    return df, path


def add_hardware_columns(df: pd.DataFrame) -> None:
    """Add parsed GPU/node columns in place."""
    df["gpus_per_node"] = df["p_solver_slurm_gres"].map(parse_gpus_from_gres)
    df["n_nodes"] = df["p_solver_slurm_nodes"].astype(int)
    df["n_gpus"] = df["n_nodes"] * df["gpus_per_node"]


def parse_gpus_from_gres(gres: object) -> int:
    """Parse Slurm GRES strings such as gpu:4."""
    if isinstance(gres, str) and ":" in gres:
        try:
            return int(gres.rsplit(":", maxsplit=1)[1])
        except ValueError:
            return 1
    return 1


def config_label(row: pd.Series) -> str:
    """Compact label for a distributed configuration."""
    n_gpus = int(row["n_gpus"])
    n_nodes = int(row["n_nodes"])
    tasks = int(row["p_solver_slurm_ntasks_per_node"])
    if n_gpus == 1 and not row.get("p_solver_distribute_physics", False):
        return "1 GPU reference"
    node_label = "node" if n_nodes == 1 else "nodes"
    return f"{n_gpus} GPUs ({n_nodes} {node_label}, {tasks} tasks/node)"


def method_label(row: pd.Series) -> str:
    """Compact label for patch/overlap choices."""
    patch = int(row["p_solver_patch_size"])
    overlap = int(row["p_solver_overlap"])
    if patch == 0:
        return "non-distributed"
    return f"patch {patch}, overlap {overlap}"


def summarize_configs(df: pd.DataFrame) -> pd.DataFrame:
    """Create one inference summary row per image/configuration."""
    config_cols = [col for col in CONFIG_COLUMNS if col in df.columns]
    final_rows = df.copy()
    final_rows["max_iter"] = final_rows.groupby(config_cols, dropna=False)[
        "stop_val"
    ].transform("max")
    final_rows = final_rows[final_rows["stop_val"] == final_rows["max_iter"]]

    final_agg = (
        final_rows.groupby(config_cols, dropna=False)
        .agg(
            max_iter=("max_iter", "max"),
            final_time_sec=("time", "mean"),
            final_psnr_db=("objective_psnr", "mean"),
            final_ssim=("objective_ssim", "mean")
            if "objective_ssim" in final_rows.columns
            else ("objective_psnr", "size"),
        )
        .reset_index()
    )

    step_df = df[df["stop_val"] > TIMING_WARMUP_ITERATIONS].copy()
    step_agg = (
        step_df.groupby(config_cols, dropna=False)
        .agg(
            avg_total_time_sec=("objective_total_time_sec", "mean"),
            avg_gradient_time_sec=("objective_gradient_time_sec", "mean"),
            avg_denoise_time_sec=("objective_denoise_time_sec", "mean"),
            max_gpu_mb=("objective_max_gpu_mb", "max"),
            timing_start_iter=("stop_val", "min"),
            timing_end_iter=("stop_val", "max"),
            timing_num_iters=("stop_val", "count"),
            config_label=("config_label", "first"),
            method_label=("method_label", "first"),
            n_nodes=("n_nodes", "first"),
            gpus_per_node=("gpus_per_node", "first"),
            image_mpix=("image_mpix", "first"),
        )
        .reset_index()
    )
    return final_agg.merge(step_agg, on=config_cols, how="left")


def summarize_training_configs(df: pd.DataFrame) -> pd.DataFrame:
    """Create one training summary row per solver/hardware configuration."""
    config_cols = [col for col in TRAINING_CONFIG_COLUMNS if col in df.columns]
    final_rows = df.copy()
    final_rows["max_iter"] = final_rows.groupby(config_cols, dropna=False)[
        "stop_val"
    ].transform("max")
    final_rows = final_rows[final_rows["stop_val"] == final_rows["max_iter"]]
    final_agg = (
        final_rows.groupby(config_cols, dropna=False)
        .agg(
            max_iter=("max_iter", "max"),
            final_time_sec=("time", "mean"),
            final_psnr_db=("objective_psnr", "mean"),
        )
        .reset_index()
    )

    step_df = df[df["objective_total_time_sec"].notna()].copy()
    if step_df.empty:
        raise ValueError("No per-step timing rows found in training results.")

    aggregations = {
        "avg_total_time_sec": ("objective_total_time_sec", "mean"),
        "avg_forward_time_sec": ("objective_forward_time_sec", "mean"),
        "avg_backward_time_sec": ("objective_backward_time_sec", "mean"),
        "max_gpu_mb": ("objective_max_gpu_mb", "max"),
        "timing_start_iter": ("stop_val", "min"),
        "timing_end_iter": ("stop_val", "max"),
        "timing_num_iters": ("stop_val", "count"),
        "config_label": ("config_label", "first"),
        "n_nodes": ("n_nodes", "first"),
        "gpus_per_node": ("gpus_per_node", "first"),
    }
    for output_col, input_col in [
        ("max_forward_gpu_mb", "objective_forward_max_gpu_mb"),
        ("max_backward_gpu_mb", "objective_backward_max_gpu_mb"),
    ]:
        if input_col in step_df.columns:
            aggregations[output_col] = (input_col, "max")

    for col in [
        "objective_comm_time_sec",
        "objective_communication_time_sec",
        "objective_all_reduce_time_sec",
        "objective_sync_time_sec",
        "objective_forward_cuda_sec",
        "objective_forward_cpu_sec",
        "objective_backward_cuda_sec",
        "objective_backward_cpu_sec",
        "objective_forward_comm_sec",
        "objective_backward_comm_sec",
        "objective_comm_cuda_sec",
        "objective_comm_sync_sec",
    ]:
        if col in step_df.columns:
            aggregations[col] = (col, "mean")

    step_agg = (
        step_df.groupby(config_cols, dropna=False).agg(**aggregations).reset_index()
    )
    summary = final_agg.merge(step_agg, on=config_cols, how="left")
    summary["avg_other_time_sec"] = np.clip(
        summary["avg_total_time_sec"]
        - summary["avg_forward_time_sec"].fillna(0)
        - summary["avg_backward_time_sec"].fillna(0),
        a_min=0,
        a_max=None,
    )
    summary["training_image_size"] = summary.apply(training_image_size, axis=1)
    summary["training_mpix"] = (
        summary["training_image_size"].astype(float) ** 2
    ) / 1_000_000
    summary["effective_batch_size"] = summary["p_solver_max_batch_size"].astype(float)
    summary["throughput_per_sec"] = (
        summary["effective_batch_size"] / summary["avg_total_time_sec"]
    )
    return summary.sort_values(
        ["training_image_size", "p_solver_max_batch_size", "n_gpus"]
    )


def best_per_gpu(summary: pd.DataFrame) -> pd.DataFrame:
    """Select one representative configuration per GPU count."""
    sort_col = "avg_total_time_sec"
    if sort_col not in summary.columns:
        sort_col = "final_time_sec"
    return (
        summary.sort_values(["n_gpus", sort_col], na_position="last")
        .groupby("n_gpus", as_index=False, dropna=False)
        .first()
        .sort_values("n_gpus")
    )


def training_image_size(row: pd.Series) -> int:
    """Return the logical training image size for mixed result schemas."""
    solver_size = row.get("p_solver_image_size")
    if pd.notna(solver_size):
        return int(solver_size)
    return int(row["p_dataset_image_size"])


def format_image_size(size: object) -> str:
    """Format square image sizes for labels."""
    try:
        size_int = int(size)
    except (TypeError, ValueError):
        return str(size)
    return f"{size_int}x{size_int}"


def configure_matplotlib() -> None:
    """Apply a consistent style for presentation-friendly static plots."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d1d5db",
            "axes.labelcolor": "#111827",
            "axes.titlecolor": "#111827",
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply common labels and remove visual clutter."""
    ax.set_title(title, loc="left", pad=14, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#374151")


def set_gpu_axis(ax, values: pd.Series | np.ndarray | list[float]) -> None:
    """Use a log2 GPU axis when more than one distinct GPU count is present."""
    gpus = sorted({int(v) for v in values})
    if len(gpus) <= 1:
        ax.set_xticks(gpus)
        ax.set_xticklabels([str(v) for v in gpus])
        return
    ax.set_xscale("log", base=2)
    ax.set_xticks(gpus)
    ax.set_xticklabels([str(v) for v in gpus])


def write_figure(fig, output_dir: Path, filename: str, *, dpi: int = 220) -> str:
    """Write a Matplotlib figure and return its local filename."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=dpi)
    plt.close(fig)
    return filename
