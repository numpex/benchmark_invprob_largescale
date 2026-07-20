"""Shared helpers for benchmark visualization modules."""

from __future__ import annotations

import pickle
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

# benchopt stores non-scalar parameters (a 3D image_size, a per-axis patch_size)
# as a pickled blob behind this marker.
BENCHOPT_PKL_PREFIX = b"\x00benchopt-pkl\x00"

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
        # From the decoded dims, so a 3D volume works as well as a 2D square.
        df["image_mpix"] = df["p_dataset_image_size"].map(
            lambda size: float(np.prod(normalize_size(size))) / 1_000_000
        )
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


def decode_param(value):
    """Decode a benchopt parameter, which pickles non-scalar values into bytes."""
    if isinstance(value, (bytes, bytearray)) and value.startswith(BENCHOPT_PKL_PREFIX):
        return pickle.loads(bytes(value)[len(BENCHOPT_PKL_PREFIX) :])
    return value


def normalize_size(size) -> tuple[int, ...]:
    """Spatial-size tuple from an image/patch param (int, [s], [h, w] or [d, h, w])."""
    size = decode_param(size)
    if isinstance(size, (list, tuple, np.ndarray)):
        dims = tuple(int(s) for s in size)
        return dims * 2 if len(dims) == 1 else dims
    return (int(size), int(size))


def size_label(size) -> str:
    """Format an image/patch size for labels, e.g. ``4096x4096`` or ``8x256x256``."""
    return "x".join(str(d) for d in normalize_size(size))


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
    """Compact label for patch/overlap choices.

    Patch and overlap may be per-axis (a 3D run), in which case they are shown as
    ``8x256x256`` rather than a single number.
    """
    patch = decode_param(row["p_solver_patch_size"])
    overlap = decode_param(row["p_solver_overlap"])
    if isinstance(patch, (list, tuple, np.ndarray)):
        return f"patch {size_label(patch)}, overlap {size_label(overlap)}"
    if int(patch) == 0:
        return "non-distributed"
    return f"patch {int(patch)}, overlap {int(overlap)}"


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
