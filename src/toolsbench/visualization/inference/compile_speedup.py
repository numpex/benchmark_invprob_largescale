"""Visualizations comparing torch.compile modes for inference.

Two independent entry points, one per experiment config:

- :func:`create_compile_speedup_visualizations` — PnP ``tomography_2d_compile``:
  1st-iteration vs steady-state cost across compile modes (None/pre/post/fused).
- :func:`create_denoiser_compile_visualizations` — ``denoiser_compile``:
  steady-state eager-vs-compiled speedup per denoiser and shape (2D/3D), plus
  an optional roofline scatter (arithmetic intensity vs speedup). Results that
  sweep several GPU architectures get one panel per GPU; older single-GPU
  results render as a single panel, unchanged.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from toolsbench.visualization.common import (
    COLORWAY,
    DEFAULT_OUTPUT_DIR,
    configure_matplotlib,
    decode_param,
    load_results,
    normalize_size,
    style_axes,
    write_figure,
)

STABLE_WINDOW = 5
COMPILE_ORDER = ["None", "pre", "post", "fused"]

# Marker per spatial dimensionality, shared by the denoiser speedup/roofline plots.
DIM_MARKERS = {2: "o", 3: "s"}

# Roofline hardware reference, keyed by the Slurm constraint recorded in the
# results: (peak TF32 tensor-core TFLOPS, peak memory bandwidth TB/s).
GPU_PEAKS = {
    "h100": (494.0, 3.35),   # HBM3
    "a100": (156.0, 2.039),  # HBM2e
}

# The FLOP/byte where a device switches from memory- to compute-bound; points
# left of it are memory-bound, right are compute-bound.
GPU_RIDGE = {gpu: tflops / bw for gpu, (tflops, bw) in GPU_PEAKS.items()}

RIDGE_POINT = GPU_RIDGE["h100"]

UNKNOWN_GPU = ""


def create_compile_speedup_visualizations(
    results: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Create the 1st-iteration vs stable-iteration compile-speedup bar chart."""
    configure_matplotlib()
    df, results_path = load_results(results)
    if "objective_total_time_sec" not in df.columns:
        raise ValueError("Results are missing 'objective_total_time_sec'.")

    summary = _compile_summary(df)

    output_path = Path(output_dir) / f"compile_speedup_{results_path.stem}"
    output_path.mkdir(parents=True, exist_ok=True)
    _plot_compile_speedup(summary, output_path)
    return output_path


def _per_run_timings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(compile_mode, n_gpus, idx_rep) first- and stable-iteration timing."""
    df = df.copy()
    df["compile_mode"] = df["p_solver_compile"].fillna("None").astype(str)

    rows = []
    for (compile_mode, n_gpus, idx_rep), group in df.groupby(
        ["compile_mode", "n_gpus", "idx_rep"], dropna=False
    ):
        timed = group.dropna(subset=["objective_total_time_sec"]).sort_values("stop_val")
        if timed.empty:
            continue
        rows.append(
            {
                "compile_mode": compile_mode,
                "n_gpus": int(n_gpus),
                "idx_rep": idx_rep,
                "first_iter_sec": float(timed.iloc[0]["objective_total_time_sec"]),
                "stable_iter_sec": float(
                    timed.tail(STABLE_WINDOW)["objective_total_time_sec"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _compile_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run timings into mean/std across repetitions, with speedup."""
    per_run = _per_run_timings(df)
    if per_run.empty:
        raise ValueError("No timed iterations found for compile-speedup comparison.")

    summary = per_run.groupby(["compile_mode", "n_gpus"], as_index=False).agg(
        first_iter_mean=("first_iter_sec", "mean"),
        first_iter_std=("first_iter_sec", "std"),
        stable_iter_mean=("stable_iter_sec", "mean"),
        stable_iter_std=("stable_iter_sec", "std"),
    )
    summary[["first_iter_std", "stable_iter_std"]] = summary[
        ["first_iter_std", "stable_iter_std"]
    ].fillna(0.0)

    baseline_rows = summary[
        (summary["n_gpus"] == summary["n_gpus"].min())
        & (summary["compile_mode"] == "None")
    ]
    if baseline_rows.empty:
        raise ValueError("No 1-GPU compile=None baseline run found.")
    baseline_stable = float(baseline_rows.iloc[0]["stable_iter_mean"])
    summary["speedup"] = baseline_stable / summary["stable_iter_mean"]

    summary["compile_order"] = summary["compile_mode"].apply(
        lambda mode: COMPILE_ORDER.index(mode) if mode in COMPILE_ORDER else len(COMPILE_ORDER)
    )
    return summary.sort_values(["n_gpus", "compile_order"]).reset_index(drop=True)


def _plot_compile_speedup(summary: pd.DataFrame, output_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(11.5, 6.2))

    x = np.arange(len(summary))
    width = 0.36

    ax.bar(
        x - width / 2,
        summary["first_iter_mean"],
        width,
        yerr=summary["first_iter_std"],
        capsize=4,
        color=COLORWAY[0],
        label="1st iteration",
    )
    bars_stable = ax.bar(
        x + width / 2,
        summary["stable_iter_mean"],
        width,
        yerr=summary["stable_iter_std"],
        capsize=4,
        color=COLORWAY[1],
        label="stable",
    )

    for bar, speedup in zip(bars_stable, summary["speedup"], strict=False):
        ax.annotate(
            f"{speedup:.2f}x",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color="#111827",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{int(row.n_gpus)} GPU\n{row.compile_mode}" for row in summary.itertuples()]
    )
    ax.set_yscale("log")
    style_axes(
        ax,
        "torch.compile: 1st-Iteration vs Stable-Iteration Cost",
        "Configuration",
        "Time / iteration (s, log scale)",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    return write_figure(fig, output_path, "compile_speedup.png")


# ---------------------------------------------------------------------------
# Denoiser compile benchmark (denoiser_compile config)
# ---------------------------------------------------------------------------

def _shape_label(size) -> str:
    dims = normalize_size(size)
    if len(dims) == 3:
        d, h, w = dims
        return f"{d}³" if d == h == w else f"{d}×{h}×{w}"
    return str(dims[0]) if dims[0] == dims[1] else f"{dims[0]}×{dims[1]}"


def _dim_of(size) -> int:
    return 3 if len(normalize_size(size)) == 3 else 2


def _area(size) -> int:
    dims = normalize_size(size)
    prod = 1
    for d in dims:
        prod *= d
    return prod


def _load_denoiser_df(results: str | Path) -> tuple[pd.DataFrame, Path]:
    """Load the denoiser parquet directly.

    Unlike :func:`load_results`, this does not assume a scalar square
    ``image_size`` (which would break on 3D/list sizes stored as pickled bytes).
    """
    path = Path(results)
    if path.is_dir():
        parquet_files = sorted(path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        path = parquet_files[-1]
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path), path


def _gpu_series(df: pd.DataFrame) -> pd.Series:
    """GPU architecture per row, from the Slurm constraint the run was pinned to.

    Results produced before the multi-architecture sweep have no such column;
    they collapse to a single :data:`UNKNOWN_GPU` group.
    """
    if "p_solver_slurm_constraint" not in df.columns:
        return pd.Series(UNKNOWN_GPU, index=df.index)
    return df["p_solver_slurm_constraint"].fillna(UNKNOWN_GPU).astype(str)


def _denoiser_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (gpu, denoiser, shape, dim) with eager/compiled time and speedup.

    ``gpu`` is a grouping key, not a cosmetic label: without it the eager and
    compiled times of different GPUs are averaged together and the resulting
    speedup describes no real machine.
    """
    df = df.copy()
    df["compile_mode"] = df["p_solver_compile"].fillna("None").astype(str)
    df["gpu"] = _gpu_series(df)
    df["denoiser"] = df["p_solver_denoiser"].astype(str)
    size = df["p_dataset_image_size"].map(decode_param)
    df["shape"] = size.map(_shape_label)
    df["dim"] = size.map(_dim_of)
    df["area"] = size.map(_area)

    has_ai = "objective_arith_intensity" in df.columns

    rows = []
    for (gpu, denoiser, shape, dim, area, mode, _rep), group in df.groupby(
        ["gpu", "denoiser", "shape", "dim", "area", "compile_mode", "idx_rep"],
        dropna=False,
    ):
        timed = group.dropna(subset=["objective_denoise_time_sec"]).sort_values("stop_val")
        if timed.empty:
            continue
        rows.append(
            {
                "gpu": gpu,
                "denoiser": denoiser,
                "shape": shape,
                "dim": int(dim),
                "area": int(area),
                "compile_mode": mode,
                "stable_sec": float(
                    timed.tail(STABLE_WINDOW)["objective_denoise_time_sec"].mean()
                ),
                "arith_intensity": (
                    float(timed["objective_arith_intensity"].dropna().mean())
                    if has_ai else float("nan")
                ),
            }
        )
    per_run = pd.DataFrame(rows)
    if per_run.empty:
        return per_run

    keys = ["gpu", "denoiser", "shape", "dim", "area"]
    time_by_mode = per_run.pivot_table(
        index=keys, columns="compile_mode", values="stable_sec", aggfunc="mean"
    )
    ai = per_run.groupby(keys, as_index=False)["arith_intensity"].mean()

    summary = time_by_mode.reset_index().merge(ai, on=keys, how="left")
    if "None" not in summary.columns or "pre" not in summary.columns:
        raise ValueError("Need both compile=None and compile=pre runs for speedup.")
    summary = summary.dropna(subset=["None", "pre"])
    summary = summary.rename(columns={"None": "eager_sec", "pre": "compiled_sec"})
    summary["speedup"] = summary["eager_sec"] / summary["compiled_sec"]
    return summary.sort_values(["gpu", "dim", "area", "denoiser"]).reset_index(drop=True)


def _denoiser_colors(summary: pd.DataFrame) -> dict[str, str]:
    return {
        d: COLORWAY[i % len(COLORWAY)]
        for i, d in enumerate(sorted(summary["denoiser"].unique()))
    }


def _legend_handles(colors: dict[str, str], styles: dict[str, str] | None = None):
    denoiser_handles = [
        Line2D([0], [0], color=c, marker="o", markersize=9, label=d)
        for d, c in colors.items()
    ]
    dim_handles = [
        Line2D([0], [0], color="gray", marker=m, linestyle="None", markersize=9, label=f"{d}D")
        for d, m in DIM_MARKERS.items()
    ]
    gpu_handles = [
        Line2D([0], [0], color="#374151", linestyle=s, lw=2, label=g.upper())
        for g, s in (styles or {}).items()
    ]
    return denoiser_handles + dim_handles + gpu_handles


def _gpus_of(summary: pd.DataFrame) -> list[str]:
    """GPUs present in the results; a single unnamed group means legacy data."""
    return sorted(summary["gpu"].unique())


# Line style is the GPU channel on the merged speedup plot, leaving colour for
# the denoiser and marker for the dimensionality.
GPU_LINESTYLES = ["-", "--", ":", "-."]

_SPEEDUP_TITLE = "torch.compile Steady-State Speedup (eager / compiled)"


def _plot_denoiser_speedup(summary: pd.DataFrame, output_path: Path) -> str:
    """All GPUs on one axes: colour = denoiser, marker = dim, line style = GPU."""
    colors = _denoiser_colors(summary)
    gpus = _gpus_of(summary)
    styles = {g: GPU_LINESTYLES[i % len(GPU_LINESTYLES)] for i, g in enumerate(gpus)}
    shapes = (
        summary[["shape", "dim", "area"]]
        .drop_duplicates()
        .sort_values(["dim", "area"])
    )
    labels = shapes["shape"].tolist()
    xpos = {lab: i for i, lab in enumerate(labels)}
    n2d = int((shapes["dim"] == 2).sum())

    # With every GPU on one axes, the same denoiser's curves converge and their
    # labels collide; alternate the offset so each GPU annotates a different side.
    if len(gpus) == 1:
        offsets = {gpus[0]: (0, 8)}
    else:
        offsets = {g: ((0, 8) if i % 2 else (0, -15)) for i, g in enumerate(gpus)}

    fig, ax = plt.subplots(figsize=(11 if len(gpus) == 1 else 13, 6.5))
    for gpu in gpus:
        for denoiser in sorted(summary["denoiser"].unique()):
            for dim in (2, 3):
                sub = summary[
                    (summary["gpu"] == gpu)
                    & (summary["denoiser"] == denoiser)
                    & (summary["dim"] == dim)
                ].sort_values("area")
                if sub.empty:
                    continue
                xs = [xpos[s] for s in sub["shape"]]
                ys = sub["speedup"].to_numpy()
                ax.plot(xs, ys, linestyle=styles[gpu], marker=DIM_MARKERS[dim],
                        color=colors[denoiser], markersize=9, linewidth=2)
                for x, y in zip(xs, ys):
                    if np.isfinite(y):
                        ax.annotate(f"{y:.2f}x", (x, y), xytext=offsets[gpu],
                                    textcoords="offset points", ha="center",
                                    fontsize=8, color=colors[denoiser])

    if 0 < n2d < len(labels):
        ax.axvline(n2d - 0.5, color="black", ls="--", lw=1.2, alpha=0.4)
    ax.axhline(1.0, color="gray", ls=":", lw=1.3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    # Legend only gains a GPU section once there is more than one to tell apart.
    multi_gpu = len(gpus) > 1 and gpus != [UNKNOWN_GPU]
    ax.legend(
        handles=_legend_handles(colors, styles if multi_gpu else None),
        title="Denoiser / dim / GPU" if multi_gpu else "Denoiser / dim",
        loc="center left", bbox_to_anchor=(1.01, 0.5) if multi_gpu else None,
    )
    style_axes(ax, _SPEEDUP_TITLE, "Shape  (2D  |  3D)", "Speedup")
    fig.tight_layout()
    return write_figure(fig, output_path, "denoiser_compile_speedup.png")


_ROOFLINE_TITLE = "Roofline: Arithmetic Intensity vs Compile Speedup"


def _ridge_of(gpu: str) -> float | None:
    """Ridge point for a GPU, or None when its peak numbers are not known.

    A GPU absent from :data:`GPU_RIDGE` is plotted without a ridge rather than
    against another device's, which would put the memory/compute boundary in
    the wrong place.
    """
    if gpu == UNKNOWN_GPU:
        return RIDGE_POINT
    return GPU_RIDGE.get(gpu.lower())


def _plot_one_roofline(summary: pd.DataFrame, output_path: Path, gpu: str) -> str:
    """Roofline for a single GPU, written to its own file."""
    colors = _denoiser_colors(summary)
    panel = summary[summary["gpu"] == gpu]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for denoiser in sorted(panel["denoiser"].unique()):
        for dim in (2, 3):
            sub = panel[(panel["denoiser"] == denoiser) & (panel["dim"] == dim)]
            sub = sub[np.isfinite(sub["arith_intensity"]) & np.isfinite(sub["speedup"])]
            if sub.empty:
                continue
            ax.scatter(sub["arith_intensity"], sub["speedup"], s=170,
                       color=colors[denoiser], marker=DIM_MARKERS[dim],
                       edgecolors="black", linewidths=0.8, zorder=10)

    ridge = _ridge_of(gpu)
    ridge_handle = []
    if ridge is not None:
        ax.axvline(ridge, color="gray", ls="--", lw=1.5)
        ridge_handle = [Line2D([0], [0], color="gray", ls="--", lw=1.5,
                               label=f"Ridge point ({ridge:.0f} FLOP/byte)")]
    ax.axhline(1.0, color="black", ls=":", lw=1, alpha=0.4)
    ax.set_xscale("log")
    ax.legend(handles=_legend_handles(colors) + ridge_handle, loc="best", fontsize=10)
    style_axes(
        ax,
        f"{_ROOFLINE_TITLE} — {gpu.upper()}" if gpu else _ROOFLINE_TITLE,
        "Arithmetic Intensity (FLOP / byte)",
        "Speedup (eager / compiled)",
    )
    fig.tight_layout()
    # Legacy results carry no GPU name, so they keep the original filename.
    suffix = f"_{gpu.lower()}" if gpu else ""
    return write_figure(fig, output_path, f"denoiser_roofline{suffix}.png")


def _plot_denoiser_roofline(summary: pd.DataFrame, output_path: Path) -> list[str]:
    """One standalone roofline figure per GPU, since each has its own ridge."""
    return [_plot_one_roofline(summary, output_path, gpu) for gpu in _gpus_of(summary)]


def create_denoiser_compile_visualizations(
    results: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    roofline: bool = False,
) -> Path:
    """Create denoiser eager-vs-compiled speedup (and optional roofline) figures."""
    configure_matplotlib()
    df, results_path = _load_denoiser_df(results)
    if "objective_denoise_time_sec" not in df.columns:
        raise ValueError("Results are missing 'objective_denoise_time_sec'.")

    summary = _denoiser_summary(df)
    if summary.empty:
        raise ValueError("No timed denoiser iterations found for speedup comparison.")

    output_path = Path(output_dir) / f"denoiser_compile_{results_path.stem}"
    output_path.mkdir(parents=True, exist_ok=True)
    _plot_denoiser_speedup(summary, output_path)
    if roofline:
        _plot_denoiser_roofline(summary, output_path)
    return output_path
