"""Visualizations comparing torch.compile modes for inference.

Two independent entry points, one per experiment config:

- :func:`create_compile_speedup_visualizations` — PnP ``tomography_2d_compile``:
  1st-iteration vs steady-state cost across compile modes (None/pre/post/fused).
- :func:`create_denoiser_compile_visualizations` — ``denoiser_compile``:
  steady-state eager-vs-compiled speedup per architecture and shape (2D/3D),
  plus an optional roofline scatter (arithmetic intensity vs speedup).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from toolsbench.visualization.common import (
    COLORWAY,
    DEFAULT_OUTPUT_DIR,
    configure_matplotlib,
    load_results,
    style_axes,
    write_figure,
)

STABLE_WINDOW = 5
COMPILE_ORDER = ["None", "pre", "post", "fused"]

# Marker per spatial dimensionality, shared by the denoiser speedup/roofline plots.
DIM_MARKERS = {2: "o", 3: "s"}

# Roofline hardware reference: H100 SXM, TF32 tensor cores (matches the demo).
# RIDGE_POINT is the FLOP/byte where the device switches from memory- to
# compute-bound; points left of it are memory-bound, right are compute-bound.
H100_PEAK_TFLOPS = 494.0
H100_PEAK_BW_TBs = 3.35
RIDGE_POINT = (H100_PEAK_TFLOPS * 1e12) / (H100_PEAK_BW_TBs * 1e12)


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

_PKL_PREFIX = b"\x00benchopt-pkl\x00"


def _decode_param(value):
    """Decode a benchopt parameter that may be pickled (e.g. a list image_size)."""
    if isinstance(value, (bytes, bytearray)) and value.startswith(_PKL_PREFIX):
        return pickle.loads(bytes(value)[len(_PKL_PREFIX):])
    return value


def _normalize_size(size) -> tuple[int, ...]:
    """Spatial-size tuple from a decoded image_size (int/[s]/[h,w]/[d,h,w])."""
    if isinstance(size, (list, tuple)):
        if len(size) == 1:
            return (int(size[0]), int(size[0]))
        return tuple(int(s) for s in size)
    return (int(size), int(size))


def _shape_label(size) -> str:
    dims = _normalize_size(size)
    if len(dims) == 3:
        d, h, w = dims
        return f"{d}³" if d == h == w else f"{d}×{h}×{w}"
    return str(dims[0]) if dims[0] == dims[1] else f"{dims[0]}×{dims[1]}"


def _dim_of(size) -> int:
    return 3 if len(_normalize_size(size)) == 3 else 2


def _area(size) -> int:
    dims = _normalize_size(size)
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


def _denoiser_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (arch, shape, dim) with eager/compiled steady time and speedup."""
    df = df.copy()
    df["compile_mode"] = df["p_solver_compile"].fillna("None").astype(str)
    df["arch"] = df["p_solver_denoiser"].astype(str)
    size = df["p_dataset_image_size"].map(_decode_param)
    df["shape"] = size.map(_shape_label)
    df["dim"] = size.map(_dim_of)
    df["area"] = size.map(_area)

    has_ai = "objective_arith_intensity" in df.columns

    rows = []
    for (arch, shape, dim, area, mode, _rep), group in df.groupby(
        ["arch", "shape", "dim", "area", "compile_mode", "idx_rep"], dropna=False
    ):
        timed = group.dropna(subset=["objective_denoise_time_sec"]).sort_values("stop_val")
        if timed.empty:
            continue
        rows.append(
            {
                "arch": arch,
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

    keys = ["arch", "shape", "dim", "area"]
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
    return summary.sort_values(["dim", "area", "arch"]).reset_index(drop=True)


def _arch_colors(summary: pd.DataFrame) -> dict[str, str]:
    return {a: COLORWAY[i % len(COLORWAY)] for i, a in enumerate(sorted(summary["arch"].unique()))}


def _legend_handles(colors: dict[str, str]):
    arch_handles = [
        Line2D([0], [0], color=c, marker="o", markersize=9, label=a)
        for a, c in colors.items()
    ]
    dim_handles = [
        Line2D([0], [0], color="gray", marker=m, linestyle="None", markersize=9, label=f"{d}D")
        for d, m in DIM_MARKERS.items()
    ]
    return arch_handles + dim_handles


def _plot_denoiser_speedup(summary: pd.DataFrame, output_path: Path) -> str:
    colors = _arch_colors(summary)
    shapes = (
        summary[["shape", "dim", "area"]]
        .drop_duplicates()
        .sort_values(["dim", "area"])
    )
    labels = shapes["shape"].tolist()
    xpos = {lab: i for i, lab in enumerate(labels)}
    n2d = int((shapes["dim"] == 2).sum())

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for arch in sorted(summary["arch"].unique()):
        for dim in (2, 3):
            sub = summary[(summary["arch"] == arch) & (summary["dim"] == dim)].sort_values("area")
            if sub.empty:
                continue
            xs = [xpos[s] for s in sub["shape"]]
            ys = sub["speedup"].to_numpy()
            ax.plot(xs, ys, "-", marker=DIM_MARKERS[dim], color=colors[arch],
                    markersize=9, linewidth=2)
            for x, y in zip(xs, ys):
                if np.isfinite(y):
                    ax.annotate(f"{y:.2f}x", (x, y), xytext=(0, 8),
                                textcoords="offset points", ha="center",
                                fontsize=9, color=colors[arch])

    if 0 < n2d < len(labels):
        ax.axvline(n2d - 0.5, color="black", ls="--", lw=1.2, alpha=0.4)
    ax.axhline(1.0, color="gray", ls=":", lw=1.3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.legend(handles=_legend_handles(colors), title="Architecture / dim")
    style_axes(
        ax,
        "torch.compile Steady-State Speedup (eager / compiled)",
        "Shape  (2D  |  3D)",
        "Speedup",
    )
    fig.tight_layout()
    return write_figure(fig, output_path, "denoiser_compile_speedup.png")


def _plot_denoiser_roofline(summary: pd.DataFrame, output_path: Path, ridge: float = RIDGE_POINT) -> str:
    colors = _arch_colors(summary)
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for arch in sorted(summary["arch"].unique()):
        for dim in (2, 3):
            sub = summary[(summary["arch"] == arch) & (summary["dim"] == dim)]
            sub = sub[np.isfinite(sub["arith_intensity"]) & np.isfinite(sub["speedup"])]
            if sub.empty:
                continue
            ax.scatter(sub["arith_intensity"], sub["speedup"], s=170, color=colors[arch],
                       marker=DIM_MARKERS[dim], edgecolors="black", linewidths=0.8, zorder=10)

    ax.axvline(ridge, color="gray", ls="--", lw=1.5)
    ax.axhline(1.0, color="black", ls=":", lw=1, alpha=0.4)
    ax.set_xscale("log")
    ridge_handle = [Line2D([0], [0], color="gray", ls="--", lw=1.5,
                           label=f"Ridge point ({ridge:.0f} FLOP/byte)")]
    ax.legend(handles=_legend_handles(colors) + ridge_handle, loc="best", fontsize=10)
    style_axes(
        ax,
        "Roofline: Arithmetic Intensity vs Compile Speedup",
        "Arithmetic Intensity (FLOP / byte)",
        "Speedup (eager / compiled)",
    )
    fig.tight_layout()
    return write_figure(fig, output_path, "denoiser_roofline.png")


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
