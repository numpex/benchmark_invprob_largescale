"""Communication breakdown for distributed PnP inference.

For the ``comm_inference_2D`` / ``comm_inference_3D`` experiments: a distributed
PnP run profiled with ``profiler_mode: torch``, which records per-section CUDA and
communication time.

The decomposition
-----------------
The torch profiler attributes communication to the section that contains it, by
walking each section's event subtree, so ``{sec}_comm_sec`` is a *subset* of
``{sec}_cuda_sec`` rather than something to add on top::

    iteration        = denoise_cuda + gradient_cuda
    denoiser compute = denoise_cuda  - denoise_comm
    denoiser comm    = denoise_comm
    gradient compute = gradient_cuda - gradient_comm
    gradient comm    = gradient_comm

The residual overhead (total_time - denoise - gradient) is excluded, as is
``comm_sync_sec`` -- NCCL kernels outside any section, which mostly spin-wait on
inter-rank skew and are reported by the profiler separately.

Figures
-------
``waste_attribution.png``
    Stacked GPU-seconds per iteration (n_gpus * part), normalized by the
    baseline's ``b * T(b)``. Bar height is exactly 1/E, so 1.0 is perfect scaling
    and growth above it is waste, attributed to the part that grew.

``{denoiser,gradient}_time.png``
    That section's compute and comm in absolute seconds against GPU count, with
    an ideal 1/N reference. The waste figure normalizes everything to the
    baseline, which hides how each part moves on its own; these show whether
    compute actually tracks 1/N and how comm behaves, on log axes because the
    series can span two decades.

Timing window
-------------
Rows without timings are dropped, then the first ``skip_warmup`` *timed*
iterations of each configuration are discarded. The absolute ``stop_val`` where
timings start depends on the solver's ``profiler_warmup``, so the skip is
relative rather than a fixed ``stop_val``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolsbench.visualization.common import (
    DEFAULT_OUTPUT_DIR,
    configure_matplotlib,
    load_results,
    make_output_path,
    size_label,
    write_figure,
)

DEFAULT_SKIP_WARMUP = 3

# section -> (cuda column, comm column). Comm is nested inside the cuda total.
SECTIONS = {
    "denoiser": ("objective_denoise_cuda_sec", "objective_denoise_comm_sec"),
    "gradient": ("objective_gradient_cuda_sec", "objective_gradient_comm_sec"),
}
# Dark shade = compute, light shade = that section's communication.
PART_COLORS = {
    "denoiser compute": "#f97316",
    "denoiser comm": "#fdba74",
    "gradient compute": "#2563eb",
    "gradient comm": "#93c5fd",
}
PARTS = list(PART_COLORS)
TOTAL_COLOR = "#111827"
# Section names are too wide to repeat above every bar.
SECTION_ABBREVIATIONS = {"denoiser": "den", "gradient": "grad"}

COMM_REQUIRED_COLUMNS = {col for pair in SECTIONS.values() for col in pair}


def create_comm_inference_visualizations(
    results: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    skip_warmup: int = DEFAULT_SKIP_WARMUP,
) -> Path:
    """Create the communication-breakdown visualizations for a PnP inference run."""
    configure_matplotlib()
    df, results_path = load_results(results)
    missing = sorted(COMM_REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            f"Results are missing {', '.join(missing)}. This experiment needs a "
            "PnP run recorded with profiler_mode=torch."
        )

    rows = _add_metrics(_summarize(df, skip_warmup))
    image_size = size_label(df["p_dataset_image_size"].iloc[0])

    output_path = make_output_path(output_dir, "comm_inference", results_path)
    _plot_waste_attribution(rows, image_size, output_path, skip_warmup)
    for section in SECTIONS:
        _plot_section_time(rows, image_size, output_path, skip_warmup, section)
    return output_path


def _summarize(df: pd.DataFrame, skip_warmup: int) -> pd.DataFrame:
    """Average the per-section timings per configuration over steady-state iterations."""
    df = df.copy()
    # A single GPU can run non-distributed (pure compute) or distributed (tiled);
    # those must not be averaged together, so distribution is part of the key.
    if {"p_solver_distribute_physics", "p_solver_distribute_denoiser"} <= set(
        df.columns
    ):
        df["distributed"] = df["p_solver_distribute_physics"].astype(bool) | df[
            "p_solver_distribute_denoiser"
        ].astype(bool)
    else:
        df["distributed"] = df["n_gpus"] > 1

    timing_columns = [col for pair in SECTIONS.values() for col in pair]
    df = df[df[timing_columns].notna().all(axis=1)]
    if df.empty:
        raise ValueError("No rows carry per-section timings.")

    keys = ["n_gpus", "distributed"]
    timed_index = df.groupby(keys, dropna=False)["stop_val"].rank(method="first")
    df = df[timed_index > skip_warmup]
    if df.empty:
        raise ValueError(
            f"Every configuration has <= {skip_warmup} timed iterations; nothing "
            "left after skipping warmup."
        )

    aggregations = {col: (col, "mean") for col in timing_columns}
    aggregations["n_iters"] = ("stop_val", "count")
    return (
        df.groupby(keys, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(keys)
    )


def _add_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """Split each section into compute/comm and add efficiency and cost columns."""
    rows = summary.copy()
    for section, (cuda_col, comm_col) in SECTIONS.items():
        # comm is nested inside the section's cuda total, so compute is the remainder.
        rows[f"{section} comm"] = rows[comm_col]
        rows[f"{section} compute"] = rows[cuda_col] - rows[comm_col]
    rows["total"] = rows[PARTS].sum(axis=1)
    rows["config"] = [
        (
            f"{int(n)} GPU\nbaseline"
            if i == 0
            else f"{int(n)} GPU\n{'tiled' if int(n) == 1 else 'dist'}"
        )
        for i, n in enumerate(rows["n_gpus"])
    ]

    baseline = rows.iloc[0]
    ideal_gpu_seconds = float(baseline["n_gpus"]) * float(baseline["total"])
    rows["efficiency"] = ideal_gpu_seconds / (rows["n_gpus"] * rows["total"]) * 100
    rows["inv_efficiency"] = rows["n_gpus"] * rows["total"] / ideal_gpu_seconds
    for part in PARTS:
        rows[f"{part} cost"] = rows["n_gpus"] * rows[part] / ideal_gpu_seconds
    # Share of each part within its own section: how much of the denoiser (or the
    # gradient) it is, independent of how big that section is. compute + comm =
    # 100% for each section.
    for section, (cuda_col, comm_col) in SECTIONS.items():
        comm_pct = rows[comm_col] / rows[cuda_col] * 100
        rows[f"{section} comm pct in section"] = comm_pct
        rows[f"{section} compute pct in section"] = 100 - comm_pct
    return rows


def _plot_waste_attribution(
    rows: pd.DataFrame,
    image_size: str,
    output_path: Path,
    skip_warmup: int,
) -> str:
    """Plot stacked GPU-second cost per part, in units of one ideal iteration."""
    baseline_gpus = int(rows["n_gpus"].iloc[0])
    x = np.arange(len(rows))

    # Widen with the number of bars: the per-bar labels need a fixed horizontal
    # slot, so a fixed width starts overlapping once there are many GPU counts.
    fig, ax = plt.subplots(figsize=(max(10.5, 2.1 * len(rows)), 6.4))
    bottom = np.zeros(len(rows))
    for part in PARTS:
        values = rows[f"{part} cost"].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=PART_COLORS[part],
            alpha=0.95,
            width=0.6,
            label=part,
        )
        pcts = rows[f"{part} pct in section"].to_numpy(dtype=float)
        for xi, (value, base, pct) in enumerate(zip(values, bottom, pcts, strict=True)):
            if value >= 0.1:
                ax.text(
                    xi,
                    base + value / 2,
                    f"{pct:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
        bottom += values

    # How much of each section is communication. The comm bands are far too thin
    # to label in place, so the shares go above the bar.
    for xi in range(len(rows)):
        row = rows.iloc[xi]
        offset = 6
        for section in reversed(SECTIONS):
            ax.annotate(
                f"{SECTION_ABBREVIATIONS[section]} "
                f"{row[f'{section} comm pct in section']:.1f}%",
                (xi, bottom[xi]),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=PART_COLORS[f"{section} comm"],
            )
            offset += 12
        ax.annotate(
            f"{bottom[xi]:.2f}x  E={row['efficiency']:.0f}%",
            (xi, bottom[xi]),
            xytext=(0, offset + 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=TOTAL_COLOR,
        )

    # Labelled through the legend: every bar top is near 1.0 and carries labels,
    # so an in-axes annotation has nowhere to sit without colliding.
    ax.axhline(
        1.0,
        color=TOTAL_COLOR,
        linestyle="--",
        linewidth=1.3,
        alpha=0.55,
        label="ideal (1.0x)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(list(rows["config"]))
    ax.set_ylim(0, float(bottom.max()) * 1.18)
    ax.set_title(
        f"{image_size} PnP - where the GPU-seconds go ({baseline_gpus}-GPU baseline)",
        loc="left",
        pad=14,
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Configuration")
    ax.set_ylabel("GPU-seconds per iteration (1.0 = baseline cost)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e5e7eb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", ncols=5, frameon=False, fontsize=9)
    fig.text(
        0.99,
        -0.01,
        "Iteration is denoiser + gradient CUDA time; comm is the part of each "
        "section spent in NCCL. Above each bar: share of the den(oiser) and "
        f"grad(ient) spent in comm. Skips {skip_warmup} warmup iterations.",
        ha="right",
        va="bottom",
        color="#6b7280",
        fontsize=9,
    )
    fig.tight_layout()
    return write_figure(fig, output_path, "waste_attribution.png", dpi=200)


def _plot_section_time(
    rows: pd.DataFrame,
    image_size: str,
    output_path: Path,
    skip_warmup: int,
    section: str,
) -> str:
    """Plot one section's compute and comm time in seconds against GPU count."""
    gpus = rows["n_gpus"].to_numpy(dtype=float)
    compute = rows[f"{section} compute"].to_numpy(dtype=float)
    comm = rows[f"{section} comm"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    # Ideal 1/N scaling anchored at the baseline's compute time.
    ax.plot(
        gpus,
        compute[0] * gpus[0] / gpus,
        linestyle="--",
        linewidth=1.4,
        color=TOTAL_COLOR,
        alpha=0.5,
        label="ideal 1/N scaling",
    )
    # Compute labelled above its points, comm below: the two series can nearly
    # coincide (a gradient whose time is almost all communication), and a fixed
    # offset would stack the labels on top of each other.
    for values, part, label_dy in (
        (compute, f"{section} compute", 10),
        (comm, f"{section} comm", -16),
    ):
        ax.plot(
            gpus,
            values,
            marker="o",
            markersize=7,
            linewidth=2.6,
            color=PART_COLORS[part],
            label=part,
        )
        for gpu, value in zip(gpus, values, strict=True):
            ax.annotate(
                f"{value:.2f}" if value >= 1 else f"{value:.3f}",
                (gpu, value),
                xytext=(0, label_dy),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=PART_COLORS[part],
                fontweight="bold",
            )

    # Both series together can span two decades, so a linear axis would flatten
    # whichever is smaller.
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(gpus)
    ax.set_xticklabels([str(int(gpu)) for gpu in gpus])
    ax.set_title(
        f"{image_size} PnP - {section} time vs GPU count",
        loc="left",
        pad=14,
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Time per iteration (s, log scale)")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#e5e7eb")
    ax.grid(True, which="minor", color="#f3f4f6", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False)
    fig.text(
        0.99,
        -0.01,
        f"Compute is the {section} CUDA time outside NCCL; comm is the NCCL time "
        f"inside the {section} section. Skips {skip_warmup} warmup iterations.",
        ha="right",
        va="bottom",
        color="#6b7280",
        fontsize=9,
    )
    fig.tight_layout()
    return write_figure(fig, output_path, f"{section}_time.png", dpi=200)
