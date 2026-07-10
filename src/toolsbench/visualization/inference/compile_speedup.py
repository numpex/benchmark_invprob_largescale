"""Visualizations comparing torch.compile modes for PnP inference.

Compares 1st-iteration cost (includes JIT/compile overhead) against
steady-state ("stable") per-iteration cost, with speedup relative to the
1-GPU, ``compile=None`` baseline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        label=f"stable",
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
