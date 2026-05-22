"""Plot training and validation metrics from a cryo run's CSV files.

Produces one figure with two panels:
  - Train loss per epoch (log scale)
  - Val metric per epoch: FSC frequency (equivariant runs) or PSNR (supervised runs)

Usage (standalone):
    python src/toolscryo/plot_metrics.py --run-dir runs/<name>/
    python src/toolscryo/plot_metrics.py --run-dir runs/<name>/ --save summary.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(run_dir: Path, save: Path | str | None = None) -> None:
    """Load CSVs from *run_dir*/metrics/ and save (or show) a summary figure."""
    metrics_dir = Path(run_dir) / "metrics"
    if not metrics_dir.exists():
        print(f"[plot_metrics] no metrics/ dir found in {run_dir}, skipping.")
        return

    train_csv = metrics_dir / "train_epochs.csv"
    val_csv = metrics_dir / "val_epochs.csv"

    train_df = pd.read_csv(train_csv) if train_csv.exists() else pd.DataFrame()
    val_df = pd.read_csv(val_csv) if val_csv.exists() else pd.DataFrame()

    # Detect individual loss columns (exclude meta / total / PSNR / FSC columns)
    _skip = {"epoch", "lr", "step", "gradient_norm", "TotalLoss"}
    individual_loss_cols = [
        c for c in train_df.columns
        if c not in _skip
        and "psnr" not in c.lower()
        and "fsc" not in c.lower()
        and not train_df.empty
    ]
    # Layout: [losses (total + components) | PSNR]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training summary", fontsize=13)

    # ── Train losses (total + components) in log scale ──────────────────────
    ax = axes[0]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx = 0
    if not train_df.empty and "TotalLoss" in train_df.columns:
        ax.plot(train_df["epoch"], train_df["TotalLoss"], "o-",
                color=colors[color_idx], linewidth=2, label="TotalLoss")
        color_idx += 1
    for col in individual_loss_cols:
        ax.plot(train_df["epoch"], train_df[col], "s--",
                color=colors[color_idx % len(colors)], label=col)
        color_idx += 1
    ax.set_yscale("log")
    ax.set_title("Train loss (log scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # ── Val metric: FSC (equivariant) or PSNR (supervised) ─────────────────
    ax = axes[1]
    fsc_col_val  = next((c for c in val_df.columns  if "fsc"  in c.lower()), None)
    psnr_col_val = next((c for c in val_df.columns  if "psnr" in c.lower()), None)
    psnr_col_trn = next((c for c in train_df.columns if "psnr" in c.lower()), None)

    if not val_df.empty and fsc_col_val:
        # ── Equivariant: FSC resolution in Å ──
        epochs = val_df["epoch"]
        ax.plot(epochs, val_df[fsc_col_val], "s-",
                color="steelblue", linewidth=2, label="mean")
        q1_col = next((c for c in val_df.columns if "q1" in c.lower()), None)
        q3_col = next((c for c in val_df.columns if "q3" in c.lower()), None)
        if q1_col and q3_col:
            ax.plot(epochs, val_df[q1_col], "--", color="steelblue",
                    linewidth=1, alpha=0.7, label="Q1")
            ax.plot(epochs, val_df[q3_col], ":",  color="steelblue",
                    linewidth=1, alpha=0.7, label="Q3")
            ax.fill_between(epochs, val_df[q1_col], val_df[q3_col],
                            color="steelblue", alpha=0.15, label="Q1–Q3")
        ax.set_title("Val FSC resolution @ 0.143 threshold (↓ better)")
        ax.set_ylabel("Resolution (Å)")
    elif (not val_df.empty and psnr_col_val) or (not train_df.empty and psnr_col_trn):
        # ── Supervised: PSNR ──
        if not train_df.empty and psnr_col_trn:
            ax.plot(train_df["epoch"], train_df[psnr_col_trn], "o-",
                    color="darkorange", label="Train")
        if not val_df.empty and psnr_col_val:
            ax.plot(val_df["epoch"], val_df[psnr_col_val], "s--",
                    color="seagreen", label="Val")
        ax.set_title("PSNR")
        ax.set_ylabel("PSNR (dB)")
    else:
        ax.text(0.5, 0.5, "No val metric data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("Val metric")
        ax.set_ylabel("")

    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot_metrics] saved to {out}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cryo training/val metrics.")
    parser.add_argument(
        "--run-dir", required=True, help="Run directory (containing metrics/)"
    )
    parser.add_argument(
        "--save", default=None, help="Save figure to this path (e.g. summary.png)"
    )
    args = parser.parse_args()
    plot_metrics(Path(args.run_dir), save=args.save)


if __name__ == "__main__":
    main()
