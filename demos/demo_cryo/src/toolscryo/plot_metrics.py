"""Plot training and validation metrics from a cryo run's CSV files.

Produces one figure with three panels:
  - Train loss per epoch
  - Train PSNR per epoch
  - Val PSNR per epoch

Usage (standalone):
    python src/toolscryo/plot_metrics.py --run-dir runs/<name>/
    python src/toolscryo/plot_metrics.py --run-dir runs/<name>/ --save summary.png
"""

from __future__ import annotations

import argparse
from pathlib import Path


def plot_metrics(run_dir: Path, save: Path | str | None = None) -> None:
    """Load CSVs from *run_dir*/metrics/ and save (or show) a summary figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[plot_metrics] skipped — missing dependency: {exc}")
        return

    metrics_dir = Path(run_dir) / "metrics"
    if not metrics_dir.exists():
        print(f"[plot_metrics] no metrics/ dir found in {run_dir}, skipping.")
        return

    train_csv = metrics_dir / "train_epochs.csv"
    val_csv = metrics_dir / "val_epochs.csv"

    train_df = pd.read_csv(train_csv) if train_csv.exists() else pd.DataFrame()
    val_df = pd.read_csv(val_csv) if val_csv.exists() else pd.DataFrame()

    # Detect individual loss columns (exclude meta / total / PSNR columns)
    _skip = {"epoch", "lr", "step", "gradient_norm", "TotalLoss"}
    individual_loss_cols = [
        c for c in train_df.columns
        if c not in _skip and "psnr" not in c.lower() and not train_df.empty
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

    # ── Train + Val PSNR ────────────────────────────────────────────────────
    ax = axes[1]
    psnr_col = next((c for c in train_df.columns if "psnr" in c.lower()), None)
    if not train_df.empty and psnr_col:
        ax.plot(train_df["epoch"], train_df[psnr_col], "o-",
                color="darkorange", label="Train")
    psnr_col_val = next((c for c in val_df.columns if "psnr" in c.lower()), None)
    if not val_df.empty and psnr_col_val:
        ax.plot(val_df["epoch"], val_df[psnr_col_val], "s--",
                color="seagreen", label="Val")
    ax.set_title("PSNR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
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
