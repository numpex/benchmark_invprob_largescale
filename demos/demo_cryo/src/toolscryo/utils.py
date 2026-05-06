from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import deepinv as dinv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from deepinv.loss.metric import PSNR


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_config_json(path: Path, cfg_dict: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)


def append_metrics_row(path: Path | str, row: dict) -> None:
    """Append one row to a CSV file, writing a header on first write."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class PerfProbe:
    """Context manager that measures wall time and peak GPU memory for a code block.

    """
    def __enter__(self) -> "PerfProbe":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_s: float = time.perf_counter() - self._t0
        self.peak_mb: float = (
            torch.cuda.max_memory_allocated() / 1e6
            if torch.cuda.is_available() else 0.0
        )


class ExponentialLRWithFloor(torch.optim.lr_scheduler.ExponentialLR):
    """ExponentialLR that clamps the LR at ``lr_min``."""
    def __init__(self, optimizer, gamma, lr_min: float = 0.0, **kwargs):
        self.lr_min = lr_min
        super().__init__(optimizer, gamma, **kwargs)
    def get_lr(self):  # type: ignore[override]
        return [max(self.lr_min, lr) for lr in super().get_lr()]


class CsvTrainer(dinv.Trainer):
    """dinv.Trainer subclass that additionally logs every epoch to CSV files
    and saves mid-slice PNG reconstructions at each eval step.

    Creates two files (rank-0 only):
      - ``metrics/train_epochs.csv`` — train loss + gradient norm per epoch
      - ``metrics/val_epochs.csv``   — validation PSNR (and other eval metrics) per epoch

    Set ``trainer._metrics_dir``, ``trainer._images_dir``, and ``trainer._ckpt_dir``
    (Paths) after construction.  Set ``trainer._grad_accum_steps`` (int, default 1)
    to enable gradient accumulation over N batches before each optimizer step.
    """

    def compute_loss(self, physics, x, y, train=True, epoch=None, step=False):  # type: ignore[override]
        """Adds gradient accumulation: accumulates over ``_grad_accum_steps`` batches."""
        # ── Timing: reset accumulators on first batch of each epoch ──────────
        if train:
            if epoch != getattr(self, "_current_train_epoch", None):
                self._current_train_epoch = epoch
                self._train_epoch_start = time.perf_counter()
                self._train_batch_count = 0
                self._epoch_fwd_model_time = 0.0
                self._epoch_fwd_loss_time  = 0.0
                self._epoch_bwd_time       = 0.0
            self._train_batch_count = getattr(self, "_train_batch_count", 0) + 1

        accum = max(1, getattr(self, "_grad_accum_steps", 1))
        if accum == 1:
            result = super().compute_loss(
                physics, x, y, train=train, epoch=epoch, step=step
            )
            return result

        # Track how many batches have been seen in the current window.
        self._accum_count = getattr(self, "_accum_count", 0)
        at_window_start = self._accum_count % accum == 0
        self._accum_count += 1
        at_window_end = self._accum_count % accum == 0

        # Parent handles zero_grad / optimizer.step when step=True.
        # We replicate only what changes: loss scaling and gating of those calls.
        logs = {}
        if train and step and at_window_start:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.enable_grad() if train else torch.no_grad():
            with PerfProbe() as p_model:
                x_net = self.model_inference(y=y, physics=physics, x=x, train=train)
            loss_total = 0
            if train or self.compute_eval_losses:
                with PerfProbe() as p_loss:
                    for k, loss_fn in enumerate(self.losses):
                        loss = loss_fn(
                            x=x,
                            x_net=x_net,
                            y=y,
                            physics=physics,
                            model=self.model,
                            epoch=epoch,
                        )
                        loss_total += loss.mean()
                        meters = (
                            self.logs_losses_train[k] if train else self.logs_losses_eval[k]
                        )
                        meters.update(loss.detach().cpu().numpy())
                        if len(self.losses) > 1:
                            logs[loss_fn.__class__.__name__] = meters.avg
                meters = (
                    self.logs_total_loss_train if train else self.logs_total_loss_eval
                )
                meters.update(loss_total.item())
                logs["TotalLoss"] = meters.avg

        if train:
            with PerfProbe() as p_bwd:
                (loss_total / accum).backward()
            norm = self.check_clip_grad()
            if norm is not None:
                logs["gradient_norm"] = self.check_grad_val.avg
            if step and at_window_end:
                self.optimizer.step()

            self._epoch_fwd_model_time += p_model.elapsed_s
            self._epoch_fwd_loss_time  += p_loss.elapsed_s if (train or self.compute_eval_losses) else 0.0
            self._epoch_bwd_time       += p_bwd.elapsed_s

            if self.verbose:
                batch = self._train_batch_count
                print(
                    f"[step] ep={epoch} batch={batch}  "
                    f"model={p_model.elapsed_s:.2f}s/{p_model.peak_mb:.0f}MB  "
                    f"loss={p_loss.elapsed_s:.2f}s/{p_loss.peak_mb:.0f}MB  "
                    f"bwd={p_bwd.elapsed_s:.2f}s/{p_bwd.peak_mb:.0f}MB",
                    flush=True,
                )

        return loss_total, x_net, logs

    def compute_metrics(self, x, x_net, y, physics, logs, train=True, epoch=None):  # type: ignore[override]
        if not train:
            self._val_batch_count = getattr(self, "_val_batch_count", 0) + 1
            if getattr(self, "_val_start", None) is None:
                self._val_start = time.perf_counter()
        x_net, logs = super().compute_metrics(
            x, x_net, y, physics, logs, train=train, epoch=epoch
        )
        if not train and x_net is not None:
            self._save_val_image(epoch, x, y, x_net)
        return x_net, logs

    def log_metrics_mlops(self, logs: dict, step: int, train: bool = True) -> None:  # type: ignore[override]
        # Only rank-0 writes CSV
        if not self.verbose:
            return

        metrics_dir = getattr(self, "_metrics_dir", None)
        if metrics_dir is None:
            return

        row = {
            "epoch": step,
            "lr": self.optimizer.param_groups[0]["lr"],
            **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
        }
        fname = "train_epochs.csv" if train else "val_epochs.csv"
        append_metrics_row(Path(metrics_dir) / fname, row)

        if train:
            scheduler = getattr(self, "_scheduler", None)
            if scheduler is not None:
                scheduler.step()
            n = max(1, getattr(self, "_train_batch_count", 1))
            t_total = time.perf_counter() - getattr(self, "_train_epoch_start", time.perf_counter())
            t_model = getattr(self, "_epoch_fwd_model_time", 0.0)
            t_loss  = getattr(self, "_epoch_fwd_loss_time",  0.0)
            t_bwd   = getattr(self, "_epoch_bwd_time", 0.0)
            t_fwd   = t_model + t_loss
            print(
                f"[time] train epoch={step}  total={t_total:.1f}s  "
                f"fwd={t_fwd:.1f}s (model={t_model:.1f}s  loss/physics={t_loss:.1f}s)  "
                f"bwd={t_bwd:.1f}s  other={t_total-t_fwd-t_bwd:.1f}s  "
                f"per_img={t_total/n:.2f}s  n={n}",
                flush=True,
            )
            self._val_start = time.perf_counter()
            self._val_batch_count = 0

            alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
            total_gb = (
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).total_memory
                / 1024**3
            )
            print(
                f"[gpu] step={step}  max_alloc={alloc_gb:.2f} GB / {total_gb:.1f} GB",
                flush=True,
            )
            torch.cuda.reset_peak_memory_stats()
        else:
            n_val = max(1, getattr(self, "_val_batch_count", 1))
            t_val = time.perf_counter() - getattr(self, "_val_start", time.perf_counter())
            print(f"[time] val   epoch={step}  total={t_val:.1f}s  per_img={t_val/n_val:.2f}s  n={n_val}", flush=True)
            self._val_start = None

        # ── Manual checkpoint every ckp_interval epochs (rank 0 only) ──────
        # log_metrics_mlops with train=False fires once per eval pass = once per epoch.
        ckpt_dir = getattr(self, "_ckpt_dir", None)
        if not train and ckpt_dir is not None:
            ckp_interval = getattr(self, "ckp_interval", 10)
            ep = step
            if ep % ckp_interval == 0:
                ckpt_path = Path(ckpt_dir) / f"ckp_{ep:04d}.pth"
                torch.save(
                    {
                        "epoch": ep,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"[ckpt] saved {ckpt_path}", flush=True)

            # ── Best checkpoint (track highest val PSNR) ─────────────────
            val_psnr = logs.get("PSNR", None)
            if val_psnr is not None:
                best = getattr(self, "_best_val_psnr", float("-inf"))
                if val_psnr > best:
                    self._best_val_psnr = val_psnr
                    best_path = Path(ckpt_dir) / "ckp_best.pth"
                    torch.save(
                        {
                            "epoch": ep,
                            "val_psnr": val_psnr,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                        },
                        best_path,
                    )
                    print(f"[ckpt] new best PSNR={val_psnr:.4f} dB → saved {best_path}", flush=True)

    def _save_val_image(self, epoch, x, y, x_net) -> None:
        """Save mid-slice PNG for one validation image (rank 0 only)."""
        if not self.verbose:
            return
        images_dir = getattr(self, "_images_dir", None)
        if images_dir is None:
            return

        if epoch != getattr(self, "_plot_epoch", None):
            self._plot_epoch = epoch
            self._plot_img_idx = 0
        img_idx = self._plot_img_idx
        self._plot_img_idx += 1

        if img_idx == 0 and epoch == 0:
            print(
                f"[data] x shape={tuple(x.shape)}  y shape={tuple(y.shape)}  x_net shape={tuple(x_net.shape)}",
                flush=True,
            )

        def _slices(t: torch.Tensor):
            v = t[0, 0].detach().cpu().float()  # (D, H, W)
            D, H, W = v.shape
            return [v[D // 2, :, :].numpy(), v[:, H // 2, :].numpy(), v[:, :, W // 2].numpy()]

        # PSNR on full 3-D volume
        _gt_t   = x[0:1].detach().cpu().float()
        _pred_t = x_net[0:1].detach().cpu().float()
        _meas_t = y[0:1].detach().cpu().float()
        psnr_val  = float(PSNR(max_pixel=None)(_pred_t, _gt_t))
        psnr_meas = float(PSNR(max_pixel=None)(_meas_t, _gt_t))

        axis_labels = ["axial (D)", "coronal (H)", "sagittal (W)"]
        col_titles  = ["GT (icecream)", f"Input ({psnr_meas:.2f} dB)", f"Pred ({psnr_val:.2f} dB)"]
        vols        = [x, y, x_net]

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for row, (label, *row_slices) in enumerate(
            zip(axis_labels, *[_slices(v) for v in vols])
        ):
            vmin = float(row_slices[0].min())
            vmax = float(row_slices[0].max())
            for col, (ax, img) in enumerate(zip(axes[row], row_slices)):
                ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
                ax.axis("off")
                if row == 0:
                    ax.set_title(col_titles[col])
                if col == 0:
                    ax.set_ylabel(label)

        fig.suptitle(f"Eval epoch {epoch + 1}  —  vol {img_idx:02d}  —  mid-slices per axis")
        fig.tight_layout()
        out = Path(images_dir) / f"epoch{epoch + 1:04d}" / f"vol{img_idx:02d}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)

    def plot(self, epoch, physics, x, y, x_net, train=True):  # type: ignore[override]
        """Suppress the default plot — images are saved in compute_metrics."""
        return
