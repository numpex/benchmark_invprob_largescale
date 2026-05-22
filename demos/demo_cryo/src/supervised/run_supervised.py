from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import deepinv as dinv
import torch
from deepinv.distributed import DistributedContext, distribute
from deepinv.loss import SupLoss
from deepinv.loss.metric import PSNR

from toolscryo.utils import (
    dump_config_json,
    ensure_dir,
    seed_everything,
)

from .dataloader import CryoDataConfig, build_cryo_dataloaders
from toolscryo.plot_metrics import plot_metrics
from toolscryo.trainer import BaseTrainer, ExponentialLRWithFloor


class SupervisedTrainer(BaseTrainer):
    """BaseTrainer + mid-slice PNG saving during val."""

    _main_metric = "PSNR"
    _main_metric_higher_is_better = True

    def compute_metrics(self, x, x_net, y, physics, logs, train=True, epoch=None):  # type: ignore[override]
        x_net, logs = super().compute_metrics(x, x_net, y, physics, logs, train=train, epoch=epoch)
        if not train and x_net is not None:
            self._save_val_image(epoch, x, y, x_net)
        return x_net, logs

    def _save_val_image(self, epoch, x, y, x_net) -> None:
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
            v = t[0, 0].detach().cpu().float()
            D, H, W = v.shape
            return [v[D // 2, :, :].numpy(), v[:, H // 2, :].numpy(), v[:, :, W // 2].numpy()]

        _gt_t   = x[0:1].detach().cpu().float()
        _pred_t = x_net[0:1].detach().cpu().float()
        _meas_t = y[0:1].detach().cpu().float()
        psnr_val  = float(PSNR(max_pixel=None)(_pred_t, _gt_t))
        psnr_meas = float(PSNR(max_pixel=None)(_meas_t, _gt_t))

        axis_labels = ["axial (D)", "coronal (H)", "sagittal (W)"]
        col_titles  = ["GT", f"Input ({psnr_meas:.2f} dB)", f"Pred ({psnr_val:.2f} dB)"]
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


@dataclass
class RunConfig:
    # ── Data ────────────────────────────────────────────────────────────────
    output_dir: str = "./runs/demo_cryo_supervised"
    input_dir: str = "./dataset/empiar-11058"
    batch_size: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    max_train_vols: int | None = None  # None = use all remaining after val split
    max_val_vols: int = 10  # explicit number of validation volumes
    target_shape: tuple[int, int, int] | None = (
        None  # (D, H, W) resample; None = no resize
    )
    seed: int = 0

    # ── Training ────────────────────────────────────────────────────────────
    num_epochs: int = 100
    learning_rate: float = 1e-4
    # Exponential LR decay: lr(n) = max(scheduler_lr_min, lr * scheduler_gamma^n)
    # Default: 1e-4 → 1e-5 at epoch 10 (gamma = 10^{-1/10} ≈ 0.7943), floor at 5e-6.
    scheduler_gamma: float = 0.7943     # per-epoch multiplicative decay
    scheduler_lr_min: float = 5e-6      # minimum LR floor
    grad_clip: float | None = 1.0
    ckp_interval: int = 10  # checkpoint every N epochs
    eval_interval: int = 1

    # ── Distributed / patching ──────────────────────────────────────────────
    distribute_model: bool = True
    patch_size: tuple[int, int, int] = (64, 64, 64)
    overlap: tuple[int, int, int] = (8, 8, 8)
    max_batch_size: int | None = 2
    checkpoint_batches: str | int | None = "auto"
    grad_accumulation_steps: int = (
        1  # accumulate gradients over N batches before stepping
    )


def run_training(cfg: RunConfig) -> None:
    seed_everything(int(cfg.seed))

    output_dir = ensure_dir(cfg.output_dir)
    dump_config_json(output_dir / "config.json", asdict(cfg))

    data_cfg = CryoDataConfig(
        input_dir=cfg.input_dir,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=int(cfg.prefetch_factor),
        persistent_workers=bool(cfg.persistent_workers),
        max_train_vols=cfg.max_train_vols,
        max_val_vols=int(cfg.max_val_vols),
        target_shape=cfg.target_shape,
        seed=int(cfg.seed),
    )

    with DistributedContext(seed=int(cfg.seed), seed_offset=False, cleanup=True) as ctx:
        rank = int(ctx.rank)

        data_bundle = build_cryo_dataloaders(data_cfg)

        # ── Model ────────────────────────────────────────────────────────────
        backbone = dinv.models.UNet(
            in_channels=1,
            out_channels=1,
            scales=4,
            residual=True,
            batch_norm="biasfree",
            dim=3,
        ).to(ctx.device)

        if cfg.distribute_model:
            backbone = distribute(
                backbone,
                ctx,
                patch_size=tuple(int(v) for v in cfg.patch_size),
                overlap=tuple(int(v) for v in cfg.overlap),
                tiling_dims=(-3, -2, -1),
                max_batch_size=cfg.max_batch_size,
                checkpoint_batches=cfg.checkpoint_batches,
            )

        model = dinv.models.ArtifactRemoval(backbone, mode="direct")

        physics = dinv.physics.Physics().to(ctx.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
        scheduler = ExponentialLRWithFloor(
            optimizer,
            gamma=float(cfg.scheduler_gamma),
            lr_min=float(cfg.scheduler_lr_min),
        )
        accum = max(1, int(cfg.grad_accumulation_steps))

        trainer = SupervisedTrainer(
            model=model,
            physics=physics,
            optimizer=optimizer,
            train_dataloader=data_bundle.train_loader,
            eval_dataloader=data_bundle.val_loader,
            epochs=int(cfg.num_epochs),
            losses=SupLoss(),
            metrics=PSNR(max_pixel=None),
            online_measurements=False,
            device=ctx.device,
            save_path=None,  # intentionally None: Trainer's makedirs runs on ALL ranks
            # and crashes ranks 1+ with exist_ok=False → hang.
            # We handle checkpointing manually in log_metrics_mlops.
            ckp_interval=int(cfg.ckp_interval),
            eval_interval=int(cfg.eval_interval),
            grad_clip=cfg.grad_clip,
            check_grad=cfg.grad_clip is not None,
            plot_images=True,
            verbose=rank == 0,
            show_progress_bar=rank == 0,
            log_train_batch=False,
            optimizer_step_multi_dataset=False,  # lets compute_loss own zero_grad+step (needed for accum)
        )
        trainer._scheduler = scheduler
        trainer._metrics_dir = ensure_dir(output_dir / "metrics")
        trainer._images_dir = ensure_dir(output_dir / "images")
        trainer._ckpt_dir = (
            ensure_dir(output_dir / "checkpoints") if rank == 0 else None
        )
        trainer._grad_accum_steps = accum

        trainer.train()

        # ── Save final checkpoint (rank 0) ───────────────────────────────────
        if rank == 0 and trainer._ckpt_dir is not None:
            ckpt_path = Path(trainer._ckpt_dir) / "ckp_final.pth"
            torch.save(
                {
                    "epoch": cfg.num_epochs,
                    "model_state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"[ckpt] saved final {ckpt_path}", flush=True)

            plot_metrics(output_dir, save=output_dir / "metrics" / "summary.png")