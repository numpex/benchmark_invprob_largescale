from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import deepinv as dinv
import torch
from deepinv.distributed import DistributedContext, distribute
from deepinv.loss import SupLoss
from deepinv.loss.metric import PSNR

from .dataloader import CryoDataConfig, build_cryo_dataloaders
from toolscryo.plot_metrics import plot_metrics
from toolscryo.utils import (
    CsvTrainer,
    dump_config_json,
    ensure_dir,
    ExponentialLRWithFloor,
    seed_everything,
)


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


# CsvTrainer lives in toolscryo.utils — imported above for backwards compatibility.


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

        trainer = CsvTrainer(
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