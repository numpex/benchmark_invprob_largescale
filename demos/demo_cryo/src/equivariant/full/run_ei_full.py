"""run_ei_full.py — Self-supervised EI training on **full volumes** with distributed tiling.

Method 3 (Equivariant, full-volume variant):
  - The network is wrapped with ``deepinv.distributed.distribute`` for tiled
    sliding-window inference, exactly as in the supervised Method 1.
  - Forward physics: MissingWedge at native volume resolution (wedge_double_size=False).
  - Loss: ObsLoss (cross half-set, Fourier domain) +
          EqLoss  (equivariance under rotated wedge, Fourier domain).
  - No ground-truth required — trains on EVN volumes; uses paired EVN+ODD when available.
  - Gradient accumulation over multiple volumes simulates a larger effective batch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import deepinv as dinv
import mrcfile
import numpy as np
import torch
from deepinv.distributed import DistributedContext, distribute

from .dataset import CryoEIFullDataset, EIFullDataConfig, build_ei_full_dataloaders
from equivariant.physics import MissingWedge
from equivariant.transform import Rotate3D
from equivariant.losses import ObsLoss, EqLoss
from deepinv.loss.metric import PSNR
import torch.nn as nn
from toolscryo.utils import CsvTrainer
from toolscryo.utils import dump_config_json, ensure_dir, ExponentialLRWithFloor, seed_everything
from toolscryo.plot_metrics import plot_metrics


# ---------------------------------------------------------------------------
# Full-volume GT evaluator
# ---------------------------------------------------------------------------

class FullVolEvaluator:
    """Load full val volumes from disk, run model, compute PSNR vs icecream GT.
    """

    def __init__(
        self,
        val_evn_paths: list[Path],
        val_icecream_paths: list[Path | None],
        device: str | torch.device = "cpu",
        target_shape: tuple[int, int, int] | None = None,
        seed: int = 42,
    ) -> None:
        self.val_evn_paths      = val_evn_paths
        self.val_icecream_paths = val_icecream_paths
        self.device             = torch.device(device)
        self.target_shape       = target_shape
        self.seed               = seed
        self._images_dir: Path | None = None

    def __call__(self, epoch: int, model: nn.Module) -> float | None:
        import torch.nn.functional as F

        seed_everything(self.seed)
        model.eval()
        psnr_all: list[float] = []

        with torch.no_grad():
            for vol_idx, (evn_path, ice_path) in enumerate(
                zip(self.val_evn_paths, self.val_icecream_paths)
            ):
                if ice_path is None:
                    continue

                def _load(path: Path) -> torch.Tensor:
                    # Mirrors _load_and_prepare exactly: resample → centre-crop → normalise
                    vol_np = CryoEIFullDataset._load_mrc(path)       # (Y,X,Z) raw float
                    vol = torch.from_numpy(vol_np)                   # (D,H,W)
                    if self.target_shape is not None:
                        vol = F.interpolate(
                            vol.unsqueeze(0).unsqueeze(0),
                            size=self.target_shape,
                            mode="trilinear",
                            align_corners=False,
                        ).squeeze(0).squeeze(0)
                    # Centre-crop to cube
                    D, H, W = vol.shape
                    S = min(D, H, W)
                    d0, h0, w0 = (D - S) // 2, (H - S) // 2, (W - S) // 2
                    vol = vol[d0:d0 + S, h0:h0 + S, w0:w0 + S]
                    # Normalise after crop
                    vol = (vol - vol.mean()) / (vol.std() + 1e-8)
                    return vol.unsqueeze(0).unsqueeze(0)             # (1,1,S,S,S)

                x  = _load(evn_path).to(self.device)
                gt = _load(ice_path).to(self.device)
                fx = model(x)

                psnr_val   = float(PSNR(max_pixel=None)(fx.cpu(), gt.cpu()))
                psnr_input = float(PSNR(max_pixel=None)(x.cpu(),   gt.cpu()))
                psnr_all.append(psnr_val)

                if self._images_dir is not None:
                    self._save_figure_full(epoch, vol_idx, x.cpu(), fx.cpu(), gt.cpu(), psnr_val, psnr_input)

        return float(np.mean(psnr_all)) if psnr_all else None

    def _save_figure_full(
        self,
        epoch: int,
        vol_idx: int,
        x: torch.Tensor,
        fx: torch.Tensor,
        gt: torch.Tensor,
        psnr: float,
        psnr_input: float = float("nan"),
    ) -> None:
        assert self._images_dir is not None

        def _slices(t: torch.Tensor):
            """Return mid-slice along D, H, W axes as numpy arrays."""
            v = t[0, 0].float()   # (D, H, W)
            D, H, W = v.shape
            return [
                v[D // 2, :, :].numpy(),   # axial   (H×W)
                v[:, H // 2, :].numpy(),   # coronal (D×W)
                v[:, :, W // 2].numpy(),   # sagittal(D×H)
            ]

        axis_labels = ["axial (D)", "coronal (H)", "sagittal (W)"]
        col_titles  = ["Icecream", f"Input corrected ({psnr_input:.2f} dB)", f"Pred f(x) ({psnr:.2f} dB)"]
        vols        = [gt, x, fx]

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for row, (label, *slices_per_vol) in enumerate(
            zip(axis_labels, *[_slices(v) for v in vols])
        ):
            vmin = float(slices_per_vol[0].min())
            vmax = float(slices_per_vol[0].max())
            for col, (ax, img) in enumerate(zip(axes[row], slices_per_vol)):
                ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
                ax.axis("off")
                if row == 0:
                    ax.set_title(col_titles[col])
                if col == 0:
                    ax.set_ylabel(label)

        fig.suptitle(f"Epoch {epoch} | Vol {vol_idx} | mid-slices per axis")
        fig.tight_layout()
        fname = Path(self._images_dir) / f"epoch{epoch:04d}" / f"vol{vol_idx:02d}.png"
        fname.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname, dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RunEIFullConfig:
    # ── Data ────────────────────────────────────────────────────────────────
    output_dir: str = "./runs/demo_cryo_ei_full"
    input_dir: str = "./dataset/empiar-11058"
    max_train_vols: int | None = None
    max_val_vols: int = 5
    seed: int = 0
    # Trilinear resample to (D, H, W) — same as supervised CryoDataConfig.target_shape.
    # None = use each volume's native resolution (auto-resolved for physics).
    target_shape: tuple[int, int, int] | None = None

    # ── DataLoader ──────────────────────────────────────────────────────────
    batch_size: int = 1                  # always 1 full volume per step; kept for API parity
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True

    # ── Physics ─────────────────────────────────────────────────────────────
    tilt_max: float = 60.0
    tilt_min: float = -60.0
    use_spherical_support: bool = True
    wedge_double_size: bool = False     
    wedge_low_support: float = 0.0
    ref_wedge_support: float = 1.0

    # ── EI loss ─────────────────────────────────────────────────────────────
    eq_weight: float = 2.0
    use_fourier: bool = False
    view_as_real: bool = True
    # If True, use plain spatial MSE for the equivariance loss instead of
    # Fourier-masked fourier_loss_batch (icecream eq_use_direct). Default False = paper setting.
    eq_use_direct: bool = False

    # ── Distributed model tiling (mirrors supervised RunConfig) ─────────────
    patch_size: tuple[int, int, int] = (64, 64, 64)
    overlap: tuple[int, int, int] = (8, 8, 8)
    max_batch_size: int | None = 2
    checkpoint_batches: str | int | None = "auto"

    # ── Training ────────────────────────────────────────────────────────────
    num_epochs: int = 100
    learning_rate: float = 1e-4
    # Exponential LR decay: lr(n) = max(scheduler_lr_min, lr * scheduler_gamma^n)
    # Default: 1e-4 → 1e-5 at epoch 10 (gamma = 10^{-1/10} ≈ 0.7943), floor at 5e-6.
    scheduler_gamma: float = 0.7943     # per-epoch multiplicative decay
    scheduler_lr_min: float = 5e-6      # minimum LR floor
    grad_clip: float | None = 1.0
    ckp_interval: int = 10
    eval_interval: int = 1
    grad_accumulation_steps: int = 4    # accumulate over N volumes before each optimiser step

    # ── Evaluation ──────────────────────────────────────────────────────────
    use_icecream_gt: bool = False


# ---------------------------------------------------------------------------
# Trainer subclass — adds optional GT evaluator PSNR hook
# ---------------------------------------------------------------------------

class EIFullTrainer(CsvTrainer):
    """CsvTrainer that optionally calls ``_gt_evaluator`` after each val pass."""

    _gt_evaluator: FullVolEvaluator | None = None

    def _save_val_image(self, *args, **kwargs) -> None:  # type: ignore[override]
        """Suppressed: EI figures are produced by FullVolEvaluator with the GT."""

    def compute_loss(self, physics, x, y, train=True, **kwargs):  # type: ignore[override]
        """During validation, skip model inference and loss computation entirely.

        The only useful val-time work is the GT PSNR computed by ``_gt_evaluator``
        inside ``log_metrics_mlops``, which fires on the last val batch regardless.
        Skipping here reduces val cost from ~4 model calls per volume to zero
        (the evaluator adds exactly 1 call per val volume).
        """
        if not train:
            # Return (loss, x_net, logs) with dummy values — deepinv step() unpacks 3.
            # metrics=[] and plot_images=False, so x_net=None is safe.
            return torch.tensor(0.0, device=y.device), None, {}
        return super().compute_loss(physics, x, y, train=True, **kwargs)

    def log_metrics_mlops(self, logs: dict, step: int, train: bool = True) -> None:  # type: ignore[override]
        if not train and self._gt_evaluator is not None:
            # All ranks call the evaluator together so the distribute'd model
            # NCCL collectives are properly synchronised across ranks.
            mean_psnr = self._gt_evaluator(step, self.model)
            self.model.train()   # restore train mode after eval
            if mean_psnr is not None and self.verbose:  # rank-0 only print
                logs["gt_psnr_db"] = mean_psnr
                print(f"[gt-eval] epoch={step}  mean_psnr={mean_psnr:.2f} dB", flush=True)
        super().log_metrics_mlops(logs, step, train=train)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_vol_size(data_bundle, cfg: RunEIFullConfig) -> int:
    """Return the spatial side length to use for MissingWedge physics.

    Loads the first EVN volume and returns its minimum spatial dimension.
    """
    first_path = data_bundle.train_paths[0]
    vol = np.array(
        mrcfile.open(str(first_path), permissive=True).data,
        dtype=np.float32,
    )
    # MRC is (Z, Y, X); after moveaxis we get (Y, X, Z) = (D, H, W).
    # Use the minimum dimension as the cube side (conservative choice).
    vol_size = int(min(vol.shape))
    print(f"[ei-full] auto vol_size={vol_size}  (from {first_path.name})", flush=True)
    return vol_size


# ---------------------------------------------------------------------------
# Training entry-point
# ---------------------------------------------------------------------------

def run_training(cfg: RunEIFullConfig) -> None:
    seed_everything(int(cfg.seed))

    output_dir = ensure_dir(cfg.output_dir)
    dump_config_json(output_dir / "config.json", asdict(cfg))

    data_cfg = EIFullDataConfig(
        input_dir=cfg.input_dir,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=int(cfg.prefetch_factor),
        persistent_workers=bool(cfg.persistent_workers),
        max_train_vols=cfg.max_train_vols,
        max_val_vols=int(cfg.max_val_vols),
        seed=int(cfg.seed),
        target_shape=cfg.target_shape,
        use_icecream_gt=cfg.use_icecream_gt,
    )

    with DistributedContext(seed=int(cfg.seed), seed_offset=False, cleanup=True) as ctx:
        rank = int(ctx.rank)

        data_bundle = build_ei_full_dataloaders(data_cfg)

        # ── Resolve volume size for physics ─────────────────────────────────
        if cfg.target_shape is not None:
            vol_size = int(min(cfg.target_shape))
            print(f"[ei-full] target_shape={cfg.target_shape} → physics crop_size={vol_size}", flush=True)
        else:
            vol_size = _resolve_vol_size(data_bundle, cfg)

        # ── Physics ──────────────────────────────────────────────────────────
        # MissingWedge — volumes are always cubic (centre-cropped in _load_and_prepare).
        # wedge_double_size=False: 2×vol_size FFT would be infeasible for large volumes.
        physics = MissingWedge(
            tilt_max=float(cfg.tilt_max),
            tilt_min=float(cfg.tilt_min),
            crop_size=vol_size,
            use_spherical_support=bool(cfg.use_spherical_support),
            wedge_double_size=bool(cfg.wedge_double_size),
            wedge_low_support=float(cfg.wedge_low_support),
            ref_wedge_support=float(cfg.ref_wedge_support),
            device=str(ctx.device),
        ).to(ctx.device)

        # ── Transform ────────────────────────────────────────────────────────
        transform = Rotate3D(n_trans=1)

        # ── Model ────────────────────────────────────────────────────────────
        # Exact same distribute pattern as supervised Method 1.
        backbone = dinv.models.UNet(
            in_channels=1,
            out_channels=1,
            scales=4,
            residual=True,
            batch_norm="biasfree",
            dim=3,
        ).to(ctx.device)

        backbone = distribute(
            backbone,
            ctx,
            patch_size=tuple(int(v) for v in cfg.patch_size),
            overlap=tuple(int(v) for v in cfg.overlap),
            tiling_dims=(-3, -2, -1),
            max_batch_size=cfg.max_batch_size,
            checkpoint_batches=cfg.checkpoint_batches,
        )

        model = backbone  # EqLoss/ObsLoss call model(x) directly — no ArtifactRemoval wrapper

        if rank == 0:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[ei-full] model params={n_params:,}", flush=True)
            print(
                f"[ei-full] vol_size={vol_size}  patch_size={cfg.patch_size}  "
                f"overlap={cfg.overlap}  max_batch_size={cfg.max_batch_size}  "
                f"checkpoint_batches={cfg.checkpoint_batches}",
                flush=True,
            )

        # ── Losses ───────────────────────────────────────────────────────────
        # ObsLoss and EqLoss are unchanged — they work at any spatial extent.
        losses = [
            ObsLoss(physics, weight=1.0,
                    use_fourier=bool(cfg.use_fourier), view_as_real=bool(cfg.view_as_real)),
            EqLoss(physics, transform, weight=float(cfg.eq_weight),
                   use_fourier=bool(cfg.use_fourier), view_as_real=bool(cfg.view_as_real),
                   eq_use_direct=bool(cfg.eq_use_direct)),
        ]

        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
        scheduler = ExponentialLRWithFloor(
            optimizer,
            gamma=float(cfg.scheduler_gamma),
            lr_min=float(cfg.scheduler_lr_min),
        )

        trainer = EIFullTrainer(
            model=model,
            physics=physics,
            optimizer=optimizer,
            train_dataloader=data_bundle.train_loader,
            eval_dataloader=data_bundle.val_loader,
            epochs=int(cfg.num_epochs),
            losses=losses,
            metrics=[],                       # no paired GT → no PSNR
            online_measurements=False,        # (y, x) pairs come from DataLoader
            device=ctx.device,
            save_path=None,
            ckp_interval=int(cfg.ckp_interval),
            eval_interval=int(cfg.eval_interval),
            grad_clip=cfg.grad_clip,
            check_grad=cfg.grad_clip is not None,
            plot_images=False,
            verbose=rank == 0,
            show_progress_bar=rank == 0,
            log_train_batch=False,
            optimizer_step_multi_dataset=False,
        )

        images_dir = ensure_dir(output_dir / "images") if rank == 0 else None
        trainer._metrics_dir      = ensure_dir(output_dir / "metrics")
        trainer._images_dir       = images_dir
        trainer._ckpt_dir         = ensure_dir(output_dir / "checkpoints") if rank == 0 else None
        trainer._scheduler        = scheduler
        trainer._grad_accum_steps = max(1, int(cfg.grad_accumulation_steps))
        trainer.ckp_interval      = int(cfg.ckp_interval)

        # ── GT evaluator (optional) ──────────────────────────────────────────
        # IMPORTANT: _gt_evaluator must be set on ALL ranks, not just rank 0.
        # FullVolEvaluator calls the distribute'd model which requires all ranks
        # to participate in the NCCL collective together.  Setting it only on
        # rank 0 causes a deadlock: rank 0 calls model(x) in the evaluator while
        # ranks 1-3 have already moved to the next epoch and call model(x) there.
        if cfg.use_icecream_gt and any(
            p is not None for p in data_bundle.val_icecream_paths
        ):
            evaluator = FullVolEvaluator(
                val_evn_paths      = data_bundle.val_paths,
                val_icecream_paths = data_bundle.val_icecream_paths,
                device             = ctx.device,
                target_shape       = cfg.target_shape,
                seed               = int(cfg.seed),
            )
            # Only rank 0 saves figures; all ranks run inference.
            evaluator._images_dir = images_dir  # None on non-rank-0
            trainer._gt_evaluator = evaluator
            if rank == 0:
                n_ice = sum(p is not None for p in data_bundle.val_icecream_paths)
                print(f"[gt-eval] enabled for {n_ice} val volumes", flush=True)
        else:
            trainer._gt_evaluator = None

        trainer.train()

        # ── Final checkpoint ──────────────────────────────────────────────────
        if rank == 0 and trainer._ckpt_dir is not None:
            ckpt_path = Path(trainer._ckpt_dir) / "ckp_final.pth"
            torch.save(
                {
                    "epoch": cfg.num_epochs,
                    "model_state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"[ckpt] saved final {ckpt_path}", flush=True)

            plot_metrics(output_dir, save=output_dir / "metrics" / "summary.png")
