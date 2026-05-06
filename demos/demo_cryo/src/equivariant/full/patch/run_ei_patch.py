"""run_ei_patch.py — Self-supervised EI training on 72³ patches.

Method 2 (Equivariant, patch variant):
  - Forward physics: MissingWedge (Fourier-space mask)
  - Loss: ObsLoss (cross half-set, Fourier domain) +
          EqLoss  (equivariance under rotated wedge, Fourier domain)
  - No ground-truth required — trains on EVN volumes; uses paired EVN+ODD when available
  - Evaluation: data-fidelity loss on held-out volumes (FSC is a separate post-hoc step)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import deepinv as dinv
import torch
import torch.nn as nn
from deepinv.distributed import DistributedContext
from deepinv.loss.metric import PSNR

from .dataset import CryoEIPatchDataset, EIPatchDataConfig, build_ei_patch_dataloaders
from equivariant.physics import MissingWedge
from equivariant.transform import Rotate3D
from equivariant.losses import ObsLoss, EqLoss
from toolscryo.utils import CsvTrainer  # generic — reusable
from toolscryo.utils import dump_config_json, ensure_dir, ExponentialLRWithFloor, seed_everything
from toolscryo.plot_metrics import plot_metrics


@dataclass
class RunEIPatchConfig:
    # ── Data ────────────────────────────────────────────────────────────────
    output_dir: str = "./runs/demo_cryo_ei_patch"
    input_dir: str = "./dataset/empiar-11058"
    max_train_vols: int | None = None
    max_val_vols: int = 5
    seed: int = 0
    normalize_crops: bool = False      # per-crop re-normalisation (volume always normalised at load)

    # ── Patch ───────────────────────────────────────────────────────────────
    crop_size: int = 72
    # Icecream semantics: this "batch_size" is crops-per-volume (n_crops),
    # not DataLoader batch size.
    batch_size: int = 4
    # Legacy alias for previous config naming; ignored when batch_size is set.
    n_crops_per_vol: int | None = None
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True

    # ── Physics ─────────────────────────────────────────────────────────────
    tilt_max: float = 60.0
    tilt_min: float = -60.0
    use_spherical_support: bool = True
    wedge_double_size: bool = True

    # ── EI loss ─────────────────────────────────────────────────────────────
    eq_weight: float = 2.0            # weight of EqLoss relative to ObsLoss
    use_fourier: bool = False
    view_as_real: bool = True

    # ── GT evaluation ───────────────────────────────────────────────
    use_icecream_gt: bool = False     # enable PSNR + figure saving vs GT volume

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


# ---------------------------------------------------------------------------
# Icecream GT evaluator
# ---------------------------------------------------------------------------

class PatchEvaluator:
    """Samples fixed patches from val volumes, runs model, computes PSNR vs GT.

    At each call saves a figure per crop into ``images/epoch{N:04d}/``.
    PSNR is returned and logged into ``metrics/val_epochs.csv`` (``gt_psnr_db`` column).

    :param val_evn_paths:      EVN volume paths (val split).
    :param val_icecream_paths: GT volume paths (parallel, may be None).
    :param physics:            MissingWedge physics (for computing A(f(x))).
    :param crop_size:          Cubic patch side length.
    :param n_crops:            Number of crops to sample per vol (averaged for PSNR).
    :param device:             Torch device.
    :param seed:               RNG seed for reproducible crop locations.
    """

    def __init__(
        self,
        val_evn_paths: list[Path],
        val_icecream_paths: list[Path | None],
        physics: MissingWedge,
        crop_size: int,
        n_crops: int = 4,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> None:
        self.val_evn_paths      = val_evn_paths
        self.val_icecream_paths = val_icecream_paths
        self.physics            = physics
        self.crop_size          = crop_size
        self.n_crops            = n_crops
        self.device             = torch.device(device)
        self.seed               = seed
        self._images_dir: Path | None = None

    def __call__(self, epoch: int, model: nn.Module) -> float | None:
        """Run evaluation. Returns mean PSNR over all crops (or None if no GT)."""
        seed_everything(self.seed)
        model.eval()
        psnr_all: list[float] = []
        rng = np.random.default_rng(self.seed)

        with torch.no_grad():
            for vol_idx, (evn_path, ice_path) in enumerate(
                zip(self.val_evn_paths, self.val_icecream_paths)
            ):
                if ice_path is None:
                    continue

                evn_np = CryoEIPatchDataset._load_mrc(evn_path)
                ice_np = CryoEIPatchDataset._load_mrc(ice_path)
                D, H, W = evn_np.shape
                cs = self.crop_size

                psnr_vol: list[float] = []

                for crop_i in range(self.n_crops):
                    d0 = int(rng.integers(0, max(1, D - cs)))
                    h0 = int(rng.integers(0, max(1, H - cs)))
                    w0 = int(rng.integers(0, max(1, W - cs)))

                    def _crop(vol, d0=d0, h0=h0, w0=w0):
                        c = vol[d0:d0+cs, h0:h0+cs, w0:w0+cs]
                        if c.shape != (cs, cs, cs):
                            pad = np.zeros((cs, cs, cs), dtype=np.float32)
                            sd, sh, sw = c.shape
                            pad[:sd, :sh, :sw] = c
                            c = pad
                        return torch.from_numpy(c).unsqueeze(0).unsqueeze(0)  # (1,1,cs,cs,cs)

                    x  = _crop(evn_np).to(self.device)
                    gt = _crop(ice_np).to(self.device)
                    fx = model(x)

                    psnr_val   = float(PSNR(max_pixel=None)(fx.cpu(), gt.cpu()))
                    psnr_input = float(PSNR(max_pixel=None)(x.cpu(),  gt.cpu()))
                    psnr_vol.append(psnr_val)

                    if self._images_dir is not None:
                        self._save_figure(epoch, vol_idx, crop_i, x.cpu(), fx.cpu(), gt.cpu(), psnr_val, psnr_input)

                if psnr_vol:
                    psnr_all.append(float(np.mean(psnr_vol)))

        return float(np.mean(psnr_all)) if psnr_all else None

    def _save_figure(
        self,
        epoch: int,
        vol_idx: int,
        crop_idx: int,
        x: torch.Tensor,
        fx: torch.Tensor,
        gt: torch.Tensor,
        psnr: float,
        psnr_input: float = float("nan"),
    ) -> None:
        """Save a 1×3 mid-slice PNG: GT | Input (EVN) | f(x) predicted."""
        assert self._images_dir is not None

        def _mid_slice(t: torch.Tensor) -> np.ndarray:
            v = t[0, 0].float()
            if v.ndim == 3:
                v = v[v.shape[0] // 2]
            return v.numpy()

        gt_sl   = _mid_slice(gt)
        evn_sl  = _mid_slice(x)
        pred_sl = _mid_slice(fx)

        vmin, vmax = float(gt_sl.min()), float(gt_sl.max())

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, img, title in zip(
            axes,
            [gt_sl, evn_sl, pred_sl],
            ["GT", f"Input EVN ({psnr_input:.2f} dB)", f"f(x) predicted ({psnr:.2f} dB)"],
        ):
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(f"Epoch {epoch} | Vol {vol_idx} | Crop {crop_idx} | mid-slice")
        fig.tight_layout()
        fname = Path(self._images_dir) / f"epoch{epoch:04d}" / f"vol{vol_idx:02d}_crop{crop_idx:02d}.png"
        fname.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname, dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Trainer subclass with GT evaluator hook
# ---------------------------------------------------------------------------

class EIPatchTrainer(CsvTrainer):
    """CsvTrainer that additionally calls ``_gt_evaluator`` after each val pass."""

    _gt_evaluator: PatchEvaluator | None = None

    def _save_val_image(self, *args, **kwargs) -> None:  # type: ignore[override]
        """Suppressed: EI figures are produced by PatchEvaluator with the GT."""

    def log_metrics_mlops(self, logs: dict, step: int, train: bool = True) -> None:  # type: ignore[override]
        if (
            not train
            and self._gt_evaluator is not None
        ):
            mean_psnr = self._gt_evaluator(step, self.model)
            if mean_psnr is not None:
                logs["gt_psnr_db"] = mean_psnr
                print(f"[gt-eval] epoch={step}  mean_psnr={mean_psnr:.2f} dB", flush=True)
        super().log_metrics_mlops(logs, step, train=train)


# ---------------------------------------------------------------------------
# Training entry-point
# ---------------------------------------------------------------------------


def run_training(cfg: RunEIPatchConfig) -> None:
    seed_everything(int(cfg.seed))

    output_dir = ensure_dir(cfg.output_dir)
    dump_config_json(output_dir / "config.json", asdict(cfg))

    # Exact icecream behavior: number of crops sampled per volume item comes
    # from ``batch_size``.
    n_crops_per_vol = int(cfg.batch_size)
    if cfg.n_crops_per_vol is not None and int(cfg.n_crops_per_vol) != n_crops_per_vol:
        print(
            f"[ei-data] INFO: ignoring legacy n_crops_per_vol={cfg.n_crops_per_vol}; "
            f"using batch_size={cfg.batch_size} as crops-per-volume (icecream semantics).",
            flush=True,
        )

    data_cfg = EIPatchDataConfig(
        input_dir=cfg.input_dir,
        crop_size=int(cfg.crop_size),
        batch_size=n_crops_per_vol,
        n_crops_per_vol=cfg.n_crops_per_vol,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=int(cfg.prefetch_factor),
        persistent_workers=bool(cfg.persistent_workers),
        max_train_vols=cfg.max_train_vols,
        max_val_vols=int(cfg.max_val_vols),
        seed=int(cfg.seed),
        normalize_crops=bool(cfg.normalize_crops),
        use_icecream_gt=cfg.use_icecream_gt,
    )

    with DistributedContext(seed=int(cfg.seed), seed_offset=False, cleanup=True) as ctx:
        rank = int(ctx.rank)

        data_bundle = build_ei_patch_dataloaders(data_cfg)

        # ── Physics ──────────────────────────────────────────────────────────
        physics = MissingWedge(
            tilt_max=float(cfg.tilt_max),
            tilt_min=float(cfg.tilt_min),
            crop_size=int(cfg.crop_size),
            use_spherical_support=bool(cfg.use_spherical_support),
            wedge_double_size=bool(cfg.wedge_double_size),
            device=str(ctx.device),
        ).to(ctx.device)

        # ── Transform ────────────────────────────────────────────────────────
        transform = Rotate3D(n_trans=1)

        # ── Model ────────────────────────────────────────────────────────────
        model = dinv.models.UNet(
            in_channels=1,
            out_channels=1,
            scales=4,
            residual=True,
            batch_norm="biasfree",
            dim=3,
        ).to(ctx.device)

        if rank == 0:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"The model has {n_params} trainable parameters")

        # ── Losses ───────────────────────────────────────────────────────────
        # Icecream-style: cross half-set data-fidelity + equivariance,
        # both in the Fourier domain under the wedge mask.
        # Falls back automatically to single half-set when x == y (EVN-only).
        losses = [
            ObsLoss(physics, weight=1.0,
                    use_fourier=bool(cfg.use_fourier), view_as_real=bool(cfg.view_as_real)),
            EqLoss(physics, transform, weight=float(cfg.eq_weight),
                   use_fourier=bool(cfg.use_fourier), view_as_real=bool(cfg.view_as_real)),
        ]

        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
        scheduler = ExponentialLRWithFloor(
            optimizer,
            gamma=float(cfg.scheduler_gamma),
            lr_min=float(cfg.scheduler_lr_min),
        )

        trainer = EIPatchTrainer(
            model=model,
            physics=physics,
            optimizer=optimizer,
            train_dataloader=data_bundle.train_loader,
            eval_dataloader=data_bundle.val_loader,
            epochs=int(cfg.num_epochs),
            losses=losses,
            metrics=[],           # no GT → no PSNR
            online_measurements=False,   # y comes from the dataloader (FBP patches)
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
        trainer._metrics_dir = ensure_dir(output_dir / "metrics")
        trainer._images_dir  = images_dir
        trainer._ckpt_dir    = ensure_dir(output_dir / "checkpoints") if rank == 0 else None
        trainer._scheduler        = scheduler
        trainer._grad_accum_steps = 1
        trainer.ckp_interval = int(cfg.ckp_interval)

        # ── GT evaluator (optional) ───────────────────────────────────────────────
        if rank == 0 and cfg.use_icecream_gt and any(
            p is not None for p in data_bundle.val_icecream_paths
        ):
            evaluator = PatchEvaluator(
                val_evn_paths=data_bundle.val_paths,
                val_icecream_paths=data_bundle.val_icecream_paths,
                physics=physics,
                crop_size=int(cfg.crop_size),
                n_crops=4,
                device=ctx.device,
                seed=int(cfg.seed),
            )
            evaluator._images_dir  = images_dir
            trainer._gt_evaluator = evaluator
            n_ice = sum(p is not None for p in data_bundle.val_icecream_paths)
            print(f"[gt-eval] enabled for {n_ice} val volumes", flush=True)
        else:
            trainer._gt_evaluator = None

        trainer.train()

        # ── Save final checkpoint ─────────────────────────────────────────────
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
