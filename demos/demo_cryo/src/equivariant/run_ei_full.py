"""run_ei_full.py — Self-supervised EI training on **full volumes** with distributed tiling.

Equivariant, full-volume variant:
  - The network is wrapped with ``deepinv.distributed.distribute`` for tiled
    sliding-window inference, exactly as in the supervised Method.
  - Forward physics: MissingWedge at native volume resolution (wedge_double_size=False).
  - Loss: ObsLoss (cross half-set, Fourier domain) +
          EqLoss  (equivariance under rotated wedge, Fourier domain).
  - No ground-truth required — trains on EVN volumes; uses paired EVN+ODD when available.
  - Gradient accumulation over multiple volumes simulates a larger effective batch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import deepinv as dinv
import mrcfile
import numpy as np
import torch
from deepinv.distributed import DistributedContext, distribute

from equivariant.dataset_full import EIFullDataConfig, build_ei_full_dataloaders
from equivariant.physics import MissingWedge
from equivariant.transform import Rotate3D
from equivariant.utils import fsc_shell, save_fsc_figure, save_resolution_histogram, save_slice_figure, GpuFSC
from toolscryo.trainer import BaseTrainer, ExponentialLRWithFloor
from toolscryo.utils import dump_config_json, ensure_dir, seed_everything
from toolscryo.plot_metrics import plot_metrics


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
    # If True, disable the box window in ObsLoss and EqLoss (pass window=None).
    # Useful to test whether the window boundary at ±patch_size voxels causes artifacts.
    no_window: bool = False
    loss_icecream: bool = False  # If True, use icecream's original loss implementation instead of deepinv Loss subclasses.

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
    # FSC(f(EVN), f(ODD)) at each eval_interval epoch.
    eval_fsc: bool = True
    fsc_threshold: float = 0.143        # gold-standard FSC resolution criterion
    # Fallback pixel size (Å/px) used when MRC header voxel_size == 0.
    # If None and header is zero, falls back to 1.0 (resolution in shells, not Å).
    pixel_size_angstrom: float | None = None


# ---------------------------------------------------------------------------
# Trainer subclass — adds optional GT evaluator PSNR hook
# ---------------------------------------------------------------------------

class EIFullTrainer(BaseTrainer):
    """BaseTrainer subclass for self-supervised EI training on full volumes.

    Overrides the val step to run FSC(f(EVN), f(ODD)) directly inside
    ``compute_loss`` as deepinv iterates the val loader — avoiding the
    double-pass that occurred when a separate FullVolEvaluator re-iterated
    the same loader after the deepinv val loop.

    Attributes set after construction (in ``run_training``):
      ``_val_pixel_sizes``  list of Å/px per val volume
      ``_fsc_threshold``    FSC threshold (default 0.143)
    """

    _main_metric = "fsc_res_angstrom"
    _main_metric_higher_is_better = False  # lower resolution value = better

    def _save_vol_slices(self, images_dir, epoch, vol_idx, x, y, f_evn_t, f_odd_t, prefix=""):
        """Save raw slice figures for one volume (shared by train and val)."""
        evn_np = x.squeeze().cpu().numpy()
        odd_np = y.squeeze().cpu().numpy()
        f_evn  = f_evn_t.squeeze().cpu().numpy()
        f_odd  = f_odd_t.squeeze().cpu().numpy()
        save_slice_figure(
            images_dir, epoch, vol_idx,
            [evn_np, odd_np, f_evn, f_odd],
            labels=["Input EVN", "Input ODD", "f(EVN)", "f(ODD)"],
            title=f"{prefix}Epoch {epoch} | Vol {vol_idx} — raw",
            fname=f"vol{vol_idx:02d}_raw.png",
        )

    def forward_pass(self, x, y, physics, train):
        """Compute f(ODD) and f(EVN) once; val is handled directly in compute_loss."""
        if not train:
            return torch.zeros_like(y), None
        x_net = self.model_inference(y=y, physics=physics, x=x, train=True)  
        y_net = self.model_inference(y=x, physics=physics, x=y, train=True)   
        # Cache outputs for train-slice visualisation in compute_loss.
        self._last_train_xnet = x_net
        self._last_train_ynet = y_net
        return x_net, y_net

    def compute_loss(self, physics, x, y, train=True, epoch=None, step=False):  # type: ignore[override]
        if not train:
            # x = EVN tensor, y = ODD tensor (as yielded by the val DataLoader).
            # Run FSC(f(EVN), f(ODD)) for this volume and accumulate.
            # All ranks call the distributed model together → NCCL collectives stay in sync.
            if epoch != getattr(self, "_val_fsc_epoch", None):
                self._val_fsc_epoch = epoch
                self._val_resolutions: list[float] = []
                self._val_vol_idx = 0

            thr       = float(getattr(self, "_fsc_threshold", 0.143))
            px_list   = getattr(self, "_val_pixel_sizes", [])
            vol_idx   = self._val_vol_idx
            px        = px_list[vol_idx] if vol_idx < len(px_list) else 1.0

            with torch.no_grad():
                f_evn_t = self.model(x)
                f_odd_t = self.model(y)
            if hasattr(self.device, 'type') and self.device.type == "cuda":
                torch.cuda.synchronize()

            if not hasattr(self, "_gpu_fsc"):
                self._gpu_fsc = GpuFSC(f_evn_t.shape[-1], device=f_evn_t.device)

            fsc_curve = self._gpu_fsc(f_evn_t, f_odd_t)

            D         = int(f_evn_t.shape[-1])
            k         = fsc_shell(fsc_curve, thr)
            res       = D * px / max(k, 1)
            self._val_resolutions.append(res)

            # Inference image: recon = 0.5 * (f(A(f(EVN))) + f(A(f(ODD))))
            # Must be computed on ALL ranks (distributed model uses NCCL collectives).
            with torch.no_grad():
                recon_t = 0.5 * (self.model(physics.A(f_evn_t)) + self.model(physics.A(f_odd_t)))

            images_dir = getattr(self, "_images_dir", None)
            if images_dir is not None:
                save_fsc_figure(images_dir, epoch, f"vol{vol_idx:02d}.png",
                                fsc_curve, k, res,
                                f"Epoch {epoch} | Vol {vol_idx}",
                                thr, vol_size=D, pixel_size=px)
                self._save_vol_slices(images_dir, epoch, vol_idx, x, y, f_evn_t, f_odd_t)
                recon_np = recon_t.squeeze().cpu().numpy()
                x_np = x.squeeze().cpu().numpy()
                y_np = y.squeeze().cpu().numpy()
                save_slice_figure(
                    images_dir, epoch, vol_idx,
                    [x_np, y_np, recon_np],
                    labels=["EVN", "ODD", "recon"],
                    title=f"Epoch {epoch} | Vol {vol_idx} — inference recon",
                    fname=f"vol{vol_idx:02d}_recon.png",
                )

            self._val_vol_idx += 1
            # Return f_evn as x_net so deepinv can log it if needed.
            return torch.tensor(0.0, device=y.device), f_evn_t.detach(), {}

        result = super().compute_loss(physics, x, y, train=True, epoch=epoch, step=step)

        # ── Train slice figures — one per volume per epoch (mirrors val) ──
        train_images_dir = getattr(self, "_train_images_dir", None)
        if train_images_dir is not None:
            if epoch != getattr(self, "_train_slice_epoch", None):
                self._train_slice_epoch = epoch
                self._train_vol_idx = 0
            vol_idx = self._train_vol_idx
            self._save_vol_slices(
                train_images_dir, epoch, vol_idx,
                x, y,
                self._last_train_ynet.detach(),
                self._last_train_xnet.detach(),
                prefix="Train ",
            )
            self._train_vol_idx += 1

        return result

    def log_metrics_mlops(self, logs: dict, step: int, train: bool = True) -> None:  # type: ignore[override]
        if not train:
            resolutions = getattr(self, "_val_resolutions", [])
            if resolutions:
                res_arr   = np.array(resolutions)
                mean_res  = float(np.mean(res_arr))
                median_res = float(np.median(res_arr))
                q1_res    = float(np.percentile(res_arr, 25))
                q3_res    = float(np.percentile(res_arr, 75))
                logs["fsc_res_angstrom"] = mean_res
                logs["fsc_res_median"]   = median_res
                logs["fsc_res_q1"]       = q1_res
                logs["fsc_res_q3"]       = q3_res
                images_dir = getattr(self, "_images_dir", None)
                if images_dir is not None:
                    save_resolution_histogram(
                        images_dir, step, resolutions, mean_res, median_res, q1_res, q3_res,
                        threshold_label=str(getattr(self, "_fsc_threshold", 0.143)),
                    )
                if self.verbose:
                    print(
                        f"[fsc-eval] epoch={step}  "
                        f"mean={mean_res:.1f} Å  median={median_res:.1f} Å  "
                        f"Q1={q1_res:.1f} Å  Q3={q3_res:.1f} Å  (lower=better)",
                        flush=True,
                    )
        super().log_metrics_mlops(logs, step, train=train)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_val_header_info(
    val_loader,
    fallback_pixel_size: float | None = None,
) -> tuple[list[float], int]:
    """Read pixel sizes and count paired volumes from val dataset MRC headers.

    Returns ``(val_pixel_sizes, n_paired_val)`` by opening only the header of
    each EVN file — no voxel data loaded.

    When the MRC header reports ``voxel_size == 0``, uses *fallback_pixel_size*
    (from config ``pixel_size_angstrom``) if provided, else 1.0 (resolution
    will be in shell units rather than Å).
    """
    val_ds = val_loader.dataset
    val_pixel_sizes: list[float] = []
    n_paired_val = 0
    for evn_p, odd_p in zip(val_ds.evn_paths, val_ds.odd_paths):
        with mrcfile.open(str(evn_p), permissive=True, mode="r") as mrc:
            px = float(mrc.voxel_size.x)
        if px <= 0.0:
            px = fallback_pixel_size if fallback_pixel_size is not None else 1.0
        val_pixel_sizes.append(px)
        if odd_p is not None:
            n_paired_val += 1
    return val_pixel_sizes, n_paired_val


def _resolve_vol_size(data_bundle, cfg: RunEIFullConfig) -> int:
    """Return the spatial side length to use for MissingWedge physics.

    Reads only the MRC header (nx, ny, nz) — no volume data loaded.
    """
    first_path = data_bundle.train_loader.dataset.evn_paths[0]
    with mrcfile.open(str(first_path), permissive=True, mode="r") as mrc:
        nx, ny, nz = int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz)
    # MRC stores (nz=Z, ny=Y, nx=X); min gives the smallest spatial dimension.
    vol_size = min(nx, ny, nz)
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
        if bool(cfg.loss_icecream):
            print("[loss] using icecream's original loss implementation (not deepinv Loss subclasses)", flush=True)
            from equivariant.losses_icecream import ObsLoss, EqLoss
        else:
            print("[loss] using deepinv Loss subclasses ObsLoss and EqLoss", flush=True)
            from equivariant.losses import ObsLoss, EqLoss

        losses = [
            ObsLoss(physics, weight=1.0,
                    use_fourier=bool(cfg.use_fourier), view_as_real=bool(cfg.view_as_real),
                    no_window=bool(cfg.no_window)),
            EqLoss(physics, transform, weight=float(cfg.eq_weight),
                   use_fourier=bool(cfg.use_fourier), view_as_real=bool(cfg.view_as_real),
                   eq_use_direct=bool(cfg.eq_use_direct), no_window=bool(cfg.no_window)),
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

        images_dir       = ensure_dir(output_dir / "val_images")   if rank == 0 else None
        train_images_dir = ensure_dir(output_dir / "train_images") if rank == 0 else None
        trainer._metrics_dir      = ensure_dir(output_dir / "metrics")
        trainer._images_dir       = images_dir
        trainer._train_images_dir = train_images_dir
        trainer._ckpt_dir         = ensure_dir(output_dir / "checkpoints") if rank == 0 else None
        trainer._scheduler        = scheduler
        trainer._grad_accum_steps = max(1, int(cfg.grad_accumulation_steps))
        trainer.ckp_interval      = int(cfg.ckp_interval)

        # ── FSC evaluation setup ──────────────────────────────────────────────
        # pixel sizes and threshold are set on ALL ranks so the distributed
        # model's NCCL collectives stay synchronised across ranks during val.
        val_pixel_sizes, n_paired_val = _read_val_header_info(
            data_bundle.val_loader,
            fallback_pixel_size=cfg.pixel_size_angstrom,
        )
        if rank == 0:
            print(f"[fsc-eval] val pixel sizes (Å/px): {[f'{v:.2f}' for v in val_pixel_sizes]}", flush=True)

        if cfg.eval_fsc and n_paired_val > 0:
            trainer._val_pixel_sizes = val_pixel_sizes
            trainer._fsc_threshold   = float(cfg.fsc_threshold)
            if rank == 0:
                print(f"[fsc-eval] enabled for {n_paired_val} paired val volumes  thr={cfg.fsc_threshold}", flush=True)
        else:
            trainer._val_pixel_sizes = []
            trainer._fsc_threshold   = float(cfg.fsc_threshold)
            if rank == 0 and not cfg.eval_fsc:
                print("[fsc-eval] disabled (eval_fsc=False)", flush=True)
            elif rank == 0:
                print("[fsc-eval] disabled — no paired ODD val volumes found", flush=True)

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
