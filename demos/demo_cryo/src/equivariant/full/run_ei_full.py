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

import numpy as np
import torch
from deepinv.distributed import DistributedContext, distribute

from equivariant.full.dataset_full import EIFullDataConfig, build_ei_full_dataloaders
from equivariant.physics import MissingWedge
from equivariant.transform import Rotate3D
from equivariant.losses import EqLoss, ObsLoss
from equivariant.utils import GpuFSC, _read_mrc_vol_size, _read_pixel_sizes, fsc_shell, half_set_recon, save_fsc_figure, save_resolution_histogram, save_slice_figure
from toolscryo.trainer import EIBaseTrainer, ExponentialLRWithFloor
from toolscryo.utils import build_ei_model, dump_config_json, ensure_dir, seed_everything
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

    # ── Mixed precision ──────────────────────────────────────────────────────
    use_mixed_precision: bool = False  # fp16 forward + scaled backward (~2x faster, same as icecream default)

    # ── Model selection ──────────────────────────────────────────────────────
    # "unet"   → icecream_orig UNet3D (no norm, dropout, exact icecream arch)
    # "drunet" → dinv.models.DRUNet  (residual U-Net with noise-level conditioning)
    model_type: str = "unet"
    unet_f_maps: int = 64          # unet only: base feature-map count
    unet_num_levels: int = 4       # unet only: encoder depth
    unet_dropout: float = 0.1      # unet only: dropout probability
    drunet_nb: int = 4             # drunet only: residual blocks per scale
    drunet_sigma: float = 0.0      # drunet only: fixed noise-level injected into the noise map

    # ── Evaluation ──────────────────────────────────────────────────────────
    # FSC(f(EVN), f(ODD)) at each eval_interval epoch.
    eval_fsc: bool = True
    fsc_threshold: float = 0.143        # gold-standard FSC resolution criterion
    # Fallback pixel size (Å/px) used when MRC header voxel_size == 0.
    # If None and header is zero, falls back to 1.0 (resolution in shells, not Å).
    pixel_size_angstrom: float | None = None

    # ── Pretrained init ──────────────────────────────────────────────────────
    # Path to a local .pth checkpoint whose weights are loaded before training
    # begins.  Set to None (default) to train from a random initialisation.
    pretrained_ckpt: str | None = None



# ---------------------------------------------------------------------------
# Trainer subclass — adds optional GT evaluator PSNR hook
# ---------------------------------------------------------------------------

class EIFullTrainer(EIBaseTrainer):
    """EIBaseTrainer subclass for self-supervised EI training on full volumes.

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
            with torch.no_grad():
                recon_t = half_set_recon(self.model, physics, f_evn_t, f_odd_t)

            images_dir = getattr(self, "_images_dir", None)
            if images_dir is not None:
                save_fsc_figure(images_dir, epoch, f"vol{vol_idx:02d}.png",
                                fsc_curve, k, res,
                                f"Epoch {epoch} | Vol {vol_idx}",
                                thr, vol_size=D, pixel_size=px)
                recon_np = recon_t.squeeze().cpu().numpy()
                recon_np = (recon_np - recon_np.mean()) / (recon_np.std() + 1e-8)
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

        # ── Train slice figures — one per volume per eval_interval epoch ──
        # Track epoch/vol on all ranks so the vol_idx is consistent.
        if epoch != getattr(self, "_train_slice_epoch", None):
            self._train_slice_epoch = epoch
            self._train_vol_idx = 0
        vol_idx = self._train_vol_idx

        if epoch % self.eval_interval == 0:
            with torch.no_grad():
                f_evn_t = self._last_train_xnet.detach()
                f_odd_t = self._last_train_ynet.detach()
                recon_t = half_set_recon(self.model, physics, f_evn_t, f_odd_t)

            train_images_dir = getattr(self, "_train_images_dir", None)
            if train_images_dir is not None:
                recon_np = recon_t.squeeze().cpu().numpy()
                recon_np = (recon_np - recon_np.mean()) / (recon_np.std() + 1e-8)
                x_np = x.squeeze().cpu().numpy()
                y_np = y.squeeze().cpu().numpy()
                save_slice_figure(
                    train_images_dir, epoch, vol_idx,
                    [x_np, y_np, recon_np],
                    labels=["EVN", "ODD", "recon"],
                    title=f"Train Epoch {epoch} | Vol {vol_idx} — inference recon",
                    fname=f"vol{vol_idx:02d}_recon.png",
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
    """Read pixel sizes and count paired volumes from val dataset MRC headers."""
    val_ds = val_loader.dataset
    val_pixel_sizes = _read_pixel_sizes(val_ds.evn_paths, fallback=fallback_pixel_size)
    n_paired_val = sum(1 for p in val_ds.odd_paths if p is not None)
    return val_pixel_sizes, n_paired_val


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
        fallback_tilt_min=cfg.tilt_min,
        fallback_tilt_max=cfg.tilt_max,
    )

    with DistributedContext(seed=int(cfg.seed), seed_offset=False, cleanup=True) as ctx:
        rank = int(ctx.rank)

        data_bundle = build_ei_full_dataloaders(data_cfg)

        # ── Resolve volume size for physics ─────────────────────────────────
        if cfg.target_shape is not None:
            vol_size = int(min(cfg.target_shape))
            print(f"[ei-full] target_shape={cfg.target_shape} → physics crop_size={vol_size}", flush=True)
        else:
            first_path = data_bundle.train_loader.dataset.evn_paths[0]
            vol_size = _read_mrc_vol_size(first_path)
            print(f"[ei-full] auto vol_size={vol_size}  (from {first_path.name})", flush=True)

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
        distribute_kwargs = dict(
            patch_size=tuple(int(v) for v in cfg.patch_size),
            overlap=tuple(int(v) for v in cfg.overlap),
            tiling_dims=(-3, -2, -1),
            max_batch_size=cfg.max_batch_size,
            checkpoint_batches=cfg.checkpoint_batches,
        )

        wrapper, model_info = build_ei_model(
            cfg.model_type, cfg.unet_f_maps, cfg.unet_num_levels, cfg.unet_dropout,
            cfg.drunet_nb, cfg.drunet_sigma, ctx.device,
        )

        # ── Load pretrained weights before distributing ──────────────────────
        if cfg.pretrained_ckpt is not None:
            ckpt = torch.load(cfg.pretrained_ckpt, map_location=ctx.device, weights_only=True)
            state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
            if any(k.startswith("processor.") for k in state):
                state = {k.removeprefix("processor."): v for k, v in state.items()}
                if rank == 0:
                    print("[ei-full] stripped 'processor.' prefix from checkpoint keys (old format)", flush=True)
            wrapper.load_state_dict(state, strict=True)
            if rank == 0:
                print(f"[ei-full] loaded pretrained weights from {cfg.pretrained_ckpt}", flush=True)

        model = distribute(wrapper, ctx, type_object="denoiser", **distribute_kwargs)

        if rank == 0:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[ei-full] model={model_info}  params={n_params:,}", flush=True)
            print(
                f"[ei-full] vol_size={vol_size}  patch_size={cfg.patch_size}  "
                f"overlap={cfg.overlap}  max_batch_size={cfg.max_batch_size}  "
                f"checkpoint_batches={cfg.checkpoint_batches}",
                flush=True,
            )

        # ── Losses ───────────────────────────────────────────────────────────
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
            eval_dataloader=data_bundle.val_loader if len(data_bundle.val_loader.dataset) > 0 else None,
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
        trainer._grad_accum_steps    = max(1, int(cfg.grad_accumulation_steps))
        trainer.ckp_interval         = int(cfg.ckp_interval)

        if cfg.use_mixed_precision:
            trainer._enable_mixed_precision()
            if rank == 0:
                print("[ei-full] mixed precision enabled", flush=True)


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
                    "model_state_dict": trainer.model.processor.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"[ckpt] saved final {ckpt_path}", flush=True)

            plot_metrics(output_dir, save=output_dir / "metrics" / "summary.png")
