"""run_ei_patch.py — Self-supervised EI training on cubic patches.

Equivariant, patch-based variant:
  - Data: random crop_size³ patches from paired EVN+ODD half-set volumes.
  - Physics: MissingWedge with wedge_double_size=True (icecream default —
    feasible for small patches; 72³ → 145³ FFT is cheap).
  - Loss: ObsLoss (cross half-set Fourier domain) +
          EqLoss  (equivariance under rotated wedge, Fourier domain).
    Both are imported from losses.py — exact icecream equivalents.
  - Model: plain UNet (no distribute wrapper — patches are small).
  - No ground-truth required.

Results match icecream because the loss formulas, wedge construction, and
rotation set are identical.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from deepinv.distributed import DistributedContext

import mrcfile
import numpy as np

from equivariant.patch.dataset_patch import EIPatchDataConfig, build_ei_patch_dataloaders
from equivariant.losses import EqLoss, ObsLoss, _symmetrize_and_binarize
from equivariant.physics import MissingWedge
from equivariant.transform import Rotate3D
from equivariant.utils import (
    _center_crop, _find_mrc, _znorm, save_slice_figure,
    GpuFSC, fsc_shell, save_fsc_figure, _read_pixel_sizes,
)
from equivariant.patch.run_ei_patch_inference import patch_inference, _load_comparison, _load_vol_normalized



from toolscryo.trainer import EIBaseTrainer, ExponentialLRWithFloor
from toolscryo.utils import build_ei_model, dump_config_json, ensure_dir, seed_everything
from toolscryo.plot_metrics import plot_metrics

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RunEIPatchConfig:
    # ── Data ────────────────────────────────────────────────────────────────
    output_dir: str = "./runs/demo_cryo_ei_patch"
    input_dir: str = "./dataset/empiar-11058"
    max_train_vols: int | None = None
    max_val_vols: int = 5
    seed: int = 0

    # ── Patch ───────────────────────────────────────────────────────────────
    crop_size: int = 72
    n_crops_per_vol: int = 10
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    normalize: bool = True

    # ── Physics ─────────────────────────────────────────────────────────────
    tilt_max: float = 60.0
    tilt_min: float = -60.0
    use_spherical_support: bool = True
    # wedge_double_size=True is icecream's default and fine for patches
    # (72³ → 145³ FFT is cheap; full volumes cannot afford this).
    wedge_double_size: bool = True
    wedge_low_support: float = 0.0
    ref_wedge_support: float = 1.0

    # ── EI loss ─────────────────────────────────────────────────────────────
    eq_weight: float = 2.0
    use_fourier: bool = False
    view_as_real: bool = True
    eq_use_direct: bool = False
    no_window: bool = False

    # ── Training ────────────────────────────────────────────────────────────
    num_epochs: int = 100
    learning_rate: float = 1e-4
    scheduler_gamma: float = 1.0
    scheduler_lr_min: float = 5e-6
    grad_clip: float | None = 1.0
    ckp_interval: int = 10
    eval_interval: int = 1
    grad_accumulation_steps: int = 1
    log_every_n_epochs: int = 100  # print loss summary every N epochs (no per-epoch progress bar)

    # ── Inference (post-training sliding-window) ─────────────────────────────
    infer_stride: int = 36         # icecream default stride for sliding-window inference
    infer_batch_size: int = 0      # 0 = same as training batch_size
    infer_downsample: int = 1      # spatial downsample factor before inference (1 = no downsampling)
    infer_train: bool = True       # run sliding-window inference on training volumes
    infer_val: bool = True         # run sliding-window inference on validation volumes
    save_mrc: bool = False         # save reconstruction as .mrc file

    # ── Batching ────────────────────────────────────────────────────────────
    # ── Mixed precision ──────────────────────────────────────────────────────
    use_mixed_precision: bool = False  # fp16 forward + scaled backward (~2x faster, same as icecream default)

    # ── Model selection ──────────────────────────────────────────────────────
    # "unet"   → icecream_orig UNet3D (no norm, dropout, exact icecream arch)
    # "drunet" → dinv.models.DRUNet  (residual U-Net with noise-level conditioning)
    model_type: str = "drunet"
    unet_f_maps: int = 64          # unet only: base feature-map count
    unet_num_levels: int = 4       # unet only: encoder depth
    unet_dropout: float = 0.1      # unet only: dropout probability
    drunet_nb: int = 4             # drunet only: residual blocks per scale
    drunet_sigma: float = 0.0      # drunet only: fixed noise-level injected into the noise map

    # ── Evaluation ──────────────────────────────────────────────────────────
    # FSC on patches is not meaningful; kept False by default.
    # Post-hoc FSC on full volumes can be run separately via run_ei_inference.
    eval_fsc: bool = False
    fsc_threshold: float = 0.143
    pixel_size_angstrom: float | None = None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class EIPatchTrainer(EIBaseTrainer):
    """EIBaseTrainer subclass for self-supervised EI training on patches.

    Train step: computes f(EVN) and f(ODD), then runs ObsLoss + EqLoss.
    Val step: computes the same losses on held-out patches (no FSC).

    Attributes set after construction (in run_training):
      ``_images_dir``   Path | None — for mid-slice PNG figures
    """

    def _save_patch_slices(self, images_dir, epoch, batch_idx, x, y, f_evn_t, f_odd_t, prefix=""):
        """Save mid-slice figures for one patch batch."""
        evn_np = x[0].squeeze().cpu().numpy()
        odd_np = y[0].squeeze().cpu().numpy()
        f_evn  = f_evn_t[0].squeeze().cpu().numpy()
        f_odd  = f_odd_t[0].squeeze().cpu().numpy()
        save_slice_figure(
            images_dir, epoch, batch_idx,
            [evn_np, odd_np, f_evn, f_odd],
            labels=["Input EVN", "Input ODD", "f(EVN)", "f(ODD)"],
            title=f"{prefix}Epoch {epoch} | Batch {batch_idx}",
            fname=f"batch{batch_idx:04d}_raw.png",
        )

    def log_metrics_mlops(self, logs: dict, step: int, train: bool = True) -> None:  # type: ignore[override]
        super().log_metrics_mlops(logs, step, train=train)
        if not getattr(self, "_is_rank0", self.verbose):
            return
        n = getattr(self, "_log_every_n_epochs", 100)
        if train:
            if not hasattr(self, "_block_start_time"):
                self._block_start_time = time.perf_counter()
            if step % n == 0:
                block_elapsed = time.perf_counter() - self._block_start_time
                loss_str = "  ".join(f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, float))
                print(
                    f"[epoch {step:>5d}]  {loss_str}  "
                    f"[last {n} ep: {block_elapsed:.1f}s, {block_elapsed/n:.2f}s/ep]",
                    flush=True,
                )
                self._block_start_time = time.perf_counter()

    def compute_loss(self, physics, x, y, train=True, epoch=None, step=False):  # type: ignore[override]
        if not train:
            # Val: run f(EVN) and f(ODD) under no_grad for loss logging only.
            with torch.no_grad():
                x_net = self.model(x)
                y_net = self.model(y)
            loss_total = torch.tensor(0.0, device=x.device)
            logs: dict = {}
            for k, loss_fn in enumerate(self.losses):
                loss = loss_fn(
                    x=x, x_net=x_net, y=y, y_net=y_net,
                    physics=physics, model=self.model, epoch=epoch,
                )
                loss_total = loss_total + loss.mean()
                meters = self.logs_losses_eval[k]
                meters.update(loss.detach().cpu().numpy())
                if len(self.losses) > 1:
                    logs[loss_fn.__class__.__name__] = meters.avg
            self.logs_total_loss_eval.update(loss_total.item())
            logs["TotalLoss"] = self.logs_total_loss_eval.avg
            return loss_total, x_net, logs

        result = super().compute_loss(physics, x, y, train=True, epoch=epoch, step=step)

        # Save one mid-slice figure every log_every_n_epochs (first batch of that epoch only)
        images_dir = getattr(self, "_images_dir", None)
        if images_dir is not None:
            n = getattr(self, "_log_every_n_epochs", 100)
            if epoch != getattr(self, "_train_slice_epoch", None):
                self._train_slice_epoch = epoch
                self._train_batch_counter = 0
            if self._train_batch_counter == 0 and epoch % n == 0:
                self._save_patch_slices(
                    images_dir, epoch, 0,
                    x, y,
                    self._last_train_ynet.detach(),
                    self._last_train_xnet.detach(),
                    prefix="Train ",
                )
            self._train_batch_counter = getattr(self, "_train_batch_counter", 0) + 1

        return result


# ---------------------------------------------------------------------------
# Training entry-point
# ---------------------------------------------------------------------------

def run_training(cfg: RunEIPatchConfig) -> None:
    seed_everything(int(cfg.seed))

    output_dir = ensure_dir(cfg.output_dir)
    dump_config_json(output_dir / "config.json", asdict(cfg))

    data_cfg = EIPatchDataConfig(
        input_dir=cfg.input_dir,
        crop_size=int(cfg.crop_size),
        n_crops_per_vol=int(cfg.n_crops_per_vol),
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=int(cfg.prefetch_factor),
        persistent_workers=bool(cfg.persistent_workers),
        max_train_vols=cfg.max_train_vols,
        max_val_vols=int(cfg.max_val_vols),
        seed=int(cfg.seed),
        normalize=bool(cfg.normalize),
        fallback_tilt_min=cfg.tilt_min,
        fallback_tilt_max=cfg.tilt_max,
    )

    with DistributedContext(seed=int(cfg.seed), seed_offset=False, cleanup=True) as ctx:
        rank = int(ctx.rank)

        data_bundle = build_ei_patch_dataloaders(data_cfg, rank=rank, world_size=ctx.world_size)

        if rank == 0:
            train_ds = data_bundle.train_loader.dataset
            val_ds   = data_bundle.val_loader.dataset
            print("[ei-patch] Train volumes:")
            for p in train_ds.evn_paths:
                print(f"  {p.parent.name} / {p.name}")
            print("[ei-patch] Val volumes:")
            for p in val_ds.evn_paths:
                print(f"  {p.parent.name} / {p.name}")

        # ── Physics ──────────────────────────────────────────────────────────
        physics = MissingWedge(
            tilt_max=float(cfg.tilt_max),
            tilt_min=float(cfg.tilt_min),
            crop_size=int(cfg.crop_size),
            use_spherical_support=bool(cfg.use_spherical_support),
            wedge_double_size=bool(cfg.wedge_double_size),
            wedge_low_support=float(cfg.wedge_low_support),
            ref_wedge_support=float(cfg.ref_wedge_support),
            device=str(ctx.device),
        ).to(ctx.device)

        # ── Transform ────────────────────────────────────────────────────────
        transform = Rotate3D(n_trans=1)

        # ── Model ────────────────────────────────────────────────────────────
        # No distribute wrapper — patches are small (72³), single forward pass fits in VRAM.
        model, model_info = build_ei_model(
            cfg.model_type, cfg.unet_f_maps, cfg.unet_num_levels, cfg.unet_dropout,
            cfg.drunet_nb, cfg.drunet_sigma, ctx.device,
        )

        if ctx.world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ctx.local_rank])
            if rank == 0:
                print(f"[ei-patch] DDP enabled: {ctx.world_size} GPUs  effective_batch={cfg.batch_size * ctx.world_size}", flush=True)

        if rank == 0:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[ei-patch] model={model_info}  params={n_params:,}", flush=True)
            print(
                f"[ei-patch] crop_size={cfg.crop_size}  batch_size={cfg.batch_size}  "
                f"wedge_double_size={cfg.wedge_double_size}  eq_weight={cfg.eq_weight}",
                flush=True,
            )

        # ── Losses — icecream-style ───────────────────────────────────────────
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

        trainer = EIPatchTrainer(
            model=model,
            physics=physics,
            optimizer=optimizer,
            train_dataloader=data_bundle.train_loader,
            eval_dataloader=None if cfg.max_val_vols == 0 else data_bundle.val_loader,
            epochs=int(cfg.num_epochs),
            losses=losses,
            metrics=[],                       # no paired GT → no PSNR
            online_measurements=False,        # (evn, odd, tilt_params) from DataLoader
            device=ctx.device,
            save_path=None,
            ckp_interval=int(cfg.ckp_interval),
            eval_interval=int(cfg.eval_interval),
            grad_clip=cfg.grad_clip,
            check_grad=cfg.grad_clip is not None,
            plot_images=False,
            verbose=False,
            show_progress_bar=False,
            log_train_batch=False,
            optimizer_step_multi_dataset=False,
            freq_update_progress_bar=100,
        )
        trainer._is_rank0 = (rank == 0)
        trainer._log_timing = False
        trainer._log_every_n_epochs = int(cfg.log_every_n_epochs)
        trainer._train_sampler = data_bundle.train_sampler

        if cfg.use_mixed_precision:
            trainer._enable_mixed_precision()
            if rank == 0:
                print("[ei-patch] mixed precision enabled", flush=True)

        images_dir = ensure_dir(output_dir / "train_images") if rank == 0 else None
        trainer._metrics_dir      = ensure_dir(output_dir / "metrics")
        trainer._images_dir       = images_dir
        trainer._ckpt_dir         = ensure_dir(output_dir / "checkpoints") if rank == 0 else None
        trainer._scheduler        = scheduler
        trainer._grad_accum_steps = max(1, int(cfg.grad_accumulation_steps))
        trainer.ckp_interval      = int(cfg.ckp_interval)

        trainer.train()

        # ── Final checkpoint ──────────────────────────────────────────────────
        raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        if rank == 0 and trainer._ckpt_dir is not None:
            ckpt_path = Path(trainer._ckpt_dir) / "ckp_final.pth"
            torch.save(
                {
                    "epoch": cfg.num_epochs,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"[ckpt] saved final {ckpt_path}", flush=True)

            plot_metrics(output_dir, save=output_dir / "metrics" / "summary.png")

        # ── Inference on volumes — each rank handles its own subset ──────────
        recon_dir = ensure_dir(output_dir / "reconstructions")
        train_ds = data_bundle.train_loader.dataset
        val_ds   = data_bundle.val_loader.dataset
        infer_datasets = []
        if cfg.infer_train and len(train_ds.evn_vols) > 0:
            infer_datasets.append(("train", train_ds))
        if cfg.infer_val and len(val_ds.evn_vols) > 0:
            infer_datasets.append(("val", val_ds))
        stride = max(1, int(cfg.infer_stride))
        infer_bs = int(cfg.infer_batch_size) if cfg.infer_batch_size > 0 else int(cfg.batch_size)
        infer_ds = max(1, int(cfg.infer_downsample))
        # wedge_input: matches icecream's get_real_binary_filter(wedge_full[:-1,:-1,:-1])
        wedge_cpu = _symmetrize_and_binarize(physics.mask[:-1, :-1, :-1]).cpu()

        if rank == 0:
            n_total = sum(len(ds.evn_vols) for _, ds in infer_datasets)
            print(f"\n[ei-patch] Running inference on {n_total} volume(s) (distributed across {ctx.world_size} GPU(s)) ...", flush=True)
        infer_images_dir = ensure_dir(output_dir / "inference_images")
        _gpu_fsc = GpuFSC(cfg.crop_size, device=ctx.device)
        raw_model.eval()
        for split_label, ds in infer_datasets:
            if rank == 0:
                print(f"[ei-patch] Reconstructing {split_label} volumes ({len(ds.evn_vols)}) ...", flush=True)
            # Each rank processes volumes at indices rank, rank+world_size, rank+2*world_size, ...
            for i in range(rank, len(ds.evn_vols), ctx.world_size):
                tilt = ds._tilt_ranges[i]
                if tilt is None:
                    tilt = (cfg.tilt_min, cfg.tilt_max)
                tilt_min_i, tilt_max_i = tilt

                # Rebuild physics for this volume's tilt angles if different
                if tilt_min_i != cfg.tilt_min or tilt_max_i != cfg.tilt_max:
                    physics_i = MissingWedge(
                        tilt_max=float(tilt_max_i), tilt_min=float(tilt_min_i),
                        crop_size=int(cfg.crop_size),
                        use_spherical_support=bool(cfg.use_spherical_support),
                        wedge_double_size=bool(cfg.wedge_double_size),
                        wedge_low_support=float(cfg.wedge_low_support),
                        ref_wedge_support=float(cfg.ref_wedge_support),
                        device="cpu",
                    )
                    wedge_i = _symmetrize_and_binarize(physics_i.mask[:-1, :-1, :-1]).cpu()
                else:
                    wedge_i = wedge_cpu

                tomo_name = ds.evn_paths[i].parent.name
                # Direct load (mrcfile.open) — avoids _open_mrc_mmap lru_cache which
                # keeps mmap pages resident in RSS across volumes (each 2GB accumulates).
                evn_vol = _load_vol_normalized(ds.evn_paths[i], ds.normalize)
                odd_vol = _load_vol_normalized(ds.odd_paths[i], ds.normalize) if ds.odd_paths[i] is not None else evn_vol

                if infer_ds > 1:
                    evn_vol = torch.nn.functional.avg_pool3d(
                        evn_vol.unsqueeze(0).unsqueeze(0).float(),
                        kernel_size=infer_ds, stride=infer_ds,
                    ).squeeze()
                    odd_vol = torch.nn.functional.avg_pool3d(
                        odd_vol.unsqueeze(0).unsqueeze(0).float(),
                        kernel_size=infer_ds, stride=infer_ds,
                    ).squeeze()
                    print(f"  downsampled ×{infer_ds} → {tuple(evn_vol.shape)}", flush=True)
                t0 = time.perf_counter()
                print(f"  [{tomo_name}] EVN inference ...", flush=True)
                recon_evn = patch_inference(
                    evn_vol, raw_model, wedge_i,
                    crop_size=int(cfg.crop_size), stride=stride,
                    infer_batch_size=infer_bs,
                    device=ctx.device, pre_pad=True,
                )
                t_evn = time.perf_counter() - t0

                t1 = time.perf_counter()
                print(f"  [{tomo_name}] ODD inference ...", flush=True)
                recon_odd = patch_inference(
                    odd_vol, raw_model, wedge_i,
                    crop_size=int(cfg.crop_size), stride=stride,
                    infer_batch_size=infer_bs,
                    device=ctx.device, pre_pad=True,
                )
                t_odd = time.perf_counter() - t1
                print(
                    f"  [{tomo_name}] done  EVN={t_evn:.1f}s  ODD={t_odd:.1f}s  "
                    f"total={t_evn+t_odd:.1f}s",
                    flush=True,
                )

                recon = 0.5 * (recon_evn + recon_odd)

                # Gold-standard FSC between the two half-reconstructions
                recon_evn_t = torch.from_numpy(recon_evn).to(ctx.device)
                recon_odd_t = torch.from_numpy(recon_odd).to(ctx.device)
                fsc_curve_i = _gpu_fsc(recon_evn_t, recon_odd_t)
                del recon_evn_t, recon_odd_t, recon_evn, recon_odd
                torch.cuda.empty_cache()

                px_i  = _read_pixel_sizes([ds.evn_paths[i]], cfg.pixel_size_angstrom)[0]
                D_i   = int(recon.shape[-1])
                k_i   = fsc_shell(fsc_curve_i, threshold=cfg.fsc_threshold)
                res_i = D_i * px_i / max(k_i, 1)
                fsc_str = f"FSC@{cfg.fsc_threshold}={res_i:.1f} Å (shell {k_i})"
                print(f"  [{tomo_name}] {fsc_str}", flush=True)

                save_fsc_figure(
                    infer_images_dir, epoch=0,
                    fname=f"{split_label}_{tomo_name}_fsc.png",
                    fsc_curve=fsc_curve_i, res_shell=k_i, res_angstrom=res_i,
                    title=f"{tomo_name} ({split_label}) | {fsc_str}",
                    threshold=cfg.fsc_threshold, vol_size=D_i,
                    pixel_size=px_i if cfg.pixel_size_angstrom else None,
                )

                # Save MRC before figure so recon can be freed early.
                if cfg.save_mrc:
                    evn_stem = ds.evn_paths[i].stem
                    odd_stem = ds.odd_paths[i].stem if ds.odd_paths[i] is not None else evn_stem
                    out_path = recon_dir / f"{evn_stem}_{odd_stem}_recon.mrc"
                    recon_zyx = np.moveaxis(recon.astype(np.float32), 2, 0)
                    with mrcfile.new(str(out_path), overwrite=True) as mrc:
                        mrc.set_data(recon_zyx)
                    del recon_zyx
                    print(f"  saved {out_path.name}", flush=True)

                # Slice comparison figure.
                # Crop FIRST then znorm — avoids creating a full 530 MB znorm
                # intermediate for each volume (_znorm returns a full copy).
                # Each crop is ~64 MB; originals are freed immediately after.
                tomo_dir = ds.evn_paths[i].parent
                icecream_path = _find_mrc(tomo_dir, "vol_*[Ii]cecream*", "vol_*[Ii]ce[Cc]ream*")
                isonet_path   = _find_mrc(tomo_dir, "vol_*[Ii]so[Nn]et*", "vol_*DDW*")

                recon_crop = _znorm(_center_crop(recon))
                del recon
                evn_crop = _znorm(_center_crop(evn_vol.numpy()))
                del evn_vol
                odd_crop = _znorm(_center_crop(odd_vol.numpy()))
                del odd_vol

                icecream_np = _load_comparison(icecream_path)
                isonet_np   = _load_comparison(isonet_path)

                cols   = [evn_crop, odd_crop, recon_crop]
                labels = ["EVN", "ODD", "ours"]
                if icecream_np is not None:
                    cols.append(_znorm(_center_crop(icecream_np)))
                    labels.append("IceCream")
                del icecream_np
                if isonet_np is not None:
                    cols.append(_znorm(_center_crop(isonet_np)))
                    labels.append("IsoNet")
                del isonet_np

                save_slice_figure(
                    infer_images_dir, epoch=0, vol_idx=i,
                    cols=cols, labels=labels,
                    title=f"{tomo_name} ({split_label}) | {fsc_str}",
                    subdir=".", fname=f"{split_label}_{tomo_name}_recon.png",
                )
                del cols

        if rank == 0:
            print(f"[ei-patch] Inference images saved to {infer_images_dir}", flush=True)
