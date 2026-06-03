"""run_ei_inference.py — Inference-only evaluation for a trained EI full-volume model.

Loads a pretrained checkpoint and runs on the validation split (same seed as
training).  Per volume saves:
  vol{i}_halves.png   — EVN | ODD | f(EVN) | f(ODD)
  vol{i}_methods.png  — EVN | ODD | IsoNet | IceCream | ours
  vol{i}_fsc.png      — FSC curve
  vol{i}_stats.csv    — min/max/q01/q99 per (image, plane)
  vol{i}_recon.mrc    — reconstructed volume  (save_recon_mrc=True, off by default)

Summary: resolution_histogram.png + results.json

Invoked via main.py  (local or SLURM):
    python main.py --config configs/conf_ei_inference.yml
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import deepinv as dinv
import mrcfile
import numpy as np
import torch
from deepinv.distributed import DistributedContext, distribute

from equivariant.dataset_full import CryoEIFullDataset, EIFullDataConfig, _make_full_loader
from equivariant.physics import MissingWedge
from equivariant.utils import (
    GpuFSC,
    _discover_pairs,
    _split_pairs,
    fsc_shell,
    save_fsc_figure,
    save_resolution_histogram,
    save_slice_figure,
    save_slice_stats_csv,
)
from toolscryo.utils import dump_config_json, ensure_dir, seed_everything


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RunEIInferenceConfig:
    # ── Checkpoint ──────────────────────────────────────────────────────────
    checkpoint_path: str = ""           # required: path to the .pth checkpoint

    # ── Data ────────────────────────────────────────────────────────────────
    output_dir: str = "./runs/inference"
    input_dir: str = "./dataset/empiar-11058"
    max_infer_vols: int = 5             # number of volumes to evaluate
    seed: int = 0                       # must match training seed to get the same val split
    target_shape: tuple[int, int, int] | None = None

    # ── DataLoader ──────────────────────────────────────────────────────────
    num_workers: int = 1
    pin_memory: bool = True
    prefetch_factor: int = 1
    persistent_workers: bool = True

    # ── Physics (must match training config) ────────────────────────────────
    tilt_max: float = 60.0
    tilt_min: float = -60.0
    use_spherical_support: bool = True
    wedge_double_size: bool = False
    wedge_low_support: float = 0.0
    ref_wedge_support: float = 1.0

    # ── Model / tiling (must match training config) ─────────────────────────
    patch_size: tuple[int, int, int] = (64, 64, 64)
    overlap: tuple[int, int, int] = (8, 8, 8)
    max_batch_size: int | None = 2
    checkpoint_batches: str | int | None = "auto"

    # ── FSC ─────────────────────────────────────────────────────────────────
    fsc_threshold: float = 0.143
    pixel_size_angstrom: float | None = None    # fallback when MRC header voxel_size == 0

    # ── Comparison volume globs (searched inside each tomo_* directory) ─────
    icecream_glob: str = "vol_*[Ii]cecream*"
    isonet_glob: str = "vol_*[Ii]so[Nn]et*"
    isonet_fallback_glob: str = "vol_*DDW*"

    # ── Output options ───────────────────────────────────────────────────────
    save_recon_mrc: bool = False        # save reconstructed volume as .mrc (default off)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_mrc(tomo_dir: Path, *globs: str) -> Path | None:
    """Return the first file matching any glob in *tomo_dir*, or None."""
    for glob in globs:
        matches = sorted(tomo_dir.glob(glob))
        if matches:
            return matches[0]
    return None


def _load_mrc_vol(path: Path, target_shape: tuple | None = None) -> np.ndarray:
    """Load MRC, reorder axes, optional resample, centre-crop to cube, normalise.

    Mirrors ``CryoEIFullDataset._load_and_prepare`` but returns a float32
    numpy array of shape (D, H, W) instead of a tensor.
    """
    with mrcfile.open(str(path), permissive=True, mode="r") as f:
        vol_np = np.array(f.data, dtype=np.float32)   # (Z, Y, X)

    vol = torch.from_numpy(np.moveaxis(vol_np, 0, 2))  # (D, H, W)

    if target_shape is not None:
        vol = torch.nn.functional.interpolate(
            vol.unsqueeze(0).unsqueeze(0),
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    D, H, W = vol.shape
    S = min(D, H, W)
    d0, h0, w0 = (D - S) // 2, (H - S) // 2, (W - S) // 2
    vol = vol[d0:d0 + S, h0:h0 + S, w0:w0 + S]

    mu, sigma = vol.mean(), vol.std()
    vol = (vol - mu) / (sigma + 1e-8)
    return vol.numpy()   # (D, H, W) float32


def _save_mrc(path: Path, vol_dhw: np.ndarray) -> None:
    """Save a (D, H, W) float32 numpy array as an MRC file (axis order: Z, Y, X)."""
    # Reverse of moveaxis(0→2): move axis 2 back to position 0 → (Z, Y, X)
    vol_zyx = np.moveaxis(vol_dhw.astype(np.float32), 2, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(vol_zyx)


def _read_pixel_sizes(
    evn_paths: list[Path],
    fallback: float | None = None,
) -> list[float]:
    """Read voxel_size.x from each EVN MRC header (header-only, no data loaded)."""
    sizes = []
    for p in evn_paths:
        with mrcfile.open(str(p), permissive=True, mode="r") as mrc:
            px = float(mrc.voxel_size.x)
        if px <= 0.0:
            px = fallback if fallback is not None else 1.0
        sizes.append(px)
    return sizes


def _resolve_vol_size_from_paths(evn_paths: list[Path]) -> int:
    """Read the first MRC header to determine the cubic volume side length."""
    with mrcfile.open(str(evn_paths[0]), permissive=True, mode="r") as mrc:
        nx, ny, nz = int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz)
    vol_size = min(nx, ny, nz)
    print(f"[inference] auto vol_size={vol_size}  (from {evn_paths[0].name})", flush=True)
    return vol_size


# ---------------------------------------------------------------------------
# Inference entry-point
# ---------------------------------------------------------------------------

def run_inference(cfg: RunEIInferenceConfig) -> None:
    seed_everything(int(cfg.seed))

    output_dir = ensure_dir(cfg.output_dir)
    images_dir = ensure_dir(output_dir / "inference_images")
    dump_config_json(output_dir / "config.json", asdict(cfg))

    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must be set in the config.")

    # ── Build val DataLoader (same split as training when seed matches) ──────
    data_cfg = EIFullDataConfig(
        input_dir=cfg.input_dir,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=int(cfg.prefetch_factor),
        persistent_workers=bool(cfg.persistent_workers),
        max_val_vols=int(cfg.max_infer_vols),
        max_train_vols=0,    # no training split needed
        seed=int(cfg.seed),
        target_shape=cfg.target_shape,
    )

    input_dir = Path(cfg.input_dir)
    all_evn, all_odd = _discover_pairs(input_dir, data_cfg.evn_glob, data_cfg.odd_glob)
    _, _, val_evn, val_odd = _split_pairs(
        all_evn, all_odd, cfg.max_infer_vols, cfg.seed, max_train=0
    )

    if not val_evn:
        raise RuntimeError(f"No volumes found in {input_dir} — check input_dir and globs.")

    val_ds = CryoEIFullDataset(val_evn, val_odd, target_shape=cfg.target_shape)
    val_loader = _make_full_loader(val_ds, shuffle=False, cfg=data_cfg)

    pixel_sizes = _read_pixel_sizes(val_evn, fallback=cfg.pixel_size_angstrom)

    with DistributedContext(seed=int(cfg.seed), seed_offset=False, cleanup=True) as ctx:
        rank = int(ctx.rank)

        # ── Physics ──────────────────────────────────────────────────────────
        if cfg.target_shape is not None:
            vol_size = int(min(cfg.target_shape))
        else:
            vol_size = _resolve_vol_size_from_paths(val_evn)

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

        # ── Model ─────────────────────────────────────────────────────────────
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

        # ── Load checkpoint ───────────────────────────────────────────────────
        ckpt_path = Path(cfg.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location=ctx.device)
        state = ckpt.get("model_state_dict", ckpt)
        backbone.load_state_dict(state)
        backbone.eval()

        if rank == 0:
            n_params = sum(p.numel() for p in backbone.parameters())
            print(
                f"[inference] loaded checkpoint {ckpt_path.name}  "
                f"params={n_params:,}  vol_size={vol_size}",
                flush=True,
            )

        # ── Inference loop ────────────────────────────────────────────────────
        resolutions: list[float] = []
        gpu_fsc: GpuFSC | None = None
        results: list[dict] = []

        for vol_idx, (evn, odd) in enumerate(val_loader):
            evn = evn.to(ctx.device)   # (1, 1, D, H, W)
            odd = odd.to(ctx.device)

            with torch.no_grad():
                f_evn_t = backbone(evn)
                f_odd_t = backbone(odd)
                recon_t = 0.5 * (backbone(physics.A(f_evn_t)) + backbone(physics.A(f_odd_t)))

            if hasattr(ctx.device, "type") and ctx.device.type == "cuda":
                torch.cuda.synchronize()

            # ── FSC ───────────────────────────────────────────────────────────
            if gpu_fsc is None:
                gpu_fsc = GpuFSC(f_evn_t.shape[-1], device=f_evn_t.device)

            fsc_curve = gpu_fsc(f_evn_t, f_odd_t)
            D   = int(f_evn_t.shape[-1])
            px  = pixel_sizes[vol_idx] if vol_idx < len(pixel_sizes) else 1.0
            k   = fsc_shell(fsc_curve, cfg.fsc_threshold)
            res = D * px / max(k, 1)
            resolutions.append(res)

            tomo_name = val_ds.evn_paths[vol_idx].parent.name

            if rank == 0:
                print(
                    f"[inference] vol{vol_idx:02d} ({tomo_name})  "
                    f"FSC@{cfg.fsc_threshold}={res:.1f} Å  (shell {k})",
                    flush=True,
                )

            # ── numpy conversion ──────────────────────────────────────────────
            evn_np   = evn.squeeze().cpu().numpy()
            odd_np   = odd.squeeze().cpu().numpy()
            f_evn_np = f_evn_t.squeeze().cpu().numpy()
            f_odd_np = f_odd_t.squeeze().cpu().numpy()
            recon_np = recon_t.squeeze().cpu().numpy()

            # ── IsoNet / IceCream comparison volumes ──────────────────────────
            tomo_dir = val_ds.evn_paths[vol_idx].parent
            isonet_path   = _find_mrc(tomo_dir, cfg.isonet_glob, cfg.isonet_fallback_glob)
            icecream_path = _find_mrc(tomo_dir, cfg.icecream_glob)

            isonet_np: np.ndarray | None = None
            icecream_np: np.ndarray | None = None
            if rank == 0:
                if isonet_path is not None:
                    try:
                        isonet_np = _load_mrc_vol(isonet_path, cfg.target_shape)
                        print(f"  [isonet]   {isonet_path.name}", flush=True)
                    except Exception as exc:
                        print(f"  [isonet]   FAILED to load {isonet_path}: {exc}", flush=True)
                else:
                    print(f"  [isonet]   not found in {tomo_dir}", flush=True)

                if icecream_path is not None:
                    try:
                        icecream_np = _load_mrc_vol(icecream_path, cfg.target_shape)
                        print(f"  [icecream] {icecream_path.name}", flush=True)
                    except Exception as exc:
                        print(f"  [icecream] FAILED to load {icecream_path}: {exc}", flush=True)
                else:
                    print(f"  [icecream] not found in {tomo_dir}", flush=True)

            if rank == 0:
                subdir = "."

                # ── Figure 1: halves — EVN | ODD | f(EVN) | f(ODD) ───────────
                save_slice_figure(
                    images_dir, epoch=0, vol_idx=vol_idx,
                    cols=[evn_np, odd_np, f_evn_np, f_odd_np],
                    labels=["EVN", "ODD", "f(EVN)", "f(ODD)"],
                    title=f"{tomo_name} | f(EVN) vs f(ODD)",
                    subdir=subdir,
                    fname=f"vol{vol_idx:02d}_halves.png",
                )

                # ── Figure 2: methods — EVN | ODD | IsoNet | IceCream | ours ──
                methods_cols   = [evn_np, odd_np, isonet_np, icecream_np, recon_np]
                methods_labels = ["EVN",  "ODD",  "IsoNet",  "IceCream",  "ours"]
                # Filter out None columns
                valid_pairs = [(v, lbl) for v, lbl in zip(methods_cols, methods_labels) if v is not None]
                valid_cols, valid_labels = zip(*valid_pairs) if valid_pairs else ([], [])
                save_slice_figure(
                    images_dir, epoch=0, vol_idx=vol_idx,
                    cols=list(valid_cols),
                    labels=list(valid_labels),
                    title=f"{tomo_name} | method comparison",
                    subdir=subdir,
                    fname=f"vol{vol_idx:02d}_methods.png",
                )

                # ── FSC figure ────────────────────────────────────────────────
                save_fsc_figure(
                    images_dir, epoch=0,
                    fname=f"vol{vol_idx:02d}_fsc.png",
                    fsc_curve=fsc_curve, res_shell=k, res_angstrom=res,
                    title=f"{tomo_name} | FSC  {res:.1f} Å",
                    threshold=cfg.fsc_threshold,
                    vol_size=D, pixel_size=px,
                )

                # ── CSV stats — all volumes (for both figures) ────────────────
                all_labels = ["EVN", "ODD", "f(EVN)", "f(ODD)", "ours", "IsoNet", "IceCream"]
                all_vols   = [evn_np, odd_np, f_evn_np, f_odd_np, recon_np, isonet_np, icecream_np]
                save_slice_stats_csv(
                    images_dir / f"vol{vol_idx:02d}_stats.csv",
                    labels=all_labels,
                    vols=all_vols,
                )

                # ── Optional: save recon MRC ──────────────────────────────────
                if cfg.save_recon_mrc:
                    recon_mrc_path = images_dir / f"vol{vol_idx:02d}_recon.mrc"
                    _save_mrc(recon_mrc_path, recon_np)
                    print(f"  [recon mrc] saved {recon_mrc_path.name}", flush=True)

            results.append({
                "vol_idx":    vol_idx,
                "tomo":       tomo_name,
                "fsc_shell":  int(k),
                "fsc_res_angstrom": float(res),
                "pixel_size": float(px),
            })

        # ── Summary ───────────────────────────────────────────────────────────
        if rank == 0 and resolutions:
            res_arr    = np.array(resolutions)
            mean_res   = float(np.mean(res_arr))
            median_res = float(np.median(res_arr))
            q1_res     = float(np.percentile(res_arr, 25))
            q3_res     = float(np.percentile(res_arr, 75))

            save_resolution_histogram(
                images_dir, epoch=0,
                resolutions_angstrom=resolutions,
                mean_res=mean_res, median_res=median_res,
                q1_res=q1_res, q3_res=q3_res,
                threshold_label=str(cfg.fsc_threshold),
            )

            summary = {
                "fsc_threshold":   cfg.fsc_threshold,
                "n_vols":          len(resolutions),
                "mean_res_angstrom":   mean_res,
                "median_res_angstrom": median_res,
                "q1_res_angstrom":     q1_res,
                "q3_res_angstrom":     q3_res,
                "per_volume":      results,
            }
            with open(output_dir / "results.json", "w") as f:
                json.dump(summary, f, indent=2)

            print(
                f"\n[inference] DONE  n={len(resolutions)}  "
                f"mean={mean_res:.1f} Å  median={median_res:.1f} Å  "
                f"Q1={q1_res:.1f} Å  Q3={q3_res:.1f} Å  (lower=better)",
                flush=True,
            )



