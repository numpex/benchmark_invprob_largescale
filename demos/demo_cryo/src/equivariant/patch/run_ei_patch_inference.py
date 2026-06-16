"""run_ei_patch_inference.py — Inference for a trained patch-based EI model.

Matches icecream's inference_util.inference exactly:
  1. Pre-pad the volume.
  2. Slide a crop_size³ window with stride, extracting overlapping patches.
  3. For each patch:  output = f(A(f(patch)))  — model applied twice, wedge in between.
  4. Reassemble via window-weighted overlap averaging.
  5. Average EVN and ODD reconstructions:  recon = 0.5 * (result_evn + result_odd).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mrcfile
import numpy as np
import torch
import torch.nn as nn

from equivariant.physics import MissingWedge
from equivariant.losses import _initialize_window, _symmetrize_and_binarize
from equivariant.utils import (
    GpuFSC,
    _discover_pairs,
    _find_mrc,
    _read_pixel_sizes,
    _resolve_tlt_ranges,
    _save_mrc,
    _split_pairs,
    fsc_shell,
    save_fsc_figure,
    save_resolution_histogram,
    save_slice_figure,
    save_slice_stats_csv,
)
from toolscryo.utils import build_ei_model, dump_config_json, ensure_dir, seed_everything


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RunEIPatchInferenceConfig:
    # ── Checkpoint ──────────────────────────────────────────────────────────
    checkpoint_path: str = ""

    # ── Data ────────────────────────────────────────────────────────────────
    output_dir: str = "./runs/patch_inference"
    input_dir: str = "./dataset/empiar-11058"
    max_infer_vols: int = 5
    seed: int = 0
    normalize: bool = True           # must match training config

    # ── Physics (must match training config) ────────────────────────────────
    tilt_max: float = 60.0
    tilt_min: float = -60.0
    use_spherical_support: bool = True
    wedge_double_size: bool = True   # patch default
    wedge_low_support: float = 0.0
    ref_wedge_support: float = 1.0

    # ── Model (must match training config) ──────────────────────────────────
    crop_size: int = 72              # patch size used during training
    model_type: str = "unet"
    unet_f_maps: int = 64          # unet only
    unet_num_levels: int = 4       # unet only
    unet_dropout: float = 0.1      # unet only
    drunet_nb: int = 4             # drunet only
    drunet_sigma: float = 0.0      # drunet only

    # ── Sliding-window inference ─────────────────────────────────────────────
    stride: int = 36                 # icecream default: crop_size // 2
    infer_batch_size: int = 4        # crops processed in parallel on GPU
    pre_pad: bool = True             # pad volume before sliding window

    # ── FSC ─────────────────────────────────────────────────────────────────
    fsc_threshold: float = 0.143
    pixel_size_angstrom: float | None = None

    # ── Comparison globs ─────────────────────────────────────────────────────
    icecream_glob: str = "vol_*[Ii]cecream*"
    isonet_glob: str = "vol_*[Ii]so[Nn]et*"
    isonet_fallback_glob: str = "vol_*DDW*"

    # ── Output ───────────────────────────────────────────────────────────────
    save_recon_mrc: bool = False


# ---------------------------------------------------------------------------
# Core sliding-window inference  (mirrors icecream's inference_util.inference)
# ---------------------------------------------------------------------------

def _compute_padd(N: int, filt_size: int, stride: int) -> int:
    w = (N - filt_size) // stride + 1
    N_rec = (w - 1) * stride + filt_size
    return (N_rec - N) % filt_size


def patch_inference(
    vol: torch.Tensor,
    model: nn.Module,
    wedge: torch.Tensor,
    crop_size: int,
    stride: int,
    infer_batch_size: int,
    device: torch.device,
    pre_pad: bool = True,
) -> np.ndarray:
    """Sliding-window f(A(f(.))) inference — mirrors icecream's inference_util.inference exactly.

    :param vol: (D, H, W) CPU float32 tensor (globally normalised).
    :param wedge: wedge_input mask (mask_size³) on CPU.
    :returns: (D, H, W) float32 numpy array.
    """
    pre_pad_size = crop_size // 4

    vol_fbp = vol.to(torch.float16)
    if pre_pad:
        vol_fbp = torch.nn.functional.pad(vol_fbp, (pre_pad_size, 0, pre_pad_size, 0, pre_pad_size, 0))
    wedge_dev = wedge.to(device)

    pad_i = _compute_padd(vol_fbp.shape[0], crop_size, stride)
    pad_j = _compute_padd(vol_fbp.shape[1], crop_size, stride)
    pad_k = _compute_padd(vol_fbp.shape[2], crop_size, stride)
    vol_fbp_pad = torch.nn.functional.pad(vol_fbp, (0, pad_k, 0, pad_j, 0, pad_i))
    del vol_fbp

    N1_orig, N2_orig, N3_orig = vol.shape
    vol_est = torch.zeros((N1_orig, N2_orig, N3_orig))
    if pre_pad:
        vol_est = torch.nn.functional.pad(vol_est, (pre_pad_size, 0, pre_pad_size, 0, pre_pad_size, 0))
    N1_pad, N2_pad, N3_pad = vol_est.shape
    pad_i = _compute_padd(vol_est.shape[0], crop_size, stride)
    pad_j = _compute_padd(vol_est.shape[1], crop_size, stride)
    pad_k = _compute_padd(vol_est.shape[2], crop_size, stride)
    vol_est = torch.nn.functional.pad(vol_est, (0, pad_k, 0, pad_j, 0, pad_i))
    N1, N2, N3 = vol_est.shape
    mask = torch.zeros_like(vol_est)
    window = _initialize_window(crop_size).cpu()

    positions = [
        (i, j, k)
        for i in range(0, N1, stride)
        for j in range(0, N2, stride)
        for k in range(0, N3, stride)
        if i + crop_size <= N1 and j + crop_size <= N2 and k + crop_size <= N3
    ]

    use_amp = device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(positions), infer_batch_size):
            batch_positions = positions[batch_start:batch_start + infer_batch_size]
            batch = torch.stack([
                vol_fbp_pad[i:i + crop_size, j:j + crop_size, k:k + crop_size]
                for i, j, k in batch_positions
            ]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(batch[:, None])[:, 0]        # f(crop)
            output = output.float()
            output = _apply_wedge_batch(output, wedge_dev)  # A(f(crop))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(output[:, None])[:, 0]       # f(A(f(crop)))
            out_cpu = output.detach().cpu()
            for b, (i, j, k) in enumerate(batch_positions):
                vol_est[i:i + crop_size, j:j + crop_size, k:k + crop_size] += out_cpu[b] * window
                mask[  i:i + crop_size, j:j + crop_size, k:k + crop_size] += window

    del vol_fbp_pad, wedge_dev
    torch.cuda.empty_cache()

    mask[mask == 0] = 1
    vol_est = vol_est / mask
    del mask
    vol_est = vol_est[:N1_pad, :N2_pad, :N3_pad]
    vol_est_np = vol_est.numpy().copy()
    del vol_est

    if pre_pad:
        vol_est_np = vol_est_np[pre_pad_size:, pre_pad_size:, pre_pad_size:]

    return vol_est_np


def _apply_wedge_batch(x: torch.Tensor, wedge: torch.Tensor) -> torch.Tensor:
    """Apply wedge mask via FFT  (B, D, H, W) → (B, D, H, W)."""
    B, D, H, W = x.shape
    mask_shape = tuple(wedge.shape)
    X = torch.fft.fftshift(torch.fft.fftn(x, s=mask_shape, dim=(-3, -2, -1)), dim=(-3, -2, -1))
    X = X * wedge
    out = torch.fft.ifftn(torch.fft.ifftshift(X, dim=(-3, -2, -1)), dim=(-3, -2, -1)).real
    return out[..., :D, :H, :W]


# ---------------------------------------------------------------------------
# Volume I/O helpers
# ---------------------------------------------------------------------------

def _load_vol_normalized(path: Path, normalize: bool) -> torch.Tensor:
    """Load MRC → (D, H, W) CPU tensor with optional global normalization."""
    with mrcfile.open(str(path), permissive=True, mode="r") as mrc:
        vol_np = np.array(mrc.data, dtype=np.float32)
    vol_t = torch.from_numpy(np.moveaxis(vol_np, 0, 2))  # (Z,Y,X) → (D,H,W)
    if normalize:
        vol_t = (vol_t - vol_t.mean()) / (vol_t.std() + 1e-8)
    return vol_t


def _load_comparison(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    try:
        with mrcfile.open(str(path), permissive=True, mode="r") as mrc:
            vol_np = np.array(mrc.data, dtype=np.float32)
        vol_t = torch.from_numpy(np.moveaxis(vol_np, 0, 2))
        vol_t = (vol_t - vol_t.mean()) / (vol_t.std() + 1e-8)
        return vol_t.numpy()
    except Exception as exc:
        print(f"  WARNING: could not load {path}: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Per-volume inference  (function scope = automatic memory cleanup on return)
# ---------------------------------------------------------------------------

def _infer_one_volume(
    vol_idx: int,
    evn_path: Path,
    odd_path: Path | None,
    tilt_range: tuple[float, float] | None,
    model: nn.Module,
    cfg: RunEIPatchInferenceConfig,
    device: torch.device,
    stride: int,
    images_dir: Path,
) -> dict:
    tomo_dir  = evn_path.parent
    tomo_name = tomo_dir.name
    tilt_min, tilt_max = tilt_range if tilt_range is not None else (cfg.tilt_min, cfg.tilt_max)

    print(
        f"[patch-infer] vol{vol_idx:02d} ({tomo_name})  tilt=[{tilt_min:.1f}, {tilt_max:.1f}]°",
        flush=True,
    )

    physics = MissingWedge(
        tilt_max=float(tilt_max), tilt_min=float(tilt_min),
        crop_size=int(cfg.crop_size),
        use_spherical_support=bool(cfg.use_spherical_support),
        wedge_double_size=bool(cfg.wedge_double_size),
        wedge_low_support=float(cfg.wedge_low_support),
        ref_wedge_support=float(cfg.ref_wedge_support),
        device="cpu",
    )
    wedge_input = _symmetrize_and_binarize(physics.mask[:-1, :-1, :-1]).cpu()

    evn_vol = _load_vol_normalized(evn_path, cfg.normalize)
    odd_vol = _load_vol_normalized(odd_path, cfg.normalize) if odd_path is not None else evn_vol

    infer_kw = dict(
        model=model, wedge=wedge_input,
        crop_size=int(cfg.crop_size), stride=stride,
        infer_batch_size=int(cfg.infer_batch_size),
        device=device, pre_pad=bool(cfg.pre_pad),
    )
    print("  running inference on EVN ...", flush=True)
    recon_evn = patch_inference(evn_vol, **infer_kw)
    print("  running inference on ODD ...", flush=True)
    recon_odd = patch_inference(odd_vol, **infer_kw)

    recon_np = 0.5 * (recon_evn + recon_odd)

    # FSC between half-reconstructions
    evn_t = torch.from_numpy(recon_evn).unsqueeze(0).unsqueeze(0).to(device)
    odd_t = torch.from_numpy(recon_odd).unsqueeze(0).unsqueeze(0).to(device)
    fsc_curve = GpuFSC(evn_t.shape[-1], device=device)(evn_t, odd_t)
    px  = _read_pixel_sizes([evn_path], cfg.pixel_size_angstrom)[0]
    D   = int(recon_evn.shape[-1])
    k   = fsc_shell(fsc_curve, cfg.fsc_threshold)
    res = D * px / max(k, 1)
    fsc_str = f"FSC@{cfg.fsc_threshold}={res:.1f} Å (shell {k})"
    print(f"  {fsc_str}", flush=True)

    # Figures — all volume locals freed when this function returns
    evn_np      = evn_vol.numpy()
    odd_np      = odd_vol.numpy()
    isonet_np   = _load_comparison(_find_mrc(tomo_dir, cfg.isonet_glob, cfg.isonet_fallback_glob))
    icecream_np = _load_comparison(_find_mrc(tomo_dir, cfg.icecream_glob))

    save_slice_figure(
        images_dir, epoch=0, vol_idx=vol_idx,
        cols=[evn_np, odd_np, recon_evn, recon_odd],
        labels=["EVN", "ODD", "recon(EVN)", "recon(ODD)"],
        title=f"{tomo_name} | recon(EVN) vs recon(ODD) | {fsc_str}",
        subdir=".", fname=f"vol{vol_idx:02d}_halves.png",
    )

    methods = [(evn_np, "EVN"), (odd_np, "ODD"), (isonet_np, "IsoNet"),
               (icecream_np, "IceCream"), (recon_np, "ours")]
    valid = [(v, lbl) for v, lbl in methods if v is not None]
    if valid:
        vcols, vlabels = zip(*valid)
        save_slice_figure(
            images_dir, epoch=0, vol_idx=vol_idx,
            cols=list(vcols), labels=list(vlabels),
            title=f"{tomo_name} | method comparison | {fsc_str}",
            subdir=".", fname=f"vol{vol_idx:02d}_methods.png",
        )

    save_fsc_figure(
        images_dir, epoch=0, fname=f"vol{vol_idx:02d}_fsc.png",
        fsc_curve=fsc_curve, res_shell=k, res_angstrom=res,
        title=f"{tomo_name} | FSC {res:.1f} Å",
        threshold=cfg.fsc_threshold, vol_size=D, pixel_size=px,
    )

    save_slice_stats_csv(
        images_dir / f"vol{vol_idx:02d}_stats.csv",
        labels=["EVN", "ODD", "recon(EVN)", "recon(ODD)", "ours", "IsoNet", "IceCream"],
        vols=[evn_np, odd_np, recon_evn, recon_odd, recon_np, isonet_np, icecream_np],
    )

    if cfg.save_recon_mrc:
        mrc_path = images_dir / f"vol{vol_idx:02d}_recon.mrc"
        _save_mrc(mrc_path, recon_np)
        print(f"  saved {mrc_path.name}", flush=True)

    return {
        "vol_idx": vol_idx, "tomo": tomo_name,
        "fsc_shell": int(k), "fsc_res_angstrom": float(res), "pixel_size": float(px),
    }


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def run_inference(cfg: RunEIPatchInferenceConfig) -> None:
    seed_everything(int(cfg.seed))

    output_dir = ensure_dir(cfg.output_dir)
    images_dir = ensure_dir(output_dir / "inference_images")
    dump_config_json(output_dir / "config.json", asdict(cfg))

    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must be set.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[patch-infer] device={device}", flush=True)

    # Discover volumes (same split as training when seed matches)
    input_dir = Path(cfg.input_dir)
    all_evn, all_odd, all_tlt = _discover_pairs(input_dir)

    all_tilt_ranges = _resolve_tlt_ranges(all_tlt)

    _, _, val_evn, val_odd, _, val_tilt = _split_pairs(
        all_evn, all_odd, cfg.max_infer_vols, cfg.seed, max_train=0,
        extra=all_tilt_ranges,
    )

    if not val_evn:
        raise RuntimeError(f"No volumes found in {input_dir}.")

    # Model
    model, model_info = build_ei_model(
        cfg.model_type, cfg.unet_f_maps, cfg.unet_num_levels, cfg.unet_dropout,
        cfg.drunet_nb, cfg.drunet_sigma, device,
    )

    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    print(
        f"[patch-infer] loaded {Path(cfg.checkpoint_path).name}  "
        f"model={model_info}  params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    stride = int(cfg.stride) if cfg.stride > 0 else cfg.crop_size // 2

    # Main loop — one function call per volume; all memory freed on return
    results = []
    for vol_idx, (evn_path, odd_path, tilt_range) in enumerate(zip(val_evn, val_odd, val_tilt)):
        result = _infer_one_volume(vol_idx, evn_path, odd_path, tilt_range, model, cfg, device, stride, images_dir)
        results.append(result)
        torch.cuda.empty_cache()

    # Summary
    resolutions = [r["fsc_res_angstrom"] for r in results]
    if resolutions:
        res_arr = np.array(resolutions)
        mean_res, median_res = float(np.mean(res_arr)), float(np.median(res_arr))
        q1_res, q3_res = float(np.percentile(res_arr, 25)), float(np.percentile(res_arr, 75))

        save_resolution_histogram(
            images_dir, epoch=0, resolutions_angstrom=resolutions,
            mean_res=mean_res, median_res=median_res,
            q1_res=q1_res, q3_res=q3_res,
            threshold_label=str(cfg.fsc_threshold),
        )
        with open(output_dir / "results.json", "w") as f:
            json.dump({
                "fsc_threshold": cfg.fsc_threshold, "n_vols": len(resolutions),
                "mean_res_angstrom": mean_res, "median_res_angstrom": median_res,
                "q1_res_angstrom": q1_res, "q3_res_angstrom": q3_res,
                "per_volume": results,
            }, f, indent=2)

        print(
            f"\n[patch-infer] DONE  n={len(resolutions)}  "
            f"mean={mean_res:.1f} Å  median={median_res:.1f} Å  (lower=better)",
            flush=True,
        )
