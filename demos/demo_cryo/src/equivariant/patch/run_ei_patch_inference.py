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
import deepinv as dinv

from equivariant.physics import MissingWedge
from equivariant.losses import _initialize_window, _symmetrize_and_binarize
from equivariant.utils import (
    GpuFSC,
    _discover_pairs,
    _find_mrc,
    _read_pixel_sizes,
    _read_tlt,
    _save_mrc,
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
    scales: int = 4
    bias: bool = False

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


def _crop_volumes(volume: torch.Tensor, stride: int, size: int, window_inp: torch.Tensor | None = None) -> torch.Tensor:
    """Extract overlapping crops from a volume into a CPU tensor (mirrors icecream's crop_volumes)."""
    N1, N2, N3 = volume.shape
    n1 = (N1 - size) // stride + 1
    n2 = (N2 - size) // stride + 1
    n3 = (N3 - size) // stride + 1
    out = torch.empty((n1 * n2 * n3, size, size, size), dtype=volume.dtype, device="cpu")
    idx = 0
    with torch.no_grad():
        for i in range(0, N1, stride):
            for j in range(0, N2, stride):
                for k in range(0, N3, stride):
                    if i + size > N1 or j + size > N2 or k + size > N3:
                        continue
                    crop = volume[i:i + size, j:j + size, k:k + size]
                    if window_inp is not None:
                        crop = crop * window_inp
                    out[idx] = crop.cpu()
                    idx += 1
    return out[:idx]


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

    Uses float16 + autocast like icecream. batch_size and stride are configurable.

    :param vol: (D, H, W) CPU float32 tensor (globally normalised).
    :param wedge: wedge_input mask (mask_size³) on CPU.
    :returns: (D, H, W) float32 numpy array.
    """
    pre_pad_size = crop_size // 4

    # Convert to float16 and move to device — matches icecream's vol_fbp = vol_input.to(float16)
    vol_fbp = vol.to(torch.float16)
    if pre_pad:
        vol_fbp = torch.nn.functional.pad(vol_fbp, (pre_pad_size, 0, pre_pad_size, 0, pre_pad_size, 0))
    vol_fbp = vol_fbp.to(device)
    wedge_dev = wedge.to(device)

    # Stride-pad the input
    window_inp = torch.ones((crop_size, crop_size, crop_size), device=device, dtype=torch.float16)
    pad_i = _compute_padd(vol_fbp.shape[0], crop_size, stride)
    pad_j = _compute_padd(vol_fbp.shape[1], crop_size, stride)
    pad_k = _compute_padd(vol_fbp.shape[2], crop_size, stride)
    vol_fbp_pad = torch.nn.functional.pad(vol_fbp, (0, pad_k, 0, pad_j, 0, pad_i))
    del vol_fbp  # GPU float16 no longer needed after padding

    # Extract all crops to CPU float16 — matches icecream's crop_volumes
    crops = _crop_volumes(vol_fbp_pad, stride, crop_size, window_inp=window_inp)
    del vol_fbp_pad, window_inp  # GPU tensors no longer needed
    torch.cuda.empty_cache()

    # Build reassembly arrays on CPU before inference so output accumulates
    # directly — eliminates the output_crops buffer (saves ~6 GB for large vols)
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

    # Pre-compute crop positions (same traversal order as _crop_volumes)
    positions = [
        (i, j, k)
        for i in range(0, N1, stride)
        for j in range(0, N2, stride)
        for k in range(0, N3, stride)
        if i + crop_size <= N1 and j + crop_size <= N2 and k + crop_size <= N3
    ]

    use_amp = device.type == "cuda"
    model.eval()
    count = 0
    with torch.no_grad():
        for start in range(0, len(crops), infer_batch_size):
            batch = crops[start:start + infer_batch_size].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(batch[:, None])[:, 0]        # f(crop)
            output = output.float()
            output = _apply_wedge_batch(output, wedge_dev)  # A(f(crop))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(output[:, None])[:, 0]       # f(A(f(crop)))
            out_cpu = output.detach().cpu()
            B = out_cpu.shape[0]
            for b in range(B):
                i, j, k = positions[count + b]
                vol_est[i:i + crop_size, j:j + crop_size, k:k + crop_size] += out_cpu[b] * window
                mask[  i:i + crop_size, j:j + crop_size, k:k + crop_size] += window
            count += B

    del wedge_dev, crops
    torch.cuda.empty_cache()

    mask[mask == 0] = 1
    vol_est = vol_est / mask
    del mask
    vol_est = vol_est[:N1_pad, :N2_pad, :N3_pad]
    vol_est_np = vol_est.numpy().copy()  # copy so vol_est tensor is freed immediately
    del vol_est

    if pre_pad:
        vol_est_np = vol_est_np[pre_pad_size:, pre_pad_size:, pre_pad_size:]

    return vol_est_np


def _apply_wedge_batch(x: torch.Tensor, wedge: torch.Tensor) -> torch.Tensor:
    """Apply wedge mask via FFT, matching icecream's get_measurement (batch version).

    :param x: (B, D, H, W)
    :param wedge: (M, M, M)  mask_size³
    :returns: (B, D, H, W) wedge-masked and IFFT'd back to crop_size
    """
    B, D, H, W = x.shape
    mask_shape = tuple(wedge.shape)
    X = torch.fft.fftshift(
        torch.fft.fftn(x, s=mask_shape, dim=(-3, -2, -1)),
        dim=(-3, -2, -1),
    )
    X = X * wedge
    out = torch.fft.ifftn(
        torch.fft.ifftshift(X, dim=(-3, -2, -1)),
        dim=(-3, -2, -1),
    ).real
    return out[..., :D, :H, :W]


# ---------------------------------------------------------------------------
# Volume I/O helpers
# ---------------------------------------------------------------------------

def _load_vol_normalized(path: Path, normalize: bool) -> torch.Tensor:
    """Load MRC → (D, H, W) CPU tensor with optional global normalization."""
    vol_np = np.array(mrcfile.open(str(path), permissive=True).data, dtype=np.float32)
    vol_t = torch.from_numpy(np.moveaxis(vol_np, 0, 2))  # (Z,Y,X) → (D,H,W)
    if normalize:
        vol_t = (vol_t - vol_t.mean()) / (vol_t.std() + 1e-8)
    return vol_t


def _load_comparison(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    try:
        vol_np = np.array(mrcfile.open(str(path), permissive=True).data, dtype=np.float32)
        vol_t = torch.from_numpy(np.moveaxis(vol_np, 0, 2))
        vol_t = (vol_t - vol_t.mean()) / (vol_t.std() + 1e-8)
        return vol_t.numpy()
    except Exception as exc:
        print(f"  WARNING: could not load {path}: {exc}", flush=True)
        return None


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

    # ── Discover volumes (same split as training when seed matches) ──────────
    input_dir = Path(cfg.input_dir)
    all_evn, all_odd, all_tlt = _discover_pairs(input_dir)

    all_tilt_ranges: list[tuple[float, float] | None] = []
    for tlt_path in all_tlt:
        if tlt_path is not None:
            try:
                all_tilt_ranges.append(_read_tlt(tlt_path))
            except Exception:
                all_tilt_ranges.append(None)
        else:
            all_tilt_ranges.append(None)

    _, _, val_evn, val_odd, _, val_tilt = _split_pairs(
        all_evn, all_odd, cfg.max_infer_vols, cfg.seed, max_train=0,
        extra=all_tilt_ranges,
    )

    if not val_evn:
        raise RuntimeError(f"No volumes found in {input_dir}.")

    # ── Model ────────────────────────────────────────────────────────────────
    model = dinv.models.UNet(
        in_channels=1,
        out_channels=1,
        scales=int(cfg.scales),
        bias=bool(cfg.bias),
        dim=3,
    ).to(device)

    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(
        f"[patch-infer] loaded {Path(cfg.checkpoint_path).name}  "
        f"params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    stride = int(cfg.stride) if cfg.stride > 0 else cfg.crop_size // 2

    resolutions: list[float] = []
    results: list[dict] = []
    gpu_fsc: GpuFSC | None = None

    for vol_idx, (evn_path, odd_path, tilt_range) in enumerate(zip(val_evn, val_odd, val_tilt)):
        tomo_dir = evn_path.parent
        tomo_name = tomo_dir.name

        # Resolve tilt angles for this volume
        if tilt_range is not None:
            tilt_min, tilt_max = tilt_range
        else:
            tilt_min, tilt_max = cfg.tilt_min, cfg.tilt_max

        print(
            f"[patch-infer] vol{vol_idx:02d} ({tomo_name})  "
            f"tilt=[{tilt_min:.1f}, {tilt_max:.1f}]°",
            flush=True,
        )

        # Build per-volume physics (tilt may differ per tomo)
        physics = MissingWedge(
            tilt_max=float(tilt_max),
            tilt_min=float(tilt_min),
            crop_size=int(cfg.crop_size),
            use_spherical_support=bool(cfg.use_spherical_support),
            wedge_double_size=bool(cfg.wedge_double_size),
            wedge_low_support=float(cfg.wedge_low_support),
            ref_wedge_support=float(cfg.ref_wedge_support),
            device="cpu",
        )
        # wedge_input matches icecream: get_real_binary_filter(wedge_full[:-1,:-1,:-1])
        wedge_input = _symmetrize_and_binarize(physics.mask[:-1, :-1, :-1]).cpu()

        # Load and normalize volumes
        evn_vol = _load_vol_normalized(evn_path, cfg.normalize)
        odd_vol = _load_vol_normalized(odd_path, cfg.normalize) if odd_path is not None else evn_vol

        # Run f(A(f(.))) sliding-window inference on each half
        print("  running inference on EVN ...", flush=True)
        recon_evn = patch_inference(
            evn_vol, model, wedge_input,
            crop_size=int(cfg.crop_size),
            stride=stride,
            infer_batch_size=int(cfg.infer_batch_size),
            device=device,
            pre_pad=bool(cfg.pre_pad),
        )
        print("  running inference on ODD ...", flush=True)
        recon_odd = patch_inference(
            odd_vol, model, wedge_input,
            crop_size=int(cfg.crop_size),
            stride=stride,
            infer_batch_size=int(cfg.infer_batch_size),
            device=device,
            pre_pad=bool(cfg.pre_pad),
        )

        # Final reconstruction — average of both half reconstructions
        recon_np = 0.5 * (recon_evn + recon_odd)

        # FSC(recon_evn, recon_odd) — gold-standard FSC between the two half-reconstructions
        # (avoids full-volume GPU inference which OOMs on large tomograms)
        recon_evn_t = torch.from_numpy(recon_evn).unsqueeze(0).unsqueeze(0).to(device)
        recon_odd_t = torch.from_numpy(recon_odd).unsqueeze(0).unsqueeze(0).to(device)
        if gpu_fsc is None:
            gpu_fsc = GpuFSC(recon_evn_t.shape[-1], device=device)
        fsc_curve = gpu_fsc(recon_evn_t, recon_odd_t)
        del recon_evn_t, recon_odd_t
        px  = _read_pixel_sizes([evn_path], cfg.pixel_size_angstrom)[0]
        D   = int(recon_evn.shape[-1])
        k   = fsc_shell(fsc_curve, cfg.fsc_threshold)
        res = D * px / max(k, 1)
        resolutions.append(res)
        print(
            f"  FSC@{cfg.fsc_threshold} = {res:.1f} Å  (shell {k})",
            flush=True,
        )

        # Comparison volumes
        isonet_path   = _find_mrc(tomo_dir, cfg.isonet_glob, cfg.isonet_fallback_glob)
        icecream_path = _find_mrc(tomo_dir, cfg.icecream_glob)
        isonet_np   = _load_comparison(isonet_path)
        icecream_np = _load_comparison(icecream_path)

        evn_np   = evn_vol.numpy()
        odd_np   = odd_vol.numpy()
        f_evn_np = recon_evn   # f(A(f(EVN))) from sliding-window inference
        f_odd_np = recon_odd   # f(A(f(ODD))) from sliding-window inference

        # Figures
        save_slice_figure(
            images_dir, epoch=0, vol_idx=vol_idx,
            cols=[evn_np, odd_np, f_evn_np, f_odd_np],
            labels=["EVN", "ODD", "recon(EVN)", "recon(ODD)"],
            title=f"{tomo_name} | recon(EVN) vs recon(ODD)",
            subdir=".", fname=f"vol{vol_idx:02d}_halves.png",
        )

        methods_cols   = [evn_np, odd_np, isonet_np, icecream_np, recon_np]
        methods_labels = ["EVN",  "ODD",  "IsoNet",  "IceCream",  "ours"]
        valid = [(v, lbl) for v, lbl in zip(methods_cols, methods_labels) if v is not None]
        if valid:
            vcols, vlabels = zip(*valid)
            save_slice_figure(
                images_dir, epoch=0, vol_idx=vol_idx,
                cols=list(vcols), labels=list(vlabels),
                title=f"{tomo_name} | method comparison",
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
            vols=[evn_np, odd_np, f_evn_np, f_odd_np, recon_np, isonet_np, icecream_np],
        )

        if cfg.save_recon_mrc:
            mrc_path = images_dir / f"vol{vol_idx:02d}_recon.mrc"
            _save_mrc(mrc_path, recon_np)
            print(f"  saved {mrc_path.name}", flush=True)

        results.append({
            "vol_idx": vol_idx, "tomo": tomo_name,
            "fsc_shell": int(k), "fsc_res_angstrom": float(res), "pixel_size": float(px),
        })

    # Summary
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
