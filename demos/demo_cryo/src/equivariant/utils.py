"""Shared utilities for equivariant cryo-ET training.

Contains dataset discovery helpers (``EIDataBundle``, ``_discover_pairs``,
``_split_pairs``), I/O helpers (``_find_mrc``, ``_save_mrc``,
``_read_mrc_vol_size``, ``_read_pixel_sizes``), volume preprocessing
utilities (``_center_crop``, ``_znorm``), and visualisation helpers
(``save_slice_figure``, ``GpuFSC``) used by both the patch and full-volume
variants.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class EIDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_sampler: object | None = None  # DistributedSampler when DDP is active


def _read_tlt(path: Path) -> tuple[float, float]:
    """Read a tlt file and return (tilt_min, tilt_max) in degrees.

    Tlt files are plain text with one floating-point angle per line.
    """
    angles = np.loadtxt(str(path))
    return float(angles.min()), float(angles.max())


def _find_tlt_for_dir(tomo_dir: Path) -> Path | None:
    """Find the full-series tlt file for a tomo directory.

    Looks for ``angles_*.tlt`` files, preferring the full-series file
    (i.e. excluding ``*_split1.tlt`` / ``*_split2.tlt``).  Falls back to
    any tlt file if no full-series file is found.
    """
    candidates = sorted(tomo_dir.glob("angles_*.tlt"))
    full_series = [
        p for p in candidates
        if not (p.stem.endswith("_split1") or p.stem.endswith("_split2"))
    ]
    if full_series:
        return full_series[0]
    if candidates:
        return candidates[0]
    return None


def _resolve_tlt_ranges(
    tlt_paths: list[Path | None],
) -> list[tuple[float, float] | None]:
    """Read tilt ranges from tlt files, returning None for missing or unreadable ones."""
    ranges: list[tuple[float, float] | None] = []
    for tlt_path in tlt_paths:
        if tlt_path is None:
            ranges.append(None)
        else:
            try:
                ranges.append(_read_tlt(tlt_path))
            except Exception as e:
                print(f"[ei-data] WARNING: could not read {tlt_path}: {e}")
                ranges.append(None)
    return ranges


def _discover_pairs(
    input_dir: Path,
    evn_glob: str = "vol*split1*.mrc",
    odd_glob: str = "vol*split2*.mrc",
) -> tuple[list[Path], list[Path | None], list[Path | None]]:
    """Discover EVN and ODD volumes, and the per-tomo tlt file if present.

    Returns three parallel lists ``(evn_paths, odd_paths, tlt_paths)``.
    ``tlt_paths[i]`` is the path to the full-series tlt file for the i-th
    tomo, or ``None`` if no tlt file was found.
    """
    evn_paths: list[Path] = []
    odd_paths: list[Path | None] = []
    tlt_paths: list[Path | None] = []

    for tomo_dir in sorted(input_dir.glob("tomo_*")):
        evn_matches = sorted(tomo_dir.glob(evn_glob))
        if not evn_matches:
            # Fallback: any *IsoNet*.mrc (old convention)
            evn_matches = sorted(tomo_dir.glob("vol*IsoNet*.mrc"))
        if not evn_matches:
            print(f"[ei-data] WARNING: no EVN volume in {tomo_dir}, skipping.")
            continue

        odd_matches = sorted(tomo_dir.glob(odd_glob))
        evn_paths.append(evn_matches[0])
        odd_paths.append(odd_matches[0] if odd_matches else None)
        tlt_paths.append(_find_tlt_for_dir(tomo_dir))

    n_paired  = sum(p is not None for p in odd_paths)
    n_evnonly = len(evn_paths) - n_paired
    n_tlt     = sum(p is not None for p in tlt_paths)
    print(
        f"[ei-data] discovered {len(evn_paths)} tomo dirs: "
        f"{n_paired} paired EVN+ODD, {n_evnonly} EVN-only, {n_tlt} with tlt files."
    )
    return evn_paths, odd_paths, tlt_paths


def _split_pairs(
    evn_paths: list[Path],
    odd_paths: list[Path | None],
    n_val: int,
    seed: int,
    max_train: int | None,
    extra: list | None = None,
) -> tuple[list, list, list, list, list, list]:
    """Shuffle and split (evn, odd, extra) lists into train / val.

    ``extra`` is any parallel list (e.g. tilt_ranges); ``None`` values are
    fine.  Returns ``(train_evn, train_odd, val_evn, val_odd, train_extra, val_extra)``.
    """
    rng = random.Random(seed)
    indices = list(range(len(evn_paths)))
    rng.shuffle(indices)
    n_val = max(0, min(n_val, len(indices) - 1))
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]
    if max_train is not None:
        train_idx = train_idx[:max_train]
    _extra = extra if extra is not None else [None] * len(evn_paths)
    train_evn   = [evn_paths[i] for i in train_idx]
    train_odd   = [odd_paths[i] for i in train_idx]
    train_extra = [_extra[i] for i in train_idx]
    val_evn     = [evn_paths[i] for i in val_idx]
    val_odd     = [odd_paths[i] for i in val_idx]
    val_extra   = [_extra[i] for i in val_idx]
    return train_evn, train_odd, val_evn, val_odd, train_extra, val_extra


# ---------------------------------------------------------------------------
# MRC I/O / volume helpers
# ---------------------------------------------------------------------------

def _find_mrc(tomo_dir: Path, *globs: str) -> Path | None:
    """Return the first file matching any glob in *tomo_dir*, or None."""
    for glob in globs:
        matches = sorted(tomo_dir.glob(glob))
        if matches:
            return matches[0]
    return None


def _save_mrc(path: Path, vol_dhw: np.ndarray) -> None:
    """Save a (D, H, W) float32 numpy array as an MRC file (axis order: Z, Y, X)."""
    vol_zyx = np.moveaxis(vol_dhw.astype(np.float32), 2, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(vol_zyx)


def _read_mrc_vol_size(path: Path) -> int:
    """Read an MRC header and return the smallest spatial dimension (cubic vol side)."""
    with mrcfile.open(str(path), permissive=True, mode="r") as mrc:
        nx, ny, nz = int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz)
    return min(nx, ny, nz)


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


def _center_crop(vol: np.ndarray, size: int = 512) -> np.ndarray:
    """Center-crop (D, H, W) to a cube of min(size, smallest dim)."""
    D, H, W = vol.shape
    s = min(size, D, H, W)
    d0, h0, w0 = (D - s) // 2, (H - s) // 2, (W - s) // 2
    return vol[d0:d0 + s, h0:h0 + s, w0:w0 + s]


def _znorm(vol: np.ndarray) -> np.ndarray:
    """Z-score normalise a volume in-place (returns float32)."""
    mu, sigma = float(vol.mean()), float(vol.std())
    return ((vol - mu) / (sigma + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Shared visualisation / FSC helpers
# ---------------------------------------------------------------------------

def fsc_shell(fsc_curve: np.ndarray, threshold: float) -> int:
    """Return first shell index where FSC drops below *threshold* (or last shell)."""
    below = np.where(fsc_curve < threshold)[0]
    return int(below[0]) if len(below) > 0 else int(len(fsc_curve) - 1)


def save_fsc_figure(
    images_dir: Path,
    epoch: int,
    fname: str,
    fsc_curve: np.ndarray,
    res_shell: int,
    res_angstrom: float,
    title: str,
    threshold: float = 0.143,
    vol_size: int | None = None,
    pixel_size: float | None = None,
) -> None:
    """Save an FSC curve PNG with threshold and resolution marker lines.

    The x-axis shows resolution in Å when *vol_size* and *pixel_size* are
    provided, otherwise falls back to shell index.
    """
    n = len(fsc_curve)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fsc_curve, lw=1.5)
    ax.axhline(threshold, color="r", ls="--", label=f"thr={threshold}")

    if vol_size and pixel_size:
        tick_shells = np.linspace(max(1, n // 8), n - 1, num=8, dtype=int)
        ax.axvline(res_shell, color="orange", ls=":", label=f"{res_angstrom:.1f} Å (shell {res_shell})")
        ax.set_xticks(tick_shells)
        ax.set_xticklabels([f"{vol_size * pixel_size / k:.1f}" for k in tick_shells],
                           rotation=30, ha="right", fontsize=7)
        ax.set_xlabel("Resolution (Å)")
    else:
        ax.axvline(res_shell, color="orange", ls=":", label=f"shell={res_shell} ({res_angstrom:.1f} Å)")
        ax.set_xlabel("Shell index")

    ax.set_ylabel("FSC")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    out = Path(images_dir) / f"fsc_epoch{epoch:04d}" / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def save_resolution_histogram(
    images_dir: Path,
    epoch: int,
    resolutions_angstrom: list[float],
    mean_res: float,
    median_res: float,
    q1_res: float,
    q3_res: float,
    threshold_label: str = "0.143",
) -> None:
    """Save a histogram of per-volume FSC resolutions (Å).

    Vertical lines mark mean, median, Q1 and Q3.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    n = len(resolutions_angstrom)
    ax.hist(resolutions_angstrom, bins=max(5, n // 2 + 1), color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(mean_res,   color="tomato",    ls="-",  lw=1.8, label=f"mean   = {mean_res:.1f} Å")
    ax.axvline(median_res, color="mediumpurple", ls="-", lw=1.8, label=f"median = {median_res:.1f} Å")
    ax.axvline(q1_res,     color="goldenrod", ls="--", lw=1.4, label=f"Q1     = {q1_res:.1f} Å")
    ax.axvline(q3_res,     color="goldenrod", ls=":",  lw=1.4, label=f"Q3     = {q3_res:.1f} Å")
    ax.set_xlabel("FSC resolution (Å)  —  lower is better")
    ax.set_ylabel("# volumes")
    ax.set_title(f"Epoch {epoch} | FSC@{threshold_label} resolution ({n} vols)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(images_dir) / f"fsc_epoch{epoch:04d}" / "resolution_histogram.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def save_slice_figure(
    images_dir: Path,
    epoch: int,
    vol_idx: int,
    cols: list[np.ndarray],
    labels: list[str] | None = None,
    title: str | None = None,
    subdir: str | None = None,
    fname: str | None = None,
) -> None:
    """Save a 3×N PNG with all three orthogonal mid-slices.

    Rows: XY (z-mid) / XZ (y-mid) / YZ (x-mid).
    Cols: one column per entry in *cols*  (caller chooses labels).
    N = len(cols), so any number of columns is supported.
    """
    def _slices(v: np.ndarray, max_px: int = 256) -> list[np.ndarray]:
        # Slice first (returns views, no copy), then convert only the small 2-D result.
        d, h, w = v.shape
        slcs = [v[d // 2, :, :], v[:, h // 2, :], v[:, :, w // 2]]
        out = []
        for s in slcs:
            # Stride-subsample to keep at most max_px pixels per axis
            sr, sc = max(1, s.shape[0] // max_px), max(1, s.shape[1] // max_px)
            out.append(s[::sr, ::sc].astype(np.float32))
        return out

    n_cols = len(cols)
    if labels is None:
        labels = [f"Col {i}" for i in range(n_cols)]
    planes = ["XY (z-mid)", "XZ (y-mid)", "YZ (x-mid)"]

    # Pre-compute slices once per column (4 calls) — avoids recomputing 12× inside the loop.
    pre = [_slices(v) for v in cols]

    # Per-row 1st–99th percentile across all columns — clips outliers while keeping
    # columns comparable within each row (handles missing-wedge contrast differences).
    row_ranges = [
        (float(np.percentile(np.concatenate([slc_list[r].ravel() for slc_list in pre]), 1)),
         float(np.percentile(np.concatenate([slc_list[r].ravel() for slc_list in pre]), 99)))
        for r in range(3)
    ]

    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12))
    if n_cols == 1:
        axes = axes[:, np.newaxis]  # keep 2-D indexing consistent
    for row, plane in enumerate(planes):
        vmin, vmax = row_ranges[row]
        for col, (label, slc_list) in enumerate(zip(labels, pre)):
            ax = axes[row, col]
            ax.imshow(slc_list[row], cmap="gray", vmin=vmin, vmax=vmax)
            ax.axis("off")
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(plane)
    fig.suptitle(title if title is not None else f"Epoch {epoch} | Vol {vol_idx}")
    fig.tight_layout()

    folder = subdir if subdir is not None else f"fsc_epoch{epoch:04d}"
    out = Path(images_dir) / folder / (fname if fname is not None else f"vol{vol_idx:02d}_slices.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)


class GpuFSC:
    """Fourier Shell Correlation computed entirely on GPU (float32).

    Matches FCC/FSC from utils_FSC.py (phiArray=[0.0] case):
      1. Build radial shell index map once at construction (cached).
      2. torch.fft.fftn + fftshift on GPU in complex64.
      3. Shell-bin numerator and denominators via scatter_add.
      4. Return FSC curve as a small 1-D numpy array.

    Args:
        vol_size: side length of the cubic volume (D = H = W).
        device:   torch device string or object.
    """

    def __init__(self, vol_size: int, device: str | torch.device = "cuda") -> None:
        self.device   = torch.device(device)
        self.vol_size = vol_size

        D    = vol_size
        half = D / 2.0
        c    = torch.arange(D, dtype=torch.float32, device=self.device) - half
        z, y, x = torch.meshgrid(c, c, c, indexing="ij")
        rho  = torch.sqrt(x * x + y * y + z * z)
        self._shells  = torch.round(rho).long().reshape(-1)
        self._rhomax  = int(np.ceil(np.sqrt(3.0) * half) + 2)

    def __call__(self, vol1: torch.Tensor, vol2: torch.Tensor) -> np.ndarray:
        """Return FSC curve as 1-D numpy array (same format as ``FSC(a,b)[:,0]``)."""
        v1 = vol1.squeeze().to(self.device, dtype=torch.float32)
        v2 = vol2.squeeze().to(self.device, dtype=torch.float32)

        F1 = torch.fft.fftshift(torch.fft.fftn(v1))
        F2 = torch.fft.fftshift(torch.fft.fftn(v2))

        cross = (F1 * F2.conj()).real.reshape(-1)
        pow1  = (F1.real ** 2 + F1.imag ** 2).reshape(-1)
        pow2  = (F2.real ** 2 + F2.imag ** 2).reshape(-1)

        sh = self._shells.clamp(0, self._rhomax - 1)
        z_ = torch.zeros(self._rhomax, dtype=torch.float32, device=self.device)
        num  = z_.clone().scatter_add_(0, sh, cross)
        den1 = z_.clone().scatter_add_(0, sh, pow1)
        den2 = z_.clone().scatter_add_(0, sh, pow2)

        denom = torch.sqrt(den1 * den2)
        fsc   = torch.where(denom > 0.0, num / denom, torch.zeros_like(num))
        return fsc.cpu().numpy()


# ---------------------------------------------------------------------------
# Self-supervised reconstruction helper
# ---------------------------------------------------------------------------

def half_set_recon(
    model: "torch.nn.Module",
    physics,
    f_evn: "torch.Tensor",
    f_odd: "torch.Tensor",
) -> "torch.Tensor":
    """Self-supervised reconstruction: 0.5 * (f(A(f_evn)) + f(A(f_odd)))."""
    return 0.5 * (model(physics.A(f_evn)) + model(physics.A(f_odd)))


# ---------------------------------------------------------------------------
# CSV stats helper
# ---------------------------------------------------------------------------

def save_slice_stats_csv(
    path: Path,
    labels: list[str],
    vols: list["np.ndarray | None"],
) -> None:
    """Write per-image per-plane statistics to a CSV file.

    For each valid (non-None) volume in *vols*, extracts the three orthogonal
    mid-plane slices (XY, XZ, YZ) at full resolution and records:
    min, max, 1st-percentile (q01), 99th-percentile (q99).

    :param path: Destination CSV file path (parent dirs created automatically).
    :param labels: One label per entry in *vols*.
    :param vols: List of float32 numpy arrays of shape (D, H, W), or None for missing volumes.
    """
    import csv as _csv

    planes = ["XY", "XZ", "YZ"]
    rows: list[dict] = []
    for label, vol in zip(labels, vols):
        if vol is None:
            continue
        d, h, w = vol.shape
        mid_slices = [
            vol[d // 2, :, :],   # XY  (z-mid)
            vol[:, h // 2, :],   # XZ  (y-mid)
            vol[:, :, w // 2],   # YZ  (x-mid)
        ]
        for plane, slc in zip(planes, mid_slices):
            arr = slc.ravel().astype(np.float32)
            rows.append({
                "image": label,
                "plane": plane,
                "min":   float(arr.min()),
                "max":   float(arr.max()),
                "q01":   float(np.percentile(arr, 1)),
                "q99":   float(np.percentile(arr, 99)),
            })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=["image", "plane", "min", "max", "q01", "q99"])
        writer.writeheader()
        writer.writerows(rows)


