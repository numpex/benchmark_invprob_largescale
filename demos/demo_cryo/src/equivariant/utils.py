"""Shared utilities for equivariant cryo-ET training.

Contains dataset discovery helpers (``EIDataBundle``, ``_discover_pairs``,
``_split_pairs``) and visualisation helpers (``save_slice_figure``) used by
both the patch and full-volume variants.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class EIDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader


def _discover_pairs(
    input_dir: Path,
    evn_glob: str = "vol*split1*.mrc",
    odd_glob: str = "vol*split2*.mrc",
) -> tuple[list[Path], list[Path | None]]:
    """Discover EVN and ODD volumes.

    Returns two parallel lists ``(evn_paths, odd_paths)``.
    """
    evn_paths: list[Path] = []
    odd_paths: list[Path | None] = []

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

    n_paired  = sum(p is not None for p in odd_paths)
    n_evnonly = len(evn_paths) - n_paired
    print(
        f"[ei-data] discovered {len(evn_paths)} tomo dirs: "
        f"{n_paired} paired EVN+ODD, {n_evnonly} EVN-only."
    )
    return evn_paths, odd_paths


def _split_pairs(
    evn_paths: list[Path],
    odd_paths: list[Path | None],
    n_val: int,
    seed: int,
    max_train: int | None,
) -> tuple[list, list, list, list]:
    """Shuffle and split paired (evn, odd) lists into train / val."""
    rng = random.Random(seed)
    indices = list(range(len(evn_paths)))
    rng.shuffle(indices)
    n_val = max(1, min(n_val, len(indices) - 1))
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]
    if max_train is not None:
        train_idx = train_idx[:max_train]
    train_evn = [evn_paths[i] for i in train_idx]
    train_odd = [odd_paths[i] for i in train_idx]
    val_evn   = [evn_paths[i] for i in val_idx]
    val_odd   = [odd_paths[i] for i in val_idx]
    return train_evn, train_odd, val_evn, val_odd


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
    q1_res: float,
    q3_res: float,
    threshold_label: str = "0.143",
) -> None:
    """Save a histogram of per-volume FSC resolutions (Å).

    Vertical lines mark mean, Q1 and Q3.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    n = len(resolutions_angstrom)
    ax.hist(resolutions_angstrom, bins=max(5, n // 2 + 1), color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(mean_res, color="tomato",   ls="-",  lw=1.8, label=f"mean = {mean_res:.1f} Å")
    ax.axvline(q1_res,   color="goldenrod", ls="--", lw=1.4, label=f"Q1   = {q1_res:.1f} Å")
    ax.axvline(q3_res,   color="goldenrod", ls=":",  lw=1.4, label=f"Q3   = {q3_res:.1f} Å")
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
    col0: np.ndarray,
    col1: np.ndarray,
    col2: np.ndarray,
    col3: np.ndarray,
    labels: list[str] | None = None,
    title: str | None = None,
    subdir: str | None = None,
    fname: str | None = None,
) -> None:
    """Save a 3×4 PNG with all three orthogonal mid-slices.

    Rows: XY (z-mid) / XZ (y-mid) / YZ (x-mid).
    Cols: col0 | col1 | col2 | col3  (caller chooses labels).
    """
    def _slices(v: np.ndarray, max_px: int = 256) -> list[np.ndarray]:
        v = v.astype(np.float32)
        d, h, w = v.shape
        slcs = [v[d // 2, :, :], v[:, h // 2, :], v[:, :, w // 2]]
        out = []
        for s in slcs:
            # Stride-subsample to keep at most max_px pixels per axis
            sr, sc = max(1, s.shape[0] // max_px), max(1, s.shape[1] // max_px)
            out.append(s[::sr, ::sc])
        return out

    cols = [col0, col1, col2, col3]
    if labels is None:
        labels = ["Col 0", "Col 1", "Col 2", "Col 3"]
    planes = ["XY (z-mid)", "XZ (y-mid)", "YZ (x-mid)"]
    all_vals = np.concatenate([c.ravel() for c in cols])
    vmin = float(np.percentile(all_vals, 1))
    vmax = float(np.percentile(all_vals, 99))

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for row, plane in enumerate(planes):
        for col, (vol, label) in enumerate(zip(cols, labels)):
            ax = axes[row, col]
            ax.imshow(_slices(vol)[row], cmap="gray", vmin=vmin, vmax=vmax)
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
    fig.savefig(out, dpi=120)
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
