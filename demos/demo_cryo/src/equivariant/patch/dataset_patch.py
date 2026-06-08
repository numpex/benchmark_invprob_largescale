"""CryoEIPatchDataset — patch dataset for equivariant imaging on cryo-ET half-sets.

Yields paired (evn_patch, odd_patch) random cubic crops from the same spatial
location in EVN and ODD half-set MRC volumes, following the same discovery and
normalisation conventions as CryoEIFullDataset.

Differences from the full-volume variant:
  - ``__getitem__`` returns the same 3-tuple ``(evn_patch, odd_patch, tilt_params)``
    so it plugs directly into EIPatchTrainer without changes.
  - Patches are random cubic crops of side ``crop_size`` extracted from the same
    coordinates in both EVN and ODD; this preserves the cross half-set pairing.
  - ``__len__`` = len(evn_paths) * n_crops_per_vol (virtual epoch length).
  - Volumes are loaded lazily on each __getitem__ call (no in-memory caching).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from equivariant.utils import EIDataBundle, _discover_pairs, _read_tlt, _split_pairs


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EIPatchDataConfig:
    input_dir: str = "./dataset/empiar-11058"
    crop_size: int = 72
    n_crops_per_vol: int = 10          # virtual epoch = n_vols * n_crops_per_vol
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    max_train_vols: int | None = None
    max_val_vols: int = 5
    seed: int = 0
    normalize: bool = True             # zero-mean, unit-std per patch
    # Glob patterns — same as CryoEIFullDataset
    evn_glob: str = "vol*split1*.mrc"
    odd_glob: str = "vol*split2*.mrc"
    # Fallback tilt range when no tlt file is found.
    fallback_tilt_min: float = -60.0
    fallback_tilt_max: float = 60.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CryoEIPatchDataset(Dataset):
    """Yields ``(evn_patch, odd_patch, tilt_params)`` random cubic crops.

    Both patches are cropped from the **same random spatial position** so the
    cross half-set pairing is preserved — ObsLoss and EqLoss can compare them
    exactly as they compare full volumes in CryoEIFullDataset.

    Normalisation (zero-mean, unit-std) is applied independently to each patch
    after cropping, matching icecream's per-patch normalisation.

    When only EVN is available, ``odd_patch`` is a copy of ``evn_patch`` so the
    single-half ObsLoss fallback ``L = fourier_loss(y, f(y), wedge)`` still works.

    :param list[Path] evn_paths: Paths to EVN half-set MRC volumes.
    :param list[Path | None] odd_paths: Paths to ODD half-set MRC volumes (or None).
    :param int crop_size: Cubic crop side length (default 72).
    :param int n_crops_per_vol: Virtual crops per volume per epoch (default 10).
    :param bool normalize: Standardise each patch independently (default True).
    :param list tilt_ranges: Per-volume (tilt_min, tilt_max) or None.
    :param float fallback_tilt_min: Used when tilt_ranges[i] is None.
    :param float fallback_tilt_max: Used when tilt_ranges[i] is None.
    """

    def __init__(
        self,
        evn_paths: list[Path],
        odd_paths: list[Path | None],
        crop_size: int = 72,
        n_crops_per_vol: int = 10,
        normalize: bool = False,
        tilt_ranges: list[tuple[float, float] | None] | None = None,
        fallback_tilt_min: float = -60.0,
        fallback_tilt_max: float = 60.0,
    ) -> None:
        assert len(evn_paths) == len(odd_paths)
        self.evn_paths         = evn_paths
        self.odd_paths         = odd_paths
        self.crop_size         = crop_size
        self.n_crops_per_vol   = n_crops_per_vol
        self.normalize         = normalize
        self.fallback_tilt_min = fallback_tilt_min
        self.fallback_tilt_max = fallback_tilt_max
        self._tilt_ranges: list[tuple[float, float] | None] = (
            tilt_ranges if tilt_ranges is not None else [None] * len(evn_paths)
        )

        # Load all volumes into CPU memory once at init — matches icecream's load_data.
        print(f"[ei-patch] Loading {len(evn_paths)} volume pair(s) into memory ...")
        self.evn_vols: list[torch.Tensor] = []
        self.odd_vols: list[torch.Tensor | None] = []
        for i, (evn_p, odd_p) in enumerate(zip(evn_paths, odd_paths)):
            evn_t = self._load_vol(evn_p)
            self.evn_vols.append(evn_t)
            if odd_p is not None:
                self.odd_vols.append(self._load_vol(odd_p))
                print(f"[ei-patch]   [{i}] EVN {evn_p.name}  shape={list(evn_t.shape)}")
            else:
                self.odd_vols.append(None)
                print(f"[ei-patch]   [{i}] EVN {evn_p.name}  shape={list(evn_t.shape)}  (no ODD)")

        n_paired = sum(p is not None for p in self.odd_vols)
        n_tlt    = sum(t is not None for t in self._tilt_ranges)
        print(
            f"[ei-patch] CryoEIPatchDataset: {len(evn_paths)} vols "
            f"({n_paired} paired EVN+ODD), crop_size={crop_size}, "
            f"n_crops_per_vol={n_crops_per_vol}"
            + (f", {n_tlt} with tlt" if n_tlt else
               f" (fallback tilt [{fallback_tilt_min}, {fallback_tilt_max}]°)")
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.evn_vols) * self.n_crops_per_vol

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        vol_idx = idx % len(self.evn_vols)

        evn_vol = self.evn_vols[vol_idx]   # (D, H, W) tensor, already in memory
        odd_vol = self.odd_vols[vol_idx]
        if odd_vol is None:
            odd_vol = evn_vol

        # Same random crop coordinates for both halves
        D, H, W = evn_vol.shape
        cs = self.crop_size
        d0 = random.randint(0, max(0, D - cs))
        h0 = random.randint(0, max(0, H - cs))
        w0 = random.randint(0, max(0, W - cs))

        evn_patch = evn_vol[d0:d0 + cs, h0:h0 + cs, w0:w0 + cs].unsqueeze(0)  # (1, cs, cs, cs)
        odd_patch = odd_vol[d0:d0 + cs, h0:h0 + cs, w0:w0 + cs].unsqueeze(0)

        tilt = self._tilt_ranges[vol_idx]
        if tilt is None:
            tilt = (self.fallback_tilt_min, self.fallback_tilt_max)
        tilt_params = {
            "tilt_min": torch.tensor(tilt[0], dtype=torch.float32),
            "tilt_max": torch.tensor(tilt[1], dtype=torch.float32),
        }
        return evn_patch, odd_patch, tilt_params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_vol(self, path: Path) -> torch.Tensor:
        """Load MRC into a CPU float32 tensor, normalize whole volume if requested."""
        vol_np = np.array(
            mrcfile.open(str(path), permissive=True).data,
            dtype=np.float32,
        )
        # MRC stores (Z, Y, X); moveaxis → (Y, X, Z) = (D, H, W), matching icecream
        vol_np = np.moveaxis(vol_np, 0, 2)
        vol_t = torch.from_numpy(vol_np)
        if self.normalize:
            vol_t = (vol_t - vol_t.mean()) / (vol_t.std() + 1e-8)
        return vol_t


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def _make_patch_loader(
    dataset: Dataset,
    shuffle: bool,
    cfg: EIPatchDataConfig,
) -> DataLoader:
    kwargs: dict = dict(
        dataset=dataset,
        batch_size=int(cfg.batch_size),
        shuffle=shuffle,
        drop_last=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"]    = int(cfg.prefetch_factor)
    return DataLoader(**kwargs)


def build_ei_patch_dataloaders(cfg: EIPatchDataConfig) -> EIDataBundle:
    """Build train / val DataLoaders over paired EVN+ODD patch crops."""
    input_dir = Path(cfg.input_dir)
    all_evn, all_odd, all_tlt = _discover_pairs(
        input_dir, cfg.evn_glob, cfg.odd_glob
    )
    

    all_tilt_ranges: list[tuple[float, float] | None] = []
    for tlt_path in all_tlt:
        if tlt_path is not None:
            try:
                all_tilt_ranges.append(_read_tlt(tlt_path))
            except Exception as e:
                print(f"[ei-data] WARNING: could not read {tlt_path}: {e}")
                all_tilt_ranges.append(None)
        else:
            all_tilt_ranges.append(None)

    train_evn, train_odd, val_evn, val_odd, train_tlt, val_tlt = _split_pairs(
        all_evn, all_odd, cfg.max_val_vols, cfg.seed, cfg.max_train_vols,
        extra=all_tilt_ranges,
    )

    ds_kwargs = dict(
        crop_size=int(cfg.crop_size),
        n_crops_per_vol=int(cfg.n_crops_per_vol),
        normalize=bool(cfg.normalize),
        fallback_tilt_min=cfg.fallback_tilt_min,
        fallback_tilt_max=cfg.fallback_tilt_max,
    )
    train_ds = CryoEIPatchDataset(train_evn, train_odd, tilt_ranges=train_tlt, **ds_kwargs)
    val_ds   = CryoEIPatchDataset(val_evn,   val_odd,   tilt_ranges=val_tlt,   **ds_kwargs)

    print(
        f"[ei-patch] total={len(all_evn)}  "
        f"train_vols={len(train_evn)}  val_vols={len(val_evn)}  "
        f"train_patches={len(train_ds)}  val_patches={len(val_ds)}"
    )

    return EIDataBundle(
        train_loader=_make_patch_loader(train_ds, shuffle=True,  cfg=cfg),
        val_loader  =_make_patch_loader(val_ds,   shuffle=False, cfg=cfg),
    )
