"""CryoEIFullDataset — full-volume dataset for equivariant imaging on full tomograms.

Yields paired (evn_vol, odd_vol) full sub-tomogram volumes from cryo-ET
half-set MRCs, following the same discovery / normalisation conventions as
CryoEIPatchDataset but without any spatial cropping.

Differences from the patch variant:
  - ``__getitem__`` returns the **full** volume (1, D, H, W) instead of
    ``n_crops`` random sub-patches.
  - DataLoader ``batch_size`` is always 1; effective batch size is controlled
    by gradient accumulation.
  - Optional ``target_shape`` trilinearly resamples volumes to a fixed (D, H, W)
    shape, matching supervised CryoDataConfig.target_shape semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from equivariant.utils import EIDataBundle, _discover_pairs, _split_pairs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EIFullDataConfig:
    input_dir: str = "./dataset/empiar-11058"
    num_workers: int = 1
    pin_memory: bool = True
    prefetch_factor: int = 1
    persistent_workers: bool = True
    max_train_vols: int | None = None
    max_val_vols: int = 5
    seed: int = 0
    # If set, volumes are trilinearly resampled to this (D, H, W) shape after
    # loading — same semantics as supervised CryoDataConfig.target_shape.
    target_shape: tuple[int, int, int] | None = None
    # Glob patterns used to discover EVN and ODD volumes inside each tomo_* dir.
    evn_glob: str = "vol*split1*.mrc"
    odd_glob: str = "vol*split2*.mrc"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CryoEIFullDataset(Dataset):
    """Yields one full-volume item as ``(evn_vol, odd_vol)``, each shape ``(1, D, H, W)``.

    Volume normalisation (zero-mean, unit-std) is applied at load time —
    matching icecream's ``load_volume`` behaviour.

    When only EVN is available, ``odd_vol`` is a copy of ``evn_vol`` so the
    single-mode ObsLoss fallback ``L = fourier_loss(y, f(y), wedge)`` works.

    :param list[Path] evn_paths: Paths to EVN half-set MRC volumes.
    :param list[Path] odd_paths: Paths to ODD half-set MRC volumes.
    :param tuple | None target_shape: If set, trilinearly resample each volume to
        this (D, H, W) shape after loading — same as supervised ``target_shape``.
    """

    def __init__(
        self,
        evn_paths: list[Path],
        odd_paths: list[Path],
        target_shape: tuple[int, int, int] | None = None,
    ) -> None:
        assert len(evn_paths) == len(odd_paths)
        self.evn_paths    = evn_paths
        self.odd_paths    = odd_paths
        self.target_shape = target_shape

        print(
            f"[ei-full] CryoEIFullDataset: {len(evn_paths)} paired EVN+ODD vols [lazy]"
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.evn_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        evn = self._load_and_prepare(self.evn_paths[idx])   # (1, D, H, W)
        odd = self._load_and_prepare(self.odd_paths[idx])
        return evn, odd

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_and_prepare(self, path: Path) -> torch.Tensor:
        """Load MRC, reorder axes, optional resample, centre-crop to cube, normalise → (1, D, H, W)."""
        # MRC stores (Z, Y, X); moveaxis → (Y, X, Z) = (D, H, W)
        vol_np = np.array(
            mrcfile.open(str(path), permissive=True).data,
            dtype=np.float32,
        )
        vol = torch.from_numpy(np.moveaxis(vol_np, 0, 2))  # (D, H, W)

        if self.target_shape is not None:
            # interpolate expects (B, C, D, H, W)
            vol = torch.nn.functional.interpolate(
                vol.unsqueeze(0).unsqueeze(0),
                size=self.target_shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)  # back to (D, H, W)

        # Centre-crop to cube of side min(D, H, W)
        # TODO: different strategies for non-cubic volumes? 
        D, H, W = vol.shape
        S = min(D, H, W)
        d0, h0, w0 = (D - S) // 2, (H - S) // 2, (W - S) // 2
        vol = vol[d0:d0 + S, h0:h0 + S, w0:w0 + S]  # (S, S, S)

        # Normalise after crop so stats reflect the kept region
        mu = vol.mean()
        sigma = vol.std()
        vol = (vol - mu) / (sigma + 1e-8)

        return vol.unsqueeze(0)  # (1, S, S, S)


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def _make_full_loader(
    dataset: Dataset,
    shuffle: bool,
    cfg: EIFullDataConfig,
) -> DataLoader:
    kwargs: dict = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=shuffle,
        drop_last=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(**kwargs)


def build_ei_full_dataloaders(cfg: EIFullDataConfig) -> EIDataBundle:
    """Build train / val DataLoaders over full cryo-ET volumes.
    """
    input_dir = Path(cfg.input_dir)
    all_evn, all_odd = _discover_pairs(
        input_dir, cfg.evn_glob, cfg.odd_glob
    )
    train_evn, train_odd, val_evn, val_odd = _split_pairs(
        all_evn, all_odd, cfg.max_val_vols, cfg.seed, cfg.max_train_vols
    )

    train_ds = CryoEIFullDataset(train_evn, train_odd, target_shape=cfg.target_shape)
    val_ds   = CryoEIFullDataset(val_evn,   val_odd,   target_shape=cfg.target_shape)

    print(
        f"[ei-full] total={len(all_evn)}  "
        f"train_vols={len(train_evn)}  val_vols={len(val_evn)}"
    )

    return EIDataBundle(
        train_loader = _make_full_loader(train_ds, shuffle=True,  cfg=cfg),
        val_loader   = _make_full_loader(val_ds,   shuffle=False, cfg=cfg),
    )
