"""CryoEIPatchDataset — self-supervised patch dataset for equivariant imaging.

Yields paired (evn_patch, odd_patch) cubic sub-tomogram patches from cryo-ET
half-set volumes, following the same approach as IceIceBreaker/icecream.

Volume discovery (per ``tomo_*`` directory):
  - EVN: first file matching ``evn_glob`` (default ``*evn*corrected*.mrc``)
  - ODD: first file matching ``odd_glob`` (default ``*odd*corrected*.mrc``)

If both EVN and ODD volumes are found, they are returned paired at the **same**
spatial crop location so the cross-consistency loss
``||A(f(EVN)) - ODD||² + ||A(f(ODD)) - EVN||²`` can be computed.

If only an EVN volume is found (no ODD reconstruction available), the dataset
falls back to returning ``(evn_patch, evn_patch)`` so training still works,
just without cross-consistency.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from equivariant.dataset_utils import EIDataBundle, _discover_pairs, _split_pairs


def _single_item_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep volume-wise semantics: DataLoader batch_size=1, pass item through unchanged."""
    return batch[0]


@dataclass
class EIPatchDataConfig:
    input_dir: str = "./dataset/empiar-11058"
    crop_size: int = 72
    # Icecream semantics: this "batch_size" is the number of random crops
    # generated per volume item (MultiVolume.n_crops), NOT DataLoader batch size.
    batch_size: int = 4
    # Legacy alias kept for backward compatibility with older config names.
    # When provided, it can be used by callers to populate ``batch_size``.
    n_crops_per_vol: int | None = None
    num_workers: int = 1
    pin_memory: bool = True
    prefetch_factor: int = 1
    persistent_workers: bool = True
    max_train_vols: int | None = None
    max_val_vols: int = 5
    seed: int = 0
    normalize_crops: bool = False      # optional per-crop re-normalisation (default off, matches icecream)
    # Volume-level normalisation (zero-mean, unit-std on the full volume) is
    # always applied at load time — matching icecream's load_volume behaviour.
    # Glob patterns used to discover EVN and ODD volumes inside each tomo_* dir.
    # The first match per directory wins.  If no ODD match is found, the dataset
    # falls back to returning (evn_patch, evn_patch) for that volume.
    evn_glob: str = "*evn*corrected*.mrc"
    odd_glob: str = "*odd*corrected*.mrc"
    use_icecream_gt: bool = False   # discover *icecream*.mrc files for GT comparison


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CryoEIPatchDataset(Dataset):
    """Yields one volume-item as (evn_crops, odd_crops), each with ``n_crops`` random crops.

    Both patches are extracted from the **same spatial location** so that the
    cross-consistency loss ``||A(f(EVN)) - ODD||²`` is well-defined.

        Matches icecream MultiVolume semantics:
            - ``__len__`` = number of volumes
            - ``__getitem__(i)`` samples ``n_crops_per_vol`` random locations *inside volume i*
            - returns stacked crop tensors with crop axis first

    :param list[Path] evn_paths: Paths to EVN half-set MRC volumes.
    :param list[Path | None] odd_paths: Paths to ODD half-set MRC volumes,
        or ``None`` entries for tomograms where ODD is unavailable.
    :param int crop_size: Cubic crop side length.
    :param int n_crops_per_vol: Number of random crops sampled per volume item.
    :param bool normalize_crops: If True, re-standardise each patch after cropping
        (icecream's ``normalize_crops`` flag, default False).
        Volumes are always normalised at load time regardless of this flag.
    """

    def __init__(
        self,
        evn_paths: list[Path],
        odd_paths: list[Path | None],
        crop_size: int = 72,
        n_crops_per_vol: int = 10,
        normalize_crops: bool = False,
    ) -> None:
        assert len(evn_paths) == len(odd_paths), (
            "evn_paths and odd_paths must have the same length"
        )
        self.evn_paths = evn_paths
        self.odd_paths = odd_paths
        self.crop_size = crop_size
        self.n_crops_per_vol = n_crops_per_vol
        self.normalize_crops = normalize_crops

        # Lazy loading: volumes are read from disk in __getitem__, not here.
        n_paired = sum(p is not None for p in odd_paths)
        print(
            f"[ei-data] CryoEIPatchDataset: {len(evn_paths)} vols "
            f"({n_paired} paired EVN+ODD, {len(evn_paths)-n_paired} EVN-only) [lazy]"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vol_paths(self) -> list[Path]:
        """Backwards-compat alias → list of EVN paths."""
        return self.evn_paths

    def __len__(self) -> int:
        # One dataset item per volume, like icecream MultiVolume.
        return len(self.evn_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        evn_vol = torch.from_numpy(self._load_mrc(self.evn_paths[idx]))
        odd_path = self.odd_paths[idx]
        odd_vol = torch.from_numpy(self._load_mrc(odd_path)) if odd_path is not None else None

        evn_crops, odd_crops = self._sample_crops(evn_vol, odd_vol, self.n_crops_per_vol)

        if self.normalize_crops:
            evn_crops = self._standardise_batch(evn_crops)
            odd_crops = self._standardise_batch(odd_crops)

        # (n_crops, 1, cs, cs, cs) each
        return evn_crops.unsqueeze(1), odd_crops.unsqueeze(1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mrc(path: Path) -> np.ndarray:
        """Load MRC, reorder axes to match icecream (moveaxis 0→2), normalise volume."""
        vol = np.array(
            mrcfile.open(str(path), permissive=True).data,
            dtype=np.float32,
        )  # (Z, Y, X) as stored in MRC
        vol = np.moveaxis(vol, 0, 2)      # → (Y, X, Z), matches icecream load_volume
        # Always normalise the full volume — matches icecream's load_volume
        mu, sigma = vol.mean(), vol.std()
        vol = (vol - mu) / (sigma + 1e-8)
        return vol  # (Y, X, Z), zero-mean unit-std

    def _sample_crops(
        self,
        vol1: torch.Tensor,
        vol2: torch.Tensor | None,
        n_crops: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample n_crops shared-location crops, matching icecream crop_volumes behavior."""
        n1, n2, n3 = vol1.shape
        cs = self.crop_size

        crops_1 = []
        crops_2 = []
        for _ in range(n_crops):
            # Keep icecream semantics exactly: upper bound excluded, no padding fallback.
            start1 = int(np.random.randint(0, n1 - cs))
            start2 = int(np.random.randint(0, n2 - cs))
            start3 = int(np.random.randint(0, n3 - cs))
            crops_1.append(vol1[start1:start1 + cs, start2:start2 + cs, start3:start3 + cs])
            if vol2 is None:
                crops_2.append(crops_1[-1].clone())
            else:
                crops_2.append(vol2[start1:start1 + cs, start2:start2 + cs, start3:start3 + cs])

        return torch.stack(crops_1), torch.stack(crops_2)

    @staticmethod
    def _standardise_batch(crops: torch.Tensor) -> torch.Tensor:
        mu = crops.mean(dim=(-1, -2, -3), keepdim=True)
        sigma = crops.std(dim=(-1, -2, -3), keepdim=True).clamp(min=1e-8)
        return (crops - mu) / sigma


# ---------------------------------------------------------------------------
# Discovery helpers  (implementations live in equivariant.dataset_utils)
# ---------------------------------------------------------------------------

_ICECREAM_GLOB = "*icecream*.mrc"


def _make_loader(dataset: Dataset, shuffle: bool, cfg: EIPatchDataConfig) -> DataLoader:
    kwargs: dict = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=shuffle,
        drop_last=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        collate_fn=_single_item_collate,
    )
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(**kwargs)


def build_ei_patch_dataloaders(cfg: EIPatchDataConfig) -> EIDataBundle:
    # Exact icecream semantics: crops-per-volume is controlled by ``batch_size``.
    n_crops = int(cfg.batch_size)

    input_dir = Path(cfg.input_dir)
    all_evn, all_odd, all_ice = _discover_pairs(
        input_dir, cfg.evn_glob, cfg.odd_glob, cfg.use_icecream_gt
    )
    if not all_evn:
        raise RuntimeError(
            f"No EVN volumes found under {input_dir} "
            f"(evn_glob='{cfg.evn_glob}')."
        )

    train_evn, train_odd, val_evn, val_odd, val_ice = _split_pairs(
        all_evn, all_odd, all_ice, cfg.max_val_vols, cfg.seed, cfg.max_train_vols
    )

    train_ds = CryoEIPatchDataset(
        train_evn, train_odd, cfg.crop_size, n_crops, cfg.normalize_crops
    )
    val_ds = CryoEIPatchDataset(
        val_evn, val_odd, cfg.crop_size, n_crops, cfg.normalize_crops
    )

    print(
        f"[ei-data] total={len(all_evn)}  "
        f"train_vols={len(train_evn)}  val_vols={len(val_evn)}"
    )
    print(
        f"[ei-data] train_vol_items={len(train_ds)}  val_vol_items={len(val_ds)}  "
        f"crops_per_item={n_crops}"
    )

    return EIDataBundle(
        train_loader=_make_loader(train_ds, shuffle=True,  cfg=cfg),
        val_loader  =_make_loader(val_ds,   shuffle=False, cfg=cfg),
        train_paths=train_evn,
        val_paths=val_evn,
        val_icecream_paths=val_ice,
    )
