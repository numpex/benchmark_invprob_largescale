"""Shared dataset utilities for equivariant cryo-ET training.

Contains discovery helpers and ``EIDataBundle`` used by both the patch and
full-volume dataset variants.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader


_ICECREAM_GLOB = "*icecream*.mrc"


@dataclass
class EIDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_paths: list[Path]
    val_paths: list[Path]
    val_icecream_paths: list[Path | None]   # parallel to val_paths; None if no icecream GT


def _discover_pairs(
    input_dir: Path,
    evn_glob: str = "*evn*corrected*.mrc",
    odd_glob: str = "*odd*corrected*.mrc",
    use_icecream_gt: bool = False,
) -> tuple[list[Path], list[Path | None], list[Path | None]]:
    """Discover EVN (and optionally ODD / icecream-denoised) volumes.

    Returns three parallel lists ``(evn_paths, odd_paths, icecream_paths)``.
    ``odd_paths[i]`` and ``icecream_paths[i]`` are ``None`` when unavailable.
    """
    evn_paths: list[Path] = []
    odd_paths: list[Path | None] = []
    icecream_paths: list[Path | None] = []

    for tomo_dir in sorted(input_dir.glob("tomo_*")):
        evn_matches = sorted(tomo_dir.glob(evn_glob))
        if not evn_matches:
            # Fallback: any *corrected.mrc (old convention)
            evn_matches = sorted(tomo_dir.glob("*_corrected.mrc"))
        if not evn_matches:
            print(f"[ei-data] WARNING: no EVN volume in {tomo_dir}, skipping.")
            continue

        odd_matches = sorted(tomo_dir.glob(odd_glob))
        ice_matches = sorted(tomo_dir.glob(_ICECREAM_GLOB)) if use_icecream_gt else []

        evn_paths.append(evn_matches[0])
        odd_paths.append(odd_matches[0] if odd_matches else None)
        icecream_paths.append(ice_matches[0] if ice_matches else None)

    n_paired  = sum(p is not None for p in odd_paths)
    n_ice     = sum(p is not None for p in icecream_paths)
    n_evnonly = len(evn_paths) - n_paired
    print(
        f"[ei-data] discovered {len(evn_paths)} tomo dirs: "
        f"{n_paired} paired EVN+ODD, {n_evnonly} EVN-only, {n_ice} with icecream GT."
    )
    return evn_paths, odd_paths, icecream_paths


def _split_pairs(
    evn_paths: list[Path],
    odd_paths: list[Path | None],
    icecream_paths: list[Path | None],
    n_val: int,
    seed: int,
    max_train: int | None,
) -> tuple[list, list, list, list, list]:
    """Shuffle and split paired (evn, odd, icecream) lists into train / val."""
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
    val_ice   = [icecream_paths[i] for i in val_idx]
    return train_evn, train_odd, val_evn, val_odd, val_ice
