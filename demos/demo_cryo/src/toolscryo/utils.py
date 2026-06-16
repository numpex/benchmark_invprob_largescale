from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_config_json(path: Path, cfg_dict: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)


def append_metrics_row(path: Path | str, row: dict) -> None:
    """Append one row to a CSV file, writing a header on first write."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class DRUNetWrapper(torch.nn.Module):
    """Wraps DRUNet so model(x) works — injects a fixed sigma as a float.

    DRUNet.forward(x, sigma) requires a noise level.  Passing sigma as a
    Python float uses the 3D-safe branch: torch.ones((B,1,*x.shape[2:]))*sigma,
    unlike the tensor branch which hard-codes 2D expand calls.
    """

    def __init__(self, drunet: torch.nn.Module, sigma: float = 0.0) -> None:
        super().__init__()
        self.drunet = drunet
        self.sigma = sigma

    def forward(self, x: torch.Tensor, physics=None, **kwargs) -> torch.Tensor:
        return self.drunet(x, self.sigma)


def build_ei_model(
    model_type: str,
    unet_f_maps: int,
    unet_num_levels: int,
    unet_dropout: float,
    drunet_nb: int,
    drunet_sigma: float,
    device,
) -> tuple[torch.nn.Module, str]:
    """Build IceCreamUNetWrapper (unet) or DRUNetWrapper (drunet) on *device*.

    Returns ``(bare_model, info_str)`` where ``info_str`` describes the architecture.
    Both model types share the same field names in Run*Config so callers can pass
    cfg fields directly.
    """
    import deepinv as dinv
    from icecream_orig.models import IceCreamUNetWrapper
    from icecream_orig.models.unet3d_bf import UNet3D as _IceCreamUNet3D

    if model_type == "unet":
        _inner = _IceCreamUNet3D(
            in_channels=1,
            out_channels=1,
            f_maps=unet_f_maps,
            num_levels=unet_num_levels,
            layer_order="cr",
            use_bias=False,
            dropout_prob=unet_dropout,
        ).to(device)
        model = IceCreamUNetWrapper(_inner)
        info = (
            f"unet  f_maps={unet_f_maps}  num_levels={unet_num_levels}  "
            f"dropout={unet_dropout}"
        )
    elif model_type == "drunet":
        _nc = tuple(unet_f_maps * (2 ** i) for i in range(4))
        model = DRUNetWrapper(
            dinv.models.DRUNet(
                in_channels=1,
                out_channels=1,
                nc=_nc,
                nb=drunet_nb,
                pretrained=None,
                dim=3,
            ).to(device),
            sigma=drunet_sigma,
        )
        info = f"drunet  nc={_nc}  nb={drunet_nb}  sigma={drunet_sigma}"
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'unet' or 'drunet'.")
    return model, info


class PerfProbe:
    """Context manager that measures wall time and peak GPU memory for a code block.

    """
    def __enter__(self) -> "PerfProbe":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_s: float = time.perf_counter() - self._t0
        self.peak_mb: float = (
            torch.cuda.max_memory_allocated() / 1e6
            if torch.cuda.is_available() else 0.0
        )

