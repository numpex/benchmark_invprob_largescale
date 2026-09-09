"""Compact point-source catalogues, rendering, and measurement refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def remove_gt_sources(
    image: torch.Tensor, kernel_size: int = 7, threshold: float = 0.0
) -> torch.Tensor:
    """Remove isolated positive target pixels, matching the radio demo."""
    positive = (image > threshold).float()
    kernel = torch.ones(
        1, 1, kernel_size, kernel_size, device=image.device, dtype=image.dtype
    )
    count = F.conv2d(positive.unsqueeze(0), kernel, padding=kernel_size // 2).squeeze(0)
    result = image.clone()
    result[(positive > 0) & (count == 1)] = 0.0
    return result


@dataclass(frozen=True)
class SourceCatalog:
    """PyBDSF point-source initialization in zero-based image coordinates."""

    amplitude: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    major: torch.Tensor | None = None
    minor: torch.Tensor | None = None
    position_angle: torch.Tensor | None = None

    def __post_init__(self) -> None:
        fields = (self.amplitude, self.x, self.y)
        if any(tensor.ndim != 1 for tensor in fields):
            raise ValueError("Source catalogue amplitude, x, and y must be 1-D.")
        if not (self.amplitude.shape == self.x.shape == self.y.shape):
            raise ValueError(
                "Source catalogue amplitude, x, and y must have equal size."
            )

    def __len__(self) -> int:
        return int(self.amplitude.numel())

    def to(self, *args, **kwargs) -> "SourceCatalog":
        def move(tensor):
            return None if tensor is None else tensor.to(*args, **kwargs)

        return SourceCatalog(
            amplitude=move(self.amplitude),
            x=move(self.x),
            y=move(self.y),
            major=move(self.major),
            minor=move(self.minor),
            position_angle=move(self.position_angle),
        )

    def pin_memory(self) -> "SourceCatalog":
        return self.to(device="cpu")._map(torch.Tensor.pin_memory)

    def _map(self, function) -> "SourceCatalog":
        def apply(tensor):
            return None if tensor is None else function(tensor)

        return SourceCatalog(
            amplitude=apply(self.amplitude),
            x=apply(self.x),
            y=apply(self.y),
            major=apply(self.major),
            minor=apply(self.minor),
            position_angle=apply(self.position_angle),
        )


@dataclass(frozen=True)
class SourceParams:
    amplitude: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor


def save_source_catalog(path: str | Path, catalog: SourceCatalog) -> None:
    """Save a catalogue without pickled Python objects."""
    values = {
        "amplitude": catalog.amplitude.detach().cpu().numpy(),
        "x": catalog.x.detach().cpu().numpy(),
        "y": catalog.y.detach().cpu().numpy(),
    }
    for name in ("major", "minor", "position_angle"):
        tensor = getattr(catalog, name)
        if tensor is not None:
            values[name] = tensor.detach().cpu().numpy()
    np.savez(Path(path), **values)


def load_source_catalog(path: str | Path) -> SourceCatalog:
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as values:
            required = {"amplitude", "x", "y"}
            missing = required.difference(values.files)
            if missing:
                raise ValueError(f"missing arrays: {sorted(missing)}")

            def tensor(name: str) -> torch.Tensor | None:
                if name not in values.files:
                    return None
                array = np.asarray(values[name], dtype=np.float32)
                return torch.from_numpy(np.ascontiguousarray(array))

            catalog = SourceCatalog(
                amplitude=tensor("amplitude"),
                x=tensor("x"),
                y=tensor("y"),
                major=tensor("major"),
                minor=tensor("minor"),
                position_angle=tensor("position_angle"),
            )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid source catalogue {path}: {exc}") from exc
    return catalog


def render_sources(
    sources: SourceCatalog | SourceParams,
    image_shape: tuple[int, ...],
) -> torch.Tensor:
    """Render point sources with differentiable bilinear sub-pixel splatting.

    ``x`` is the FITS/PyBDSF horizontal coordinate (last tensor dimension) and
    ``y`` is the vertical coordinate (penultimate tensor dimension).
    """
    if len(image_shape) not in (2, 3, 4):
        raise ValueError(f"Expected a 2-D, C,H,W, or B,C,H,W shape, got {image_shape}.")
    height, width = int(image_shape[-2]), int(image_shape[-1])
    amplitude, x, y = sources.amplitude, sources.x, sources.y
    if not (amplitude.shape == x.shape == y.shape):
        raise ValueError("Source amplitude, x, and y must have equal shape.")

    flat = amplitude.new_zeros(height * width)
    if amplitude.numel() != 0:
        x0 = torch.floor(x)
        y0 = torch.floor(y)
        fx = x - x0
        fy = y - y0
        x0 = x0.to(torch.long)
        y0 = y0.to(torch.long)

        for xi, yi, weight in (
            (x0, y0, (1.0 - fx) * (1.0 - fy)),
            (x0 + 1, y0, fx * (1.0 - fy)),
            (x0, y0 + 1, (1.0 - fx) * fy),
            (x0 + 1, y0 + 1, fx * fy),
        ):
            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            safe_index = yi.clamp(0, height - 1) * width + xi.clamp(0, width - 1)
            flat.scatter_add_(
                0, safe_index, amplitude * weight * valid.to(weight.dtype)
            )

    image = flat.reshape(height, width)
    return image.reshape((1,) * (len(image_shape) - 2) + (height, width))
