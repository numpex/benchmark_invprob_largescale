"""FITS image helpers shared by radio training and data preparation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from astropy.io import fits


def get_fits_image_size(image_path: str | Path) -> int:
    """Return the native spatial size of a square FITS image."""
    with fits.open(image_path, memmap=False) as hdul:
        shape = tuple(size for size in hdul[0].shape if size != 1)

    if len(shape) not in (2, 3):
        raise ValueError(
            f"Unexpected FITS image shape {shape}; expected two spatial dimensions "
            "and, optionally, a channel dimension."
        )
    height, width = shape[-2:]
    if height != width:
        raise ValueError(
            "Radio interferometry currently requires a square FITS image, got "
            f"spatial shape {(height, width)} from {image_path}."
        )
    return int(height)


def load_fits_image(image_path: str | Path, normalize: bool = False) -> np.ndarray:
    """Load a FITS image as a contiguous float32 ``(C, H, W)`` array."""
    with fits.open(image_path, memmap=False) as hdul:
        image = np.array(hdul[0].data, dtype=np.float32, copy=True)

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    max_value = float(np.max(image))
    if normalize and max_value > 1.0:
        image = image / max_value

    image = np.squeeze(image)
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    elif image.ndim != 3:
        raise ValueError(
            f"Unexpected FITS image shape {image.shape}, expected 2D or 3D after squeeze."
        )

    image_size = get_fits_image_size(image_path)
    if image.shape[-2:] != (image_size, image_size):
        raise ValueError(
            f"FITS data shape {image.shape} does not match its native spatial size "
            f"{image_size}."
        )
    return np.ascontiguousarray(image, dtype=np.float32)


def get_cellsize_from_fits_wcs(fits_file: str | Path) -> float:
    """Return pixel angular size in radians per pixel from FITS WCS."""
    header = fits.getheader(fits_file)
    cdelt1 = header.get("CDELT1")
    cdelt2 = header.get("CDELT2")
    if cdelt1 is None and cdelt2 is None:
        raise ValueError("FITS header has no CDELT1/CDELT2")
    if cdelt1 is not None and cdelt2 is not None:
        pixel_scale_deg = 0.5 * (abs(float(cdelt1)) + abs(float(cdelt2)))
    else:
        pixel_scale_deg = abs(float(cdelt1 if cdelt1 is not None else cdelt2))
    return math.radians(pixel_scale_deg)
