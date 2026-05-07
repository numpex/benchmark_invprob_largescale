"""MissingWedge — deepinv LinearPhysics wrapping icecream's Fourier wedge mask.

Mask construction matches icecream's initialize_wedge exactly:
    - get_wedge_3d_new(mask_size) → shape (mask_size+1)³
    - symmetrize: (mask + flipped_mask) / 2
    - binarize:   values > 0.1 → 1
    - keep full (mask_size+1)³ as ``mask`` (icecream ``wedge_full``)

Forward operator A matches icecream's get_measurement exactly:
    1. fftn(x, s=mask.shape)  — zero-pad volume to mask.shape before FFT
    2. fftshift                — center DC before masking
    3. multiply by binary wedge mask
    4. ifftshift + ifftn.real  — back to real space at mask_size
    5. crop output to original x shape

Volumes are always cubic (centre-cropped to min-dim before reaching here):
    When wedge_double_size=True  (icecream default):
        mask_size = 2 * crop_size, and ``mask`` has shape (mask_size+1)³
    When wedge_double_size=False:
        mask_size = crop_size      →  no zero-padding, mask applied at native size

The operator is self-adjoint (A = Aᵀ) because the mask is real and binary.
"""
from __future__ import annotations

import numpy as np
import torch
import deepinv as dinv

from icecream_orig.utils.utils import get_wedge_3d_new, symmetrize_3D


class MissingWedge(dinv.physics.LinearPhysics):
    """Missing-wedge Fourier-mask operator for cryo-ET.

    Matches icecream's EquivariantTrainer exactly for cubic volumes.
    When ``volume_shape`` is provided (non-cubic volumes), the wedge is built
    at ``max(volume_shape)`` cubic and then **cropped** to the actual volume
    dimensions.  Cropping the centred Fourier grid is physically correct: it
    retains the low-frequency region that corresponds to the spatial resolution
    of each axis.

    :param float tilt_max: Maximum tilt angle in degrees (default 60).
    :param float tilt_min: Minimum tilt angle in degrees (default -60).
    :param int crop_size: Cubic patch side length — used for wedge_ref and window.
    :param tuple[int,int,int] | None volume_shape: Actual (D, H, W) of the
        volume the mask will be applied to.  ``None`` = cubic (legacy behaviour).
    :param bool use_spherical_support: Enforce spherical support in Fourier space (default True).
    :param bool wedge_double_size: Build mask at 2× the cubic side (icecream default True).
    :param float wedge_low_support: Radius² of a low-frequency ball forced to 1 in the input wedge
        (icecream ``wedge_low_support``). 0 = no leakage (icecream default); 0.1 = 10% low-freq kept.
    :param float ref_wedge_support: Same parameter for the reference wedge used in EqLoss
        (icecream ``ref_wedge_support``). 1.0 = full unit sphere set to 1 (icecream default),
        meaning the equivariance reference sees the complete spectrum.
    :param str device: Device string (default 'cpu').
    """

    def __init__(
        self,
        tilt_max: float = 60.0,
        tilt_min: float = -60.0,
        crop_size: int = 72,
        volume_shape: tuple[int, int, int] | None = None,
        use_spherical_support: bool = True,
        wedge_double_size: bool = True,
        wedge_low_support: float = 0.0,
        ref_wedge_support: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        # ── Resolve effective spatial dimensions ─────────────────────────
        if volume_shape is not None:
            D, H, W = int(volume_shape[0]), int(volume_shape[1]), int(volume_shape[2])
        else:
            D = H = W = crop_size  # cubic — legacy behaviour

        self._volume_shape = (D, H, W)

        # ── Build wedge mask ─────────────────────────────────────────────
        # get_wedge_3d_new only works correctly for cubic inputs (internal
        # broadcast assumes equal axes). Build cubic at max_dim, then crop
        # the centred Fourier grid to the actual (D+1, H+1, W+1) shape.
        max_dim = max(D, H, W)
        mask_size = max_dim * 2 if wedge_double_size else max_dim

        wedge_np, _ = get_wedge_3d_new(
            mask_size,
            tilt_max,
            tilt_min,
            low_support=wedge_low_support,
            use_spherical_support=use_spherical_support,
        )
        mask = torch.from_numpy(wedge_np.astype(np.float32))  # (mask_size+1,)*3

        # symmetrize + binarize — matches icecream's get_real_binary_filter
        mask_sym = symmetrize_3D(mask)
        mask = (mask + mask_sym) / 2
        mask[mask > 0.1] = 1.0

        # Crop centred to (D+1, H+1, W+1). No-op for cubic D==H==W==max_dim.
        full_side = mask_size + 1
        dD = (full_side - (D + 1)) // 2
        dH = (full_side - (H + 1)) // 2
        dW = (full_side - (W + 1)) // 2
        mask = mask[dD: dD + D + 1, dH: dH + H + 1, dW: dW + W + 1]  # (D+1, H+1, W+1)

        # Stored at (D+1, H+1, W+1) — analogous to icecream's wedge_full.
        self.register_buffer("mask", mask)

        # wedge_ref: native resolution (no doubling — ref is never doubled in icecream).
        # Build cubic at max_dim and crop to (D+1, H+1, W+1), then [:-1,:-1,:-1] → (D,H,W).
        wedge_ref_np, _ = get_wedge_3d_new(
            max_dim,
            tilt_max,
            tilt_min,
            low_support=ref_wedge_support,
            use_spherical_support=use_spherical_support,
        )
        mask_ref = torch.from_numpy(wedge_ref_np.astype(np.float32))  # (max_dim+1,)*3
        mask_ref_sym = symmetrize_3D(mask_ref)
        mask_ref = (mask_ref + mask_ref_sym) / 2
        mask_ref[mask_ref > 0.1] = 1.0
        full_side_ref = max_dim + 1
        dD_r = (full_side_ref - (D + 1)) // 2
        dH_r = (full_side_ref - (H + 1)) // 2
        dW_r = (full_side_ref - (W + 1)) // 2
        mask_ref = mask_ref[dD_r: dD_r + D + 1, dH_r: dH_r + H + 1, dW_r: dW_r + W + 1]
        # [:-1,:-1,:-1] → (D, H, W)
        self.register_buffer("mask_ref", mask_ref[:-1, :-1, :-1])

    # ------------------------------------------------------------------
    # deepinv LinearPhysics interface
    # ------------------------------------------------------------------

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply missing-wedge mask in Fourier space.

        Matches icecream's ``get_measurement``:
          - zero-pads x to mask shape via ``fftn(s=mask.shape)``
          - centers DC with fftshift before masking
          - crops output back to original x spatial shape

        :param torch.Tensor x: Input tensor of shape (B, C, D, H, W).
        :return: Wedge-masked volume, same shape as x.
        """
        # Remember original spatial size for output crop
        D, H, W = x.shape[-3], x.shape[-2], x.shape[-1]
        mask_dims = tuple(self.mask.shape)  # (mask_size+1,)*3

        # Zero-pad to mask shape, shift DC to center, apply mask
        X = torch.fft.fftshift(
            torch.fft.fftn(x, s=mask_dims, dim=(-3, -2, -1)),
            dim=(-3, -2, -1),
        )
        # mask may have shape (mask_size+1,)³ — broadcast over batch/channel dims
        X_masked = X * self.mask

        # Back to real space, then crop to original volume size
        out = torch.fft.ifftn(
            torch.fft.ifftshift(X_masked, dim=(-3, -2, -1)),
            dim=(-3, -2, -1),
        ).real
        return out[..., :D, :H, :W]

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Adjoint = A (self-adjoint because mask is real and binary)."""
        return self.A(y)
