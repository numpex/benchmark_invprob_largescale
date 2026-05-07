"""Icecream-style losses for the cryo-ET equivariant trainer.

Implements the two loss terms from icecream's EquivariantTrainer.compute_loss:

1. ``ObsLoss`` — Cross half-set data-fidelity in the Fourier domain
       L_obs = fourier_loss(EVN, f(ODD), W) + fourier_loss(ODD, f(EVN), W)
   If only one half-set is available (paired=False), reduces to:
       L_obs = fourier_loss(y, f(y), W)   (self-consistent, single half)

2. ``EqLoss`` — Equivariance under random cube-symmetry rotations,
   measured in the Fourier domain under the *rotated* wedge, matching icecream's
   fourier_loss_batch variant.

Both are ``deepinv.loss.Loss`` subclasses so they plug straight into the deepinv
Trainer ``losses`` list.

The ``fourier_loss`` formula (icecream default, ``use_fourier=False, view_as_real=True``):
    FFT both target and estimate to ``mask_size`` (zero-pad),
    apply the binary wedge mask in centred Fourier space,
    IFFT back to real space (take ``.real``),
    crop back to ``crop_size³``,
    compute plain MSE — **no** ``1/sqrt(N³)`` normalisation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from deepinv.loss import Loss

from .transform import Rotate3D
from icecream_orig.utils.utils import (
    fourier_loss as _ic_fourier_loss,
    fourier_loss_batch as _ic_fourier_loss_batch,
    get_measurement as _ic_get_measurement,
)


def _initialize_window(shape: int | tuple[int, int, int]) -> torch.Tensor:
    """Build the 3-D box window used in icecream's ``initialize_window``.

    Inner region ``[s//4 : -s//4]`` along each axis = 1, rest = 0.
    Accepts either a single int (cubic) or a (D, H, W) tuple.
    """
    if isinstance(shape, int):
        shape = (shape, shape, shape)
    D, H, W = shape
    w = np.zeros((D, H, W), dtype=np.float32)
    qd, qh, qw = D // 4, H // 4, W // 4
    w[qd:-qd, qh:-qh, qw:-qw] = 1.0
    return torch.from_numpy(w)


# ---------------------------------------------------------------------------
# Low-level Fourier loss helpers  (exact icecream clones)
# ---------------------------------------------------------------------------

def _fourier_loss(
    target: torch.Tensor,
    estimate: torch.Tensor,
    wedge: torch.Tensor,
    criteria: nn.Module,
    window: torch.Tensor | None = None,
    use_fourier: bool = False,
    view_as_real: bool = True,
) -> torch.Tensor:
    """Delegates to icecream_orig fourier_loss.

    Handles (B, C, D, H, W) by reshaping to (B*C, D, H, W) before the call.
    """
    t = target.reshape(-1, *target.shape[-3:])
    e = estimate.reshape(-1, *estimate.shape[-3:])
    return _ic_fourier_loss(t, e, wedge, criteria, use_fourier=use_fourier, view_as_real=view_as_real, window=window)


def _fourier_loss_batch(
    target: torch.Tensor,
    estimate: torch.Tensor,
    wedge: torch.Tensor,       # (B, M, M, M)  — per-sample rotated wedge
    criteria: nn.Module,
    window: torch.Tensor | None = None,
    use_fourier: bool = False,
    view_as_real: bool = True,
) -> torch.Tensor:
    """Delegates to icecream_orig fourier_loss_batch.

    Handles (B, C, D, H, W) by reshaping to (B*C, D, H, W) before the call.
    """
    t = target.reshape(-1, *target.shape[-3:])
    e = estimate.reshape(-1, *estimate.shape[-3:])
    return _ic_fourier_loss_batch(t, e, wedge, criteria, use_fourier=use_fourier, view_as_real=view_as_real, window=window)


# ---------------------------------------------------------------------------
# Helper: apply wedge mask (icecream's get_measurement)
# ---------------------------------------------------------------------------

def _apply_wedge(x: torch.Tensor, wedge: torch.Tensor) -> torch.Tensor:
    """Delegates to icecream_orig get_measurement.

    Handles (B, C, D, H, W) by reshaping to (B*C, D, H, W) and restoring shape.

    :param x:     (B, [C,] D, H, W)
    :param wedge: (M, M, M)
    :return: same shape as x
    """
    shape = x.shape
    x_4d = x.reshape(-1, *shape[-3:])
    out = _ic_get_measurement(x_4d, wedge)
    return out.reshape(*shape[:-3], *out.shape[-3:])


# ---------------------------------------------------------------------------
# Helper: rotate a binary wedge mask
# ---------------------------------------------------------------------------

def _rotate_wedge(wedge: torch.Tensor, kx: int, ky: int, kz: int, axis: int) -> torch.Tensor:
    """Apply the same (kx,ky,kz,axis) rotation used in batch_rot_4vol to a 3D wedge.

    :param wedge: (M, M, M)
    :return: (M, M, M) rotated wedge
    """
    w = wedge.unsqueeze(0)          # (1, M, M, M)
    w = torch.rot90(w, k=kx, dims=(1, 2))
    w = torch.rot90(w, k=ky, dims=(1, 3))
    w = torch.rot90(w, k=kz, dims=(2, 3))
    if axis != -1:
        w = torch.flip(w, [axis + 1])
    return w.squeeze(0)             # (M, M, M)


# ---------------------------------------------------------------------------
# 1. Data-fidelity loss  (obs_loss in icecream)
# ---------------------------------------------------------------------------

class ObsLoss(Loss):
    """Cross half-set data-fidelity loss in the Fourier domain.

    **Paired mode** (EVN + ODD available, ``x ≠ y``):
        L = fourier_loss(ODD, A(f(EVN)), W) + fourier_loss(EVN, A(f(ODD)), W)

    **Single half-set mode** (``x == y`` or only EVN):
        L = fourier_loss(y, A(f(y)), W)

    The wedge mask ``W`` used here is the *input* wedge (``wedge_input``,
    cropped to ``crop_size³``, same as icecream's ``wedge_input_set[idx]``).

    :param MissingWedge physics: the MissingWedge physics object (provides ``mask``).
    :param float weight: loss weight (default 1.0).
    :param bool paired: if True always use cross half-set formula;
        if False always use single half-set.  Default ``None`` = auto-detect
        by checking ``torch.equal(x, y)`` each forward call.
    """

    def __init__(self, physics, weight: float = 1.0, paired: bool | None = None,
                 use_fourier: bool = False, view_as_real: bool = True) -> None:
        super().__init__()
        self.weight      = weight
        self.paired      = paired
        self.use_fourier = use_fourier
        self.view_as_real = view_as_real
        self._physics = physics          # kept as plain ref, not nn.Module child
        self._criteria = nn.MSELoss(reduction="mean")

    @property
    def _wedge_input(self) -> torch.Tensor:
        """(mask_size)³ wedge — matches icecream's ``wedge_full[:-1,:-1,:-1]``."""
        return self._physics.mask[:-1, :-1, :-1]

    @property
    def _window(self) -> torch.Tensor:
        """Box window matching the actual volume shape."""
        return _initialize_window(self._physics._volume_shape)

    def forward(
        self,
        x: torch.Tensor,        # EVN patch  (B, 1, D, H, W)
        y: torch.Tensor,        # ODD patch  (same shape)
        x_net: torch.Tensor,    # f(x) = f(EVN)
        physics,                # unused — wedge comes from self._physics
        model,                  
        **kwargs,
    ) -> torch.Tensor:
        wedge = self._wedge_input.to(x.device)

        # Determine whether we have a genuine paired batch
        is_paired = (self.paired is True) or (
            self.paired is None and not torch.equal(x, y)
        )

        window = self._window.to(x.device)

        if is_paired:
            # f(EVN) is already computed as x_net;  compute f(ODD) separately
            est_evn = x_net                          # f(EVN),  (B,1,D,H,W)
            est_odd = model(y)                       # f(ODD)

            # Pass estimates raw — fourier_loss applies wedge internally (matches icecream)
            loss = (
                _fourier_loss(y, est_evn, wedge, self._criteria, window=window,
                              use_fourier=self.use_fourier, view_as_real=self.view_as_real)
                + _fourier_loss(x, est_odd, wedge, self._criteria, window=window,
                                use_fourier=self.use_fourier, view_as_real=self.view_as_real)
            )
        else:
            # Single half-set: y == x (EVN only)
            loss = _fourier_loss(y, x_net, wedge, self._criteria, window=window,
                                 use_fourier=self.use_fourier, view_as_real=self.view_as_real)

        return self.weight * loss


# ---------------------------------------------------------------------------
# 2. Equivariance loss  (equi_loss_est in icecream)
# ---------------------------------------------------------------------------

class EqLoss(Loss):
    """Equivariance loss under random cube-symmetry rotation, Fourier domain.

    For a randomly sampled rotation T (from the 40-element cubic group):

        est_rot      = T(f(y))           # rotate the network estimate
        wedge_rot    = T(W_full)         # rotate the full-size wedge mask
        ref          = A_ref(est_rot)    # apply crop_size wedge to rotated estimate
        remeasured   = f(A_input(est_rot))  # re-denoise the re-measured rotated estimate

    Loss:
        L_eq = fourier_loss_batch(ref, remeasured, wedge_rot)

    **Paired mode**: both EVN and ODD estimates are rotated and the loss is
    summed symmetrically.  **Single mode**: only ``x_net = f(y)`` is used.

    ``W_full``   = physics.mask at full mask_size (the double-size mask)
    ``A_ref``    = apply crop_size wedge  (``wedge_ref``)
    ``A_input``  = apply crop_size wedge  (``wedge_input``)

    Note: icecream uses the *same* crop_size mask for both ``wedge_ref`` and
    ``wedge_input`` in the equivariance branch (they differ only in whether
    spherical support is used; here we keep them identical for simplicity).

    :param MissingWedge physics: provides mask buffers.
    :param Rotate3D transform: the 3D rotation transform (for sampling k_idx).
    :param float weight: loss weight (default 2.0).
    :param bool paired: same semantics as ObsLoss.
    """

    def __init__(
        self,
        physics,
        transform: Rotate3D,
        weight: float = 2.0,
        paired: bool | None = None,
        min_distance: float = 0.5,
        use_fourier: bool = False,
        view_as_real: bool = True,
        eq_use_direct: bool = False,
    ) -> None:
        super().__init__()
        self.weight        = weight
        self.paired        = paired
        self.use_fourier   = use_fourier
        self.eq_use_direct = eq_use_direct
        self.view_as_real = view_as_real
        self._physics  = physics
        self._transform = transform
        self._criteria  = nn.MSELoss(reduction="mean")
        self._valid_k_sets = self._compute_valid_k_sets(min_distance)

    def _compute_valid_k_sets(self, min_distance: float) -> list[int]:
        """Return indices into ``Rotate3D._KSET`` where the rotated wedge differs
        from the original by ``distance > min_distance``.

        Mirrors icecream's ``generate_all_cube_symmetries_torch`` filter.
        Volumes are always cubic so all 40 rotations are shape-preserving.
        """
        wedge = self._physics.mask[:-1, :-1, :-1].float()
        norm_w = torch.linalg.norm(wedge)
        valid = []
        for i, (kx, ky, kz, axis) in enumerate(Rotate3D._KSET):
            w_rot = _rotate_wedge(wedge, kx, ky, kz, axis)
            dist = torch.linalg.norm(w_rot - wedge) / norm_w
            if dist.item() > min_distance:
                valid.append(i)
        return valid if valid else list(range(len(Rotate3D._KSET)))

    @property
    def _wedge_full(self) -> torch.Tensor:
        """Full (mask_size)³ wedge — used for rotation."""
        return self._physics.mask

    @property
    def _wedge_input(self) -> torch.Tensor:
        """(mask_size)³ wedge (``wedge_full[:-1,:-1,:-1]``) — used for A_input and fourier_loss_batch."""
        return self._physics.mask[:-1, :-1, :-1]

    @property
    def _wedge_ref(self) -> torch.Tensor:
        """(D,H,W) wedge at native volume resolution — matches icecream's ``wedge_ref`` for A_ref."""
        return self._physics.mask_ref

    @property
    def _window(self) -> torch.Tensor:
        """Box window matching the actual volume shape."""
        return _initialize_window(self._physics._volume_shape)

    def _rotate_batch_per_sample(self, x: torch.Tensor, k_indices: torch.Tensor) -> torch.Tensor:
        """Rotate each sample in a batch with its own k-index (icecream batch_rot behavior)."""
        out = []
        for i in range(x.shape[0]):
            out.append(self._transform.transform(x[i:i + 1], k_idx=int(k_indices[i].item())))
        return torch.cat(out, dim=0)

    def _rotate_wedge_batch(
        self,
        wedge_full: torch.Tensor,
        k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate full wedge per sample, then crop + binarize (icecream get_real_binary_filters_batch path)."""
        w_batch = []
        for i in range(k_indices.shape[0]):
            kx, ky, kz, axis = Rotate3D._KSET[int(k_indices[i].item())]
            w_rot = _rotate_wedge(wedge_full, kx, ky, kz, axis)
            w_rot = w_rot[:-1, :-1, :-1]
            w_rot = (w_rot > 0.1).float()
            w_batch.append(w_rot)
        return torch.stack(w_batch, dim=0)

    def _one_sided_eq_loss(
        self,
        est: torch.Tensor,        # (B, 1, D, H, W)  network output
        model: nn.Module,
        wedge_full: torch.Tensor,
        wedge_ref: torch.Tensor,
        wedge_input: torch.Tensor,
        window: torch.Tensor,
        k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute single-branch equivariance term with per-sample rotations.

        This is used for EVN-only fallback where x==y.
        """
        est_rot = self._rotate_batch_per_sample(est, k_indices)
        w_rot_batch = self._rotate_wedge_batch(wedge_full, k_indices)

        ref = _apply_wedge(est_rot, wedge_ref)                   # A_ref(T(x̂))

        remeasured_inp = _apply_wedge(est_rot, wedge_input)      # A_input(T(x̂))
        remeasured     = model(remeasured_inp)                    # f(A_input(T(x̂)))

        if self.eq_use_direct:
            # Plain spatial MSE, no Fourier masking (icecream eq_use_direct=True branch)
            return self._criteria(ref, remeasured)
        return _fourier_loss_batch(ref, remeasured, w_rot_batch, self._criteria, window=window,
                                   use_fourier=self.use_fourier, view_as_real=self.view_as_real)

    def _paired_eq_loss(
        self,
        est_1: torch.Tensor,
        est_2: torch.Tensor,
        model: nn.Module,
        wedge_full: torch.Tensor,
        wedge_ref: torch.Tensor,
        wedge_input: torch.Tensor,
        window: torch.Tensor,
        k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Icecream-equivalent paired equivariant term (cross-coupled).

        Matches eq_trainer compute_loss:
          est_1_rot, est_2_rot = batch_rot_4vol(...)
          wedge_rot = rotated_wedge[:-1,:-1,:-1]
          est_1_ref = A_ref(est_1_rot), est_2_ref = A_ref(est_2_rot)
          est_1_rot_est = f(A_input(est_1_rot)), est_2_rot_est = f(A_input(est_2_rot))
          L_eq = fourier_loss_batch(est_2_ref, est_1_rot_est, wedge_rot)
               + fourier_loss_batch(est_1_ref, est_2_rot_est, wedge_rot)
        """
        est_1_rot = self._rotate_batch_per_sample(est_1, k_indices)
        est_2_rot = self._rotate_batch_per_sample(est_2, k_indices)
        wedge_rot_batch = self._rotate_wedge_batch(wedge_full, k_indices)

        est_1_ref = _apply_wedge(est_1_rot, wedge_ref)
        est_2_ref = _apply_wedge(est_2_rot, wedge_ref)

        est_1_rot_inp = _apply_wedge(est_1_rot, wedge_input)
        est_2_rot_inp = _apply_wedge(est_2_rot, wedge_input)

        est_1_rot_est = model(est_1_rot_inp)
        est_2_rot_est = model(est_2_rot_inp)

        if self.eq_use_direct:
            # Plain spatial MSE, no Fourier masking (icecream eq_use_direct=True branch)
            return self._criteria(est_2_ref, est_1_rot_est) + self._criteria(est_1_ref, est_2_rot_est)
        return (
            _fourier_loss_batch(est_2_ref, est_1_rot_est, wedge_rot_batch, self._criteria, window=window,
                                use_fourier=self.use_fourier, view_as_real=self.view_as_real)
            + _fourier_loss_batch(est_1_ref, est_2_rot_est, wedge_rot_batch, self._criteria, window=window,
                                  use_fourier=self.use_fourier, view_as_real=self.view_as_real)
        )

    def forward(
        self,
        x: torch.Tensor,        # EVN patch
        y: torch.Tensor,        # ODD patch (== x if single half-set)
        x_net: torch.Tensor,    # f(x) = f(EVN)
        physics,
        model: nn.Module,
        **kwargs,
    ) -> torch.Tensor:
        wedge_full  = self._wedge_full.to(x.device)
        wedge_ref   = self._wedge_ref.to(x.device)
        wedge_input = self._wedge_input.to(x.device)
        window      = self._window.to(x.device)

        # Sample one valid rotation index per sample (icecream batch_rot_* behavior).
        pool = self._valid_k_sets
        bsz = x.shape[0]
        rand_idx = torch.randint(len(pool), (bsz,), device=x.device)
        k_indices = torch.tensor([pool[int(i.item())] for i in rand_idx], device=x.device)

        is_paired = (self.paired is True) or (
            self.paired is None and not torch.equal(x, y)
        )

        if is_paired:
            est_odd = model(y)   # f(ODD)  — needed for second symmetric term
            loss = self._paired_eq_loss(
                x_net,
                est_odd,
                model,
                wedge_full,
                wedge_ref,
                wedge_input,
                window,
                k_indices,
            )
        else:
            # EVN-only fallback: keep training functional without paired half-set.
            loss = self._one_sided_eq_loss(
                x_net,
                model,
                wedge_full,
                wedge_ref,
                wedge_input,
                window,
                k_indices,
            )

        return self.weight * loss
