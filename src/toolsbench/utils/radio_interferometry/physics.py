from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from astropy import constants as const
from deepinv.physics import LinearPhysics, stack
from deepinv.utils import TensorList
import pytorch_finufft as py_nufft


def _identity(x: torch.Tensor, **_kwargs) -> torch.Tensor:
    return x


def _finufft_options(upsampfac: float, use_cuda: bool) -> dict[str, float | int]:
    """Return compatible FINUFFT options for the requested fine-grid factor."""
    upsampfac = float(upsampfac)
    if upsampfac <= 1.0:
        raise ValueError("nufft_k_oversampling must be greater than 1.")
    options: dict[str, float | int] = {"upsampfac": upsampfac}
    standard_horner_factors = (1.25, 2.0)
    if use_cuda and not any(
        math.isclose(upsampfac, factor, rel_tol=0.0, abs_tol=1e-12)
        for factor in standard_horner_factors
    ):
        # cuFINUFFT's default GPU Horner evaluator only has polynomial rules
        # for its standard factors. Direct evaluation supports arbitrary
        # factors, including the radio pipeline's configured 1.5.
        options["gpu_kerevalmeth"] = 0
    return options


def _balanced_plane_assignment(
    active_plane_ids: Sequence[int],
    counts: torch.Tensor,
    world_size: int,
) -> list[list[tuple[int, int]]]:
    """Balance plane count first, then greedily balance visibility count.

    Each returned pair is ``(global_index, plane_id)``. Keeping the number of
    planes per rank within one balances fixed phase-screen memory and NUFFT
    cost, while placing the most populated planes first balances point-dependent
    work. Deterministic tie-breaking lets every rank derive the same ownership
    without communication.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive.")

    num_planes = len(active_plane_ids)
    planes_per_rank, remainder = divmod(num_planes, world_size)
    capacities = [planes_per_rank + int(rank < remainder) for rank in range(world_size)]
    assignments: list[list[tuple[int, int]]] = [[] for _ in range(world_size)]
    visibility_loads = [0] * world_size

    jobs = [
        (global_index, int(plane_id), int(counts[plane_id].item()))
        for global_index, plane_id in enumerate(active_plane_ids)
    ]
    jobs.sort(key=lambda job: (-job[2], job[0]))

    for global_index, plane_id, visibility_count in jobs:
        eligible_ranks = [
            rank
            for rank in range(world_size)
            if len(assignments[rank]) < capacities[rank]
        ]
        rank = min(
            eligible_ranks,
            key=lambda candidate: (
                visibility_loads[candidate],
                len(assignments[candidate]),
                candidate,
            ),
        )
        assignments[rank].append((global_index, plane_id))
        visibility_loads[rank] += visibility_count

    for assignment in assignments:
        assignment.sort(key=lambda pair: pair[0])
    return assignments


def _equal_count_plane_indices(
    w_lambda: torch.Tensor,
    w_planes: int,
) -> list[torch.Tensor]:
    """Return contiguous-in-w plane indices with near-equal populations.

    Points are sorted by their w coordinate and the sorted index sequence is
    split into ``w_planes`` consecutive chunks. Plane populations therefore
    differ by at most one, and no two planes interleave in w.
    """
    if int(w_planes) <= 0:
        raise ValueError("w_planes must be positive.")
    if w_lambda.ndim != 1:
        raise ValueError("w_lambda must be one-dimensional.")
    if w_lambda.numel() == 0:
        return []

    sorted_indices = torch.argsort(w_lambda)
    return [
        indices
        for indices in torch.tensor_split(sorted_indices, int(w_planes))
        if indices.numel() > 0
    ]


class IdentityNoise(torch.nn.Module):
    def forward(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        return x


@dataclass
class DirtyImagerConfig:
    imaging_npixel: int
    imaging_cellsize: float
    binning_factor: float = 1.25
    nufft_k_oversampling: float = 1.5
    combine_across_frequencies: bool = True
    weighting_chunk_points: int | None = 262144
    w_phase_row_chunk_size: int = 256


@dataclass
class RadioPhysicsResult:
    """Physics plus the metadata needed to adopt a rank-local shard."""

    physics: LinearPhysics | list[LinearPhysics] | None
    measurements: torch.Tensor | TensorList | None
    weights: torch.Tensor | None
    global_indices: list[int] | None
    num_operators: int
    global_measurement_count: int
    from_shard: bool = False


class MyRadioInterferometry(LinearPhysics):
    """Benchmark-compatible radio interferometry operator.

    The NUFFT is intentionally called with ``norm=None`` to preserve the
    benchmark radio scaling. ``dataWeight`` is the square root of the
    statistical visibility weight, so this operator represents
    ``sqrt(W) F`` and its normal operator is ``F* W F``.
    """

    def __init__(
        self,
        img_size: tuple[int, int] | torch.Tensor,
        samples_loc: torch.Tensor,
        dataWeight: torch.Tensor | None = None,
        image_phase: torch.Tensor | None = None,
        real_projection: bool = True,
        nufft_k_oversampling: float = 1.5,
        device: torch.device | str = "cpu",
        **kwargs,
    ) -> None:

        super().__init__(
            A=_identity,
            A_adjoint=_identity,
            noise_model=IdentityNoise(),
            sensor_model=_identity,
            device=device,
            **kwargs,
        )

        if isinstance(img_size, torch.Tensor):
            img_size = tuple(int(v) for v in img_size.detach().cpu().tolist())
        self.img_size = tuple(int(v) for v in img_size)
        if len(self.img_size) != 2:
            raise ValueError(f"Expected 2-D image size, got {self.img_size}.")

        if dataWeight is None:
            dataWeight = torch.tensor([1.0], device=device)

        self.real_projection = bool(real_projection)
        self.nufft_k_oversampling = float(nufft_k_oversampling)
        _finufft_options(self.nufft_k_oversampling, use_cuda=False)

        self.register_buffer("samples_loc", samples_loc.to(device))
        self.register_buffer("dataWeight", dataWeight.to(device))
        if image_phase is None:
            image_phase = torch.ones(self.img_size, dtype=torch.cfloat, device=device)
        if tuple(image_phase.shape) != self.img_size:
            raise ValueError(
                "image_phase must have the same spatial shape as img_size; got "
                f"{tuple(image_phase.shape)} and {self.img_size}."
            )
        # A w-plane applies conj(image_phase) before the Fourier transform and
        # image_phase after its adjoint.  Keeping it in the elementary operator
        # makes each plane a valid LinearPhysics instance for deepinv.stack.
        self.register_buffer(
            "image_phase", image_phase.to(device=device, dtype=torch.cfloat)
        )

        # FITS grids center an even-sized image between pixels (index
        # (N-1)/2), while finufft's mode ordering puts the phase-center pixel
        # at index N//2. Empirically measured via a residual-power sweep
        # against real MS data: a +0.5 pixel shift on both axes aligns
        # FITS-grid images with the operator's native grid.
        center_phase = torch.exp(
            -0.5j * (self.samples_loc[0] + self.samples_loc[1])
        ).to(torch.cfloat)
        self.register_buffer("center_phase", center_phase)

        self.to(device)

    def adj_projection(self, x: torch.Tensor) -> torch.Tensor:
        return torch.real(x).float() if self.real_projection else x

    def setWeight(self, w: torch.Tensor) -> None:
        self.dataWeight = w.to(self.dataWeight)

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        vis = py_nufft.functional.finufft_type2(
            self.samples_loc,
            x.to(torch.cfloat) * torch.conj(self.image_phase),
            modeord=0,
            isign=-1,
            **_finufft_options(
                self.nufft_k_oversampling,
                use_cuda=self.samples_loc.device.type == "cuda",
            ),
        )
        return vis * self.dataWeight * self.center_phase

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        y = y * self.dataWeight * torch.conj(self.center_phase)
        image = py_nufft.functional.finufft_type1(
            self.samples_loc,
            y,
            self.img_size,
            modeord=0,
            isign=1,
            **_finufft_options(
                self.nufft_k_oversampling,
                use_cuda=self.samples_loc.device.type == "cuda",
            ),
        )
        return self.adj_projection(image * self.image_phase)


class DeepinvDirtyImager(torch.nn.Module):
    """Dirty imager used by the benchmark radio dataset."""

    def __init__(
        self,
        config: DirtyImagerConfig,
        device: torch.device = torch.device("cpu"),
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.verbose = int(verbose)

    def to_device(
        self,
        tensor: torch.Tensor,
        non_blocking: bool = True,
        pin_memory: bool = False,
    ) -> torch.Tensor:
        if self.device.type == "cuda":
            if pin_memory and tensor.device.type == "cpu":
                tensor = tensor.pin_memory()
            return tensor.to(self.device, non_blocking=non_blocking)
        return tensor.to(self.device)

    def load_visibilities(
        self,
        visibility_path: str | Path,
        visibility_format: str = "MS",
        visibility_column: str = "DATA",
        chunk_rows: int | None = 65536,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from casacore.tables import table

        if visibility_format != "MS":
            raise NotImplementedError(
                f"Visibility format {visibility_format} not supported, only MS is supported."
            )
        if chunk_rows is not None and int(chunk_rows) <= 0:
            raise ValueError("visibility_chunk_rows must be positive or null.")

        visibility_path = str(visibility_path)
        with table(visibility_path + "/SPECTRAL_WINDOW", readonly=True) as tb:
            chan_freqs_np = tb.getcol("CHAN_FREQ")[0].astype(np.float32, copy=False)
        n_freq = int(len(chan_freqs_np))

        with table(visibility_path, readonly=True) as tb:
            n_rows = int(tb.nrows())
            first_visibility = tb.getcell(visibility_column, 0)
            if first_visibility.ndim != 2 or first_visibility.shape[0] != n_freq:
                raise ValueError(
                    f"Unexpected {visibility_column} cell shape "
                    f"{first_visibility.shape}; expected ({n_freq}, n_correlations)."
                )
            n_correlations = int(first_visibility.shape[1])
            if n_correlations not in {1, 4}:
                raise ValueError(
                    f"Expected one or four correlations, got {n_correlations}."
                )

            uvw_np = np.empty((n_rows, 3), dtype=np.float32)
            visibilities_np = np.empty((n_rows, n_freq), dtype=np.complex64)
            rows_per_chunk = n_rows if chunk_rows is None else int(chunk_rows)
            for start_row in range(0, n_rows, rows_per_chunk):
                n_chunk = min(rows_per_chunk, n_rows - start_row)
                row_slice = slice(start_row, start_row + n_chunk)

                uvw_chunk = tb.getcol("UVW", startrow=start_row, nrow=n_chunk)
                np.copyto(uvw_np[row_slice], uvw_chunk, casting="unsafe")
                del uvw_chunk

                if n_correlations == 4:
                    visibility_chunk = tb.getcolslice(
                        visibility_column,
                        [0, 0],
                        [n_freq - 1, 3],
                        [1, 3],
                        startrow=start_row,
                        nrow=n_chunk,
                    )
                    target = visibilities_np[row_slice]
                    np.add(
                        visibility_chunk[..., 0],
                        visibility_chunk[..., 1],
                        out=target,
                    )
                    target *= 0.5
                else:
                    visibility_chunk = tb.getcol(
                        visibility_column,
                        startrow=start_row,
                        nrow=n_chunk,
                    )
                    np.copyto(
                        visibilities_np[row_slice],
                        visibility_chunk[..., 0],
                        casting="unsafe",
                    )
                del visibility_chunk

            if self.verbose:
                print(
                    f"Data loaded: {n_rows} rows x {n_freq} channels "
                    f"in chunks of {rows_per_chunk}",
                    flush=True,
                )
                print(f"Available columns: {tb.colnames()}", flush=True)

        if self.verbose:
            print(f"Number of channels: {n_freq}", flush=True)

        uvw = self.to_device(torch.from_numpy(uvw_np))
        visibilities = self.to_device(torch.from_numpy(visibilities_np))
        freqs = self.to_device(torch.from_numpy(chan_freqs_np))
        return uvw, visibilities, freqs

    def normalize_uv_coords(
        self,
        uvw: torch.Tensor,
        freqs: torch.Tensor,
        visibilities: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if visibilities.ndim == 3 and visibilities.shape[2] == 4:
            visibilities = 0.5 * (visibilities[:, :, 0] + visibilities[:, :, 3])
        elif visibilities.ndim == 3 and visibilities.shape[2] == 1:
            visibilities = visibilities[:, :, 0]
        elif visibilities.ndim != 2:
            raise ValueError(
                f"Expected visibility shape (rows, channels), got "
                f"{tuple(visibilities.shape)}."
            )

        n_vis, n_freq = visibilities.shape

        # Pre-allocation for better performance
        total_points = n_vis * n_freq
        samples_locs = torch.zeros(
            (2, total_points), dtype=torch.float32, device=self.device
        )
        all_w_lambda = torch.zeros(
            total_points, dtype=torch.float32, device=self.device
        )
        all_visibilities = torch.zeros(
            total_points, dtype=torch.complex64, device=self.device
        )

        # Vectorized calculations
        cellsize_2pi = self.config.imaging_cellsize * 2 * np.pi

        # Processing by frequency (more memory efficient)
        for i, freq in enumerate(freqs):
            start_idx = i * n_vis
            end_idx = (i + 1) * n_vis

            # Vectorized calculation of normalized UV coordinates
            wavelength = const.c.value / freq
            uvw_lambda = uvw / wavelength
            uvw_norm = (uvw_lambda * cellsize_2pi).T
            uvw_norm = torch.stack((-uvw_norm[1], uvw_norm[0]), dim=0)

            samples_locs[:, start_idx:end_idx] = uvw_norm
            all_w_lambda[start_idx:end_idx] = uvw_lambda[:, 2]
            all_visibilities[start_idx:end_idx] = visibilities[:, i]

        # Reshape for compatibility with rest of code
        visibilities_reshaped = all_visibilities.unsqueeze(0).unsqueeze(0)

        return samples_locs, all_w_lambda, visibilities_reshaped

    def uniform_weighting(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        im_size: torch.Tensor,
        weight_gridsize: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Strict uniform weighting with bounded point-chunk temporaries."""

        print(
            u.shape,
            v.shape,
            im_size,
            weight_gridsize,
            self.config.weighting_chunk_points,
            flush=True,
        )

        dtype = torch.float32
        N0, N1 = [int(i * weight_gridsize) for i in im_size]
        n_points = int(u.numel())
        configured_chunk_points = self.config.weighting_chunk_points
        if configured_chunk_points is None:
            chunk_points = max(n_points, 1)
        else:
            chunk_points = int(configured_chunk_points)
            if chunk_points <= 0:
                raise ValueError("weighting_chunk_points must be positive or None.")

        counts = torch.zeros(N0 * N1, dtype=dtype, device=self.device)
        valid_mask = torch.empty(n_points, dtype=torch.bool, device=self.device)

        def chunk_indices(start: int, end: int):
            u_chunk = u[start:end]
            v_chunk = v[start:end]
            flip_mask = v_chunk < 0
            u_sym = torch.where(flip_mask, -u_chunk, u_chunk)
            v_sym = torch.where(flip_mask, -v_chunk, v_chunk)
            p = ((u_sym + np.pi) * N0 / (2 * np.pi)).floor().to(torch.int64)
            q = ((v_sym + np.pi) * N1 / (2 * np.pi)).floor().to(torch.int64)
            valid = (p >= 0) & (p < N0) & (q >= 0) & (q < N1)
            return p, q, valid

        # First pass: accumulate global cell populations without retaining
        # full-length p/q/index tensors.
        for start in range(0, n_points, chunk_points):
            end = min(start + chunk_points, n_points)
            p, q, valid = chunk_indices(start, end)
            valid_mask[start:end] = valid
            if bool(valid.any()):
                indices = p[valid] * N1 + q[valid]
                counts.add_(torch.bincount(indices, minlength=N0 * N1).to(dtype))

        # Second pass: recompute bounded indices and emit one weight per point.
        weights = torch.zeros(n_points, dtype=dtype, device=self.device)
        for start in range(0, n_points, chunk_points):
            end = min(start + chunk_points, n_points)
            p, q, valid = chunk_indices(start, end)
            if bool(valid.any()):
                indices = p[valid] * N1 + q[valid]
                chunk_weights = weights[start:end]
                chunk_weights[valid] = 1.0 / torch.clamp(counts[indices], min=1.0)

        if not bool(valid_mask.any()):
            return torch.empty((0,), dtype=dtype, device=self.device), valid_mask
        if bool(valid_mask.all()):
            return weights, valid_mask
        return weights[valid_mask], valid_mask

    def bin_uv_data(
        self,
        uv_coords: torch.Tensor,
        w_lambda: torch.Tensor,
        visibilities: torch.Tensor,
        weights: torch.Tensor,
        grid_size: int = 512,
    ):
        """Bin UV data to reduce number of visibilities"""

        vis = visibilities.squeeze().to(torch.complex64)  # [N]
        w = weights.squeeze().to(torch.float32)  # [N]
        u = uv_coords[0].to(torch.float32)  # [N]
        v = uv_coords[1].to(torch.float32)  # [N]
        wl = w_lambda.to(torch.float32)  # [N]

        # Grid indices: [-pi, pi] -> [0, grid_size)
        p = (
            ((u + np.pi) * grid_size / (2 * np.pi))
            .floor()
            .clamp(0, grid_size - 1)
            .to(torch.int64)
        )
        q = (
            ((v + np.pi) * grid_size / (2 * np.pi))
            .floor()
            .clamp(0, grid_size - 1)
            .to(torch.int64)
        )

        idx = p * grid_size + q  # [N]
        max_bins = grid_size * grid_size

        sum_wu = torch.zeros(max_bins, dtype=torch.float32, device=self.device)
        sum_wv = torch.zeros(max_bins, dtype=torch.float32, device=self.device)
        sum_wvr = torch.zeros(max_bins, dtype=torch.float32, device=self.device)
        sum_wvi = torch.zeros(max_bins, dtype=torch.float32, device=self.device)
        sum_w = torch.zeros(max_bins, dtype=torch.float32, device=self.device)
        sum_wl = torch.zeros(max_bins, dtype=torch.float32, device=self.device)

        sum_wu.index_add_(0, idx, w * u)
        sum_wv.index_add_(0, idx, w * v)
        sum_wvr.index_add_(0, idx, w * vis.real.to(torch.float32))
        sum_wvi.index_add_(0, idx, w * vis.imag.to(torch.float32))
        sum_w.index_add_(0, idx, w)
        sum_wl.index_add_(0, idx, w * wl)

        mask = sum_w > 0
        w_bin = sum_w[mask]  # [M]
        u_binned = sum_wu[mask] / w_bin  # [M]
        v_binned = sum_wv[mask] / w_bin  # [M]
        vis_binned = (sum_wvr[mask] + 1j * sum_wvi[mask]) / w_bin  # [M]
        wl_binned = sum_wl[mask] / w_bin  # [M]

        binned_uv = torch.stack([u_binned, v_binned], dim=0)  # [2, M]
        vis_binned = vis_binned.unsqueeze(0).unsqueeze(0)  # [1, 1, M]

        return binned_uv, w_bin, vis_binned, wl_binned

    def classic_imaging(self, visibilities, samples_locs, weights):
        physics = MyRadioInterferometry(
            img_size=(self.config.imaging_npixel, self.config.imaging_npixel),
            samples_loc=samples_locs,
            dataWeight=weights,
            real_projection=True,
            nufft_k_oversampling=self.config.nufft_k_oversampling,
            device=self.device,
        )
        # Weighted least squares is ||sqrt(W) (F x - y)||^2. Pairing the
        # weighted operator with sqrt(W)y makes all data-fidelity calls,
        # adjoints, and operator-norm computations use the same objective.
        visibilities = visibilities * weights

        if self.verbose:
            print("visibilities", visibilities.shape, flush=True)
            print("samples_locs", samples_locs.shape, flush=True)
            print("weights", weights.shape, flush=True)

        return physics, visibilities, weights

    def _build_n_minus_1(self, n_pixel: int) -> torch.Tensor:
        """Build the image-domain w term with bounded row temporaries."""
        row_chunk_size = int(self.config.w_phase_row_chunk_size)
        if row_chunk_size <= 0:
            raise ValueError("w_phase_row_chunk_size must be positive.")
        coords = (
            torch.arange(n_pixel, dtype=torch.float32, device=self.device)
            - (n_pixel - 1) / 2
        ) * float(self.config.imaging_cellsize)
        coords_squared = coords.square()
        n_minus_1 = torch.empty(
            (n_pixel, n_pixel), dtype=torch.float32, device=self.device
        )
        for start in range(0, n_pixel, row_chunk_size):
            end = min(start + row_chunk_size, n_pixel)
            tile = 1.0 - coords_squared[start:end, None] - coords_squared[None, :]
            tile.clamp_(min=0.0).sqrt_().sub_(1.0)
            n_minus_1[start:end].copy_(tile)
        return n_minus_1

    def _build_phase_screen(
        self, n_minus_1: torch.Tensor, plane_w: float
    ) -> torch.Tensor:
        """Build one retained complex screen without a full complex temporary."""
        row_chunk_size = int(self.config.w_phase_row_chunk_size)
        n_pixel = int(n_minus_1.shape[0])
        screen = torch.empty(n_minus_1.shape, dtype=torch.complex64, device=self.device)
        scale = -2.0 * math.pi * float(plane_w)
        for start in range(0, n_pixel, row_chunk_size):
            end = min(start + row_chunk_size, n_pixel)
            angle = n_minus_1[start:end] * scale
            torch.cos(angle, out=screen.real[start:end])
            torch.sin(angle, out=screen.imag[start:end])
        return screen

    def w_stacking(
        self,
        w_planes,
        samples_locs,
        w_lambda,
        visibilities,
        weights,
        w_binning: str = "equal_width",
        shard_rank: int | None = None,
        shard_world_size: int | None = None,
    ) -> RadioPhysicsResult:
        """Build either a complete w-stack or this rank's local plane shard."""
        img_size = (self.config.imaging_npixel, self.config.imaging_npixel)
        n_pixel = img_size[0]
        l_max = (n_pixel / 2) * float(self.config.imaging_cellsize)
        if w_planes is None:
            w_planes = max(1, int(2.0 * w_lambda.abs().max().item() * l_max**2) + 1)
        if int(w_planes) <= 0:
            raise ValueError("w_planes must be positive or null.")
        w_binning = str(w_binning).lower()
        if w_binning not in {"equal_width", "equal_count"}:
            raise ValueError(
                "w_binning must be either 'equal_width' or 'equal_count', "
                f"got {w_binning!r}."
            )

        plane_indices = None
        if w_binning == "equal_width":
            boundaries = torch.linspace(
                w_lambda.min().item(),
                w_lambda.max().item(),
                int(w_planes) + 1,
                device=self.device,
            )
            bin_ids = torch.bucketize(w_lambda, boundaries[1:-1])
            counts = torch.bincount(bin_ids, minlength=int(w_planes))
            active_plane_ids = torch.nonzero(counts, as_tuple=False).flatten().tolist()
        else:
            plane_indices = _equal_count_plane_indices(w_lambda, int(w_planes))
            counts = torch.tensor(
                [indices.numel() for indices in plane_indices],
                dtype=torch.int64,
                device=w_lambda.device,
            )
            active_plane_ids = list(range(len(plane_indices)))

        sharded = shard_rank is not None or shard_world_size is not None
        if sharded:
            if shard_rank is None or shard_world_size is None:
                raise ValueError(
                    "shard_rank and shard_world_size must be set together."
                )
            shard_rank = int(shard_rank)
            shard_world_size = int(shard_world_size)
            if shard_world_size <= 0 or not 0 <= shard_rank < shard_world_size:
                raise ValueError(
                    f"Invalid physics shard rank {shard_rank}/{shard_world_size}."
                )

        if not active_plane_ids:
            raise ValueError("Cannot build w-stacking physics with no visibilities.")
        indexed_planes = list(enumerate(active_plane_ids))
        if self.verbose:
            print(
                f"w-stacking binning={w_binning} "
                f"visibilities_per_plane="
                f"{[int(counts[plane_id].item()) for plane_id in active_plane_ids]}",
                flush=True,
            )
        if sharded:
            assignments = _balanced_plane_assignment(
                active_plane_ids, counts, shard_world_size
            )
            local_indexed_planes = assignments[shard_rank]
            if self.verbose and shard_rank == 0:
                visibility_loads = [
                    sum(int(counts[plane_id].item()) for _, plane_id in assignment)
                    for assignment in assignments
                ]
                print(
                    "w-stacking balanced allocation: "
                    f"planes_per_rank={[len(assignment) for assignment in assignments]} "
                    f"visibilities_per_rank={visibility_loads}",
                    flush=True,
                )
        else:
            local_indexed_planes = indexed_planes

        # Generate and retain only this rank's screens, one plane at a time.
        # Empty ranks skip the image-sized n-minus-one allocation entirely.
        n_minus_1 = self._build_n_minus_1(n_pixel) if local_indexed_planes else None
        planes = []
        plane_measurements = []
        global_indices = []
        for global_index, plane_id in local_indexed_planes:
            selection = (
                bin_ids == plane_id
                if plane_indices is None
                else plane_indices[plane_id]
            )
            plane_w = float(w_lambda[selection].mean().item())
            screen = self._build_phase_screen(n_minus_1, plane_w)
            planes.append(
                MyRadioInterferometry(
                    img_size=img_size,
                    samples_loc=samples_locs[:, selection],
                    dataWeight=weights[selection],
                    image_phase=screen,
                    real_projection=True,
                    nufft_k_oversampling=self.config.nufft_k_oversampling,
                    device=self.device,
                )
            )
            plane_measurements.append(visibilities[..., selection] * weights[selection])
            global_indices.append(global_index)

        if sharded:
            if self.verbose:
                print(
                    f"w-stacking shard {shard_rank}/{shard_world_size}: "
                    f"local_planes={global_indices} global_planes={len(indexed_planes)}",
                    flush=True,
                )
            return RadioPhysicsResult(
                physics=planes or None,
                measurements=TensorList(plane_measurements) if planes else None,
                weights=None,
                global_indices=global_indices or None,
                num_operators=len(indexed_planes),
                global_measurement_count=int(samples_locs.shape[1]),
                from_shard=True,
            )

        physics = stack(*planes)
        physics.forw = _identity
        physics.A_adj = _identity
        physics.noise_model = IdentityNoise()
        physics.sensor_model = _identity
        physics.img_size = img_size
        physics.n_planes = len(planes)
        return RadioPhysicsResult(
            physics=physics,
            measurements=TensorList(plane_measurements),
            weights=weights,
            global_indices=None,
            num_operators=len(planes),
            global_measurement_count=int(samples_locs.shape[1]),
        )

    def create_deepinv_physics(
        self,
        visibility_path: str | Path,
        visibility_format: str,
        visibility_column: str,
        bin_data: bool = False,
        w_stacking: bool = True,
        w_planes: Optional[int] = None,
        w_binning: str = "equal_width",
        imaging_npixel: Optional[int] = None,
        binning_factor: Optional[float] = None,
        max_visibilities: Optional[int] = None,
        visibility_chunk_rows: int | None = 65536,
        shard_rank: int | None = None,
        shard_world_size: int | None = None,
    ) -> RadioPhysicsResult:
        uvw, visibilities, freqs = self.load_visibilities(
            visibility_path,
            visibility_format,
            visibility_column,
            chunk_rows=visibility_chunk_rows,
        )
        print(
            f"Loaded {uvw.shape[0]} visibilities with {visibilities.shape[1]} frequencies.",
            flush=True,
        )

        if max_visibilities is not None:
            n_freq = int(visibilities.shape[1])
            total_points = int(uvw.shape[0]) * n_freq
            if total_points > int(max_visibilities):
                n_rows = max(1, math.ceil(int(max_visibilities) / max(1, n_freq)))
                keep_rows = torch.linspace(
                    0,
                    int(uvw.shape[0]) - 1,
                    steps=n_rows,
                    device=uvw.device,
                ).long()
                uvw = uvw[keep_rows]
                visibilities = visibilities[keep_rows]

        samples_locs, w_lambda, visibilities = self.normalize_uv_coords(
            uvw, freqs, visibilities
        )

        print(
            f"Normalized UV coordinates to {samples_locs.shape[1]} points.", flush=True
        )

        imaging_npixel = (
            int(imaging_npixel)
            if imaging_npixel is not None
            else int(self.config.imaging_npixel)
        )
        binning_factor = (
            float(binning_factor)
            if binning_factor is not None
            else float(self.config.binning_factor)
        )

        im_size = torch.tensor([imaging_npixel, imaging_npixel], device=self.device)

        weights, valid_mask = self.uniform_weighting(
            samples_locs[0], samples_locs[1], im_size
        )

        print(
            f"Computed uniform weights for {weights.shape[0]} valid points.", flush=True
        )

        if not bool(valid_mask.all()):
            samples_locs = samples_locs[:, valid_mask]
            visibilities = visibilities[:, :, valid_mask]
            w_lambda = w_lambda[valid_mask]
        del valid_mask

        if max_visibilities is not None and samples_locs.shape[1] > int(
            max_visibilities
        ):
            n_vis = samples_locs.shape[1]
            keep = torch.linspace(
                0,
                n_vis - 1,
                steps=int(max_visibilities),
                device=samples_locs.device,
            ).long()
            samples_locs = samples_locs[:, keep]
            weights = weights[keep]
            visibilities = visibilities[:, :, keep]
            w_lambda = w_lambda[keep]

        if bin_data:
            samples_locs, weights, visibilities, w_lambda = self.bin_uv_data(
                samples_locs,
                w_lambda,
                visibilities,
                weights,
                grid_size=int(imaging_npixel * binning_factor),
            )
            print(f"Binned UV data to {samples_locs.shape[1]} points.", flush=True)

        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Visibility weights must be finite and non-negative.")
        sqrt_weights = weights.sqrt()

        if w_stacking:
            result = self.w_stacking(
                w_planes,
                samples_locs,
                w_lambda,
                visibilities,
                sqrt_weights,
                w_binning=w_binning,
                shard_rank=shard_rank,
                shard_world_size=shard_world_size,
            )
        else:
            if shard_rank is not None or shard_world_size is not None:
                raise ValueError(
                    "Physics sharding is currently supported only with w_stacking."
                )
            physics, visibilities, weights = self.classic_imaging(
                visibilities, samples_locs, sqrt_weights
            )
            result = RadioPhysicsResult(
                physics=physics,
                measurements=visibilities,
                weights=weights,
                global_indices=None,
                num_operators=1,
                global_measurement_count=int(samples_locs.shape[1]),
            )

        print(
            f"Created DeepInv physics with {samples_locs.shape[1]} points.", flush=True
        )

        return result


def create_radio_physics(
    ms_path: str | Path,
    imaging_npixel: int,
    imaging_cellsize: float,
    device: torch.device,
    visibility_column: str = "DATA",
    w_stacking: bool = True,
    w_planes: int = 32,
    w_binning: str = "equal_width",
    bin_data: bool = False,
    binning_factor: float = 1.25,
    nufft_k_oversampling: float = 1.5,
    max_visibilities: int | None = None,
    visibility_chunk_rows: int | None = 65536,
    weighting_chunk_points: int | None = 262144,
    w_phase_row_chunk_size: int = 256,
    shard_rank: int | None = None,
    shard_world_size: int | None = None,
    noise_level: float = 0.0,
    verbose: int = 0,
) -> RadioPhysicsResult:
    """Build the current FINUFFT radio operator from one Measurement Set."""
    imager = DeepinvDirtyImager(
        DirtyImagerConfig(
            imaging_npixel=int(imaging_npixel),
            imaging_cellsize=float(imaging_cellsize),
            binning_factor=float(binning_factor),
            nufft_k_oversampling=float(nufft_k_oversampling),
            combine_across_frequencies=False,
            weighting_chunk_points=weighting_chunk_points,
            w_phase_row_chunk_size=int(w_phase_row_chunk_size),
        ),
        device=device,
        verbose=verbose,
    )
    result = imager.create_deepinv_physics(
        visibility_path=ms_path,
        visibility_format="MS",
        visibility_column=visibility_column,
        w_stacking=w_stacking,
        w_planes=w_planes,
        w_binning=w_binning,
        bin_data=bin_data,
        imaging_npixel=int(imaging_npixel),
        binning_factor=float(binning_factor),
        max_visibilities=max_visibilities,
        visibility_chunk_rows=visibility_chunk_rows,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
    )
    if noise_level > 0:
        from deepinv.physics import GaussianNoise

        noise_model = GaussianNoise(sigma=float(noise_level))
        if isinstance(result.measurements, TensorList):
            result.measurements = TensorList(
                [noise_model(value) for value in result.measurements]
            )
        elif result.measurements is not None:
            result.measurements = noise_model(result.measurements)

    leaves = (
        []
        if result.physics is None
        else getattr(
            result.physics,
            "physics_list",
            (
                result.physics
                if isinstance(result.physics, (list, tuple))
                else [result.physics]
            ),
        )
    )
    for physics in leaves:
        physics.noise_model = IdentityNoise()
    return result
