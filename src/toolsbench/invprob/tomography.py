from dataclasses import dataclass
from typing import Optional

import torch
from deepinv.physics import GaussianNoise, TomographyWithAstra

from toolsbench.data import DataConfig, Tomography2D, Tomography3D
from toolsbench.invprob.base import (
    BaseInvProb,
    InvProb,
    InvProbConfig,
    build_problem_params,
)


@dataclass
class _TomographyParams:
    data: str = "2d"
    num_operators: int = 1
    num_angles: int = 100
    num_projections: int = 100
    noise_level: float = 0.01
    seed: int = 42
    geometry_type_2d: str = "parallel"
    geometry_type_3d: str = "conebeam"
    detector_spacing_2d: float = 1.0
    pixel_spacing_2d: float | tuple[float, float] = 1.0
    detector_pixels_2d: int | None = None
    detector_pixels_3d: tuple[int, int] = (972, 768)
    object_spacing_3d: tuple[float, float, float] = (0.1, 0.1, 0.1)
    use_dataset_sinogram: bool = True


class TomographyInvProb(BaseInvProb):
    """ASTRA-backed 2D/3D tomography inverse problem."""

    def get_invprob(self, invprob_config: InvProbConfig) -> InvProb:
        params = build_problem_params(_TomographyParams, invprob_config.params)
        self._validate_config(params)
        data_kind = params.data.lower()
        if data_kind in {"2d", "tomography_2d"}:
            return self._get_2d_invprob(invprob_config, params)
        if data_kind in {"3d", "tomography_3d"}:
            return self._get_3d_invprob(invprob_config, params)
        raise ValueError(
            f"Unsupported tomography data '{params.data}'. "
            "Choose one of ['2d', 'tomography_2d', '3d', 'tomography_3d']."
        )

    def _get_2d_invprob(
        self, invprob_config: InvProbConfig, params: _TomographyParams
    ) -> InvProb:
        device = torch.device(invprob_config.device)
        data = Tomography2D().get_data(
            DataConfig(
                size=invprob_config.size,
                batch_size=invprob_config.batch_size,
                channels=1,
                data_type=invprob_config.data_type,
                device=device,
                data_path=invprob_config.data_path,
            )
        )
        ground_truth = data["data"]
        if ground_truth.shape[1] != 1:
            ground_truth = ground_truth[:, :1]
        if ground_truth.shape[-2] != ground_truth.shape[-1]:
            raise ValueError(
                "TomographyInvProb 2D expects square images, got "
                f"{tuple(ground_truth.shape[-2:])}."
            )

        angles = self._angles_2d(invprob_config, params, device)
        split_indices = self._split_indices(params.num_angles, params.num_operators)
        angles_list = list(torch.tensor_split(angles, split_indices))

        full_physics = self._build_2d_physics(
            params=params,
            angles=angles,
            img_size=tuple(ground_truth.shape[-2:]),
            device=device,
            index=0,
        )
        with torch.no_grad():
            full_measurement = full_physics(ground_truth).contiguous()
        measurements = [
            measurement.contiguous()
            for measurement in torch.tensor_split(
                full_measurement, split_indices, dim=2
            )
        ]

        physics_factory = self._create_2d_physics_factory(
            params=params,
            angles_list=angles_list,
            img_size=tuple(ground_truth.shape[-2:]),
        )

        return InvProb(
            ground_truth=ground_truth,
            measurements=measurements,
            physics=physics_factory,
            ground_truth_shape=ground_truth.shape,
            num_operators=params.num_operators,
            min_pixel=ground_truth.min().item(),
            max_pixel=ground_truth.max().item(),
        )

    def _get_3d_invprob(
        self, invprob_config: InvProbConfig, params: _TomographyParams
    ) -> InvProb:
        if not params.use_dataset_sinogram:
            raise NotImplementedError(
                "Generating 3D ASTRA measurements by forward pass is not implemented."
            )

        device = torch.device(invprob_config.device)
        data = Tomography3D().get_data(
            DataConfig(
                size=invprob_config.size,
                batch_size=invprob_config.batch_size,
                channels=1,
                data_type=invprob_config.data_type,
                device=device,
                data_path=invprob_config.data_path,
            )
        )
        ground_truth = data["ground_truth"]
        img_shape = tuple(ground_truth.shape[-3:])

        trajectory = self._subsample_angles(data["vecs"], params.num_projections)
        sinogram = self._subsample_angles(data["sinogram"], params.num_projections)
        measurements_factory = self._create_3d_measurements_factory(
            sinogram=sinogram,
            num_operators=params.num_operators,
        )
        measurements = [
            measurements_factory(i, device, None)
            for i in range(params.num_operators)
        ]

        physics_factory = self._create_3d_physics_factory(
            params=params,
            trajectory=trajectory,
            img_shape=img_shape,
        )

        return InvProb(
            ground_truth=ground_truth,
            measurements=measurements,
            physics=physics_factory,
            ground_truth_shape=ground_truth.shape,
            num_operators=params.num_operators,
            min_pixel=ground_truth.min().item(),
            max_pixel=ground_truth.max().item(),
        )

    def _create_2d_physics_factory(
        self,
        params: _TomographyParams,
        angles_list: list[torch.Tensor],
        img_size: tuple[int, int],
    ):
        def factory(index: int, device: torch.device, shared: Optional[dict] = None):
            return self._build_2d_physics(
                params=params,
                angles=angles_list[index].to(device),
                img_size=img_size,
                device=torch.device(device),
                index=index,
            )

        return factory

    def _build_2d_physics(
        self,
        params: _TomographyParams,
        angles: torch.Tensor,
        img_size: tuple[int, int],
        device: torch.device,
        index: int,
    ) -> TomographyWithAstra:
        rng = torch.Generator(device=device).manual_seed(params.seed + index)
        return TomographyWithAstra(
            img_size=img_size,
            angles=angles,
            n_detector_pixels=params.detector_pixels_2d,
            detector_spacing=params.detector_spacing_2d,
            pixel_spacing=params.pixel_spacing_2d,
            geometry_type=params.geometry_type_2d,
            normalize=False,
            device=device,
            noise_model=GaussianNoise(sigma=params.noise_level, rng=rng),
        )

    def _create_3d_physics_factory(
        self,
        params: _TomographyParams,
        trajectory: torch.Tensor,
        img_shape: tuple[int, int, int],
    ):
        splits = self._projection_splits(trajectory.shape[0], params.num_operators)
        trajectory = trajectory.detach().cpu()

        def factory(index: int, device: torch.device, shared: Optional[dict] = None):
            start, end = splits[index]
            trajectory_subset = trajectory[start:end].clone().to(device)
            physics = TomographyWithAstra(
                img_size=img_shape,
                angles=end - start,
                n_detector_pixels=params.detector_pixels_3d,
                pixel_spacing=params.object_spacing_3d,
                geometry_type=params.geometry_type_3d,
                geometry_vectors=trajectory_subset,
                normalize=False,
                device=device,
            )
            physics._angle_range = (start, end)
            physics._operator_idx = index
            return physics

        return factory

    def _create_3d_measurements_factory(
        self,
        sinogram: torch.Tensor,
        num_operators: int,
    ):
        sinogram_tensor = self._format_3d_sinogram(sinogram).contiguous()
        splits = self._projection_splits(sinogram_tensor.shape[3], num_operators)

        def factory(index: int, device: torch.device, shared: Optional[dict] = None):
            start, end = splits[index]
            return sinogram_tensor[:, :, :, start:end, :].to(device).contiguous()

        return factory

    def _format_3d_sinogram(self, sinogram: torch.Tensor) -> torch.Tensor:
        if sinogram.ndim == 3:
            # (angles, detector_h, detector_v)
            # -> (1, 1, detector_h, angles, detector_v)
            return sinogram.permute(1, 0, 2).unsqueeze(0).unsqueeze(0)
        if sinogram.ndim == 4:
            # (batch, angles, detector_h, detector_v)
            # -> (batch, 1, detector_h, angles, detector_v)
            return sinogram.permute(0, 2, 1, 3).unsqueeze(1)
        if sinogram.ndim == 5:
            return sinogram
        raise ValueError(
            "3D tomography sinogram must have shape (angles, detector_h, "
            "(batch, angles, detector_h, detector_v), or ASTRA format "
            f"(batch, channels, detector_h, angles, detector_v); got {tuple(sinogram.shape)}."
        )

    def _angles_2d(
        self,
        invprob_config: InvProbConfig,
        params: _TomographyParams,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.linspace(
            0,
            180,
            params.num_angles + 1,
            dtype=invprob_config.data_type,
            device=device,
        )[:-1]

    def _subsample_angles(
        self, tensor: torch.Tensor, num_angles: int
    ) -> torch.Tensor:
        angle_dim = self._angle_dim(tensor)
        total = tensor.shape[angle_dim]
        if num_angles >= total:
            return tensor
        indices = torch.linspace(0, total - 1, steps=num_angles, dtype=torch.long)
        return tensor.index_select(angle_dim, indices.to(tensor.device))

    def _angle_dim(self, tensor: torch.Tensor) -> int:
        if tensor.ndim in {2, 3}:
            return 0
        if tensor.ndim == 4:
            return 1
        if tensor.ndim == 5:
            return 3
        raise ValueError(
            f"Cannot infer angle dimension for shape {tuple(tensor.shape)}."
        )

    def _split_indices(self, num_angles: int, num_operators: int) -> list[int]:
        splits = self._projection_splits(num_angles, num_operators)
        return [end for _, end in splits[:-1]]

    def _projection_splits(
        self, num_angles: int, num_operators: int
    ) -> list[tuple[int, int]]:
        base, rem = divmod(int(num_angles), int(num_operators))
        sizes = [base + (1 if i < rem else 0) for i in range(num_operators)]
        edges = [0]
        for size in sizes:
            edges.append(edges[-1] + size)
        return [(edges[i], edges[i + 1]) for i in range(num_operators)]

    def _validate_config(self, params: _TomographyParams) -> None:
        if params.num_operators < 1:
            raise ValueError("num_operators must be at least 1.")
        data_kind = params.data.lower()
        if data_kind in {"2d", "tomography_2d"}:
            if params.num_angles < params.num_operators:
                raise ValueError("num_angles must be at least num_operators.")
        elif data_kind in {"3d", "tomography_3d"}:
            if params.num_projections < params.num_operators:
                raise ValueError("num_projections must be at least num_operators.")
        if params.noise_level < 0:
            raise ValueError("noise_level must be non-negative.")
