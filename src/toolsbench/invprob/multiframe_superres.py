from dataclasses import dataclass
from typing import Type

import torch
from deepinv.physics import (
    Blur,
    Downsampling,
    GaussianNoise,
    StackedPhysics,
    compose,
    stack,
)
from deepinv.physics.functional import gaussian_blur

from toolsbench.data import DataConfig, HighResColorImagingData, SyntheticData
from toolsbench.data.base import BaseData
from toolsbench.invprob.base import (
    BaseInvProb,
    InvProb,
    InvProbConfig,
    build_problem_params,
)


@dataclass
class _MultiFrameSuperResParams:
    num_frames: int = 5
    scale_factor: int = 2
    noise_std: float = 0.01
    blur_kernel_size: int = 5
    blur_sigma: float = 1.0
    data: str = "synthetic"


class MultiFrameSuperResInvProb(BaseInvProb):

    def get_invprob(self, invprob_config: InvProbConfig) -> InvProb:
        params = build_problem_params(
            _MultiFrameSuperResParams,
            invprob_config.params,
        )
        device = torch.device(invprob_config.device)
        data = self._get_data(params).get_data(
            DataConfig(
                size=invprob_config.size,
                batch_size=invprob_config.batch_size,
                channels=invprob_config.channels,
                data_type=invprob_config.data_type,
                device=device,
                data_path=invprob_config.data_path,
            )
        )

        physics = self._build_stacked_physics(
            params, invprob_config.data_type, data["data"].shape, device
        )

        with torch.no_grad():
            measurements = physics(data["data"])
            for i in range(len(measurements)):
                measurements[i] = measurements[i].clamp(0.0, 1.0)

        return InvProb(
            ground_truth=data["data"],
            measurements=measurements,
            physics=physics,
            ground_truth_shape=data["data"].shape,
            num_operators=params.num_frames,
            min_pixel=0.0,
            max_pixel=1.0,
        )

    def _get_data(self, params: _MultiFrameSuperResParams) -> BaseData:
        data_sources: dict[str, Type[BaseData]] = {
            "synthetic": SyntheticData,
            "highres_imaging": HighResColorImagingData,
        }
        key = params.data
        if key not in data_sources:
            raise ValueError(
                f"Unsupported data source '{params.data}'. "
                f"Choose one of {sorted(data_sources)}."
            )
        return data_sources[key]()

    def _build_stacked_physics(
        self,
        params: _MultiFrameSuperResParams,
        data_type: torch.dtype,
        ground_truth_shape: torch.Size,
        device: torch.device,
    ) -> StackedPhysics:
        if len(ground_truth_shape) != 4:
            raise ValueError(
                "MultiFrameSuperResInvProb expects 2D image batches with shape "
                f"(batch, channels, height, width), got {tuple(ground_truth_shape)}."
            )
        if params.num_frames < 1:
            raise ValueError("num_frames must be at least 1.")
        if params.scale_factor < 1:
            raise ValueError("scale_factor must be at least 1.")
        if params.blur_kernel_size < 1:
            raise ValueError("blur_kernel_size must be at least 1.")
        if params.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be odd.")

        _, channels, height, width = ground_truth_shape
        img_size = (channels, height, width)
        angles = torch.linspace(0, 180, params.num_frames + 1)[:-1]

        physics_list = []
        for frame_idx, angle in enumerate(angles):
            kernel = gaussian_blur(
                psf_size=(
                    params.blur_kernel_size,
                    params.blur_kernel_size,
                ),
                sigma=params.blur_sigma,
                angle=float(angle.item()),
                device=device,
                dtype=data_type,
            )
            blur = Blur(filter=kernel, padding="circular", device=device)
            generator = torch.Generator(device=device).manual_seed(frame_idx)
            downsample = Downsampling(
                img_size=img_size,
                filter=None,
                factor=params.scale_factor,
                padding="circular",
                device=device,
                noise_model=GaussianNoise(
                    sigma=params.noise_std,
                    rng=generator,
                ),
            )
            frame_physics = compose(blur, downsample)
            physics_list.append(frame_physics.to(device))

        return stack(*physics_list)
