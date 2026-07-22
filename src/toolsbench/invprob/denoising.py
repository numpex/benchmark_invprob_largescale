from dataclasses import dataclass

import torch
from deepinv.physics import Denoising, GaussianNoise, stack

from toolsbench.data import DataConfig, SyntheticData
from toolsbench.invprob.base import (
    BaseInvProb,
    InvProb,
    InvProbConfig,
    build_problem_params,
)


@dataclass
class _DenoisingParams:
    num_frames: int = 1  # number of measurements (independent noisy copies)
    noise_std: float = 0.01
    data: str = "synthetic"


class DenoisingInvProb(BaseInvProb):
    """Pure denoising problem: ``y = x + noise`` (identity forward operator).

    Dimension-agnostic — works for 2D ``(B, C, H, W)`` and 3D
    ``(B, C, D, H, W)`` signals, since both :class:`SyntheticData` and deepinv's
    :class:`Denoising` physics generalise to arbitrary spatial dimensions. Used
    as the 3D counterpart of :class:`MultiFrameSuperResInvProb`, whose super-res
    physics is 2D-only.
    """

    def get_invprob(self, invprob_config: InvProbConfig) -> InvProb:
        params = build_problem_params(_DenoisingParams, invprob_config.params)
        if params.data != "synthetic":
            raise ValueError(
                f"DenoisingInvProb only supports data='synthetic', got '{params.data}'."
            )
        device = torch.device(invprob_config.device)

        data = SyntheticData().get_data(
            DataConfig(
                size=invprob_config.size,
                batch_size=invprob_config.batch_size,
                channels=invprob_config.channels,
                data_type=invprob_config.data_type,
                device=device,
                data_path=invprob_config.data_path,
            )
        )

        if params.num_frames < 1:
            raise ValueError("num_frames must be at least 1.")

        # One denoising operator per measurement, each with independent noise.
        # Build physics on `device` so its noise generator matches the data device.
        physics_list = []
        for i in range(params.num_frames):
            generator = torch.Generator(device=device).manual_seed(i)
            physics_list.append(
                Denoising(
                    GaussianNoise(sigma=params.noise_std, rng=generator), device=device
                )
            )
        physics = stack(*physics_list)

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
