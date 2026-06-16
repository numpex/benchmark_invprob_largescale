import torch

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from deepinv.physics import Physics, StackedPhysics

from toolsbench.data.base import BaseData


@dataclass
class InvProb:
    ground_truth: torch.Tensor
    measurements: torch.Tensor | list[torch.Tensor] | Callable[[int, str | torch.device], torch.Tensor]
    physics: Physics | StackedPhysics | list[Physics] | Callable[[int, str | torch.device], Physics]
    ground_truth_shape: torch.Size
    num_operators: int = 1
    min_pixel: float = 0.0
    max_pixel: float = 1.0
    invprob_kwargs: Optional[dict] = None


@dataclass
class InvProbConfig:
    """Configuration for an inverse problem."""

    size: tuple[int, ...]
    batch_size: int = 1
    channels: int = 3
    data_type: torch.dtype = torch.float32
    device: torch.device | str = torch.device("cpu")
    data_path: str | Path = "./data"


class BaseInvProb(ABC):

    def get_invprob(self, invprob_config: InvProbConfig) -> InvProb:
        """Returns a batch of data with parameters specified in the invprob_config."""
        raise NotImplementedError
