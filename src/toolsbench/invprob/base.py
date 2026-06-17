import warnings

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, TypeVar

import torch

from deepinv.physics import Physics, StackedPhysics


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
    params: dict[str, Any] = field(default_factory=dict)


_ParamsT = TypeVar("_ParamsT")


def build_problem_params(
    params_cls: type[_ParamsT],
    params: Mapping[str, Any] | None,
) -> _ParamsT:
    """Instantiate a private problem-parameter dataclass from user params."""
    params_dict = {} if params is None else dict(params)
    allowed = {
        dataclass_field.name
        for dataclass_field in fields(params_cls)
        if dataclass_field.init
    }
    unknown = sorted(set(params_dict) - allowed)
    if unknown:
        name = params_cls.__name__.removeprefix("_")
        warnings.warn(
            f"Parameters {unknown} are not available for {name} and will not be "
            f"taken into account. Choose from {sorted(allowed)}.",
            UserWarning,
            stacklevel=2,
        )
        params_dict = {
            key: value for key, value in params_dict.items() if key in allowed
        }
    return params_cls(**params_dict)


class BaseInvProb(ABC):

    @abstractmethod
    def get_invprob(self, invprob_config: InvProbConfig) -> InvProb:
        """Returns a batch of data with parameters specified in the invprob_config."""
        raise NotImplementedError
