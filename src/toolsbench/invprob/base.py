import warnings

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping as MappingABC
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, TypeVar

import torch
from torch.utils.data import DataLoader, Dataset

from deepinv.physics import Physics, StackedPhysics


@dataclass
class InvProb(MappingABC):
    measurements: torch.Tensor | list[torch.Tensor] | Callable[[int, str | torch.device], torch.Tensor]
    physics: Physics | StackedPhysics | list[Physics] | Callable[[int, str | torch.device], Physics]
    ground_truth_shape: torch.Size
    ground_truth: Optional[torch.Tensor] = None
    num_operators: int = 1
    min_pixel: float = 0.0
    max_pixel: float = 1.0
    invprob_kwargs: Optional[dict] = None

    def asdict(self) -> dict[str, Any]:
        """Return the benchmark-facing dictionary for ``Objective.set_data``."""
        data = {
            "measurements": self.measurements,
            "physics": self.physics,
            "ground_truth_shape": self.ground_truth_shape,
            "num_operators": self.num_operators,
            "min_pixel": self.min_pixel,
            "max_pixel": self.max_pixel,
        }
        if self.ground_truth is not None:
            data["ground_truth"] = self.ground_truth
        if self.invprob_kwargs:
            data.update(self.invprob_kwargs)
        return data

    def resized(self, image_size, *, device=None) -> "InvProb":
        """Return this inverse problem restated at a different spatial size.

        ``image_size`` accepts an int or ``[s]`` (square), ``[h, w]``, or
        ``[d, h, w]``, matching the dataset's own convention. Batch and channels
        are kept, so the physics must stay valid at the new size.

        The physics operators are reset in place rather than copied -- they can
        be large and stacked -- so the returned problem shares them with this
        one, and this one should be considered spent.
        """
        # A single-element list is a wrapper, not a 1D size: the configs write
        # [s] for a square and [[d, h, w]] for a volume.
        while isinstance(image_size, (list, tuple)) and len(image_size) == 1:
            image_size = image_size[0]
        spatial = (
            tuple(int(s) for s in image_size)
            if isinstance(image_size, (list, tuple))
            else (int(image_size),) * 2
        )
        shape = torch.Size((*self.ground_truth_shape[:2], *spatial))
        if shape == self.ground_truth_shape:
            return self

        if self.ground_truth is not None:
            device = device or self.ground_truth.device
            ground_truth = torch.nn.functional.interpolate(
                self.ground_truth.to(device),
                size=spatial,
                mode="trilinear" if len(spatial) == 3 else "bilinear",
                align_corners=False,
            )
        else:
            ground_truth = torch.rand(shape, device=device)

        # Operators cache the size they were built for; clearing it makes them
        # rebuild for the new shape. 
        pending = [self.physics]
        while pending:
            operator = pending.pop()
            children = getattr(operator, "local_physics", None) or getattr(
                operator, "physics_list", None
            )
            if children:
                pending.extend(children)
            elif hasattr(operator, "imsize"):
                operator.imsize = None

        with torch.no_grad():
            measurements = self.physics.A(ground_truth)

        return replace(
            self,
            ground_truth=ground_truth,
            ground_truth_shape=shape,
            measurements=measurements,
        )

    def __getitem__(self, key: str) -> Any:
        return self.asdict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.asdict())

    def __len__(self) -> int:
        return len(self.asdict())

    def to_dataloader(self) -> DataLoader:
        """Return a single-sample DataLoader wrapping this inverse problem.

        Used to satisfy deepinv's Trainer.setup_train() validation, which only
        calls dataset.__getitem__(0) — batch_size=None disables collation entirely.
        """
        if self.ground_truth is None:
            raise ValueError("to_dataloader() requires ground_truth to be set.")
        x, y = self.ground_truth, self.measurements

        class _DS(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, _):
                return x, y

        return DataLoader(_DS(), batch_size=None)


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
