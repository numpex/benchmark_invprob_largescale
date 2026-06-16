import deepinv
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor


@dataclass
class DataConfig:
    """Configuration for a data source."""

    size: tuple[int, ...]
    batch_size: int = 1
    channels: int = 3
    data_type: torch.dtype = torch.float32
    device: torch.device | str = torch.device("cpu")
    data_path: str | Path = "./data"
    

class BaseData(ABC):

    @abstractmethod
    def get_data(self, data_config: DataConfig) -> dict[str, torch.Tensor]:
        """Returns a batch of data with parameters specified in the data_config."""
        raise NotImplementedError

    def download(self, data_path: str | Path = Path("./data")) -> Path:
        """Downloads the dataset if necessary."""
        return Path(data_path)


class DeepinvData(BaseData, ABC):
    """Base class for datasets that download and load a 2D image via deepinv utilities.

    Subclasses only need to implement :attr:`image_name`.  The download (with
    existence check) and image loading are handled here.
    """

    @property
    @abstractmethod
    def image_name(self) -> str:
        """Filename of the image to fetch from the deepinv HuggingFace dataset."""
        ...

    def get_data(self, data_config: DataConfig) -> dict[str, torch.Tensor]:
        if len(data_config.size) != 2:
            raise ValueError(
                f"{self.__class__.__name__} only supports 2D data, "
                f"got size={data_config.size}"
            )
        path = self.download(data_config.data_path)
        img = deepinv.utils.load_image(
            path,
            img_size=data_config.size,
            device=torch.device(data_config.device) if isinstance(data_config.device, str) else data_config.device,
            dtype=data_config.data_type,
            resize_mode="resize",
        )
        return {"data": img.repeat(data_config.batch_size, 1, 1, 1)}

    def download(self, data_path: str | Path = Path("./data")) -> Path:
        p = Path(data_path) / self.image_name
        if not p.exists():
            deepinv.utils.download_example(self.image_name, Path(data_path))
        return p
