import deepinv
import fnmatch
import requests
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from toolsbench.data.utils import download_file


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
        ...

    @abstractmethod
    def download(self, data_path: str | Path = Path("./data")) -> Path:
        """Downloads the dataset if necessary."""
        ...


class HFData(BaseData, ABC):
    """Base class for datasets hosted on HuggingFace Hub.

    Provides helpers to build dataset URLs and to download individual files
    or whole repository snapshots.  Concrete subclasses must declare
    :attr:`_hf_repo` as a class attribute and implement :meth:`get_data`
    and :meth:`download`.
    """

    _HF_BASE = "https://huggingface.co/datasets"
    _hf_repo: str  # concrete subclasses must set this as a class attribute

    def _hf_url(self, filename: str, branch: str = "main") -> str:
        """Return the resolve URL for *filename* in the HF dataset repo."""
        return f"{self._HF_BASE}/{self._hf_repo}/resolve/{branch}/{filename}"

    def _download_hf_file(self, filename: str, data_path: Path) -> Path:
        """Download a single *filename* from the HF repo into *data_path*."""
        url = self._hf_url(filename)
        return download_file(url, data_path / filename)

    def _download_hf_snapshot(
        self,
        data_path: Path,
        allow_patterns: str | list[str] | None = None,
    ) -> Path:
        """Download all (or filtered) files from the HF repo into *data_path*.

        Uses the public HuggingFace HTTP API and ``requests`` to avoid a hard
        dependency on ``huggingface_hub``.
        """
        data_path = Path(data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        patterns = self._normalize_hf_patterns(allow_patterns)
        for filename in self._list_hf_files():
            if patterns is not None and not any(
                fnmatch.fnmatch(filename, pattern) for pattern in patterns
            ):
                continue
            download_file(self._hf_url(filename), data_path / filename)
        return data_path

    def _list_hf_files(self, branch: str = "main") -> list[str]:
        url = f"https://huggingface.co/api/datasets/{self._hf_repo}/tree/{branch}"
        response = requests.get(url, params={"recursive": "1"}, timeout=(10, 60))
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected HuggingFace API response from {url}.")
        return [
            item["path"]
            for item in payload
            if item.get("type") == "file" and "path" in item
        ]

    @staticmethod
    def _normalize_hf_patterns(
        allow_patterns: str | list[str] | None,
    ) -> list[str] | None:
        if allow_patterns is None:
            return None
        if isinstance(allow_patterns, str):
            return [allow_patterns]
        return list(allow_patterns)


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
            device=(
                torch.device(data_config.device)
                if isinstance(data_config.device, str)
                else data_config.device
            ),
            dtype=data_config.data_type,
            resize_mode="resize",
        )
        return {"data": img.repeat(data_config.batch_size, 1, 1, 1)}

    def download(self, data_path: str | Path = Path("./data")) -> Path:
        p = Path(data_path) / self.image_name
        if not p.exists():
            deepinv.utils.download_example(self.image_name, Path(data_path))
        return p
