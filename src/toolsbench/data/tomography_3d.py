import warnings
import torch
from pathlib import Path

import requests

from toolsbench.data.base import BaseData, DataConfig


class Tomography3D(BaseData):
    """3D cone-beam CT dataset (Walnut) loaded from HuggingFace.

    Data loading only: handles download, caching, device transfer, and type
    casting. Physics/operator construction must be done in a separate module.

    The spatial dimensions of the volume and sinogram are fixed by the dataset.
    Resizing them would invalidate the scan geometry, so ``data_config.size``
    is always ignored and a ``UserWarning`` is emitted.

    Returned dict keys
    ------------------
    ground_truth : Tensor, shape (B, 1, D, H, W)
    sinogram     : Tensor, shape (B, angles, det_h, det_v)
    vecs         : Tensor, shape (angles, 12)  — cone-beam geometry vectors
    """

    _FILENAME = "Walnut-CBCT_8.pt"
    _URL = (
        "https://huggingface.co/datasets/romainvo/ct_examples"
        "/resolve/main/Walnut-CBCT_8.pt"
    )
    _CHUNK_SIZE = 8 * 1024 * 1024
    _PROGRESS_STEP = 128 * 1024 * 1024

    def get_data(self, data_config: DataConfig) -> dict[str, torch.Tensor]:
        warnings.warn(
            f"{self.__class__.__name__}: data_config.size is ignored. "
            "The spatial dimensions of the 3D volume and sinogram are fixed by "
            "the dataset; resizing them would invalidate the scan geometry.",
            UserWarning,
            stacklevel=2,
        )
        dataset = load_torch_url(
            self._URL,
            data_path=Path(data_config.data_path),
            filename=self._FILENAME,
        )
        device = (
            torch.device(data_config.device)
            if isinstance(data_config.device, str)
            else data_config.device
        )
        dtype = data_config.data_type

        # Ground truth volume: normalise to (1, 1, D, H, W) then batch
        gt = dataset["dense_reconstruction"].to(device=device, dtype=dtype)
        while gt.ndim < 5:
            gt = gt.unsqueeze(0)
        gt = gt.repeat(data_config.batch_size, *([1] * (gt.ndim - 1)))

        # Sinogram (angles, det_h, det_v) -> (B, angles, det_h, det_v)
        sino = dataset["sinogram"].to(device=device, dtype=dtype)
        while sino.ndim < 4:
            sino = sino.unsqueeze(0)
        sino = sino.repeat(data_config.batch_size, *([1] * (sino.ndim - 1)))

        # Geometry vectors — not batched
        vecs = dataset["vecs"].to(device=device, dtype=dtype)

        return {"ground_truth": gt, "sinogram": sino, "vecs": vecs}

    def download(self, data_path: str | Path = Path("./data")) -> Path:
        cache_path = Path(data_path) / self._FILENAME
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path

        tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
        if tmp_path.exists():
            tmp_path.unlink()
        self._download_file(self._URL, tmp_path, cache_path)
        return cache_path

    @classmethod
    def _download_file(cls, url: str, tmp_path: Path, cache_path: Path) -> None:
        print(f"\nDownloading tomography_3d data to {cache_path}", flush=True)
        try:
            with requests.get(url, stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length") or 0)
                if total:
                    print(
                        f"Expected download size: {cls._format_bytes(total)}",
                        flush=True,
                    )

                downloaded = 0
                next_report = cls._next_report_threshold(total)
                with tmp_path.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=cls._CHUNK_SIZE):
                        if not chunk:
                            continue
                        file.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            print(
                                cls._format_progress(downloaded, total),
                                flush=True,
                            )
                            next_report += cls._next_report_threshold(total)

            tmp_path.replace(cache_path)
            print(
                f"Downloaded tomography_3d data: {cls._format_bytes(downloaded)}",
                flush=True,
            )
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(
                "Failed to download the tomography_3d Walnut dataset from "
                f"{url}."
            ) from exc

    @classmethod
    def _next_report_threshold(cls, total: int) -> int:
        if total <= 0:
            return cls._PROGRESS_STEP
        return max(total // 20, cls._PROGRESS_STEP)

    @staticmethod
    def _format_progress(downloaded: int, total: int) -> str:
        if total <= 0:
            return f"Downloaded {Tomography3D._format_bytes(downloaded)}"
        pct = 100.0 * downloaded / total
        return (
            f"Downloaded {Tomography3D._format_bytes(downloaded)} / "
            f"{Tomography3D._format_bytes(total)} ({pct:.1f}%)"
        )

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        units = ("B", "KiB", "MiB", "GiB", "TiB")
        value = float(num_bytes)
        for unit in units:
            if value < 1024 or unit == units[-1]:
                return f"{value:.1f} {unit}"
            value /= 1024


def load_torch_url(
    url: str,
    data_path: str | Path = Path("./data"),
    filename: str = Tomography3D._FILENAME,
) -> dict[str, torch.Tensor]:
    cache_path = Path(data_path) / filename
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not (cache_path.exists() and cache_path.stat().st_size > 0):
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
        if tmp_path.exists():
            tmp_path.unlink()
        Tomography3D._download_file(url, tmp_path, cache_path)
    return torch.load(cache_path, weights_only=True)
