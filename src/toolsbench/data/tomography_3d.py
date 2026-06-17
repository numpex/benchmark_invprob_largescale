import warnings
import torch
from pathlib import Path
from deepinv.utils import load_torch_url

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

    def get_data(self, data_config: DataConfig) -> dict[str, torch.Tensor]:
        warnings.warn(
            f"{self.__class__.__name__}: data_config.size is ignored. "
            "The spatial dimensions of the 3D volume and sinogram are fixed by "
            "the dataset; resizing them would invalidate the scan geometry.",
            UserWarning,
            stacklevel=2,
        )
        dataset = self._load_or_download_dataset(Path(data_config.data_path))
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

    def _load_or_download_dataset(self, data_dir: Path) -> dict[str, torch.Tensor]:
        cache_path = data_dir / self._FILENAME
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            dataset = torch.load(cache_path, weights_only=True)
        else:
            dataset = load_torch_url(self._URL)
            torch.save(dataset, cache_path)
        return dataset
