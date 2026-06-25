import warnings
import torch
from pathlib import Path

from toolsbench.data.base import HFData, DataConfig


class Tomography3D(HFData):
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

    _hf_repo = "romainvo/ct_examples"
    _FILENAME = "Walnut-CBCT_8.pt"

    def get_data(self, data_config: DataConfig) -> dict[str, torch.Tensor]:
        warnings.warn(
            f"{self.__class__.__name__}: data_config.size is ignored. "
            "The spatial dimensions of the 3D volume and sinogram are fixed by "
            "the dataset; resizing them would invalidate the scan geometry.",
            UserWarning,
            stacklevel=2,
        )

        dataset = self._get_dataset(data_config)

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
    
    def _get_dataset(self, data_config: DataConfig) -> dict[str, torch.Tensor]:
        data_path = self.download(data_path=data_config.data_path)
        return torch.load(data_path, weights_only=True)

    def download(self, data_path: str | Path = Path("./data")) -> Path:
        return self._download_hf_file(self._FILENAME, Path(data_path))
