from pathlib import Path
from typing import Any

from toolsbench.data.base import DataConfig
from toolsbench.data.deepinv_datasets import HighResColorImagingData, Tomography2D
from toolsbench.data.synthetic import SyntheticData
from toolsbench.data.tomography_3d import Tomography3D
from toolsbench.data.radio_interferometry import RadioInterferometryData

__all__ = [
    "DataConfig",
    "HighResColorImagingData",
    "Tomography2D",
    "SyntheticData",
    "Tomography3D",
    "RadioInterferometryData",
]


def check_installed(name: str, path: str | Path, **download_kwargs: Any) -> Path:
    """Download the dataset *name* into *path* if not already present.

    Parameters
    ----------
    name : str
        Name of the dataset to check / install.
    path : str or Path
        Local directory where the dataset should be stored.
    **download_kwargs
        Dataset-specific options forwarded to its :meth:`download` method.

    Returns
    -------
    Path
        The path returned by the dataset's :meth:`download` method
        (a file for single-file datasets, a directory otherwise).
    """
    dataset_classes = {
        "highres_color_image": HighResColorImagingData,
        "tomography_2d": Tomography2D,
        "synthetic": SyntheticData,
        "tomography_3d": Tomography3D,
        "radio_interferometry": RadioInterferometryData,
    }

    if name not in dataset_classes:
        raise ValueError(
            f"Dataset '{name}' is not recognized. "
            f"Known datasets: {sorted(dataset_classes)}."
        )

    return dataset_classes[name]().download(path, **download_kwargs)
