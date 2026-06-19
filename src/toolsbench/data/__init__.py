from pathlib import Path

from toolsbench.data.base import DataConfig
from toolsbench.data.deepinv_datasets import HighResColorImagingData, Tomography2D
from toolsbench.data.synthetic import SyntheticData
from toolsbench.data.tomography_3d import Tomography3D

__all__ = [
    "DataConfig",
    "HighResColorImagingData",
    "Tomography2D",
    "SyntheticData",
    "Tomography3D",
]


def check_installed(name: str, path: str | Path) -> Path:
    """Install dataset if needed

    Parameters
    ----------
    name : str
        Name of the dataset to check.

    Returns
    -------
    bool
        True if the dataset is installed, False otherwise.
    """
    dataset_classes = {
        "highres_color_image": HighResColorImagingData,
        "tomography_2d": Tomography2D,
        "synthetic": SyntheticData,
        "tomography_3d": Tomography3D,
    }

    if name not in dataset_classes:
        raise ValueError(f"Dataset {name} is not recognized.")
    
    return dataset_classes[name]().download(path)
