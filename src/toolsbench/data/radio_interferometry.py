"""Radio interferometry FITS images from HuggingFace."""

from __future__ import annotations

from pathlib import Path

import torch

from toolsbench.data.base import DataConfig, HFData


VALID_FITS_SIZES = ("1024", "10k")


class RadioInterferometryData(HFData):
    """Radio interferometry sky images downloaded from HuggingFace."""

    _hf_repo = "bmalezieux-numpex/radio-interferometry-images"

    def get_data(self, data_config: DataConfig) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "RadioInterferometryData.get_data() is not implemented. "
            "Use toolsbench.invprob.RadioInterferometryInvProb instead."
        )

    def download(self, data_path: str | Path = Path("./data")) -> Path:
        """Download all FITS images into *data_path* and return that directory."""
        data_path = Path(data_path)
        if all(len(list((data_path / size).glob("*.fits"))) == 1 for size in VALID_FITS_SIZES):
            return data_path
        return self._download_hf_snapshot(data_path, allow_patterns="*.fits")

    @classmethod
    def select_fits_file(
        cls,
        data_path: str | Path,
        fits_size: str = "1024",
        fits_name: str | None = None,
    ) -> Path:
        """Return the source FITS image for *fits_size* under *data_path*."""
        data_path = Path(data_path)

        if fits_name:
            fits_path = data_path / fits_name
            if not fits_path.exists():
                matches = sorted((data_path / fits_size).glob(Path(fits_name).name))
                if matches:
                    fits_path = matches[0]
            if not fits_path.exists():
                raise FileNotFoundError(
                    f"Radio FITS image {fits_name!r} was not found under "
                    f"{data_path}."
                )
            return fits_path

        size_dir = data_path / fits_size
        fits_files = sorted(size_dir.glob("*.fits"))
        if len(fits_files) != 1:
            available = sorted(str(p.relative_to(data_path)) for p in data_path.rglob("*.fits"))
            raise RuntimeError(
                f"Expected exactly one FITS file under {size_dir}, found "
                f"{len(fits_files)}. Available FITS files: {available}"
            )
        return fits_files[0]

    @classmethod
    def relative_fits_name(
        cls,
        data_path: str | Path,
        fits_size: str = "1024",
        fits_name: str | None = None,
    ) -> str:
        """Return the selected FITS path relative to *data_path*."""
        data_path = Path(data_path)
        return str(cls.select_fits_file(data_path, fits_size, fits_name).relative_to(data_path))
