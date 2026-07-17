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

    def download(
        self,
        data_path: str | Path = Path("./data"),
        fits_size: str | None = None,
    ) -> Path:
        """Download the requested FITS image(s) and return their directory.

        When *fits_size* is omitted, all available image sizes are downloaded
        for backward compatibility.
        """
        data_path = Path(data_path)
        requested_sizes = VALID_FITS_SIZES if fits_size is None else (str(fits_size),)
        invalid_sizes = set(requested_sizes).difference(VALID_FITS_SIZES)
        if invalid_sizes:
            raise ValueError(
                f"Unknown radio FITS size(s) {sorted(invalid_sizes)}. "
                f"Expected one of {list(VALID_FITS_SIZES)}."
            )

        if all(
            len(list((data_path / size).glob("*.fits"))) == 1
            for size in requested_sizes
        ):
            return data_path

        allow_patterns = [f"{size}/*.fits" for size in requested_sizes]
        return self._download_hf_snapshot(data_path, allow_patterns=allow_patterns)

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
