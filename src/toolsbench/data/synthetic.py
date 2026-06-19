from pathlib import Path

import torch

from toolsbench.data.base import BaseData, DataConfig


class SyntheticData(BaseData):

    def get_data(self, data_config: DataConfig) -> dict[str, torch.Tensor]:
        """Returns a batch of synthetic data with parameters specified in the data_config."""
        img = self._generate_synthetic_signal(
            data_config.device, data_config.size, data_config.channels, data_config.data_type
        )
        return {"data": img.repeat(data_config.batch_size, *([1] * (img.ndim - 1)))}

    def download(self, data_path: str | Path = Path("./data")) -> Path:
        """No download needed for synthetic data."""
        return Path(data_path)

    def _generate_synthetic_signal(
        self, device: torch.device, size: int | tuple[int, ...], channels: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Generate a synthetic n-dimensional signal with geometric patterns.

        Creates a signal with hypersphere patterns and gradients along each axis,
        generalising naturally from 1D to arbitrary spatial dimensions.

        Parameters
        ----------
        device : torch.device
            Device to create the tensor on.
        size : int | tuple[int, ...]
            Spatial dimensions (d1, d2, ..., dn) or a single integer for square 2d images.
        channels : int
            Number of channels.
        dtype : torch.dtype
            Data type of the output tensor.

        Returns
        -------
        torch.Tensor
            Synthetic signal of shape (1, channels, d1, d2, ..., dn).
        """
        if isinstance(size, int):
            size = (size, size)
        ndim = len(size)

        # Build normalised coordinate grids in [-1, 1] for each spatial dimension
        axes = [torch.linspace(-1, 1, s, device=device) for s in size]
        grids = torch.meshgrid(*axes, indexing="ij")  # each of shape (*size)

        # Per-channel patterns: hypersphere + linear gradient along a distinct axis
        channels_list = []
        for c in range(channels):
            # Shift the centre of the hypersphere slightly per channel
            shift = 0.2 * (c / max(channels - 1, 1) - 0.5)
            r2 = sum((g - shift) ** 2 for g in grids)
            radius = r2.sqrt()

            ch = (radius < 0.4).to(dtype) * 0.8
            # Add a linear gradient along the axis corresponding to this channel (cyclic)
            ch = ch + grids[c % ndim] * 0.3 + 0.3
            # Add a high-frequency sinusoidal pattern modulated across all axes
            sinusoid = torch.ones(size, device=device, dtype=dtype)
            for g in grids:
                sinusoid = sinusoid * torch.sin(g * 8 * torch.pi)
            ch = ch + sinusoid * 0.1

            ch = torch.clamp(ch, 0, 1)
            channels_list.append(ch)

        # Stack channels and add batch dim: (1, C, d1, ..., dn)
        img = torch.stack(channels_list, dim=0).unsqueeze(0)
        return img.to(dtype)
    