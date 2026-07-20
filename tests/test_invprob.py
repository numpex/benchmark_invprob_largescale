from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
import torch
import numpy as np
from astropy.io import fits

from toolsbench.invprob.base import InvProb, InvProbConfig
from toolsbench.invprob.denoising import DenoisingInvProb
from toolsbench.invprob.multiframe_superres import MultiFrameSuperResInvProb
from toolsbench.invprob.tomography import TomographyInvProb
from toolsbench.utils.radio_interferometry.radio_utils import (
    get_fits_image_size,
    get_meerkat_visibilities_path,
    load_fits_image,
)


class TestRadioFitsLoading:
    def test_preserves_native_size_and_adapts_to_channels_first(self, tmp_path):
        fits_path = tmp_path / "native.fits"
        data = np.arange(35, dtype=np.float32).reshape(1, 5, 7)
        fits.PrimaryHDU(data).writeto(fits_path)

        with pytest.raises(ValueError, match="requires a square FITS image"):
            get_fits_image_size(fits_path)

        square_data = np.arange(25, dtype=np.float32).reshape(1, 5, 5)
        fits.PrimaryHDU(square_data).writeto(fits_path, overwrite=True)

        image = load_fits_image(fits_path)

        assert get_fits_image_size(fits_path) == 5
        assert image.shape == (1, 5, 5)
        assert image.dtype == np.float32
        np.testing.assert_array_equal(image[0], square_data[0])

    def test_sanitizes_non_finite_values_without_resizing(self, tmp_path):
        fits_path = tmp_path / "non_finite.fits"
        data = np.array([[np.nan, np.inf], [-np.inf, 3.0]], dtype=np.float32)
        fits.PrimaryHDU(data).writeto(fits_path)

        image = load_fits_image(fits_path)

        assert image.shape == (1, 2, 2)
        np.testing.assert_array_equal(
            image,
            np.array([[[0.0, 0.0], [0.0, 3.0]]], dtype=np.float32),
        )

    def test_cache_key_uses_source_fits_bytes(self, tmp_path):
        fits_path = tmp_path / "source.fits"
        image = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
        fits.PrimaryHDU(image).writeto(fits_path)

        cache_path = get_meerkat_visibilities_path(
            image, tmp_path, fits_path, imaging_npixel=2
        )
        differently_decoded = image.astype(np.float64) + 1.0
        same_source_path = get_meerkat_visibilities_path(
            differently_decoded, tmp_path, fits_path, imaging_npixel=2
        )
        assert same_source_path == cache_path

        fits.PrimaryHDU(image + 1.0).writeto(fits_path, overwrite=True)
        changed_source_path = get_meerkat_visibilities_path(
            image, tmp_path, fits_path, imaging_npixel=2
        )
        assert changed_source_path != cache_path


class DummyAstraPhysics:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._angle_range = None
        self._operator_idx = None


@pytest.fixture
def mock_tomo3d_dataset():
    return {
        "dense_reconstruction": torch.ones(5, 6, 7),
        "sinogram": torch.ones(8, 3, 4),
        "vecs": torch.ones(8, 12),
    }


class TestMultiFrameSuperResInvProb:
    def test_get_invprob_shapes_dtype_and_num_physics(self):
        cfg = InvProbConfig(
            size=(16, 16),
            batch_size=2,
            channels=1,
            data_type=torch.float32,
            device=torch.device("cpu"),
            params={
                "num_frames": 3,
                "scale_factor": 2,
                "noise_std": 0.0,
                "data": "synthetic",
            },
        )

        invprob = MultiFrameSuperResInvProb().get_invprob(cfg)

        assert invprob.ground_truth.shape == (2, 1, 16, 16)
        assert invprob.ground_truth.dtype == torch.float32
        assert invprob.ground_truth_shape == torch.Size((2, 1, 16, 16))
        assert invprob.num_operators == 3
        assert len(invprob.physics) == 3
        assert len(invprob.measurements) == 3

        for measurement in invprob.measurements:
            assert measurement.shape == (2, 1, 8, 8)
            assert measurement.dtype == torch.float32

    def test_warns_and_ignores_unknown_params(self):
        cfg = InvProbConfig(
            size=(16, 16),
            params={
                "num_frame": 3,
                "num_frames": 2,
                "noise_std": 0.0,
            },
        )

        with pytest.warns(UserWarning, match="will not be taken into account"):
            invprob = MultiFrameSuperResInvProb().get_invprob(cfg)

        assert invprob.num_operators == 2


class TestTomographyInvProb:
    def test_get_invprob_shapes_dtype_and_num_physics(
        self, tmp_path, mock_tomo3d_dataset
    ):
        cfg = InvProbConfig(
            size=(5, 6, 7),
            batch_size=2,
            channels=1,
            data_type=torch.float32,
            device=torch.device("cpu"),
            data_path=str(tmp_path),
            params={
                "data": "3d",
                "num_operators": 3,
                "num_projections": 6,
            },
        )

        with patch(
            "toolsbench.data.tomography_3d.Tomography3D._get_dataset",
            return_value=mock_tomo3d_dataset,
        ):
            with pytest.warns(UserWarning, match="size is ignored"):
                invprob = TomographyInvProb().get_invprob(cfg)

        assert invprob.ground_truth.shape == (2, 1, 5, 6, 7)
        assert invprob.ground_truth.dtype == torch.float32
        assert invprob.ground_truth_shape == torch.Size((2, 1, 5, 6, 7))
        assert invprob.num_operators == 3
        assert len(invprob.measurements) == 3
        assert callable(invprob.physics)

        for measurement in invprob.measurements:
            assert measurement.shape == (2, 1, 3, 2, 4)
            assert measurement.dtype == torch.float32

        with patch(
            "toolsbench.invprob.tomography.TomographyWithAstra",
            DummyAstraPhysics,
        ):
            physics = [
                invprob.physics(i, torch.device("cpu"), None)
                for i in range(invprob.num_operators)
            ]

        assert len(physics) == 3
        assert [operator._operator_idx for operator in physics] == [0, 1, 2]
        assert [operator._angle_range for operator in physics] == [
            (0, 2),
            (2, 4),
            (4, 6),
        ]


class TestDenoisingInvProb:
    def _cfg(self, size, num_frames=1):
        return InvProbConfig(
            size=size,
            batch_size=1,
            channels=3,
            data_type=torch.float32,
            device=torch.device("cpu"),
            params={"num_frames": num_frames, "noise_std": 0.05, "data": "synthetic"},
        )

    def test_denoising_invprob_3d(self):
        # 3D volume with stacked, independently-noised measurements.
        ip = DenoisingInvProb().get_invprob(self._cfg((4, 16, 16), num_frames=3))
        assert tuple(ip.ground_truth.shape) == (1, 3, 4, 16, 16)
        assert ip.num_operators == 3
        assert len(ip.measurements) == 3
        assert not torch.allclose(ip.measurements[0], ip.measurements[1])

    def test_denoising_invprob_2d(self):
        ip = DenoisingInvProb().get_invprob(self._cfg((16, 16)))
        assert tuple(ip.ground_truth.shape) == (1, 3, 16, 16)
        assert ip.num_operators == 1


class TestInvProbResized:
    def _problem(self, shape, ground_truth=None):
        physics = MagicMock()
        physics.A.return_value = torch.zeros(shape)
        return InvProb(
            measurements=torch.zeros(shape),
            physics=physics,
            ground_truth_shape=torch.Size(shape),
            ground_truth=ground_truth,
        )

    def test_same_size_is_noop(self):
        ip = self._problem((1, 1, 8, 8))
        assert ip.resized([8, 8]) is ip

    def test_resizes_ground_truth_and_recomputes_measurements(self):
        gt = torch.rand(1, 1, 8, 8)
        ip = self._problem((1, 1, 8, 8), ground_truth=gt)
        out = ip.resized([16, 16])
        assert out.ground_truth_shape == torch.Size((1, 1, 16, 16))
        assert out.ground_truth.shape == (1, 1, 16, 16)
        ip.physics.A.assert_called_once()

    def test_no_ground_truth_falls_back_to_random(self):
        ip = self._problem((1, 1, 8, 8))
        out = ip.resized([16, 16], device=torch.device("cpu"))
        assert out.ground_truth.shape == (1, 1, 16, 16)

    def test_resets_imsize_flat_and_nested(self):
        class Leaf:
            imsize = (8, 8)

        leaf = Leaf()
        parent = SimpleNamespace(local_physics=[leaf], A=lambda x: torch.zeros_like(x))
        ip = self._problem((1, 1, 8, 8))
        ip.physics = parent
        ip.resized([16, 16], device=torch.device("cpu"))
        assert leaf.imsize is None
