import pytest
import torch
from pathlib import Path
from unittest.mock import patch

from toolsbench.data import (
    DataConfig,
    HighResColorImagingData,
    SyntheticData,
    Tomography2D,
    Tomography3D,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _2d_config(tmp_path, batch_size=2, h=64, w=64, channels=3):
    return DataConfig(
        size=(h, w),
        batch_size=batch_size,
        channels=channels,
        data_type=torch.float32,
        device=torch.device("cpu"),
        data_path=str(tmp_path),
    )


# ---------------------------------------------------------------------------
# SyntheticData
# ---------------------------------------------------------------------------


class TestSyntheticData:

    def test_output_shape_2d(self):
        cfg = DataConfig(size=(32, 48), batch_size=3, channels=2)
        data = SyntheticData().get_data(cfg)
        assert data["data"].shape == (3, 2, 32, 48)

    def test_output_shape_3d(self):
        cfg = DataConfig(size=(8, 16, 16), batch_size=2, channels=4)
        data = SyntheticData().get_data(cfg)
        assert data["data"].shape == (2, 4, 8, 16, 16)

    def test_output_dtype(self):
        cfg = DataConfig(
            size=(16, 16), batch_size=1, channels=1, data_type=torch.float16
        )
        data = SyntheticData().get_data(cfg)
        assert data["data"].dtype == torch.float16

    def test_output_range(self):
        cfg = DataConfig(size=(32, 32), batch_size=1, channels=3)
        data = SyntheticData().get_data(cfg)
        assert data["data"].min() >= 0.0
        assert data["data"].max() <= 1.0

    def test_no_download_needed(self):
        SyntheticData().download()


# ---------------------------------------------------------------------------
# HighResColorImagingData
# ---------------------------------------------------------------------------


class TestHighResColorImagingData:

    def test_download_and_shape(self, tmp_path):
        cfg = _2d_config(tmp_path, batch_size=2, h=64, w=64, channels=3)
        data = HighResColorImagingData().get_data(cfg)
        assert data["data"].shape == (2, 3, 64, 64)

    def test_file_cached_after_first_call(self, tmp_path):
        cfg = _2d_config(tmp_path, h=32, w=32)
        HighResColorImagingData().get_data(cfg)
        assert (tmp_path / "butterfly.png").exists()
        HighResColorImagingData().get_data(cfg)

    def test_raises_on_non_2d(self, tmp_path):
        cfg = DataConfig(size=(8, 16, 16), data_path=str(tmp_path))
        with pytest.raises(ValueError, match="2D"):
            HighResColorImagingData().get_data(cfg)

    def test_output_dtype(self, tmp_path):
        cfg = _2d_config(tmp_path, h=32, w=32)
        data = HighResColorImagingData().get_data(cfg)
        assert data["data"].dtype == torch.float32


# ---------------------------------------------------------------------------
# Tomography2D
# ---------------------------------------------------------------------------


class TestTomography2D:

    def test_download_and_shape(self, tmp_path):
        cfg = _2d_config(tmp_path, batch_size=2, h=64, w=64, channels=1)
        data = Tomography2D().get_data(cfg)
        assert data["data"].shape[0] == 2
        assert data["data"].shape[-2:] == torch.Size([64, 64])

    def test_file_cached_after_first_call(self, tmp_path):
        cfg = _2d_config(tmp_path, h=32, w=32)
        Tomography2D().get_data(cfg)
        assert (tmp_path / "SheppLogan.png").exists()
        Tomography2D().get_data(cfg)

    def test_raises_on_non_2d(self, tmp_path):
        cfg = DataConfig(size=(8, 16, 16), data_path=str(tmp_path))
        with pytest.raises(ValueError, match="2D"):
            Tomography2D().get_data(cfg)

    def test_output_dtype(self, tmp_path):
        cfg = _2d_config(tmp_path, h=32, w=32)
        data = Tomography2D().get_data(cfg)
        assert data["data"].dtype == torch.float32


# ---------------------------------------------------------------------------
# Tomography3D  (download mocked — no network required)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tomo3d_dataset():
    return {
        "dense_reconstruction": torch.zeros(50, 60, 60),  # (D, H, W)
        "sinogram": torch.zeros(80, 30, 25),  # (angles, det_h, det_v)
        "vecs": torch.zeros(80, 12),  # (angles, 12)
    }


def _mock_tomo3d_download(dataset):
    def _write_dataset(data_path=Path("./data")):
        cache_path = Path(data_path) / Tomography3D._FILENAME
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            torch.save(dataset, cache_path)
        return cache_path

    return patch(
        "toolsbench.data.tomography_3d.Tomography3D.download",
        side_effect=_write_dataset,
    )


class TestTomography3D:

    def test_output_keys(self, tmp_path, mock_tomo3d_dataset):
        with _mock_tomo3d_download(mock_tomo3d_dataset):
            cfg = DataConfig(size=(50, 60, 60), data_path=str(tmp_path))
            with pytest.warns(UserWarning, match="size is ignored"):
                data = Tomography3D().get_data(cfg)
        assert set(data.keys()) == {"ground_truth", "sinogram", "vecs"}

    def test_output_shapes(self, tmp_path, mock_tomo3d_dataset):
        with _mock_tomo3d_download(mock_tomo3d_dataset):
            cfg = DataConfig(size=(50, 60, 60), batch_size=2, data_path=str(tmp_path))
            with pytest.warns(UserWarning):
                data = Tomography3D().get_data(cfg)
        assert data["ground_truth"].shape == (2, 1, 50, 60, 60)
        assert data["sinogram"].shape == (2, 80, 30, 25)
        assert data["vecs"].shape == (80, 12)

    def test_warns_about_size(self, tmp_path, mock_tomo3d_dataset):
        with _mock_tomo3d_download(mock_tomo3d_dataset):
            cfg = DataConfig(size=(64, 64, 64), data_path=str(tmp_path))
            with pytest.warns(UserWarning, match="size is ignored"):
                Tomography3D().get_data(cfg)

    def test_output_dtype(self, tmp_path, mock_tomo3d_dataset):
        with _mock_tomo3d_download(mock_tomo3d_dataset):
            cfg = DataConfig(
                size=(50, 60, 60), data_path=str(tmp_path), data_type=torch.float32
            )
            with pytest.warns(UserWarning):
                data = Tomography3D().get_data(cfg)
        assert data["ground_truth"].dtype == torch.float32
        assert data["sinogram"].dtype == torch.float32
        assert data["vecs"].dtype == torch.float32

    def test_device_transfer(self, tmp_path, mock_tomo3d_dataset):
        with _mock_tomo3d_download(mock_tomo3d_dataset):
            cfg = DataConfig(size=(50, 60, 60), data_path=str(tmp_path), device="cpu")
            with pytest.warns(UserWarning):
                data = Tomography3D().get_data(cfg)
        for v in data.values():
            assert v.device.type == "cpu"
