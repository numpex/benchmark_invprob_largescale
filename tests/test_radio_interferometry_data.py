"""Tests for selective radio-interferometry dataset downloads."""

from unittest.mock import patch

import pytest

from toolsbench.data import RadioInterferometryData, check_installed


def test_download_only_requested_size(tmp_path):
    dataset = RadioInterferometryData()
    with patch.object(
        dataset, "_download_hf_snapshot", return_value=tmp_path
    ) as download:
        assert dataset.download(tmp_path, fits_size="1024") == tmp_path

    download.assert_called_once_with(
        tmp_path,
        allow_patterns=["1024/*.fits"],
    )


def test_requested_size_uses_cached_file(tmp_path):
    size_dir = tmp_path / "1024"
    size_dir.mkdir()
    (size_dir / "image.fits").touch()
    dataset = RadioInterferometryData()

    with patch.object(dataset, "_download_hf_snapshot") as download:
        assert dataset.download(tmp_path, fits_size="1024") == tmp_path

    download.assert_not_called()


def test_download_without_size_keeps_all_sizes_behavior(tmp_path):
    dataset = RadioInterferometryData()
    with patch.object(
        dataset, "_download_hf_snapshot", return_value=tmp_path
    ) as download:
        dataset.download(tmp_path)

    download.assert_called_once_with(
        tmp_path,
        allow_patterns=["1024/*.fits", "10k/*.fits"],
    )


def test_rejects_unknown_size(tmp_path):
    with pytest.raises(ValueError, match="Unknown radio FITS size"):
        RadioInterferometryData().download(tmp_path, fits_size="2048")


def test_check_installed_forwards_download_options(tmp_path):
    with patch.object(
        RadioInterferometryData,
        "download",
        return_value=tmp_path,
    ) as download:
        assert (
            check_installed(
                "radio_interferometry",
                tmp_path,
                fits_size="1024",
            )
            == tmp_path
        )

    download.assert_called_once_with(tmp_path, fits_size="1024")
