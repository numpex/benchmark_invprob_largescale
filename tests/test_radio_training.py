"""Focused tests for the preprocessed radio training data path."""

from dataclasses import fields
from pathlib import Path

import pytest
import torch

from toolsbench.utils.radio_interferometry.data import (
    RadioDataConfig,
    _build_loader,
    build_radio_dataloaders,
    discover_radio_entries,
)
from toolsbench.utils.radio_interferometry.physics import (
    MyRadioInterferometry,
    _balanced_plane_assignment,
    _equal_count_plane_indices,
    _finufft_options,
)
from toolsbench.utils.radio_interferometry import physics as radio_physics

EXAMPLE = Path(__file__).resolve().parents[2] / "demo_radio/radio_dataset_example"


def test_discover_preprocessed_example():
    entries = discover_radio_entries(EXAMPLE, image_size=1024)
    assert len(entries) == 1
    entry = entries[0]
    assert Path(entry.fits_path).is_file()
    assert Path(entry.ms_path).is_dir()
    assert Path(entry.x_init_path).is_file()
    assert Path(entry.norm_path).is_file()


def test_radio_loader_rejects_batching_variable_physics():
    with pytest.raises(ValueError, match="batch_size=1"):
        _build_loader(
            dataset=[],
            sampler=[],
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=1,
            persistent_workers=False,
        )


def test_balanced_plane_assignment_is_complete_and_balanced():
    counts = torch.tensor([100, 90, 20, 10])
    assignments = _balanced_plane_assignment([0, 1, 2, 3], counts, 2)
    assert sorted(item for shard in assignments for item in shard) == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]
    assert [len(shard) for shard in assignments] == [2, 2]
    assert sum(int(counts[plane]) for _, plane in assignments[0]) == 110
    assert sum(int(counts[plane]) for _, plane in assignments[1]) == 110


def test_equal_count_planes_are_balanced_contiguous_and_complete():
    w_lambda = torch.tensor([9.0, -2.0, 4.0, 0.0, 1.0, 8.0, 3.0])
    planes = _equal_count_plane_indices(w_lambda, 3)

    assert [indices.numel() for indices in planes] == [3, 2, 2]
    assert sorted(torch.cat(planes).tolist()) == list(range(w_lambda.numel()))
    for left, right in zip(planes, planes[1:]):
        assert w_lambda[left].max() <= w_lambda[right].min()


def test_nufft_oversampling_is_forwarded_in_both_directions(monkeypatch):
    calls = {}

    def fake_type2(points, targets, **kwargs):
        calls["type2"] = kwargs
        return torch.zeros(
            *targets.shape[:-2],
            points.shape[-1],
            dtype=torch.complex64,
        )

    def fake_type1(points, values, output_shape, **kwargs):
        calls["type1"] = kwargs
        return torch.zeros(
            *values.shape[:-1],
            *output_shape,
            dtype=torch.complex64,
        )

    monkeypatch.setattr(radio_physics.py_nufft.functional, "finufft_type2", fake_type2)
    monkeypatch.setattr(radio_physics.py_nufft.functional, "finufft_type1", fake_type1)
    physics = MyRadioInterferometry(
        img_size=(4, 4),
        samples_loc=torch.zeros(2, 3),
        nufft_k_oversampling=1.25,
    )

    physics.A(torch.ones(1, 1, 4, 4))
    physics.A_adjoint(torch.ones(1, 1, 3, dtype=torch.complex64))

    assert calls["type2"]["upsampfac"] == 1.25
    assert calls["type1"]["upsampfac"] == 1.25


def test_nonstandard_cuda_oversampling_uses_direct_kernel_evaluation():
    assert _finufft_options(1.5, use_cuda=True) == {
        "upsampfac": 1.5,
        "gpu_kerevalmeth": 0,
    }
    assert _finufft_options(1.25, use_cuda=True) == {"upsampfac": 1.25}
    assert _finufft_options(1.5, use_cuda=False) == {"upsampfac": 1.5}


def test_loader_configuration_keeps_torch_options():
    config = RadioDataConfig(
        input_dir=str(EXAMPLE),
        num_workers=3,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    assert config.num_workers == 3
    assert config.pin_memory is True
    assert config.prefetch_factor == 4
    assert config.persistent_workers is True


def test_radio_data_config_exposes_physics_sharding():
    config = RadioDataConfig(
        input_dir=str(EXAMPLE),
        w_stacking=True,
        w_planes=2,
        w_binning="equal_count",
        physics_sharding=True,
    )
    assert config.physics_sharding is True
    assert config.w_planes == 2
    assert config.w_binning == "equal_count"


def test_radio_training_uses_one_unsplit_dataset():
    parameter_names = {field.name for field in fields(RadioDataConfig)}
    assert "train_fraction" not in parameter_names
    assert "max_val_samples" not in parameter_names

    bundle = build_radio_dataloaders(
        RadioDataConfig(input_dir=str(EXAMPLE), num_workers=0)
    )
    assert len(bundle.dataset) == 1
    assert bundle.loader.dataset is bundle.dataset
    assert not hasattr(bundle, "val_loader")
