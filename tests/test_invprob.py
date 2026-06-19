from unittest.mock import patch

import pytest
import torch

from toolsbench.invprob.base import InvProbConfig
from toolsbench.invprob.multiframe_superres import MultiFrameSuperResInvProb
from toolsbench.invprob.tomography import TomographyInvProb


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
