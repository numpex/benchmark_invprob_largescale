from __future__ import annotations

import pandas as pd
import pytest

from toolsbench.visualization.website.inference_scaling import (
    _overlap_work_multiplier,
    _quality_preservation_payload,
    _summarize_communication,
)


@pytest.mark.parametrize(
    ("image_size", "expected"),
    [(2048, 1.5625), (4096, 1.5625), (8192, 1.41015625)],
)
def test_overlap_work_multiplier_matches_deepinv_grid(image_size, expected):
    multiplier = _overlap_work_multiplier(
        image_size=image_size,
        patch_size=448,
        overlap=32,
        dimensions=2,
    )
    assert multiplier == pytest.approx(expected)


def test_quality_preservation_requires_single_process_reference():
    rows = []
    for image_size, gpu_values in {
        2048: {1: [10.0, 20.0], 2: [10.01, 19.99]},
        8192: {2: [11.0, 21.0], 4: [11.0, 21.02]},
    }.items():
        for gpu_count, psnr_values in gpu_values.items():
            for iteration, psnr in enumerate(psnr_values):
                rows.append(
                    {
                        "p_dataset_image_size": image_size,
                        "n_gpus": gpu_count,
                        "stop_val": iteration,
                        "objective_psnr": psnr,
                        "p_solver_distribute_denoiser": not (
                            image_size == 2048 and gpu_count == 1
                        ),
                    }
                )

    payload = _quality_preservation_payload(pd.DataFrame(rows), {})

    assert payload["methodology"]["baselineGpuCountByImageSize"] == {
        "2048": 1,
    }
    assert [value["psnrDifferenceDb"] for value in payload["values"]] == [
        0.01,
        -0.01,
    ]


def test_communication_summary_excludes_initial_iterations():
    rows = []
    for stop_val, total, gradient, denoise, comm in [
        (1, 100.0, 40.0, 60.0, 20.0),
        (2, 100.0, 40.0, 60.0, 20.0),
        (3, 10.0, 4.0, 6.0, 2.0),
        (4, 14.0, 6.0, 8.0, 4.0),
    ]:
        rows.append(
            {
                "n_gpus": 2,
                "n_nodes": 1,
                "p_solver_distribute_physics": True,
                "p_solver_distribute_denoiser": True,
                "stop_val": stop_val,
                "objective_total_time_sec": total,
                "objective_gradient_cuda_sec": gradient,
                "objective_denoise_cuda_sec": denoise,
                "objective_gradient_comm_sec": comm / 2,
                "objective_denoise_comm_sec": comm / 2,
                "objective_comm_cuda_sec": comm,
                "objective_comm_sync_sec": 0.5,
            }
        )

    assert _summarize_communication(pd.DataFrame(rows), "test") == [
        {
            "problem": "test",
            "gpuCount": 2,
            "nodeCount": 1,
            "mode": "distributed",
            "iterationWallSec": 12.0,
            "computeCudaSec": 9.0,
            "communicationCudaSec": 3.0,
            "synchronizationCudaSec": 0.5,
            "communicationSharePct": 25.0,
            "computeSpeedup": 1.0,
            "idealComputeSpeedup": 1.0,
        }
    ]
