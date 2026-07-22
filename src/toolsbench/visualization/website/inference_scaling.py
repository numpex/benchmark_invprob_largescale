"""Plot-ready website data for the distributed PnP inference finding.

Scientific selection, aggregation, scaling baselines, and overlap-work
calculations live here. The Astro site only maps fields to visual channels.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from toolsbench.visualization.common import (
    TIMING_WARMUP_ITERATIONS,
    add_hardware_columns,
    best_per_gpu,
    load_results,
    summarize_configs,
)

SCHEMA_VERSION = 1
FINDING_ID = "inference_scaling"
SCALING_CONFIG = "benchmark_inference/configs/experiments/strong_scaling_inference.yml"
COMMUNICATION_REQUIRED_COLUMNS = {
    "p_solver_slurm_nodes",
    "p_solver_slurm_gres",
    "p_solver_distribute_physics",
    "p_solver_distribute_denoiser",
    "stop_val",
    "objective_total_time_sec",
    "objective_gradient_cuda_sec",
    "objective_denoise_cuda_sec",
    "objective_gradient_comm_sec",
    "objective_denoise_comm_sec",
    "objective_comm_cuda_sec",
    "objective_comm_sync_sec",
}


def create_inference_scaling_website_data(
    *,
    scaling_results: str | Path,
    comm_2d_results: str | Path,
    comm_3d_results: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Create the plot-ready datasets used by the inference scaling finding."""
    scaling_df, scaling_path = load_results(scaling_results)
    scaling_summary = summarize_configs(scaling_df)
    _validate_scaling_recipe(scaling_summary)
    comm_2d_df, comm_2d_path = _load_communication_results(comm_2d_results)
    comm_3d_df, comm_3d_path = _load_communication_results(comm_3d_results)

    finding_dir = Path(output_dir) / FINDING_ID
    finding_dir.mkdir(parents=True, exist_ok=True)
    scaling_provenance = _provenance(scaling_df, scaling_path, SCALING_CONFIG)
    communication_provenance = {
        "sources": [
            _provenance(comm_2d_df, comm_2d_path, None),
            _provenance(comm_3d_df, comm_3d_path, None),
        ]
    }
    outputs = [
        (
            finding_dir / "timing-breakdown.json",
            _timing_breakdown_payload(scaling_summary, scaling_provenance),
        ),
        (
            finding_dir / "quality-preservation.json",
            _quality_preservation_payload(scaling_df, scaling_provenance),
        ),
        (
            finding_dir / "scaling-efficiency.json",
            _scaling_efficiency_payload(scaling_summary, scaling_provenance),
        ),
        (
            finding_dir / "communication-scaling.json",
            _communication_payload(comm_2d_df, comm_3d_df, communication_provenance),
        ),
    ]
    for path, payload in outputs:
        _write_json(path, payload)
    return [path for path, _ in outputs]


def _validate_scaling_recipe(summary: pd.DataFrame) -> None:
    """Reject ambiguous inputs rather than silently selecting configurations."""
    expected_sizes = {2048, 4096, 8192}
    actual_sizes = set(summary["p_dataset_image_size"].astype(int).unique())
    if actual_sizes != expected_sizes:
        raise ValueError(
            f"Expected image sizes {sorted(expected_sizes)}, "
            f"found {sorted(actual_sizes)}"
        )
    recipe_columns = [
        "p_solver_patch_size",
        "p_solver_overlap",
        "p_solver_max_batch_size",
    ]
    distributed = summary[summary["p_solver_distribute_denoiser"].astype(bool)]
    configurations = distributed[recipe_columns].drop_duplicates()
    if len(configurations) != 1:
        raise ValueError(
            "The website recipe requires one patch/overlap/batch configuration; "
            "filter the parquet explicitly before exporting."
        )
    actual = tuple(int(configurations.iloc[0][column]) for column in recipe_columns)
    expected = (448, 32, 32)
    if actual != expected:
        raise ValueError(f"Expected patch/overlap/batch {expected}, found {actual}")


def _timing_breakdown_payload(
    summary: pd.DataFrame, provenance: dict[str, Any]
) -> dict[str, Any]:
    values = []
    for _, row in summary.sort_values(["p_dataset_image_size", "n_gpus"]).iterrows():
        physics = _finite(row["avg_gradient_time_sec"])
        denoising = _finite(row["avg_denoise_time_sec"])
        total = _finite(row["avg_total_time_sec"])
        values.append(
            {
                "imageSize": int(row["p_dataset_image_size"]),
                "gpuCount": int(row["n_gpus"]),
                "mode": (
                    "distributed"
                    if bool(row["p_solver_distribute_denoiser"])
                    else "non-distributed"
                ),
                "workMultiplier": _number(_row_work_multiplier(row)),
                "physicsSec": physics,
                "denoisingSec": denoising,
                "overheadSec": _number(max(total - physics - denoising, 0.0)),
                "totalSec": total,
            }
        )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": "inference-scaling-timing-breakdown",
        "provenance": provenance,
        "methodology": {
            "aggregation": (
                "mean over iterations 3-10 within one benchmark run; "
                "iterations 1-2 excluded"
            ),
            "timingStartIteration": int(summary["timing_start_iter"].min()),
            "timingEndIteration": int(summary["timing_end_iter"].max()),
            "independentRepetitions": 1,
            "overheadDefinition": "total iteration minus physics and denoising wall time",
        },
        "values": values,
    }


def _quality_preservation_payload(
    scaling_df: pd.DataFrame, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Compare PSNR trajectories where a single-process reference exists."""
    required = {
        "p_dataset_image_size",
        "n_gpus",
        "stop_val",
        "objective_psnr",
        "p_solver_distribute_denoiser",
    }
    missing = sorted(required.difference(scaling_df.columns))
    if missing:
        raise ValueError(f"Missing quality columns: {', '.join(missing)}")

    single_process_sizes = scaling_df.loc[
        ~scaling_df["p_solver_distribute_denoiser"].astype(bool),
        "p_dataset_image_size",
    ].unique()
    quality_df = scaling_df[
        scaling_df["p_dataset_image_size"].isin(single_process_sizes)
    ]
    trajectories = (
        quality_df.dropna(subset=["objective_psnr"])
        .groupby(["p_dataset_image_size", "n_gpus", "stop_val"], dropna=False)[
            "objective_psnr"
        ]
        .mean()
        .reset_index()
    )
    values: list[dict[str, Any]] = []
    baseline_by_size: dict[str, int] = {}
    max_absolute_difference = 0.0
    for image_size, group in trajectories.groupby("p_dataset_image_size", sort=True):
        baseline_gpus = int(group["n_gpus"].min())
        baseline_by_size[str(int(image_size))] = baseline_gpus
        reference = (
            group[group["n_gpus"] == baseline_gpus][["stop_val", "objective_psnr"]]
            .rename(columns={"objective_psnr": "reference_psnr"})
            .copy()
        )
        comparisons = group[group["n_gpus"] != baseline_gpus]
        compared = comparisons.merge(
            reference, on="stop_val", how="inner", validate="many_to_one"
        )
        if len(compared) != len(comparisons):
            raise ValueError(f"Incomplete PSNR reference trajectory for {image_size}")
        for _, row in compared.sort_values(["n_gpus", "stop_val"]).iterrows():
            difference = float(row["objective_psnr"] - row["reference_psnr"])
            max_absolute_difference = max(max_absolute_difference, abs(difference))
            values.append(
                {
                    "imageSize": int(image_size),
                    "iteration": int(row["stop_val"]),
                    "gpuCount": int(row["n_gpus"]),
                    "baselineGpuCount": baseline_gpus,
                    "psnrDb": _finite(row["objective_psnr"]),
                    "baselinePsnrDb": _finite(row["reference_psnr"]),
                    "psnrDifferenceDb": _number(difference),
                }
            )

    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": "inference-scaling-quality-preservation",
        "provenance": provenance,
        "methodology": {
            "comparison": (
                "PSNR at each iteration minus PSNR from the single-process "
                "configuration at the same iteration; image sizes without a "
                "single-process reference are excluded"
            ),
            "baselineGpuCountByImageSize": baseline_by_size,
            "iterations": sorted(
                int(value) for value in trajectories["stop_val"].unique()
            ),
            "independentRepetitions": 1,
            "maxAbsoluteDifferenceDb": _number(max_absolute_difference),
        },
        "values": values,
    }


def _scaling_efficiency_payload(
    summary: pd.DataFrame, provenance: dict[str, Any]
) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for image_size, group in summary.groupby("p_dataset_image_size", sort=True):
        rows = best_per_gpu(group.copy())
        baseline = rows.iloc[0]
        baseline_gpus = int(baseline["n_gpus"])
        baseline_time = float(baseline["avg_total_time_sec"])
        baseline_work = _row_work_multiplier(baseline)
        for _, row in rows.iterrows():
            gpu_count = int(row["n_gpus"])
            total_time = float(row["avg_total_time_sec"])
            work = _row_work_multiplier(row)
            speedup = baseline_time / total_time
            absolute_efficiency = speedup * baseline_gpus / gpu_count * 100
            work_normalized_efficiency = absolute_efficiency * work / baseline_work
            values.append(
                {
                    "imageSize": int(image_size),
                    "gpuCount": gpu_count,
                    "baselineGpuCount": baseline_gpus,
                    "totalSec": _number(total_time),
                    "speedup": _number(speedup),
                    "absoluteEfficiencyPct": _number(absolute_efficiency),
                    "workNormalizedEfficiencyPct": _number(work_normalized_efficiency),
                    "workMultiplier": _number(work),
                    "usefulComputePct": _number(100 / work),
                }
            )
    distributed_recipe = summary[
        summary["p_solver_distribute_denoiser"].astype(bool)
    ].iloc[0]
    patch_size = int(distributed_recipe["p_solver_patch_size"])
    overlap = int(distributed_recipe["p_solver_overlap"])
    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": "inference-scaling-efficiency",
        "provenance": provenance,
        "methodology": {
            "absoluteEfficiency": "(baseline time / time) * (baseline GPUs / GPUs)",
            "workNormalizedEfficiency": (
                "absolute efficiency * current work multiplier / baseline work multiplier"
            ),
            "tileWork": (
                "number of halo-padded denoiser tiles times tile area; the "
                "non-distributed full-image baseline has multiplier 1"
            ),
            "patchSize": patch_size,
            "overlapRadius": overlap,
            "independentRepetitions": 1,
        },
        "values": values,
    }


def _row_work_multiplier(row: pd.Series) -> float:
    if not bool(row["p_solver_distribute_denoiser"]):
        return 1.0
    return _overlap_work_multiplier(
        image_size=int(row["p_dataset_image_size"]),
        patch_size=int(row["p_solver_patch_size"]),
        overlap=int(row["p_solver_overlap"]),
        dimensions=2,
    )


def _overlap_work_multiplier(
    *, image_size: int, patch_size: int, overlap: int, dimensions: int
) -> float:
    """Return halo-padded tile elements divided by useful image elements."""
    if min(image_size, patch_size) <= 0 or overlap < 0 or dimensions <= 0:
        raise ValueError("Image, patch, overlap, and dimension values are invalid")
    tiles_per_axis = math.ceil(image_size / patch_size)
    processed = (tiles_per_axis * (patch_size + 2 * overlap)) ** dimensions
    return processed / image_size**dimensions


def _load_communication_results(
    results: str | Path,
) -> tuple[pd.DataFrame, Path]:
    path = Path(results)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        path = files[-1]
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path).copy()
    missing = sorted(COMMUNICATION_REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Missing communication columns: {', '.join(missing)}")
    add_hardware_columns(df)
    return df, path


def _communication_payload(
    comm_2d: pd.DataFrame,
    comm_3d: pd.DataFrame,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    values = _summarize_communication(comm_2d, "2D 4096x4096")
    values.extend(_summarize_communication(comm_3d, "3D 512x512x512"))
    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": "inference-communication-scaling",
        "provenance": provenance,
        "methodology": {
            "aggregation": (
                f"mean over iterations strictly after iteration "
                f"{TIMING_WARMUP_ITERATIONS} within one benchmark run"
            ),
            "computeCuda": (
                "physics and denoising CUDA section durations minus communication "
                "kernels attributed to those sections"
            ),
            "communicationCuda": (
                "NCCL/c10d device time attributed to physics and denoising sections"
            ),
            "synchronizationCuda": (
                "communication device time outside named sections; reported "
                "separately because it primarily reflects rank skew"
            ),
            "caution": (
                "CUDA section spans and communication kernels can overlap; compute "
                "and communication are diagnostics, not an additive wall-time split"
            ),
            "independentRepetitions": 1,
        },
        "values": values,
    }


def _summarize_communication(df: pd.DataFrame, problem: str) -> list[dict[str, Any]]:
    measured = df[
        (df["stop_val"] > TIMING_WARMUP_ITERATIONS)
        & df["objective_total_time_sec"].notna()
    ].copy()
    group_columns = [
        "n_gpus",
        "n_nodes",
        "p_solver_distribute_physics",
        "p_solver_distribute_denoiser",
    ]
    metric_columns = [
        "objective_total_time_sec",
        "objective_gradient_cuda_sec",
        "objective_denoise_cuda_sec",
        "objective_gradient_comm_sec",
        "objective_denoise_comm_sec",
        "objective_comm_cuda_sec",
        "objective_comm_sync_sec",
    ]
    summary = measured.groupby(group_columns, dropna=False)[metric_columns].mean()
    summary = summary.reset_index().sort_values(
        ["p_solver_distribute_denoiser", "n_gpus"]
    )
    distributed = summary[
        summary["p_solver_distribute_physics"] & summary["p_solver_distribute_denoiser"]
    ]
    baseline_compute = None
    baseline_gpus = None
    if not distributed.empty:
        baseline_row = distributed.sort_values("n_gpus").iloc[0]
        baseline_compute = _compute_cuda_seconds(baseline_row)
        baseline_gpus = int(baseline_row["n_gpus"])

    values = []
    for _, row in summary.iterrows():
        is_distributed = bool(
            row["p_solver_distribute_physics"] and row["p_solver_distribute_denoiser"]
        )
        compute = _compute_cuda_seconds(row)
        communication = float(row["objective_comm_cuda_sec"])
        section_cuda = float(
            row["objective_gradient_cuda_sec"] + row["objective_denoise_cuda_sec"]
        )
        values.append(
            {
                "problem": problem,
                "gpuCount": int(row["n_gpus"]),
                "nodeCount": int(row["n_nodes"]),
                "mode": "distributed" if is_distributed else "non-distributed",
                "iterationWallSec": _finite(row["objective_total_time_sec"]),
                "computeCudaSec": _number(compute),
                "communicationCudaSec": _number(communication),
                "synchronizationCudaSec": _finite(row["objective_comm_sync_sec"]),
                "communicationSharePct": _number(
                    100 * communication / section_cuda if section_cuda else 0
                ),
                "computeSpeedup": (
                    _number(baseline_compute / compute)
                    if is_distributed and baseline_compute
                    else None
                ),
                "idealComputeSpeedup": (
                    _number(int(row["n_gpus"]) / baseline_gpus)
                    if is_distributed and baseline_gpus
                    else None
                ),
            }
        )
    return values


def _compute_cuda_seconds(row: pd.Series) -> float:
    physics = max(
        float(row["objective_gradient_cuda_sec"])
        - float(row["objective_gradient_comm_sec"]),
        0.0,
    )
    denoising = max(
        float(row["objective_denoise_cuda_sec"])
        - float(row["objective_denoise_comm_sec"]),
        0.0,
    )
    return physics + denoising


def _provenance(df: pd.DataFrame, path: Path, config: str | None) -> dict[str, Any]:
    metadata = {
        "source": path.as_posix(),
        "sourceSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "config": config,
    }
    for source_column, output_key in [
        ("run_date", "runDate"),
        ("version-cuda", "accelerator"),
        ("platform", "platform"),
        ("benchmark-git-tag", "benchmarkGitTag"),
    ]:
        if source_column not in df.columns:
            continue
        values = df[source_column].dropna().unique()
        metadata[output_key] = _json_scalar(values[0]) if len(values) == 1 else None
    return metadata


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _number(value: float) -> float:
    return round(float(value), 6)


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Website result data must be finite, got {value!r}")
    return _number(number)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)
