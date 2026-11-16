"""Website data for the cryo-ET training finding (Adastra MI300A).

Training time is forward + backward, median of steps 3-5 (steps 1-2 are
warm-up). Efficiency is measured against the 4-GPU run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from toolsbench.visualization.website.common import (
    SCHEMA_VERSION,
    finite,
    number,
    provenance,
    write_json,
)

FINDING_ID = "cryo_training"
CONFIGS = "benchmark_training/configs/cryo/"
DEVICE_MEMORY_GB = 128
FIRST_STEP = 3
METHODOLOGY = {
    "trainSec": "forward + backward time, median of steps 3-5",
    "peakMemoryGb": "peak allocated GPU memory of forward and backward",
    "efficiencyPct": "100 x t4 x 4 / (t x gpuCount), t per volume",
    "deviceMemoryGb": DEVICE_MEMORY_GB,
    "miopenFindMode": "FAST (kernels picked by heuristic, no tuning)",
}

#: Runs that do not appear in a parquet, with the reason.
PATCH_SIZE_FAILED = [{"patchSize": 192, "status": "MIOpen int32 limit"}]
CHECKPOINT_OOM = [
    {"checkpointMode": "never", "gpuCount": g, "status": "out of memory"}
    for g in (4, 8, 16, 64)
]

#: (mixed_precision, compile_model, num_operators) -> variant label. Rows
#: not listed (num_operators 2) are left out.
SPEEDUP_VARIANTS = {
    ("off", False, "none"): "baseline",
    ("bf16", False, "none"): "bf16",
    ("off", True, "none"): "compile",
    ("off", False, "auto"): "distributed physics",
    ("bf16", True, "auto"): "all combined",
}


def create_cryo_training_website_data(
    *,
    speedup_results: str | Path,
    patch_size_results: str | Path,
    checkpointing_results: str | Path,
    tiling_results: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Write speedup, patch-size, checkpointing and tiling-vs-DDP JSON."""
    out = Path(output_dir) / FINDING_ID
    out.mkdir(parents=True, exist_ok=True)
    payloads = {
        "speedup": _speedup(Path(speedup_results)),
        "patch-size": _patch_size(Path(patch_size_results)),
        "checkpointing": _checkpointing(Path(checkpointing_results)),
        "tiling-vs-ddp": _tiling(Path(tiling_results)),
    }
    paths = []
    for name, payload in payloads.items():
        write_json(out / f"{name}.json", payload)
        paths.append(out / f"{name}.json")
    return paths


def _steps(path: Path) -> pd.DataFrame:
    """Steps 3-5 with training time and peak memory columns."""
    df = pd.read_parquet(path)
    df["gpuCount"] = df.p_solver_slurm_nodes * df.p_solver_slurm_ntasks_per_node
    df["trainSec"] = df.objective_forward_time_sec + df.objective_backward_time_sec
    mb = ["objective_forward_max_gpu_mb", "objective_backward_max_gpu_mb"]
    df["peakMemoryGb"] = df[mb].max(axis=1) / 1024
    return df[df.stop_val >= FIRST_STEP]


def _summary(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Median times and max memory per configuration."""
    g = df.groupby(keys, dropna=False)
    cols = ["trainSec", "objective_forward_time_sec", "objective_backward_time_sec"]
    out = g[cols].median()
    out["peakMemoryGb"] = g.peakMemoryGb.max()
    return out.reset_index()


def _row(r, **extra) -> dict[str, Any]:
    return {
        "trainSec": finite(r.trainSec),
        "forwardSec": finite(r.objective_forward_time_sec),
        "backwardSec": finite(r.objective_backward_time_sec),
        "peakMemoryGb": finite(r.peakMemoryGb),
        "status": "measured",
        **extra,
    }


def _payload(name: str, config: str, path: Path, df, values: list) -> dict:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": f"cryo-{name}",
        "provenance": provenance(df, path, CONFIGS + config),
        "methodology": METHODOLOGY,
        "values": values,
    }


def _speedup(path: Path) -> dict:
    """One change at a time against fp32, no compile, replicated physics."""
    df = _steps(path)
    df["p_solver_num_operators"] = df.p_solver_num_operators.fillna("none")
    keys = ["p_solver_mixed_precision", "p_solver_compile_model"]
    s = _summary(df, keys + ["p_solver_num_operators"])
    s["variant"] = [SPEEDUP_VARIANTS.get(k) for k in s.iloc[:, :3].itertuples(False)]
    s = s.dropna(subset=["variant"])
    t0 = s[s.variant == "baseline"].trainSec.iloc[0]
    values = [
        _row(
            r,
            variant=r.variant,
            speedup=number(t0 / r.trainSec),
            speedupPct=number(100 * (t0 / r.trainSec - 1)),
        )
        for r in s.itertuples()
    ]
    return _payload("speedup", "speedup.yml", path, df, values)


def _patch_size(path: Path) -> dict:
    df = _steps(path)
    values = [
        _row(r, patchSize=int(r.p_solver_patch_size))
        for r in _summary(df, ["p_solver_patch_size"]).itertuples()
    ]
    return _payload(
        "patch-size", "patch_size.yml", path, df, values + PATCH_SIZE_FAILED
    )


def _checkpointing(path: Path) -> dict:
    df = _steps(path)
    s = _summary(df, ["p_solver_checkpoint_batches", "gpuCount"])
    t4 = s[(s.gpuCount == 4) & (s.p_solver_checkpoint_batches == "always")]
    t4 = t4.trainSec.iloc[0]
    values = [
        _row(
            r,
            checkpointMode=r.p_solver_checkpoint_batches,
            gpuCount=int(r.gpuCount),
            efficiencyPct=number(100 * t4 * 4 / (r.trainSec * r.gpuCount)),
        )
        for r in s.itertuples()
    ]
    return _payload(
        "checkpointing", "checkpointing.yml", path, df, values + CHECKPOINT_OOM
    )


def _tiling(path: Path) -> dict:
    """A DDP step covers gpuCount / gpusPerVolume volumes: compare per volume."""
    df = _steps(path)
    df["gpusPerVolume"] = df.p_solver_inner_world_size.fillna(df.gpuCount)
    s = _summary(df, ["gpuCount", "gpusPerVolume"])
    s["perVolumeSec"] = s.trainSec * s.gpusPerVolume / s.gpuCount
    t4 = s[(s.gpuCount == 4) & (s.gpusPerVolume == 4)].perVolumeSec.iloc[0]
    values = [
        _row(
            r,
            gpuCount=int(r.gpuCount),
            gpusPerVolume=int(r.gpusPerVolume),
            tilesAllGpus=bool(r.gpusPerVolume == r.gpuCount),
            perVolumeSec=finite(r.perVolumeSec),
            efficiencyPct=number(100 * t4 * 4 / (r.perVolumeSec * r.gpuCount)),
        )
        for r in s.itertuples()
    ]
    return _payload("tiling-vs-ddp", "dp_vs_tiling_tomo_ei.yml", path, df, values)
