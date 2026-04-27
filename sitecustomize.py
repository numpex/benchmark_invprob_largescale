"""Global Python startup customizations for benchmark worker processes.

This module is imported automatically by Python at startup (when available on
``sys.path``). We define Dragon symbols eagerly imported by SimAI-Bench so
non-Dragon environments (e.g., local Ray workers) can import cleanly.
"""

import builtins
import os
import re
import typing


def _fix_cuda_visible_devices() -> None:
    """Recover CUDA_VISIBLE_DEVICES from SLURM env vars at Python startup.

    Ray (and some SLURM configurations) spawn worker processes with
    ``CUDA_VISIBLE_DEVICES=""`` (empty string), which hides all GPUs from the
    CUDA runtime.  The CUDA runtime reads this variable on first use, so it
    must be corrected **before** any CUDA library (including torch) is imported.
    Doing this in sitecustomize.py guarantees it runs first.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    # Non-empty value means GPUs are already explicitly assigned — leave it.
    if visible is not None and visible.strip():
        return

    for env_name in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS", "NVIDIA_VISIBLE_DEVICES"):
        raw = os.environ.get(env_name, None)
        if not raw:
            continue
        raw = raw.strip()
        if not raw or raw in {"None", "none", "null", "NoDevFiles", "N/A"}:
            continue
        # Preserve UUID-style identifiers (e.g. "GPU-abc123") as-is.
        if "GPU-" in raw or "MIG-" in raw:
            os.environ["CUDA_VISIBLE_DEVICES"] = raw
            return
        gpu_ids = re.findall(r"\d+", raw)
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
            return


_fix_cuda_visible_devices()


if not hasattr(builtins, "Task"):
    builtins.Task = object
if not hasattr(builtins, "Any"):
    builtins.Any = typing.Any
if not hasattr(builtins, "Sequence"):
    builtins.Sequence = typing.Sequence
