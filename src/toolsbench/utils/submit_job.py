"""Submit or run radio data generation jobs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import yaml


def get_repo_root() -> Path:
    """Return benchmark_invprob_largescale root."""
    return Path(__file__).resolve().parents[3]


def resolve_config_path(config_arg: str) -> Path:
    config_path = Path(config_arg)
    if config_path.exists():
        return config_path

    candidate = get_repo_root() / config_arg
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Config file not found: {config_arg}")


def resolve_image_path(config: dict, image_override: str | None) -> Path:
    if image_override:
        image_path = Path(image_override).expanduser()
    else:
        image_path = Path(config["singularity"]["image_path"]).expanduser()
        if not image_path.is_absolute():
            image_path = get_repo_root() / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Container image not found: {image_path}")

    return image_path


def run_simulation(
    config: dict,
    image_override: str | None = None,
) -> None:
    """Run generation inside the container (local or compute node)."""
    repo_root = get_repo_root()
    image_path = resolve_image_path(config, image_override)

    runtime = shutil.which("apptainer") or shutil.which("singularity")
    if runtime is None:
        raise RuntimeError("Neither apptainer nor singularity is available.")

    working_dir = config["singularity"].get("working_dir", "/workspace")
    mount_point = config["singularity"].get("mount_point", "/workspace")

    cache_dir = repo_root / "debug_output" / "cache"
    mpl_dir = repo_root / "debug_output" / "mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)

    container_cache = f"{mount_point}/debug_output/cache"
    container_mpl = f"{mount_point}/debug_output/mpl_cache"
    config_path = f"{working_dir}/install_scripts/config_slurm.yaml"

    cmd = [
        runtime,
        "exec",
        "--nv",
        "-B",
        f"{repo_root}:{mount_point}",
        "--env",
        f"XDG_CACHE_HOME={container_cache},MPLCONFIGDIR={container_mpl}",
        "--pwd",
        working_dir,
        str(image_path),
        "python",
        f"{mount_point}/src/toolsbench/utils/generate_radio_data.py",
        "--config",
        config_path,
    ]

    print(f"Running command: {' '.join(cmd)}", flush=True)
    # Stream logs directly to Slurm output/error files.
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Simulation failed with code {result.returncode}")


def run_simulation_with_params(
    config: dict,
    job_params: dict,
    image_override: str | None = None,
) -> None:
    """Run generation inside the container with explicit job parameters.

    Unlike ``run_simulation()``, which reads a fixed YAML config file, this
    function accepts job parameters as a dict, writes a temporary YAML
    (accessible from the container through the bind-mount), and passes it to
    ``generate_radio_data.py``.

    Parameters
    ----------
    config : dict
        Must contain a ``singularity`` key with at least ``image_path``, and
        optionally ``mount_point`` and ``working_dir``.
    job_params : dict
        Parameters for the ``job`` section consumed by
        ``generate_radio_data.py``. Must include at least ``fits_name``,
        ``image_size`` (a list), and ``data_path`` (container-absolute path).
    image_override : str or None
        Override the container image path from config.
    """
    repo_root = get_repo_root()
    image_path = resolve_image_path(config, image_override)

    runtime = shutil.which("apptainer") or shutil.which("singularity")
    if runtime is None:
        raise RuntimeError("Neither apptainer nor singularity is available.")

    working_dir = config["singularity"].get("working_dir", "/workspace")
    mount_point = config["singularity"].get("mount_point", "/workspace")

    cache_dir = repo_root / "debug_output" / "cache"
    mpl_dir = repo_root / "debug_output" / "mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)

    container_cache = f"{mount_point}/debug_output/cache"
    container_mpl = f"{mount_point}/debug_output/mpl_cache"

    # Write a temporary job config inside the bind-mounted repo so the
    # container can read it at a deterministic container-absolute path.
    temp_dir = repo_root / "benchmark_inference" / "data" / "radio_interferometry"
    temp_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=temp_dir,
            prefix="_prepare_",
            suffix=".yaml",
            mode="w",
            delete=False,
        ) as tmp:
            yaml.dump({"job": job_params}, tmp)
            tmp_path = Path(tmp.name)

        container_config_path = (
            f"{mount_point}/{tmp_path.relative_to(repo_root)}"
        )

        cmd = [
            runtime,
            "exec",
            "--nv",
            "-B",
            f"{repo_root}:{mount_point}",
            "--env",
            f"XDG_CACHE_HOME={container_cache},MPLCONFIGDIR={container_mpl}",
            "--pwd",
            working_dir,
            str(image_path),
            "python",
            f"{mount_point}/src/toolsbench/utils/generate_radio_data.py",
            "--config",
            container_config_path,
        ]

        print(f"Running command: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"Simulation failed with return code {result.returncode}"
            )
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def submit_slurm_job(
    config: dict,
    image_override: str | None = None,
) -> None:
    try:
        import submitit
    except ImportError as exc:
        raise RuntimeError(
            "Slurm mode requires submitit. Install it or run with --local."
        ) from exc

    slurm_conf = config.get("slurm", {})
    folder = slurm_conf.get("folder", "logs")
    Path(folder).mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=folder)

    kwargs = {}
    keys = [
        "job_name",
        "time",
        "nodes",
        "gres",
        "cpus_per_task",
        "gpus_per_task",
        "ntasks_per_node",
        "mem",
        "partition",
        "account",
    ]
    for key in keys:
        value = slurm_conf.get(key)
        if value not in (None, ""):
            kwargs[f"slurm_{key}"] = value

    cpus_per_gpu = slurm_conf.get("cpus_per_gpu")
    if cpus_per_gpu not in (None, ""):
        gpus_per_task = slurm_conf.get("gpus_per_task")
        if gpus_per_task not in (None, ""):
            kwargs["slurm_cpus_per_gpu"] = cpus_per_gpu
        else:
            print(
                "Ignoring slurm.cpus_per_gpu because slurm.gpus_per_task is not set.",
                flush=True,
            )

    additional = {}
    for key in ["hint", "constraint", "qos"]:
        value = slurm_conf.get(key)
        if value not in (None, ""):
            additional[key] = value
    if additional:
        kwargs["slurm_additional_parameters"] = additional

    if "setup" in slurm_conf:
        kwargs["slurm_setup"] = slurm_conf["setup"]

    executor.update_parameters(**kwargs)
    print(f"Submitting Slurm job with parameters: {kwargs}")
    job = executor.submit(run_simulation, config, image_override)
    print(f"Submitted job {job.job_id}, waiting for completion...")

    poll_interval_seconds = int(slurm_conf.get("poll_interval_seconds", 30))
    wait_timeout_seconds = int(slurm_conf.get("wait_timeout_seconds", 7200))
    deadline = time.time() + wait_timeout_seconds

    def is_done() -> bool:
        done_attr = getattr(job, "done", None)
        if callable(done_attr):
            try:
                return bool(done_attr())
            except TypeError:
                pass
        elif done_attr is not None:
            return bool(done_attr)

        state = str(getattr(job, "state", "")).upper()
        return state in {"DONE", "FAILED", "CANCELLED", "TIMEOUT"}

    while not is_done():
        state = "unknown"
        try:
            state = str(job.state)
        except Exception:
            pass
        print(
            f"Job {job.job_id} still running (state={state}). "
            f"Polling every {poll_interval_seconds}s.",
            flush=True,
        )
        if time.time() >= deadline:
            try:
                job.cancel()
            except Exception:
                pass
            raise TimeoutError(
                f"Job {job.job_id} exceeded wait_timeout_seconds="
                f"{wait_timeout_seconds}."
            )
        time.sleep(poll_interval_seconds)

    # Re-raise any remote exception if the Slurm job failed.
    job.result()
    print("Job completed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit or run Karabo radio data generation"
    )
    parser.add_argument(
        "--config",
        default="install_scripts/config_slurm.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of submitting to Slurm",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Override container image path",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    print(f"Loading config from {config_path}")
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if args.local:
        print("Running locally...")
        run_simulation(config, args.image_path)
    else:
        submit_slurm_job(config, args.image_path)


if __name__ == "__main__":
    main()
