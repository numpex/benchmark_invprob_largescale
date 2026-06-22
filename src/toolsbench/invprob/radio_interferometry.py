from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from toolsbench.data.radio_interferometry import RadioInterferometryData
from toolsbench.invprob.base import (
    BaseInvProb,
    InvProb,
    InvProbConfig,
    build_problem_params,
)


@dataclass
class RadioSimulationCache:
    """Host-side paths for one radio simulation cache entry."""

    simulation_hash: str
    ms_path: Path
    metadata_path: Path
    resized_fits_path: Path
    source_fits_path: Path

    @property
    def is_complete(self) -> bool:
        return self.ms_path.exists() and self.metadata_path.exists()


@dataclass
class _RadioInterferometryParams:
    image_size: int = 256
    batch_size: int = 1
    fits_size: str = "1024"
    fits_name: str | None = None
    noise_level: float = 0.1
    seed: int = 42
    pos_ra: float = 0.0
    pos_dec: float = 0.0
    random_position: bool = True
    use_gpus: bool = True
    number_of_time_steps: int = 64
    start_frequency_hz: float = 1.300e9
    end_frequency_hz: float = 1.340e9
    number_of_channels: int = 8
    add_noise: bool = True
    pol_mode: str = "Full"
    run_on_slurm: bool = False
    slurm_folder: str = "debug_output/slurm_logs"
    slurm_job_name: str = "karabo_simulator"
    slurm_nodes: int = 1
    slurm_ntasks_per_node: int = 1
    slurm_cpus_per_task: int = 40
    slurm_gres: str = "gpu:4"
    slurm_time: int = 60
    slurm_hint: str = "nomultithread"
    slurm_account: str | None = None
    slurm_constraint: str = "v100-32g"
    slurm_poll_interval_seconds: int = 30
    slurm_wait_timeout_seconds: int = 1200
    slurm_setup: str | list[str] = "module purge\nmodule load singularity\nset -x"


def get_karabo_image_path(repo_root: str | Path | None = None) -> Path:
    allowed_dir = os.environ.get("SINGULARITY_ALLOWED_DIR")
    if allowed_dir:
        return Path(allowed_dir).expanduser() / "karabo.sif"
    elif repo_root is not None:
        return Path(repo_root) / "tools" / "karabo.sif"
    return Path(__file__).resolve().parents[3] / "tools" / "karabo.sif"


def get_meerkat_cache_dir(data_path: str | Path) -> Path:
    cache_dir = Path(data_path) / "meerkat_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_radio_simulation_cache(
    data_path: str | Path,
    params: Mapping[str, Any] | _RadioInterferometryParams | None = None,
    **kwargs: Any,
) -> RadioSimulationCache:
    """Return the deterministic cache paths for a radio simulation."""
    radio_params = _coerce_params(params, kwargs)
    source_fits_path = RadioInterferometryData.select_fits_file(
        data_path,
        fits_size=radio_params.fits_size,
        fits_name=radio_params.fits_name,
    )
    resized_fits_path, resized_img = _ensure_resized_fits(
        source_fits_path=source_fits_path,
        data_path=Path(data_path),
        image_size=int(radio_params.image_size),
    )

    from toolsbench.utils.radio_interferometry.radio_utils import (
        get_meerkat_visibilities_path,
    )

    ms_path = get_meerkat_visibilities_path(
        resized_img,
        get_meerkat_cache_dir(data_path),
        resized_fits_path.name,
        int(radio_params.image_size),
        int(radio_params.number_of_time_steps),
        float(radio_params.start_frequency_hz),
        float(radio_params.end_frequency_hz),
        int(radio_params.number_of_channels),
        float(radio_params.pos_ra),
        float(radio_params.pos_dec),
        bool(radio_params.random_position),
        bool(radio_params.add_noise),
        str(radio_params.pol_mode),
        bool(radio_params.use_gpus),
    )
    return RadioSimulationCache(
        simulation_hash=ms_path.stem,
        ms_path=ms_path,
        metadata_path=ms_path.with_suffix(".meta.json"),
        resized_fits_path=resized_fits_path,
        source_fits_path=source_fits_path,
    )


def run_simulation(
    data_path: str | Path,
    params: Mapping[str, Any] | _RadioInterferometryParams | None = None,
    **kwargs: Any,
) -> RadioSimulationCache:
    """Ensure a Karabo MeerKAT simulation exists and return its cache paths."""
    radio_params = _coerce_params(params, kwargs)
    cache = get_radio_simulation_cache(data_path, radio_params)
    if cache.is_complete:
        print(
            f"Radio simulation cache hit: simulator_hash={cache.simulation_hash}",
            flush=True,
        )
        return cache

    config = _simulation_config(data_path, radio_params, cache.source_fits_path)
    if radio_params.run_on_slurm:
        _submit_slurm_job(config, radio_params, data_path)
    else:
        _run_container(config, data_path)

    cache = get_radio_simulation_cache(data_path, radio_params)
    if not cache.is_complete:
        raise RuntimeError(
            "Radio simulation finished but the expected cache entry is incomplete: "
            f"ms_path={cache.ms_path}, metadata_path={cache.metadata_path}."
        )
    return cache


class RadioInterferometryInvProb(BaseInvProb):
    """DeepInv inverse problem backed by cached Karabo MeerKAT simulations."""

    def get_invprob(self, invprob_config: InvProbConfig) -> InvProb:
        params = build_problem_params(
            _RadioInterferometryParams,
            invprob_config.params,
        )
        if params.batch_size != 1 or invprob_config.batch_size != 1:
            raise ValueError("RadioInterferometryInvProb currently supports batch_size=1.")

        device = torch.device(invprob_config.device)
        cache = get_radio_simulation_cache(invprob_config.data_path, params)
        if not cache.is_complete:
            raise FileNotFoundError(
                "Radio simulation cache entry is missing. Run dataset.prepare() "
                "or call toolsbench.invprob.radio_interferometry.run_simulation "
                "with the same parameters first. Expected "
                f"{cache.ms_path} and {cache.metadata_path}."
            )

        metadata = _load_metadata(cache.metadata_path)
        ground_truth = _load_ground_truth(cache.resized_fits_path, device)

        from deepinv.physics import GaussianNoise
        from toolsbench.utils.radio_interferometry.deepinv_imager import (
            DeepinvDirtyImager,
            DirtyImagerConfig,
        )

        imager_config = DirtyImagerConfig(
            imaging_npixel=int(params.image_size),
            imaging_cellsize=float(metadata["imaging_cellsize"]),
            combine_across_frequencies=False,
        )
        imager = DeepinvDirtyImager(imager_config, device=device)
        physics, measurements, weights = imager.create_deepinv_physics(
            visibility_path=str(cache.ms_path),
            visibility_format="MS",
            visibility_column="DATA",
        )

        if float(params.noise_level) > 0:
            rng = torch.Generator(device=device).manual_seed(int(params.seed))
            physics.noise_model = GaussianNoise(sigma=float(params.noise_level), rng=rng)
            measurements = physics.noise_model(measurements)

        return InvProb(
            ground_truth=ground_truth,
            measurements=measurements,
            physics=physics,
            ground_truth_shape=ground_truth.shape,
            num_operators=1,
            min_pixel=float(ground_truth.min()),
            max_pixel=float(ground_truth.max()),
            invprob_kwargs={"weights": weights},
        )


def _coerce_params(
    params: Mapping[str, Any] | _RadioInterferometryParams | None,
    kwargs: Mapping[str, Any],
) -> _RadioInterferometryParams:
    if isinstance(params, _RadioInterferometryParams):
        if kwargs:
            merged = {**params.__dict__, **kwargs}
            return build_problem_params(_RadioInterferometryParams, merged)
        return params
    params_dict: dict[str, Any] = {}
    if params is not None:
        params_dict.update(dict(params))
    params_dict.update(kwargs)
    return build_problem_params(_RadioInterferometryParams, params_dict)


def _ensure_resized_fits(
    source_fits_path: Path,
    data_path: Path,
    image_size: int,
) -> tuple[Path, np.ndarray]:
    from astropy.io import fits
    from toolsbench.utils.radio_interferometry.radio_utils import (
        load_and_resize_image,
        load_new_header,
    )

    cache_dir = get_meerkat_cache_dir(data_path)
    resized_fits_path = cache_dir / f"{source_fits_path.stem}_{image_size}.fits"
    if resized_fits_path.exists():
        with fits.open(resized_fits_path, memmap=False) as hdul:
            resized_img = np.array(hdul[0].data, dtype=np.float32, copy=True)
    else:
        resized_img = load_and_resize_image(source_fits_path, image_size, normalize=False)
        header = load_new_header(source_fits_path, image_size)
        fits.PrimaryHDU(resized_img, header=header).writeto(
            resized_fits_path,
            overwrite=True,
        )

    if not resized_img.dtype.isnative:
        resized_img = resized_img.byteswap().view(resized_img.dtype.newbyteorder("="))
    return resized_fits_path, np.ascontiguousarray(resized_img, dtype=np.float32)


def _simulation_config(
    data_path: str | Path,
    params: _RadioInterferometryParams,
    source_fits_path: Path,
) -> dict[str, Any]:
    data_path = Path(data_path)
    fits_name = str(source_fits_path.relative_to(data_path))
    return {
        "job": {
            "fits_name": fits_name,
            "pos_ra": float(params.pos_ra),
            "pos_dec": float(params.pos_dec),
            "random_position": bool(params.random_position),
            "image_size": [int(params.image_size)],
            "data_path": None,
            "use_gpus": bool(params.use_gpus),
            "number_of_time_steps": int(params.number_of_time_steps),
            "start_frequency_hz": float(params.start_frequency_hz),
            "end_frequency_hz": float(params.end_frequency_hz),
            "number_of_channels": int(params.number_of_channels),
            "add_noise": bool(params.add_noise),
            "pol_mode": str(params.pol_mode),
        }
    }


def _container_paths(
    host_data_path: str | Path,
    config: dict[str, Any],
) -> tuple[list[str], str, str]:
    repo_root = get_repo_root()
    mount_point = "/workspace"
    binds = ["-B", f"{repo_root}:{mount_point}"]

    host_data_path = Path(host_data_path).resolve()
    host_data_path.mkdir(parents=True, exist_ok=True)
    try:
        rel = host_data_path.relative_to(repo_root)
        container_data_path = f"{mount_point}/{rel}"
    except ValueError:
        container_data_path = "/benchmark_data"
        binds.extend(["-B", f"{host_data_path}:{container_data_path}"])

    config["job"]["data_path"] = container_data_path
    return binds, mount_point, "/workspace/benchmark_inference"


def _run_container(config: dict[str, Any], host_data_path: str | Path) -> None:
    repo_root = get_repo_root()
    image_path = get_karabo_image_path(repo_root)
    if not image_path.exists():
        raise FileNotFoundError(
            f"Karabo container image not found at {image_path}. "
            "Run `benchopt install -d radio_interferometry` first."
        )

    runtime = shutil.which("apptainer") or shutil.which("singularity")
    if runtime is None:
        raise RuntimeError("Neither apptainer nor singularity is available.")

    config = {"job": dict(config["job"])}
    binds, mount_point, working_dir = _container_paths(host_data_path, config)

    cache_dir = repo_root / "debug_output" / "cache"
    mpl_dir = repo_root / "debug_output" / "mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = repo_root / "debug_output"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_config_host = tmp_dir / "_run_radio_simulation_config.yaml"
    with tmp_config_host.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)

    container_config_path = f"{mount_point}/debug_output/{tmp_config_host.name}"
    cmd = [
        runtime,
        "exec",
        "--nv",
        *binds,
        "--env",
        f"XDG_CACHE_HOME={mount_point}/debug_output/cache,"
        f"MPLCONFIGDIR={mount_point}/debug_output/mpl_cache",
        "--pwd",
        working_dir,
        str(image_path),
        "python",
        f"{mount_point}/src/toolsbench/utils/radio_interferometry/generate_radio_data.py",
        "--config",
        container_config_path,
    ]

    print(f"Running radio simulation: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
    finally:
        tmp_config_host.unlink(missing_ok=True)


def _submit_slurm_job(
    config: dict[str, Any],
    params: _RadioInterferometryParams,
    host_data_path: str | Path,
) -> None:
    try:
        import submitit
    except ImportError as exc:
        raise RuntimeError("Slurm radio simulation requires submitit.") from exc

    folder = Path(params.slurm_folder)
    if not folder.is_absolute():
        folder = get_repo_root() / folder
    folder.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=str(folder))
    additional = {}
    if params.slurm_hint:
        additional["hint"] = params.slurm_hint
    if params.slurm_constraint:
        additional["constraint"] = params.slurm_constraint

    kwargs: dict[str, Any] = {
        "slurm_job_name": params.slurm_job_name,
        "slurm_time": params.slurm_time,
        "slurm_nodes": params.slurm_nodes,
        "slurm_ntasks_per_node": params.slurm_ntasks_per_node,
        "slurm_cpus_per_task": params.slurm_cpus_per_task,
        "slurm_gres": params.slurm_gres,
    }
    if params.slurm_account:
        kwargs["slurm_account"] = params.slurm_account
    if additional:
        kwargs["slurm_additional_parameters"] = additional
    setup = params.slurm_setup
    if isinstance(setup, str):
        setup = [line for line in setup.splitlines() if line.strip()]
    if setup:
        kwargs["slurm_setup"] = setup

    executor.update_parameters(**kwargs)
    print(f"Submitting radio simulation Slurm job with parameters: {kwargs}", flush=True)
    job = executor.submit(_run_container, config, host_data_path)
    print(f"Submitted radio simulation job {job.job_id}.", flush=True)

    deadline = time.time() + int(params.slurm_wait_timeout_seconds)
    while not _job_done(job):
        if time.time() >= deadline:
            try:
                job.cancel()
            except Exception:
                pass
            raise TimeoutError(
                f"Radio simulation job {job.job_id} exceeded "
                f"{params.slurm_wait_timeout_seconds} seconds."
            )
        print(
            f"Radio simulation job {job.job_id} still running "
            f"(state={getattr(job, 'state', 'unknown')}).",
            flush=True,
        )
        time.sleep(int(params.slurm_poll_interval_seconds))

    job.result()


def _job_done(job: Any) -> bool:
    done = getattr(job, "done", None)
    if callable(done):
        return bool(done())
    state = str(getattr(job, "state", "")).upper()
    return state in {"DONE", "FAILED", "CANCELLED", "TIMEOUT"}


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_ground_truth(fits_path: Path, device: torch.device) -> torch.Tensor:
    from astropy.io import fits

    with fits.open(fits_path, memmap=False) as hdul:
        img_np = np.array(hdul[0].data, dtype=np.float32, copy=True)
    if not img_np.dtype.isnative:
        img_np = img_np.byteswap().view(img_np.dtype.newbyteorder("="))
    img = torch.from_numpy(np.ascontiguousarray(img_np, dtype=np.float32))
    if img.ndim == 2:
        img = img.unsqueeze(0)
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.ndim != 4:
        raise ValueError(f"Unexpected radio ground truth shape {tuple(img.shape)}.")
    return img.to(device)
