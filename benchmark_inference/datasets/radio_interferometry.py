"""Benchopt dataset for the radio interferometry inverse problem."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from benchopt import BaseDataset, config
from deepinv.distributed import DistributedContext

from toolsbench.data import check_installed
from toolsbench.invprob import InvProbConfig, RadioInterferometryInvProb
from toolsbench.invprob.radio_interferometry import run_simulation
from toolsbench.utils import setup_distributed_env


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
HOST_WORKSPACE_PATH = BENCHMARK_DIR.parent
KARABO_IMAGE_PATH = BENCHMARK_DIR / "tools" / "karabo.sif"


def _singularity_allowed_dir() -> Path | None:
    allowed_dir = os.environ.get("SINGULARITY_ALLOWED_DIR")
    if allowed_dir:
        return Path(allowed_dir).expanduser()

    try:
        result = subprocess.run(
            [
                "bash",
                "-lc",
                "module load singularity >/dev/null 2>&1 && "
                'printf "%s" "${SINGULARITY_ALLOWED_DIR:-}"',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    allowed_dir = result.stdout.strip()
    if result.returncode == 0 and allowed_dir:
        return Path(allowed_dir).expanduser()
    return None


def _karabo_image_path() -> Path:
    allowed_dir = _singularity_allowed_dir()
    if allowed_dir:
        return allowed_dir / "karabo.sif"
    return KARABO_IMAGE_PATH


class Dataset(BaseDataset):
    name = "radio_interferometry"

    install_cmd = "shell"
    install_script = "install_radio.sh"

    parameters = {
        "image_size": [256],
        "fits_size": ["1024"],
        "noise_level": [0.1],
        "seed": [42],
        "pos_ra": [0.0],
        "pos_dec": [0.0],
        "random_position": [True],
        "use_gpus": [True],
        "number_of_time_steps": [64],
        "start_frequency_hz": [1.300e9],
        "end_frequency_hz": [1.340e9],
        "number_of_channels": [8],
        "add_noise": [True],
        "pol_mode": ["Full"],
        "run_on_slurm": [False],
        "slurm_folder": ["debug_output/slurm_logs"],
        "slurm_job_name": ["karabo_simulator"],
        "slurm_nodes": [1],
        "slurm_ntasks_per_node": [1],
        "slurm_cpus_per_task": [40],
        "slurm_gres": ["gpu:4"],
        "slurm_time": [60],
        "slurm_hint": ["nomultithread"],
        "slurm_account": [None],
        "slurm_constraint": ["v100-32g"],
        "slurm_poll_interval_seconds": [30],
        "slurm_wait_timeout_seconds": [1200],
        "slurm_setup": ["module purge\nmodule load singularity\nset -x"],
    }

    prepare_cache_ignore = (
        "noise_level",
        "seed",
        "slurm_folder",
        "slurm_job_name",
        "slurm_nodes",
        "slurm_ntasks_per_node",
        "slurm_cpus_per_task",
        "slurm_gres",
        "slurm_time",
        "slurm_hint",
        "slurm_account",
        "slurm_constraint",
        "slurm_poll_interval_seconds",
        "slurm_wait_timeout_seconds",
        "slurm_setup",
    )

    @classmethod
    def is_installed(cls, env_name=None, quiet=True, **kwargs):
        if _singularity_allowed_dir():
            return _karabo_image_path().exists()
        runtime_available = bool(shutil.which("apptainer") or shutil.which("singularity"))
        return runtime_available and _karabo_image_path().exists()

    def prepare(self, env_name=None, **kwargs):
        data_path = Path(config.get_data_path(key="radio_interferometry"))
        check_installed("radio_interferometry", data_path)
        cache = run_simulation(
            data_path,
            params=self._invprob_params(),
            karabo_image_path=_karabo_image_path(),
            host_workspace_path=HOST_WORKSPACE_PATH,
        )
        print(
            f"Radio simulation ready: simulator_hash={cache.simulation_hash}",
            flush=True,
        )

    def get_data(self):
        setup_distributed_env()
        with DistributedContext(seed=self.seed, cleanup=False) as ctx:
            invprob_conf = InvProbConfig(
                size=(int(self.image_size), int(self.image_size)),
                batch_size=1,
                channels=1,
                device=ctx.device,
                data_path=Path(config.get_data_path(key="radio_interferometry")),
                params=self._invprob_params(),
            )
            return RadioInterferometryInvProb().get_invprob(invprob_conf).asdict()

    def _invprob_params(self) -> dict:
        return {
            "image_size": int(self.image_size),
            "fits_size": str(self.fits_size),
            "noise_level": float(self.noise_level),
            "seed": int(self.seed),
            "pos_ra": float(self.pos_ra),
            "pos_dec": float(self.pos_dec),
            "random_position": bool(self.random_position),
            "use_gpus": bool(self.use_gpus),
            "number_of_time_steps": int(self.number_of_time_steps),
            "start_frequency_hz": float(self.start_frequency_hz),
            "end_frequency_hz": float(self.end_frequency_hz),
            "number_of_channels": int(self.number_of_channels),
            "add_noise": bool(self.add_noise),
            "pol_mode": str(self.pol_mode),
            "run_on_slurm": bool(self.run_on_slurm),
            "slurm_folder": str(self.slurm_folder),
            "slurm_job_name": str(self.slurm_job_name),
            "slurm_nodes": int(self.slurm_nodes),
            "slurm_ntasks_per_node": int(self.slurm_ntasks_per_node),
            "slurm_cpus_per_task": int(self.slurm_cpus_per_task),
            "slurm_gres": str(self.slurm_gres),
            "slurm_time": int(self.slurm_time),
            "slurm_hint": str(self.slurm_hint),
            "slurm_account": self.slurm_account,
            "slurm_constraint": str(self.slurm_constraint),
            "slurm_poll_interval_seconds": int(self.slurm_poll_interval_seconds),
            "slurm_wait_timeout_seconds": int(self.slurm_wait_timeout_seconds),
            "slurm_setup": self.slurm_setup,
        }
