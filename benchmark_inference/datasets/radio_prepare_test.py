"""Radio interferometry test dataset using Dataset.prepare() for simulation.

How prepare() and get_data() communicate
-----------------------------------------
They communicate through the filesystem — there is no direct data passing via
benchopt.

  prepare()  runs the Karabo MeerKAT simulator inside the Singularity container
             and writes {hash}.ms + {hash}.meta.json to
             data/radio_interferometry/meerkat_cache/.  The hash is computed
             deterministically by get_meerkat_visibilities_path() from the
             image data and simulation parameters.

  get_data() recomputes the same hash (same function, same parameters) to find
             the MS file written by prepare(), then builds the DeepInverse
             physics operator from it.

  benchopt's joblib cache around _prepare(dataset) acts only as a guard: it
  skips re-running prepare() when the effective parameters haven't changed.
  It stores a None result — the real handoff is 100% on disk.

Cache key
---------
All simulation observation parameters are in the benchopt parameters dict and
therefore part of the prepare cache key by default.  The exceptions are:

  noise_level  applied post-simulation in get_data() via GaussianNoise
  seed         controls randomness in get_data(), not the simulator
  use_gpus     runtime performance switch — same MS regardless of GPU/CPU
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from benchopt import BaseDataset
from benchopt import config as benchopt_config


class Dataset(BaseDataset):
    name = "radio_prepare_test"

    install_cmd = "shell"
    install_script = "install_radio_test.sh"

    # Simulation parameters are proper benchopt parameters so benchopt manages
    # the prepare cache key automatically.  Parameters that do not affect the
    # MS content are listed in prepare_cache_ignore.
    parameters = {
        # ── image / benchmark ──────────────────────────────────────────
        "fits_name": [None],
        "image_size": [256],
        "noise_level": [0.1],
        "seed": [42],
        # ── observation geometry ───────────────────────────────────────
        "pos_ra": [155.66367],
        "pos_dec": [-30.7130],
        "random_position": [True],
        # ── frequency / time ───────────────────────────────────────────
        "number_of_time_steps": [1024],
        "start_frequency_hz": [1.300e9],
        "end_frequency_hz": [1.340e9],
        "number_of_channels": [32],
        # ── simulator options ──────────────────────────────────────────
        "add_noise": [True],
        "pol_mode": ["Full"],
        "use_gpus": [True],
        # ── singularity container ──────────────────────────────────────────────────
        # null → resolved at runtime as get_data_path("containers")/karabo.sif
        # Set to a repo-relative or absolute path to override.
        "singularity_image_path": [None],
        "singularity_mount_point": ["/workspace"],
        "singularity_working_dir": ["/workspace/benchmark_inference"],
    }

    # noise_level / seed: applied post-simulation in get_data(), not by the simulator.
    # use_gpus: runtime performance switch only — same MS regardless of GPU/CPU.
    # singularity_*: container infrastructure, does not affect the MS content.
    prepare_cache_ignore = (
        "noise_level",
        "seed",
        "use_gpus",
        "singularity_image_path",
        "singularity_mount_point",
        "singularity_working_dir",
    )

    # ------------------------------------------------------------------
    # Image path resolution
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_image_path(cls, image_path_param):
        """Resolve the Singularity image path from a parameter value.

        None           → get_data_path("containers") / "karabo.sif"
        relative str   → repo_root / value
        absolute str   → used as-is
        """
        if image_path_param is None:
            try:
                containers_dir = Path(
                    benchopt_config.get_data_path("containers")
                )
            except Exception:
                # Fallback: benchmark_inference/data/containers (relative to this file)
                containers_dir = Path(__file__).parent.parent / "data" / "containers"
            return containers_dir / "karabo.sif"

        image_path = Path(image_path_param)
        if not image_path.is_absolute():
            from toolsbench.utils.submit_job import get_repo_root
            image_path = get_repo_root() / image_path
        return image_path

    # ------------------------------------------------------------------
    # is_installed
    # ------------------------------------------------------------------

    @classmethod
    def is_installed(cls, env_name=None, quiet=True, **kwargs):
        """Return True when all runtime dependencies are available.

        Checks that required Python packages are importable and that the
        Singularity image pulled by install_radio_test.sh is present on disk.
        Data readiness is a separate concern handled by prepare().
        """
        try:
            import astropy
            import deepinv
            import torchkbnufft
            from deepinv.distributed import DistributedContext

            _ = (astropy, deepinv, torchkbnufft, DistributedContext)
        except ImportError:
            return False

        try:
            image_path = cls._resolve_image_path(
                cls.parameters["singularity_image_path"][0]
            )
            if not image_path.exists():
                return False
        except Exception:
            return False

        return True

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        fits_name=None,
        image_size=256,
        noise_level=0.1,
        seed=42,
        pos_ra=155.66367,
        pos_dec=-30.7130,
        random_position=True,
        number_of_time_steps=1024,
        start_frequency_hz=1.300e9,
        end_frequency_hz=1.340e9,
        number_of_channels=32,
        add_noise=True,
        pol_mode="Full",
        use_gpus=True,
        singularity_image_path=None,
        singularity_mount_point="/workspace",
        singularity_working_dir="/workspace/benchmark_inference",
    ):
        super().__init__()
        self.fits_name = fits_name
        self.image_size = image_size
        self.noise_level = noise_level
        self.seed = seed
        self.pos_ra = pos_ra
        self.pos_dec = pos_dec
        self.random_position = random_position
        self.number_of_time_steps = number_of_time_steps
        self.start_frequency_hz = start_frequency_hz
        self.end_frequency_hz = end_frequency_hz
        self.number_of_channels = number_of_channels
        self.add_noise = add_noise
        self.pol_mode = pol_mode
        self.use_gpus = use_gpus
        self.singularity_image_path = singularity_image_path
        self.singularity_mount_point = singularity_mount_point
        self.singularity_working_dir = singularity_working_dir

    # ------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------

    def prepare(self):
        """Run the Karabo MeerKAT simulation inside the Singularity container.

        Simulation parameters come from self (benchopt dataset parameters).
        Singularity container settings also come from self — no YAML is read here.
        Writes {hash}.ms + {hash}.meta.json to meerkat_cache/; the hash is
        recomputed identically by get_data() to locate the file.
        """
        from toolsbench.utils import load_cached_example
        from toolsbench.utils.submit_job import get_repo_root, run_simulation_with_params

        print(f"[DEBUG prepare()] fits_name={self.fits_name!r}, image_size={self.image_size}, number_of_channels={self.number_of_channels}", flush=True)
        if self.fits_name is None:
            raise ValueError(
                "fits_name must be set. "
                "Configure it via --config configs/radio_prepare_test_config.yaml."
            )

        image_path = self._resolve_image_path(self.singularity_image_path)

        # Ensure the source FITS file exists on the host before launching the
        # container (the container accesses it through the bind-mount).
        data_path = Path(benchopt_config.get_data_path(key="radio_interferometry"))
        data_path.mkdir(parents=True, exist_ok=True)
        if not (data_path / self.fits_name).exists():
            load_cached_example(
                self.fits_name,
                cache_dir=data_path,
                grayscale=True,
                device="cpu",
            )

        # Compute the container-absolute data path.
        # repo_root (benchmark_invprob_largescale) is mounted at singularity_mount_point,
        # so the relative path from repo_root maps 1-to-1 into the container.
        repo_root = get_repo_root()
        container_data_path = (
            f"{self.singularity_mount_point}/{data_path.relative_to(repo_root)}"
        )

        job_params = {
            "fits_name": self.fits_name,
            "image_size": [self.image_size],  # generate_radio_data iterates a list
            "data_path": container_data_path,
            "pos_ra": self.pos_ra,
            "pos_dec": self.pos_dec,
            "random_position": self.random_position,
            "number_of_time_steps": self.number_of_time_steps,
            "start_frequency_hz": self.start_frequency_hz,
            "end_frequency_hz": self.end_frequency_hz,
            "number_of_channels": self.number_of_channels,
            "add_noise": self.add_noise,
            "pol_mode": self.pol_mode,
            "use_gpus": self.use_gpus,
        }

        singularity_cfg = {
            "image_path": str(image_path),
            "mount_point": self.singularity_mount_point,
            "working_dir": self.singularity_working_dir,
        }
        run_simulation_with_params(
            config={"singularity": singularity_cfg},
            job_params=job_params,
        )

    # ------------------------------------------------------------------
    # get_data
    # ------------------------------------------------------------------

    def get_data(self):
        """Load the data produced by prepare().

        Recomputes the expected MS path from the simulation parameters (same
        hash as prepare()), then builds the DeepInverse physics operator.

        Raises FileNotFoundError if prepare() has not been run yet.
        """
        from deepinv.distributed import DistributedContext
        from deepinv.physics import GaussianNoise
        from toolsbench.utils import load_cached_example
        from toolsbench.utils.deepinv_imager import DeepinvDirtyImager, DirtyImagerConfig
        from toolsbench.utils.radio_utils import (
            get_meerkat_visibilities_path,
            load_and_resize_image,
            load_new_header,
        )

        if self.fits_name is None:
            raise ValueError("fits_name must be set.")

        data_path = Path(benchopt_config.get_data_path(key="radio_interferometry"))
        ms_cache_dir = data_path / "meerkat_cache"

        # ── Load (or create) the resized FITS ──────────────────────────────
        # prepare() writes the resized FITS to ms_cache_dir as a side-effect
        # of generate_radio_data.py.  If it's already there we skip the
        # download; otherwise we fall back so get_data() works standalone.
        fits_stem = Path(self.fits_name).stem
        cached_resized_fits_path = ms_cache_dir / f"{fits_stem}_{self.image_size}.fits"

        if cached_resized_fits_path.exists():
            with fits.open(cached_resized_fits_path, memmap=False) as hdul:
                img_np = np.array(hdul[0].data, dtype=np.float32, copy=True)
        else:
            load_cached_example(
                self.fits_name,
                cache_dir=data_path,
                grayscale=True,
                device="cpu",
            )
            source_fits_path = data_path / self.fits_name
            img_np = load_and_resize_image(source_fits_path, self.image_size)
            new_header = load_new_header(source_fits_path, self.image_size)
            ms_cache_dir.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(img_np, header=new_header).writeto(
                cached_resized_fits_path, overwrite=True
            )

        if not img_np.dtype.isnative:
            img_np = img_np.byteswap().view(img_np.dtype.newbyteorder("="))

        # ── Recompute the MS path (same hash as prepare()) ─────────────────
        # Both sides call the same deterministic function with self.* params.
        ms_path = get_meerkat_visibilities_path(
            img_np,
            ms_cache_dir,
            self.fits_name,
            self.image_size,
            number_of_time_steps=self.number_of_time_steps,
            start_frequency_hz=self.start_frequency_hz,
            end_frequency_hz=self.end_frequency_hz,
            number_of_channels=self.number_of_channels,
            random_position=self.random_position,
        )

        if not ms_path.exists():
            raise FileNotFoundError(
                f"Measurement Set not found at {ms_path}.\n"
                "Run  benchopt prepare . -d radio_prepare_test  first."
            )

        metadata_path = ms_path.with_suffix(".meta.json")
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {metadata_path}.\n"
                "Re-run  benchopt prepare . -d radio_prepare_test ."
            )

        with metadata_path.open() as f:
            metadata = json.load(f)
        imaging_cellsize = float(metadata["imaging_cellsize"])

        # ── Set up distributed context ─────────────────────────────────────
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            try:
                import submitit

                submitit.helpers.TorchDistributedEnvironment().export(
                    set_cuda_visible_devices=False
                )
            except (ImportError, RuntimeError):
                pass

        with DistributedContext(seed=self.seed, cleanup=False) as ctx:
            device = ctx.device

            img = torch.from_numpy(img_np)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            ground_truth = img.to(device)

            imager_config = DirtyImagerConfig(
                imaging_npixel=self.image_size,
                imaging_cellsize=imaging_cellsize,
                combine_across_frequencies=False,
            )
            imager = DeepinvDirtyImager(imager_config, device=device)
            physics, measurements = imager.create_deepinv_physics(
                visibility_path=str(ms_path),
                visibility_format="MS",
                visibility_column="DATA",
            )

            # noise_level is applied here, not during simulation, so it does
            # not affect the prepare cache key.
            if self.noise_level > 0:
                physics.noise_model = GaussianNoise(sigma=self.noise_level)
                measurements = physics.noise_model(measurements)

            return dict(
                ground_truth=ground_truth,
                measurement=measurements,
                physics=physics,
                min_pixel=0.0,
                max_pixel=1.0,
                ground_truth_shape=ground_truth.shape,
            )
