Radio Interferometry
====================

Dataset name: ``radio_interferometry``

Data and Inverse Problem
------------------------

Radio interferometers do not measure a sky image directly. Pairs of antennas
sample complex spatial frequencies called *visibilities*. Reconstructing the sky
therefore amounts to recovering an image from incomplete, non-Cartesian Fourier
measurements, with sampling weights determined by the telescope observation.

This use case downloads a FITS sky image and uses Karabo to simulate a MeerKAT
observation. The simulation produces a Measurement Set and a metadata file. At
dataset load time, the cached visibilities are converted into a DeepInv physics
operator and measurement tensor by the benchmark's dirty-imaging interface. The
FITS image is retained as ground truth, and visibility weights are passed to the
solver.

There are two distinct noise controls. ``add_noise`` belongs to the Karabo
observation simulation and therefore changes the prepared cache. ``noise_level``
adds reproducible Gaussian noise through the DeepInv physics when the benchmark
dataset is loaded.

What ``benchopt install`` Does
------------------------------

Radio interferometry provides a custom shell installer. It checks for Apptainer
or Singularity, loads the ``singularity`` module when the environment supports
modules, and pulls the Karabo image from an OCI registry if it is absent. The
image is stored as ``benchmark_inference/tools/karabo.sif``.

On Jean Zay, the installer also detects ``SINGULARITY_ALLOWED_DIR`` and ensures
that ``karabo.sif`` is present there, using the site container-copy command when
available. An existing image is reused. The registry can be overridden with the
``KARABO_IMAGE_URI`` environment variable.

What ``benchopt prepare`` Does
------------------------------

Preparation first downloads the FITS file selected by ``fits_size``, either "1024" or "10k". It then
builds a deterministic cache key from the observation parameters and checks for a
matching Measurement Set and metadata file below the dataset's
``meerkat_cache/`` directory.

If the cache is missing, preparation runs the Karabo container either directly
or in a separate SLURM job according to ``run_on_slurm``. Preparation waits for a
submitted simulation to finish and verifies that both cache files were created.
Matching completed simulations are reused.

Run preparation for every observation configuration before launching the main
benchmark, especially on clusters where compute nodes have restricted network or
container-registry access. The simulation SLURM settings below control this
preparation job; they are separate from the parallel configuration used by
``benchopt run``.

Image and Observation Parameters
--------------------------------

``fits_size`` (default ``"1024"``)
   Selects the source-image collection. Supported values are ``"1024"`` and
   ``"10k"``.

``pos_ra`` and ``pos_dec`` (defaults ``0.0``)
   Right ascension and declination used for a fixed observation position.

``random_position`` (default ``true``)
   Ask the simulator to select the sky position randomly rather than relying
   only on the fixed coordinates.

``number_of_time_steps`` (default ``64``)
   Number of temporal samples in the simulated observation.

``start_frequency_hz`` and ``end_frequency_hz``
   Frequency interval of the observation; defaults are ``1.300e9`` and
   ``1.340e9`` Hz.

``number_of_channels`` (default ``8``)
   Number of frequency channels across the interval.

``pol_mode`` (default ``"Full"``)
   Polarization mode passed to the simulator.

``use_gpus`` (default ``true``)
   Enable GPU use inside the Karabo simulation.

``add_noise`` (default ``true``)
   Enable the simulator's observation noise.

``noise_level`` (default ``0.1``)
   Standard deviation of the additional Gaussian noise applied when creating
   the DeepInv measurements.

``seed`` (default ``42``)
   Seed for the additional DeepInv noise.

Preparation-job Parameters
--------------------------

``run_on_slurm`` (default ``false``)
   Run Karabo immediately in the current allocation, or submit a dedicated
   simulation job when set to ``true``.

``slurm_folder`` (default ``debug_output/slurm_logs``)
   Submitit log directory for the simulation job.

``slurm_job_name`` (default ``karabo_simulator``)
   Name of the simulation job.

``slurm_nodes`` (default ``1``)
   Number of nodes requested for the simulation.

``slurm_ntasks_per_node`` (default ``1``)
   Number of simulation tasks started on each node.

``slurm_cpus_per_task`` (default ``40``)
   CPU cores requested for each simulation task.

``slurm_gres`` (default ``gpu:4``)
   Generic GPU resource request for the simulation.

``slurm_time`` (default ``60``)
   Wall-time value passed to Submitit for the simulation job.

``slurm_hint`` (default ``nomultithread``)
   SLURM hint applied to the simulation job.

``slurm_account`` (default ``null``)
   Optional SLURM account used for the simulation.

``slurm_constraint`` (default ``v100-32g``)
   GPU constraint requested for the simulation.

``slurm_setup`` (default loads the Singularity module)
   Commands executed before the simulation job starts.

``slurm_poll_interval_seconds`` (default ``30``)
   Interval at which preparation checks the submitted job.

``slurm_wait_timeout_seconds`` (default ``1200``)
   Maximum time preparation waits before cancelling the simulation job.

See ``benchmark_inference/configs/examples/radio_interferometry.yml`` for a
complete configuration.
