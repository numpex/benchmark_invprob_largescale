Configuration Guide
===================

Each benchmark keeps its configuration next to its datasets, objective, and
solvers. A cluster run combines two YAML files with different responsibilities:

- ``--parallel-config`` configures the execution backend and the SLURM
  environment shared by all jobs.
- ``--config`` selects the objective, dataset, solver, parameter grid, and
  BenchOpt run options for one experiment.

This page uses ``benchmark_inference`` throughout. The same organization and
commands apply to ``benchmark_training`` by replacing the benchmark and
configuration paths.

Configuration Organization
--------------------------

The current layout is:

.. code-block:: text

   benchmark_inference/configs/
   ├── config_parallel.yml
   ├── examples/
   │   ├── multiframe_superres.yml
   │   ├── radio_interferometry.yml
   │   ├── tomography_2d.yml
   │   └── tomography_3d.yml
   └── experiments/
       ├── denoiser_compile.yml
       ├── reconstruction_quality.yml
       ├── strong_scaling_inference.yml
       └── tomography_2d_compile.yml

   benchmark_training/configs/
   ├── config_parallel.yml
   ├── config_parallel_nsys.yml
   ├── synthetic_unrolled.yml
   └── experiments/
       ├── batch_size.yml
       ├── checkpointing.yml
       ├── comm_time.yml
       ├── strong_scaling.yml
       └── weak_scaling.yml

Use ``examples/`` for representative imaging pipelines and ``experiments/``
for focused performance studies.

Parallel Configuration
----------------------

``benchmark_inference/configs/config_parallel.yml`` contains settings that are
common to every submitted job. The checked-in file has the following structure:

.. code-block:: yaml

   backend: submitit
   slurm_time: 1800
   slurm_stderr_to_stdout: true
   slurm_python: python
   slurm_additional_parameters:
     cpus-per-task: 10
     qos: qos_gpu-dev
     account: null
     constraint: v100-32g
   slurm_setup:
     - module purge
     - module load pytorch-gpu/py3/2.7.0
     - export NCCL_DEBUG=INFO

Before submitting, adapt at least ``account``, ``qos``, ``constraint``, and
``slurm_setup`` to your cluster. ``slurm_time`` is expressed in seconds in the
current configuration. ``slurm_python`` must resolve to a Python interpreter
that can import the benchmark dependencies on the compute nodes.

This file defines scheduler-wide defaults, but it does **not** define the number
of GPUs or nodes for an individual experiment. Those resources belong to the
solver grid in the experiment configuration.

Experiment Configuration
------------------------

An experiment YAML follows the same top-level structure for inference and
training:

.. code-block:: yaml

   objective:
     - reconstruction_objective

   dataset:
     - multiframe_superres:
         image_size: 256
         num_operators: 8
         noise_level: 0.1

   solver:
     - PnP:
         denoiser: drunet
         denoiser_sigma: 0.005
         step_size: 0.1
         init_method: [zeros]
         profiler_mode: custom
         profiler_warmup: 0
         profiler_active: 0

   max-runs: 2
   n-repetitions: 1
   plot: true
   html: true

The sections have distinct roles:

``objective``
   Selects how solver results are scored. The reconstruction objective reports
   quality metrics and forwards profiling measurements returned by the solver.

``dataset``
   Selects the inverse problem and its acquisition parameters. Dataset names
   correspond to implementations in ``benchmark_inference/datasets/``.

``solver``
   Selects the algorithm, its numerical parameters, profiling options, and its
   execution-resource grid. Solver names correspond to implementations in
   ``benchmark_inference/solvers/``.

Run options
   ``max-runs`` controls the number of solver callback steps,
   ``n-repetitions`` controls timing repetitions, and ``plot`` and ``html``
   control report generation.

You can inspect the names discovered by BenchOpt before creating a configuration:

.. code-block:: bash

   benchopt info benchmark_inference/.
   benchopt info benchmark_training/.

Execution Grids
---------------

Comma-separated parameter names define coupled values. Each inner list is one
configuration rather than an independent Cartesian product:

.. code-block:: yaml

   solver:
     - PnP:
         slurm_gres, slurm_ntasks_per_node, slurm_nodes, distribute_physics, distribute_denoiser, patch_size, overlap, max_batch_size:
           - ["gpu:1", 1, 1, false, false,   0,  0, 0]
           - ["gpu:2", 2, 1, true,  true,  448, 32, 0]

The first row requests one GPU and runs without distribution. The second requests
two GPUs on one node, starts two tasks on that node, and distributes both the
physics and denoising work.

The main resource and distribution fields are:

- ``slurm_gres``: SLURM generic-resource request per node, such as ``gpu:2``.
- ``slurm_ntasks_per_node``: number of distributed processes started per node.
- ``slurm_nodes``: number of nodes requested for the job.
- ``distribute_physics``: distribute the forward/adjoint physics operations.
- ``distribute_denoiser``: distribute patch-based denoiser evaluation.
- ``patch_size`` and ``overlap``: spatial tiling parameters for the denoiser.
- ``max_batch_size``: maximum number of patches evaluated together on a device.

Keep resource fields coupled when a particular algorithm setting only makes
sense for a matching GPU topology. Ordinary list-valued parameters that are not
coupled are expanded by BenchOpt as a parameter grid.

The pattern generalizes directly to training. Training uses ``UnrolledPnP`` and
fields such as ``distribute_model`` and ``checkpoint_batches`` instead of the
inference-specific ``distribute_physics`` and ``distribute_denoiser`` fields:

.. code-block:: yaml

   solver:
     - UnrolledPnP:
         slurm_gres, slurm_ntasks_per_node, slurm_nodes, distribute_model, patch_size, overlap, max_batch_size, checkpoint_batches:
           - ["gpu:1", 1, 1, true, 512, 32, 4, always]
           - ["gpu:2", 2, 1, true, 512, 32, 4, always]

Running a Configuration
-----------------------

Run commands from the repository root. For the inference example:

.. code-block:: bash

   benchopt run benchmark_inference/. \
       --parallel-config benchmark_inference/configs/config_parallel.yml \
       --config benchmark_inference/configs/examples/multiframe_superres.yml \

To create a new study, copy the closest example or experiment, give the copy a
descriptive name, adjust its dataset and solver grids, and keep it under the
corresponding benchmark's ``configs/`` directory.
