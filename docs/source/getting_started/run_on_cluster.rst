Run on a Cluster
================

The benchmark can submit inference and training experiments to a SLURM cluster
through BenchOpt's Submitit backend. This page uses
`Jean Zay <http://www.idris.fr/jean-zay/>`_ as a concrete example. Adapt module,
account, QoS, and GPU names to your own cluster.

The login and compute nodes must be able to access the cloned repository and the
Python environment used to install it.

Load the Environment
--------------------

Start by loading the cluster-provided Python and GPU software stack. The
checked-in parallel configuration currently uses this Jean Zay module:

.. code-block:: bash

   module purge
   module load pytorch-gpu/py3/2.7.0

This module provides Python and a GPU-enabled PyTorch installation. Module names
change over time, so use the version recommended by the cluster when the example
above is no longer available.

Create a Virtual Environment
----------------------------

A virtual environment is optional, but it keeps benchmark dependencies isolated
from other projects. Using ``--system-site-packages`` makes the PyTorch stack
provided by the loaded module visible inside the environment:

.. code-block:: bash

   python -m venv --system-site-packages benchmark_env
   source benchmark_env/bin/activate

Create the environment on a shared filesystem. On later sessions, load the same
module before activating it:

.. code-block:: bash

   module purge
   module load pytorch-gpu/py3/2.7.0
   source /path/to/benchmark_env/bin/activate

If you use the module environment directly, omit the virtual-environment commands
and install into an appropriate writable Python environment according to your
cluster policy.

Clone and Install the Benchmark
-------------------------------

Clone the repository, enter it, and install the project with ``pip``:

.. code-block:: bash

   git clone https://github.com/numpex/benchmark_invprob_largescale.git
   cd benchmark_invprob_largescale
   python -m pip install .

For the optional radio interferometry dependencies, install the corresponding
extra instead:

.. code-block:: bash

   python -m pip install '.[radio]'

Check that the active environment can discover both benchmarks:

.. code-block:: bash

   benchopt info benchmark_inference/.
   benchopt info benchmark_training/.

Configure SLURM
---------------

Edit the parallel configuration for the benchmark you want to run:

- ``benchmark_inference/configs/config_parallel.yml``;
- ``benchmark_training/configs/config_parallel.yml``; or
- ``benchmark_training/configs/config_parallel_nsys.yml`` for an Nsight Systems
  profiling run.

At minimum, review ``account``, ``qos``, ``constraint``, ``slurm_time``, and
``slurm_setup``. The setup commands execute inside every submitted job. They must
load the same software stack used during installation and, when applicable,
activate the shared virtual environment:

.. code-block:: yaml

   backend: submitit
   slurm_time: 1800
   slurm_python: python
   slurm_additional_parameters:
     cpus-per-task: 10
     qos: qos_gpu-dev
     account: your_account
     constraint: v100-32g
   slurm_setup:
     - module purge
     - module load pytorch-gpu/py3/2.7.0
     - source /path/to/benchmark_env/bin/activate
     - export NCCL_DEBUG=INFO

If you did not create a virtual environment, remove its ``source`` line.
``slurm_python`` must resolve to an interpreter that can import ``benchopt``,
``toolsbench``, DeepInv, and the remaining project dependencies on the compute
nodes.

The experiment YAML separately specifies resources for each solver run through
``slurm_gres``, ``slurm_ntasks_per_node``, and ``slurm_nodes``. See
:doc:`config_guide` for the full configuration model.

Submit an Experiment
--------------------

From the repository root, submit the multi-frame super-resolution example with:

.. code-block:: bash

   benchopt run benchmark_inference/. \
       --parallel-config benchmark_inference/configs/config_parallel.yml \
       --config benchmark_inference/configs/examples/multiframe_superres.yml \

.. code-block:: bash

   benchopt run benchmark_training/. \
       --parallel-config benchmark_training/configs/config_parallel.yml \
       --config benchmark_training/configs/synthetic_unrolled.yml \

BenchOpt expands the configured dataset and solver grids, Submitit creates the
corresponding SLURM jobs, and each distributed solver starts the requested
processes across its assigned GPUs and nodes. Submission files and logs are
written below the benchmark's ``benchopt_run/`` directory.

Collect Results
---------------------------

Completed benchmark tables and reports are stored in the selected benchmark's
``outputs/`` directory. Filenames are timestamped by default, for example:

.. code-block:: text

   benchmark_inference/outputs/benchopt_run_<timestamp>.parquet
   benchmark_inference/outputs/benchmark_inference_benchopt_run_<timestamp>.html

The HTML report contains convergence and quality curves together with the timing
and resource metrics returned by the benchmark. Solver profiling may also create
per-rank metric files in ``outputs/``.
