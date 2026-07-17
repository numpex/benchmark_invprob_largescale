# Benchmarking Inverse Problems at Scale

How well does an inverse-problem method perform when the unknown is a
multi-megapixel image or a full 3D volume? This repository provides reproducible
[BenchOpt](https://benchopt.github.io) and [DeepInv](https://deepinv.github.io)
benchmarks for reconstruction quality, runtime, GPU memory, communication, and
distributed scaling on one GPU, multiple GPUs, or multi-node SLURM clusters.

For measurements

\[
y = A x + n,
\]

the benchmark studies the complete large-scale pipeline: the acquisition
operator, learned prior, solver parameterization, memory footprint, and
communication between devices.

## Benchmarks

- **Inference** (`benchmark_inference/`) measures the quality and cost of solving
  large inverse problems with plug-and-play reconstruction algorithms and
  standalone denoisers. It records PSNR/SSIM and per-step timing and memory for
  physics-gradient and denoising stages.
- **Training** (`benchmark_training/`) measures supervised optimization of
  unrolled plug-and-play networks. One benchmark iteration is one complete
  training step, enabling strong/weak scaling, communication, patch batching,
  and activation-checkpointing studies.

Both benchmarks use shared solver, profiling, distributed-computing, and
visualization utilities from `src/toolsbench/`.

## Use Cases

- **Multi-frame super-resolution:** reconstruct a high-resolution color image
  from blurred, downsampled, and noisy frames.
- **2D tomography:** reconstruct a slice from noisy ASTRA-backed parallel-beam
  projections.
- **3D tomography:** reconstruct the real Walnut cone-beam CT volume using its
  measured sinogram and scanner geometry.
- **Radio interferometry:** recover a sky image from sparse, non-Cartesian
  Fourier measurements simulated for a configurable MeerKAT observation.
- **Synthetic workloads:** generate scalable 2D super-resolution and 3D
  denoising problems without external data, for controlled scaling studies.

Inference uses an iterative PnP solver whose physics and denoiser can be
distributed independently. Training uses an unrolled PGD/PnP model with
spatially distributed patch processing. The available profilers range from
lightweight wall-clock and peak-memory measurements to PyTorch operator traces
and NVIDIA Nsight Systems timelines.

See the [use-case documentation](docs/source/use_cases/index.rst) and
[solver and profiler documentation](docs/source/solvers/index.rst) for the data,
algorithms, multi-GPU execution, and full parameter lists.

## Installation

The project requires Python 3.12 or later. For a local installation with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Optional dependency groups are available for radio interferometry and the
documentation:

```bash
uv sync --extra radio
uv sync --extra docs
```

Inspect the datasets, solvers, objectives, and parameters discovered by
BenchOpt:

```bash
uv run benchopt info benchmark_inference/.
uv run benchopt info benchmark_training/.
```

## Running an Experiment

Experiment YAML files select the objective, dataset, solver parameter grid,
profiling window, and run options. Prepare reusable input data before launching
an inference experiment, especially when compute nodes do not have internet
access:

```bash
uv run benchopt install benchmark_inference/. \
    --config benchmark_inference/configs/examples/multiframe_superres.yml
uv run benchopt prepare benchmark_inference/. \
    --config benchmark_inference/configs/examples/multiframe_superres.yml
uv run benchopt run benchmark_inference/. \
    --config benchmark_inference/configs/examples/multiframe_superres.yml
```

Run the synthetic unrolled-training benchmark with:

```bash
uv run benchopt run benchmark_training/. \
    --config benchmark_training/configs/synthetic_unrolled.yml
```

Other ready-to-run inference examples cover 2D/3D tomography and radio
interferometry under `benchmark_inference/configs/examples/`. Experiment grids
for scaling, compilation, communication, patch batching, and checkpointing are
under each benchmark's `configs/experiments/` directory.

The [configuration guide](docs/source/getting_started/config_guide.rst) explains
coupled parameter grids and distributed resource settings.

## Running on a SLURM Cluster

The checked-in parallel configurations use BenchOpt's Submitit backend. On a
cluster, install the environment on a shared filesystem accessible from login
and compute nodes. For example, on Jean Zay:

```bash
module purge
module load pytorch-gpu/py3/2.7.0
python -m venv --system-site-packages benchmark_env
source benchmark_env/bin/activate
python -m pip install .
```

Install `.[radio]` instead when radio-interferometry dependencies are required.
On later sessions, load the same module before activating the environment.

Edit the appropriate parallel configuration before submitting:

- `benchmark_inference/configs/config_parallel.yml`
- `benchmark_training/configs/config_parallel.yml`
- `benchmark_training/configs/config_parallel_nsys.yml` for Nsight Systems
  profiling

At minimum, adapt the SLURM account, QoS, GPU constraint, time limit, and
`slurm_setup` commands. `slurm_python` must resolve on compute nodes to an
interpreter that can import BenchOpt, `toolsbench`, DeepInv, and the project
dependencies. GPU counts and process topology are selected separately in the
experiment YAML through `slurm_gres`, `slurm_ntasks_per_node`, and
`slurm_nodes`.

Submit inference and training experiments from the repository root:

```bash
benchopt run benchmark_inference/. \
    --parallel-config benchmark_inference/configs/config_parallel.yml \
    --config benchmark_inference/configs/examples/multiframe_superres.yml

benchopt run benchmark_training/. \
    --parallel-config benchmark_training/configs/config_parallel.yml \
    --config benchmark_training/configs/synthetic_unrolled.yml
```

BenchOpt expands the dataset/solver grids, Submitit creates the SLURM jobs, and
the solvers start the requested ranks across GPUs and nodes. Submission logs are
stored below the selected benchmark's `benchopt_run/` directory.

The complete operational walkthrough is in
[Run on a Cluster](docs/source/getting_started/run_on_cluster.rst).

## Results and Documentation

Completed Parquet results and HTML reports are written below the selected
benchmark's `outputs/` directory. Reports contain quality/convergence curves and
timing/resource metrics; profiling can additionally create per-rank CSV or
trace files.

Build the documentation locally with:

```bash
uv sync --extra docs
uv run make -C docs html
```

The generated site is available under `docs/build/html/`.
