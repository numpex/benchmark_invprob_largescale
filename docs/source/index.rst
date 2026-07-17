Benchmarking Inverse Problems at Scale
======================================

How well does an inverse-problem method perform when the unknown is no longer a
small test image, but a multi-megapixel image or a full 3D volume? This benchmark
turns that question into reproducible experiments, from reconstruction quality
to GPU memory, runtime, and distributed scaling.

Built with `BenchOpt <https://benchopt.github.io>`_ and
`DeepInv <https://deepinv.github.io>`_, the suite provides a common framework for
evaluating large-scale imaging pipelines on a single GPU, across multiple GPUs,
and on multi-node SLURM clusters.

Inverse problems recover an unknown signal :math:`x` from indirect and noisy
measurements :math:`y`:

.. math::

   y = A x + n,

where :math:`A` models the acquisition system and :math:`n` represents
measurement noise. At scale, reconstruction quality is only part of the story:
the forward operator, learned prior, memory footprint, and communication between
devices can all become limiting factors. The benchmark measures these trade-offs
under controlled, repeatable conditions.

Imaging Problems and Datasets
-----------------------------

The suite covers complementary acquisition models and data regimes, with images
and volumes ranging from roughly one million to one hundred million unknowns:

- **Multi-frame super-resolution** reconstructs a high-resolution color image
  from several blurred, downsampled, and noisy frames. Varying the image size and
  number of frames stresses both the stacked forward model and the denoising
  prior.
- **2D tomography** reconstructs a slice from noisy parallel-beam projections.
  The number of angles, detector samples, and operators can be varied to study
  increasingly expensive projection and backprojection steps.
- **3D tomography** reconstructs a volumetric walnut from cone-beam CT
  measurements. This real-world geometry combines a large 3D unknown with a
  demanding forward operator and memory footprint.
- **Radio interferometry** recovers a sky image from sparse Fourier-domain
  measurements generated from a configurable telescope observation. It exposes
  the computational challenges of wide, high-dynamic-range astronomical images
  and non-Cartesian sampling.

Benchmark Inference
-------------------

The inference benchmark targets the **quality, speed, and memory cost of solving
large inverse problems**. It evaluates plug-and-play reconstruction algorithms
and standalone denoisers while varying image size, physics complexity,
solver settings, and hardware resources.

Alongside PSNR or SSIM, it records per-step timings and GPU memory for the
physics gradient and denoising stages. Its experiments are designed to answer
questions such as:

- Does adding GPUs reduce reconstruction time for a fixed problem size?
- Which stage limits performance: the forward model, the denoiser, or
  communication?
- What is the quality and performance impact of patching, overlap, or
  compilation?
- How large a reconstruction can a given GPU configuration process?

Benchmark Training
------------------

The training benchmark targets the **cost of optimizing unrolled reconstruction
networks at large scale**. Each benchmark iteration monitors one supervised
training step of an unrolled plug-and-play model.

It focuses on the choices that determine whether training scales:
strong and weak scaling across GPUs and nodes, communication overhead, patch
batch size, and activation checkpointing. The resulting measurements show where
additional hardware improves throughput, where communication dominates, and
which memory-saving strategies enable larger images or volumes.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started/run_on_cluster
   getting_started/config_guide
   use_cases/index
   solvers/index

Explore the Benchmark
---------------------

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Run on a Cluster
      :link: getting_started/run_on_cluster
      :link-type: doc
      :class-card: benchmark-card

      Load the environment, configure SLURM, and launch an experiment.

   .. grid-item-card:: Use Cases
      :link: use_cases/index
      :link-type: doc
      :class-card: benchmark-card

      Explore the data, inverse problems, preparation steps, and parameters.

   .. grid-item-card:: Configuration Guide
      :link: getting_started/config_guide
      :link-type: doc
      :class-card: benchmark-card

      Build experiment grids for datasets, solvers, and distributed resources.

   .. grid-item-card:: Solvers and Profiling
      :link: solvers/index
      :link-type: doc
      :class-card: benchmark-card

      Understand the algorithms, multi-GPU execution, parameters, and profilers.
