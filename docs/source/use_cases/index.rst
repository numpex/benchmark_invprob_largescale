Use Cases
=========

The benchmark provides datasets for several imaging modalities and controlled
workloads. Each page describes the source data, forward model, BenchOpt
installation and preparation steps, and parameters available in experiment YAML.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Multi-frame Super-resolution
      :link: multiframe_super_resolution
      :link-type: doc
      :class-card: benchmark-card

      Recover a high-resolution color image from multiple blurred, downsampled,
      and noisy observations.

   .. grid-item-card:: Tomography
      :link: tomography
      :link-type: doc
      :class-card: benchmark-card

      Reconstruct 2D slices or 3D volumes from ASTRA-backed parallel-beam and
      cone-beam projections.

   .. grid-item-card:: Radio Interferometry
      :link: radio_interferometry
      :link-type: doc
      :class-card: benchmark-card

      Recover a sky image from sparse Fourier measurements simulated for a
      configurable telescope observation.

   .. grid-item-card:: Synthetic Workloads
      :link: synthetic
      :link-type: doc
      :class-card: benchmark-card

      Generate scalable 2D and 3D signals for controlled super-resolution and
      denoising workloads.

Installing and Preparing Data
-----------------------------

Installation and preparation are separate operations:

- ``benchopt install`` installs requirements declared by the selected benchmark
  components. A dataset may also provide a custom installer, as radio
  interferometry does for its simulation container.
- ``benchopt prepare`` calls the selected dataset's ``prepare()`` method. This is
  where reusable input files or simulations are downloaded and cached. It does
  not construct the tensors and operators for every benchmark run; that happens
  when BenchOpt loads the dataset.

Both commands can select one dataset directly or use an experiment configuration:

.. code-block:: bash

   benchopt install benchmark_inference/. -d tomography_3d
   benchopt prepare benchmark_inference/. -d tomography_3d

   benchopt install benchmark_inference/. \
       --config benchmark_inference/configs/examples/tomography_3d.yml
   benchopt prepare benchmark_inference/. \
       --config benchmark_inference/configs/examples/tomography_3d.yml

Preparation is cached by BenchOpt. Use ``benchopt prepare --force`` when a cached
preparation must be repeated.

.. toctree::
   :hidden:
   :maxdepth: 1

   multiframe_super_resolution
   tomography
   radio_interferometry
   synthetic
