Synthetic Workloads
===================

Dataset name: ``simulated``

Data and Inverse Problem
------------------------

The synthetic dataset creates signals directly on the target device. Each
channel combines a geometric region, a smooth directional gradient, and a
higher-frequency sinusoidal pattern, with values clipped to ``[0, 1]``. Because
there is no external source image, the workload can be scaled to arbitrary 2D
image or 3D volume shapes without introducing data-loading differences.

The inverse problem depends on dimensionality:

- An integer or two-element ``image_size`` creates a 2D multi-frame
  super-resolution problem. Each frame is blurred, downsampled by two, and
  corrupted by independent Gaussian noise, using the same fixed acquisition
  settings as the real-image super-resolution use case.
- A three-element ``image_size`` creates a 3D denoising problem. The forward
  operator is the identity and each measurement is an independently corrupted
  copy of the synthetic volume. Super-resolution is not used because the current
  multi-frame implementation is 2D-only.

The operators are stacked in both cases, so ``num_operators`` controls the number
of measurements made from the same ground truth.

What ``benchopt install`` Does
------------------------------

The dataset declares no additional requirements or custom installer. Its PyTorch
and DeepInv dependencies are part of the project installation.

What ``benchopt prepare`` Does
------------------------------

Preparation is intentionally empty: no file is downloaded and no tensor is
stored. The signal, measurements, and physics are generated when BenchOpt loads
the dataset for a run.

Available Dataset Parameters
----------------------------

``image_size`` (default ``2048``)
   Spatial shape of the signal. Use an integer for a square 2D image, two values
   for a rectangular 2D image, or three values for a 3D volume. For example,
   ``[128, 128, 128]`` selects the 3D denoising path.

``batch_size`` (default ``1``)
   Number of copies of the generated signal.

``channels`` (default ``3``)
   Number of independently generated signal channels.

``num_operators`` (default grid ``1, 8, 16``)
   Number of super-resolution frames in 2D or noisy copies in 3D.

``noise_level`` (default ``0.1``)
   Standard deviation of Gaussian measurement noise.

``seed`` (default ``42``)
   Seed used to initialize the dataset's distributed context and preserve
   reproducible execution.

A configuration can select either dimensionality:

.. code-block:: yaml

   dataset:
     - simulated:
         image_size: 2048
         batch_size: 1
         channels: 3
         num_operators: 8
         noise_level: 0.1
         seed: 42
     - simulated:
         image_size: [128, 128, 128]
         batch_size: 1
         channels: 1
         num_operators: 4
         noise_level: 0.1
         seed: 42
