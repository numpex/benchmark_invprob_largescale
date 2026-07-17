Multi-frame Super-resolution
============================

Dataset name: ``multiframe_superres``

Data and Inverse Problem
------------------------

For each frame, the benchmark applies an oriented Gaussian blur, downsamples the
blurred image by a factor of two, and adds independent Gaussian noise. With
:math:`K` frames, the measurements are

.. math::

   y_k = S B_k x + n_k, \qquad k=1,\ldots,K,

where :math:`B_k` is the blur for frame :math:`k` and :math:`S` is the
subsampling operator. Blur orientations are distributed over 180 degrees. The
operators are represented as a DeepInv stacked physics object so a solver can
process or distribute the frames separately.

The dataset interface fixes the downsampling factor to ``2``, the blur kernel to
``5 x 5``, and the blur standard deviation to ``1.0``. These values are part of
the current use case rather than configurable dataset parameters.

What ``benchopt install`` Does
------------------------------

``multiframe_superres`` declares no additional dataset-specific software
requirements. Installation still resolves requirements from the selected
objective and solvers, but it does not download the image. When the repository
has already been installed with its Python dependencies, there is normally no
extra dataset installation step.

What ``benchopt prepare`` Does
------------------------------

Preparation downloads ``butterfly.png`` into BenchOpt's data directory for the
``multiframe_superres`` dataset if it is not already cached. Preparation is
independent of image size, batch size, noise level, and number of frames, so a
single downloaded source image serves every configuration. Preparing on a login
node is recommended when compute nodes cannot access the internet.

Available Dataset Parameters
----------------------------

``image_size`` (default ``2048``)
   Target image size. An integer produces a square 2D image.

``batch_size`` (default ``1``)
   Number of copies of the source image in the generated batch.

``channels`` (default ``3``)
   Number of image channels requested from the data interface. This use case is
   intended for color data.

``num_operators`` (default grid ``1, 8, 16``)
   Number of low-resolution frames and therefore the number of stacked physics
   operators.

``noise_level`` (default ``0.1``)
   Standard deviation of the Gaussian noise added to every frame.

``seed`` (default ``42``)
   Seed used to initialize the distributed dataset context. Frame noise remains
   reproducible across runs.

A minimal dataset section is:

.. code-block:: yaml

   dataset:
     - multiframe_superres:
         image_size: 2048
         batch_size: 1
         channels: 3
         num_operators: 8
         noise_level: 0.1
         seed: 42
