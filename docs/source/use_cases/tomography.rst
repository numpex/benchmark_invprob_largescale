Tomography
==========

Dataset names: ``tomography_2d`` and ``tomography_3d``

Both tomography datasets use DeepInv's ``TomographyWithAstra`` physics, backed by
the ASTRA Toolbox, but they represent different data and acquisition geometries.
The 2D case generates a sinogram from a reference slice, whereas the 3D case uses
measured cone-beam data and its supplied scanner trajectory.

2D Data and Inverse Problem
---------------------------

The 2D dataset downloads a Shepp--Logan phantom and resizes it to the requested
square image size. It constructs evenly spaced projection angles over 180
degrees, creates an ASTRA parallel-beam operator, and generates noisy projections
from the phantom.

When ``num_operators`` is greater than one, the angle sequence and sinogram are
split into contiguous groups. Each group defines one physics operator and one
measurement tensor. Together they still represent the complete acquisition.

3D Data and Inverse Problem
---------------------------

The 3D dataset downloads ``Walnut-CBCT_8.pt``. It contains a dense reference
volume, a cone-beam sinogram, and ASTRA geometry vectors describing the source
and detector pose for every projection. The physical volume and detector geometry
are fixed by the acquisition; resizing either would make the supplied geometry
inconsistent. Consequently, ``image_size`` is accepted by the BenchOpt dataset
interface but ignored when the Walnut data is loaded.

``num_projections`` selects an approximately uniform subset of the stored views.
The selected sinogram and matching geometry vectors are split across
``num_operators``. An ASTRA cone-beam operator is then created for each split.
The current implementation requires the stored sinogram: setting
``use_dataset_sinogram`` to ``false`` raises ``NotImplementedError`` because a
new 3D sinogram is not generated with a forward pass.

What ``benchopt install`` Does
------------------------------

Neither tomography dataset declares an additional custom installer. The ASTRA,
DeepInv, and PyTorch dependencies are normal project requirements. With the
project already installed, ``benchopt install`` has no tomography-specific file
to install; input data is handled by preparation.

What ``benchopt prepare`` Does
------------------------------

For ``tomography_2d``, preparation downloads ``SheppLogan.png`` into BenchOpt's
shared tomography data directory. For ``tomography_3d``, it downloads the Walnut
``.pt`` archive from the ``romainvo/ct_examples`` dataset on Hugging Face.
Existing non-empty files are reused. Preparation is cached independently of the
parameter grid because all configurations use the same source file.

Available 2D Parameters
-----------------------

``image_size`` (default ``512``)
   Square reconstruction size, or an explicit two-element spatial shape. The
   loaded phantom is resized accordingly; the final shape must be square.

``batch_size`` (default ``1``)
   Number of phantom images in the batch.

``num_operators`` (default ``1``)
   Number of groups into which the angles and measurements are split. It cannot
   exceed ``num_angles``.

``num_angles`` (default ``100``)
   Number of evenly spaced projection angles used to generate the 2D sinogram.

``num_projections`` (default ``100``)
   Present in the common tomography interface but not used by the current 2D
   implementation; use ``num_angles`` to control 2D sampling.

``noise_level`` (default ``0.01``)
   Standard deviation of Gaussian noise applied by the 2D ASTRA physics.

``seed`` (default ``42``)
   Base seed for the per-operator 2D noise generators.

``geometry_type_2d`` (default ``parallel``)
   ASTRA geometry used for the 2D acquisition.

``detector_spacing_2d`` (default ``1.0``)
   Spacing between detector elements.

``pixel_spacing_2d`` (default ``1.0``)
   Physical spacing of image pixels; a scalar or two-element spacing can be
   passed through the underlying physics interface.

Available 3D Parameters
-----------------------

``image_size`` (default ``512``)
   Accepted for a uniform dataset interface but ignored for the fixed Walnut
   volume.

``batch_size`` (default ``1``)
   Number of copies of the stored volume and sinogram.

``num_operators`` (default ``1``)
   Number of contiguous projection groups. It cannot exceed
   ``num_projections``.

``num_projections`` (default ``100``)
   Number of stored Walnut views selected before splitting.

``num_angles`` (default ``100``)
   Present in the common interface but not used by the 3D path; sampling is
   controlled by ``num_projections``.

``geometry_type_3d`` (default ``conebeam``)
   ASTRA geometry type. The supplied geometry vectors describe the actual
   cone-beam trajectory.

``use_dataset_sinogram`` (default ``true``)
   Use the stored Walnut sinogram. This must currently remain ``true``.

``noise_level`` (default ``0.01``) and ``seed`` (default ``42``)
   Accepted by the dataset interface. They affect generated 2D measurements but
   are not applied to the stored 3D sinogram in the current implementation.

Example configurations are available under
``benchmark_inference/configs/examples/tomography_2d.yml`` and
``benchmark_inference/configs/examples/tomography_3d.yml``.
