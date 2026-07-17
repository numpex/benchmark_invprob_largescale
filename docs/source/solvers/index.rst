Solvers and Profiling
=====================

The benchmark provides an iterative plug-and-play solver for reconstruction and
an unrolled plug-and-play solver for supervised training. Both use the same
distributed execution context and profiling interface, making it possible to
compare numerical quality and hardware behavior across single-GPU, multi-GPU,
and multi-node runs.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Plug-and-Play Reconstruction
      :link: pnp
      :link-type: doc
      :class-card: benchmark-card

      Alternate a data-fidelity gradient step and a learned denoising prior,
      with independent distribution of physics and denoising.

   .. grid-item-card:: Unrolled PnP Training
      :link: unrolling
      :link-type: doc
      :class-card: benchmark-card

      Train an unrolled PGD model one supervised optimization step at a time,
      with distributed physics and patch-based model execution.

   .. grid-item-card:: Profilers
      :link: profiling
      :link-type: doc
      :class-card: benchmark-card

      Choose lightweight wall-clock metrics, operator-level PyTorch traces, or
      Nsight Systems timelines.

.. toctree::
   :hidden:
   :maxdepth: 1

   pnp
   unrolling
   profiling
