Unrolled Plug-and-Play Training
===============================

Usage and Principle
-------------------

The training benchmark exposes this solver as ``UnrolledPnP``. It unfolds a
fixed number of proximal-gradient iterations into a differentiable network.
Each stage contains an L2 data-fidelity update followed by a DRUNet PnP prior;
the step sizes and prior parameters are trainable alongside the denoiser.

One BenchOpt callback step is one complete supervised training step on the
in-memory batch: unrolled forward pass, MSE loss, backward pass, and Adam update.
Consequently, ``max-runs`` controls training steps rather than unrolled depth;
``n_iter`` controls the number of reconstruction stages evaluated inside each
training step. The profiler labels the forward and backward regions separately.

A minimal configuration is:

.. code-block:: yaml

   solver:
     - UnrolledPnP:
         denoiser: drunet
         n_iter: 4
         init_stepsize: 0.8
         denoiser_sigma: 0.05
         distribute_model: true
         patch_size: 512
         overlap: 32
         max_batch_size: 1
         checkpoint_batches: auto
         profiler_mode: custom

Multi-GPU Distribution
----------------------

As in inference, distributed runs use one process per rank and normally one GPU
per process. When a distributed context is active, the stacked physics is always
partitioned across ranks and its contributions are reduced using a mean. This
shares the acquisition work used by every unrolled stage.

With ``distribute_model: true``, the full unrolled model is evaluated on
overlapping spatial patches distributed among ranks. Local patch batches run
the forward and backward graphs, and overlap-aware collective reduction
reassembles the full reconstruction. The model is distributed spatially rather
than by giving each rank an independent data batch, which is appropriate for a
single very large image or volume.

``patch_size`` and ``overlap`` determine the spatial decomposition;
``max_batch_size`` controls how many local patches are processed together;
``checkpoint_batches`` controls the activation-memory/recomputation policy
accepted by the distributed model wrapper. These settings should be tuned as a
group: smaller batches and more checkpointing lower peak memory, while larger
batches and less recomputation generally improve throughput. Ensure there are
enough patches to keep all ranks busy.

Parameters
----------

Architecture and optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``denoiser`` (default: ``drunet``)
   Denoiser used as the PnP prior in every unrolled stage. DRUNet is currently
   the only supported value.

``n_iter`` (default: ``4``)
   Number of unfolded PGD stages per forward pass. Increasing it adds both
   reconstruction depth and activation memory.

``init_stepsize`` (default: ``0.8``)
   Initial value assigned to the trainable step size of every unfolded stage.

``denoiser_sigma`` (default: ``0.05``)
   Noise-level parameter passed to the denoiser in each stage.

``grad_clip`` (default: ``1.0``)
   Gradient clipping threshold stored on the DeepInv trainer. The instrumented
   training step currently does not call the trainer clipping hook, so this
   parameter does not alter the update.

``image_size`` (default: ``None``)
   Optional solver-side square image size. When it differs from the dataset
   size, the ground truth is bilinearly resized and measurements are regenerated
   with the updated physics. Omit it to preserve the dataset dimensions.

Distribution and memory
~~~~~~~~~~~~~~~~~~~~~~~

``distribute_model`` (default: ``False``)
   Enable overlapping-patch distribution of the unrolled model. Physics is
   distributed whenever more than one rank is active, independently of this
   switch.

``patch_size`` (default: ``128``)
   Patch extent along the tiled spatial dimensions.

``overlap`` (default: ``32``)
   Neighboring-patch overlap used when assembling the model output.

``max_batch_size`` (default: ``1``)
   Maximum number of local patches processed in one model call. Lower values
   reduce peak activation memory.

``checkpoint_batches`` (default: ``auto``)
   Activation-checkpointing policy forwarded to the distributed model wrapper.
   Experiment configurations use ``always`` to favor memory savings, ``never``
   to favor speed, and ``auto`` for automatic selection.

``deterministic`` (default: ``True``)
   Requests deterministic behavior from the distributed context. Disable it
   only when nondeterministic kernels are acceptable for the experiment.

Execution and reporting
~~~~~~~~~~~~~~~~~~~~~~~

``slurm_nodes`` (default: ``1``)
   Number of nodes requested for the solver run.

``slurm_ntasks_per_node`` (default: ``1``)
   Number of distributed worker processes per node for SLURM execution.

``slurm_gres`` (default: ``gpu:1``)
   SLURM generic-resource request per node.

``torchrun_nproc_per_node`` (default: ``1``)
   Processes per node when launched through ``torchrun``.

``name_prefix`` (default: ``unrolled_pnp``)
   Prefix used to identify the run and profiler artifacts.

``profiler_mode``, ``profiler_warmup``, ``profiler_active``, ``profiler_trace_dir``, ``profiler_per_step``, ``profiler_repeat``, ``profiler_save_file``
   Profiling backend, recording window, and output options. See
   :doc:`profiling`.
