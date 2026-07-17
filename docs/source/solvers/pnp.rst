Plug-and-Play Reconstruction
============================

Usage and Principle
-------------------

The inference benchmark exposes this solver as ``PnP``. One BenchOpt callback
step performs one plug-and-play proximal-gradient iteration. Starting from
:math:`x_k`, the solver first takes an L2 data-fidelity gradient step and then
applies DRUNet as an implicit image prior:

.. math::

   z_k &= x_k - \gamma \nabla f(x_k), \\
   x_{k+1} &= D_{\sigma}(z_k).

The result is clipped to the dataset signal range by default. An optional
relaxation blends the gradient-step result and denoised result instead of
replacing it directly. The objective evaluates the current reconstruction after
each callback, so ``max-runs`` in the experiment YAML controls the number of PnP
iterations.

A minimal configuration is:

.. code-block:: yaml

   solver:
     - PnP:
         denoiser: drunet
         denoiser_sigma: 0.01
         step_size: 0.1
         init_method: zeros
         profiler_mode: custom

Multi-GPU Distribution
----------------------

Distributed execution uses one process per rank and ordinarily one GPU per
process. The process topology comes from SLURM or ``torchrun``; a DeepInv
distributed context assigns the local device and coordinates collectives. Every
rank follows the same callback decision and receives the assembled iterate.

The two expensive stages can be distributed independently:

``distribute_physics``
   Partitions the stacked acquisition operators across ranks. Each rank computes
   its local forward/adjoint contribution and the data-fidelity result is reduced
   across ranks. This is most useful when there are several acquisition
   operators, views, or frames.

``distribute_denoiser``
   Tiles the full image or volume into overlapping patches, assigns patches to
   ranks, evaluates DRUNet locally, and combines them with overlap-aware blending
   and an all-reduce. For useful scaling, create at least as many patches as
   ranks. ``patch_size``, ``overlap``, and ``max_batch_size`` trade communication
   and seam suppression against memory and throughput.

Either switch can be enabled alone. Enable both when physics and prior are both
large enough to benefit; otherwise the collective overhead may outweigh the
local work. Keep ``slurm_gres``, ``slurm_ntasks_per_node``, ``slurm_nodes``, and
the distribution switches coupled in a BenchOpt parameter grid. See the
:doc:`../getting_started/config_guide` for the coupled-list syntax.

Parameters
----------

Algorithm and prior
~~~~~~~~~~~~~~~~~~~

``denoiser`` (default: ``drunet``)
   Learned denoiser used by the PnP prior. DRUNet is currently the only
   supported value; a 2D or 3D model is selected from the ground-truth shape.

``denoiser_sigma`` (default: ``0.05``)
   Noise-level parameter :math:`\sigma` passed to the denoiser at every
   iteration.

``step_size`` (default: ``None``)
   Gradient step size :math:`\gamma`. When omitted, the solver estimates the
   operator norm and uses ``step_size_scale / ||A||``.

``step_size_scale`` (default: ``0.99``)
   Safety factor applied only to the automatically computed step size.

``denoiser_lambda_relaxation`` (default: ``None``)
   If set to :math:`\lambda`, blends the gradient-step and denoised values with
   weight :math:`\alpha=\gamma\lambda/(1+\gamma\lambda)`. If omitted, the
   denoised value becomes the next iterate directly.

``init_method`` (default: ``pseudo_inverse``)
   Initial reconstruction. ``zeros`` creates a zero tensor;
   ``pseudo_inverse`` uses :math:`A^\dagger y`; and ``adjoint`` uses
   :math:`A^T y`. Operator-based initializations are peak-normalized and clipped
   when the dataset supplies a signal range.

``norm_strategy`` (default: ``clip``)
   ``clip`` clamps the denoiser output to the dataset range. ``dynamic`` maps
   that range to ``[0, 1]`` before denoising and maps it back afterward.

Distribution and memory
~~~~~~~~~~~~~~~~~~~~~~~

``distribute_physics`` (default: ``False``)
   Distribute stacked physics and data-fidelity work across ranks.

``distribute_denoiser`` (default: ``False``)
   Distribute overlapping denoiser patches across ranks.

``patch_size`` (default: ``128``)
   Patch extent along every tiled spatial dimension.

``overlap`` (default: ``32``)
   Neighboring-patch overlap used for smooth reconstruction.

``max_batch_size`` (default: ``0``)
   Maximum local patch batch passed to the denoiser. It is forwarded to the
   distributed processing layer; use a small positive value when memory-bound
   and increase it for throughput when memory permits.

Compilation
~~~~~~~~~~~

``compile`` (default: ``None``)
   Placement of :func:`torch.compile`. ``pre`` compiles the denoiser and
   compatible forward/adjoint methods before distributed wrapping; ``post``
   compiles the wrapped denoiser and data-fidelity gradient; ``fused`` compiles
   an entire gradient--denoise--clip iteration. ``fused`` requires
   ``norm_strategy: clip``. The first compiled iteration includes compilation
   cost, so use profiler warmup iterations when measuring steady state.

Execution and reporting
~~~~~~~~~~~~~~~~~~~~~~~

``slurm_nodes`` (default: ``1``)
   Number of nodes requested for the solver run.

``slurm_ntasks_per_node`` (default: ``1``)
   Number of distributed worker processes per node for SLURM execution.

``slurm_gres`` (default: ``gpu:1``)
   SLURM generic-resource request per node.

``torchrun_nproc_per_node`` (default: ``1``)
   Processes per node when the run is launched through ``torchrun``.

``name_prefix`` (default: ``pnp``)
   Prefix used to identify the run and profiler artifacts.

``profiler_mode``, ``profiler_warmup``, ``profiler_active``, ``profiler_trace_dir``, ``profiler_per_step``, ``profiler_repeat``
   Profiling backend and recording-window options. See :doc:`profiling`.
