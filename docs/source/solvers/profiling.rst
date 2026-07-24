Profiling Solvers
=================

All solver backends use the same profiling interface. Named regions become
``gradient`` and ``denoise`` for PnP inference, and ``forward`` and ``backward``
for unrolled training. At the end of an iteration, current measurements are
returned through the BenchOpt objective alongside reconstruction metrics.

The common recording window is controlled by:

``profiler_mode`` (default: ``custom``)
   One of ``None``, ``custom``, ``torch``, or ``nvidia``. In YAML, use ``null``
   for Python ``None``.

``profiler_warmup`` (default: ``0``)
   Number of complete solver iterations to execute before recording. Warmup is
   particularly important for CUDA initialization and :func:`torch.compile`.

``profiler_active`` (default: ``0``)
   Number of iterations to record after warmup. Zero records every remaining
   iteration.

``profiler_save_file`` (default: ``False``)
   Write profiler CSV data under ``outputs/`` when the backend supports it. Both
   ``PnP`` and ``UnrolledPnP`` expose this parameter; metrics are still returned
   to BenchOpt when file output is disabled.

How Profiling Works
-------------------

Each solver wraps its named regions in ``profiler.track_step(name)`` and calls
``profiler.end_iteration()`` once per BenchOpt iteration. ``track_step`` opens the
recording context for that region (wall-clock timing on every backend, plus
operator- and communication-level capture on the torch backend);
``end_iteration`` closes the iteration, computes ``total_time_sec`` and
``max_gpu_mb``, and stores the per-iteration values that BenchOpt reads through
``get_current_metrics()``. Recording is limited to the window set by
``profiler_warmup`` and ``profiler_active``; outside that window the solver runs
at full speed and reports nothing.

No Profiling
------------

**Principle.** ``profiler_mode: null`` selects a no-op profiler. Named sections
remain valid context managers but do not synchronize devices, collect metrics,
or create files.

**What it measures.** Nothing. Use it when measuring through an external system
or when profiler overhead would interfere with the experiment.

Custom Profiler
---------------

**Principle.** ``profiler_mode: custom`` surrounds named regions with a
high-resolution wall clock. On CUDA it synchronizes the selected GPU around
each region and resets/reads PyTorch peak allocated memory statistics.

**What it measures.** Each active iteration reports ``total_time_sec`` and
``max_gpu_mb``. It also reports ``<section>_time_sec`` and
``<section>_max_gpu_mb`` for every named region. ``total_time_sec`` spans the
whole iteration, including the BenchOpt objective callback that scores the
reconstruction (e.g. PSNR/SSIM against the ground truth) and the optimizer step,
so it exceeds the sum of the ``<section>_time_sec`` values. GPU
memory is peak tensor memory allocated through PyTorch, not total
device usage or reserved allocator capacity. On CPU, memory fields are zero.

**Output and trade-offs.** Metrics are returned directly to BenchOpt. With file
saving enabled, one row per active iteration is written to
``outputs/<run-name>_gpu_metrics.csv``. Explicit CUDA synchronization makes the
timings easy to interpret. In distributed runs the denoiser/physics collectives already synchronize the ranks,
so the added cost is negligible. This is the preferred backend for lightweight
scaling and memory studies.

PyTorch Profiler
----------------

**Principle.** ``profiler_mode: torch`` uses :mod:`torch.profiler`. Named solver
regions are recorded as user annotations, and their child PyTorch operators are
collected from the CPU/CUDA event tree. Communication operators are identified
from NCCL, Gloo, and c10d annotations and attributed to their containing region
where possible.

**What it measures.** In per-step mode, BenchOpt receives
``<section>_cpu_sec``, ``<section>_cuda_sec``, and
``<section>_comm_sec`` together with ``comm_cuda_sec`` for communication inside
named regions, ``comm_sync_sec`` for communication outside them,
``total_time_sec``, and ``max_gpu_mb``. Detailed operator rows contain
``cpu_sec``, ``cuda_sec``, ``self_cuda_sec``, ``mem_mb``, ``self_mem_mb``, and
call ``count`` grouped by iteration, section, and operator.

**Modes and output.** The additional parameters are:

``profiler_per_step`` (default: ``True``)
   When true, each iteration is a separate profiler cycle. Section summaries
   are returned to BenchOpt and optional CSV rows retain the iteration index.
   When false, it runs ``torch.profiler`` in standard scheduled mode: one cycle
   over the whole window, operator rows aggregated with ``iter=agg`` (a lighter
   CSV), and only wall time and peak memory returned per iteration. This mode
   can export a Chrome trace via ``profiler_trace_dir``.

``profiler_repeat`` (default: ``1``)
   With ``profiler_per_step: false``, number of warmup/active schedule cycles.
   Zero repeats indefinitely. It has no effect in per-step mode.

``profiler_trace_dir`` (default: ``None``)
   With ``profiler_per_step: false``, export a Chrome trace named
   ``rank_<rank>.pt.trace.json`` to this directory. A trace directory is
   incompatible with per-step mode because the profiler is reset each
   iteration.

With file saving enabled, detailed rows are written to
``outputs/<run-name>_gpu_metrics.csv``. PyTorch profiling has substantially more
overhead than the custom backend; select a short active window representative of
steady-state execution. 

NVIDIA Nsight Systems Profiler
------------------------------

**Principle.** ``profiler_mode: nvidia`` adds NVTX ranges for every named region
and for each active iteration (``iter_N``). It also calls the CUDA profiler start
and stop APIs at the recording-window boundaries. NVIDIA Nsight Systems,
launched around the actual worker process, captures these markers and the CUDA
timeline.

**What it measures.** Internally, the backend returns per-iteration
``total_time_sec`` and ``max_gpu_mb``. The external ``nsys`` report provides the
detailed CPU/GPU timeline, kernel activity, memory activity, idle gaps, and NVTX
region summaries. Without ``nsys`` attached, the markers are inert and only the
two internal metrics are available.

**Output and distributed use.** With file saving enabled, internal metrics are
written to ``outputs/<run-name>_gpu_metrics.csv``. Nsight writes its own
``.nsys-rep`` output according to the worker launch command. With Submitit, do
not wrap the outer ``benchopt run`` command: it only submits and polls for a
remote job. Instead, prefix the real SLURM worker through ``slurm_python``. The
repository provides ``benchmark_training/configs/config_parallel_nsys.yml`` as
an example that creates rank-specific reports. For local execution, wrap the
normal run directly:

.. code-block:: bash

   nsys profile -o my_run --force-overwrite=true \
       --trace=cuda,nvtx --trace-fork-before-exec=true \
       --cuda-memory-usage=true --sample=none \
       benchopt run benchmark_training/. --config <experiment.yml>

Warmup and active parameters control both the NVTX iteration ranges and the
CUDA capture range. Nsight Systems is the best choice when a scalar timing does
not explain synchronization, overlap, or communication behavior.
