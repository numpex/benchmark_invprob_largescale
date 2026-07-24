Troubleshooting
===============

This page collects the failures most often seen when running the benchmark on a
cluster, with the checks that usually resolve them. It assumes the setup in
:doc:`getting_started/run_on_cluster`.

The Job Hangs
-------------

The submission is accepted but nothing progresses: no reconstruction metrics
appear, the log stops after the banner, and waiting several minutes changes
nothing. The common causes are missing preparation and a job that is not
actually running.

**Datasets were not prepared.** BenchOpt calls each dataset's ``prepare()``
before the run, which is where reusable inputs are downloaded, simulated, and
cached. Compute nodes may not have internet access, so a first-time download
launched from inside the job blocks indefinitely. Prepare the data from a
login node, which does have network access, before submitting:

.. code-block:: bash

   benchopt prepare benchmark_inference/. --config <experiment.yml>

Preparation is cached by BenchOpt, so this is a one-time cost per dataset. See
:doc:`use_cases/index` for per-dataset preparation details.

**Denoiser weights were not cached.** Pretrained networks weights are downloaded 
from the internet on first use (``pretrained="download"``), which blocks on an offline 
compute node the same way an unprepared dataset does.
``benchopt prepare`` caches these weights into the shared torch hub cache
(``~/.cache/torch/hub/checkpoints/``) alongside the dataset inputs, so running it
from a login node covers both.


**The job is not actually running.** A "hang" is often a job still sitting in
the queue or already failed at submission. Check its state:

.. code-block:: bash

   squeue --me

If the job is pending (``PD``) with a reason like ``(Resources)`` or
``(Priority)``, the requested resources are simply not free yet. If it never
appears at all, the submission was rejected; if it stays ``PD`` with a QoS reason
such as ``(QOSMaxJobsPerUserLimit)`` or ``(QOSMaxWallDurationPerJobLimit)`` in the
reason column, it was accepted but capped by the QoS. Each QoS caps the number
of simultaneous or queued jobs and the maximum wall time, and a development QoS
is far more restrictive than a production one.
Requesting more jobs than the QoS allows, or a ``slurm_time`` longer than the QoS
permits, is refused. Review ``account``, ``qos``, and ``slurm_time`` in the
parallel configuration against your cluster's QoS table, and inspect the Submitit
log files under the benchmark's ``benchopt_run/`` directory for the rejection
message.

The Job Runs but Stops Early
----------------------------

The run starts and produces some results, then ends before completing every
configured solver or all ``max-runs`` iterations. Two independent time limits can
cause this.

**BenchOpt per-solver timeout.** BenchOpt stops a solver once a single run
exceeds its ``--timeout``, which defaults to **100 seconds**. A large image, a
slow forward operator, or many iterations can cross this limit and the solver is
recorded as timed out rather than failed. Raise or remove the limit explicitly:

.. code-block:: bash

   benchopt run benchmark_inference/. --config <experiment.yml> --timeout 30m
   # or, to disable it entirely:
   benchopt run benchmark_inference/. --config <experiment.yml> --no-timeout

**SLURM wall time.** The job itself is killed when it exceeds ``slurm_time`` in
the parallel configuration (the checked-in examples use ``1800`` seconds).
Increase it for long experiments, keeping it within what the selected QoS allows.

In both cases the cause is stated in the log: check the Submitit output under
``benchopt_run/`` for a timeout notice or a SLURM ``TIMEOUT`` / ``CANCELLED``
line.
