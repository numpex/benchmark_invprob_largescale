import re

from benchopt import BaseSolver
from benchopt.stopping_criterion import NoCriterion
from deepinv.distributed import DistributedContext

from toolsbench.invprob.base import InvProb
from toolsbench.profiler import create_profiler
from toolsbench.solver.denoiser import DenoiserSolver
from toolsbench.utils.solver_utils import (
    build_solver_name,
    get_device_from_context,
    setup_distributed_env,
)


class Solver(BaseSolver):
    """Denoiser throughput probe: x <- denoiser(x), eager vs torch.compile."""

    name = "Denoiser"
    sampling_strategy = "callback"
    stopping_criterion = NoCriterion(strategy="callback")

    parameters = {
        "denoiser": ["drunet"],
        "denoiser_sigma": [0.05],
        "compile": [None],
        "distribute_denoiser": [False],
        "patch_size": [128],
        "overlap": [32],
        "max_batch_size": [0],
        "roofline": [True],
        "slurm_nodes": [1],
        "slurm_ntasks_per_node": [1],
        "slurm_gres": ["gpu:1"],
        "torchrun_nproc_per_node": [1],
        "name_prefix": ["denoiser"],
        "profiler_mode": ["custom"],
        "profiler_warmup": [0],
        "profiler_active": [0],
        "profiler_trace_dir": [None],
        "profiler_per_step": [True],
        "profiler_repeat": [1],
    }

    def set_objective(
        self,
        measurements,
        physics,
        ground_truth_shape,
        num_operators,
        min_pixel=0.0,
        max_pixel=1.0,
        weights=None,
        **kwargs,
    ):
        self.problem = InvProb(
            measurements=measurements,
            physics=physics,
            ground_truth_shape=ground_truth_shape,
            num_operators=num_operators,
            min_pixel=min_pixel,
            max_pixel=max_pixel,
            invprob_kwargs={"weights": weights} if weights is not None else None,
        )
        self.ctx = None
        self._algo = None
        self.world_size = setup_distributed_env()
        self.distributed_mode = self.world_size > 1
        self.run_name = build_solver_name(
            self.name_prefix,
            self.slurm_nodes,
            self.slurm_ntasks_per_node,
            self.torchrun_nproc_per_node,
            self.distributed_mode,
        )
        self.name = re.sub(r"_rank\d+$", "", re.sub(r"_\d{8}_\d{6}_", "_", self.run_name))

    def run(self, cb):
        # distribute() tiles the denoiser even on a single rank, so a context is
        # needed whenever the denoiser is distributed, not only in multi-GPU runs.
        if self.distributed_mode or self.distribute_denoiser:
            with DistributedContext(seed=42, cleanup=True) as ctx:
                self.ctx = ctx
                self._run_with_context(cb, ctx)
        else:
            self._run_with_context(cb, ctx=None)

    def _run_with_context(self, cb, ctx):
        device = get_device_from_context(ctx)
        profiler = create_profiler(
            self.profiler_mode, device, self.run_name,
            warmup=self.profiler_warmup, active=self.profiler_active,
            trace_dir=self.profiler_trace_dir,
            per_step=self.profiler_per_step,
            repeat=self.profiler_repeat,
        )
        with profiler:
            self._algo = DenoiserSolver(
                problem=self.problem,
                device=device,
                profiler=profiler,
                ctx=ctx,
                distributed_mode=self.distributed_mode,
                denoiser=self.denoiser,
                denoiser_sigma=self.denoiser_sigma,
                compile=self.compile,
                distribute_denoiser=self.distribute_denoiser,
                patch_size=self.patch_size,
                overlap=self.overlap,
                max_batch_size=self.max_batch_size,
                roofline=self.roofline,
            )
            self._algo.run(cb)
        profiler.finalize(ctx)

    def get_result(self):
        result = dict(name=self.run_name)
        result.update(self._algo.get_result())
        return result

    def get_next(self, stop_val):
        return stop_val + 1
