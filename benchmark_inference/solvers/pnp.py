import re

from benchopt import BaseSolver
from deepinv.distributed import DistributedContext

from toolsbench.invprob.base import InvProb
from toolsbench.profiler import create_profiler
from toolsbench.solver.pnp import PnPSolver
from toolsbench.utils.solver_utils import (
    build_solver_name,
    get_device_from_context,
    setup_distributed_env,
)


class Solver(BaseSolver):
    """PnP solver."""

    name = "PnP"
    sampling_strategy = "callback"

    parameters = {
        "denoiser": ["drunet"],
        "image_size": [None],
        "patch_size": [128],
        "overlap": [32],
        "max_batch_size": [0],
        "denoiser_lambda_relaxation": [None],
        "step_size": [None],
        "step_size_scale": [0.99],
        "denoiser_sigma": [0.05],
        "distribute_physics": [False],
        "distribute_denoiser": [False],
        "init_method": ["pseudo_inverse"],
        "norm_strategy": ["clip"],
        "compile": [None],
        "slurm_nodes": [1],
        "slurm_ntasks_per_node": [1],
        "slurm_gres": ["gpu:1"],
        "torchrun_nproc_per_node": [1],
        "name_prefix": ["pnp"],
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
        self.distributed_mode = self.distribute_denoiser or self.distribute_physics
        self.run_name = build_solver_name(
            self.name_prefix,
            self.slurm_nodes,
            self.slurm_ntasks_per_node,
            self.torchrun_nproc_per_node,
            self.distributed_mode,
        )
        self.name = re.sub(
            r"_rank\d+$", "", re.sub(r"_\d{8}_\d{6}_", "_", self.run_name)
        )

    def run(self, cb):
        if self.distributed_mode:
            with DistributedContext(seed=42, cleanup=True) as ctx:
                self.ctx = ctx
                self._run_with_context(cb, ctx)
        else:
            self._run_with_context(cb, ctx=None)

    def _run_with_context(self, cb, ctx):
        device = get_device_from_context(ctx)
        profiler = create_profiler(
            self.profiler_mode,
            device,
            self.run_name,
            warmup=self.profiler_warmup,
            active=self.profiler_active,
            trace_dir=self.profiler_trace_dir,
            per_step=self.profiler_per_step,
            repeat=self.profiler_repeat,
        )
        with profiler:
            self._algo = PnPSolver(
                problem=self.problem,
                device=device,
                profiler=profiler,
                ctx=ctx,
                distributed_mode=self.distributed_mode,
                denoiser=self.denoiser,
                patch_size=self.patch_size,
                overlap=self.overlap,
                max_batch_size=self.max_batch_size,
                step_size=self.step_size,
                step_size_scale=self.step_size_scale,
                denoiser_sigma=self.denoiser_sigma,
                denoiser_lambda_relaxation=self.denoiser_lambda_relaxation,
                distribute_physics=self.distribute_physics,
                distribute_denoiser=self.distribute_denoiser,
                init_method=self.init_method,
                norm_strategy=self.norm_strategy,
                compile=self.compile,
                image_size=self.image_size,
            )
            self._algo.run(cb)
        profiler.finalize(ctx)

    def get_result(self):
        result = dict(name=self.run_name)
        result.update(self._algo.get_result())
        return result

    def get_next(self, stop_val):
        return stop_val + 1
