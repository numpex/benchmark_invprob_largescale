import torch
import torch.nn.functional as F

from benchopt import BaseSolver
from benchopt.stopping_criterion import NoCriterion
from deepinv.distributed import DistributedContext

from toolsbench.invprob.base import InvProb
from toolsbench.profiler import create_profiler
from toolsbench.solver.unrolled_pnp import UnrolledPnPSolver
from toolsbench.utils.solver_utils import (
    build_solver_name,
    get_device_from_context,
    setup_distributed_env,
)


class Solver(BaseSolver):
    """Unrolled PnP training solver (one training step per iteration)."""

    name = "UnrolledPnP"

    sampling_strategy = "callback"
    # Disable convergence checking — run for exactly max_runs training steps.
    stopping_criterion = NoCriterion()

    parameters = {
        # --- Model architecture ---
        "denoiser": ["drunet"],
        "n_iter": [4],
        "init_stepsize": [0.8],
        "denoiser_sigma": [0.05],
        # --- Optimizer ---
        "grad_clip": [1.0],
        # --- Distributed processing ---
        "distribute_model": [False],
        "patch_size": [128],
        "overlap": [32],
        "max_batch_size": [1],
        "checkpoint_batches": ["auto"],
        # --- Image size (solver-side, None means use dataset size) ---
        "image_size": [None],
        # --- SLURM / torchrun ---
        "slurm_nodes": [1],
        "slurm_ntasks_per_node": [1],
        "slurm_gres": ["gpu:1"],
        "torchrun_nproc_per_node": [1],
        # --- Logging / profiling ---
        "name_prefix": ["unrolled_pnp"],
        "profiler_mode": ["custom"],
        "profiler_warmup": [0],
        "profiler_active": [0],
        "profiler_trace_dir": [None],
        "profiler_per_step": [True],
        "profiler_repeat": [1],
        "profiler_save_file": [False],
    }

    def set_objective(
        self,
        ground_truth,
        measurements,
        physics,
        ground_truth_shape,
        num_operators,
        min_pixel=0.0,
        max_pixel=1.0,
        **kwargs,
    ):
        dataset_image_size = ground_truth.shape[-1]
        if self.image_size is not None and self.image_size != dataset_image_size:
            ground_truth = F.interpolate(
                ground_truth,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            for frame_physics in physics.physics_list:
                subs = frame_physics.physics_list if hasattr(frame_physics, "physics_list") else [frame_physics]
                for sub in subs:
                    if hasattr(sub, "imsize"):
                        sub.imsize = None
            with torch.no_grad():
                measurements = physics(ground_truth)
        self.problem = InvProb(
            ground_truth=ground_truth,
            measurements=measurements,
            physics=physics,
            ground_truth_shape=ground_truth.shape,
            num_operators=num_operators,
            min_pixel=min_pixel,
            max_pixel=max_pixel,
        )
        self.ctx = None
        self._algo = None
        self.world_size = setup_distributed_env()
        self.distributed_mode = self.world_size > 1
        self.name = build_solver_name(
            self.name_prefix,
            self.slurm_nodes,
            self.slurm_ntasks_per_node,
            self.torchrun_nproc_per_node,
            self.distributed_mode,
        )

    def run(self, cb):
        if self.distributed_mode:
            with DistributedContext(seed=42, cleanup=True, deterministic=True) as ctx:
                self.ctx = ctx
                self._run_with_context(cb, ctx)
        else:
            self._run_with_context(cb, ctx=None)

    def _run_with_context(self, cb, ctx):
        device = get_device_from_context(ctx)
        profiler = create_profiler(
            self.profiler_mode, device, self.name,
            warmup=self.profiler_warmup, active=self.profiler_active,
            trace_dir=self.profiler_trace_dir,
            per_step=self.profiler_per_step,
            repeat=self.profiler_repeat,
            save_file=self.profiler_save_file,
        )
        with profiler:
            self._algo = UnrolledPnPSolver(
                problem=self.problem,
                device=device,
                profiler=profiler,
                ctx=ctx,
                distributed_mode=self.distributed_mode,
                denoiser=self.denoiser,
                n_iter=self.n_iter,
                init_stepsize=self.init_stepsize,
                denoiser_sigma=self.denoiser_sigma,
                grad_clip=self.grad_clip,
                distribute_model=self.distribute_model,
                patch_size=self.patch_size,
                overlap=self.overlap,
                max_batch_size=self.max_batch_size,
                checkpoint_batches=self.checkpoint_batches,
            )
            self._algo.run(cb)
        profiler.finalize(ctx)

    def get_result(self):
        if self._algo is None:
            return {"reconstruction": None}
        result = dict(name=self.name, ground_truth=self.problem.ground_truth)
        result.update(self._algo.get_result())
        return result

    def get_next(self, stop_val):
        return stop_val + 1
