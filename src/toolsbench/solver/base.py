from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from benchopt import BaseSolver
from deepinv.distributed import DistributedContext
from deepinv.physics import Physics, StackedPhysics

from toolsbench.profiler import create_profiler
from toolsbench.utils import create_drunet_denoiser, setup_distributed_env
from toolsbench.utils.solver_utils import build_solver_name, get_device_from_context


@dataclass
class SolverObjective:
    """Data handed to any inference solver via set_objective."""

    measurement: torch.Tensor
    physics: Physics | StackedPhysics | Callable
    ground_truth_shape: torch.Size
    num_operators: int
    min_pixel: float = 0.0
    max_pixel: float = 1.0
    weights: Optional[torch.Tensor] = None


class BaseInvprobSolver(BaseSolver):
    """Shared scaffold for inverse-problem solvers.

    Subclasses must implement :meth:`_run_algorithm`.
    """

    sampling_strategy = "callback"

    parameters = {
        "denoiser": ["drunet"],
        "patch_size": [128],
        "overlap": [32],
        "max_batch_size": [0],
        "slurm_nodes": [1],
        "slurm_ntasks_per_node": [1],
        "slurm_gres": ["gpu:1"],
        "torchrun_nproc_per_node": [1],
        "name_prefix": ["pnp"],
        "profiler_mode": ["custom"],
        "profiler_warmup": [0],
        "profiler_active": [0],
    }

    def set_objective(
        self,
        measurement,
        physics,
        ground_truth_shape,
        num_operators,
        min_pixel=0.0,
        max_pixel=1.0,
        weights=None,
    ):
        self.problem = SolverObjective(
            measurement=measurement,
            physics=physics,
            ground_truth_shape=ground_truth_shape,
            num_operators=num_operators,
            min_pixel=min_pixel,
            max_pixel=max_pixel,
            weights=weights,
        )
        self.ctx = None
        self.profiler = None
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
            with DistributedContext(seed=42, cleanup=True) as ctx:
                self.ctx = ctx
                self._run_with_context(cb, ctx)
        else:
            self._run_with_context(cb, ctx=None)

    def _run_with_context(self, cb, ctx):
        self.device = get_device_from_context(ctx)
        self.profiler = create_profiler(
            self.profiler_mode, self.device, self.name,
            warmup=self.profiler_warmup, active=self.profiler_active,
        )
        with self.profiler:
            self._run_algorithm(cb, ctx)
        self.profiler.finalize(ctx)

    @abstractmethod
    def _run_algorithm(self, cb, ctx):
        """Implement the solver's core algorithm (setup, iteration loop, cleanup)."""
        raise NotImplementedError

    def _create_denoiser(self, device):
        if self.denoiser == "drunet":
            return create_drunet_denoiser(
                self.problem.ground_truth_shape, device, torch.float32
            )
        raise ValueError(f"Unknown denoiser: {self.denoiser}")

    def get_next(self, stop_val):
        return stop_val + 1
