from dataclasses import dataclass
from typing import Callable, Optional

import torch
from deepinv.distributed import distribute
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.physics import Physics, StackedPhysics, stack
from deepinv.utils.tensorlist import TensorList

from toolsbench.utils import create_drunet_denoiser
from toolsbench.utils.solver_utils import (
    distributed_callback_iter,
    initialize_reconstruction,
    sync_and_barrier,
)


def compute_step_size_from_operator(operator, ground_truth: torch.Tensor) -> float:
    """Step size = 1 / Lipschitz constant of the forward operator."""
    with torch.no_grad():
        x_example = torch.zeros_like(ground_truth, device=ground_truth.device, dtype=ground_truth.dtype)
        lipschitz_constant = operator.compute_norm(x_example, local_only=False)
        return 1.0 / lipschitz_constant if lipschitz_constant > 0 else 1.0


@dataclass
class SolverObjective:
    """Data handed to the inference solver via set_objective."""

    measurement: torch.Tensor
    physics: Physics | StackedPhysics | Callable
    ground_truth_shape: torch.Size
    num_operators: int
    min_pixel: float = 0.0
    max_pixel: float = 1.0
    weights: Optional[torch.Tensor] = None



class PnPSolver:
    """Plug-and-Play algorithm: gradient step + denoiser prior.

    Standalone algorithm class — no benchopt dependency.
    Instantiate with all config, call run(cb), then get_result().
    """

    def __init__(
        self,
        problem,
        device,
        profiler,
        ctx,
        distributed_mode,
        *,
        denoiser="drunet",
        patch_size=128,
        overlap=32,
        max_batch_size=0,
        step_size=None,
        step_size_scale=0.99,
        denoiser_sigma=0.05,
        denoiser_lambda_relaxation=None,
        distribute_physics=False,
        distribute_denoiser=False,
        init_method="pseudo_inverse",
        norm_strategy="clip",
    ):
        self.problem = problem
        self.device = device
        self.profiler = profiler
        self.ctx = ctx
        self.distributed_mode = distributed_mode
        self.denoiser = denoiser
        self.patch_size = patch_size
        self.overlap = overlap
        self.max_batch_size = max_batch_size
        self.step_size = step_size
        self.step_size_scale = step_size_scale
        self.denoiser_sigma = denoiser_sigma
        self.denoiser_lambda_relaxation = denoiser_lambda_relaxation
        self.distribute_physics = distribute_physics
        self.distribute_denoiser = distribute_denoiser
        self.init_method = init_method
        self.norm_strategy = norm_strategy
        self.reconstruction = None

    def run(self, cb):
        measurement = self.problem.measurement
        if hasattr(measurement, "to"):
            measurement = measurement.to(self.device)
        elif isinstance(measurement, list):
            measurement = TensorList([m.to(self.device) for m in measurement])

        if self.ctx is not None and self.distribute_physics:
            physics = distribute(
                self.problem.physics,
                self.ctx,
                num_operators=self.problem.num_operators,
                type_object="linear_physics",
            )
        elif callable(self.problem.physics) and not isinstance(self.problem.physics, Physics):
            physics = stack(*[
                self.problem.physics(i, self.device, None)
                for i in range(self.problem.num_operators)
            ])
        else:
            physics = self.problem.physics
            if hasattr(physics, "to"):
                physics = physics.to(self.device)

        prior, data_fidelity = self._setup_components()
        print("Components set up.")

        step_size = self._compute_step_size(physics)
        print("Step size computed:", step_size)

        self.reconstruction = self._initialize_reconstruction(physics, measurement)
        print("Reconstruction initialized.")

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        print("Starting PnP iterations.")
        self._run_iterations(prior, data_fidelity, physics, measurement, step_size, cb)

        sync_and_barrier(self.device, self.ctx)

    def _setup_components(self):
        if self.denoiser == "drunet":
            denoiser = create_drunet_denoiser(
                self.problem.ground_truth_shape, self.device, torch.float32
            )
        else:
            raise ValueError(f"Unknown denoiser: {self.denoiser}")

        if self.ctx is not None and self.distribute_denoiser:
            denoiser = distribute(
                denoiser,
                self.ctx,
                patch_size=self.patch_size,
                overlap=self.overlap,
                tiling_dims=(
                    (-3, -2, -1) if len(self.problem.ground_truth_shape) == 5 else (-2, -1)
                ),
                max_batch_size=self.max_batch_size,
            )

        prior = PnP(denoiser=denoiser)
        data_fidelity = L2()

        if self.ctx is not None and self.distribute_physics:
            data_fidelity = distribute(data_fidelity, self.ctx)

        return prior, data_fidelity

    def _compute_step_size(self, physics):
        if isinstance(self.step_size, float):
            return self.step_size
        x_example = torch.zeros(self.problem.ground_truth_shape, device=self.device)
        return compute_step_size_from_operator(physics, x_example) * self.step_size_scale

    def _initialize_reconstruction(self, physics, measurement):
        with torch.no_grad():
            return initialize_reconstruction(
                signal_shape=self.problem.ground_truth_shape,
                operator=physics,
                measurements=measurement,
                device=self.device,
                method=self.init_method,
                clip_range=(self.problem.min_pixel, self.problem.max_pixel),
                weights=self.problem.weights,
            )

    def _run_iterations(self, prior, data_fidelity, physics, measurements, step_size, cb):
        sig_min, sig_max = self.problem.min_pixel, self.problem.max_pixel

        with torch.no_grad():
            for _ in distributed_callback_iter(cb, self.distributed_mode, self.device, self.ctx):
                with self.profiler.track_step("gradient"):
                    grad = data_fidelity.grad(self.reconstruction, measurements, physics)
                    self.reconstruction = self.reconstruction - step_size * grad

                with self.profiler.track_step("denoise"):
                    if self.norm_strategy == "dynamic":
                        scale = sig_max - sig_min
                        self.reconstruction = (self.reconstruction - sig_min) / scale

                        if self.denoiser_lambda_relaxation is None:
                            self.reconstruction = prior.prox(
                                self.reconstruction, sigma_denoiser=self.denoiser_sigma
                            )
                        else:
                            denoised = prior.prox(
                                self.reconstruction, sigma_denoiser=self.denoiser_sigma
                            )
                            lamda = self.denoiser_lambda_relaxation
                            alpha = (step_size * lamda) / (1 + step_size * lamda)
                            self.reconstruction = (
                                (1 - alpha) * self.reconstruction + alpha * denoised
                            )

                        self.reconstruction = self.reconstruction * scale + sig_min
                    else:
                        if self.denoiser_lambda_relaxation is None:
                            self.reconstruction = prior.prox(
                                self.reconstruction, sigma_denoiser=self.denoiser_sigma
                            )
                        else:
                            x_denoised = prior.prox(
                                self.reconstruction, sigma_denoiser=self.denoiser_sigma
                            )
                            lamda = self.denoiser_lambda_relaxation
                            alpha = (step_size * lamda) / (1 + step_size * lamda)
                            self.reconstruction = (
                                (1 - alpha) * self.reconstruction + alpha * x_denoised
                            )
                        self.reconstruction = torch.clamp(
                            self.reconstruction, sig_min, sig_max
                        )

                self.profiler.end_iteration()

    def get_result(self):
        result = dict(reconstruction=self.reconstruction)
        if self.profiler is not None:
            result.update(self.profiler.get_current_metrics())
        return result
