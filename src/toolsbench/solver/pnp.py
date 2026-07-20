
import torch
from deepinv.distributed import distribute
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.physics import Physics, stack
from toolsbench.utils import create_drunet_denoiser
from toolsbench.utils.solver_utils import (
    distributed_callback_iter,
    initialize_reconstruction,
    measurement_to_device,
    sync_and_barrier,
)


def compute_step_size_from_operator(operator, ground_truth: torch.Tensor) -> float:
    """Step size = 1 / Lipschitz constant of the forward operator."""
    with torch.no_grad():
        x_example = torch.zeros_like(ground_truth, device=ground_truth.device, dtype=ground_truth.dtype)
        lipschitz_constant = operator.compute_norm(x_example, local_only=False)
        return 1.0 / lipschitz_constant if lipschitz_constant > 0 else 1.0



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
        compile=None,
        image_size=None,
    ):
        if image_size is not None:
            problem = problem.resized(image_size, device=device)
        self.shape = tuple(problem.ground_truth_shape)
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
        self.compile = compile
        self.reconstruction = None

    def run(self, cb):
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

        measurement = measurement_to_device(self.problem.measurements, self.device)

        prior, data_fidelity = self._setup_components()
        print("Components set up.")

        step_size = self._compute_step_size(physics)
        print("Step size computed:", step_size)

        if self.compile == "pre":
            local_ops = (
                list(physics.local_physics) if hasattr(physics, "local_physics")
                else list(physics.physics_list) if hasattr(physics, "physics_list")
                else [physics]
            )
            for op in local_ops:
                if hasattr(op, "xray_transform"):
                    # ASTRA-backed ops rebuild a fresh type(self) on every
                    # call, so torch.compile can never cache a graph here.
                    continue
                if hasattr(op, "A"):
                    op.A = torch.compile(op.A)
                if hasattr(op, "A_adjoint"):
                    op.A_adjoint = torch.compile(op.A_adjoint)

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
                self.shape, self.device, torch.float32
            )
        else:
            raise ValueError(f"Unknown denoiser: {self.denoiser}")

        if self.compile == "pre":
            denoiser = torch.compile(denoiser)

        if self.ctx is not None and self.distribute_denoiser:
            denoiser = distribute(
                denoiser,
                self.ctx,
                patch_size=self.patch_size,
                overlap=self.overlap,
                tiling_dims=(
                    (-3, -2, -1) if len(self.shape) == 5 else (-2, -1)
                ),
                max_batch_size=self.max_batch_size,
                type_object="denoiser",
            )

        if self.compile == "post":
            denoiser = torch.compile(denoiser)

        prior = PnP(denoiser=denoiser)
        data_fidelity = L2()

        if self.ctx is not None and self.distribute_physics:
            data_fidelity = distribute(data_fidelity, self.ctx)

        if self.compile == "post":
            data_fidelity.grad = torch.compile(data_fidelity.grad)

        return prior, data_fidelity

    def _compute_step_size(self, physics):
        if isinstance(self.step_size, float):
            return self.step_size
        x_example = torch.zeros(self.shape, device=self.device)
        return compute_step_size_from_operator(physics, x_example) * self.step_size_scale

    def _initialize_reconstruction(self, physics, measurement):
        with torch.no_grad():
            return initialize_reconstruction(
                signal_shape=self.shape,
                operator=physics,
                measurements=measurement,
                device=self.device,
                method=self.init_method,
                clip_range=(self.problem.min_pixel, self.problem.max_pixel),
                weights=self.problem.invprob_kwargs.get("weights") if self.problem.invprob_kwargs else None,
            )

    def _run_iterations(self, prior, data_fidelity, physics, measurements, step_size, cb):
        sig_min, sig_max = self.problem.min_pixel, self.problem.max_pixel

        if self.compile == "fused":
            if self.norm_strategy != "clip":
                raise ValueError("compile='fused' only supports norm_strategy='clip'")

            def pnp_step(x):
                grad = data_fidelity.grad(x, measurements, physics)
                x = x - step_size * grad
                if self.denoiser_lambda_relaxation is None:
                    x = prior.prox(x, sigma_denoiser=self.denoiser_sigma)
                else:
                    denoised = prior.prox(x, sigma_denoiser=self.denoiser_sigma)
                    lamda = self.denoiser_lambda_relaxation
                    alpha = (step_size * lamda) / (1 + step_size * lamda)
                    x = (1 - alpha) * x + alpha * denoised
                return torch.clamp(x, sig_min, sig_max)

            pnp_step = torch.compile(pnp_step)

            with torch.no_grad():
                for _ in distributed_callback_iter(cb, self.distributed_mode, self.device, self.ctx):
                    with self.profiler.track_step("pnp"):
                        self.reconstruction = pnp_step(self.reconstruction)
                    self.profiler.end_iteration(self.ctx)
            return

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

                self.profiler.end_iteration(self.ctx)

    def get_result(self):
        result = dict(reconstruction=self.reconstruction)
        # A resized problem carries its own ground truth; the objective scores
        # against that rather than the dataset's, whose shape no longer matches.
        if self.problem.ground_truth is not None:
            result["ground_truth"] = self.problem.ground_truth
        if self.profiler is not None:
            result.update(self.profiler.get_current_metrics())
        return result
