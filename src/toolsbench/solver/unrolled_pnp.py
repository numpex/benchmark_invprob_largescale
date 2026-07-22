"""Standalone unrolled PnP training solver.

Mirrors :mod:`toolsbench.solver.pnp` (inference) but adds a supervised
backward pass.  Each benchopt iteration corresponds to **one training step**
(forward + backward + optimizer update) on a single in-memory batch, exactly
like the inference solver runs one PnP iteration per callback.

The actual training step reuses deepinv's :class:`deepinv.training.Trainer`
machinery through :class:`BenchTrainer`, which only overrides
:meth:`compute_loss` to time the forward / backward / optimizer regions via the
shared :class:`toolsbench.profiler.BenchProfiler` API.
"""

import torch
from deepinv.distributed import DistributedContext, distribute
from deepinv.optim import PGD
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.training import Trainer
from toolsbench.utils import create_drunet_denoiser
from toolsbench.utils.solver_utils import (
    distributed_callback_iter,
    measurement_to_device,
    sync_and_barrier,
)


class BenchTrainer(Trainer):
    """deepinv Trainer instrumented to time forward / backward / optimizer.

    ``profiler`` is attached as a plain attribute (the parent is a dataclass).
    Call :meth:`run_step` once per benchopt iteration.
    """

    def compute_loss(self, physics, x, y, train=True, epoch=None, step=False):
        """Single training step with profiler timing on each region.

        Reimplements the upstream forward/backward/optimizer flow, dropping the
        eval-loss branch (training only) and wrapping each region in
        ``profiler.track_step`` so per-step timings land in the CSV/benchopt
        metrics, just like the inference solver's gradient/denoise steps.
        """
        logs = {}

        if train and step:
            self.optimizer.zero_grad(set_to_none=True)

        with self.profiler.track_step("forward"):
            x_net = self.model_inference(y=y, physics=physics, x=x, train=True)
            loss_total = 0
            for k, loss_fn in enumerate(self.losses):
                loss = loss_fn(
                    x=x,
                    x_net=x_net,
                    y=y,
                    physics=physics,
                    model=self.model,
                    epoch=epoch,
                )
                loss_total += loss.mean()
                self.logs_losses_train[k].update(loss.detach().cpu().numpy())
            self.logs_total_loss_train.update(loss_total.item())
            logs["TotalLoss"] = self.logs_total_loss_train.avg

        if train:
            with self.profiler.track_step("backward"):
                loss_total.backward()
            if step:
                # Optimizer step is negligible vs. fwd/bwd; left untimed but
                # still included in the iteration's total_time_sec.
                self.optimizer.step()

        return loss_total, x_net, logs

    def run_step(self, x, y, physics, epoch: int):
        """Run one supervised training step and return the detached output."""
        self.model.train()
        _, x_net, _ = self.compute_loss(
            physics, x, y, train=True, epoch=epoch, step=True
        )
        return x_net.detach()


class UnrolledPnPSolver:
    """Unrolled PGD + PnP prior trained one step per benchopt iteration.

    Standalone algorithm class — no benchopt dependency.  Instantiate with all
    config, call ``run(cb)``, then ``get_result()``.
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
        n_iter=4,
        init_stepsize=0.8,
        denoiser_sigma=0.05,
        learning_rate=1e-5,
        model_learning_rate=1e-5,
        train_algo_params=True,
        lambda_relaxation=False,
        grad_clip=1.0,
        distribute_model=False,
        patch_size=128,
        overlap=32,
        max_batch_size=1,
        checkpoint_batches="auto",
        deterministic=True,
        image_size=None,
    ):
        if image_size is not None:
            problem = problem.resized(image_size, device=device)
        self.problem = problem
        self.device = device
        self.profiler = profiler
        self.ctx = ctx
        self.distributed_mode = distributed_mode
        self.denoiser = denoiser
        self.n_iter = n_iter
        self.init_stepsize = init_stepsize
        self.denoiser_sigma = denoiser_sigma
        self.learning_rate = learning_rate
        self.model_learning_rate = model_learning_rate
        self.train_algo_params = train_algo_params
        self.lambda_relaxation = lambda_relaxation
        self.grad_clip = grad_clip
        self.distribute_model = distribute_model
        self.patch_size = patch_size
        self.overlap = overlap
        self.max_batch_size = max_batch_size
        self.checkpoint_batches = checkpoint_batches
        self.deterministic = deterministic
        self.reconstruction = None

    def run(self, cb):
        if self.distribute_model and self.ctx is None:
            self.ctx = DistributedContext(deterministic=self.deterministic)
        x = self.problem.ground_truth.to(self.device)
        y = self._measurement_to_device(self.problem.measurements)
        physics = self._setup_physics()

        self.model, denoiser_params = self._setup_components()
        optimizer = self._setup_optimizer(denoiser_params)
        print("Components set up.")

        trainer = BenchTrainer(
            model=self.model,
            physics=physics,
            optimizer=optimizer,
            train_dataloader=self.problem.to_dataloader(),
            epochs=1,
            # losses: default deepinv SupLoss (supervised MSE).
            metrics=None,  # benchopt objective computes the metric
            grad_clip=self.grad_clip,
            device=self.device,
            save_path=None,
            verbose=False,
            show_progress_bar=False,
            compute_train_metrics=False,
            non_blocking_transfers=False,
        )
        trainer.profiler = self.profiler
        trainer.setup_train(train=True)

        # Initial (untrained) reconstruction so benchopt's first callback
        # evaluation at step 0 has a valid result, mirroring the inference solver.
        self.reconstruction = trainer.model_inference(
            y=y, physics=physics, x=x, train=False
        ).detach()

        print("Starting unrolled PnP training (one step per iteration).")
        epoch = 0
        for _ in distributed_callback_iter(
            cb, self.distributed_mode, self.device, self.ctx
        ):
            self.reconstruction = trainer.run_step(x, y, physics, epoch)
            self.profiler.end_iteration(self.ctx)
            epoch += 1

        sync_and_barrier(self.device, self.ctx)

    def _measurement_to_device(self, measurement):
        return measurement_to_device(measurement, self.device)

    def _setup_physics(self):
        physics = self.problem.physics
        if self.ctx is not None:
            return distribute(
                physics,
                self.ctx,
                num_operators=self.problem.num_operators,
                type_object="linear_physics",
                reduction="mean",
            )
        if hasattr(physics, "to"):
            physics = physics.to(self.device)
        return physics

    def _setup_components(self):
        if self.denoiser == "drunet":
            denoiser = create_drunet_denoiser(
                self.problem.ground_truth_shape, self.device, torch.float32
            )
        else:
            raise ValueError(f"Unknown denoiser: {self.denoiser}")

        # Collect raw denoiser params before any wrapping (distribute / compile).
        denoiser_params = list(denoiser.parameters())

        data_fidelity = L2()
        prior = PnP(denoiser=denoiser)

        if self.train_algo_params:
            trainable_params = ["stepsize", "g_param"]
            if self.lambda_relaxation:
                trainable_params.append("beta")
        else:
            trainable_params = []

        model = PGD(
            stepsize=[float(self.init_stepsize)] * self.n_iter,
            sigma_denoiser=self.denoiser_sigma,
            beta=[1.0] * self.n_iter,
            trainable_params=trainable_params,
            data_fidelity=data_fidelity,
            max_iter=self.n_iter,
            prior=prior,
            unfold=True,
        )

        # Save params_algo before distribute() wraps the model — the wrapper
        # does not forward that attribute.
        self._pgd_params_algo = model.params_algo

        if self.distribute_model and self.ctx is not None:
            model = distribute(
                model,
                self.ctx,
                patch_size=self.patch_size,
                overlap=self.overlap,
                max_batch_size=self.max_batch_size,
                checkpoint_batches=self.checkpoint_batches,
            )

        return model, denoiser_params

    def _setup_optimizer(self, denoiser_params):
        if self.train_algo_params:
            algo_params = (
                list(self._pgd_params_algo["stepsize"])
                + list(self._pgd_params_algo["g_param"])
                + (
                    list(self._pgd_params_algo["beta"])
                    if self.lambda_relaxation
                    else []
                )
            )
            return torch.optim.Adam(
                [
                    {"params": algo_params, "lr": float(self.learning_rate)},
                    {"params": denoiser_params, "lr": float(self.model_learning_rate)},
                ]
            )
        return torch.optim.Adam(
            [{"params": denoiser_params, "lr": float(self.model_learning_rate)}]
        )

    def get_result(self):
        result = dict(
            reconstruction=self.reconstruction, ground_truth=self.problem.ground_truth
        )
        if self.profiler is not None:
            result.update(self.profiler.get_current_metrics())
        return result
