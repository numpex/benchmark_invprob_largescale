"""Source-aware unrolled radio training with timed, solver-owned loading."""

from __future__ import annotations

import gc
import math

import torch
from deepinv.distributed import distribute

from .builder import create_alternating_source_model
from .model import (
    DynamicRangePnP,
    create_drunet,
    set_radio_model_params,
)
from .iteration import make_joint_initialization
from toolsbench.utils.radio_interferometry import (
    RadioDataConfig,
    build_radio_dataloaders,
)
from toolsbench.utils.radio_interferometry.sources import SourceCatalog


class RadioTrainingSolver:
    """One radio forward/backward/optimizer update per Benchopt callback."""

    def __init__(self, data_config, ctx, profiler, **config):
        self.data_config = RadioDataConfig(**data_config)
        self.ctx = ctx
        self.profiler = profiler
        self.config = config
        self.device = ctx.device
        self.reconstruction = torch.zeros(1, 1, 1, 1, device=self.device)
        self.ground_truth = torch.ones_like(self.reconstruction)
        self.min_pixel = 0.0
        self.max_pixel = 1.0
        self.epoch = 0
        self.iterator = None

    def setup(self):
        cfg = self.data_config
        sharding_override = self.config["physics_sharding"]
        if sharding_override is not None:
            cfg.physics_sharding = bool(sharding_override)
        if cfg.physics_sharding and not cfg.w_stacking:
            raise ValueError("physics_sharding requires w_stacking=true.")
        if cfg.physics_sharding and int(self.ctx.inner_world_size) < 2:
            raise ValueError(
                "physics_sharding requires inner_world_size >= 2; use pure DDP "
                "with physics_sharding=false."
            )
        cfg.shard_rank = int(self.ctx.inner_rank) if cfg.physics_sharding else 0
        cfg.shard_world_size = (
            int(self.ctx.inner_world_size) if cfg.physics_sharding else 1
        )
        bundle = build_radio_dataloaders(cfg, ctx=self.ctx)
        self.loader = bundle.loader
        self.sampler = bundle.sampler
        self.model = self._build_model(bundle.ground_truth_shape)
        if int(self.ctx.dp_world_size) > 1:
            self.model = self.ctx.distributed_data_parallel(self.model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(self.config["model_learning_rate"])
        )

    def _build_model(self, shape):
        def spatial_pair(value):
            if isinstance(value, (list, tuple)):
                return tuple(int(v) for v in value)
            return (int(value), int(value))

        denoiser = create_drunet(
            tuple(int(v) for v in shape),
            self.device,
            pretrained=self.config["drunet_pretrained"],
        ).train()
        if self.config["distribute_denoiser"]:
            denoiser = distribute(
                denoiser,
                self.ctx,
                type_object="denoiser",
                patch_size=spatial_pair(self.config["patch_size"]),
                overlap=spatial_pair(self.config["overlap"]),
                tiling_dims=(-2, -1),
                max_batch_size=self.config["max_batch_size"],
                checkpoint_batches=self.config["checkpoint_batches"],
            )
        elif int(self.ctx.inner_world_size) > 1:
            raise ValueError("inner_world_size > 1 requires distribute_denoiser=true.")

        prior = DynamicRangePnP(
            denoiser,
            norm_strategy=self.config["norm_strategy"],
            denoiser_lambda_relaxation=self.config["denoiser_lambda_relaxation"],
        )
        fidelity = None
        if self.data_config.w_stacking:
            from deepinv.optim.data_fidelity import L2

            fidelity = distribute(L2(), self.ctx, type_object="data_fidelity")
        return create_alternating_source_model(
            prior=prior,
            n_iter=int(self.config["n_iter"]),
            step_size=(
                float(self.config["step_size"])
                if self.config["step_size"] is not None
                else 1.0
            ),
            denoiser_sigma=float(self.config["denoiser_sigma"]),
            source_mode=self.config["source_refinement"],
            source_max_shift=float(self.config["source_max_shift"]),
            source_lambda_amp=float(self.config["source_lambda_amp"]),
            source_lambda_pos=float(self.config["source_lambda_pos"]),
            source_amplitude_step=float(self.config["source_amplitude_step"]),
            source_position_step=float(self.config["source_position_step"]),
            source_recompute_residual=bool(self.config["source_recompute_residual"]),
            train_source_update=bool(self.config["train_source_update"]),
            data_fidelity=fidelity,
            train_denoiser_sigma=bool(self.config["train_denoiser_sigma"]),
            train_stepsize=bool(self.config["train_stepsize"]),
        ).to(self.device)

    @property
    def base_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _next_batch(self):
        while True:
            if self.iterator is None:
                if hasattr(self.sampler, "set_epoch"):
                    self.sampler.set_epoch(self.epoch)
                self.iterator = iter(self.loader)
            try:
                return next(self.iterator)
            except StopIteration:
                self.epoch += 1
                self.iterator = None

    def _step_size(self, physics, x, cached):
        if self.config["step_size"] is not None:
            return float(self.config["step_size"])
        if cached is not None and float(cached) > 0:
            return float(self.config["step_size_scale"]) / float(cached)
        with torch.no_grad():
            value = physics.compute_norm(
                torch.zeros_like(x),
                max_iter=int(self.config["norm_max_iter"]),
                tol=float(self.config["norm_tol"]),
                verbose=False,
                local_only=False,
            )
        return float(self.config["step_size_scale"]) / max(
            float(value.detach().cpu()), 1e-12
        )

    def _loss(self, prediction, target, clip_range):
        kind = self.config["loss"]
        if kind == "mse":
            return torch.nn.functional.mse_loss(prediction, target)
        if kind == "asinh_mse":
            beta = target.square().mean().sqrt().clamp(min=1e-12).item()
            return torch.nn.functional.mse_loss(
                torch.asinh(prediction / beta), torch.asinh(target / beta)
            )
        if kind == "log_mse":
            lo, hi = clip_range
            scale = max(float(hi) - float(lo), 1e-12)

            def stretch(value):
                value = ((value - lo) / scale).clamp(0, 1)
                return torch.log1p(1000.0 * value) / math.log1p(1000.0)

            return torch.nn.functional.mse_loss(stretch(prediction), stretch(target))
        raise ValueError("loss must be mse, asinh_mse, or log_mse.")

    def run_step(self):
        with self.profiler.track_step("data_loading"):
            batch = self._next_batch()
            batch = batch.to(self.device, non_blocking=self.device.type == "cuda")
            x = batch.x
            clip_range = (batch.min_pixel, batch.max_pixel)

        with self.profiler.track_step("physics_setup"):
            if batch.physics_from_shard:
                physics = distribute(
                    batch.physics,
                    self.ctx,
                    type_object="linear_physics",
                    from_shard=True,
                    num_operators=batch.num_physics_operators,
                    global_indices=batch.physics_global_indices,
                )
            else:
                physics = batch.physics
            step_size = self._step_size(physics, x, batch.lipschitz)
            source_catalog = batch.source_catalog
            if source_catalog is None:
                empty = x.new_empty(0)
                source_catalog = SourceCatalog(empty, empty, empty)
            model_init = make_joint_initialization(
                batch.x_init,
                source_catalog,
                measurement_size=batch.global_measurement_count,
            )
            diffuse_init = model_init["est"][0]
            model_range = (
                float(diffuse_init.detach().amin()),
                float(diffuse_init.detach().amax()),
            )
            set_radio_model_params(
                self.base_model,
                step_size,
                model_range,
                train_stepsize=bool(self.config["train_stepsize"]),
            )

        self.optimizer.zero_grad(set_to_none=True)
        self.profiler.snapshot_memory("before_forward")
        try:
            with self.profiler.track_step("forward"):
                prediction = self.model(batch.measurements, physics, init=model_init)
                state = getattr(
                    getattr(self.base_model, "get_output", None), "last_state", None
                )
                diffuse_prediction = (
                    state["est"][0]
                    if state is not None
                    else prediction - model_init["source_image"]
                )
                diffuse_target = (
                    batch.x_diffuse.unsqueeze(0)
                    if batch.x_diffuse.ndim == 3
                    else batch.x_diffuse
                )
                image_loss = self._loss(prediction, x, clip_range)
                diffuse_loss = self._loss(
                    diffuse_prediction, diffuse_target, clip_range
                )
                alpha = float(self.config["loss_alpha"])
                loss = alpha * image_loss + (1.0 - alpha) * diffuse_loss
        except Exception:
            self.profiler.snapshot_memory("forward_failure")
            raise

        self.profiler.snapshot_memory("before_backward")
        try:
            with self.profiler.track_step("backward"):
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                loss.backward()
        except Exception:
            self.profiler.snapshot_memory("backward_failure")
            raise
        with self.profiler.track_step("optimizer"):
            clip = float(self.config["clip_grad_norm"])
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

        self.reconstruction = prediction.detach()
        self.ground_truth = x.detach()
        self.min_pixel, self.max_pixel = clip_range
        del loss, image_loss, diffuse_loss, prediction, x, batch
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            gc.collect()
            torch.cuda.empty_cache()

    def get_result(self):
        result = {
            "reconstruction": self.reconstruction,
            "ground_truth": self.ground_truth,
            "min_pixel": self.min_pixel,
            "max_pixel": self.max_pixel,
        }
        result.update(self.profiler.get_current_metrics())
        return result
