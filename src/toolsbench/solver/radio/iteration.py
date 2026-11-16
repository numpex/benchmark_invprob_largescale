"""Joint diffuse/compact-source iterations for radio reconstruction."""

from __future__ import annotations

import torch
from deepinv.optim.optim_iterators import OptimIterator

from toolsbench.utils.radio_interferometry.sources import (
    SourceCatalog,
    SourceParams,
    render_sources,
)


def _source_params(
    source_init: SourceCatalog,
    log_amplitude: torch.Tensor,
    dx_raw: torch.Tensor,
    dy_raw: torch.Tensor,
    max_shift: float,
) -> SourceParams:
    return SourceParams(
        amplitude=source_init.amplitude * torch.exp(log_amplitude),
        x=source_init.x + float(max_shift) * torch.tanh(dx_raw),
        y=source_init.y + float(max_shift) * torch.tanh(dy_raw),
    )


def source_parameter_gradients(
    image_gradient: torch.Tensor,
    source_params: SourceParams,
    dx_raw: torch.Tensor,
    dy_raw: torch.Tensor,
    max_shift: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the bilinear renderer VJP using four pixels per source.

    ``image_gradient`` is the gradient with respect to the summed sky image,
    normally ``A_adjoint(A(d + S(theta)) - y)``. The returned tensors are
    gradients with respect to log-amplitude and the raw bounded x/y offsets.
    """
    if image_gradient.ndim != 4 or image_gradient.shape[:2] != (1, 1):
        raise ValueError(
            "Source refinement expects a single image shaped (1,1,H,W), got "
            f"{tuple(image_gradient.shape)}."
        )
    amplitude, x, y = (
        source_params.amplitude,
        source_params.x,
        source_params.y,
    )
    if amplitude.numel() == 0:
        return amplitude, dx_raw, dy_raw

    height, width = image_gradient.shape[-2:]
    x0_float = torch.floor(x)
    y0_float = torch.floor(y)
    fx = x - x0_float
    fy = y - y0_float
    x0 = x0_float.to(torch.long)
    y0 = y0_float.to(torch.long)
    flat_gradient = image_gradient.reshape(-1)

    def gather(x_index: torch.Tensor, y_index: torch.Tensor) -> torch.Tensor:
        valid = (x_index >= 0) & (x_index < width) & (y_index >= 0) & (y_index < height)
        safe_x = x_index.clamp(0, width - 1)
        safe_y = y_index.clamp(0, height - 1)
        values = flat_gradient[safe_y * width + safe_x]
        return values * valid.to(values.dtype)

    g00 = gather(x0, y0)
    g10 = gather(x0 + 1, y0)
    g01 = gather(x0, y0 + 1)
    g11 = gather(x0 + 1, y0 + 1)

    amplitude_gradient = (
        (1.0 - fx) * (1.0 - fy) * g00
        + fx * (1.0 - fy) * g10
        + (1.0 - fx) * fy * g01
        + fx * fy * g11
    )
    x_gradient = amplitude * ((1.0 - fy) * (g10 - g00) + fy * (g11 - g01))
    y_gradient = amplitude * ((1.0 - fx) * (g01 - g00) + fx * (g11 - g10))

    log_amplitude_gradient = amplitude * amplitude_gradient
    dx_draw = float(max_shift) * (1.0 - torch.tanh(dx_raw).square())
    dy_draw = float(max_shift) * (1.0 - torch.tanh(dy_raw).square())
    return (
        log_amplitude_gradient,
        x_gradient * dx_draw,
        y_gradient * dy_draw,
    )


class FirstOrderSourceUpdate(torch.nn.Module):
    """One first-order source update through the summed image model."""

    def __init__(
        self,
        mode: str = "amplitude_position",
        max_shift: float = 0.75,
        lambda_amp: float = 1e-3,
        lambda_pos: float = 1e-3,
    ) -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode == "position":
            mode = "amplitude_position"
        if mode not in {"fixed", "amplitude", "amplitude_position"}:
            raise ValueError(
                "source mode must be 'fixed', 'amplitude', or " "'amplitude_position'."
            )
        if max_shift <= 0:
            raise ValueError("source max_shift must be positive.")
        self.mode = mode
        self.max_shift = float(max_shift)
        self.lambda_amp = float(lambda_amp)
        self.lambda_pos = float(lambda_pos)

    def forward(
        self,
        image_gradient: torch.Tensor,
        source_init: SourceCatalog,
        log_amplitude: torch.Tensor,
        dx_raw: torch.Tensor,
        dy_raw: torch.Tensor,
        amplitude_step: torch.Tensor,
        position_step: torch.Tensor,
        measurement_size: int,
        image_shape: tuple[int, ...],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        SourceParams,
        torch.Tensor,
    ]:
        params = _source_params(
            source_init, log_amplitude, dx_raw, dy_raw, self.max_shift
        )
        if len(source_init) == 0 or self.mode == "fixed":
            return (
                log_amplitude,
                dx_raw,
                dy_raw,
                params,
                render_sources(params, image_shape),
            )

        grad_amplitude, grad_x, grad_y = source_parameter_gradients(
            image_gradient, params, dx_raw, dy_raw, self.max_shift
        )
        data_scale = 1.0 / max(1, int(measurement_size))
        grad_amplitude = data_scale * grad_amplitude
        grad_amplitude = grad_amplitude + self.lambda_amp * log_amplitude
        log_amplitude = log_amplitude - amplitude_step * grad_amplitude

        if self.mode == "amplitude_position":
            grad_x = data_scale * grad_x + self.lambda_pos * dx_raw
            grad_y = data_scale * grad_y + self.lambda_pos * dy_raw
            dx_raw = dx_raw - position_step * grad_x
            dy_raw = dy_raw - position_step * grad_y

        params = _source_params(
            source_init, log_amplitude, dx_raw, dy_raw, self.max_shift
        )
        return (
            log_amplitude,
            dx_raw,
            dy_raw,
            params,
            render_sources(params, image_shape),
        )


class AlternatingSourcePGDIteration(OptimIterator):
    """One first-order source update followed by one PGD/DRUNet update."""

    def __init__(
        self,
        source_mode: str = "amplitude_position",
        source_max_shift: float = 0.75,
        source_lambda_amp: float = 1e-3,
        source_lambda_pos: float = 1e-3,
        source_recompute_residual: bool = False,
    ) -> None:
        super().__init__(has_cost=False)
        self.source_update = FirstOrderSourceUpdate(
            mode=source_mode,
            max_shift=source_max_shift,
            lambda_amp=source_lambda_amp,
            lambda_pos=source_lambda_pos,
        )
        self.source_recompute_residual = bool(source_recompute_residual)

    @staticmethod
    def _positive_scalar(value, reference: torch.Tensor) -> torch.Tensor:
        raw = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        return torch.nn.functional.softplus(raw)

    def forward(
        self,
        X,
        cur_data_fidelity,
        cur_prior,
        cur_params,
        measurement,
        physics,
        *args,
        **kwargs,
    ):
        diffuse, log_amplitude, dx_raw, dy_raw = X["est"]
        source_init = X["source_init"]
        source_image = X["source_image"]
        measurement_size = X.get("measurement_size")
        if measurement_size is None:
            # Preserve the ordinary, non-sharded API for direct model calls.
            measurement_size = measurement.flatten().numel()

        sky = diffuse + source_image
        if getattr(physics, "from_shard", False):
            # DistributedDataFidelity evaluates A(x), residuals, and VJPs with
            # gather=False internally. gather=True reduces only the resulting
            # image-shaped gradients so all unrolled replicas remain aligned.
            image_gradient = cur_data_fidelity.grad(
                sky, measurement, physics, gather=True
            )
        else:
            prediction = physics.A(sky)
            residual = prediction - measurement
            image_gradient = physics.A_adjoint(residual)
        amplitude_step = self._positive_scalar(
            cur_params["source_log_amplitude_step"], diffuse
        )
        position_step = self._positive_scalar(
            cur_params["source_log_position_step"], diffuse
        )
        (
            log_amplitude,
            dx_raw,
            dy_raw,
            source_params,
            source_image_next,
        ) = self.source_update(
            image_gradient=image_gradient,
            source_init=source_init,
            log_amplitude=log_amplitude,
            dx_raw=dx_raw,
            dy_raw=dy_raw,
            amplitude_step=amplitude_step,
            position_step=position_step,
            # Keep source-step scaling independent of the number of local
            # planes owned by this rank.
            measurement_size=measurement_size,
            image_shape=diffuse.shape,
        )

        # By default both blocks share one data-fidelity gradient. The source
        # correction is visible to the diffuse block at the next unrolled stage,
        # which keeps active refinement at one A/A* pair per stage. The optional
        # strict mode recomputes the residual immediately after the source step.
        if (
            len(source_init) == 0
            or self.source_update.mode == "fixed"
            or not self.source_recompute_residual
        ):
            data_gradient = image_gradient
        else:
            updated_sky = diffuse + source_image_next
            if getattr(physics, "from_shard", False):
                data_gradient = cur_data_fidelity.grad(
                    updated_sky, measurement, physics, gather=True
                )
            else:
                residual = residual + physics.A(source_image_next - source_image)
                data_gradient = physics.A_adjoint(residual)

        z = diffuse - cur_params["stepsize"] * data_gradient

        diffuse_next = cur_prior.prox(
            z,
            cur_params["g_param"],
            gamma=cur_params["lambda"] * cur_params["stepsize"],
        )
        diffuse_next = self.relaxation_step(diffuse_next, diffuse, cur_params["beta"])
        return {
            "est": (diffuse_next, log_amplitude, dx_raw, dy_raw),
            "source_init": source_init,
            "source_params": source_params,
            "source_image": source_image_next,
            "measurement_size": measurement_size,
            "cost": None,
        }


def make_joint_initialization(
    x_init: torch.Tensor,
    source_init: SourceCatalog,
    measurement_size: int | None = None,
) -> dict:
    """Create the initial joint state ``(d, alpha, dx, dy)``."""
    source_image = render_sources(source_init, x_init.shape)
    zeros = torch.zeros_like(source_init.amplitude)
    if measurement_size is not None and int(measurement_size) <= 0:
        raise ValueError("measurement_size must be positive.")
    return {
        "est": (x_init - source_image, zeros, zeros.clone(), zeros.clone()),
        "source_init": source_init,
        "source_params": SourceParams(
            source_init.amplitude, source_init.x, source_init.y
        ),
        "source_image": source_image,
        "measurement_size": (
            None if measurement_size is None else int(measurement_size)
        ),
    }
