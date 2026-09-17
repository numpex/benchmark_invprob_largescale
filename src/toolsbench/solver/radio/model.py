"""Radio denoiser prior construction and parameter updates."""

from __future__ import annotations

import torch
from deepinv.optim.prior import PnP

_AUTOCAST_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def resolve_autocast_dtype(value) -> torch.dtype | None:
    if value is None or value is False:
        return None
    if isinstance(value, torch.dtype):
        return value
    key = str(value).lower()
    if key in {"none", "null", "false", "off"}:
        return None
    if key not in _AUTOCAST_DTYPES:
        raise ValueError("autocast_dtype must be None, 'bfloat16', or 'float16'.")
    return _AUTOCAST_DTYPES[key]


def compile_denoiser(denoiser, mode="pre"):
    if mode is None or str(mode).lower() in {"none", "off", "false"}:
        return denoiser
    normalized = str(mode).lower()
    if normalized != "pre":
        raise ValueError("compile must be either 'pre' or None.")
    return torch.compile(
        denoiser,
        fullgraph=True,
        mode="max-autotune",
    )


class DynamicRangePnP(PnP):
    """PnP prior with the radio benchmark dynamic normalization scheme."""

    def __init__(
        self,
        denoiser,
        norm_strategy: str = "dynamic",
        clip_range: tuple[float, float] = (0.0, 1.0),
        denoiser_lambda_relaxation: float | None = None,
        eps: float = 1e-12,
        autocast_dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__(denoiser=denoiser)
        self.norm_strategy = str(norm_strategy)
        self.clip_range = (float(clip_range[0]), float(clip_range[1]))
        self.denoiser_lambda_relaxation = denoiser_lambda_relaxation
        self.eps = float(eps)
        self.autocast_dtype = resolve_autocast_dtype(autocast_dtype)

    def set_clip_range(self, clip_range: tuple[float, float]) -> None:
        self.clip_range = (float(clip_range[0]), float(clip_range[1]))

    def _denoise(self, x, sigma_denoiser):
        if self.autocast_dtype is None:
            return self.denoiser(x, sigma_denoiser)
        if x.device.type not in {"cpu", "cuda"}:
            raise RuntimeError(
                f"autocast is unsupported for device type {x.device.type!r}."
            )
        with torch.autocast(
            device_type=x.device.type, dtype=self.autocast_dtype, enabled=True
        ):
            denoised = self.denoiser(x, sigma_denoiser)
        return denoised.to(dtype=x.dtype)

    def prox(self, x, sigma_denoiser, *args, gamma=1.0, **kwargs):
        if self.norm_strategy == "dynamic":
            sig_min, sig_max = self.clip_range
            scale = max(sig_max - sig_min, self.eps)
            x_unit = (x - sig_min) / scale
            denoised = self._denoise(x_unit, sigma_denoiser)
            if self.denoiser_lambda_relaxation is not None:
                lamda = float(self.denoiser_lambda_relaxation)
                step_size = _to_float(gamma)
                alpha = (step_size * lamda) / (1.0 + step_size * lamda)
                denoised = (1.0 - alpha) * x_unit + alpha * denoised
            return denoised * scale + sig_min

        denoised = self._denoise(x, sigma_denoiser)
        if self.denoiser_lambda_relaxation is not None:
            lamda = float(self.denoiser_lambda_relaxation)
            step_size = _to_float(gamma)
            alpha = (step_size * lamda) / (1.0 + step_size * lamda)
            denoised = (1.0 - alpha) * x + alpha * denoised
        if self.clip_range is not None:
            denoised = denoised.clamp(self.clip_range[0], self.clip_range[1])
        return denoised


def create_drunet(
    ground_truth_shape: tuple[int, ...],
    device: torch.device,
    pretrained: str | None = "download",
):
    import deepinv as dinv

    if len(ground_truth_shape) != 4:
        raise ValueError(
            f"Radio demo expects 2-D tensors shaped (B,C,H,W), got {ground_truth_shape}."
        )
    channels = int(ground_truth_shape[1])
    if channels != 1:
        raise ValueError(f"Expected single-channel radio image, got {channels}.")

    return dinv.models.DRUNet(
        in_channels=1,
        out_channels=1,
        pretrained=pretrained,
        device=device,
    )


def set_radio_model_params(
    model,
    step_size: float,
    clip_range: tuple[float, float],
    train_stepsize: bool = False,
) -> None:
    # Once trainable, only seed stepsize on the first call so later gradient
    # updates from the optimizer are not overwritten on every forward pass.
    already_initialized = getattr(model, "_stepsize_initialized", False)
    if (
        (not train_stepsize or not already_initialized)
        and hasattr(model, "params_algo")
        and "stepsize" in model.params_algo
    ):
        steps = model.params_algo["stepsize"]
        if isinstance(steps, torch.nn.ParameterList):
            for param in steps:
                param.data.fill_(float(step_size))
        elif isinstance(steps, list):
            for i in range(len(steps)):
                steps[i] = float(step_size)
        else:
            model.params_algo["stepsize"] = [float(step_size)]
        if train_stepsize:
            model._stepsize_initialized = True

    priors = getattr(model, "prior", [])
    for prior in priors:
        if hasattr(prior, "set_clip_range"):
            prior.set_clip_range(clip_range)


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)
