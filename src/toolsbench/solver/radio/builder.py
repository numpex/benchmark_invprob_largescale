"""DeepInv construction helpers for joint source/diffuse unfolding."""

from __future__ import annotations

import math

import torch

from .iteration import AlternatingSourcePGDIteration


class _JointReconstructionOutput:
    """``get_output`` callable that also caches the refined final state.

    ``optim_builder`` only returns the summed tensor from ``get_output``, so
    without this cache callers cannot recover the post-refinement diffuse and
    source components separately from the pre-refinement ``model_init``.
    """

    def __init__(self) -> None:
        self.last_state: dict | None = None

    def __call__(self, state: dict) -> torch.Tensor:
        self.last_state = state
        return state["est"][0] + state["source_image"]


def create_alternating_source_model(
    prior,
    n_iter: int,
    step_size: float,
    denoiser_sigma: float,
    source_mode: str = "amplitude_position",
    source_max_shift: float = 0.75,
    source_lambda_amp: float = 1e-3,
    source_lambda_pos: float = 1e-3,
    source_amplitude_step: float = 1.0,
    source_position_step: float = 1.0,
    source_recompute_residual: bool = False,
    train_source_update: bool = True,
    data_fidelity=None,
    train_denoiser_sigma: bool = False,
    train_stepsize: bool = False,
):
    """Build a DeepInv unfolded model over the joint ``(d, theta)`` state."""
    import deepinv as dinv

    if source_amplitude_step <= 0 or source_position_step <= 0:
        raise ValueError("Source amplitude and position steps must be positive.")

    def inverse_softplus(value: float) -> float:
        return math.log(math.expm1(float(value)))

    iterator = AlternatingSourcePGDIteration(
        source_mode=source_mode,
        source_max_shift=source_max_shift,
        source_lambda_amp=source_lambda_amp,
        source_lambda_pos=source_lambda_pos,
        source_recompute_residual=source_recompute_residual,
    )
    params_algo = {
        "stepsize": float(step_size),
        "g_param": float(denoiser_sigma),
        "lambda": 1.0,
        "beta": 1.0,
        "source_log_amplitude_step": [
            inverse_softplus(source_amplitude_step) for _ in range(int(n_iter))
        ],
        "source_log_position_step": [
            inverse_softplus(source_position_step) for _ in range(int(n_iter))
        ],
    }
    mode = str(source_mode).lower()
    if mode == "position":
        mode = "amplitude_position"
    trainable_params: list[str] = []
    if train_source_update and mode != "fixed":
        trainable_params.append("source_log_amplitude_step")
        if mode == "amplitude_position":
            trainable_params.append("source_log_position_step")
    if data_fidelity is None:
        data_fidelity = dinv.optim.data_fidelity.L2()
    if train_denoiser_sigma:
        trainable_params.append("g_param")
    if train_stepsize:
        trainable_params.append("stepsize")
    return dinv.optim.optim_builder(
        iteration=iterator,
        max_iter=int(n_iter),
        params_algo=params_algo,
        data_fidelity=data_fidelity,
        prior=prior,
        unfold=True,
        trainable_params=trainable_params,
        get_output=_JointReconstructionOutput(),
    )
