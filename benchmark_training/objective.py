"""Reconstruction objective for unrolled-model training benchmarking.

Mirrors the inference objective: each benchopt iteration is one training step,
the solver returns its current reconstruction, and the objective scores it with
PSNR / SSIM / MSE.  The only difference from inference is that ``get_objective``
also forwards ``ground_truth`` so the supervised training loss can use it.
"""

import torch
from benchopt import BaseObjective
from deepinv.loss.metric import PSNR


class Objective(BaseObjective):
    """Training objective scoring reconstruction quality per training step."""

    name = "reconstruction_objective"

    def set_data(
        self,
        ground_truth=None,
        measurements=None,
        physics=None,
        min_pixel=0.0,
        max_pixel=1.0,
        ground_truth_shape=None,
        num_operators=None,
        **kwargs,
    ):
        """Set the data from a Dataset to compute the objective.

        Parameters
        ----------
        ground_truth : torch.Tensor
            Ground truth image (used both for supervision and scoring).
        measurements : torch.Tensor or TensorList
            Noisy measurements.
        physics : Physics
            Forward operator.
        min_pixel, max_pixel : float, optional
            Pixel value range for metrics.
        ground_truth_shape : tuple, optional
            Shape of the ground truth tensor.
        num_operators : int, optional
            Number of operators in the stacked physics.
        **kwargs :
            Extra dataset-specific parameters forwarded to the solver.
        """
        self.ground_truth = ground_truth
        self.measurements = measurements
        self.physics = physics
        self._extra_kwargs = kwargs
        self.ground_truth_shape = ground_truth_shape
        if self.ground_truth_shape is None and ground_truth is not None:
            self.ground_truth_shape = ground_truth.shape
        self.num_operators = num_operators if num_operators is not None else 1
        self.psnr_metric = PSNR(max_pixel=max_pixel)
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel

    def get_objective(self):
        """Returns a dict passed to Solver.set_objective.

        Includes ``ground_truth`` (unlike the inference objective) so the
        supervised training loss can be computed.
        """
        objective = dict(
            ground_truth_shape=self.ground_truth_shape,
            num_operators=self.num_operators,
            min_pixel=self.min_pixel,
            max_pixel=self.max_pixel,
            **self._extra_kwargs,
        )
        if self.ground_truth is not None:
            objective["ground_truth"] = self.ground_truth
        if self.measurements is not None:
            objective["measurements"] = self.measurements
        if self.physics is not None:
            objective["physics"] = self.physics
        return objective

    def evaluate_result(
        self,
        reconstruction,
        name,
        ground_truth=None,
        min_pixel=None,
        max_pixel=None,
        **kwargs,
    ):
        """Score the reconstruction returned by the solver for this step.

        Parameters
        ----------
        reconstruction : torch.Tensor
            Reconstruction from the current training step.
        name : str
            Name identifier for the solver/configuration.
        **kwargs : dict
            Optional per-step / GPU metrics from the profiler.

        Returns
        -------
        dict
            ``value`` (negative PSNR for minimization), ``psnr`` plus any forwarded metrics.
        """
        with torch.no_grad():
            gt = ground_truth if ground_truth is not None else self.ground_truth
            if gt is None:
                raise ValueError(
                    "The training solver must return its current ground_truth."
                )
            lo = self.min_pixel if min_pixel is None else float(min_pixel)
            hi = self.max_pixel if max_pixel is None else float(max_pixel)
            reconstruction = reconstruction.to(gt.device)
            reconstruction = torch.clamp(reconstruction, min=lo, max=hi)
            ground_truth = torch.clamp(gt, min=lo, max=hi)

            psnr_tensor = PSNR(max_pixel=hi)(reconstruction, ground_truth)
            psnr = (
                psnr_tensor.mean().item()
                if psnr_tensor.numel() > 1
                else psnr_tensor.item()
            )

        result = dict(value=-psnr, psnr=psnr)
        for key, value in kwargs.items():
            if value is not None:
                result[key] = value
        return result

    def get_one_result(self):
        """Return one solution for which the objective can be evaluated."""
        return dict(
            reconstruction=torch.zeros(1, 1, 1, 1),
            ground_truth=torch.ones(1, 1, 1, 1),
            min_pixel=0.0,
            max_pixel=1.0,
            name="test_result",
        )
