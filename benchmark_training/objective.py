"""Reconstruction objective for unrolled-model training benchmarking.

Mirrors the inference objective: each benchopt iteration is one training step,
the solver returns its current reconstruction, and the objective scores it with
PSNR / SSIM / MSE.  The only difference from inference is that ``get_objective``
also forwards ``ground_truth`` so the supervised training loss can use it.
"""


import torch
from benchopt import BaseObjective
from deepinv.loss.metric import PSNR, SSIM, MSE

from toolsbench.utils import save_comparison_figure


class Objective(BaseObjective):
    """Training objective scoring reconstruction quality per training step."""

    name = "reconstruction_objective"

    def set_data(
        self,
        ground_truth,
        measurements,
        physics,
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
        self.ground_truth_shape = (
            ground_truth_shape if ground_truth_shape is not None else ground_truth.shape
        )
        self.num_operators = num_operators if num_operators is not None else 1
        self.psnr_metric = PSNR(max_pixel=max_pixel)
        self.ssim_metric = SSIM(max_pixel=max_pixel)
        self.mse_metric = MSE()
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.evaluation_count = 0

    def get_objective(self):
        """Returns a dict passed to Solver.set_objective.

        Includes ``ground_truth`` (unlike the inference objective) so the
        supervised training loss can be computed.
        """
        return dict(
            ground_truth=self.ground_truth,
            measurements=self.measurements,
            physics=self.physics,
            ground_truth_shape=self.ground_truth_shape,
            num_operators=self.num_operators,
            min_pixel=self.min_pixel,
            max_pixel=self.max_pixel,
            **self._extra_kwargs,
        )

    def evaluate_result(self, reconstruction, name, **kwargs):
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
            ``value`` (negative PSNR for minimization), ``psnr``, ``ssim``,
            ``mse`` plus any forwarded metrics.
        """
        with torch.no_grad():
            reconstruction = reconstruction.to(self.ground_truth.device)
            reconstruction = torch.clamp(
                reconstruction, min=self.min_pixel, max=self.max_pixel
            )
            ground_truth = torch.clamp(
                self.ground_truth, min=self.min_pixel, max=self.max_pixel
            )

            psnr_tensor = self.psnr_metric(reconstruction, ground_truth)
            ssim_tensor = self.ssim_metric(reconstruction, ground_truth)
            mse_tensor = self.mse_metric(reconstruction, ground_truth)

            psnr = (
                psnr_tensor.mean().item()
                if psnr_tensor.numel() > 1
                else psnr_tensor.item()
            )
            ssim = (
                ssim_tensor.mean().item()
                if ssim_tensor.numel() > 1
                else ssim_tensor.item()
            )
            mse = (
                mse_tensor.mean().item()
                if mse_tensor.numel() > 1
                else mse_tensor.item()
            )

            output_dir = "evaluation_output/" + name.replace("/", "_").replace("..", "")
            self.evaluation_count += 1
            save_comparison_figure(
                self.ground_truth,
                reconstruction,
                metrics={"psnr": psnr, "ssim": ssim, "mse": mse},
                output_dir=output_dir,
                filename=f"eval_{self.evaluation_count:04d}.png",
                evaluation_count=self.evaluation_count,
                vmin=self.min_pixel,
                vmax=self.max_pixel,
            )

        result = dict(value=-psnr, psnr=psnr, ssim=ssim, mse=mse)
        for key, value in kwargs.items():
            if value is not None:
                result[key] = value
        return result

    def get_one_result(self):
        """Return one solution for which the objective can be evaluated."""
        return dict(
            reconstruction=self.ground_truth + self.ground_truth.std(),
            name="test_result",
        )
