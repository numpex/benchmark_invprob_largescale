import torch
from deepinv.distributed import distribute

from toolsbench.utils import create_denoiser
from toolsbench.utils.solver_utils import (
    distributed_callback_iter,
    profile_roofline,
    sync_and_barrier,
)


class DenoiserSolver:
    """Denoiser throughput probe: repeated denoiser forward passes.

    Standalone algorithm class — no benchopt dependency. Each iteration is a
    single denoiser call ``x <- denoiser(x)``; this is a timing loop, not a
    reconstruction algorithm, so the reported PSNR is not meaningful.

    ``compile`` selects where :func:`torch.compile` is applied:

    - ``None``: eager.
    - ``"pre"``: compile the denoiser before ``distribute``.
    - ``"post"``: compile the distributed wrapper (requires
      ``distribute_denoiser=True``; ``distribute`` also tiles on a single GPU).

    ``image_size`` restates the problem at a different spatial size before
    timing; ``None`` keeps the dataset's own size.
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
        denoiser_sigma=0.05,
        compile=None,
        distribute_denoiser=False,
        patch_size=128,
        overlap=32,
        max_batch_size=0,
        roofline=True,
        image_size=None,
    ):
        if compile == "post" and not distribute_denoiser:
            raise ValueError(
                "compile='post' requires distribute_denoiser=True: without it there "
                "is no distributed wrapper to compile."
            )

        self.problem = problem
        self.device = device
        self.profiler = profiler
        self.ctx = ctx
        self.distributed_mode = distributed_mode
        self.denoiser = denoiser
        self.denoiser_sigma = denoiser_sigma
        self.compile = compile
        self.distribute_denoiser = distribute_denoiser
        self.patch_size = patch_size
        self.overlap = overlap
        self.max_batch_size = max_batch_size
        self.roofline = roofline
        if image_size is not None:
            self.problem = problem.resized(image_size, device=device)
        self.shape = tuple(self.problem.ground_truth_shape)
        self.reconstruction = None
        self.reference = None
        self.roofline_metrics = {}

    def run(self, cb):

        noisy = next(iter(self.problem.measurements))
        self.reconstruction = (
            noisy.to(self.device).clone()
            if noisy.shape == self.shape
            else torch.rand(self.shape, device=self.device)
        )
        self.reference = self.reconstruction.clone()
        print(f"Reconstruction initialized with shape {tuple(self.shape)}.")

        if self.roofline:
            self.roofline_metrics = self._profile_roofline()
            print("Roofline:", self.roofline_metrics)

        denoiser = self._setup_denoiser()
        print("Denoiser set up.")

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        print("Starting denoiser iterations.")
        self._run_iterations(denoiser, cb)

        sync_and_barrier(self.device, self.ctx)

    def _setup_denoiser(self):
        denoiser = create_denoiser(
            self.denoiser, self.shape, self.device, torch.float32
        )

        if self.compile == "pre":
            denoiser = torch.compile(denoiser, fullgraph=True, mode="max-autotune")

        if self.ctx is not None and self.distribute_denoiser:
            denoiser = distribute(
                denoiser,
                self.ctx,
                patch_size=self.patch_size,
                overlap=self.overlap,
                tiling_dims=((-3, -2, -1) if len(self.shape) == 5 else (-2, -1)),
                max_batch_size=self.max_batch_size,
                type_object="denoiser",
            )

        if self.compile == "post":
            denoiser = torch.compile(denoiser, fullgraph=True, mode="max-autotune")

        return denoiser

    def _profile_roofline(self):
        """Roofline of one un-tiled, eager forward pass on the full image.

        Runs on a throw-away eager model, so it is unaffected by torch.compile
        and by tiling. Skipped (empty metrics) if the full image does not fit.
        """
        model = create_denoiser(self.denoiser, self.shape, self.device, torch.float32)
        try:
            metrics = profile_roofline(model, self.reconstruction, self.denoiser_sigma)
        except torch.cuda.OutOfMemoryError:
            print(
                "Roofline profiling skipped: full-image forward does not fit in memory."
            )
            metrics = {}
        finally:
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return metrics

    def _run_iterations(self, denoiser, cb):
        with torch.no_grad():
            for _ in distributed_callback_iter(
                cb, self.distributed_mode, self.device, self.ctx
            ):
                with self.profiler.track_step("denoise"):
                    self.reconstruction = denoiser(
                        self.reconstruction, sigma=self.denoiser_sigma
                    ).clone()
                self.profiler.end_iteration(self.ctx)

    def get_result(self):
        result = dict(reconstruction=self.reconstruction, ground_truth=self.reference)
        result.update(self.roofline_metrics)
        if self.profiler is not None:
            result.update(self.profiler.get_current_metrics())
        return result
