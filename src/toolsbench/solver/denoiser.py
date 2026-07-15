import torch
from deepinv.distributed import distribute
from deepinv.physics import Physics, stack

from toolsbench.utils import create_denoiser
from toolsbench.utils.solver_utils import (
    distributed_callback_iter,
    initialize_reconstruction,
    measurement_to_device,
    sync_and_barrier,
)


def profile_roofline(model, x, sigma, bytes_per_elem=4):
    """FLOPs, memory traffic and arithmetic intensity of one forward pass.

    Counts conv/linear FLOPs and the bytes touched by every leaf module
    (inputs, outputs, weights) through forward hooks, so the numbers describe
    the denoiser workload itself, independent of how it is executed.
    """
    total_flops, total_bytes = [0], [0]

    def hook(module, inp, out):
        for t in inp:
            if isinstance(t, torch.Tensor):
                total_bytes[0] += t.numel() * bytes_per_elem
        if isinstance(out, torch.Tensor):
            total_bytes[0] += out.numel() * bytes_per_elem
        weight = getattr(module, "weight", None)
        if weight is not None:
            total_bytes[0] += weight.numel() * bytes_per_elem

        conv_types = (
            torch.nn.Conv2d, torch.nn.Conv3d,
            torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
        )
        if isinstance(module, conv_types) and isinstance(out, torch.Tensor):
            weight = module.weight
            out_spatial = out.numel() // (out.shape[0] * out.shape[1])
            kernel_ops = int(torch.tensor(weight.shape[1:]).prod().item())
            batch = inp[0].shape[0] if isinstance(inp[0], torch.Tensor) else 1
            total_flops[0] += 2 * kernel_ops * weight.shape[0] * out_spatial * batch
        elif isinstance(module, torch.nn.Linear) and isinstance(inp[0], torch.Tensor):
            batch = int(torch.tensor(inp[0].shape[:-1]).prod().item())
            total_flops[0] += 2 * module.in_features * module.out_features * batch

    hooks = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if not list(m.children())
    ]
    try:
        with torch.no_grad():
            model(x, sigma=sigma)
    finally:
        for h in hooks:
            h.remove()

    flops, mem_bytes = total_flops[0], total_bytes[0]
    return dict(
        flops=flops,
        mem_bytes=mem_bytes,
        arith_intensity=flops / mem_bytes if mem_bytes > 0 else 0.0,
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
        init_method="pseudo_inverse",
        compile=None,
        distribute_denoiser=False,
        patch_size=128,
        overlap=32,
        max_batch_size=0,
        roofline=True,
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
        self.init_method = init_method
        self.compile = compile
        self.distribute_denoiser = distribute_denoiser
        self.patch_size = patch_size
        self.overlap = overlap
        self.max_batch_size = max_batch_size
        self.roofline = roofline
        self.reconstruction = None
        self.roofline_metrics = {}

    def run(self, cb):
        measurement = measurement_to_device(self.problem.measurements, self.device)
        physics = self._setup_physics()

        self.reconstruction = self._initialize_reconstruction(physics, measurement)
        print("Reconstruction initialized.")

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

    def _setup_physics(self):
        """Physics is only needed to build the initial image."""
        if callable(self.problem.physics) and not isinstance(self.problem.physics, Physics):
            return stack(*[
                self.problem.physics(i, self.device, None)
                for i in range(self.problem.num_operators)
            ])
        physics = self.problem.physics
        if hasattr(physics, "to"):
            physics = physics.to(self.device)
        return physics

    def _setup_denoiser(self):
        denoiser = create_denoiser(
            self.denoiser, self.problem.ground_truth_shape, self.device, torch.float32
        )

        if self.compile == "pre":
            denoiser = torch.compile(denoiser, fullgraph=True, mode="max-autotune")

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
                type_object="denoiser",
            )

        if self.compile == "post":
            denoiser = torch.compile(denoiser, fullgraph=True, mode="max-autotune")

        return denoiser

    def _initialize_reconstruction(self, physics, measurement):
        with torch.no_grad():
            return initialize_reconstruction(
                signal_shape=self.problem.ground_truth_shape,
                operator=physics,
                measurements=measurement,
                device=self.device,
                method=self.init_method,
                clip_range=(self.problem.min_pixel, self.problem.max_pixel),
                weights=self.problem.invprob_kwargs.get("weights") if self.problem.invprob_kwargs else None,
            )

    def _profile_roofline(self):
        """Roofline of one un-tiled, eager forward pass on the full image.

        Runs on a throw-away eager model, so it is unaffected by torch.compile
        and by tiling. Skipped (empty metrics) if the full image does not fit.
        """
        model = create_denoiser(
            self.denoiser, self.problem.ground_truth_shape, self.device, torch.float32
        )
        try:
            metrics = profile_roofline(model, self.reconstruction, self.denoiser_sigma)
        except torch.cuda.OutOfMemoryError:
            print("Roofline profiling skipped: full-image forward does not fit in memory.")
            metrics = {}
        finally:
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return metrics

    def _run_iterations(self, denoiser, cb):
        with torch.no_grad():
            for _ in distributed_callback_iter(cb, self.distributed_mode, self.device, self.ctx):
                with self.profiler.track_step("denoise"):
                    # .clone() copies the output out of torch.compile's static
                    # CUDA-graph buffer (mode="max-autotune"); without it, feeding
                    # the output back as the next input raises "accessing tensor
                    # output of CUDAGraphs that has been overwritten".
                    self.reconstruction = denoiser(
                        self.reconstruction, sigma=self.denoiser_sigma
                    ).clone()
                self.profiler.end_iteration(self.ctx)

    def get_result(self):
        result = dict(reconstruction=self.reconstruction)
        result.update(self.roofline_metrics)
        if self.profiler is not None:
            result.update(self.profiler.get_current_metrics())
        return result
