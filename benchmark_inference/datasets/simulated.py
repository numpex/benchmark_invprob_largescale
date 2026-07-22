"""Multiframe super-resolution dataset for benchmarking large scale inverse problems."""

from benchopt import BaseDataset, config
from deepinv.distributed import DistributedContext

from toolsbench.invprob import (
    DenoisingInvProb,
    MultiFrameSuperResInvProb,
    InvProbConfig,
)
from toolsbench.utils import save_measurements_figure, setup_distributed_env


class Dataset(BaseDataset):
    # Name of the Dataset, used to select it in the CLI
    name = "simulated"

    parameters = {
        "image_size": [2048],
        "batch_size": [1],
        "channels": [3],
        "num_operators": [1, 8, 16],
        "noise_level": [0.1],
        "seed": [42],
    }

    def prepare(self):
        return

    def get_data(self):
        """Load the data for this Dataset.

        Creates stacked physics operators and measurements using deepinv examples.
        Returns dictionary with keys expected by Objective.set_data().
        """
        setup_distributed_env()

        # Use cleanup=False to keep process group alive for solver
        # Solver will handle cleanup when it's done
        with DistributedContext(seed=self.seed, cleanup=False) as ctx:
            print(f"DistributedContext: rank {ctx.rank} / {ctx.world_size}")

            # Setup device
            device = ctx.device

            # Normalize image_size: int or [s] -> square 2D; [h, w] -> 2D;
            # [d, h, w] -> 3D volume. A 3-element size selects 3D; the super-res
            # physics is 2D-only, so 3D uses a pure denoising problem instead.
            size = self.image_size
            if isinstance(size, (list, tuple)):
                size = int(size[0]) if len(size) == 1 else tuple(size)
            is_3d = isinstance(size, tuple) and len(size) == 3
            if is_3d:
                params = {
                    "num_frames": self.num_operators,
                    "noise_std": self.noise_level,
                    "data": "synthetic",
                }
            else:
                params = {
                    "num_frames": self.num_operators,
                    "scale_factor": 2,
                    "noise_std": self.noise_level,
                    "blur_kernel_size": 5,
                    "blur_sigma": 1.0,
                    "data": "synthetic",
                }

            invprob_conf = InvProbConfig(
                size=size,
                batch_size=self.batch_size,
                channels=self.channels,
                device=device,
                data_path=config.get_data_path(key="simulated"),
                params=params,
            )

            if is_3d:
                invprob = DenoisingInvProb().get_invprob(invprob_conf)
            else:
                invprob = MultiFrameSuperResInvProb().get_invprob(invprob_conf)

            if ctx.rank == 0:
                # Save debug visualization
                save_measurements_figure(
                    invprob.ground_truth,
                    invprob.measurements,
                    filename="simulated.png",
                )

        return invprob.asdict()
