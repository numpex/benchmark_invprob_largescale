"""Synthetic training dataset for unrolled-model benchmarking.

Uses a synthetic ground-truth signal with stacked blur + downsampling physics
(via :class:`MultiFrameSuperResInvProb`).  Returns a single in-memory batch,
mirroring the inference ``simulated`` dataset; the training solver runs one
gradient step per benchopt iteration on this batch.
"""

from benchopt import BaseDataset, config
from deepinv.distributed import DistributedContext

from toolsbench.invprob import InvProbConfig, MultiFrameSuperResInvProb
from toolsbench.utils import save_measurements_figure, setup_distributed_env


class Dataset(BaseDataset):
    # Name of the Dataset, used to select it in the CLI
    name = "synthetic"

    parameters = {
        "image_size": [256],
        "batch_size": [1],
        "channels": [3],
        "num_operators": [1, 8, 16],
        "noise_level": [0.1],
        "seed": [42],
    }

    def prepare(self):
        return

    def get_data(self):
        """Build a synthetic inverse problem (single batch).

        Returns the dict consumed by ``Objective.set_data``.
        """
        setup_distributed_env()

        # cleanup=False keeps the process group alive for the solver.
        with DistributedContext(seed=self.seed, cleanup=False) as ctx:
            print(f"DistributedContext: rank {ctx.rank} / {ctx.world_size}")

            device = ctx.device

            invprob_conf = InvProbConfig(
                size=self.image_size,
                batch_size=self.batch_size,
                channels=self.channels,
                device=device,
                data_path=config.get_data_path(key="synthetic"),
                params={
                    "num_frames": self.num_operators,
                    "scale_factor": 2,
                    "noise_std": self.noise_level,
                    "blur_kernel_size": 5,
                    "blur_sigma": 1.0,
                    "data": "synthetic",
                },
            )

            invprob = MultiFrameSuperResInvProb().get_invprob(invprob_conf)

            if ctx.rank == 0:
                save_measurements_figure(
                    invprob.ground_truth,
                    invprob.measurements,
                    filename="synthetic_train.png",
                )

        return invprob
