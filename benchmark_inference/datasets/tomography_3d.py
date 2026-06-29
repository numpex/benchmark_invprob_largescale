from benchopt import BaseDataset, config
from deepinv.distributed import DistributedContext

from toolsbench.data import check_installed
from toolsbench.invprob import InvProbConfig, TomographyInvProb
from toolsbench.utils import save_measurements_figure, setup_distributed_env


class Dataset(BaseDataset):
    # Name of the Dataset, used to select it in the CLI
    name = "tomography_3d"
    prepare_cache_ignore = "all"

    parameters = {
        "image_size": [512],
        "batch_size": [1],
        "num_operators": [1],
        "num_angles": [100],
        "num_projections": [100],
        "noise_level": [0.01],
        "seed": [42],
        "geometry_type_3d": ["conebeam"],
        "use_dataset_sinogram": [True],
    }

    def prepare(self):
        check_installed("tomography_3d", config.get_data_path(key="tomography"))

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
            size = self._get_size()

            invprob_conf = InvProbConfig(
                size=size,
                batch_size=self.batch_size,
                channels=1,
                device=device,
                data_path=config.get_data_path(key="tomography"),
                params={
                    "data": "3d",
                    "num_operators": self.num_operators,
                    "num_angles": self.num_angles,
                    "num_projections": self.num_projections,
                    "noise_level": self.noise_level,
                    "seed": self.seed,
                    "geometry_type_3d": self.geometry_type_3d,
                    "use_dataset_sinogram": self.use_dataset_sinogram,
                },
            )

            invprob = TomographyInvProb().get_invprob(invprob_conf)

            if ctx.rank == 0:
                # Save debug visualization
                save_measurements_figure(
                    invprob.ground_truth,
                    invprob.measurements,
                    filename="tomography_3d.png",
                )

        return invprob.asdict()

    def _get_size(self):
        if isinstance(self.image_size, int):
            return (self.image_size, self.image_size)
        return tuple(self.image_size)
