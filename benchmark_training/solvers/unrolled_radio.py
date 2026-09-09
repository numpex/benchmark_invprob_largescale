"""Benchopt adapter for source-aware unrolled radio training."""

from benchopt import BaseSolver
from benchopt.stopping_criterion import NoCriterion
from deepinv.distributed import DistributedContext

from toolsbench.profiler import create_profiler
from toolsbench.solver.radio import RadioTrainingSolver
from toolsbench.utils import setup_distributed_env
from toolsbench.utils.solver_utils import build_solver_name


class Solver(BaseSolver):
    name = "UnrolledRadio"
    sampling_strategy = "callback"
    stopping_criterion = NoCriterion()

    parameters = {
        "n_iter": [3],
        "step_size": [None],
        "step_size_scale": [0.99],
        "norm_max_iter": [100],
        "norm_tol": [1e-3],
        "denoiser_sigma": [1e-3],
        "denoiser_lambda_relaxation": [None],
        "norm_strategy": ["dynamic"],
        "drunet_pretrained": ["download"],
        "train_denoiser_sigma": [False],
        "train_stepsize": [False],
        "model_learning_rate": [1e-6],
        "loss": ["mse"],
        "loss_alpha": [1.0],
        "clip_grad_norm": [1e-2],
        "source_refinement": ["fixed"],
        "source_max_shift": [0.75],
        "source_lambda_amp": [1e-3],
        "source_lambda_pos": [1e-3],
        "source_amplitude_step": [1.0],
        "source_position_step": [1.0],
        "source_recompute_residual": [False],
        "train_source_update": [False],
        "distribute_denoiser": [False],
        "patch_size": [[512, 512]],
        "overlap": [[64, 64]],
        "max_batch_size": [1],
        "checkpoint_batches": ["auto"],
        # A solver parameter so sharding can be coupled to the SLURM topology
        # without creating invalid Cartesian products in experiment configs.
        "physics_sharding": [None],
        # Number of ranks cooperating on one sample. Remaining orthogonal
        # groups are data-parallel replicas managed by DDP.
        "inner_world_size": [1],
        "device_mode": ["auto"],
        "gradient_reduction": ["mean"],
        "deterministic": [True],
        "name_prefix": ["unrolled_radio"],
        "profiler_mode": ["custom"],
        "profiler_warmup": [0],
        "profiler_active": [0],
        "profiler_trace_dir": [None],
        "profiler_per_step": [True],
        "profiler_repeat": [1],
        "profiler_save_file": [False],
        "slurm_nodes": [1],
        "slurm_ntasks_per_node": [1],
        "slurm_gres": ["gpu:1"],
        "torchrun_nproc_per_node": [1],
    }

    def set_objective(self, radio_data_config, ground_truth_shape, **kwargs):
        del kwargs
        self.radio_data_config = dict(radio_data_config)
        self.ground_truth_shape = tuple(ground_truth_shape)
        self.world_size = setup_distributed_env()
        if self.world_size % int(self.inner_world_size):
            raise ValueError("inner_world_size must divide the launched WORLD_SIZE.")
        self._algo = None
        self.name = build_solver_name(
            self.name_prefix,
            self.slurm_nodes,
            self.slurm_ntasks_per_node,
            self.torchrun_nproc_per_node,
            self.world_size > 1,
        )

    def run(self, cb):
        with DistributedContext(
            seed=int(self.radio_data_config["seed"]),
            seed_offset=True,
            cleanup=True,
            deterministic=bool(self.deterministic),
            device_mode=(
                None if str(self.device_mode).lower() == "auto" else self.device_mode
            ),
            inner_world_size=int(self.inner_world_size),
            gradient_reduction=self.gradient_reduction,
        ) as ctx:
            profiler = create_profiler(
                self.profiler_mode,
                ctx.device,
                self.name,
                warmup=self.profiler_warmup,
                active=self.profiler_active,
                trace_dir=self.profiler_trace_dir,
                per_step=self.profiler_per_step,
                repeat=self.profiler_repeat,
                save_file=self.profiler_save_file,
            )
            self._algo = RadioTrainingSolver(
                self.radio_data_config,
                ctx,
                profiler,
                **{
                    key: getattr(self, key)
                    for key in (
                        "n_iter",
                        "step_size",
                        "step_size_scale",
                        "norm_max_iter",
                        "norm_tol",
                        "denoiser_sigma",
                        "denoiser_lambda_relaxation",
                        "norm_strategy",
                        "drunet_pretrained",
                        "train_denoiser_sigma",
                        "train_stepsize",
                        "model_learning_rate",
                        "loss",
                        "loss_alpha",
                        "clip_grad_norm",
                        "source_refinement",
                        "source_max_shift",
                        "source_lambda_amp",
                        "source_lambda_pos",
                        "source_amplitude_step",
                        "source_position_step",
                        "source_recompute_residual",
                        "train_source_update",
                        "distribute_denoiser",
                        "patch_size",
                        "overlap",
                        "max_batch_size",
                        "checkpoint_batches",
                        "physics_sharding",
                    )
                },
            )
            self._algo.setup()
            with profiler:
                while cb():
                    self._algo.run_step()
                    profiler.end_iteration(ctx)
            profiler.finalize(ctx)

    def get_result(self):
        if self._algo is None:
            return {
                "reconstruction": __import__("torch").zeros(1, 1, 1, 1),
                "ground_truth": __import__("torch").ones(1, 1, 1, 1),
                "name": self.name,
            }
        return {"name": self.name, **self._algo.get_result()}

    def get_next(self, stop_val):
        return stop_val + 1
