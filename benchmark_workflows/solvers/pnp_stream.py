from __future__ import annotations

import builtins
import os
import shutil
import tempfile
import time
import typing
from pathlib import Path

import torch
from benchopt import BaseSolver

# SimAI-Bench currently imports Dragon executor symbols eagerly.
# Define fallbacks so non-Dragon installs can still import high-level APIs.
if not hasattr(builtins, "Task"):
    builtins.Task = object
if not hasattr(builtins, "Any"):
    builtins.Any = typing.Any
if not hasattr(builtins, "Sequence"):
    builtins.Sequence = typing.Sequence

try:
    from SimAIBench import (
        DataStore,
        OchestratorConfig,
        ServerManager,
        SystemConfig,
        Workflow,
    )
    from SimAIBench import server_registry
    _SIMAIBENCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on runtime environment.
    DataStore = OchestratorConfig = ServerManager = SystemConfig = Workflow = None
    server_registry = None
    _SIMAIBENCH_IMPORT_ERROR = exc

from toolsbench.utils.simai_components import (
    error_key,
    pnp_consumer_component,
    producer_component,
    result_key,
)


class Solver(BaseSolver):
    """Single-process PnP solver using SimAI-Bench high-level APIs."""

    name = "PnPStreamSimAIBench"
    sampling_strategy = "run_once"

    parameters = {
        "step_size": [0.8],
        "denoiser_sigma": [0.05],
        "denoiser_kernel_size": [3],
        "denoiser_lambda_relaxation": [None],
        "inner_iterations": [1],
        "batch_size": [1, 4],
        "batch_wait_s": [0.0],
        "device": ["cpu"],
        "poll_interval_s": [0.01],
        "result_timeout_s": [300.0],
        "orchestrator_name": ["process-pool"],
        "server_type": ["filesystem"],
        "ncpus": [2],
        "ngpus": [0],
        # Submitit/SLURM parameters (used by benchopt --parallel-config backend=submitit)
        "slurm_nodes": [1],
        "slurm_ntasks_per_node": [1],
        "slurm_gres": ["gpu:1"],
    }

    def set_objective(
        self,
        stream_dataloader,
        physics_spec,
        stream_spec,
        ground_truth_shape,
        min_pixel=0.0,
        max_pixel=1.0,
    ):
        self.stream_dataloader = stream_dataloader
        self.stream_records = self._materialize_stream_records(stream_dataloader)
        self.physics_spec = physics_spec
        self.physics_mode = self._infer_physics_mode(self.physics_spec)
        self.stream_spec = dict(stream_spec)
        self.stream_spec["max_packets"] = min(
            int(self.stream_spec.get("max_packets", len(self.stream_records))),
            len(self.stream_records),
        )
        self.stream_spec.setdefault(
            "clamp_generated_measurements",
            self.physics_mode != "radio",
        )
        print(f"Materialized {len(self.stream_records)} stream record(s).", flush=True)
        self.ground_truth_shape = tuple(ground_truth_shape)
        self.min_pixel = float(min_pixel)
        self.max_pixel = float(max_pixel)

        self.reconstruction = torch.zeros(self.ground_truth_shape, dtype=torch.float32)
        self.trace = {}
        self.name = self.__class__.name

    @staticmethod
    def _materialize_stream_records(stream_dataloader):
        def copy_record_fields(record):
            copied = {"physics_spec": dict(record["physics_spec"])}
            if "image_path" in record:
                copied["image_path"] = record["image_path"]
            elif "image" in record:
                copied["image"] = record["image"].detach().cpu()
            else:
                raise KeyError("Each stream sample must contain 'image_path' or 'image'.")

            for optional_key in (
                "measurement",
                "measurement_path",
                "measurement_format",
                "measurement_source_path",
                "measurements_path",
            ):
                if optional_key in record:
                    value = record[optional_key]
                    if isinstance(value, torch.Tensor):
                        copied[optional_key] = value.detach().cpu()
                    else:
                        copied[optional_key] = value

            # Backward compatibility for old key naming in stream samples.
            if "measurements_path" in copied and "measurement_path" not in copied:
                copied["measurement_path"] = copied["measurements_path"]
            return copied

        dataset = getattr(stream_dataloader, "dataset", None)
        raw_records = getattr(dataset, "records", None)
        if raw_records is None:
            records = []
            for sample in stream_dataloader:
                records.append(copy_record_fields(sample))
        else:
            records = [copy_record_fields(record) for record in raw_records]

        if len(records) == 0:
            raise ValueError("stream_dataloader yielded no samples.")
        return records

    @staticmethod
    def _infer_physics_mode(physics_spec):
        if isinstance(physics_spec, dict):
            if "physics_mode" in physics_spec:
                return str(physics_spec["physics_mode"]).lower()
            if "samples_locs" in physics_spec:
                return "radio"
            if "blur_sigma" in physics_spec:
                return "blur"
        return "blur"

    def _build_workflow(self, server_info, key_prefix, compute_device):
        pnp_cfg = {
            "step_size": float(self.step_size),
            "denoiser_sigma": float(self.denoiser_sigma),
            "denoiser_kernel_size": int(self.denoiser_kernel_size),
            "denoiser_lambda_relaxation": self.denoiser_lambda_relaxation,
            "inner_iterations": int(self.inner_iterations),
            "batch_size": int(self.batch_size),
            "batch_wait_s": float(self.batch_wait_s),
            "min_pixel": float(self.min_pixel),
            "max_pixel": float(self.max_pixel),
            "device": str(compute_device),
            "poll_interval_s": float(self.poll_interval_s),
            "physics_mode": self.physics_mode,
            "denoiser_kind": "auto",
            "normalize_for_denoiser": self.physics_mode == "radio",
            "use_weighted_adjoint_init": self.physics_mode == "radio",
            "normalize_adjoint_init": self.physics_mode == "radio",
        }
        ncpus = int(self.ncpus)
        ngpus = int(self.ngpus)

        workflow = Workflow(
            orchestrator_config=OchestratorConfig(
                name=self.orchestrator_name,
                submit_loop_sleep_time=1,
            ),
            system_config=SystemConfig(name="local", ncpus=ncpus, ngpus=ngpus),
        )
        workflow.register_component(
            name="producer",
            executable=producer_component,
            type="local",
            args={
                "server_info": server_info,
                "key_prefix": key_prefix,
                "stream_records": self.stream_records,
                "stream_spec": self.stream_spec,
            },
        )
        workflow.register_component(
            name="pnp_consumer",
            executable=pnp_consumer_component,
            type="local",
            args={
                "server_info": server_info,
                "key_prefix": key_prefix,
                "physics_spec": {"physics_mode": self.physics_mode},
                "ground_truth_shape": self.ground_truth_shape,
                "pnp_cfg": pnp_cfg,
            },
        )
        return workflow

    @staticmethod
    def _ensure_worker_pythonpath():
        """Ensure workers can import sitecustomize startup fallbacks."""
        benchmark_root = str(Path(__file__).resolve().parents[2])
        current = os.environ.get("PYTHONPATH", "")
        paths = [p for p in current.split(os.pathsep) if p]
        if benchmark_root not in paths:
            os.environ["PYTHONPATH"] = (
                benchmark_root if not current else benchmark_root + os.pathsep + current
            )

    @staticmethod
    def _wait_for_result(result_reader, key, err_key, timeout_s=30.0):
        start = time.perf_counter()
        while time.perf_counter() - start < timeout_s:
            if result_reader.poll_staged_data(err_key):
                payload = result_reader.stage_read(err_key)
                component = payload.get("component", "unknown")
                error = payload.get("error", "unknown error")
                tb = payload.get("traceback", "")
                raise RuntimeError(
                    f"Workflow component '{component}' failed: {error}\n{tb}"
                )
            if result_reader.poll_staged_data(key):
                return result_reader.stage_read(key)
            time.sleep(0.1)
        raise RuntimeError(
            f"Timed out waiting for workflow result key '{key}'. "
            "Check SimAI-Bench logs for component failures."
        )

    @staticmethod
    def _ensure_simaibench_logdir():
        """Ensure SimAI-Bench can create logs under cwd/logs."""
        logs_dir = Path.cwd() / "logs"
        if logs_dir.exists() and not logs_dir.is_dir():
            raise RuntimeError(
                f"Cannot create SimAI-Bench logs: '{logs_dir}' exists and is not a directory."
            )
        logs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ensure_cuda_visible_devices():
        """Set CUDA_VISIBLE_DEVICES via submitit and disable Ray's GPU isolation.

        Two things are needed:
        1. submitit reads the SLURM GPU allocation and sets CUDA_VISIBLE_DEVICES
           in the *current* (parent) process so Ray workers inherit the right value.
        2. RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 prevents Ray from
           overriding CUDA_VISIBLE_DEVICES with "" in worker processes for tasks
           that don't explicitly request GPU resources (Ray's default isolation
           behaviour). Must be set before workflow.launch() / ray.init().
        """
        # Step 1: populate CUDA_VISIBLE_DEVICES from SLURM via submitit.
        try:
            import submitit
            submitit.helpers.TorchDistributedEnvironment().export(
                set_cuda_visible_devices=True
            )
            print(
                "[solver] CUDA_VISIBLE_DEVICES set via submitit: "
                f"'{os.environ.get('CUDA_VISIBLE_DEVICES', '')}'",
                flush=True,
            )
        except (ImportError, RuntimeError) as exc:
            print(f"[solver] submitit GPU export skipped ({exc}).", flush=True)

        # Step 2: tell Ray not to override CUDA_VISIBLE_DEVICES in workers.
        os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")

    def run(self, n_iter=None):
        del n_iter
        if _SIMAIBENCH_IMPORT_ERROR is not None:
            raise ImportError(
                "SimAIBench is required to run PnPStreamSimAIBench. "
                "Install/activate SimAIBench in the runtime environment."
            ) from _SIMAIBENCH_IMPORT_ERROR
        self._ensure_cuda_visible_devices()
        self._ensure_worker_pythonpath()
        self._ensure_simaibench_logdir()
        compute_device = (
            torch.device("cuda")
            if self.device == "cuda" and torch.cuda.is_available()
            else torch.device("cpu")
        )
        tmp_dir = tempfile.mkdtemp(prefix="infer_inverse_simai_")
        server = None
        try:
            if self.server_type == "filesystem":
                server_config = server_registry.create_config(
                    type="filesystem",
                    server_address=tmp_dir,
                    nshards=64,
                )
            elif self.server_type == "redis":
                redis_server_exe = shutil.which("redis-server")
                if redis_server_exe is None:
                    raise RuntimeError(
                        "server_type='redis' requires a 'redis-server' executable "
                        "available on PATH."
                    )
                server_config = server_registry.create_config(
                    type="redis",
                    server_address="localhost:6379",
                    redis_server_exe=redis_server_exe,
                    is_clustered=False,
                )
            else:
                raise ValueError(
                    f"Unsupported server_type '{self.server_type}'. "
                    "Expected one of: filesystem, redis."
                )
            server = ServerManager("stream_server", config=server_config)
            server.start_server()

            server_info = server.get_server_info()
            key_prefix = f"run_{time.time_ns()}"

            print("Building workflow...", flush=True)
            workflow = self._build_workflow(
                server_info=server_info,
                key_prefix=key_prefix,
                compute_device=compute_device,
            )
            print("Launching workflow...", flush=True)
            workflow.launch()

            print("Waiting for result...", flush=True)
            result_reader = DataStore("result_reader", server_info=server_info)
            output = self._wait_for_result(
                result_reader=result_reader,
                key=result_key(key_prefix),
                err_key=error_key(key_prefix),
                timeout_s=float(self.result_timeout_s),
            )

            self.reconstruction = output["reconstruction"]
            self.trace = output["trace"]
        finally:
            if server is not None:
                server.stop_server()
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def get_result(self):
        return dict(
            reconstruction=self.reconstruction,
            trace=self.trace,
            name=self.name,
        )
