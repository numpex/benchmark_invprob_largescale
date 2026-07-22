from __future__ import annotations

import builtins
import copy
import os
import re
import subprocess
import time
import traceback
import typing
from pathlib import Path

import numpy as np
import torch
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.physics.blur import Blur, gaussian_blur

# SimAI-Bench eagerly imports Dragon symbols. Define non-Dragon fallbacks.
if not hasattr(builtins, "Task"):
    builtins.Task = object
if not hasattr(builtins, "Any"):
    builtins.Any = typing.Any
if not hasattr(builtins, "Sequence"):
    builtins.Sequence = typing.Sequence

try:
    from SimAIBench import DataStore

    _SIMAIBENCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on runtime environment.
    DataStore = None
    _SIMAIBENCH_IMPORT_ERROR = exc

_PHYSICS_PAYLOAD_CACHE = {}


def _require_simaibench():
    if _SIMAIBENCH_IMPORT_ERROR is not None:
        raise ImportError(
            "SimAIBench is required to run stream producer/consumer components."
        ) from _SIMAIBENCH_IMPORT_ERROR


class BoxBlurDenoiser(torch.nn.Module):
    """Simple denoiser used for a minimal PnP baseline."""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = int(kernel_size)

    def forward(self, x, sigma=None):
        del sigma
        if x.ndim == 4:
            return torch.nn.functional.avg_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )
        if x.ndim == 5:
            return torch.nn.functional.avg_pool3d(
                x,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )
        return x


def _to_device(payload, device):
    if isinstance(payload, torch.Tensor):
        return payload.to(device)
    if isinstance(payload, list):
        return [_to_device(p, device) for p in payload]
    if isinstance(payload, tuple):
        return tuple(_to_device(p, device) for p in payload)
    return payload


def _concat_payloads(payloads):
    """Concatenate homogeneous payloads along batch dimension."""
    first = payloads[0]
    if isinstance(first, torch.Tensor):
        if first.ndim == 0:
            return torch.stack(payloads, dim=0)
        return torch.cat(payloads, dim=0)
    if isinstance(first, list):
        return [_concat_payloads([p[i] for p in payloads]) for i in range(len(first))]
    if isinstance(first, tuple):
        return tuple(
            _concat_payloads([p[i] for p in payloads]) for i in range(len(first))
        )
    raise TypeError(f"Unsupported payload type for batching: {type(first)}")


def _run_pnp_updates(
    reconstruction,
    measurement,
    physics,
    data_fidelity,
    prior,
    pnp_cfg,
):
    normalize_for_denoiser = bool(pnp_cfg.get("normalize_for_denoiser", False))
    sig_min = float(pnp_cfg["min_pixel"])
    sig_max = float(pnp_cfg["max_pixel"])
    scale = max(sig_max - sig_min, 1e-12)

    with torch.no_grad():
        for _ in range(int(pnp_cfg["inner_iterations"])):
            grad = data_fidelity.grad(reconstruction, measurement, physics)
            reconstruction = reconstruction - pnp_cfg["step_size"] * grad

            if normalize_for_denoiser:
                normalized = (reconstruction - sig_min) / scale
                denoised = prior.prox(
                    normalized,
                    sigma_denoiser=pnp_cfg["denoiser_sigma"],
                )
                if pnp_cfg["denoiser_lambda_relaxation"] is None:
                    reconstruction = denoised * scale + sig_min
                else:
                    lam = float(pnp_cfg["denoiser_lambda_relaxation"])
                    alpha = (pnp_cfg["step_size"] * lam) / (
                        1.0 + pnp_cfg["step_size"] * lam
                    )
                    relaxed = (1.0 - alpha) * normalized + alpha * denoised
                    reconstruction = relaxed * scale + sig_min
            else:
                denoised = prior.prox(
                    reconstruction,
                    sigma_denoiser=pnp_cfg["denoiser_sigma"],
                )
                if pnp_cfg["denoiser_lambda_relaxation"] is None:
                    reconstruction = denoised
                else:
                    lam = float(pnp_cfg["denoiser_lambda_relaxation"])
                    alpha = (pnp_cfg["step_size"] * lam) / (
                        1.0 + pnp_cfg["step_size"] * lam
                    )
                    reconstruction = (1.0 - alpha) * reconstruction + alpha * denoised

            reconstruction = reconstruction.clamp(sig_min, sig_max)
    return reconstruction


def _clone_payload(payload):
    if isinstance(payload, torch.Tensor):
        return payload.detach().clone()
    if isinstance(payload, list):
        return [_clone_payload(x) for x in payload]
    if isinstance(payload, tuple):
        return tuple(_clone_payload(x) for x in payload)
    return payload


def payload_nbytes(payload) -> int:
    if isinstance(payload, torch.Tensor):
        return payload.nelement() * payload.element_size()
    if isinstance(payload, list):
        return sum(payload_nbytes(x) for x in payload)
    if isinstance(payload, tuple):
        return sum(payload_nbytes(x) for x in payload)
    return 0


def packet_key(key_prefix: str, packet_id: int) -> str:
    return f"{key_prefix}:packet:{packet_id:08d}"


def eos_key(key_prefix: str) -> str:
    return f"{key_prefix}:eos"


def result_key(key_prefix: str) -> str:
    return f"{key_prefix}:result"


def error_key(key_prefix: str) -> str:
    return f"{key_prefix}:error"


def _normalize_image_tensor(image):
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected image tensor, got {type(image)}")
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim != 4:
        raise ValueError(f"Unsupported image tensor shape: {tuple(image.shape)}")
    return image.to(dtype=torch.float32)


def _load_image_from_path(image_path):
    image_path = str(image_path)
    suffix = Path(image_path).suffix.lower()

    if suffix in {".pt", ".pth", ".ckpt"}:
        return torch.load(image_path, map_location="cpu")

    if suffix in {".fits", ".fit", ".fts"}:
        from astropy.io import fits

        with fits.open(image_path, memmap=False) as hdul:
            image_np = hdul[0].data.astype(np.float32)
        return torch.from_numpy(image_np)

    raise ValueError(
        f"Unsupported image format '{suffix}' for path '{image_path}'. "
        "Expected torch tensor (.pt) or FITS (.fits)."
    )


def _load_measurement_from_path(measurement_path, measurement_format):
    measurement_path = str(measurement_path)
    measurement_format = (measurement_format or "auto").lower()

    if measurement_format == "auto":
        suffix = Path(measurement_path).suffix.lower()
        if suffix in {".pt", ".pth", ".ckpt"}:
            measurement_format = "torch"
        elif suffix in {".npy"}:
            measurement_format = "npy"
        elif suffix in {".npz"}:
            measurement_format = "npz"
        else:
            raise ValueError(
                "Unable to infer measurement format from path "
                f"'{measurement_path}'. Set sample['measurement_format']."
            )

    if measurement_format == "torch":
        return torch.load(measurement_path, map_location="cpu")

    if measurement_format == "npy":
        return torch.from_numpy(np.load(measurement_path))

    if measurement_format == "npz":
        npz = np.load(measurement_path)
        if "measurement" not in npz:
            raise KeyError(
                f"Expected key 'measurement' in npz file '{measurement_path}'."
            )
        return torch.from_numpy(npz["measurement"])

    raise ValueError(
        f"Unsupported measurement_format '{measurement_format}' for '{measurement_path}'."
    )


def _sample_to_image_and_spec(sample):
    if not isinstance(sample, dict):
        raise TypeError(f"Expected sample dictionary, got {type(sample)}")

    image = sample.get("image", None)
    image_path = sample.get("image_path", None)
    physics_spec = sample.get("physics_spec", None)

    if image is None and image_path is not None:
        image = _load_image_from_path(image_path)
    if image is None:
        raise KeyError("Missing image payload. Expected 'image' or 'image_path'.")
    if physics_spec is None:
        raise KeyError("Missing 'physics_spec' in stream sample.")

    return _normalize_image_tensor(image), dict(physics_spec)


def _sample_to_physics_spec(sample):
    if not isinstance(sample, dict):
        raise TypeError(f"Expected sample dictionary, got {type(sample)}")
    physics_spec = sample.get("physics_spec", None)
    if physics_spec is None:
        raise KeyError("Missing 'physics_spec' in stream sample.")
    return dict(physics_spec)


def _sample_inline_measurement(sample):
    if "measurement" not in sample or sample["measurement"] is None:
        return None
    measurement = sample["measurement"]
    if not isinstance(measurement, torch.Tensor):
        measurement = torch.as_tensor(measurement)
    return measurement


def _sample_measurement_path(sample):
    measurement_path = sample.get("measurement_path", None)
    if measurement_path is None:
        return None, None

    # Legacy radio records may pass only an MS path; in that case we fall back
    # to on-the-fly forward generation from the image and physics operator.
    if Path(str(measurement_path)).suffix.lower() == ".ms" and (
        "measurement_format" not in sample
    ):
        return None, None

    return str(measurement_path), sample.get("measurement_format", "auto")


def _sample_to_measurement(sample):
    measurement = _sample_inline_measurement(sample)
    if measurement is not None:
        return measurement

    measurement_path, measurement_format = _sample_measurement_path(sample)
    if measurement_path is not None:
        measurement = _load_measurement_from_path(measurement_path, measurement_format)
        if not isinstance(measurement, torch.Tensor):
            measurement = torch.as_tensor(measurement)
        return measurement

    return None


def _packet_to_measurement(packet, compute_device):
    if "y" in packet:
        return _to_device(packet["y"], compute_device)

    measurement_path = packet.get("measurement_path", None)
    if measurement_path is not None:
        measurement = _load_measurement_from_path(
            measurement_path,
            packet.get("measurement_format", "auto"),
        )
        if not isinstance(measurement, torch.Tensor):
            measurement = torch.as_tensor(measurement)
        return measurement.to(compute_device)

    raise KeyError("Packet must contain either 'y' or 'measurement_path'.")


def _detect_physics_mode(physics_spec):
    mode = physics_spec.get("physics_mode", None)
    if mode is not None:
        return str(mode).lower()
    if "samples_locs" in physics_spec:
        return "radio"
    if "blur_sigma" in physics_spec:
        return "blur"
    raise KeyError(
        "Could not infer physics mode from physics_spec. "
        "Expected 'physics_mode' or one of {'samples_locs', 'blur_sigma'}."
    )


def _load_physics_payload(physics_spec):
    payload_path = physics_spec.get("physics_payload_path", None)
    if payload_path is None:
        return {}
    payload_path = str(payload_path)
    cached = _PHYSICS_PAYLOAD_CACHE.get(payload_path, None)
    if cached is None:
        cached = torch.load(payload_path, map_location="cpu")
        if not isinstance(cached, dict):
            raise TypeError(
                "Expected physics payload file to contain a dictionary, got "
                f"{type(cached)} at '{payload_path}'."
            )
        _PHYSICS_PAYLOAD_CACHE[payload_path] = cached
    return cached


def _get_radio_samples_locs(physics_spec):
    if "samples_locs" in physics_spec:
        return physics_spec["samples_locs"]
    payload = _load_physics_payload(physics_spec)
    if "samples_locs" not in payload:
        raise KeyError(
            "Missing radio sampling locations. Expected either "
            "'physics_spec[\"samples_locs\"]' or a payload file with key "
            "'samples_locs' referenced by 'physics_payload_path'."
        )
    return payload["samples_locs"]


def _get_radio_weights(physics_spec):
    if "weights" in physics_spec:
        return physics_spec["weights"]
    payload = _load_physics_payload(physics_spec)
    return payload.get("weights", None)


def _normalize_gpu_id_list(raw_value):
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value or value in {"None", "none", "null"}:
        return None
    if value in {"NoDevFiles", "N/A"}:
        return None

    # Preserve UUID-style identifiers as-is.
    if "GPU-" in value or "MIG-" in value:
        return value

    # Handle formats like "gpu[0,1]" or "0,1" or "0 1".
    gpu_ids = re.findall(r"\d+", value)
    if gpu_ids:
        return ",".join(gpu_ids)

    return value


def _infer_visible_gpus_from_env():
    for env_name in (
        "AVAILABLE_GPUS",
        "SLURM_STEP_GPUS",
        "SLURM_JOB_GPUS",
        "NVIDIA_VISIBLE_DEVICES",
    ):
        normalized = _normalize_gpu_id_list(os.environ.get(env_name, None))
        if normalized:
            return normalized, env_name
    return None, None


def _infer_visible_gpus_from_nvidia_smi():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    gpu_ids = []
    for line in result.stdout.splitlines():
        token = line.strip()
        if token.isdigit():
            gpu_ids.append(token)
    if gpu_ids:
        return ",".join(gpu_ids)
    return None


def _physics_spec_key(physics_spec):
    if "physics_id" in physics_spec:
        return ("physics_id", physics_spec["physics_id"])

    scalar_items = []
    for key in sorted(physics_spec.keys()):
        value = physics_spec[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            scalar_items.append((key, value))

    if scalar_items:
        return tuple(scalar_items)

    return id(physics_spec)


def _build_blur_physics(physics_spec, compute_device):
    blur_sigma = float(physics_spec["blur_sigma"])
    kernel = gaussian_blur(
        sigma=(blur_sigma, blur_sigma),
        angle=0.0,
        device=str(compute_device),
    )
    return Blur(filter=kernel, padding="circular", device=str(compute_device))


def _build_radio_physics(ground_truth_shape, physics_spec, compute_device):
    from benchmark_invprob_largescale.src.toolsbench.utils.radio_interferometry.deepinv_imager import (
        MyRadioInterferometry,
    )

    imaging_npixel = int(ground_truth_shape[-1])
    samples_locs = _to_device(_get_radio_samples_locs(physics_spec), compute_device)
    k_oversampling = float(physics_spec.get("k_oversampling", 1.5))

    return MyRadioInterferometry(
        img_size=(imaging_npixel, imaging_npixel),
        samples_loc=samples_locs,
        real_projection=True,
        k_oversampling=k_oversampling,
        device=str(compute_device),
    )


def _build_physics_from_spec(ground_truth_shape, physics_spec, compute_device):
    mode = _detect_physics_mode(physics_spec)
    if mode == "blur":
        return _build_blur_physics(
            physics_spec=physics_spec, compute_device=compute_device
        )
    if mode == "radio":
        return _build_radio_physics(
            ground_truth_shape=ground_truth_shape,
            physics_spec=physics_spec,
            compute_device=compute_device,
        )
    raise ValueError(f"Unsupported physics mode '{mode}'.")


def _build_denoiser(mode, pnp_cfg, ground_truth_shape, compute_device):
    denoiser_kind = str(pnp_cfg.get("denoiser_kind", "auto")).lower()
    if denoiser_kind == "auto":
        denoiser_kind = "drunet" if mode == "radio" else "box"

    if denoiser_kind == "box":
        return BoxBlurDenoiser(kernel_size=pnp_cfg["denoiser_kernel_size"]).to(
            compute_device
        )

    if denoiser_kind == "drunet":
        from toolsbench.utils import create_drunet_denoiser

        print("Creating DRUNet denoiser...", flush=True)
        return create_drunet_denoiser(
            ground_truth_shape=ground_truth_shape,
            device=compute_device,
            dtype=torch.float32,
        )

    raise ValueError(
        f"Unsupported denoiser_kind '{denoiser_kind}'. Expected one of: auto, box, drunet."
    )


def _adjoint_initialization(
    physics,
    measurement,
    batch_physics_spec,
    pnp_cfg,
    compute_device,
    ground_truth_shape,
):
    if hasattr(physics, "A_adjoint"):
        with torch.no_grad():
            operator_for_init = physics
            if bool(pnp_cfg.get("use_weighted_adjoint_init", False)) and hasattr(
                physics, "setWeight"
            ):
                weights = _get_radio_weights(batch_physics_spec)
                if weights is not None:
                    operator_for_init = copy.deepcopy(physics)
                    operator_for_init.setWeight(_to_device(weights, compute_device))

            reconstruction_batch = operator_for_init.A_adjoint(measurement)

            if bool(pnp_cfg.get("normalize_adjoint_init", False)):
                peak = torch.abs(reconstruction_batch).amax()
                if torch.isfinite(peak) and peak > 0:
                    reconstruction_batch = (
                        reconstruction_batch * float(pnp_cfg["max_pixel"]) / peak
                    )

            reconstruction_batch = reconstruction_batch.clamp(
                float(pnp_cfg["min_pixel"]),
                float(pnp_cfg["max_pixel"]),
            )

        return reconstruction_batch

    batch_len = (
        int(measurement.shape[0]) if isinstance(measurement, torch.Tensor) else 1
    )
    return torch.zeros(
        (batch_len, *tuple(ground_truth_shape)[1:]),
        device=compute_device,
        dtype=torch.float32,
    )


def producer_component(
    server_info,
    key_prefix,
    stream_spec,
    stream_records=None,
    stream_dataloader=None,
):
    """Workflow component: produce packet stream into SimAI DataStore."""
    _require_simaibench()
    ds = DataStore("producer", server_info=server_info)
    max_packets = int(stream_spec["max_packets"])
    rate_hz = stream_spec.get("rate_hz", None)
    include_ground_truth = bool(stream_spec.get("include_ground_truth", True))
    clamp_generated_measurements = bool(
        stream_spec.get("clamp_generated_measurements", True)
    )

    t0 = time.perf_counter()
    current_physics_key = None
    current_physics = None

    if stream_records is not None:
        source_iter = iter(stream_records)
    elif stream_dataloader is not None:
        source_iter = iter(stream_dataloader)
    else:
        raise ValueError(
            "producer_component expects either stream_records or stream_dataloader."
        )

    try:
        print(
            f"Producer starting with max_packets={max_packets} "
            f"(stream_records={'yes' if stream_records is not None else 'no'}).",
            flush=True,
        )
        for packet_id, sample in enumerate(source_iter):
            print(f"Producing packet {packet_id}...", flush=True)
            if packet_id >= max_packets:
                break

            if rate_hz is not None and float(rate_hz) > 0:
                target_t = t0 + (packet_id / float(rate_hz))
                wait_s = target_t - time.perf_counter()
                if wait_s > 0:
                    time.sleep(wait_s)

            sample_physics_spec = _sample_to_physics_spec(sample)
            measurement_path, measurement_format = _sample_measurement_path(sample)

            x_true_raw = None
            measurement = None
            if measurement_path is None or include_ground_truth:
                x_true_raw, _ = _sample_to_image_and_spec(sample)

            packet = {
                "packet_id": packet_id,
                "t_source": time.perf_counter(),
                "x_true": (
                    _clone_payload(x_true_raw)
                    if (include_ground_truth and x_true_raw is not None)
                    else None
                ),
                "physics_spec": sample_physics_spec,
            }

            if measurement_path is not None:
                packet["measurement_path"] = measurement_path
                packet["measurement_format"] = measurement_format
                try:
                    packet["nbytes"] = int(Path(measurement_path).stat().st_size)
                except OSError:
                    packet["nbytes"] = 0
            else:
                measurement = _sample_inline_measurement(sample)
                if measurement is None:
                    if x_true_raw is None:
                        x_true_raw, _ = _sample_to_image_and_spec(sample)
                    physics_key = _physics_spec_key(sample_physics_spec)
                    if current_physics is None or physics_key != current_physics_key:
                        current_physics = _build_physics_from_spec(
                            ground_truth_shape=tuple(x_true_raw.shape),
                            physics_spec=sample_physics_spec,
                            compute_device=torch.device("cpu"),
                        )
                        current_physics_key = physics_key

                    with torch.no_grad():
                        measurement = current_physics(x_true_raw)
                        if clamp_generated_measurements:
                            measurement = measurement.clamp(0.0, 1.0)
                else:
                    measurement = measurement.detach().clone()

                packet["y"] = _clone_payload(measurement)
                packet["nbytes"] = payload_nbytes(measurement)

            ds.stage_write(packet_key(key_prefix, packet_id), packet)
    except Exception as exc:
        ds.stage_write(
            error_key(key_prefix),
            {
                "component": "producer",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        print("Producer finished producing packets, writing EOS signal...", flush=True)
        ds.stage_write(eos_key(key_prefix), True)


def pnp_consumer_component(
    server_info,
    key_prefix,
    physics_spec,
    ground_truth_shape,
    pnp_cfg,
):
    """Workflow component: consume packets, run PnP, and stage result."""
    _require_simaibench()
    ds = DataStore("consumer", server_info=server_info)

    try:
        requested_device = str(pnp_cfg.get("device", "cpu")).lower()
        if requested_device == "cuda":
            visible_before = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
            if not visible_before:
                inferred_gpus, source_env = _infer_visible_gpus_from_env()
                if inferred_gpus is None:
                    inferred_gpus = _infer_visible_gpus_from_nvidia_smi()
                    source_env = "nvidia-smi" if inferred_gpus is not None else None
                if inferred_gpus is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = inferred_gpus
                    print(
                        "CUDA_VISIBLE_DEVICES was empty; recovered it from "
                        f"{source_env}: '{inferred_gpus}'",
                        flush=True,
                    )

            if not torch.cuda.is_available():
                visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
                slurm_step = str(os.environ.get("SLURM_STEP_GPUS", ""))
                slurm_job = str(os.environ.get("SLURM_JOB_GPUS", ""))
                available = str(os.environ.get("AVAILABLE_GPUS", ""))
                raise RuntimeError(
                    "PnP consumer requested CUDA but torch.cuda.is_available() is False. "
                    "Ensure SimAI-Bench components are provisioned with GPUs "
                    "(e.g. solver ngpus=1) and CUDA is visible in worker processes. "
                    f"CUDA_VISIBLE_DEVICES='{visible}', "
                    f"SLURM_STEP_GPUS='{slurm_step}', "
                    f"SLURM_JOB_GPUS='{slurm_job}', "
                    f"AVAILABLE_GPUS='{available}'."
                )
            compute_device = torch.device("cuda")
        else:
            compute_device = torch.device("cpu")
        print(f"Consumer compute device: {compute_device}", flush=True)

        base_physics_spec = dict(physics_spec or {})
        base_physics_spec.setdefault(
            "physics_mode", str(pnp_cfg.get("physics_mode", "blur"))
        )
        mode = _detect_physics_mode(base_physics_spec)
        physics = None
        active_physics_spec = dict(base_physics_spec)
        active_physics_key = None

        print("Loading denoiser for PnP prior...", flush=True)
        denoiser = _build_denoiser(
            mode=mode,
            pnp_cfg=pnp_cfg,
            ground_truth_shape=ground_truth_shape,
            compute_device=compute_device,
        )
        print("Loaded denoiser", flush=True)
        prior = PnP(denoiser=denoiser)
        data_fidelity = L2()

        reconstruction = torch.zeros(
            tuple(ground_truth_shape),
            device=compute_device,
            dtype=torch.float32,
        )
        consumed_packets = 0
        consumed_batches = 0
        consumed_bytes = 0
        first_consume_t = None
        last_consume_t = None
        packet_id = 0
        batch_size = max(1, int(pnp_cfg.get("batch_size", 1)))
        batch_wait_s = max(0.0, float(pnp_cfg.get("batch_wait_s", 0.0)))
        poll_interval_s = max(0.0, float(pnp_cfg["poll_interval_s"]))

        t_start = time.perf_counter()
        print("Consumer started, waiting for packets...", flush=True)
        while True:
            key = packet_key(key_prefix, packet_id)
            if not ds.poll_staged_data(key):
                if ds.poll_staged_data(eos_key(key_prefix)):
                    break
                time.sleep(poll_interval_s)
                continue

            batch_packets = [ds.stage_read(key)]
            packet_id += 1
            gather_start = time.perf_counter()
            while len(batch_packets) < batch_size:
                next_key = packet_key(key_prefix, packet_id)
                if ds.poll_staged_data(next_key):
                    batch_packets.append(ds.stage_read(next_key))
                    packet_id += 1
                    continue
                if ds.poll_staged_data(eos_key(key_prefix)):
                    break
                if time.perf_counter() - gather_start >= batch_wait_s:
                    break
                time.sleep(poll_interval_s)

            measurement = _concat_payloads(
                [
                    _packet_to_measurement(packet, compute_device)
                    for packet in batch_packets
                ]
            )

            batch_physics_spec = batch_packets[0].get(
                "physics_spec", active_physics_spec
            )
            batch_physics_key = _physics_spec_key(batch_physics_spec)
            if physics is None or batch_physics_key != active_physics_key:
                print(
                    "Building physics for packet "
                    f"{batch_packets[0].get('packet_id', 'unknown')}...",
                    flush=True,
                )
                physics = _build_physics_from_spec(
                    ground_truth_shape=ground_truth_shape,
                    physics_spec=batch_physics_spec,
                    compute_device=compute_device,
                )
                active_physics_spec = dict(batch_physics_spec)
                active_physics_key = batch_physics_key
            print("Starting PnP reconstruction...", flush=True)
            reconstruction_batch = _adjoint_initialization(
                physics=physics,
                measurement=measurement,
                batch_physics_spec=batch_physics_spec,
                pnp_cfg=pnp_cfg,
                compute_device=compute_device,
                ground_truth_shape=ground_truth_shape,
            )

            reconstruction_batch = _run_pnp_updates(
                reconstruction=reconstruction_batch,
                measurement=measurement,
                physics=physics,
                data_fidelity=data_fidelity,
                prior=prior,
                pnp_cfg=pnp_cfg,
            )
            print("Finished PnP reconstruction", flush=True)
            reconstruction = reconstruction_batch[:1]

            now = time.perf_counter()
            if first_consume_t is None:
                first_consume_t = now
            last_consume_t = now
            consumed_packets += len(batch_packets)
            consumed_batches += 1
            consumed_bytes += sum(
                int(packet.get("nbytes", payload_nbytes(packet.get("y", 0))))
                for packet in batch_packets
            )

        t_end = time.perf_counter()
        ds.stage_write(
            result_key(key_prefix),
            {
                "reconstruction": reconstruction.detach().cpu(),
                "trace": {
                    "t_start": t_start,
                    "t_end": t_end,
                    "first_consume_t": first_consume_t,
                    "last_consume_t": last_consume_t,
                    "consumed_packets": consumed_packets,
                    "consumed_batches": consumed_batches,
                    "consumed_bytes": consumed_bytes,
                    "dropped_packets": 0,
                },
            },
        )
    except Exception as exc:
        ds.stage_write(
            error_key(key_prefix),
            {
                "component": "pnp_consumer",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
