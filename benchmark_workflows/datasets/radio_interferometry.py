import json
import re
import hashlib
from pathlib import Path

import numpy as np
import torch
from benchopt import BaseDataset
from torch.utils.data import DataLoader, Dataset as TorchDataset


class RadioStreamDataset(TorchDataset):
    """Dataset that loads FITS images and cached radio measurement tensors."""

    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _load_fits_image(image_path):
        from astropy.io import fits

        with fits.open(image_path, memmap=False) as hdul:
            image_np = hdul[0].data.astype(np.float32)
        image = torch.from_numpy(image_np)
        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.ndim == 3:
            image = image.unsqueeze(0)
        else:
            raise ValueError(
                f"Unsupported FITS image shape {tuple(image.shape)} at {image_path}."
            )
        return image

    def __getitem__(self, idx):
        record = self.records[idx]
        sample = {
            "image": self._load_fits_image(record["image_path"]),
            "image_path": record["image_path"],
            "physics_spec": dict(record["physics_spec"]),
        }
        if "measurement" in record:
            sample["measurement"] = record["measurement"].detach().clone()
        if "measurement_path" in record:
            sample["measurement_path"] = record["measurement_path"]
        if "measurement_format" in record:
            sample["measurement_format"] = record["measurement_format"]
        if "measurement_source_path" in record:
            sample["measurement_source_path"] = record["measurement_source_path"]
        return sample


class Dataset(BaseDataset):
    """Radio interferometry stream dataset backed by on-disk FITS/MS files."""

    name = "radio_interferometry_stream"

    parameters = {
        "data_root": [""],
        "image_size": [256],
        "noise_level": [0.01],
        "max_samples": [5],
        "stream_length": [64],
        "rate_hz": [0.0],
        "queue_capacity": [4],
        "drop_policy": ["block"],
        "visibility_column": ["DATA"],
        "seed": [42],
    }

    _sample_dir_pattern = re.compile(
        r"^RA_[\-\+]?\d+(?:\.\d+)?_DEC_[\-\+]?\d+(?:\.\d+)?_size_(\d+)$"
    )

    def get_data(self):
        device = torch.device("cpu")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        records = self._build_stream_records(
            data_root=self._resolve_data_root(),
            buffer_root=self._buffer_root(),
            device=device,
        )
        stream_dataset = RadioStreamDataset(records)
        print(f"Created stream dataset with {len(stream_dataset)} samples.")
        stream_dataloader = DataLoader(
            stream_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
        )
        if len(stream_dataset) == 0:
            raise ValueError(
                "No stream sample selected. Ensure completed samples exist and that "
                "`max_samples` and `stream_length` are >= 1."
            )

        first_sample = stream_dataset[0]
        ground_truth = first_sample["image"].clone()
        physics_spec = dict(first_sample["physics_spec"])

        stream_spec = {
            "rate_hz": (None if self.rate_hz <= 0.0 else float(self.rate_hz)),
            "max_packets": len(stream_dataset),
            "queue_capacity": int(self.queue_capacity),
            "drop_policy": self.drop_policy,
            "include_ground_truth": False,
            # Measurements are precomputed from the MS files.
            "clamp_generated_measurements": False,
        }

        min_pixel = float(torch.amin(ground_truth).item())
        max_pixel = float(torch.amax(ground_truth).item())
        if max_pixel <= min_pixel:
            max_pixel = min_pixel + 1.0

        print("Dataset prepared")

        return dict(
            ground_truth=ground_truth,
            stream_dataloader=stream_dataloader,
            physics_spec=physics_spec,
            stream_spec=stream_spec,
            min_pixel=min_pixel,
            max_pixel=max_pixel,
        )

    def _resolve_data_root(self):
        configured_root = str(self.data_root).strip()
        if configured_root:
            root = Path(configured_root).expanduser()
            if not root.is_absolute():
                benchmark_root = Path(__file__).resolve().parents[1]
                root = (benchmark_root / root).resolve()
            return root
        return Path(__file__).resolve().parents[1] / "data" / "radio_interferometry"

    def _buffer_root(self):
        """Local buffer directory used to cache measurement tensors as .pt files."""
        buffer_root = Path(__file__).resolve().parents[1] / "data" / "radio_interferometry"
        buffer_root.mkdir(parents=True, exist_ok=True)
        return buffer_root

    @staticmethod
    def _single_path(paths, kind, folder):
        if len(paths) != 1:
            raise ValueError(
                f"Expected exactly one {kind} in '{folder}', found {len(paths)}."
            )
        return paths[0]

    @classmethod
    def _parse_sample_folder_size(cls, folder_name):
        match = cls._sample_dir_pattern.match(folder_name)
        if match is None:
            return None
        return int(match.group(1))

    def _discover_completed_samples(self, data_root):
        if not data_root.exists():
            raise FileNotFoundError(
                f"Radio data root does not exist: {data_root}. "
                "Set `data_root` to the folder containing RA_*_DEC_*_size_* folders."
            )

        completed = []
        discarded_incomplete = []

        for sample_dir in sorted((p for p in data_root.iterdir() if p.is_dir()), key=lambda p: p.name):
            parsed_size = self._parse_sample_folder_size(sample_dir.name)
            if parsed_size is None:
                continue

            fits_paths = sorted(
                [
                    *sample_dir.glob("*.fits"),
                    *sample_dir.glob("*.fit"),
                    *sample_dir.glob("*.fts"),
                ]
            )
            json_paths = sorted(sample_dir.glob("*.json"))
            ms_paths = sorted(p for p in sample_dir.glob("*.ms") if p.is_dir())

            has_fits = len(fits_paths) > 0
            has_json = len(json_paths) > 0
            has_ms = len(ms_paths) > 0

            if has_fits and has_json and has_ms:
                completed.append(
                    {
                        "folder": sample_dir,
                        "image_path": self._single_path(fits_paths, "FITS file", sample_dir),
                        "metadata_path": self._single_path(json_paths, "JSON metadata file", sample_dir),
                        "ms_path": self._single_path(ms_paths, "MS folder", sample_dir),
                        "image_size": parsed_size,
                    }
                )
            elif has_fits:
                # Convention: folders with only FITS correspond to simulations not run yet.
                discarded_incomplete.append(sample_dir)

        if len(completed) == 0:
            raise ValueError(
                "No completed radio samples found. Expected folders named "
                "RA_*_DEC_*_size_* containing exactly one FITS, one JSON, and one MS."
            )

        if len(discarded_incomplete) > 0:
            print(
                f"Discarded {len(discarded_incomplete)} incomplete sample folder(s) "
                "(FITS only or missing JSON/MS)."
            )

        return completed

    def _build_stream_records(self, data_root, buffer_root, device):
        from toolsbench.utils.deepinv_imager import DeepinvDirtyImager, DirtyImagerConfig

        max_samples = int(self.max_samples)
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1.")

        requested_stream_length = int(self.stream_length)
        if requested_stream_length < 1:
            raise ValueError("stream_length must be >= 1.")

        completed_samples = self._discover_completed_samples(Path(data_root))
        capped_samples = completed_samples[:max_samples]
        if len(completed_samples) > max_samples:
            print(
                f"Capped completed samples to max_samples={max_samples} "
                f"(from {len(completed_samples)} available)."
            )

        selected_samples = capped_samples[:requested_stream_length]
        buffer_root = Path(buffer_root)
        buffer_root.mkdir(parents=True, exist_ok=True)
        source_hash = hashlib.sha1(str(Path(data_root).resolve()).encode("utf-8")).hexdigest()[:12]
        source_buffer_root = buffer_root / f"source_{source_hash}"
        source_buffer_root.mkdir(parents=True, exist_ok=True)

        if len(selected_samples) < requested_stream_length:
            print(
                f"Requested stream_length={requested_stream_length}, but only "
                f"{len(selected_samples)} completed sample(s) are available."
            )

        records = []
        for stream_idx, sample in enumerate(selected_samples):
            image_path = sample["image_path"]
            metadata_path = sample["metadata_path"]
            ms_path = sample["ms_path"]
            sample_buffer_dir = source_buffer_root / sample["folder"].name
            sample_buffer_dir.mkdir(parents=True, exist_ok=True)

            cache_name = f"measurement_{str(self.visibility_column).lower()}.pt"
            measurement_cache_path = sample_buffer_dir / cache_name
            physics_payload_path = sample_buffer_dir / "physics_payload.pt"

            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            if "imaging_cellsize" not in metadata:
                raise KeyError(
                    f"Missing 'imaging_cellsize' in metadata file: {metadata_path}"
                )

            imaging_npixel = int(sample["image_size"] or self.image_size)
            imager_config = DirtyImagerConfig(
                imaging_npixel=imaging_npixel,
                imaging_cellsize=float(metadata["imaging_cellsize"]),
                combine_across_frequencies=False,
            )
            if measurement_cache_path.exists() and physics_payload_path.exists():
                print(
                    f"Using cached measurement and physics payload for sample {stream_idx} "
                    f"from {sample_buffer_dir}."
                )
            else:
                print(
                    "Creating imager and physics for sample",
                    stream_idx,
                    "with image size",
                    imaging_npixel,
                    "x",
                    imaging_npixel,
                )
                imager = DeepinvDirtyImager(imager_config, device=device)
                physics, measurements, weights = imager.create_deepinv_physics(
                    visibility_path=str(ms_path),
                    visibility_format="MS",
                    visibility_column=str(self.visibility_column),
                    bin_data=False,
                )
                print(
                    f"Caching measurements for sample {stream_idx} at {measurement_cache_path}..."
                )
                torch.save(measurements.detach().cpu(), measurement_cache_path)
                print(
                    f"Cached measurements shape: {measurements.shape}, dtype: {measurements.dtype}"
                )
                torch.save(
                    {
                        "samples_locs": physics.samples_loc.detach().cpu(),
                        "weights": weights.detach().cpu(),
                    },
                    physics_payload_path,
                )

            records.append(
                {
                    "image_path": str(image_path),
                    "measurement_path": str(measurement_cache_path),
                    "measurement_format": "torch",
                    "measurement_source_path": str(ms_path),
                    "physics_spec": {
                        "physics_mode": "radio",
                        "physics_id": sample["folder"].name,
                        "stream_index": int(stream_idx),
                        "physics_payload_path": str(physics_payload_path),
                        "k_oversampling": float(imager_config.nufft_k_oversampling),
                        "noise_level": float(self.noise_level),
                        "seed": int(self.seed),
                    },
                }
            )

        return records
