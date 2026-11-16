"""Preprocessed radio-interferometry training dataset specification.

The Benchopt dataset deliberately returns paths and configuration only.  The
solver owns the real PyTorch DataLoader so waiting for workers, MS parsing,
operator construction, transfers, forward, and backward all happen in the
timed callback loop.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from astropy.io import fits
from benchopt import BaseDataset

from toolsbench.utils.radio_interferometry import (
    RadioDataConfig,
    discover_radio_entries,
)


class Dataset(BaseDataset):
    name = "radio_interferometry"

    parameters = {
        "input_dir": ["../demo_radio/radio_dataset_example"],
        "sample_id": [None],
        "image_size": [1024],
        "batch_size": [1],
        "num_workers": [0],
        "pin_memory": [False],
        "prefetch_factor": [1],
        "persistent_workers": [False],
        "seed": [0],
        "resize_fits": [False],
        "norm_cache_dir": [None],
        "preprocessing_subdir": ["preprocessing"],
        "sources_extraction": [False],
        "visibility_column": ["DATA"],
        "w_stacking": [False],
        "w_planes": [32],
        "w_binning": ["equal_width"],
        "bin_data": [False],
        "binning_factor": [1.25],
        "nufft_k_oversampling": [1.5],
        "max_visibilities": [None],
        "visibility_chunk_rows": [65536],
        "weighting_chunk_points": [262144],
        "w_phase_row_chunk_size": [256],
        # Whether w-plane operators are split over the solver's inner process
        # group. The solver can override this in coupled topology sweeps.
        "physics_sharding": [False],
        "noise_level": [0.0],
        "imager_verbose": [0],
    }

    def _input_path(self) -> Path:
        path = Path(str(self.input_dir)).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[2] / path).resolve()
        return path

    def prepare(self):
        print(
            "Radio training data is user-provided. Expected each sample directory "
            "to contain one FITS target, one Measurement Set, one metadata JSON, "
            "and preprocessing/{x_init.fits,norm.json}; source_catalog.npz is "
            "also required when sources_extraction=true.",
            flush=True,
        )

    def get_data(self):
        config = RadioDataConfig(
            input_dir=str(self._input_path()),
            sample_id=self.sample_id,
            image_size=int(self.image_size),
            batch_size=int(self.batch_size),
            num_workers=int(self.num_workers),
            pin_memory=bool(self.pin_memory),
            prefetch_factor=int(self.prefetch_factor),
            persistent_workers=bool(self.persistent_workers),
            seed=int(self.seed),
            resize_fits=bool(self.resize_fits),
            norm_cache_dir=self.norm_cache_dir,
            preprocessing_subdir=str(self.preprocessing_subdir),
            sources_extraction=bool(self.sources_extraction),
            visibility_column=str(self.visibility_column),
            w_stacking=bool(self.w_stacking),
            w_planes=int(self.w_planes),
            w_binning=str(self.w_binning),
            bin_data=bool(self.bin_data),
            binning_factor=float(self.binning_factor),
            nufft_k_oversampling=float(self.nufft_k_oversampling),
            max_visibilities=self.max_visibilities,
            visibility_chunk_rows=self.visibility_chunk_rows,
            weighting_chunk_points=self.weighting_chunk_points,
            w_phase_row_chunk_size=int(self.w_phase_row_chunk_size),
            physics_sharding=bool(self.physics_sharding),
            noise_level=float(self.noise_level),
            imager_verbose=int(self.imager_verbose),
        )
        entries = discover_radio_entries(
            config.input_dir,
            image_size=config.image_size,
            preprocessing_subdir=config.preprocessing_subdir,
        )
        if config.sample_id is not None:
            entries = [
                entry for entry in entries if entry.sample_id == config.sample_id
            ]
            if not entries:
                raise ValueError(f"Unknown radio sample_id {config.sample_id!r}.")

        required = ["x_init_path", "norm_path"]
        if config.sources_extraction:
            required.append("source_catalog_path")
        complete_entries = [
            entry
            for entry in entries
            if all(Path(getattr(entry, field)).is_file() for field in required)
        ]
        if config.sample_id is not None and not complete_entries:
            raise FileNotFoundError(
                f"Selected radio sample {config.sample_id!r} is not fully preprocessed."
            )
        entries = complete_entries
        if not entries:
            raise FileNotFoundError(
                "No radio samples with all required data and preprocessing files "
                "were found."
            )

        with fits.open(entries[0].fits_path, memmap=False) as hdul:
            shape = tuple(int(v) for v in hdul[0].shape if int(v) != 1)
        spatial = shape[-2:]
        if spatial != (config.image_size, config.image_size) and not config.resize_fits:
            raise ValueError(
                f"First radio target has shape {spatial}, expected "
                f"{(config.image_size, config.image_size)}."
            )
        return {
            "radio_data_config": asdict(config),
            "ground_truth_shape": (1, 1, config.image_size, config.image_size),
        }
