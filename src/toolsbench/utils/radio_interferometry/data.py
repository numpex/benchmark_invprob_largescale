from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table
from deepinv.utils import TensorList
from torch.utils.data import DataLoader, Dataset, Sampler

from .sources import SourceCatalog, load_source_catalog, remove_gt_sources


@dataclass(frozen=True)
class RadioEntry:
    sample_id: str
    sample_dir: str
    fits_path: str
    ms_path: str
    metadata_path: str
    imaging_npixel: int
    imaging_cellsize: float
    norm_path: str
    x_init_path: str
    source_catalog_path: str


@dataclass
class RadioDataConfig:
    input_dir: str = "/lustre/fswork/projects/rech/fio/commun/dataset_radio_1024"
    sample_id: str | None = None
    image_size: int = 1024
    batch_size: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    seed: int = 0
    save_fits: bool = False
    resize_fits: bool = False
    norm_cache_dir: str | None = None
    preprocessing_subdir: str = "preprocessing"
    sources_extraction: bool = False
    visibility_column: str = "DATA"
    w_stacking: bool = True
    w_planes: int = 32
    w_binning: str = "equal_width"
    bin_data: bool = False
    binning_factor: float = 1.25
    nufft_k_oversampling: float = 1.5
    max_visibilities: int | None = None
    visibility_chunk_rows: int | None = 65536
    weighting_chunk_points: int | None = 262144
    w_phase_row_chunk_size: int = 256
    physics_sharding: bool = False
    shard_rank: int = 0
    shard_world_size: int = 1
    noise_level: float = 0.0
    imager_verbose: int = 0


def _physics_leaves(physics) -> list:
    if physics is None:
        return []
    if isinstance(physics, (list, tuple)):
        return list(physics)
    return list(getattr(physics, "physics_list", [physics]))


@dataclass
class RadioBatch:
    """One fully prepared radio sample, optionally resident on an accelerator."""

    x: torch.Tensor
    x_diffuse: torch.Tensor
    x_init: torch.Tensor
    physics: Any
    measurements: torch.Tensor | TensorList | None
    weights: torch.Tensor | None
    physics_global_indices: list[int] | None
    num_physics_operators: int
    global_measurement_count: int
    physics_from_shard: bool
    source_catalog: SourceCatalog | None
    lipschitz: float | None
    sample_index: int
    sample_id: str
    fits_path: str
    ms_path: str
    min_pixel: float
    max_pixel: float

    def pin_memory(self) -> "RadioBatch":
        shared_diffuse = self.x_diffuse is self.x
        for name in ("x", "x_init"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.pin_memory())
        if shared_diffuse:
            self.x_diffuse = self.x
        elif self.x_diffuse is not None:
            self.x_diffuse = self.x_diffuse.pin_memory()
        if isinstance(self.measurements, TensorList):
            self.measurements = TensorList(
                [tensor.pin_memory() for tensor in self.measurements]
            )
        elif self.measurements is not None:
            self.measurements = self.measurements.pin_memory()
        if self.source_catalog is not None:
            self.source_catalog = self.source_catalog.pin_memory()
        for physics in _physics_leaves(self.physics):
            physics._apply(lambda tensor: tensor.pin_memory())
        return self

    def to(self, device: torch.device, non_blocking: bool = False) -> "RadioBatch":
        def move(tensor):
            return (
                None
                if tensor is None
                else tensor.to(device=device, non_blocking=non_blocking)
            )

        shared_diffuse = self.x_diffuse is self.x
        self.x = move(self.x)
        self.x_diffuse = self.x if shared_diffuse else move(self.x_diffuse)
        self.x_init = move(self.x_init)
        self.measurements = move(self.measurements)
        # The physics and measurements already contain sqrt(weights). Keep the
        # original statistical weights as CPU metadata for diagnostics only.
        if self.source_catalog is not None:
            self.source_catalog = self.source_catalog.to(
                device=device, non_blocking=non_blocking
            )
        # deepinv's StackedPhysics stores its leaves in a plain list rather
        # than registered submodules, so ``stacked.to(...)`` alone would leave
        # the per-plane NUFFT coordinates on CPU.
        for physics in _physics_leaves(self.physics):
            physics.to(device=device, non_blocking=non_blocking)
        return self


@dataclass
class RadioDataBundle:
    loader: DataLoader
    sampler: Sampler[int]
    dataset: "RadioInterferometryDataset"
    ground_truth_shape: tuple[int, ...]
    min_pixel: float
    max_pixel: float


@dataclass
class RadioInferenceBundle:
    loader: DataLoader
    sampler: Sampler[int]
    dataset: "RadioInterferometryDataset"
    ground_truth_shape: tuple[int, ...]


class RadioSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, shuffle: bool, seed: int = 0) -> None:
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        order = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(order)
        return iter(order)


class RadioInterferometryDataset(Dataset):
    def __init__(
        self,
        entries: list[RadioEntry],
        image_size: int = 1024,
        resize_fits: bool = False,
        norm_cache: dict[str, float] | None = None,
        sources_extraction: bool = False,
        visibility_column: str = "DATA",
        w_stacking: bool = True,
        w_planes: int = 32,
        w_binning: str = "equal_width",
        bin_data: bool = False,
        binning_factor: float = 1.25,
        nufft_k_oversampling: float = 1.5,
        max_visibilities: int | None = None,
        visibility_chunk_rows: int | None = 65536,
        weighting_chunk_points: int | None = 262144,
        w_phase_row_chunk_size: int = 256,
        physics_sharding: bool = False,
        shard_rank: int = 0,
        shard_world_size: int = 1,
        noise_level: float = 0.0,
        imager_verbose: int = 0,
    ) -> None:
        self.entries = list(entries)
        self.image_size = int(image_size)
        self.resize_fits = bool(resize_fits)
        self.norm_cache: dict[str, float] = dict(norm_cache or {})
        self.sources_extraction = bool(sources_extraction)
        self.visibility_column = str(visibility_column)
        self.w_stacking = bool(w_stacking)
        self.w_planes = int(w_planes)
        self.w_binning = str(w_binning).lower()
        if self.w_binning not in {"equal_width", "equal_count"}:
            raise ValueError(
                "w_binning must be either 'equal_width' or 'equal_count', "
                f"got {self.w_binning!r}."
            )
        self.bin_data = bool(bin_data)
        self.binning_factor = float(binning_factor)
        self.nufft_k_oversampling = float(nufft_k_oversampling)
        self.max_visibilities = max_visibilities
        self.visibility_chunk_rows = visibility_chunk_rows
        self.weighting_chunk_points = (
            None if weighting_chunk_points is None else int(weighting_chunk_points)
        )
        self.w_phase_row_chunk_size = int(w_phase_row_chunk_size)
        self.physics_sharding = bool(physics_sharding)
        self.shard_rank = int(shard_rank)
        self.shard_world_size = int(shard_world_size)
        if self.physics_sharding and not self.w_stacking:
            raise ValueError("physics_sharding requires w_stacking=True.")
        if (
            self.shard_world_size <= 0
            or not 0 <= self.shard_rank < self.shard_world_size
        ):
            raise ValueError(
                f"Invalid physics shard rank {self.shard_rank}/{self.shard_world_size}."
            )
        self.noise_level = float(noise_level)
        self.imager_verbose = int(imager_verbose)

        for entry in self.entries:
            if entry.sample_id in self.norm_cache:
                continue
            cached_norm = load_norm_file(entry.norm_path)
            if cached_norm is not None:
                self.norm_cache[entry.sample_id] = cached_norm

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> RadioBatch:
        from .physics import create_radio_physics

        entry = self.entries[idx]
        t0 = time.perf_counter()
        x = load_fits_tensor(
            entry.fits_path,
            image_size=self.image_size,
            resize=self.resize_fits,
        )
        x_init = load_fits_tensor(entry.x_init_path, image_size=self.image_size)
        source_catalog = None
        if self.sources_extraction:
            source_catalog = load_source_catalog(entry.source_catalog_path)

        result = create_radio_physics(
            ms_path=entry.ms_path,
            imaging_npixel=entry.imaging_npixel,
            imaging_cellsize=entry.imaging_cellsize,
            device=torch.device("cpu"),
            visibility_column=self.visibility_column,
            w_stacking=self.w_stacking,
            w_planes=self.w_planes,
            w_binning=self.w_binning,
            bin_data=self.bin_data,
            binning_factor=self.binning_factor,
            nufft_k_oversampling=self.nufft_k_oversampling,
            max_visibilities=self.max_visibilities,
            visibility_chunk_rows=self.visibility_chunk_rows,
            weighting_chunk_points=self.weighting_chunk_points,
            w_phase_row_chunk_size=self.w_phase_row_chunk_size,
            shard_rank=self.shard_rank if self.physics_sharding else None,
            shard_world_size=(self.shard_world_size if self.physics_sharding else None),
            noise_level=self.noise_level,
            verbose=self.imager_verbose,
        )
        elapsed = time.perf_counter() - t0
        print(
            f"[{time.strftime('%H:%M:%S')}] [dataset] prepared idx={idx} "
            f"id={entry.sample_id} ({elapsed:.1f}s) "
            f"local_vis={_measurement_numel(result.measurements)} "
            f"global_vis={result.global_measurement_count} "
            f"planes={result.global_indices}/{result.num_operators}",
            flush=True,
        )
        x_diffuse = (
            remove_gt_sources(x, kernel_size=7, threshold=0.0)
            if self.sources_extraction
            else x
        )
        return RadioBatch(
            x=x,
            x_diffuse=x_diffuse,
            x_init=x_init,
            physics=result.physics,
            measurements=result.measurements,
            weights=result.weights,
            physics_global_indices=result.global_indices,
            num_physics_operators=result.num_operators,
            global_measurement_count=result.global_measurement_count,
            physics_from_shard=result.from_shard,
            source_catalog=source_catalog,
            lipschitz=self.norm_cache.get(entry.sample_id),
            sample_index=idx,
            sample_id=entry.sample_id,
            fits_path=entry.fits_path,
            ms_path=entry.ms_path,
            min_pixel=float(x.min().item()),
            max_pixel=float(x.max().item()),
        )


def _measurement_numel(measurements: torch.Tensor | TensorList | None) -> int:
    if measurements is None:
        return 0
    if isinstance(measurements, TensorList):
        return sum(tensor.numel() for tensor in measurements)
    return measurements.numel()


def load_fits_tensor(
    fits_path: str | Path,
    image_size: int | None = None,
    resize: bool = False,
) -> torch.Tensor:
    with fits.open(fits_path, memmap=False) as hdul:
        img_np = np.array(hdul[0].data, dtype=np.float32, copy=True)

    if not img_np.dtype.isnative:
        img_np = img_np.byteswap().view(img_np.dtype.newbyteorder("="))

    img_np = np.nan_to_num(img_np, nan=0.0, posinf=0.0, neginf=0.0)
    img_np = np.squeeze(img_np)

    if img_np.ndim == 2:
        img_np = img_np[np.newaxis, ...]
    elif img_np.ndim != 3:
        raise ValueError(
            f"Unexpected FITS image shape {img_np.shape}; expected 2-D or C,H,W."
        )

    if image_size is not None:
        _, h, w = img_np.shape
        if (h, w) != (int(image_size), int(image_size)):
            if not resize:
                raise ValueError(
                    f"{fits_path} has spatial shape {(h, w)}, expected "
                    f"{(image_size, image_size)}. Set resize_fits=true to resize."
                )
            from scipy.ndimage import zoom

            factors = (1, int(image_size) / h, int(image_size) / w)
            img_np = zoom(img_np, factors, order=3)

    img_np = np.ascontiguousarray(img_np, dtype=np.float32)
    return torch.from_numpy(img_np)


def load_fits_catalog(
    catalog_path: str | Path,
) -> torch.Tensor:
    catalog = Table.read(catalog_path)
    return catalog


def discover_radio_entries(
    input_dir: str | Path,
    image_size: int = 1024,
    preprocessing_subdir: str = "preprocessing",
) -> list[RadioEntry]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Radio dataset directory not found: {root}")

    entries: list[RadioEntry] = []
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        fits_files = sorted(sample_dir.glob("*.fits"))
        ms_dirs = [
            d for d in sorted(sample_dir.glob("*.ms")) if (d / "table.dat").exists()
        ]
        json_files = sorted(sample_dir.glob("*.json"))
        if not fits_files or not ms_dirs or not json_files:
            continue

        metadata_path = _choose_metadata(json_files)
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read radio metadata {metadata_path}: {exc}"
            ) from exc

        imaging_npixel = int(metadata.get("imaging_npixel", image_size))
        imaging_cellsize = float(metadata["imaging_cellsize"])
        preprocessing_dir = sample_dir / preprocessing_subdir

        entries.append(
            RadioEntry(
                sample_id=sample_dir.name,
                sample_dir=str(sample_dir),
                fits_path=str(fits_files[0]),
                ms_path=str(ms_dirs[0]),
                metadata_path=str(metadata_path),
                imaging_npixel=imaging_npixel,
                imaging_cellsize=imaging_cellsize,
                norm_path=str(preprocessing_dir / "norm.json"),
                x_init_path=str(preprocessing_dir / "x_init.fits"),
                source_catalog_path=str(preprocessing_dir / "source_catalog.npz"),
            )
        )

    if not entries:
        raise FileNotFoundError(
            f"No radio samples found in {root}. Expected folders containing .fits, .ms and .json."
        )
    return entries


def _choose_metadata(paths: list[Path]) -> Path:
    for path in paths:
        if path.name.endswith(".meta.json"):
            return path
    return paths[0]


def load_norm_cache(cache_dir: str | Path) -> dict[str, float]:
    """Load all cached operator norms from *cache_dir*.

    Each JSON file is expected to have at least a ``"lipschitz"`` key.
    Returns a mapping from *sample_id* (= filename stem) to the Lipschitz
    constant.  Missing or malformed files are silently skipped.
    """
    root = Path(cache_dir)
    cache: dict[str, float] = {}
    if not root.exists():
        return cache
    for p in root.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            lip = float(data["lipschitz"])
            if lip > 0:
                cache[p.stem] = lip
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass  # skip corrupted / incomplete files
    return cache


def load_norm_file(path: str | Path) -> float | None:
    """Load one preprocessing norm, returning ``None`` when it is absent."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            lipschitz = float(json.load(stream)["lipschitz"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid preprocessing norm file {path}: {exc}") from exc
    if lipschitz <= 0:
        raise ValueError(f"Invalid non-positive Lipschitz constant in {path}")
    return lipschitz


def _select_sample_entry(
    entries: list[RadioEntry], sample_id: str | None
) -> list[RadioEntry]:
    if sample_id is None:
        return entries
    selected = [entry for entry in entries if entry.sample_id == sample_id]
    if not selected:
        raise ValueError(f"Unknown sample_id: {sample_id}")
    return selected


def _collate_radio(batch: list[RadioBatch]) -> RadioBatch:
    if len(batch) != 1:
        raise ValueError(
            "Radio samples have per-sample physics and variable visibility sizes; use batch_size=1."
        )
    item = batch[0]
    item.x = item.x.unsqueeze(0)
    item.x_init = item.x_init.unsqueeze(0)
    return item


def _radio_worker_init(_worker_id: int) -> None:
    # Workers build CPU-only radio operators. Hide accelerators before the lazy
    # DeepInv/NUFFT imports in Dataset.__getitem__ so workers cannot create CUDA
    # contexts and consume training VRAM.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)


def _build_loader(
    dataset: RadioInterferometryDataset,
    sampler: Sampler[int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
) -> DataLoader:
    if int(batch_size) != 1:
        raise ValueError("Radio demo currently supports batch_size=1 only.")
    kwargs = {
        "dataset": dataset,
        "batch_size": 1,
        "sampler": sampler,
        "drop_last": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": _collate_radio,
        "worker_init_fn": _radio_worker_init,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
        # CUDA is initialized in the training parent before iteration. Spawned
        # workers stay CPU-only and avoid inheriting an unusable CUDA runtime.
        kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(**kwargs)


def build_radio_dataloaders(config: RadioDataConfig, ctx=None) -> RadioDataBundle:
    print(
        f"[data] discovering samples input_dir={config.input_dir} "
        f"selected_sample={config.sample_id!r}",
        flush=True,
    )
    started = time.perf_counter()
    entries = discover_radio_entries(
        config.input_dir,
        image_size=config.image_size,
        preprocessing_subdir=config.preprocessing_subdir,
    )
    print(
        f"[data] discovery completed in {time.perf_counter() - started:.1f}s "
        f"candidates={len(entries)}",
        flush=True,
    )
    entries = _select_sample_entry(entries, config.sample_id)

    print(f"[data] checking preprocessing for {len(entries)} sample(s)...", flush=True)
    available_entries: list[RadioEntry] = []
    for entry in entries:
        required = [entry.norm_path, entry.x_init_path]
        if config.sources_extraction:
            required.append(entry.source_catalog_path)
        missing = [path for path in required if not Path(path).is_file()]
        if missing and config.sample_id is not None:
            raise FileNotFoundError(
                f"Selected sample {entry.sample_id} is missing preprocessing "
                f"file(s): {', '.join(missing)}"
            )
        if missing:
            print(
                f"[data] skipping sample={entry.sample_id}; missing preprocessing "
                f"file(s): {', '.join(missing)}",
                flush=True,
            )
        else:
            available_entries.append(entry)

    entries = available_entries
    if not entries:
        raise FileNotFoundError(
            "Training requires preprocessed x_init and norm files"
            + (" and source catalogues" if config.sources_extraction else "")
            + ", but no samples with complete preprocessing were found."
        )
    print("[data] preprocessing check complete.", flush=True)

    norm_cache: dict[str, float] = {}
    if config.norm_cache_dir is not None:
        norm_cache = load_norm_cache(config.norm_cache_dir)
    for entry in entries:
        preprocessing_norm = load_norm_file(entry.norm_path)
        if preprocessing_norm is not None:
            norm_cache[entry.sample_id] = preprocessing_norm
    n_cached = sum(1 for entry in entries if entry.sample_id in norm_cache)
    print(
        f"[data] operator norms cached for {n_cached}/{len(entries)} samples "
        "(per-sample preprocessing norms take precedence)",
        flush=True,
    )

    print("[data] constructing dataset...", flush=True)
    dataset = RadioInterferometryDataset(
        entries,
        image_size=config.image_size,
        resize_fits=config.resize_fits,
        norm_cache=norm_cache,
        sources_extraction=config.sources_extraction,
        visibility_column=config.visibility_column,
        w_stacking=config.w_stacking,
        w_planes=config.w_planes,
        w_binning=config.w_binning,
        bin_data=config.bin_data,
        binning_factor=config.binning_factor,
        nufft_k_oversampling=config.nufft_k_oversampling,
        max_visibilities=config.max_visibilities,
        visibility_chunk_rows=config.visibility_chunk_rows,
        weighting_chunk_points=config.weighting_chunk_points,
        w_phase_row_chunk_size=config.w_phase_row_chunk_size,
        physics_sharding=config.physics_sharding,
        shard_rank=config.shard_rank,
        shard_world_size=config.shard_world_size,
        noise_level=config.noise_level,
        imager_verbose=config.imager_verbose,
    )
    if ctx is not None and int(ctx.dp_world_size) > 1:
        sampler = ctx.distributed_data_sampler(dataset, shuffle=True, seed=config.seed)
    else:
        sampler = RadioSampler(dataset, shuffle=True, seed=config.seed)

    print(
        f"[data] samples={len(dataset)}",
        flush=True,
    )

    loader = _build_loader(
        dataset,
        sampler,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
    )
    return RadioDataBundle(
        loader=loader,
        sampler=sampler,
        dataset=dataset,
        ground_truth_shape=(1, 1, config.image_size, config.image_size),
        # Per-sample ranges are loaded with each RadioBatch.
        min_pixel=0.0,
        max_pixel=1.0,
    )


def build_radio_inference_dataloader(
    config: RadioDataConfig,
) -> RadioInferenceBundle:
    """Build one deterministic loader over every selected, preprocessed sample."""
    entries = discover_radio_entries(
        config.input_dir,
        image_size=config.image_size,
        preprocessing_subdir=config.preprocessing_subdir,
    )
    entries = _select_sample_entry(entries, config.sample_id)

    missing_by_sample: dict[str, list[str]] = {}
    norm_cache: dict[str, float] = {}
    if config.norm_cache_dir is not None:
        norm_cache.update(load_norm_cache(config.norm_cache_dir))
    for entry in entries:
        required = [entry.norm_path, entry.x_init_path]
        if config.sources_extraction:
            required.append(entry.source_catalog_path)
        missing = [path for path in required if not Path(path).is_file()]
        if missing:
            missing_by_sample[entry.sample_id] = missing
            continue
        preprocessing_norm = load_norm_file(entry.norm_path)
        if preprocessing_norm is not None:
            norm_cache[entry.sample_id] = preprocessing_norm

    if missing_by_sample:
        details = "; ".join(
            f"{sample_id}: {', '.join(paths)}"
            for sample_id, paths in missing_by_sample.items()
        )
        raise FileNotFoundError(
            "Inference requires complete preprocessing for every sample. "
            f"Missing files: {details}"
        )

    dataset = RadioInterferometryDataset(
        entries,
        image_size=config.image_size,
        resize_fits=config.resize_fits,
        norm_cache=norm_cache,
        sources_extraction=config.sources_extraction,
        visibility_column=config.visibility_column,
        w_stacking=config.w_stacking,
        w_planes=config.w_planes,
        w_binning=config.w_binning,
        bin_data=config.bin_data,
        binning_factor=config.binning_factor,
        nufft_k_oversampling=config.nufft_k_oversampling,
        max_visibilities=config.max_visibilities,
        visibility_chunk_rows=config.visibility_chunk_rows,
        weighting_chunk_points=config.weighting_chunk_points,
        w_phase_row_chunk_size=config.w_phase_row_chunk_size,
        physics_sharding=config.physics_sharding,
        shard_rank=config.shard_rank,
        shard_world_size=config.shard_world_size,
        noise_level=config.noise_level,
        imager_verbose=config.imager_verbose,
    )
    sampler = RadioSampler(dataset, shuffle=False, seed=config.seed)
    loader = _build_loader(
        dataset,
        sampler,
        batch_size=1,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
    )
    x0 = load_fits_tensor(
        entries[0].fits_path,
        image_size=config.image_size,
        resize=config.resize_fits,
    )
    return RadioInferenceBundle(
        loader=loader,
        sampler=sampler,
        dataset=dataset,
        ground_truth_shape=(1,) + tuple(x0.shape),
    )
