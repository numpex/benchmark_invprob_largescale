from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_config_json(path: Path, cfg_dict: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)


def append_metrics_row(path: Path | str, row: dict) -> None:
    """Append one row to a CSV file, writing a header on first write."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class PerfProbe:
    """Context manager that measures wall time and peak GPU memory for a code block.

    """
    def __enter__(self) -> "PerfProbe":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_s: float = time.perf_counter() - self._t0
        self.peak_mb: float = (
            torch.cuda.max_memory_allocated() / 1e6
            if torch.cuda.is_available() else 0.0
        )

