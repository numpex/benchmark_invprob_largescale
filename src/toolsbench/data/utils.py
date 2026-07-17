"""Common utilities for downloading data files."""

from __future__ import annotations

from pathlib import Path

import requests

_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB
_PROGRESS_STEP = 128 * 1024 * 1024  # 128 MiB


def download_file(
    url: str,
    cache_path: Path | str,
    *,
    chunk_size: int = _CHUNK_SIZE,
    progress_step: int = _PROGRESS_STEP,
) -> Path:
    """Download *url* to *cache_path* with progress reporting.

    Downloads to a temporary ``.part`` file and atomically renames on
    success, so interrupted downloads never leave a corrupt file.

    Parameters
    ----------
    url:
        Source URL.
    cache_path:
        Destination path (parent directory is created if needed).
    chunk_size:
        Streaming chunk size in bytes.
    progress_step:
        Minimum bytes between progress log lines.

    Returns
    -------
    Path
        Absolute path to the downloaded file.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"\nDownloading {cache_path.name} to {cache_path}", flush=True)
    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            if total:
                print(f"Expected download size: {_format_bytes(total)}", flush=True)

            downloaded = 0
            next_report = _next_report_threshold(total, progress_step)
            with tmp_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        print(_format_progress(downloaded, total), flush=True)
                        next_report += _next_report_threshold(total, progress_step)

        tmp_path.replace(cache_path)
        print(f"Downloaded {cache_path.name}: {_format_bytes(downloaded)}", flush=True)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download {url} to {cache_path}.") from exc

    return cache_path


def _next_report_threshold(total: int, progress_step: int = _PROGRESS_STEP) -> int:
    if total <= 0:
        return progress_step
    return max(total // 20, progress_step)


def _format_progress(downloaded: int, total: int) -> str:
    if total <= 0:
        return f"Downloaded {_format_bytes(downloaded)}"
    pct = 100.0 * downloaded / total
    return (
        f"Downloaded {_format_bytes(downloaded)} / "
        f"{_format_bytes(total)} ({pct:.1f}%)"
    )


def _format_bytes(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    for unit in units[:-1]:
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} {units[-1]}"
