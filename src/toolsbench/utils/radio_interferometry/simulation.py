"""Legacy simulation cache and pointing helpers.

Training consumes preprocessed datasets directly; these functions remain for
the separate inference preparation path.
"""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import astropy.units as u
import numpy as np
import yaml
from astropy.coordinates import AltAz, EarthLocation, ICRS, SkyCoord
from astropy.time import Time

MEERKAT_LOCATION = EarthLocation(
    lat=-30.83 * u.deg, lon=21.33 * u.deg, height=1195.0 * u.m
)


def get_meerkat_visibilities_path(
    image: np.ndarray,
    cache_dir: Path,
    fits_file: str | Path,
    imaging_npixel: int,
    number_of_time_steps: int = 256,
    start_frequency_hz: float = 100e6,
    end_frequency_hz: float = 120e6,
    number_of_channels: int = 12,
    pos_ra: float = 155.66367,
    pos_dec: float = -30.7130,
    random_position: bool = False,
    add_noise: bool = False,
    pol_mode: str = "Full",
    use_gpus: bool = False,
):
    """Return the deterministic Measurement Set cache path."""
    del image  # Retained in the signature for the inference preparation API.
    params = {
        "fits_name": Path(fits_file).name,
        "number_of_time_steps": number_of_time_steps,
        "start_frequency_hz": start_frequency_hz,
        "end_frequency_hz": end_frequency_hz,
        "number_of_channels": number_of_channels,
        "pos_ra": pos_ra,
        "pos_dec": pos_dec,
        "random_position": random_position,
        "add_noise": add_noise,
        "pol_mode": pol_mode,
        "use_gpus": use_gpus,
        "imaging_npixel": imaging_npixel,
    }
    params_hash = hashlib.md5(str(sorted(params.items())).encode()).hexdigest()
    fits_hash = hashlib.md5()
    with Path(fits_file).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            fits_hash.update(chunk)
    full_hash = hashlib.md5((params_hash + fits_hash.hexdigest()).encode()).hexdigest()
    return Path(cache_dir) / f"{full_hash}.ms"


def _load_object(values):
    return types.SimpleNamespace(**values)


def load_config(config_path, section=None):
    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
        config = json.loads(json.dumps(config), object_hook=_load_object)
    if section is not None:
        if not hasattr(config, section):
            raise KeyError(f"Section {section!r} not found in config: {config_path}")
        return getattr(config, section)
    return config


def is_source_visible(
    ra_deg,
    dec_deg,
    obs_start_time,
    obs_duration,
    telescope_location,
    min_elevation_deg=15.0,
    n_time_samples=10,
):
    """Return whether a source stays above the elevation limit."""
    source = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    time_samples = [
        obs_start_time + i * obs_duration / (n_time_samples - 1)
        for i in range(n_time_samples)
    ]
    for value in time_samples:
        frame = AltAz(obstime=Time(value), location=telescope_location)
        if source.transform_to(frame).alt.deg < min_elevation_deg:
            return False
    return True


def draw_random_pointing(
    time: Time,
    observer: EarthLocation = MEERKAT_LOCATION,
    min_elevation_deg: float = 15.0,
    max_attempts: int = 1000,
    n_azimuth_samples: int = 360,
) -> tuple[float, float]:
    """Draw a random visible pointing at a given time and location."""
    azimuth = np.linspace(0, 360, n_azimuth_samples)
    boundary = SkyCoord(
        azimuth * u.deg,
        np.full(azimuth.size, min_elevation_deg) * u.deg,
        frame=AltAz(obstime=time, location=observer),
    ).transform_to(ICRS)
    ra = boundary.ra.deg
    dec = boundary.dec.deg
    dec_min, dec_max = np.min(dec), np.max(dec)
    ra_min, ra_max = (0, 360) if np.ptp(ra) > 180 else (np.min(ra), np.max(ra))

    for _ in range(max_attempts):
        ra_candidate = ra_min + np.random.rand() * (ra_max - ra_min)
        dec_candidate = dec_min + np.random.rand() * (dec_max - dec_min)
        candidate = SkyCoord(
            ra_candidate * u.deg, dec_candidate * u.deg, frame=ICRS
        ).transform_to(AltAz(obstime=time, location=observer))
        if candidate.alt.deg >= min_elevation_deg:
            return float(ra_candidate), float(dec_candidate)
    raise RuntimeError(
        "Could not find a valid pointing within the elevation contour after "
        f"{max_attempts} attempts."
    )
