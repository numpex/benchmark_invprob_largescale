import sys
import argparse
from pathlib import Path

# Add the src directory to sys.path when this file is executed directly in the
# Karabo container. parents[2] is ``src/toolsbench``; Python needs ``src`` in
# order to resolve the top-level ``toolsbench`` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from toolsbench.utils.radio_interferometry.karabo_utils import (
    generate_meerkat_visibilities,
)
from toolsbench.utils.radio_interferometry.radio_utils import (
    load_config,
    load_fits_image,
)


def generate_data(cfg):
    """Generate data at the FITS image's native spatial resolution."""
    data_path = cfg.data_path
    fits_name = cfg.fits_name
    pos_ra = float(cfg.pos_ra)
    pos_dec = float(cfg.pos_dec)
    random_position = bool(cfg.random_position)
    use_gpus = bool(cfg.use_gpus)
    number_of_time_steps = int(cfg.number_of_time_steps)
    start_frequency_hz = float(cfg.start_frequency_hz)
    end_frequency_hz = float(cfg.end_frequency_hz)
    number_of_channels = int(cfg.number_of_channels)
    add_noise = bool(cfg.add_noise)
    pol_mode = str(cfg.pol_mode)

    # Cache directory
    data_path = Path(data_path)
    ms_cache_dir = data_path / "meerkat_cache"
    ms_cache_dir.mkdir(parents=True, exist_ok=True)

    # Verify/Load image
    fits_file = data_path / fits_name
    if not fits_file.exists():
        print(f"File not found: {fits_file}")
        # Try finding it in default benchmark data dir relative to script
        default_data_dir = Path(__file__).resolve().parents[3] / "data"
        fits_file = default_data_dir / fits_name
        if not fits_file.exists():
            print(f"Could not find {fits_name} in {data_path} or {default_data_dir}")
            return

    try:
        image = load_fits_image(fits_file, normalize=False)
    except Exception as e:
        print(f"Could not load/process FITS image: {e}")
        return

    image_size = image.shape[-1]
    print(
        f"Generating data at native image size {image_size} "
        f"with use_gpus={use_gpus}"
    )

    # Generate visibilities
    vis_path = generate_meerkat_visibilities(
        fits_file,
        image,
        ms_cache_dir,
        use_gpus=use_gpus,
        number_of_time_steps=number_of_time_steps,
        start_frequency_hz=start_frequency_hz,
        end_frequency_hz=end_frequency_hz,
        number_of_channels=number_of_channels,
        pos_ra=pos_ra,
        pos_dec=pos_dec,
        random_position=random_position,
        add_noise=add_noise,
        pol_mode=pol_mode,
    )

    print(f"Visibilities ready for size {image_size}: {vis_path}")


def main_generation_loop(cfg):
    generate_data(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args = parser.parse_args()
    cfg = load_config(args.config, section="job")

    # Check if GPU is available
    # In karabo env, we assume cpu usually, or if torch is missing we definitely use cpu
    # But this script is running in karabo env WITHOUT torch.
    use_gpus = bool(cfg.use_gpus)
    print(f"Running generation with use_gpus={use_gpus}")

    main_generation_loop(cfg)
