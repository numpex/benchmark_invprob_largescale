"""Command line interface for benchmark visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

from toolsbench.visualization.common import (
    DEFAULT_INFERENCE_OUTPUT_DIR,
    DEFAULT_TRAINING_OUTPUT_DIR,
)
from toolsbench.visualization.inference.compile_speedup import (
    create_compile_speedup_visualizations,
    create_denoiser_compile_visualizations,
)
from toolsbench.visualization.inference.quality import create_quality_visualizations
from toolsbench.visualization.inference.scaling import create_scaling_visualizations
from toolsbench.visualization.training.batch_size import create_batch_size_visualizations
from toolsbench.visualization.training.checkpointing import (
    create_checkpointing_visualizations,
)
from toolsbench.visualization.training.comm_time import create_comm_time_visualizations
from toolsbench.visualization.training.strong_scaling import (
    create_strong_scaling_visualizations,
)
from toolsbench.visualization.training.weak_scaling import (
    create_weak_scaling_visualizations,
)

TRAINING_CREATORS = {
    "strong_scaling": create_strong_scaling_visualizations,
    "weak_scaling": create_weak_scaling_visualizations,
    "comm_time": create_comm_time_visualizations,
    "batch_size": create_batch_size_visualizations,
    "checkpointing": create_checkpointing_visualizations,
}


def build_parser(command: str) -> argparse.ArgumentParser:
    """Build a parser for one visualization command."""
    if command == "vizinference":
        return _build_inference_parser()
    if command == "viztraining":
        return _build_training_parser()
    raise ValueError(f"Unknown visualization command: {command}")


def _build_inference_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolsbench vizinference",
        description=(
            "Create inference benchmark visualizations from benchopt parquet results."
        ),
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    scaling = subparsers.add_parser("scaling", help="Visualize scaling experiments.")
    _add_results_args(scaling, DEFAULT_INFERENCE_OUTPUT_DIR)

    quality = subparsers.add_parser(
        "quality", help="Visualize reconstruction-quality experiments."
    )
    _add_results_args(quality, DEFAULT_INFERENCE_OUTPUT_DIR)

    compile_speedup = subparsers.add_parser(
        "compile_speedup",
        aliases=["compile-speedup"],
        help="Visualize torch.compile 1st-iteration vs stable-iteration speedup.",
    )
    _add_results_args(compile_speedup, DEFAULT_INFERENCE_OUTPUT_DIR)

    denoiser_compile = subparsers.add_parser(
        "denoiser_compile",
        aliases=["denoiser-compile"],
        help="Visualize denoiser eager-vs-compiled steady-state speedup (2D/3D).",
    )
    _add_results_args(denoiser_compile, DEFAULT_INFERENCE_OUTPUT_DIR)
    denoiser_compile.add_argument(
        "--roofline",
        action="store_true",
        help="Also plot the roofline (arithmetic intensity vs speedup).",
    )
    return parser


def _build_training_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolsbench viztraining",
        description=(
            "Create training benchmark visualizations from benchopt parquet results."
        ),
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    for experiment in TRAINING_CREATORS:
        subparser = subparsers.add_parser(
            experiment,
            aliases=[experiment.replace("_", "-")],
            help=f"Visualize {experiment.replace('_', ' ')} experiments.",
        )
        _add_results_args(subparser, DEFAULT_TRAINING_OUTPUT_DIR)

    all_parser = subparsers.add_parser(
        "all",
        help="Visualize every known training experiment found in a results directory.",
    )
    all_parser.add_argument(
        "--results-dir",
        default="results_training",
        help="Directory containing one folder per training experiment.",
    )
    all_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_TRAINING_OUTPUT_DIR),
        help="Directory where visualizations are written.",
    )
    return parser


def _add_results_args(
    parser: argparse.ArgumentParser,
    default_output_dir: Path,
) -> None:
    parser.add_argument(
        "--results",
        required=True,
        help="Parquet file or output directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help="Directory where visualizations are written.",
    )


def main(command: str, argv: list[str] | None = None) -> int:
    """Run a visualization subcommand."""
    parser = build_parser(command)
    args = parser.parse_args(argv)

    if command == "vizinference":
        output_paths = _run_inference(args, parser)
    elif command == "viztraining":
        output_paths = _run_training(args, parser)
    else:
        parser.error(f"Unknown visualization command: {command}")

    for output_path in output_paths:
        print(f"Wrote visualizations to {output_path}")
    return 0


def _run_inference(args, parser: argparse.ArgumentParser) -> list[Path]:
    experiment = args.experiment.replace("-", "_")
    if experiment == "scaling":
        return [create_scaling_visualizations(args.results, Path(args.output_dir))]
    if experiment == "quality":
        return [create_quality_visualizations(args.results, Path(args.output_dir))]
    if experiment == "compile_speedup":
        return [create_compile_speedup_visualizations(args.results, Path(args.output_dir))]
    if experiment == "denoiser_compile":
        return [create_denoiser_compile_visualizations(
            args.results, Path(args.output_dir), roofline=args.roofline
        )]
    parser.error(f"Unknown inference experiment: {args.experiment}")
    return []


def _run_training(args, parser: argparse.ArgumentParser) -> list[Path]:
    experiment = args.experiment.replace("-", "_")
    if experiment == "all":
        return _run_all_training(Path(args.results_dir), Path(args.output_dir))

    creator = TRAINING_CREATORS.get(experiment)
    if creator is None:
        parser.error(f"Unknown training experiment: {args.experiment}")
    return [creator(args.results, Path(args.output_dir))]


def _run_all_training(results_dir: Path, output_dir: Path) -> list[Path]:
    output_paths = []
    for experiment, creator in TRAINING_CREATORS.items():
        experiment_dir = results_dir / experiment
        if not experiment_dir.exists():
            print(f"Skipping {experiment}: {experiment_dir} does not exist")
            continue
        if not list(experiment_dir.glob("*.parquet")):
            print(f"Skipping {experiment}: no parquet file found in {experiment_dir}")
            continue
        output_paths.append(creator(experiment_dir, output_dir))
    return output_paths
