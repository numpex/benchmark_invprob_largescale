"""Command line interface for benchmark visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

from toolsbench.visualization.common import (
    DEFAULT_INFERENCE_OUTPUT_DIR,
    DEFAULT_TRAINING_OUTPUT_DIR,
)
from toolsbench.visualization.inference.comm_inference import (
    DEFAULT_SKIP_WARMUP as COMM_INFERENCE_SKIP_WARMUP,
    create_comm_inference_visualizations,
)
from toolsbench.visualization.inference.compile_speedup import (
    create_compile_speedup_visualizations,
    create_denoiser_compile_visualizations,
)
from toolsbench.visualization.inference.quality import create_quality_visualizations
from toolsbench.visualization.inference.scaling import create_scaling_visualizations
from toolsbench.visualization.training.batch_size import (
    create_batch_size_visualizations,
)
from toolsbench.visualization.training.checkpointing import (
    create_checkpointing_visualizations,
)
from toolsbench.visualization.training.comm_time import create_comm_time_visualizations
from toolsbench.visualization.training.scaling import (
    create_strong_scaling_visualizations,
)
from toolsbench.visualization.website.comm import (
    create_communication_inference_website_data,
    create_communication_training_website_data,
)
from toolsbench.visualization.website.cryo_training import (
    create_cryo_training_website_data,
)
from toolsbench.visualization.website.compile_speedup import (
    create_compile_speedup_website_data,
)
from toolsbench.visualization.website.inference_scaling import (
    create_inference_scaling_website_data,
)
from toolsbench.visualization.website.training_scaling import (
    create_training_scaling_website_data,
)

TRAINING_CREATORS = {
    "scaling": create_strong_scaling_visualizations,
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
    if command == "vizwebsite":
        return _build_website_parser()
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

    comm_inference = subparsers.add_parser(
        "comm_inference",
        aliases=["comm-inference"],
        help="Visualize the denoiser/gradient compute-vs-communication breakdown.",
    )
    _add_results_args(
        comm_inference,
        DEFAULT_INFERENCE_OUTPUT_DIR,
        results_help=(
            "Parent directory containing a comm_2D/ and a comm_3D/ subfolder, "
            "each with its own benchopt parquet."
        ),
    )
    comm_inference.add_argument(
        "--skip-warmup",
        type=int,
        default=COMM_INFERENCE_SKIP_WARMUP,
        help=(
            "Timed iterations to drop per configuration. Counted from the first "
            "iteration that carries timings, not from stop_val=0."
        ),
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
        if experiment == "comm_time":
            _add_results_args(
                subparser,
                DEFAULT_TRAINING_OUTPUT_DIR,
                results_help=(
                    "Parent directory containing a comm_2D/ and a comm_3D/ "
                    "subfolder, each with its own benchopt parquet."
                ),
            )
        else:
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


def _build_website_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolsbench vizwebsite",
        description="Create plot-ready JSON data for benchmark finding pages.",
    )
    subparsers = parser.add_subparsers(dest="finding", required=True)
    inference_scaling = subparsers.add_parser(
        "inference_scaling",
        aliases=["inference-scaling"],
        help="Export distributed PnP scaling data.",
    )
    inference_scaling.add_argument(
        "--scaling-results",
        required=True,
        help="Inference strong-scaling parquet file or containing directory.",
    )
    inference_scaling.add_argument(
        "--output-dir",
        default="site/src/data/results",
        help="Root directory where finding JSON data is written.",
    )

    training_scaling = subparsers.add_parser(
        "training_scaling",
        aliases=["training-scaling"],
        help="Export training scaling, checkpointing, and batch-size data.",
    )
    training_scaling.add_argument(
        "--scaling-results",
        required=True,
        help=(
            "Training strong-scaling parquet file or containing directory; "
            "the weak-scaling ladders are derived from it too."
        ),
    )
    training_scaling.add_argument(
        "--checkpointing-results",
        required=True,
        help="Training checkpointing parquet file or containing directory.",
    )
    training_scaling.add_argument(
        "--batch-size-results",
        required=True,
        help="Training max-batch-size parquet file or containing directory.",
    )
    training_scaling.add_argument(
        "--output-dir",
        default="site/src/data/results",
        help="Root directory where finding JSON data is written.",
    )

    cryo = subparsers.add_parser(
        "cryo_training",
        aliases=["cryo-training"],
        help="Export the cryo-ET training data (Adastra).",
    )
    for name in ("speedup", "patch-size", "checkpointing", "tiling"):
        cryo.add_argument(f"--{name}-results", required=True, help="Parquet file.")
    cryo.add_argument("--output-dir", default="site/src/data/results")

    compile_speedup = subparsers.add_parser(
        "compile_speedup",
        aliases=["compile-speedup"],
        help="Export torch.compile denoiser and PnP speedup data.",
    )
    compile_speedup.add_argument(
        "--denoiser-results",
        required=True,
        help="Denoiser compile-sweep parquet file or containing directory.",
    )
    compile_speedup.add_argument(
        "--pnp-results",
        required=True,
        help="PnP compile-sweep parquet file or containing directory.",
    )
    compile_speedup.add_argument(
        "--output-dir",
        default="site/src/data/results",
        help="Root directory where finding JSON data is written.",
    )

    for finding in ("communication_inference", "communication_training"):
        study = finding.split("_", maxsplit=1)[1]
        communication = subparsers.add_parser(
            finding,
            aliases=[finding.replace("_", "-")],
            help=f"Export the {study} compute/transfer/wait communication breakdown.",
        )
        communication.add_argument(
            "--results-2d",
            required=True,
            help=f"2D {study} communication parquet file or containing directory.",
        )
        communication.add_argument(
            "--results-3d",
            required=True,
            help=f"3D {study} communication parquet file or containing directory.",
        )
        communication.add_argument(
            "--output-dir",
            default="site/src/data/results",
            help="Root directory where finding JSON data is written.",
        )
    return parser


def _add_results_args(
    parser: argparse.ArgumentParser,
    default_output_dir: Path,
    *,
    results_help: str = "Parquet file or output directory.",
) -> None:
    parser.add_argument(
        "--results",
        required=True,
        help=results_help,
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
    elif command == "vizwebsite":
        output_paths = _run_website(args, parser)
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
        return [
            create_compile_speedup_visualizations(args.results, Path(args.output_dir))
        ]
    if experiment == "denoiser_compile":
        return [
            create_denoiser_compile_visualizations(
                args.results, Path(args.output_dir), roofline=args.roofline
            )
        ]
    if experiment == "comm_inference":
        return [
            create_comm_inference_visualizations(
                args.results, Path(args.output_dir), skip_warmup=args.skip_warmup
            )
        ]
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


def _run_website(args, parser: argparse.ArgumentParser) -> list[Path]:
    finding = args.finding.replace("-", "_")
    if finding == "inference_scaling":
        return create_inference_scaling_website_data(
            scaling_results=args.scaling_results,
            output_dir=Path(args.output_dir),
        )
    if finding == "training_scaling":
        return create_training_scaling_website_data(
            scaling_results=args.scaling_results,
            checkpointing_results=args.checkpointing_results,
            batch_size_results=args.batch_size_results,
            output_dir=Path(args.output_dir),
        )
    if finding == "cryo_training":
        return create_cryo_training_website_data(
            speedup_results=args.speedup_results,
            patch_size_results=args.patch_size_results,
            checkpointing_results=args.checkpointing_results,
            tiling_results=args.tiling_results,
            output_dir=Path(args.output_dir),
        )
    if finding == "compile_speedup":
        return create_compile_speedup_website_data(
            denoiser_results=args.denoiser_results,
            pnp_results=args.pnp_results,
            output_dir=Path(args.output_dir),
        )
    if finding in ("communication_inference", "communication_training"):
        creator = (
            create_communication_inference_website_data
            if finding == "communication_inference"
            else create_communication_training_website_data
        )
        return creator(
            results_2d=args.results_2d,
            results_3d=args.results_3d,
            output_dir=Path(args.output_dir),
        )
    parser.error(f"Unknown website finding: {args.finding}")
    return []


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
