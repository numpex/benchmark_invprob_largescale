#!/bin/bash
# Install script for radio_prepare_test dataset.
#
# This script only handles DEPENDENCIES and IMAGE PULL.
# Data generation (simulation) is handled separately by Dataset.prepare(),
# triggered via:   benchopt prepare . -d radio_prepare_test
set -euo pipefail

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

load_singularity_module() {
    if command -v module &> /dev/null; then
        log "Loading module 'singularity'..."
        if module load singularity; then
            return 0
        fi
        log "Warning: failed to load module 'singularity'."
    fi
    return 1
}

# 1. Check that apptainer or singularity is available
if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    load_singularity_module || true
fi
if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    log "Error: apptainer or singularity could not be found (even after module load)."
    exit 1
fi

# 2. Resolve paths
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$BENCHMARK_DIR")"
cd "$REPO_DIR"
log "Repository root: $REPO_DIR"

# Benchopt may pass CONDA_PREFIX as the first positional argument — consume it.
BENCHOPT_CONDA_PREFIX="${1:-}"
if [[ -n "$BENCHOPT_CONDA_PREFIX" && -d "$BENCHOPT_CONDA_PREFIX" ]]; then
    shift
fi

# 3. Resolve containers directory via benchopt config.
#    This uses the same logic as Dataset._resolve_image_path(None):
#      get_data_path("containers") / "karabo.sif"
#    Falls back to benchmark_inference/data/containers if benchopt is unavailable.
CONTAINERS_DIR=$(cd "$BENCHMARK_DIR" && python -c "
from benchopt.benchmark import Benchmark
b = Benchmark('.')
from benchopt.config import get_data_path
print(get_data_path('containers'))
" 2>/dev/null) || CONTAINERS_DIR="$BENCHMARK_DIR/data/containers"

mkdir -p "$CONTAINERS_DIR"
log "Containers directory: $CONTAINERS_DIR"

# 4. Pull the Singularity image (if not already present)
IMAGE_NAME="karabo.sif"
IMAGE_PATH="$CONTAINERS_DIR/$IMAGE_NAME"
IMAGE_URI="${KARABO_IMAGE_URI:-oras://ghcr.io/bmalezieux/karabo-image:latest}"

if [ ! -f "$IMAGE_PATH" ]; then
    log "Pulling Singularity image from $IMAGE_URI to $IMAGE_PATH ..."
    if command -v apptainer &> /dev/null; then
        apptainer pull "$IMAGE_PATH" "$IMAGE_URI"
    else
        singularity pull "$IMAGE_PATH" "$IMAGE_URI"
    fi
else
    log "Image already present at $IMAGE_PATH."
fi

log "Install complete."
log "Run 'benchopt prepare . -d radio_prepare_test' to generate simulation data."
