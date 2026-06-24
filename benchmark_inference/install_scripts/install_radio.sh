#!/bin/bash
# Download the Karabo singularity image.
# Simulation (data generation) is handled by Dataset.prepare(), not here.
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

# 1. Check for apptainer / singularity
# On Jean Zay, loading this module also exposes SINGULARITY_ALLOWED_DIR.
load_singularity_module || true
if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    log "Error: apptainer or singularity could not be found (even after module load)."
    exit 1
fi

# 2. Resolve paths
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$BENCHMARK_DIR")"
log "Repository root: $REPO_DIR"

# 3. Pull image (skip if already present)
IMAGE_NAME="karabo.sif"
TOOLS_DIR="$BENCHMARK_DIR/tools"
mkdir -p "$TOOLS_DIR"
IMAGE_PATH="$TOOLS_DIR/$IMAGE_NAME"
LEGACY_IMAGE_PATH="$REPO_DIR/$IMAGE_NAME"
IMAGE_URI="${KARABO_IMAGE_URI:-oras://ghcr.io/bmalezieux/karabo-image:latest}"

if [ ! -f "$IMAGE_PATH" ]; then
    if [ -f "$LEGACY_IMAGE_PATH" ]; then
        log "Moving existing image from $LEGACY_IMAGE_PATH to $IMAGE_PATH..."
        mv "$LEGACY_IMAGE_PATH" "$IMAGE_PATH"
    else
        log "Pulling singularity image from $IMAGE_URI to $IMAGE_PATH..."
        if command -v apptainer &> /dev/null; then
            apptainer pull "$IMAGE_PATH" "$IMAGE_URI"
        else
            singularity pull "$IMAGE_PATH" "$IMAGE_URI"
        fi
    fi
else
    log "Image already found at $IMAGE_PATH."
fi

# Jean Zay: copy image to the singularity-allowed directory if needed.
if [[ -n "${SINGULARITY_ALLOWED_DIR:-}" ]]; then
    TARGET_IMAGE="${SINGULARITY_ALLOWED_DIR%/}/${IMAGE_NAME}"
    log "SINGULARITY_ALLOWED_DIR='$SINGULARITY_ALLOWED_DIR' — ensuring image is at $TARGET_IMAGE"
    mkdir -p "$SINGULARITY_ALLOWED_DIR"
    if [[ ! -f "$TARGET_IMAGE" ]]; then
        if command -v idrcpy &> /dev/null; then
            idrcpy "$IMAGE_PATH" "$TARGET_IMAGE"
        elif command -v idrcontmgr &> /dev/null; then
            idrcontmgr cp "$IMAGE_PATH"
        else
            cp "$IMAGE_PATH" "$TARGET_IMAGE"
        fi
        if [[ ! -f "$TARGET_IMAGE" ]]; then
            log "Error: expected copied image at $TARGET_IMAGE, but it was not found."
            exit 1
        fi
    else
        log "Image already present in allowed directory."
    fi
fi

log "Singularity image ready at $IMAGE_PATH."
