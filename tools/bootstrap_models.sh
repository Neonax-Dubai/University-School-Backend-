#!/usr/bin/env bash
# Download/verify the five Zayed models, build the three TensorRT engines on this GB10,
# smoke-test everything on the GPU. Idempotent: cached, compatible engines are reused.
#   tools/bootstrap_models.sh            FORCE_REBUILD=1 tools/bootstrap_models.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_HOST="${MODELS_HOST:-/mnt/cctv/models/zayed}"
ENGINES_HOST="${ENGINES_HOST:-/mnt/cctv/cache/tensorrt/zayed}"
mkdir -p "$MODELS_HOST" "$ENGINES_HOST"
exec docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e FORCE_REBUILD="${FORCE_REBUILD:-0}" \
  -v "$MODELS_HOST":/models/zayed -v "$ENGINES_HOST":/engines/zayed -v "$HERE":/tools:ro \
  zayed/inference:25.11 python /tools/bootstrap_models.py
