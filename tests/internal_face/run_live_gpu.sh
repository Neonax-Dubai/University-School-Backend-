#!/usr/bin/env bash
# Live GPU check for the internal face API (real buffalo_l on CUDA). Read-only, offline.
#
#   tests/internal_face/run_live_gpu.sh <image> [baseline.npy]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
IMAGE_PATH="${1:-$ROOT/part-chall_b.jpg}"
BASELINE="${2:-}"
MOUNTS=(-v "$ROOT":/app:ro -v /mnt/cctv/models/zayed:/models/zayed:ro
        -v /mnt/cctv/models/zayed/insightface:/home/app/.insightface:ro
        -v "$(dirname "$IMAGE_PATH")":/imgs:ro)
BASE_ARG=""
if [ -n "$BASELINE" ]; then
  MOUNTS+=(-v "$(dirname "$BASELINE")":/baseline:ro)
  BASE_ARG="/baseline/$(basename "$BASELINE")"
fi
docker run --rm --gpus all --network none --user "$(id -u):983" \
  -e HOME=/home/app -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp/mpl \
  "${MOUNTS[@]}" --entrypoint python3 "${ZAYED_TEST_IMAGE:-zayed/inference:25.11}" -u \
  /app/tests/internal_face/live_gpu_check.py "/imgs/$(basename "$IMAGE_PATH")" "$BASE_ARG" 2>&1 |
  grep -E "^(PASS|FAIL|ALL LIVE|FAILED|      |\[FACE-API\])"
