#!/usr/bin/env bash
# Offline replay of one extracted clip through the Fall path - GPU, NO network, no production writes.
#
#   tests/fall/run_fall_replay.sh <clip.mp4> <out_dir> [--candidate] [--phase 0|1|2|all]
#
# baseline  = fall_pose_policy.py at $BASELINE_REF (default `main`, i.e. what production runs),
#             taken from git - never from a working tree someone may be editing.
# candidate = this checkout's fall_pose_policy.py (only with --candidate).
#
# The clip must sit next to its extract_clip.py sidecar (<clip>.json). Nothing is replayed into
# MediaMTX, and the container has no route to PostgreSQL, Qdrant, Redis, SeaweedFS or the dashboard.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CLIP="$(realpath "${1:?usage: run_fall_replay.sh <clip.mp4> <out_dir> [--candidate] [--phase N]}")"
OUT="$(realpath -m "${2:?out_dir}")"
shift 2
SIDECAR="${CLIP%.mp4}.json"
[ -f "$SIDECAR" ] || { echo "missing sidecar $SIDECAR" >&2; exit 2; }
mkdir -p "$OUT" && chmod 700 "$OUT"
BASE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fall_replay_baseline.XXXXXX")"
trap 'rm -rf "$BASE_DIR"' EXIT
git -C "$ROOT" show "${BASELINE_REF:-main}:fall_pose_policy.py" > "$BASE_DIR/fall_pose_policy.py"
chmod 755 "$BASE_DIR" && chmod 644 "$BASE_DIR/fall_pose_policy.py"
CANDIDATE=()
if [ "${1:-}" = "--candidate" ]; then
  CANDIDATE=(--candidate-policy /app/fall_pose_policy.py)
  shift
fi
docker run --rm --network none --gpus all --user "$(id -u):983" \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e YOLO_CONFIG_DIR=/tmp -e MPLCONFIGDIR=/tmp/mpl \
  -e POSE_L960_ENGINE=/engines/zayed/current/yolo26l-pose.engine \
  -e POSE_L960_MODEL_PT=/engines/zayed/current/yolo26l-pose.engine \
  -v "$ROOT":/app:ro -v /mnt/cctv/cache/tensorrt/zayed:/engines/zayed:ro \
  -v "$(dirname "$CLIP")":/clip:ro -v "$BASE_DIR":/baseline:ro -v "$OUT":/out \
  --entrypoint python3 zayed/inference:25.11 -u /app/tests/fall/replay_fall_clip.py \
  --clip "/clip/$(basename "$CLIP")" --sidecar "/clip/$(basename "$SIDECAR")" \
  --baseline-policy /baseline/fall_pose_policy.py "${CANDIDATE[@]}" --out-dir /out "$@" 2>&1 |
  grep -vE "^\s*$|TensorRT|TRT\]|Loading|WARNING ⚠️"
