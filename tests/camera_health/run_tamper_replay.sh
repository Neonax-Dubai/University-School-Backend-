#!/usr/bin/env bash
# Offline replay of recorded clips through the camera-health scene-change path.
# NO network, no GPU, no production writes - the container has no route to PostgreSQL,
# Qdrant, Redis, SeaweedFS, MediaMTX or the dashboard, and the harness never uses the
# manager (whose _announce() emits incidents).
#
#   tests/camera_health/run_tamper_replay.sh <out_dir> <clip.mp4> [clip.mp4 ...]
#   CANDIDATE=1 tests/camera_health/run_tamper_replay.sh ...     # this checkout's detectors.py
#
# baseline  = camera_health/ at $BASELINE_REF (default HEAD, i.e. what production runs),
#             taken from git - never from a working tree someone may be editing.
# candidate = this checkout's camera_health/ (only with CANDIDATE=1).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$(realpath -m "${1:?usage: run_tamper_replay.sh <out_dir> <clip.mp4> ...}")"; shift
[ $# -ge 1 ] || { echo "no clips given" >&2; exit 2; }
mkdir -p "$OUT"

PKG_MOUNT=()
if [ "${CANDIDATE:-0}" = "1" ]; then
  PACKAGE=/app
  LABEL_SUFFIX="candidate"
else
  BASE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tamper_replay_baseline.XXXXXX")"
  trap 'rm -rf "$BASE_DIR"' EXIT
  mkdir -p "$BASE_DIR/camera_health"
  for f in $(git -C "$ROOT" ls-tree --name-only "${BASELINE_REF:-HEAD}:camera_health/"); do
    case "$f" in *.py) git -C "$ROOT" show "${BASELINE_REF:-HEAD}:camera_health/$f" \
      > "$BASE_DIR/camera_health/$f" ;; esac
  done
  chmod -R a+rX "$BASE_DIR"
  PACKAGE=/baseline
  LABEL_SUFFIX="baseline@${BASELINE_REF:-HEAD}"
  PKG_MOUNT=(-v "$BASE_DIR:/baseline:ro")
fi

echo "[harness] package=$LABEL_SUFFIX  clips=$#  out=$OUT"
for clip in "$@"; do
  name="$(basename "${clip%.mp4}")"
  docker run --rm --network none --user "$(id -u):$(id -g)" \
    -v "$ROOT:/app:ro" "${PKG_MOUNT[@]}" \
    -v "$(dirname "$(realpath "$clip")"):/clips:ro" \
    -v "$OUT:/out" \
    -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp \
    ${CAMERA_HEALTH_ENV:+$CAMERA_HEALTH_ENV} \
    -w /app zayed/inference:25.11 \
    python3 tests/camera_health/replay_tamper_clip.py \
      --clip "/clips/$(basename "$clip")" \
      --package "$PACKAGE" \
      --label "$name [$LABEL_SUFFIX]" \
      --json "/out/${name}.${LABEL_SUFFIX//[^A-Za-z0-9]/_}.json"
done
