#!/usr/bin/env bash
# Offline replay of simultaneous C101 clips through the occupancy path - GPU, NO network, no production writes.
#
#   tests/occupancy/run_occupancy_replay.sh <out_dir> <zones.json> camera_01=<a.mp4> camera_02=<b.mp4> camera_03=<c.mp4> [--overlays 30]
#
# Nothing is replayed into MediaMTX, and the container has no route to PostgreSQL, Qdrant, Redis,
# SeaweedFS or the dashboard. Overlays show people: keep <out_dir> private and delete it afterwards.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$(realpath -m "${1:?usage: run_occupancy_replay.sh <out_dir> <zones.json> camera_XX=<clip> ...}")"
ZONES="$(realpath "${2:?zones.json}")"
shift 2
mkdir -p "$OUT" && chmod 700 "$OUT"
MOUNTS=(-v "$ROOT":/app:ro -v /mnt/cctv/cache/tensorrt/zayed:/engines/zayed:ro -v "$OUT":/out -v "$ZONES":/zones.json:ro)
ARGS=()
i=0
while [ $# -gt 0 ]; do
  case "$1" in
    camera_*=*) cam="${1%%=*}"; clip="$(realpath "${1#*=}")"; i=$((i + 1))
                MOUNTS+=(-v "$clip":/clips/$i.mp4:ro); ARGS+=(--clip "$cam=/clips/$i.mp4"); shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
docker run --rm --network none --gpus all --user "$(id -u):983" --name "${REPLAY_NAME:-occupancy-replay}" \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e YOLO_CONFIG_DIR=/tmp -e MPLCONFIGDIR=/tmp/mpl \
  "${MOUNTS[@]}" --entrypoint python3 zayed/inference:25.11 -u /app/tests/occupancy/replay_occupancy_clips.py \
  --zones /zones.json --out-dir /out "${ARGS[@]}" 2>&1 | grep -vE "^\s*$|TensorRT|TRT\]|Loading|WARNING ⚠️"
