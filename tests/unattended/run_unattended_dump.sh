#!/usr/bin/env bash
# Stage 1 of the unattended-object replay (GPU, NO network, no production writes): dump what the
# analytic would receive for a sequence of contiguous clips of ONE camera.
#
#   tests/unattended/run_unattended_dump.sh <camera_id> <out.jsonl.gz> <clip.mp4>@<offset_s> [...]
#
# Nothing is replayed into MediaMTX, and the container has no route to PostgreSQL, Qdrant, Redis,
# SeaweedFS or the dashboard. The dump holds boxes only - no pixels.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CAMERA="${1:?usage: run_unattended_dump.sh <camera_id> <out.jsonl.gz> <clip>@<offset> ...}"
OUT="$(realpath -m "${2:?out.jsonl.gz}")"
shift 2
mkdir -p "$(dirname "$OUT")"
MOUNTS=(-v "$ROOT":/app:ro -v /mnt/cctv/cache/tensorrt/zayed:/engines/zayed:ro -v "$(dirname "$OUT")":/out)
ARGS=()
i=0
for spec in "$@"; do
  clip="$(realpath "${spec%@*}")"; offset="${spec##*@}"; i=$((i + 1))
  MOUNTS+=(-v "$clip":/clips/$i.mp4:ro); ARGS+=(--clip "/clips/$i.mp4@$offset")
done
docker run --rm --network none --gpus all --user "$(id -u):983" --name "${REPLAY_NAME:-unattended-dump-$CAMERA}" \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e YOLO_CONFIG_DIR=/tmp -e MPLCONFIGDIR=/tmp/mpl \
  "${MOUNTS[@]}" --entrypoint python3 zayed/inference:25.11 -u /app/tests/unattended/dump_unattended_inputs.py \
  --camera "$CAMERA" --out "/out/$(basename "$OUT")" "${ARGS[@]}" 2>&1 | grep -vE "^\s*$|TensorRT|TRT\]|Loading|WARNING ⚠️|Ultralytics Settings|yolo settings"
