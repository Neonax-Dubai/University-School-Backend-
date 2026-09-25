#!/usr/bin/env bash
# Cold start WITHOUT Qdrant, then Qdrant appears: the real Face-ID / Re-ID must come back through
# identity_supervisor without a process restart. Fully isolated: a temporary --internal network,
# a throwaway Qdrant (no ports), removed afterwards. Uses the GPU (InsightFace, OSNet).
#
#   tests/resilience/run_identity_recovery.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
NET=zayed_resilience_test
QDRANT=zayed-resilience-qdrant
TEST=zayed-resilience-identity
cleanup() { docker rm -f "$TEST" "$QDRANT" >/dev/null 2>&1 || true; docker network rm "$NET" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup
docker network create --internal "$NET" >/dev/null
docker run -d --name "$TEST" --network "$NET" --gpus all --user "$(id -u):983" \
  -e HOME=/home/app -e PYTHONDONTWRITEBYTECODE=1 -e YOLO_CONFIG_DIR=/tmp -e MPLCONFIGDIR=/tmp/mpl \
  -e DASHBOARD_URL=http://no-dashboard.invalid:8000 -e DASHBOARD_TOKEN=unused \
  -e FACE_ID_QDRANT_HOST=$QDRANT -e FACE_ID_QDRANT_PORT=6333 -e FACE_ID_ALLOW_SHARED_QDRANT=1 \
  -e FACE_ID_QDRANT_COLLECTION=resilience_face -e KNOWN_PERSON_RECOGNITION_ENABLED=1 \
  -e KNOWN_PERSON_QDRANT_HOST=$QDRANT -e KNOWN_PERSON_QDRANT_PORT=6333 -e KNOWN_PERSON_ALLOW_SHARED_QDRANT=1 \
  -e REID_PROD_STORE=qdrant -e REID_PROD_QDRANT_HOST=$QDRANT -e REID_PROD_QDRANT_PORT=6333 \
  -e REID_PROD_ALLOW_SHARED_QDRANT=1 -e REID_PROD_QDRANT_COLLECTION=resilience_reid \
  -e REID_OBSERVABILITY_OUTPUT_PATH=/tmp/obs.jsonl -e TORCH_HOME=/models/zayed/torch \
  -e FACE_EVIDENCE_ENABLED=0 -e EVIDENCE_ENABLED=0 \
  -v "$ROOT":/app:ro -v /mnt/cctv/models/zayed:/models/zayed:ro \
  -v /mnt/cctv/models/zayed/insightface:/home/app/.insightface:ro \
  --entrypoint python3 zayed/inference:25.11 -u /app/tests/resilience/identity_recovery_live.py >/dev/null
timeout 300 bash -c "until docker logs $TEST 2>&1 | grep -qE 'READY_FOR_QDRANT|RESULT:'; do sleep 2; done"
if docker logs "$TEST" 2>&1 | grep -q READY_FOR_QDRANT; then
  sleep 8                                               # let at least one retry fail first
  echo "starting the throwaway Qdrant ($(date +%T))"
  docker run -d --name "$QDRANT" --network "$NET" qdrant/qdrant:v1.19.1 >/dev/null
fi
status=$(docker wait "$TEST")
docker logs "$TEST" 2>&1 | grep -E "COLD START|IDENTITY|FACE-ID\] could not|REID-ADAPTER\] initialisation|RESULT:|STATS" | cut -c1-240
exit "$status"
