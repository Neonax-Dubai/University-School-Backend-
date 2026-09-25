#!/usr/bin/env bash
# Resilience tests that touch NO live service: loopback fakes inside a network-less container.
#
#   tests/resilience/run_resilience.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
IMAGE="${ZAYED_TEST_IMAGE:-zayed/inference:25.11}"
docker run --rm --network none --user "$(id -u):$(id -g)" -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME=/tmp -e EVENT_OUTBOX_PATH=/tmp/unused.sqlite3 \
  -v "$ROOT":/app:ro -w /app/tests/resilience --entrypoint /bin/bash "$IMAGE" -c '
set -u -o pipefail; status=0
for t in test_outbox test_evidence_outage test_identity_supervisor; do
  echo "== $t"; python -m unittest "$t" 2>&1 | tail -4 || status=1
done
exit $status' 2>&1 | grep -E "^(==|Ran|OK|FAILED|ERROR|FAIL:)"
