#!/usr/bin/env bash
# Unit tests for the Zayed-specific inference modules. CPU only, no network.
#
#   tests/unit/run_unit.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
docker run --rm --network none --user "$(id -u):$(id -g)" -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp \
  -v "$ROOT":/app:ro -w /app/tests/unit --entrypoint /bin/bash "${ZAYED_TEST_IMAGE:-zayed/inference:25.11}" -c '
set -u -o pipefail; status=0
for t in test_phone_use test_sleeping test_evacuation test_floor_plan test_cctv_stall; do
  echo "== $t"; python -m unittest "$t" 2>&1 | tail -4 || status=1
done
exit $status' 2>&1 | grep -E "^(==|Ran|OK|FAILED|ERROR|FAIL:|AssertionError|Traceback|  File)"
