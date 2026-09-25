#!/usr/bin/env bash
# Internal face API unit tests: CPU only, no network, model stubbed.
#
#   tests/internal_face/run_internal_face.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
docker run --rm --network none --user "$(id -u):$(id -g)" -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp \
  -v "$ROOT":/app:ro -w /app/tests/internal_face --entrypoint /bin/bash \
  "${ZAYED_TEST_IMAGE:-zayed/inference:25.11}" -c '
set -u -o pipefail
python -m unittest test_internal_api 2>&1 | tail -5' 2>&1 | grep -E "^(Ran|OK|FAILED|ERROR|FAIL:|AssertionError)"
