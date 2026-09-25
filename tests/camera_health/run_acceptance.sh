#!/usr/bin/env bash
# Camera-health acceptance suite: the seven Dubai test files, copied VERBATIM,
# run against zayed_ai_inferencing/camera_health.
#
#   tests/camera_health/run_acceptance.sh
#
# test_camera_tamper_lighting.py needs 14 real CAM-R25 frames (320x180 grey).
# They are Dubai camera imagery, so they are NOT stored in this project: they
# are extracted from the frozen Dubai repository's history into a temporary
# directory for the run and deleted afterwards.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DUBAI_REPO="${DUBAI_REPO:-/home/stack/Zayed_University/dubai_ai_inferncing}"
FIXTURE_COMMIT="${FIXTURE_COMMIT:-9a511c37}"
IMAGE="${ZAYED_TEST_IMAGE:-cctv/analytics-probe:25.11}"
MAIN_LOOP="${MAIN_LOOP:-zayed_inference.py}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp "$HERE"/test_*.py "$WORK"/
mkdir -p "$WORK/test_fixtures/camera_tamper_lighting"
for f in $(git -C "$DUBAI_REPO" ls-tree --name-only "$FIXTURE_COMMIT" test_fixtures/camera_tamper_lighting/); do
  git -C "$DUBAI_REPO" show "$FIXTURE_COMMIT:$f" > "$WORK/$f"
done

# As the invoking user and without bytecode, so the temporary directory (which briefly holds
# the Dubai fixture frames) can always be deleted by the EXIT trap.
docker run --rm --network none --user "$(id -u):$(id -g)" -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$ROOT":/app:ro -v "$WORK":/t -e PYTHONPATH=/app \
  -e MULTICAM_PATH="/app/$MAIN_LOOP" -w /t "$IMAGE" bash -c '
set -u -o pipefail; status=0          # pipefail: a failing test must fail the run, not just print
for t in test_camera_health test_camera_health_recovery test_camera_health_umbrella \
         test_camera_tamper_incident test_camera_tamper_lighting; do
  echo "== $t"; python -m unittest "$t" 2>&1 | tail -3 || status=1
done
echo "== test_camera_health_telemetry"; python test_camera_health_telemetry.py | tail -1 || status=1
if [ -f "$MULTICAM_PATH" ]; then
  echo "== test_camera_health_event (handler from $MULTICAM_PATH)"; python -m unittest test_camera_health_event 2>&1 | tail -3 || status=1
else
  echo "== test_camera_health_event SKIPPED: $MULTICAM_PATH does not exist yet"; status=2
fi
exit $status' 2>&1 | grep -E "^(==|Ran|OK|FAILED|[0-9]+ passed)"
