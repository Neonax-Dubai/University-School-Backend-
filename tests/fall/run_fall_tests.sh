#!/usr/bin/env bash
# Fall-policy tests: CPU only, no network, the checkout mounted read-only.
#
#   tests/fall/run_fall_tests.sh
#
# 1. Zayed scenarios (test_fall_pose_zayed.py) against this checkout's policy. The policy production
#    runs - `git show $BASELINE_REF:fall_pose_policy.py`, default main - is passed as the "before",
#    so the Event 32 fixtures must still reproduce its false alarm.
# 2. Dubai's own test_fall_pose.py from the frozen repository, UNMODIFIED, against copies of this
#    checkout's modules: once as shipped, and once with FALL_POSE_CONFIRM_SECONDS=0.3 so the
#    structural change is measured apart from the longer hold. Every Dubai failure must be one
#    of the classified ones below, and every classified one must fail - anything else fails the run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DUBAI_REPO="${DUBAI_REPO:-/home/stack/Zayed_University/dubai_ai_inferncing}"
DUBAI_COMMIT="${DUBAI_COMMIT:-5b192cef}"
IMAGE="${ZAYED_TEST_IMAGE:-zayed/inference:25.11}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git -C "$ROOT" show "${BASELINE_REF:-main}:fall_pose_policy.py" > "$WORK/baseline_fall_pose_policy.py"
mkdir "$WORK/dubai"
cp "$ROOT"/*.py "$WORK/dubai/"
git -C "$DUBAI_REPO" show "$DUBAI_COMMIT:testing/test_fall_pose.py" > "$WORK/dubai/test_fall_pose.py"
chmod -R a+rX "$WORK"

docker run --rm --network none --user "$(id -u):$(id -g)" -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp \
  -v "$ROOT":/app:ro -v "$WORK":/w:ro -e PYTHONPATH=/app -w /app/tests/fall \
  --entrypoint /bin/bash "$IMAGE" -c '
set -u -o pipefail; status=0

echo "== Zayed fall scenarios (baseline = the policy production runs)"
if out=$(FALL_BASELINE_POLICY=/w/baseline_fall_pose_policy.py python -m unittest -v test_fall_pose_zayed 2>&1); then
  echo "$out" | grep -E " \.\.\. |^Ran |^OK"
else
  echo "$out"; status=1
fi

classify() {   # $1 label, $2 expected failing test numbers (space separated), $3 runner output
  python3 - "$1" "$2" "$3" <<"PY"
import re, sys
label, expected, output = sys.argv[1], set(sys.argv[2].split()), sys.argv[3]
failed = set(re.findall(r"^\s+FAILED (\d+) ", output, re.M))
total = re.search(r"^(\d+) passed, (\d+) failed", output, re.M)
why = {"2": "missing test data: pose_capture.json is in no repository",
       "10": "missing test data: pose_capture.json is in no repository",
       "32": "INTENDED CHANGE: pins the torso-only fallback when the ankles are hidden - the Event 32 mechanism",
       "4": "pinned to the 0.3 s hold: observes 0.76 s", "9": "pinned to the 0.3 s hold: observes 0.76 s",
       "18": "pinned to the 0.3 s hold: observes 0.76 s", "19": "pinned to the 0.3 s hold: observes 0.76 s",
       "22": "pinned to the 0.3 s hold: observes 0.96 s",
       "34": "pinned to the 0.3 s hold: asserts CONFIRM_SECONDS == 0.3"}
print(f"   {label}: {total.group(1)} passed, {total.group(2)} failed")
for n in sorted(failed, key=int):
    reason = why.get(n, "UNEXPECTED FAILURE")
    print(f"      test {n:>2}: {reason}")
bad = (failed - expected) | (expected - failed)
if bad:
    print(f"   REGRESSION CHECK FAILED - unexpected: {sorted(failed - expected, key=int)}, "
          f"expected but passed: {sorted(expected - failed, key=int)}")
    sys.exit(1)
print("   every failure is classified and every classified failure occurred")
PY
}

echo "== Dubai test_fall_pose.py (unmodified) on this checkout, CONFIRM_SECONDS as shipped"
out=$(cd /w/dubai && python test_fall_pose.py 2>&1) || true
classify "as shipped" "2 10 32 4 9 18 19 22 34" "$out" || status=1

echo "== Dubai test_fall_pose.py (unmodified) on this checkout, FALL_POSE_CONFIRM_SECONDS=0.3"
out=$(cd /w/dubai && FALL_POSE_CONFIRM_SECONDS=0.3 python test_fall_pose.py 2>&1) || true
classify "hold at 0.3 s (structural change only)" "2 10 32" "$out" || status=1

exit $status'
