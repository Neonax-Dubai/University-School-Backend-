"""
REAL face_id_adapter / reid_adapter recover through identity_supervisor after a cold start
without Qdrant. Run only by run_identity_recovery.sh, inside an isolated Docker network whose
throwaway Qdrant is started AFTER this script has seen both components fail.
"""
import os
import sys
import time

sys.path.insert(0, "/app")
import face_id_adapter                                          # noqa: E402
import identity_supervisor as ids                               # noqa: E402
import reid_adapter                                             # noqa: E402

cameras = ["camera_01"]
live = {"face": face_id_adapter.build_production_adapter(), "reid": None}
live["face"].set_enabled_cameras(cameras)
live["reid"] = reid_adapter.ReIDAdapter()
face_down, reid_down = ids.face_failed(live["face"]), ids.reid_failed(live["reid"])
print(f"COLD START without Qdrant: face_failed={face_down} reid_failed={reid_down}", flush=True)
if not (face_down and reid_down):
    print("RESULT: FAIL - both components were expected to fail without Qdrant", flush=True)
    sys.exit(1)

sup = ids.IdentitySupervisor(log=lambda m: print(m, flush=True), min_delay=3, max_delay=10)
sup.watch("Face-ID", lambda: live["face"], lambda n: live.__setitem__("face", n),
          face_id_adapter.build_production_adapter, failed=ids.face_failed,
          arm=lambda a: a.set_enabled_cameras(cameras), ready=ids.face_ready)
sup.watch("Re-ID", lambda: live["reid"], lambda n: live.__setitem__("reid", n), reid_adapter.ReIDAdapter,
          failed=ids.reid_failed)
print("READY_FOR_QDRANT", flush=True)

started = time.monotonic()
recovered = set()
while time.monotonic() - started < 240 and recovered != {"Face-ID", "Re-ID"}:
    recovered |= set(sup.check_once())
    time.sleep(1)
elapsed = time.monotonic() - started
ok = ids.face_ready(live["face"]) and live["reid"].enabled
print(f"STATS {sup.stats()}", flush=True)
print(f"RESULT: {'PASS' if ok else 'FAIL'} - recovered={sorted(recovered)} after {elapsed:.0f}s "
      f"face_ready={ids.face_ready(live['face'])} reid_enabled={live['reid'].enabled}", flush=True)
for adapter in (live["face"], live["reid"]):
    try:
        adapter.close()
    except Exception:                                           # noqa: BLE001
        pass
sys.exit(0 if ok else 1)
