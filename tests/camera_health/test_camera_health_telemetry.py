"""
Camera Health CURRENT-STATE telemetry — tests.

    ./aienv/bin/python test_camera_health_telemetry.py

No camera, no GPU, no network, no database. The manager and the publisher are
driven directly; the HTTP session is a fake.

WHAT THESE ARE FOR
------------------
The Camera Details health panel showed 100% availability / blur 5.0 / 25 fps
for three weeks because CameraHealth's model DEFAULTS are indistinguishable
from a healthy reading and nothing ever wrote them. So every test below is
about the difference between a MEASUREMENT and an absence of one: a value is
either measured and reported, or absent and reported as None. Nothing here
asserts that an unmeasured quantity has a plausible value, because that is the
bug.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import camera_health
from camera_health import detectors, metrics, state as state_mod

PASSED = 0
FAILURES = []

ALL = {k: True for k in camera_health.CAMERA_HEALTH_FEATURES}


def check(name, ok, detail=""):
    global PASSED
    if ok:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILURES.append((name, detail))
        print(f"  FAIL  {name}: {detail}")


class FakeReader:
    """cctv.CameraStream stand-in: the manager only reads frames_read."""

    def __init__(self, frames_read=0):
        self.frames_read = frames_read


def structured(seed=1, w=640, h=360, blur=0, bright=None):
    """A frame with real structure, optionally blurred or flattened."""
    import cv2
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, size=(h // 8, w // 8, 3), dtype=np.uint8)
    frame = np.kron(small, np.ones((8, 8, 1), dtype=np.uint8))
    if blur:
        frame = cv2.GaussianBlur(frame, (blur, blur), 0)
    if bright is not None:
        frame = np.full((h, w, 3), bright, dtype=np.uint8)
    return frame


def manager(features=None):
    m = camera_health.CameraHealthManager(shadow=True)
    m.set_enabled_cameras({"CAM-T1": dict(features or ALL)})
    return m


def deliver(m, camera_id, reader, t0, seconds, fps=25.0, hz=10.0, stall=False):
    """Run note_streams for `seconds` of virtual time at a real rate.

    Frames accumulate as a float and are emitted whole, so a 5 fps camera
    ticked at 10 Hz delivers 5 frames a second. Truncating per tick - the
    obvious version - silently delivered ZERO for any rate below the tick
    rate, which is a fixture that does not test what it says it does.
    """
    t = t0
    step = 1.0 / hz
    owed = 0.0
    for _ in range(int(seconds * hz)):
        t += step
        if not stall:
            owed += fps * step
            whole = int(owed)
            reader.frames_read += whole
            owed -= whole
        m.note_streams({camera_id: reader}, now=t)
    return t


def snap(m, camera_id, now):
    for row in m.snapshot(now=now):
        if row["camera_id"] == camera_id:
            return row
    return None


# ─────────────────────────────────────────────────── availability

def test_01_availability_is_none_before_the_window_fills():
    m = manager()
    r = FakeReader()
    t = deliver(m, "CAM-T1", r, 1000.0, 30)          # < AVAILABILITY_MIN_SLOTS
    s = snap(m, "CAM-T1", t)
    check("01 availability is None before the window fills",
          s["availability_pct"] is None and s["telemetry"] == "warming_up",
          f"got {s['availability_pct']} / {s['telemetry']}")


def test_02_availability_measured_once_the_window_fills():
    m = manager()
    r = FakeReader()
    t = deliver(m, "CAM-T1", r, 1000.0, 120)
    s = snap(m, "CAM-T1", t)
    check("02 availability is 100 only after being measured",
          s["availability_pct"] == 100.0 and s["telemetry"] == "live",
          f"got {s['availability_pct']} / {s['telemetry']}")


def test_03_availability_falls_when_delivery_stops():
    m = manager()
    r = FakeReader()
    t = deliver(m, "CAM-T1", r, 1000.0, 200)
    t = deliver(m, "CAM-T1", r, t, 100, stall=True)   # 100 s of silence
    s = snap(m, "CAM-T1", t)
    ok = s["availability_pct"] is not None and 60.0 <= s["availability_pct"] <= 72.0
    check("03 availability falls when delivery stops", ok,
          f"expected ~66% after 100s silence in a 300s window, got {s['availability_pct']}")


def test_04_availability_never_100_merely_because_a_frame_arrives():
    m = manager()
    r = FakeReader()
    t = deliver(m, "CAM-T1", r, 1000.0, 200)
    t = deliver(m, "CAM-T1", r, t, 100, stall=True)
    t = deliver(m, "CAM-T1", r, t, 2)                 # frames arriving again
    s = snap(m, "CAM-T1", t)
    check("04 a frame arriving now does not reset availability to 100",
          s["availability_pct"] is not None and s["availability_pct"] < 95.0,
          f"got {s['availability_pct']}")


# ─────────────────────────────────────────────────── FPS

def test_05_fps_is_none_without_enough_history():
    m = manager()
    r = FakeReader()
    t = deliver(m, "CAM-T1", r, 1000.0, 1.0)          # < FPS_MIN_SPAN_SECONDS
    s = snap(m, "CAM-T1", t)
    check("05 fps is None below the minimum span", s["actual_fps"] is None,
          f"got {s['actual_fps']}")


def test_06_fps_is_measured_not_configured():
    m = manager()
    r = FakeReader()
    t = deliver(m, "CAM-T1", r, 1000.0, 30, fps=12.0)
    s = snap(m, "CAM-T1", t)
    ok = s["actual_fps"] is not None and abs(s["actual_fps"] - 12.0) <= 1.0
    check("06 fps reflects the delivered rate, not a configured 25", ok,
          f"delivered 12 fps, reported {s['actual_fps']}")


def test_07_counter_reset_clears_the_window_and_reports_no_spike():
    m = manager()
    r = FakeReader()
    t = deliver(m, "CAM-T1", r, 1000.0, 30, fps=25.0)
    before = snap(m, "CAM-T1", t)["actual_fps"]
    r.frames_read = 0                                  # reconnect
    t = deliver(m, "CAM-T1", r, t, 1.0, fps=25.0)
    after = snap(m, "CAM-T1", t)["actual_fps"]
    check("07 a counter reset yields None, never a spike",
          before is not None and after is None,
          f"before={before} after={after} (a spike here would be a fabricated rate)")


def test_08_fps_recovers_after_a_reset():
    m = manager()
    r = FakeReader()
    t = deliver(m, "CAM-T1", r, 1000.0, 30)
    r.frames_read = 0
    t = deliver(m, "CAM-T1", r, t, 1.0)
    t = deliver(m, "CAM-T1", r, t, 20, fps=20.0)
    s = snap(m, "CAM-T1", t)
    ok = s["actual_fps"] is not None and abs(s["actual_fps"] - 20.0) <= 1.5
    check("08 fps recovers after a reset", ok, f"got {s['actual_fps']}")


def test_09_fps_is_none_when_the_counter_itself_is_stale():
    m = manager()
    r = FakeReader()
    t = deliver(m, "CAM-T1", r, 1000.0, 30)
    s = snap(m, "CAM-T1", t + 60.0)                    # nothing observed since
    check("09 fps is None when the counter is stale", s["actual_fps"] is None,
          f"got {s['actual_fps']}")


# ─────────────────────────────────────────────────── image metrics

def test_10_blur_is_none_without_a_baseline():
    m = manager()
    m.process_sample("CAM-T1", metrics.to_sample(structured()), 1000.0)
    s = snap(m, "CAM-T1", 1000.0)
    check("10 blur is None until a baseline exists", s["blur_score"] is None,
          f"got {s['blur_score']} - a number here would be invented")


def test_11_blur_reads_near_100_when_sharpness_matches_baseline():
    m = manager()
    t = 1000.0
    for i in range(40):
        t += 1.0
        m.process_sample("CAM-T1", metrics.to_sample(structured(seed=i)), t)
    s = snap(m, "CAM-T1", t)
    ok = s["blur_score"] is not None and 70.0 <= s["blur_score"] <= 140.0
    check("11 blur is ~100 when sharpness matches the baseline", ok,
          f"got {s['blur_score']}")


def test_12_blur_collapses_when_the_image_is_blurred():
    m = manager()
    t = 1000.0
    for i in range(40):
        t += 1.0
        m.process_sample("CAM-T1", metrics.to_sample(structured(seed=i)), t)
    sharp = snap(m, "CAM-T1", t)["blur_score"]
    t += 1.0
    m.process_sample("CAM-T1", metrics.to_sample(structured(seed=99, blur=15)), t)
    blurred = snap(m, "CAM-T1", t)["blur_score"]
    check("12 blur score collapses under real defocus",
          sharp is not None and blurred is not None and blurred < sharp * 0.5,
          f"sharp={sharp} blurred={blurred}")


def test_13_obstruction_is_percent_of_a_featureless_field():
    m = manager()
    t = 1000.0
    m.process_sample("CAM-T1", metrics.to_sample(structured(seed=3)), t)
    normal = snap(m, "CAM-T1", t)["obstruction_pct"]
    t += 1.0
    m.process_sample("CAM-T1", metrics.to_sample(structured(bright=2)), t)
    covered = snap(m, "CAM-T1", t)["obstruction_pct"]
    check("13 obstruction is the featureless share of the frame",
          normal is not None and covered is not None and normal < 30.0 and covered > 95.0,
          f"normal={normal} covered={covered}")


def test_14_image_metrics_are_none_for_a_camera_never_sampled():
    m = manager()
    r = FakeReader()
    t = deliver(m, "CAM-T1", r, 1000.0, 120)          # stream only, no observe()
    s = snap(m, "CAM-T1", t)
    check("14 image metrics are None when only the stream is seen",
          s["blur_score"] is None and s["obstruction_pct"] is None
          and s["availability_pct"] is not None,
          f"blur={s['blur_score']} obstruction={s['obstruction_pct']}")


def test_15_image_metrics_go_none_when_the_sample_goes_stale():
    m = manager()
    t = 1000.0
    for i in range(40):
        t += 1.0
        m.process_sample("CAM-T1", metrics.to_sample(structured(seed=i)), t)
    fresh = snap(m, "CAM-T1", t)["obstruction_pct"]
    stale = snap(m, "CAM-T1", t + 30.0)["obstruction_pct"]
    check("15 image metrics go None once the sample is stale",
          fresh is not None and stale is None, f"fresh={fresh} stale={stale}")


# ─────────────────────────────────────────────────── status / signal / issue

def test_16_signal_reflects_the_stream_not_a_camera_row():
    m = manager()
    r = FakeReader()
    t = deliver(m, "CAM-T1", r, 1000.0, 20)
    receiving = snap(m, "CAM-T1", t)["signal"]
    t = deliver(m, "CAM-T1", r, t, 20, stall=True)     # > SIGNAL_LOSS_SECONDS
    lost = snap(m, "CAM-T1", t)
    check("16 signal follows the stream, and status goes offline with it",
          receiving == "receiving" and lost["signal"] == "lost"
          and lost["status"] == "offline",
          f"receiving={receiving} then signal={lost['signal']} status={lost['status']}")


def test_17_signal_recovers():
    m = manager()
    r = FakeReader()
    t = deliver(m, "CAM-T1", r, 1000.0, 20)
    t = deliver(m, "CAM-T1", r, t, 20, stall=True)
    t = deliver(m, "CAM-T1", r, t, 5)
    s = snap(m, "CAM-T1", t)
    check("17 signal recovers and status returns to healthy",
          s["signal"] == "receiving" and s["status"] == "healthy",
          f"signal={s['signal']} status={s['status']} issue={s['issue']}")


def test_18_obstruction_shows_as_warning_and_names_the_issue():
    m = manager()
    t = 1000.0
    for i in range(20):
        t += 1.0
        m.process_sample("CAM-T1", metrics.to_sample(structured(seed=i)), t)
    for _ in range(12):                                # hold a covered lens
        t += 1.0
        m.process_sample("CAM-T1", metrics.to_sample(structured(bright=2)), t)
    s = snap(m, "CAM-T1", t)
    check("18 an active condition shows as warning and is named",
          s["status"] == "warning" and s["issue"] and s["issue"] != "None",
          f"status={s['status']} issue={s['issue']}")


def test_19_status_is_none_when_nothing_was_ever_observed():
    m = manager()
    m._states["CAM-T1"] = state_mod.CameraState("CAM-T1", 1000.0)
    s = snap(m, "CAM-T1", 1000.0)
    check("19 a tracked but unobserved camera reports nothing",
          s["status"] is None and s["telemetry"] == "none"
          and s["availability_pct"] is None and s["actual_fps"] is None,
          f"got {s}")


# ─────────────────────────────────────────────────── multi-camera

def test_20_cameras_do_not_contaminate_each_other():
    m = camera_health.CameraHealthManager(shadow=True)
    ids = ["CAM-R09", "CAM-R10", "CAM-R12", "CAM-R16", "CAM-R21", "CAM-R23"]
    m.set_enabled_cameras({c: dict(ALL) for c in ids})
    readers = {c: FakeReader() for c in ids}
    rates = {c: 5.0 * (i + 1) for i, c in enumerate(ids)}

    t = 1000.0
    step = 0.1
    owed = {c: 0.0 for c in ids}

    def tick(now, skip=()):
        for c in ids:
            if c in skip:
                continue
            owed[c] += rates[c] * step
            whole = int(owed[c])
            readers[c].frames_read += whole
            owed[c] -= whole
        m.note_streams(readers, now=now)

    for _ in range(1200):
        t += step
        tick(t)
    # one camera goes dark
    for _ in range(600):
        t += step
        tick(t, skip=("CAM-R16",))

    rows = {r["camera_id"]: r for r in m.snapshot(now=t)}
    ok = len(rows) == len(ids)
    detail = []
    for c in ids:
        r = rows.get(c, {})
        detail.append(f"{c}:fps={r.get('actual_fps')},avail={r.get('availability_pct')}")
        if c == "CAM-R16":
            ok = ok and r.get("status") == "offline"
        else:
            ok = ok and r.get("status") == "healthy"
            ok = ok and r.get("actual_fps") is not None
            ok = ok and abs(r["actual_fps"] - rates[c]) <= max(1.5, rates[c] * 0.12)
    check("20 six cameras keep independent state", ok, " ".join(detail))


def test_21_disarming_a_camera_removes_it_from_the_snapshot():
    m = manager()
    r = FakeReader()
    t = deliver(m, "CAM-T1", r, 1000.0, 90)
    assert snap(m, "CAM-T1", t) is not None
    m.set_enabled_cameras({})
    check("21 a disarmed camera leaves the snapshot", m.snapshot(now=t) == [],
          f"got {m.snapshot(now=t)}")


# ─────────────────────────────────────────────────── publisher

class FakeResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


class FakeSession:
    def __init__(self, behaviour=None):
        self.headers = {}
        self.calls = []
        self.behaviour = behaviour or (lambda payload: FakeResponse(200))

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json, timeout))
        return self.behaviour(json)


def publisher(m, session, **kw):
    return camera_health.HealthTelemetryPublisher(
        m, "http://dash.invalid", "tok", session=session, log=lambda *_a: None, **kw)


def test_22_publisher_sends_one_bulk_request_for_the_estate():
    m = camera_health.CameraHealthManager(shadow=True)
    ids = ["CAM-A", "CAM-B", "CAM-C"]
    m.set_enabled_cameras({c: dict(ALL) for c in ids})
    readers = {c: FakeReader() for c in ids}
    t = 1000.0
    for _ in range(900):
        t += 0.1
        for r in readers.values():
            r.frames_read += 2
        m.note_streams(readers, now=t)
    sess = FakeSession()
    p = publisher(m, sess)
    ok = p.publish_once()
    payload = sess.calls[0][1] if sess.calls else {}
    check("22 one bulk POST carries every camera",
          ok and len(sess.calls) == 1 and len(payload.get("cameras", [])) == 3,
          f"calls={len(sess.calls)} cameras={len(payload.get('cameras', []))}")


def test_23_publisher_survives_an_http_error():
    m = manager()
    sess = FakeSession(lambda payload: FakeResponse(500, "boom"))
    p = publisher(m, sess)
    m._states["CAM-T1"] = state_mod.CameraState("CAM-T1", time.time())
    ok = p.publish_once()
    check("23 an HTTP error is counted, not raised",
          ok is False and p.stats()["failed"] == 1 and p.stats()["errors"] == 0,
          f"{p.stats()}")


def test_24_publisher_survives_a_timeout():
    def boom(payload):
        raise TimeoutError("read timed out")
    m = manager()
    m._states["CAM-T1"] = state_mod.CameraState("CAM-T1", time.time())
    sess = FakeSession(boom)
    p = publisher(m, sess)
    ok = p.publish_once()
    check("24 a network timeout is contained",
          ok is False and p.stats()["failed"] == 1, f"{p.stats()}")


def test_25_publisher_does_not_queue_stale_snapshots():
    """A failed cycle must be forgotten, not retried later."""
    m = manager()
    m._states["CAM-T1"] = state_mod.CameraState("CAM-T1", time.time())
    outcomes = [FakeResponse(500), FakeResponse(200)]
    sess = FakeSession(lambda payload: outcomes.pop(0))
    p = publisher(m, sess)
    p.publish_once()
    p.publish_once()
    check("25 nothing is queued or replayed after a failure",
          len(sess.calls) == 2 and p.stats()["sent"] == 1 and p.stats()["failed"] == 1,
          f"calls={len(sess.calls)} {p.stats()}")


def test_26_publisher_skips_an_empty_estate():
    m = camera_health.CameraHealthManager(shadow=True)
    sess = FakeSession()
    p = publisher(m, sess)
    ok = p.publish_once()
    check("26 nothing is sent when no camera is tracked",
          ok is False and sess.calls == [] and p.stats()["skipped_empty"] == 1,
          f"{p.stats()}")


def test_27_publisher_fault_does_not_kill_the_thread():
    class Exploding:
        def snapshot(self, now=None):
            raise RuntimeError("snapshot fault")
    p = publisher(Exploding(), FakeSession(), interval=0.05)
    p.start()
    time.sleep(0.3)
    alive = p._thread is not None and p._thread.is_alive()
    p.stop()
    check("27 a fault inside the loop leaves the thread running",
          alive and p.stats()["errors"] >= 1, f"alive={alive} {p.stats()}")


def test_28_publication_failure_cannot_block_inference():
    """The frame path must stay fast while every POST hangs."""
    def hang(payload):
        time.sleep(0.5)
        return FakeResponse(200)

    m = manager()
    r = FakeReader()
    deliver(m, "CAM-T1", r, 1000.0, 90)
    p = publisher(m, FakeSession(hang), interval=0.05)
    p.start()
    try:
        frame = structured()
        worst = 0.0
        deadline = time.time() + 1.0
        while time.time() < deadline:
            t0 = time.perf_counter()
            m.observe("CAM-T1", frame)
            m.note_streams({"CAM-T1": r})
            worst = max(worst, (time.perf_counter() - t0) * 1000.0)
    finally:
        p.stop(timeout=2.0)
    check("28 a hanging publisher never blocks the frame path",
          worst < 50.0, f"worst observe+note_streams was {worst:.2f} ms")


def test_29_stop_is_idempotent_and_quick():
    p = publisher(manager(), FakeSession(), interval=0.05)
    p.start()
    t0 = time.perf_counter()
    p.stop()
    p.stop()
    check("29 stop is idempotent and prompt",
          (time.perf_counter() - t0) < 2.0, "stop took too long")


# ─────────────────────────────────────────────────── contract / safety

def test_30_snapshot_is_a_plain_copy():
    m = manager()
    r = FakeReader()
    t = deliver(m, "CAM-T1", r, 1000.0, 90)
    rows = m.snapshot(now=t)
    rows[0]["status"] = "TAMPERED-WITH"
    again = snap(m, "CAM-T1", t)
    check("30 mutating a snapshot cannot reach detector state",
          again["status"] == "healthy", f"got {again['status']}")


def test_31_snapshot_raises_no_events():
    seen = []
    m = camera_health.CameraHealthManager(
        shadow=False, event_sink=lambda *a, **k: seen.append(a))
    m.set_enabled_cameras({"CAM-T1": dict(ALL)})
    r = FakeReader()
    t = deliver(m, "CAM-T1", r, 1000.0, 90)
    before = len(seen)
    for _ in range(5):
        m.snapshot(now=t)
    check("31 taking a snapshot never emits an event", len(seen) == before,
          f"{len(seen) - before} events emitted by snapshot()")


def test_32_thresholds_are_untouched():
    for label, got, want in (("SIGNAL_LOSS_SECONDS", detectors.SIGNAL_LOSS_SECONDS, 5.0),
                             ("OBSTRUCTION_EDGE_MAX", detectors.OBSTRUCTION_EDGE_MAX, 0.15),
                             ("DEFOCUS_RATIO", detectors.DEFOCUS_RATIO, 0.50),
                             ("TAMPER_DISTANCE", detectors.TAMPER_DISTANCE, 0.65)):
        check(f"32 {label} unchanged", got == want, f"{got} != {want}")


def test_33_windows_are_the_approved_values():
    for label, got, want in (("AVAILABILITY_WINDOW_SECONDS", state_mod.AVAILABILITY_WINDOW_SECONDS, 300),
                             ("AVAILABILITY_MIN_SLOTS", state_mod.AVAILABILITY_MIN_SLOTS, 60),
                             ("FPS_WINDOW_SECONDS", state_mod.FPS_WINDOW_SECONDS, 10.0),
                             ("FPS_MIN_SPAN_SECONDS", state_mod.FPS_MIN_SPAN_SECONDS, 2.0),
                             ("PUBLISH_INTERVAL_SECONDS", camera_health.PUBLISH_INTERVAL_SECONDS, 15.0)):
        check(f"33 {label} is the approved value", got == want, f"{got} != {want}")


def main():
    print("=" * 66)
    print("Camera Health current-state telemetry")
    print("=" * 66)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:                      # noqa: BLE001
                check(name, False, f"raised {type(exc).__name__}: {exc}")
    print("=" * 66)
    print(f"{PASSED} passed, {len(FAILURES)} failed")
    for n, d in FAILURES:
        print(f"  FAILED {n}: {d}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
