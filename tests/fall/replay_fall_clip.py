"""
Offline replay of one camera clip through the production Fall path - isolated, writes nothing
but its own output files.

    python3 replay_fall_clip.py --clip CLIP.mp4 --sidecar CLIP.json \
        --baseline-policy /baseline/fall_pose_policy.py [--candidate-policy /app/fall_pose_policy.py] \
        --out-dir /out [--phase 0|1|2|all]

WHAT IT REPRODUCES (zayed_inference.py, per camera, per inferred frame)
  * frames spaced at TARGET_INFERENCE_FPS (10) out of the ~30 fps stream; `--phase` picks which
    of the three interleaved frame sets, because the live loop's phase is not recorded;
  * YOLO26-L TensorRT at 640 with the live confidence floor (0.25), then the live per-class
    filters: person >= the camera's 0.45, bags >= abandoned.OBJECT_CONFIDENCE, phones kept out of
    the tracker;
  * tracking.TrackManager with the live class groups, then the live EMA box smoothing (0.45);
  * fall_pose_adapter.FallPoseAdapter - its candidate gate, hot window and one-to-one pose
    association - on the shared YOLO26L-pose @960 runtime, run synchronously (the live queue of
    8 has never dropped a frame: fall_pose_dropped_queue_full = 0).

WHAT IT COMPARES
  Every pose observation the adapter makes is handed to TWO policies with identical arguments:
  the baseline (the file production runs today) and, when given, the candidate. The adapter itself
  sees the candidate's result. Both decisions are written side by side, so any difference between
  them is the policy and nothing else.

ISOLATION
  Run it in a container with --network none. It imports no events/outbox/evidence/dashboard code,
  opens no socket and writes only to --out-dir.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timedelta

APP = os.environ.get("REPLAY_APP_DIR", "/app")
sys.path.insert(0, APP)

import cv2  # noqa: E402

import abandoned  # noqa: E402
import fall_pose_adapter  # noqa: E402
import object_logger  # noqa: E402
import phone_use  # noqa: E402
import tracking  # noqa: E402

TARGET_FPS = 10.0
PREDICT_CONF = 0.25          # zayed_inference.predict_confidence() with abandoned + phone armed
CAMERA_CONF = 0.45           # the camera's detection_confidence (live API, 2026-09-29)
SMOOTH_ALPHA = 0.45          # zayed_inference.SMOOTH_ALPHA default
PERSON_CLASSES = {0}
CARRIED_CLASSES = set(abandoned.CANDIDATE_CLASS_IDS)
PHONE_CLASSES = {phone_use.PHONE_CLASS_ID}
KNEES_ANKLES = (13, 14, 15, 16)
HIPS = (11, 12)


def load_policy_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _r(v, nd=2):
    if v is None:
        return None
    if isinstance(v, (tuple, list)):
        return [_r(x, nd) for x in v]
    return round(float(v), nd)


class TeePolicy:
    """Hands every observation to the baseline and the candidate policy; records both."""

    def __init__(self, baseline, candidate, sink, wall_of):
        self.baseline, self.candidate, self.sink, self.wall_of = baseline, candidate, sink, wall_of
        self.uncertain = 0

    def observe(self, camera_id, track_id, kpts, confs, bbox, observed_at=None, pose_track_iou=None,
                frame_width=None, frame_height=None):
        kw = dict(observed_at=observed_at, pose_track_iou=pose_track_iou,
                  frame_width=frame_width, frame_height=frame_height)
        rb = self.baseline.observe(camera_id, track_id, kpts, confs, bbox, **kw)
        rc = self.candidate.observe(camera_id, track_id, kpts, confs, bbox, **kw) if self.candidate else None
        m = rb.measurements
        row = {
            "t": round(observed_at, 3), "wall": self.wall_of(observed_at), "track": track_id,
            "bbox": [int(v) for v in bbox], "iou": _r(pose_track_iou, 3),
            "kpts": m.get("confident_keypoints"),
            "knee_ankle_conf": [_r(confs[i], 2) for i in KNEES_ANKLES],
            "hip_conf": [_r(confs[i], 2) for i in HIPS],
            "torso": _r(m.get("torso_angle")), "trunk": _r(m.get("trunk_angle")),
            "ar": _r(m.get("body_aspect_ratio"), 3),
            "shoulder": _r(m.get("shoulder_center"), 1), "hip": _r(m.get("hip_center"), 1),
            "knee": _r(m.get("knee_center"), 1), "ankle": _r(m.get("ankle_center"), 1),
            "base_state": rb.state, "base_event": rb.event is not None, "base_reason": rb.reason,
        }
        if rc is not None:
            extra = {k: (_r(v, 3) if isinstance(v, float) else v) for k, v in rc.measurements.items()
                     if k not in m}
            row.update({"cand_state": rc.state, "cand_event": rc.event is not None,
                        "cand_reason": rc.reason, "cand_extra": extra})
            if rc.event is not None:
                row["cand_event_metadata"] = rc.event.metadata()
        if rb.event is not None:
            row["base_event_metadata"] = rb.event.metadata()
        self.sink(row)
        return rc if rc is not None else rb

    def prune(self, now=None):
        self.baseline.prune(now)
        if self.candidate:
            self.candidate.prune(now)

    def stats(self):
        return {"baseline": self.baseline.stats(),
                "candidate": self.candidate.stats() if self.candidate else None}


def run_phase(args, phase, detector, baseline_mod, candidate_mod, start_wall):
    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.clip}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / TARGET_FPS)))

    def wall_of(t):
        return (start_wall + timedelta(seconds=t)).strftime("%H:%M:%S.%f")[:-3]

    rows, frames_log = [], []
    tee = TeePolicy(baseline_mod.FallPosePolicy(), candidate_mod.FallPosePolicy() if candidate_mod else None,
                    rows.append, wall_of)
    adapter = fall_pose_adapter.FallPoseAdapter(enabled=False, decision=tee)
    adapter.enabled = True                   # synchronous: no worker thread, _process called inline
    adapter.set_enabled_cameras([args.camera])
    tracker = tracking.TrackManager(
        person_classes=PERSON_CLASSES, vehicle_classes=set(),
        class_names=dict(enumerate(object_logger.COCO_NAMES)),
        object_classes=CARRIED_CLASSES | PHONE_CLASSES | object_logger.LOGGABLE_CLASS_IDS)
    smooth = {}
    index = inferred = submitted = 0
    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = index / fps
        index += 1
        if (index - 1) % step != phase:
            continue
        inferred += 1
        result = detector.predict(source=[frame], imgsz=640, conf=PREDICT_CONF, device="cuda:0",
                                  verbose=False)[0]
        detections = []
        if result.boxes is not None and len(result.boxes):
            for cls, conf, box in zip(result.boxes.cls.cpu().numpy().astype(int),
                                      result.boxes.conf.cpu().numpy(), result.boxes.xyxy.cpu().numpy()):
                cls, conf = int(cls), float(conf)
                if cls not in PERSON_CLASSES | CARRIED_CLASSES | PHONE_CLASSES:
                    continue
                if cls in PHONE_CLASSES:          # phones go to the phone analytic, never the tracker
                    continue
                if conf < CAMERA_CONF and not (cls in CARRIED_CLASSES and conf >= abandoned.OBJECT_CONFIDENCE):
                    continue
                detections.append((cls, conf, tuple(int(v) for v in box)))
        tracked = tracker.update(args.camera, detections, now=t)
        for tr in tracked:                       # zayed_inference.py EMA box smoothing, verbatim maths
            key = tr.track_id
            rx1, ry1, rx2, ry2 = tr.bbox
            prev = smooth.get(key)
            s = ([float(rx1), float(ry1), float(rx2), float(ry2)] if prev is None else
                 [SMOOTH_ALPHA * rx1 + (1 - SMOOTH_ALPHA) * prev[0], SMOOTH_ALPHA * ry1 + (1 - SMOOTH_ALPHA) * prev[1],
                  SMOOTH_ALPHA * rx2 + (1 - SMOOTH_ALPHA) * prev[2], SMOOTH_ALPHA * ry2 + (1 - SMOOTH_ALPHA) * prev[3]])
            smooth[key] = s
            tr.bbox = [int(v) for v in s]
        for gone in [k for k in smooth if k not in {tr.track_id for tr in tracked}]:
            del smooth[gone]
        persons = [tr for tr in tracked if tr.group == tracking.PERSON_GROUP]
        frames_log.append({"t": round(t, 3), "wall": wall_of(t),
                           "persons": [{"track": p.track_id, "bbox": list(p.bbox),
                                        "ar": round((p.bbox[2] - p.bbox[0]) / max(1, p.bbox[3] - p.bbox[1]), 3)}
                                       for p in persons]})
        if adapter.submit(args.camera, frame, tracked, timestamp=t, frame_id=index - 1):
            submitted += 1
            adapter._process(adapter._queue.get_nowait())
    cap.release()
    return {
        "phase": phase, "fps": round(fps, 3), "frames_decoded": index, "frames_inferred": inferred,
        "pose_submissions": submitted, "observations": len(rows),
        "seconds": round(time.perf_counter() - t_start, 1),
        "adapter": {k: v for k, v in adapter.stats().items() if k.startswith("fall_pose_")},
        "baseline_events": [r for r in rows if r["base_event"]],
        "candidate_events": [r for r in rows if r.get("cand_event")],
    }, rows, frames_log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--sidecar", required=True, help="extract_clip.py JSON (gives the wall-clock start)")
    ap.add_argument("--camera", default="camera_01")
    ap.add_argument("--baseline-policy", required=True)
    ap.add_argument("--candidate-policy")
    ap.add_argument("--detect-engine", default="/engines/zayed/current/yolo26l.engine")
    ap.add_argument("--phase", default="all")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    side = json.load(open(args.sidecar))
    start_wall = datetime.strptime(side["requested_start"], "%Y-%m-%d %H:%M:%S")
    baseline_mod = load_policy_module(args.baseline_policy, "fall_pose_policy_baseline")
    candidate_mod = (load_policy_module(args.candidate_policy, "fall_pose_policy_candidate")
                     if args.candidate_policy else None)

    from ultralytics import YOLO
    detector = YOLO(args.detect_engine, task="detect")
    phases = [0, 1, 2] if args.phase == "all" else [int(args.phase)]
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {"clip": os.path.basename(args.clip), "camera": args.camera,
               "wall_start": side["requested_start"], "timezone": side.get("timezone"),
               "baseline_policy": args.baseline_policy, "candidate_policy": args.candidate_policy,
               "phases": []}
    for phase in phases:
        result, rows, frames_log = run_phase(args, phase, detector, baseline_mod, candidate_mod, start_wall)
        with open(os.path.join(args.out_dir, f"observations_phase{phase}.jsonl"), "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=str) + "\n")
        with open(os.path.join(args.out_dir, f"tracks_phase{phase}.jsonl"), "w") as handle:
            for row in frames_log:
                handle.write(json.dumps(row) + "\n")
        summary["phases"].append({k: v for k, v in result.items()
                                  if k not in ("baseline_events", "candidate_events")} | {
            "baseline_events": [{k: e[k] for k in ("t", "wall", "track", "bbox", "torso", "trunk", "ar", "kpts")}
                                for e in result["baseline_events"]],
            "candidate_events": [{k: e[k] for k in ("t", "wall", "track", "bbox", "torso", "trunk", "ar", "kpts")}
                                 for e in result["candidate_events"]]})
        print(f"phase {phase}: inferred={result['frames_inferred']} pose={result['pose_submissions']} "
              f"obs={result['observations']} baseline_events={len(result['baseline_events'])} "
              f"candidate_events={len(result['candidate_events']) if candidate_mod else '-'} "
              f"({result['seconds']}s)", flush=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
