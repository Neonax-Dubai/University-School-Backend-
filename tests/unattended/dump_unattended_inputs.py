"""
Stage 1 of the unattended-object replay: run recorded clips of ONE camera through the production
detection and tracking path and dump, per inferred frame, exactly what zayed_inference.py hands to
abandoned.AbandonedObjectDetector.update() - the carried-object tracks and the person boxes.

    python3 dump_unattended_inputs.py --camera camera_02 \
        --clip /clips/1.mp4@0 --clip /clips/2.mp4@600 --out /out/camera_02_A.jsonl.gz

Each --clip is PATH@OFFSET: the clip's start, in seconds from the first clip's start, so a sequence
of contiguous NVR clips keeps one clock (and one tracker) across the files.

WHAT IT REPRODUCES (zayed_inference.py process_result, one camera)
  * one inferred frame in three (TARGET_INFERENCE_FPS 10 of ~30), per clip;
  * YOLO26-L TensorRT at 640, predict floor = predict_confidence() with unattended + phone armed;
  * the camera's allowed classes (person, the unattended classes, phone - object_detection is OFF in
    production), person >= the camera's 0.45, carried >= abandoned.OBJECT_CONFIDENCE, phones kept
    out of the tracker;
  * tracking.TrackManager with the live class groups, then the live EMA box smoothing;
  * the analytic's inputs: carried tracks of abandoned.CANDIDATE_CLASS_IDS and the boxes of the
    person tracks of that frame.
Stage 2 (evaluate_unattended.py) replays the dump through any version of abandoned.py, CPU only.

ISOLATION
  Run with --network none. Imports no events/outbox/evidence/dashboard code, opens no socket and
  writes only --out.
"""
import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2  # noqa: E402

import abandoned  # noqa: E402
import object_logger  # noqa: E402
import phone_use  # noqa: E402
import tracking  # noqa: E402

TARGET_FPS = 10.0
CAMERA_CONF = 0.45
SMOOTH_ALPHA = 0.45
PERSON_CLASSES = {0}
CARRIED_CLASSES = set(abandoned.CANDIDATE_CLASS_IDS)
PHONE_CLASSES = {phone_use.PHONE_CLASS_ID}
PREDICT_CONF = min(CAMERA_CONF, abandoned.OBJECT_CONFIDENCE, phone_use.PHONE_CONFIDENCE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True)
    ap.add_argument("--clip", action="append", required=True, help="PATH@OFFSET_SECONDS (repeat, in order)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--detect-engine", default="/engines/zayed/current/yolo26l.engine")
    args = ap.parse_args()

    from ultralytics import YOLO
    detector = YOLO(args.detect_engine, task="detect")
    tracker = tracking.TrackManager(
        person_classes=PERSON_CLASSES, vehicle_classes=set(),
        class_names=dict(enumerate(object_logger.COCO_NAMES)),
        object_classes=CARRIED_CLASSES | PHONE_CLASSES | object_logger.LOGGABLE_CLASS_IDS)
    allowed = PERSON_CLASSES | CARRIED_CLASSES | PHONE_CLASSES
    smooth, predict_ms, failures, inferred, clips = {}, [], 0, 0, []
    started = time.perf_counter()
    with gzip.open(args.out, "wt") as out:
        for spec in args.clip:
            path, offset = spec.rsplit("@", 1)
            offset = float(offset)
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(round(fps / TARGET_FPS)))
            index = decoded = 0
            while True:
                if index % step != args.phase:
                    if not cap.grab():
                        break
                    index += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                t = offset + index / fps
                index += 1
                decoded += 1
                height, width = frame.shape[:2]
                t0 = time.perf_counter()
                try:
                    result = detector.predict(source=[frame], imgsz=640, conf=PREDICT_CONF, device="cuda:0",
                                              verbose=False)[0]
                except Exception as exc:                  # noqa: BLE001 - counted like the live loop
                    failures += 1
                    print(f"predict failed at t={t:.1f}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                predict_ms.append((time.perf_counter() - t0) * 1000.0)
                inferred += 1
                detections = []
                if result.boxes is not None and len(result.boxes):
                    for cls, conf, box in zip(result.boxes.cls.cpu().numpy().astype(int),
                                              result.boxes.conf.cpu().numpy(), result.boxes.xyxy.cpu().numpy()):
                        cls, conf = int(cls), float(conf)
                        if cls not in allowed:
                            continue
                        if cls in PHONE_CLASSES and conf >= phone_use.PHONE_CONFIDENCE:
                            continue                     # phones go to the phone analytic, never the tracker
                        if conf < CAMERA_CONF and not (cls in CARRIED_CLASSES and conf >= abandoned.OBJECT_CONFIDENCE):
                            continue
                        detections.append((cls, conf, tuple(int(v) for v in box)))
                tracked = tracker.update(args.camera, detections, now=t)
                for tr in tracked:                       # zayed_inference.py EMA box smoothing, verbatim maths
                    key = tr.track_id
                    rx1, ry1, rx2, ry2 = tr.bbox
                    prev = smooth.get(key)
                    s = ([float(rx1), float(ry1), float(rx2), float(ry2)] if prev is None else
                         [SMOOTH_ALPHA * v + (1 - SMOOTH_ALPHA) * p for v, p in zip((rx1, ry1, rx2, ry2), prev)])
                    smooth[key] = s
                    tr.bbox = [int(v) for v in s]
                for gone in [k for k in smooth if k not in {tr.track_id for tr in tracked}]:
                    del smooth[gone]
                objects = [[tr.track_id, tr.raw_track_id, tr.class_id, tr.class_name, round(tr.confidence, 3),
                            *tr.bbox] for tr in tracked
                           if tr.group == tracking.OBJECT_GROUP and tr.class_id in abandoned.CANDIDATE_CLASS_IDS]
                persons = [list(tr.bbox) for tr in tracked if tr.group == tracking.PERSON_GROUP]
                out.write(json.dumps({"t": round(t, 3), "w": width, "h": height, "objects": objects,
                                      "persons": persons}) + "\n")
            cap.release()
            clips.append({"clip": os.path.basename(path), "offset": offset, "fps": round(fps, 3),
                          "frames_read": index, "frames_inferred": decoded})
    wall = time.perf_counter() - started
    predict_ms.sort()
    summary = {"camera": args.camera, "clips": clips, "inferred": inferred, "prediction_failures": failures,
               "wall_seconds": round(wall, 1), "frames_per_second": round(inferred / wall, 1) if wall else None,
               "predict_ms_p50": round(predict_ms[len(predict_ms) // 2], 1) if predict_ms else None,
               "predict_ms_p95": round(predict_ms[int(len(predict_ms) * 0.95)], 1) if predict_ms else None}
    with open(args.out.replace(".jsonl.gz", ".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
