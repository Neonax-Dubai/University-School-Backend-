"""
Offline replay of simultaneous C101 clips through the production occupancy path - isolated, writes
nothing but its own output files.

    python3 replay_occupancy_clips.py --clip camera_01=A.mp4 --clip camera_02=B.mp4 --clip camera_03=C.mp4 \
        --zones zones.json --capacity 41 --out-dir /out [--overlays 30]

WHAT IT REPRODUCES (zayed_inference.py process_result, per camera, per inferred frame)
  * the clips in lock-step, one inferred frame per camera every 3rd stream frame (TARGET_INFERENCE_FPS
    10 of 30), predicted as one batch like the live loop;
  * YOLO26-L TensorRT at 640 with the live confidence floor, then the camera's person floor (0.45);
  * one shared tracking.TrackManager with the live class groups, then the live EMA box smoothing;
  * occupancy.ClassroomOccupancy.observe / snapshot / decisions - the Phase 4 module, configured
    exactly as the dashboard's /api/ai/cameras/ payload would configure it (classroom, zones).

WHAT IT RECORDS
  every 10 s of video: the published occupancy row (method, per_camera, occupancy, %), next to each
  camera's full-frame person count - so what camera_03 would have added is visible, not assumed;
  per camera: tracks, zone membership changes, prediction failures, predict latency and the cost of
  the zone test itself.

ISOLATION
  Run it with --network none. It imports no events/outbox/evidence/dashboard code, opens no socket
  and writes only to --out-dir. Overlays contain people: keep --out-dir private and delete it after.
"""
import argparse
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2  # noqa: E402

import abandoned  # noqa: E402
import object_logger  # noqa: E402
import occupancy  # noqa: E402
import phone_use  # noqa: E402
import tracking  # noqa: E402
import zones as zone_geometry  # noqa: E402

TARGET_FPS = 10.0
PREDICT_CONF = 0.25          # zayed_inference.predict_confidence() with abandoned + phone armed
CAMERA_CONF = 0.45           # the cameras' detection_confidence
SMOOTH_ALPHA = 0.45
PERSON_CLASSES = {0}
CARRIED_CLASSES = set(abandoned.CANDIDATE_CLASS_IDS)
PHONE_CLASSES = {phone_use.PHONE_CLASS_ID}


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q / 100.0 * (len(values) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", required=True, help="camera_id=path.mp4 (repeat)")
    ap.add_argument("--zones", required=True, help='{"camera_01": [{"name", "coordinates", "rules"}], ...}')
    ap.add_argument("--classroom", default="C101")
    ap.add_argument("--capacity", type=int, default=41)
    ap.add_argument("--threshold-pct", type=float, default=100.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overlays", type=float, default=0, help="save an overlay every N video seconds (0 = none)")
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--detect-engine", default="/engines/zayed/current/yolo26l.engine")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    clips = dict(c.split("=", 1) for c in args.clip)
    zone_config = json.load(open(args.zones))
    configs = []
    for camera_id in clips:
        configs.append(SimpleNamespace(
            camera_id=camera_id, features={"people_counting": True},
            classroom={"classroom_id": args.classroom, "capacity": args.capacity,
                       "overcrowding_threshold_pct": args.threshold_pct, "floor_plan": {}},
            placement={"floor_plan_homography": {}},
            zones=[dict(z, id=i + 1, type="custom", enabled=z.get("enabled", True))
                   for i, z in enumerate(zone_config.get(camera_id, []))]))
    occ = occupancy.ClassroomOccupancy()
    occ.configure(configs)
    polygons = {c.camera_id: [z["coordinates"] for z in c.zones if (z.get("rules") or {}).get("occupancy")]
                for c in configs}

    from ultralytics import YOLO
    detector = YOLO(args.detect_engine, task="detect")
    tracker = tracking.TrackManager(
        person_classes=PERSON_CLASSES, vehicle_classes=set(),
        class_names=dict(enumerate(object_logger.COCO_NAMES)),
        object_classes=CARRIED_CLASSES | PHONE_CLASSES | object_logger.LOGGABLE_CLASS_IDS)

    caps = {cam: cv2.VideoCapture(path) for cam, path in clips.items()}
    fps = {cam: cap.get(cv2.CAP_PROP_FPS) or 30.0 for cam, cap in caps.items()}
    step = max(1, int(round(min(fps.values()) / TARGET_FPS)))
    cameras = list(clips)

    smooth, full_window, index = {}, {c: [] for c in cameras}, 0
    per_cam = {c: {"frames": 0, "tracks": set(), "zone_tracks": set(), "membership": {}, "changes": 0,
                   "max_in_zone": 0, "max_full_frame": 0, "observe_us": []} for c in cameras}
    predict_ms, predict_failures, snapshots, events, frames_log = [], 0, [], [], []
    next_snapshot, next_overlay = 10.0, 0.0
    started = time.perf_counter()
    done = False
    while not done:
        frames = {}
        for cam, cap in caps.items():
            if (index % step) == args.phase:
                ok, frame = cap.read()
                if ok:
                    frames[cam] = frame
                else:
                    done = True
            elif not cap.grab():
                done = True
        t = index / fps[cameras[0]]
        index += 1
        if done or not frames:
            continue
        batch = [frames[c] for c in cameras if c in frames]
        t0 = time.perf_counter()
        try:
            results = detector.predict(source=batch, imgsz=640, conf=PREDICT_CONF, device="cuda:0", verbose=False)
        except Exception as exc:                          # noqa: BLE001 - counted like the live loop
            predict_failures += 1
            print(f"predict failed at t={t:.1f}: {type(exc).__name__}: {exc}", flush=True)
            continue
        predict_ms.append((time.perf_counter() - t0) * 1000.0)
        step_log = {"t": round(t, 2)}
        for cam, result in zip([c for c in cameras if c in frames], results):
            frame = frames[cam]
            height, width = frame.shape[:2]
            detections = []
            if result.boxes is not None and len(result.boxes):
                for cls, conf, box in zip(result.boxes.cls.cpu().numpy().astype(int),
                                          result.boxes.conf.cpu().numpy(), result.boxes.xyxy.cpu().numpy()):
                    cls, conf = int(cls), float(conf)
                    if cls not in PERSON_CLASSES | CARRIED_CLASSES or cls in PHONE_CLASSES:
                        continue
                    if conf < CAMERA_CONF and not (cls in CARRIED_CLASSES and conf >= abandoned.OBJECT_CONFIDENCE):
                        continue
                    detections.append((cls, conf, tuple(int(v) for v in box)))
            tracked = tracker.update(cam, detections, now=t)
            for tr in tracked:
                key = (cam, tr.track_id)
                rx1, ry1, rx2, ry2 = tr.bbox
                prev = smooth.get(key)
                s = ([float(rx1), float(ry1), float(rx2), float(ry2)] if prev is None else
                     [SMOOTH_ALPHA * v + (1 - SMOOTH_ALPHA) * p for v, p in zip((rx1, ry1, rx2, ry2), prev)])
                smooth[key] = s
                tr.bbox = [int(v) for v in s]
            active = {(cam, tr.track_id) for tr in tracked}
            for gone in [k for k in smooth if k[0] == cam and k not in active]:
                del smooth[gone]

            o0 = time.perf_counter()
            occ.observe(cam, tracked, t, width, height)
            per_cam[cam]["observe_us"].append((time.perf_counter() - o0) * 1e6)

            stats = per_cam[cam]
            stats["frames"] += 1
            persons, in_zone = [], 0
            for tr in tracked:
                if tr.group != tracking.PERSON_GROUP:
                    continue
                point = zone_geometry.anchor_point(tr.bbox, width, height)
                inside = bool(point and any(zone_geometry.point_in_polygon(point[0], point[1], p)
                                            for p in polygons.get(cam, [])))
                stats["tracks"].add(tr.track_id)
                if inside:
                    stats["zone_tracks"].add(tr.track_id)
                    in_zone += 1
                if stats["membership"].get(tr.track_id, inside) != inside:
                    stats["changes"] += 1
                stats["membership"][tr.track_id] = inside
                persons.append({"track": tr.track_id, "bbox": list(tr.bbox), "in_zone": inside})
            stats["max_in_zone"] = max(stats["max_in_zone"], in_zone)
            stats["max_full_frame"] = max(stats["max_full_frame"], len(persons))
            full_window[cam].append((t, len(persons)))
            step_log[cam] = persons

            if args.overlays and t >= next_overlay:
                img = frame.copy()
                for polygon in polygons.get(cam, []):
                    pts = [(int(x * width), int(y * height)) for x, y in polygon]
                    for a, b in zip(pts, pts[1:] + pts[:1]):
                        cv2.line(img, a, b, (0, 255, 255), 6)
                for p in persons:
                    x1, y1, x2, y2 = p["bbox"]
                    colour = (0, 200, 0) if p["in_zone"] else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 6)
                    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 14, colour, -1)
                label = f"{cam} t={t:.0f}s counted={in_zone}" if polygons.get(cam) else f"{cam} t={t:.0f}s NOT COUNTED"
                cv2.putText(img, label, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)
                os.makedirs(os.path.join(args.out_dir, "overlays"), exist_ok=True)
                cv2.imwrite(os.path.join(args.out_dir, "overlays", f"t{int(t):04d}_{cam}.jpg"),
                            cv2.resize(img, (width // 3, height // 3)))
        frames_log.append(step_log)
        if args.overlays and t >= next_overlay:
            next_overlay += args.overlays

        for kind, cid, meta in occ.decisions(t):
            events.append({"t": round(t, 2), "event_type": kind, "classroom_id": cid, **meta})
        if t >= next_snapshot:
            next_snapshot += 10.0
            [row] = [r for r in occ.snapshot(t) if r["classroom_id"] == args.classroom]
            full = {}
            for cam in cameras:
                window = [n for ts, n in full_window[cam] if t - ts <= occupancy.WINDOW_SECONDS]
                full_window[cam] = [(ts, n) for ts, n in full_window[cam] if t - ts <= occupancy.WINDOW_SECONDS]
                full[cam] = int(statistics.median(window)) if window else None
            snapshots.append({
                "t": round(t, 1), "method": row["method"], "occupancy": row["occupancy"],
                "occupancy_max": row["occupancy_max"], "per_camera": row["per_camera"],
                "occupancy_pct": round(100.0 * row["occupancy"] / args.capacity, 1) if row["occupancy"] is not None else None,
                "full_frame_median": full,
                "sum_if_every_camera_counted_full_frame": sum(v for v in full.values() if v is not None),
                "camera_03_in_per_camera": "camera_03" in row["per_camera"]})

    wall = time.perf_counter() - started
    for cap in caps.values():
        cap.release()
    summary = {
        "clips": clips, "zones": zone_config, "capacity": args.capacity, "threshold_pct": args.threshold_pct,
        "stride": step, "phase": args.phase, "wall_seconds": round(wall, 1),
        "inferred_steps": len(predict_ms), "prediction_failures": predict_failures,
        "throughput_camera_frames_per_second": round(sum(s["frames"] for s in per_cam.values()) / wall, 1),
        "predict_batch_ms": {"p50": round(pct(predict_ms, 50), 1), "p95": round(pct(predict_ms, 95), 1),
                             "max": round(max(predict_ms), 1)} if predict_ms else None,
        "cameras": {cam: {
            "frames": s["frames"], "person_tracks": len(s["tracks"]), "tracks_ever_in_zone": len(s["zone_tracks"]),
            "zone_membership_changes": s["changes"], "max_in_zone": s["max_in_zone"],
            "max_full_frame": s["max_full_frame"], "has_occupancy_zone": bool(polygons.get(cam)),
            "observe_us": {"p50": round(pct(s["observe_us"], 50), 1), "p95": round(pct(s["observe_us"], 95), 1),
                           "max": round(max(s["observe_us"]), 1)} if s["observe_us"] else None}
            for cam, s in per_cam.items()},
        "snapshots": len(snapshots),
        "occupancy_range": [min((s["occupancy"] for s in snapshots if s["occupancy"] is not None), default=None),
                            max((s["occupancy"] for s in snapshots if s["occupancy"] is not None), default=None)],
        "camera_03_ever_in_classroom_row": any(s["camera_03_in_per_camera"] for s in snapshots),
        "events": {k: sum(1 for e in events if e["event_type"] == k) for k in {e["event_type"] for e in events}},
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "snapshots.jsonl"), "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in snapshots)
    with open(os.path.join(args.out_dir, "events.jsonl"), "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in events)
    with open(os.path.join(args.out_dir, "frames.jsonl"), "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in frames_log)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
