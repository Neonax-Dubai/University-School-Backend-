"""
Stage 2 of the unattended-object replay: feed one camera's dump (dump_unattended_inputs.py) through
the BASELINE abandoned.py (what production runs) and the CANDIDATE (this tree's), with identical
inputs on the recording's own clock, and count what each would put in the operator's queue.

    python3 evaluate_unattended.py --dump c02_B.jsonl.gz --camera camera_02 \
        --baseline /baseline/abandoned.py --out /out/c02_B [--start-wall 15:30:00] [--set NAME=VALUE ...]

Downstream of the detector it emulates, per event:
  baseline   events.py (debounce key camera + TRACK id + type, 30 s) and the dashboard
             (ALARM_RULES: one alarm per event);
  candidate  events.py (debounce keyed by EPISODE) and the dashboard's unattended handling (one
             alarm per episode; a resolution closes it; an open alarm at the same place folds a
             re-created episode).
It also writes the statistics the association thresholds were chosen from: track hand-offs (gap,
distance, size ratio between a track that ends and the next one at that place), class flips, and the
person-attendance runs at each episode's resting position. CPU only, no network, writes only --out.
"""
import argparse
import gzip
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import types
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

DEBOUNCE_SECONDS = 30.0          # events.ZONE_COOLDOWN_SECONDS, the abandoned_object cooldown
FOLD_IOU = 0.3                   # zayed dashboard: an open alarm at the same place is the same object


class Obj:
    group = "object"

    def __init__(self, camera_id, row):
        self.camera_id = camera_id
        self.track_id, self.raw_track_id, self.class_id, self.class_name, self.confidence = row[:5]
        self.bbox = list(row[5:9])


def load(path, name, clock=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if clock is not None:                # the baseline reads time.monotonic(): give it the video clock
        module.time = types.SimpleNamespace(monotonic=lambda: clock[0])
    return module


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def wall(t, start):
    if start is None:
        return f"t={t:.0f}s"
    h, m, s = (int(x) for x in start.split(":"))
    total = h * 3600 + m * 60 + s + int(t)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))] if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--baseline", required=True, help="the production abandoned.py")
    ap.add_argument("--candidate", default=os.path.abspath(os.path.join(HERE, "..", "..", "abandoned.py")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-wall", default=None, help="NVR local time of t=0, HH:MM:SS (labels only)")
    ap.add_argument("--set", action="append", default=[], help="override a candidate module constant")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    clock = [0.0]
    base = load(args.baseline, "abandoned_baseline", clock)
    cand = load(args.candidate, "abandoned_candidate")
    for item in args.set:
        key, value = item.split("=", 1)
        setattr(cand, key, type(getattr(cand, key))(value) if not isinstance(getattr(cand, key), bool)
                else value in ("1", "true", "True"))
    features = {args.camera: {"abandoned_object": True}}
    base_det = base.AbandonedObjectDetector(features, dwell_seconds=base.DWELL_SECONDS)
    cand_det = cand.AbandonedObjectDetector(features, dwell_seconds=cand.DWELL_SECONDS)

    base_debounce, base_alarms, base_events = {}, [], []
    cand_debounce, cand_events, alarms = {}, [], []          # alarms: dicts, the dashboard emulation
    tracks = {}                                              # track_id -> stats
    base_us, cand_us, frames = [], [], 0
    seen_episodes = {}                                       # episode_id -> latest snapshot
    persons_by_t = []

    with gzip.open(args.dump, "rt") as f:
        for line in f:
            row = json.loads(line)
            t, w, h = row["t"], row["w"], row["h"]
            objs = [Obj(args.camera, r) for r in row["objects"]]
            persons = row["persons"]
            frames += 1
            persons_by_t.append((t, persons))
            for o in objs:
                s = tracks.setdefault(o.track_id, {"first": t, "last": t, "classes": defaultdict(int), "n": 0,
                                                   "first_box": o.bbox, "last_box": o.bbox, "conf": []})
                s["last"], s["last_box"], s["n"] = t, o.bbox, s["n"] + 1
                s["classes"][o.class_name] += 1
                s["conf"].append(o.confidence)

            # ---- baseline: production detector -> per-track debounce -> one alarm per event
            clock[0] = t
            t0 = time.perf_counter()
            fired = base_det.update(args.camera, objs, persons, w, h)
            base_us.append((time.perf_counter() - t0) * 1e6)
            for o, meta in fired:
                key = (args.camera, o.track_id)
                if key in base_debounce and t - base_debounce[key] < DEBOUNCE_SECONDS:
                    continue
                base_debounce[key] = t
                base_events.append({"t": t, "wall": wall(t, args.start_wall), "track_id": o.track_id,
                                    "class": o.class_name, "bbox": o.bbox, "dwell": meta["dwell_seconds"]})
                base_alarms.append(base_events[-1])

            # ---- candidate: episode detector -> episode debounce -> dashboard episode handling
            t0 = time.perf_counter()
            fired = cand_det.update(args.camera, objs, persons, w, h, now=t)
            resolved = cand_det.drain_resolutions(args.camera)
            cand_us.append((time.perf_counter() - t0) * 1e6)
            for snapshot in cand_det.episodes(args.camera):
                seen_episodes[snapshot["episode_id"]] = snapshot
            for o, meta in fired:
                key = (args.camera, meta["episode_id"])
                if key in cand_debounce and t - cand_debounce[key] < DEBOUNCE_SECONDS:
                    continue
                cand_debounce[key] = t
                event = {"t": t, "wall": wall(t, args.start_wall), "kind": "alert", "track_id": o.track_id,
                         "class": meta["object_type"], "bbox": o.bbox, **meta}
                cand_events.append(event)
                same = [a for a in alarms if a["open"] and a["episode_id"] == meta["episode_id"]]
                fold = [a for a in alarms if a["open"] and a["class"] == meta["object_type"]
                        and iou(a["bbox"], o.bbox) >= FOLD_IOU]
                if same:
                    continue
                if fold:
                    fold[0]["episode_id"], fold[0]["folded"] = meta["episode_id"], fold[0].get("folded", 0) + 1
                    continue
                alarms.append({"opened_t": t, "wall": event["wall"], "episode_id": meta["episode_id"],
                               "class": meta["object_type"], "bbox": o.bbox, "open": True, "track_ids": meta["track_ids"]})
            for o, meta in resolved:
                event = {"t": t, "wall": wall(t, args.start_wall), "kind": "resolved", "track_id": o.track_id,
                         "class": meta["object_type"], "bbox": o.bbox, **meta}
                cand_events.append(event)
                for a in alarms:
                    if a["open"] and a["episode_id"] == meta["episode_id"]:
                        a.update(open=False, closed_t=t, closed_wall=event["wall"], resolution=meta["resolution"],
                                 track_ids=meta["track_ids"])

    # ---- episodes, as the candidate saw them
    episodes = sorted(seen_episodes.values(), key=lambda e: e["first_seen"])
    alerted = [e for e in episodes if e["alerted_at"] is not None]
    alerted_ids = {e["episode_id"] for e in alerted}

    # ---- threshold evidence: hand-offs between a track that ends and the next one at that place
    ordered = sorted(tracks.items(), key=lambda kv: kv[1]["first"])
    handoffs = []
    for tid, s in ordered:
        best = None
        for pid, p in ordered:
            if pid == tid or p["last"] >= s["first"] or s["first"] - p["last"] > 600:
                continue
            (ax, ay), (bx, by) = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in (p["last_box"], s["first_box"])]
            diag = max(math.hypot(b[2] - b[0], b[3] - b[1]) for b in (p["last_box"], s["first_box"]))
            d = math.hypot(ax - bx, ay - by) / diag if diag else 99
            if d <= 1.5 and (best is None or d < best["dist_frac"]):
                areas = sorted((b[2] - b[0]) * (b[3] - b[1]) for b in (p["last_box"], s["first_box"]))
                best = {"from": pid, "to": tid, "gap_s": round(s["first"] - p["last"], 1), "dist_frac": round(d, 3),
                        "size_ratio": round(areas[1] / areas[0], 2) if areas[0] else None}
        if best:
            handoffs.append(best)

    # ---- attendance runs at each alerted episode's resting position (while it was alerted)
    def runs_at(anchor, start, end, radius_frac=cand.PROXIMITY_FRAC):
        out, run_start, last_near = [], None, None
        for t, persons in persons_by_t:
            if t < start or t > end:
                continue
            w, h = 2592, 1944
            radius = radius_frac * math.hypot(w, h)
            near = any(math.hypot(max(b[0] - anchor[0], 0, anchor[0] - b[2]), max(b[1] - anchor[1], 0, anchor[1] - b[3]))
                       <= radius for b in persons)
            if near:
                if run_start is None or t - last_near > cand.GRACE_SECONDS:
                    if run_start is not None:
                        out.append(round(last_near - run_start, 1))
                    run_start = t
                last_near = t
        if run_start is not None:
            out.append(round(last_near - run_start, 1))
        return out

    episode_rows = []
    for e in episodes:
        end = e["resolved_at"] if e["resolved_at"] is not None else frames and persons_by_t[-1][0]
        episode_rows.append({
            "episode_id": e["episode_id"], "class": e["class_name"], "state": e["state"], "resolution": e["resolution"],
            "first_seen": wall(e["first_seen"], args.start_wall), "alerted": wall(e["alerted_at"], args.start_wall) if e["alerted_at"] else None,
            "resolved": wall(e["resolved_at"], args.start_wall) if e["resolved_at"] else None,
            "track_ids": e["track_ids"], "track_changes": max(0, len(e["track_ids"]) - 1),
            "anchor": [int(v) for v in e["anchor"]], "bbox": e["anchor_bbox"],
            "attendance_runs_while_alerted": runs_at(e["anchor"], e["alerted_at"], end) if e["alerted_at"] else None})

    # ---- baseline alarms grouped by the candidate episode that covers them (same place)
    def episode_for(event):
        best = None
        for e in alerted:
            if iou(e["anchor_bbox"], event["bbox"]) > 0.2 or math.hypot(
                    (event["bbox"][0] + event["bbox"][2]) / 2 - e["anchor"][0],
                    (event["bbox"][1] + event["bbox"][3]) / 2 - e["anchor"][1]) <= 0.75 * math.hypot(
                    e["anchor_bbox"][2] - e["anchor_bbox"][0], e["anchor_bbox"][3] - e["anchor_bbox"][1]):
                best = e["episode_id"] if best is None else best
        return best
    for event in base_alarms:
        event["candidate_episode"] = episode_for(event)

    flips = {tid: dict(s["classes"]) for tid, s in tracks.items() if len(s["classes"]) > 1}
    summary = {
        "camera": args.camera, "frames": frames,
        "duration_s": round(persons_by_t[-1][0] - persons_by_t[0][0], 1) if persons_by_t else 0,
        "raw_track_ids": len(tracks),
        "raw_track_ids_lasting_10s": sum(1 for s in tracks.values() if s["last"] - s["first"] >= 10),
        "tracks_with_class_flips": len(flips),
        "baseline": {"alerts": len(base_alarms), "alarms": len(base_alarms),
                     "alert_track_ids": [e["track_id"] for e in base_alarms],
                     "distinct_alert_track_ids": len({e["track_id"] for e in base_alarms}),
                     "update_us_p50": round(pct(base_us, 50), 1), "update_us_p95": round(pct(base_us, 95), 1)},
        "candidate": {"episodes": len(episodes), "alerted_episodes": len(alerted),
                      "alert_events": sum(1 for e in cand_events if e["kind"] == "alert"),
                      "resolution_events": sum(1 for e in cand_events if e["kind"] == "resolved"),
                      "alarms_opened": len(alarms), "alarms_closed": sum(1 for a in alarms if not a["open"]),
                      "alarms_still_open": sum(1 for a in alarms if a["open"]),
                      "alerted_episode_track_ids": {e["episode_id"]: e["track_ids"] for e in alerted},
                      "update_us_p50": round(pct(cand_us, 50), 1), "update_us_p95": round(pct(cand_us, 95), 1)},
        "baseline_alarms_per_candidate_episode": {eid: sum(1 for e in base_alarms if e["candidate_episode"] == eid)
                                                 for eid in sorted(alerted_ids)},
        "baseline_alarms_outside_any_candidate_episode": sum(1 for e in base_alarms if e["candidate_episode"] is None),
        "handoff_gap_s": {"n": len(handoffs), "p50": pct([h["gap_s"] for h in handoffs], 50),
                          "p90": pct([h["gap_s"] for h in handoffs], 90), "max": max([h["gap_s"] for h in handoffs], default=None)},
        "handoff_dist_frac": {"p50": pct([h["dist_frac"] for h in handoffs], 50),
                              "p90": pct([h["dist_frac"] for h in handoffs], 90)},
        "overrides": args.set,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for name, rows in (("baseline_alarms", base_alarms), ("candidate_events", cand_events), ("candidate_alarms", alarms),
                       ("episodes", episode_rows), ("handoffs", handoffs)):
        with open(os.path.join(args.out, f"{name}.jsonl"), "w") as f:
            f.writelines(json.dumps(r, default=str) + "\n" for r in rows)
    with open(os.path.join(args.out, "tracks.json"), "w") as f:
        json.dump({tid: {"first": s["first"], "last": s["last"], "n": s["n"], "classes": dict(s["classes"]),
                         "first_box": s["first_box"], "last_box": s["last_box"],
                         "conf_p50": round(statistics.median(s["conf"]), 3)} for tid, s in tracks.items()}, f)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
