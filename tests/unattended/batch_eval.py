"""
Stage 2 over many per-clip dumps (dump_unattended_inputs.py --separate): one abandoned.py, every clip,
per-class totals. CPU only, no network, writes only --out.

    python3 batch_eval.py --dumps /dumps/new_camera_01 --camera camera_01 --module /app/abandoned.py --out /out/x.json

Per clip and per object class (the episode's majority class): detections, raw tracks, episodes,
episodes that reached UNATTENDED (no person near past the grace), alerted episodes, alert events, resolutions by reason, tracker ids per alerted episode. Two
alerted episodes of the same class group at overlapping places in one clip are listed as POSSIBLE
duplicates for a reviewer to check on the frames. Phone boxes handed to the phone analytic are counted
as held (phone_use.attach finds a person) or free.
"""
import argparse
import glob
import gzip
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import phone_use  # noqa: E402


class Obj:
    group = "object"

    def __init__(self, camera_id, row):
        self.camera_id = camera_id
        self.track_id, self.raw_track_id, self.class_id, self.class_name, self.confidence = row[:5]
        self.bbox = list(row[5:9])


def load(path):
    spec = importlib.util.spec_from_file_location("abandoned_under_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--module", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    A = load(args.module)
    group = getattr(A, "CLASS_GROUPS", {24: "bag", 26: "bag", 28: "bag"})
    clips, totals, update_us = [], {}, []
    for path in sorted(glob.glob(os.path.join(args.dumps, "*.jsonl.gz"))):
        det = A.AbandonedObjectDetector({args.camera: {"abandoned_object": True}}, dwell_seconds=A.DWELL_SECONDS)
        seen, reached, alerts, resolutions, detections, tracks = {}, set(), [], [], {}, {}
        phones_held = phones_free = 0
        for line in gzip.open(path, "rt"):
            row = json.loads(line)
            objs = [Obj(args.camera, r) for r in row["objects"]]
            for o in objs:
                detections[o.class_name] = detections.get(o.class_name, 0) + 1
                tracks.setdefault(o.class_name, set()).add(o.track_id)
            for p in row.get("phones", []):
                if phone_use.attach(p[1:5], row["persons"]) is None:
                    phones_free += 1
                else:
                    phones_held += 1
            t0 = time.perf_counter()
            for o, meta in det.update(args.camera, objs, row["persons"], row["w"], row["h"], now=row["t"]):
                alerts.append({"t": row["t"], "class": meta["object_type"], "episode_id": meta["episode_id"],
                               "bbox": o.bbox, "track_ids": meta["track_ids"]})
            for o, meta in det.drain_resolutions(args.camera):
                resolutions.append({"t": row["t"], "class": meta["object_type"], "episode_id": meta["episode_id"],
                                    "resolution": meta["resolution"]})
            update_us.append((time.perf_counter() - t0) * 1e6)
            for snapshot in det.episodes(args.camera):
                seen[snapshot["episode_id"]] = snapshot
                if snapshot["state"] in (A.UNATTENDED, A.ALERTED):
                    reached.add(snapshot["episode_id"])
        episodes = list(seen.values())
        alerted = [e for e in episodes if e["alerted_at"] is not None]
        possible_duplicates = [(a["episode_id"], b["episode_id"]) for i, a in enumerate(alerted) for b in alerted[i + 1:]
                               if group.get(a["class_id"], a["class_id"]) == group.get(b["class_id"], b["class_id"])
                               and iou(a["anchor_bbox"], b["anchor_bbox"]) > 0.1]
        per_class = {}
        for name in set(detections) | {e["class_name"] for e in episodes}:
            mine = [e for e in episodes if e["class_name"] == name]
            per_class[name] = {
                "detections": detections.get(name, 0), "raw_tracks": len(tracks.get(name, ())),
                "episodes": len(mine), "unattended_episodes": sum(1 for e in mine if e["episode_id"] in reached),
                "alerted_episodes": sum(1 for e in mine if e["alerted_at"] is not None),
                "alerts": sum(1 for a in alerts if a["class"] == name),
                "resolutions": {r: sum(1 for x in resolutions if x["class"] == name and x["resolution"] == r)
                                for r in {x["resolution"] for x in resolutions if x["class"] == name}},
                "track_ids_per_alerted": [len(e["track_ids"]) for e in mine if e["alerted_at"] is not None]}
            tot = totals.setdefault(name, {"detections": 0, "raw_tracks": 0, "episodes": 0, "unattended_episodes": 0,
                                           "alerted_episodes": 0,
                                           "alerts": 0, "resolutions": {}, "track_ids_per_alerted": [], "clips_with_alert": 0})
            for k in ("detections", "raw_tracks", "episodes", "unattended_episodes", "alerted_episodes", "alerts"):
                tot[k] += per_class[name][k]
            for r, n in per_class[name]["resolutions"].items():
                tot["resolutions"][r] = tot["resolutions"].get(r, 0) + n
            tot["track_ids_per_alerted"] += per_class[name]["track_ids_per_alerted"]
            tot["clips_with_alert"] += 1 if per_class[name]["alerts"] else 0
        clips.append({"clip": os.path.basename(path).replace(".jsonl.gz", ""), "per_class": per_class,
                      "alerts": alerts, "resolutions": resolutions, "possible_duplicates": possible_duplicates,
                      "phones_to_phone_analytic": {"held": phones_held, "free": phones_free},
                      "alerted_episodes": [{"episode_id": e["episode_id"], "class": e["class_name"],
                                            "alerted_at": e["alerted_at"], "resolved_at": e["resolved_at"],
                                            "resolution": e["resolution"], "anchor": [int(v) for v in e["anchor"]],
                                            "anchor_bbox": e["anchor_bbox"], "track_ids": e["track_ids"]} for e in alerted]})
    update_us.sort()
    result = {"camera": args.camera, "module": args.module, "clips": clips, "totals": totals,
              "possible_duplicates": sum(len(c["possible_duplicates"]) for c in clips),
              "update_us_p50": round(update_us[len(update_us) // 2], 1) if update_us else None,
              "update_us_p95": round(update_us[int(len(update_us) * 0.95)], 1) if update_us else None}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"camera": args.camera, "clips": len(clips), "totals": totals,
                      "possible_duplicates": result["possible_duplicates"],
                      "update_us_p50": result["update_us_p50"], "update_us_p95": result["update_us_p95"]}, indent=1))


if __name__ == "__main__":
    main()
