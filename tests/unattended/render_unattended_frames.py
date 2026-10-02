"""
Stage 3 of the unattended-object replay: draw what the analytic saw onto the recorded frames at given
times - candidate-object tracks (track id, class), person boxes and the candidate's episode anchors -
so a reviewer can check by eye that the tracks folded into one episode are one physical object.

    python3 render_unattended_frames.py --clip /clips/1.mp4@0 --clip /clips/2.mp4@960 \
        --dump c01_D.jsonl.gz --episodes episodes.jsonl --times 312,388,601 --out /out

The pictures show people: write them to a private directory and delete them after review.
"""
import argparse
import gzip
import json
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--episodes", default=None)
    ap.add_argument("--times", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    clips = sorted(((float(o), p) for p, o in (c.rsplit("@", 1) for c in args.clip)))
    rows = [json.loads(line) for line in gzip.open(args.dump, "rt")]
    episodes = [json.loads(line) for line in open(args.episodes)] if args.episodes else []
    for t in (float(x) for x in args.times.split(",") if x.strip()):
        offset, path = max(((o, p) for o, p in clips if o <= t), default=clips[0])
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, (t - offset) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"t={t}: no frame")
            continue
        row = min(rows, key=lambda r: abs(r["t"] - t))
        for box in row["persons"]:
            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:4]), (0, 0, 255), 4)
        for obj in row["objects"]:
            track, _, _, name, conf, x1, y1, x2, y2 = obj
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            cv2.putText(frame, f"{track} {name} {conf:.2f}", (x1, max(30, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (0, 255, 0), 3)
        for ep in episodes:
            ax, ay = ep["anchor"]
            cv2.circle(frame, (int(ax), int(ay)), 18, (0, 255, 255), -1)
            cv2.putText(frame, ep["episode_id"][-4:], (int(ax) + 22, int(ay) + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 255), 3)
        cv2.putText(frame, f"{args.label} t={t:.0f}s", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 6)
        out = os.path.join(args.out, f"{args.label}_t{int(t):05d}.jpg")
        cv2.imwrite(out, cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)))
        print(out)


if __name__ == "__main__":
    main()
