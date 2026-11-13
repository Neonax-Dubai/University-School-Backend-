"""Replay one recorded clip through the camera-health scene-change path, offline.

    python replay_tamper_clip.py --clip <file.mp4> --package <dir containing camera_health/>

Runs the SAME decision code production runs - `detectors.tamper_state()` and
`Condition.update()` - over real frames, sampled at the production cadence
(manager.SAMPLE_INTERVAL_SECONDS). The manager itself is deliberately NOT used: its
`_announce()` path emits incidents, and nothing here may reach production. There is no
network, no database, no sink and no outbox; the only output is the trace it prints.

Each sample reproduces `manager._process()` for the tamper feature, in its order:

    gray  = metrics.ensure_sample(frame)
    share = metrics.structured_share(gray)          -> obstructed
    bad, detail = detectors.tamper_state(state, gray, t, obstructed)
    action      = state.tamper.update(bad, t, TAMPER_SECONDS, TAMPER_COOLDOWN_SECONDS, detail)
    detectors.defocus_state(state, metrics.sharpness(gray), obstructed)   # learns the baseline

`defocus_state` is called last, exactly as the manager does, because it is what learns
`sharpness_baseline` - the measurement the structure and lighting guards read.

The clip supplies its own baseline: a production clip is 20 s of pre-roll before the event,
which at 1 sample/s is enough for SCENE_WARMUP_SAMPLES + SCENE_BASELINE_SAMPLES and for the
10 samples `defocus_state` needs. A clip shorter than that cannot reach PHASE_READY, and the
trace says so rather than pretending to a verdict.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--package", required=True,
                    help="directory containing the camera_health package to exercise")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default="", help="write the trace here as JSON")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.package))
    from camera_health import detectors as D          # noqa: E402
    from camera_health import metrics                 # noqa: E402
    from camera_health import state as state_mod      # noqa: E402

    interval = 1.0                                    # manager.SAMPLE_INTERVAL_SECONDS
    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.clip}")

    st = state_mod.CameraState("replay", 0.0)
    trace, next_at, raised_at = [], 0.0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t + 1e-6 < next_at:
            continue
        next_at = t + interval

        gray = metrics.ensure_sample(frame)
        share = metrics.structured_share(gray)
        obstructed = share <= D.OBSTRUCTION_EDGE_MAX

        bad, detail = D.tamper_state(st, gray, t, obstructed)
        action = None
        if bad is not None:
            action = st.tamper.update(bad, t, D.TAMPER_SECONDS, D.TAMPER_COOLDOWN_SECONDS, detail)
        D.defocus_state(st, metrics.sharpness(gray), obstructed)

        check = detail.get("lighting_check") or {}
        row = {
            "t": round(t, 2),
            "phase": st.scene_phase,
            "distance": detail.get("distance"),
            "brightness_ratio": detail.get("brightness_ratio", check.get("brightness_ratio")),
            "overlap": detail.get("structural_overlap", check.get("structural_overlap")),
            "sharpness_ratio": check.get("sharpness_ratio"),
            "illumination_transition": bool(detail.get("illumination_transition")),
            "structure_intact": detail.get("structure_intact"),
            "structure_intact_rejected": detail.get("structure_intact_rejected"),
            "lighting_rejected": check.get("lighting_rejected"),
            "bad": bad,
            "since": None if st.tamper.since is None else round(t - st.tamper.since, 1),
            "action": action,
        }
        trace.append(row)
        if action == "raise" and raised_at is None:
            raised_at = t
    cap.release()

    label = args.label or os.path.basename(args.clip)
    print(f"=== {label} ===")
    print(f"  samples {len(trace)}  final phase {trace[-1]['phase'] if trace else 'n/a'}")
    if not any(r["phase"] == "ready" for r in trace):
        print("  NEVER REACHED PHASE_READY - clip too short to form a baseline; no verdict")
    print(f"  {'t':>6} {'dist':>6} {'brt':>5} {'ovl':>5} {'shrp':>5}  {'branch':<22} "
          f"{'bad':>5} {'since':>5} action")
    for r in trace:
        branch = ("LIGHTING->rebaseline" if r["illumination_transition"]
                  else "structure_intact" if r["structure_intact"]
                  else "struct_rejected" if r["structure_intact_rejected"]
                  else "scored" if r["phase"] == "ready" else r["phase"])
        if r["lighting_rejected"]:
            branch += "/lgt_rej"
        fmt = lambda v, n=3: "-" if v is None else f"{v:.{n}f}"     # noqa: E731
        print(f"  {r['t']:6.1f} {fmt(r['distance'],3):>6} {fmt(r['brightness_ratio'],2):>5} "
              f"{fmt(r['overlap'],2):>5} {fmt(r['sharpness_ratio'],2):>5}  {branch:<22} "
              f"{str(r['bad']):>5} {str(r['since']):>5} {r['action'] or ''}")
    print(f"  VERDICT: {'RAISED at t=%.1f s' % raised_at if raised_at is not None else 'no raise'}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"clip": args.clip, "label": label, "raised_at": raised_at,
                       "trace": trace}, fh, indent=1)


if __name__ == "__main__":
    main()
