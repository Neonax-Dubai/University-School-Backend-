#!/usr/bin/env python3
"""
Standalone Fall Detection Test Script
======================================
Runs the PRODUCTION FallPoseAdapter against a local video file or RTSP stream.

Usage
-----
    python test_fall_detection.py --input /path/to/video.mp4
    python test_fall_detection.py --input rtsp://user:pass@host/Streaming/Channels/101
    python test_fall_detection.py --input video.mp4 --output out_fall.mp4 --headless
    python test_fall_detection.py --input video.mp4 --save-stills --loop

    # Override detection confidence or target FPS via env
    FALL_TEST_DETECTION_CONF=0.4 FALL_TEST_FPS=8 python test_fall_detection.py --input video.mp4

Keyboard controls (display mode)
    q / Esc   quit

Frame annotations
    Cyan box  + "Px"        confirmed person track
    RED box   + "FALL"      confirmed fall event (lingers 5 s)
    HUD top-left            cam/res/fps/ms, adapter counters, candidate-gate stats
    Red banner bottom       while fall event is active
"""
import argparse, os, sys, time
from collections import deque
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tracking
import fall_pose_adapter
import pose_l960_runtime

CAMERA_ID  = os.getenv("FALL_TEST_CAMERA_ID", "test_cam")
CONF       = float(os.getenv("FALL_TEST_DETECTION_CONF", "0.45"))
IMG_SIZE   = int(os.getenv("FALL_TEST_IMG_SIZE", "640"))
TARGET_FPS = float(os.getenv("FALL_TEST_FPS", "10.0"))

C_PERSON = (0, 200, 255)   # cyan-amber
C_FALL   = (0,   0, 255)   # red
C_WHITE  = (255,255,255)
C_GREEN  = (0, 220,  80)

COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_detector():
    from ultralytics import YOLO
    engine_dir = os.getenv("ZAYED_ENGINE_DIR", "/engines/zayed/current")
    candidates = [
        os.path.join(engine_dir, "yolo26l.engine"),
        os.getenv("FALL_TEST_DETECTOR_PT",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "models/yolo26l.pt")),
    ]
    for p in candidates:
        if os.path.exists(p):
            m = YOLO(p, task="detect")
            m.predict([np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.uint8)],
                      imgsz=IMG_SIZE, conf=CONF, device=0, verbose=False)
            print(f"[FALL-TEST] detector: {p}")
            return m
    raise FileNotFoundError(
        "No detector found. Set ZAYED_ENGINE_DIR or FALL_TEST_DETECTOR_PT.\n"
        f"Tried: {candidates}")


def _sc(w): return max(0.5, w / 1280.0)


def label_box(frame, bbox, text, colour, lw=2):
    h, w = frame.shape[:2]; s = _sc(w)
    x1,y1,x2,y2 = [int(v) for v in bbox]
    cv2.rectangle(frame,(x1,y1),(x2,y2),colour,max(1,int(lw*s)))
    if not text: return
    fs = 0.55*s; tk = max(1,int(s))
    (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, tk)
    p = int(6*s); top = max(0, y1-th-p)
    cv2.rectangle(frame,(x1,top),(x1+tw+p+2,top+th+p),colour,-1)
    cv2.putText(frame,text,(x1+int(4*s),top+th+int(s)),
                cv2.FONT_HERSHEY_SIMPLEX,fs,(0,0,0),tk,cv2.LINE_AA)


def draw_hud(frame, lines):
    h,w = frame.shape[:2]; s = _sc(w); step = int(26*s)
    for i,(text,col) in enumerate(lines):
        orig = (int(12*s), int(28*s)+i*step)
        cv2.putText(frame,text,orig,cv2.FONT_HERSHEY_SIMPLEX,
                    0.65*s,(0,0,0),max(3,int(4*s)),cv2.LINE_AA)
        cv2.putText(frame,text,orig,cv2.FONT_HERSHEY_SIMPLEX,
                    0.65*s,col,max(1,int(s)),cv2.LINE_AA)


def draw_banner(frame, lines):
    if not lines: return
    h,w = frame.shape[:2]; s = _sc(w); step = int(30*s)
    bh = step*len(lines)+int(10*s)
    ov = frame.copy()
    cv2.rectangle(ov,(0,h-bh),(w,h),(0,0,40),-1)
    cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
    for i,(text,col) in enumerate(lines):
        cv2.putText(frame,text,(int(12*s),h-bh+int(22*s)+i*step),
                    cv2.FONT_HERSHEY_SIMPLEX,0.75*s,col,
                    max(2,int(2*s)),cv2.LINE_AA)


def draw_skeleton(frame, kxy, kcf, thr=0.3):
    h,w = frame.shape[:2]; s = _sc(w)
    for a,b in COCO_SKELETON:
        if kcf[a]>=thr and kcf[b]>=thr:
            pa=(int(kxy[a][0]),int(kxy[a][1])); pb=(int(kxy[b][0]),int(kxy[b][1]))
            if pa!=(0,0) and pb!=(0,0):
                cv2.line(frame,pa,pb,(0,180,255),max(1,int(s)),cv2.LINE_AA)
    for i in range(len(kxy)):
        if kcf[i]>=thr:
            pt=(int(kxy[i][0]),int(kxy[i][1]))
            if pt!=(0,0):
                cv2.circle(frame,pt,max(2,int(3*s)),(0,255,140),-1)


# ── main loop ────────────────────────────────────────────────────────────────

def run(args):
    # ── production adapter ─────────────────────────────────────────────────
    print("[FALL-TEST] Building production FallPoseAdapter ...")
    pose_l960_runtime.shared_l960.acquire()
    adapter = fall_pose_adapter.build_production_adapter()
    adapter.set_enabled_cameras({CAMERA_ID})
    print("[FALL-TEST] Adapter armed.")

    # ── detector + tracker ─────────────────────────────────────────────────
    print("[FALL-TEST] Loading detector ...")
    det_model = _load_detector()
    tm = tracking.TrackManager(
        person_classes={0}, vehicle_classes=set(),
        class_names={0:"person"}, object_classes=set())

    # ── source ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        sys.exit(f"[FALL-TEST] Cannot open: {args.input}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 1280)
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    print(f"[FALL-TEST] Source {src_w}x{src_h} @ {src_fps:.1f} fps")

    writer = None
    if args.output:
        w4 = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, w4, min(TARGET_FPS, src_fps),
                                 (min(src_w,1280), min(src_h,720)))
        print(f"[FALL-TEST] Output: {args.output}")

    stills_dir = None
    if args.save_stills:
        stills_dir = os.path.join(os.path.dirname(os.path.abspath(args.input)),
                                  "fall_stills")
        os.makedirs(stills_dir, exist_ok=True)
        print(f"[FALL-TEST] Stills: {stills_dir}")

    frame_id    = 0
    total_falls = 0
    active      = {}          # track_id -> expiry wall-clock
    fps_win     = deque(maxlen=30)
    interval    = 1.0 / max(0.5, TARGET_FPS)
    next_due    = time.perf_counter()

    try:
        while True:
            now_pc = time.perf_counter()
            if now_pc < next_due:
                time.sleep(min(next_due - now_pc, 0.01)); continue
            next_due += interval

            ok, frame = cap.read()
            if not ok or frame is None:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                break

            frame_id   += 1
            ts          = time.time()
            canvas      = frame.copy()
            h, w        = frame.shape[:2]
            t0          = time.perf_counter()

            # ── detect + track ─────────────────────────────────────────────
            try:
                res = det_model.predict(source=[frame], imgsz=IMG_SIZE,
                                        conf=CONF, device=0, verbose=False)[0]
            except Exception as e:
                print(f"[FALL-TEST] detect: {e}"); res = None

            dets = []
            if res is not None and res.boxes is not None:
                for cid,cf,box in zip(res.boxes.cls.cpu().numpy().astype(int),
                                      res.boxes.conf.cpu().numpy(),
                                      res.boxes.xyxy.cpu().numpy()):
                    if cid == 0 and cf >= CONF:
                        dets.append((0,float(cf),
                                     (int(box[0]),int(box[1]),
                                      int(box[2]),int(box[3]))))

            tracked = tm.update(CAMERA_ID, dets, now=ts)
            persons = [t for t in tracked if t.group == tracking.PERSON_GROUP]

            # ── fall adapter ───────────────────────────────────────────────
            adapter.submit(CAMERA_ID, frame, tracked, timestamp=ts, frame_id=frame_id)

            new_msgs = []
            for ev in adapter.flush():
                total_falls += 1
                tid = str(ev.track_id)
                active[tid] = ts + 5.0
                meta = ev.metadata() if hasattr(ev,"metadata") else {}
                msg = (f"FALL track={tid}  "
                       f"torso={meta.get('torso_angle','?')} deg  "
                       f"held={meta.get('pose_duration','?')}s  "
                       f"kpts={meta.get('confident_keypoints','?')}  "
                       f"iou={meta.get('pose_track_iou','?')}")
                new_msgs.append(msg)
                print(f"[FALL-TEST] *** {msg}")
                label_box(canvas, ev.bbox, f"FALL  P{tid}", C_FALL, 4)
                if stills_dir:
                    fname = os.path.join(stills_dir,
                                         f"fall_{time.strftime('%H%M%S')}_{frame_id:06d}_t{tid}.jpg")
                    cv2.imwrite(fname, canvas)
                    print(f"[FALL-TEST] Still: {fname}")

            active = {tid:exp for tid,exp in active.items() if exp > ts}

            # ── draw tracks ────────────────────────────────────────────────
            for t in persons:
                tid = str(t.track_id)
                if tid in active:
                    label_box(canvas,t.bbox,f"FALL P{tid}",C_FALL,3)
                else:
                    label_box(canvas,t.bbox,f"P{tid}",C_PERSON,2)

            # ── HUD ────────────────────────────────────────────────────────
            ms = (time.perf_counter()-t0)*1000
            fps_win.append(time.time())
            fps = ((len(fps_win)-1)/max(1e-6,fps_win[-1]-fps_win[0])
                   if len(fps_win)>1 else 0.0)
            st = adapter.stats()
            sc = C_FALL if active else C_GREEN
            draw_hud(canvas,[
                (f"[FALL-DETECTION TEST]  cam={CAMERA_ID}  {w}x{h}  "
                 f"{fps:.1f} fps  {ms:.0f} ms", C_WHITE),
                (f"persons={len(persons)}  frame={frame_id}",C_WHITE),
                (f"adapter  submitted={st.get('fall_pose_submitted',0)}  "
                 f"processed={st.get('fall_pose_processed',0)}  "
                 f"events={st.get('fall_pose_events',0)}  "
                 f"queue={st.get('fall_pose_queue_depth',0)}",C_WHITE),
                (f"gate  candidates={st.get('fall_pose_candidates',0)}  "
                 f"skipped={st.get('fall_pose_skipped_not_candidate',0)}  "
                 f"dropped={st.get('fall_pose_dropped_queue_full',0)}",C_WHITE),
                (f"TOTAL FALLS: {total_falls}  "
                 + ("*** FALL ACTIVE ***" if active else "no active alert"), sc),
            ])

            if active:
                draw_banner(canvas,[("!!! PERSON FELL !!!",C_FALL)]
                            + [(m,C_FALL) for m in new_msgs[:2]])

            # ── output ─────────────────────────────────────────────────────
            out = canvas
            if w > 1280:
                out = cv2.resize(canvas,(1280,int(h*1280/w)),
                                 interpolation=cv2.INTER_AREA)
            if writer: writer.write(out)
            if not args.headless:
                cv2.imshow("Fall Detection Test", out)
                if cv2.waitKey(1)&0xFF in (ord("q"),27): break

    finally:
        cap.release()
        if writer: writer.release()
        if not args.headless: cv2.destroyAllWindows()
        adapter.close()
        print(f"\n[FALL-TEST] Done. frames={frame_id}  falls={total_falls}")
        print("[FALL-TEST] Stats:", adapter.stats())


def main():
    ap = argparse.ArgumentParser(
        description="Standalone Fall Detection – production FallPoseAdapter")
    ap.add_argument("--input",       required=True)
    ap.add_argument("--output",      default="", help="Save annotated MP4")
    ap.add_argument("--headless",    action="store_true")
    ap.add_argument("--loop",        action="store_true")
    ap.add_argument("--save-stills", action="store_true")
    run(ap.parse_args())

if __name__ == "__main__":
    main()
