#!/usr/bin/env python3
"""
Standalone Sleep Detection Test Script
=======================================
Runs the PRODUCTION SleepingAdapter against a local video file or RTSP stream.

Usage
-----
    python test_sleep_detection.py --input /path/to/video.mp4
    python test_sleep_detection.py --input rtsp://user:pass@host/Streaming/Channels/101
    python test_sleep_detection.py --input video.mp4 --output out_sleep.mp4 --headless
    python test_sleep_detection.py --input video.mp4 --save-stills --loop

    # Tune via env (all production thresholds apply)
    SLEEP_SECONDS=60 SLEEP_MIN_SAMPLES=15 python test_sleep_detection.py --input video.mp4

    # Speed up video replay to compress the confirmation window (120 s -> 30 s)
    SLEEP_SECONDS=30 SLEEP_MIN_SAMPLES=10 SLEEP_TEST_FPS=4 \\
        python test_sleep_detection.py --input video.mp4

Keyboard controls
    q / Esc   quit

Frame annotations
    Cyan box   + "Px"           each confirmed person track
    ORANGE box + "SLEEPING"     sleeping detection confirmed (lingers 8 s)
    Head-down indicator         a downward arrow drawn above bounding box
                                when the current frame's skeleton reads HEAD-DOWN
    Progress bar                yellow bar below each person box, fills as
                                head-down fraction accumulates during SLEEP_SECONDS
    HUD top-left  cam/res/fps/ms, submitted/processed/raised, judged/head_down
    Orange banner bottom  while sleep event is active

NOTE: The production SLEEP_SECONDS default is 120 s (2 minutes of observation).
      Use the env override above when testing with short clips.
"""
import argparse, os, sys, time
from collections import deque
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tracking
import sleeping
import pose_l960_runtime

CAMERA_ID  = os.getenv("SLEEP_TEST_CAMERA_ID", "test_cam")
CONF       = float(os.getenv("SLEEP_TEST_DETECTION_CONF", "0.45"))
IMG_SIZE   = int(os.getenv("SLEEP_TEST_IMG_SIZE", "640"))
TARGET_FPS = float(os.getenv("SLEEP_TEST_FPS", "8.0"))

C_PERSON  = (0, 200, 255)
C_SLEEP   = (0, 165, 255)   # orange
C_WHITE   = (255,255,255)
C_RED     = (0,   0, 255)
C_GREEN   = (0, 220,  80)
C_YELLOW  = (0, 220, 220)
C_CYAN    = (255, 220,   0)
C_MAGENTA = (255,   0, 200)
C_ORANGE  = (  0, 165, 255)

# COCO-17 skeleton connections for drawing pose
_SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),
    (5,7),(7,9),
    (6,8),(8,10),
    (5,11),(6,12),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16),
]
_SEG_COLOURS = {
    (0,1):(200,200,0),(0,2):(200,200,0),(1,3):(200,200,0),(2,4):(200,200,0),
    (5,6):(0,200,255),
    (5,7):(80,220,0),(7,9):(80,220,0),
    (6,8):(0,80,255),(8,10):(0,80,255),
    (5,11):(200,100,0),(6,12):(200,100,0),
    (11,12):(200,0,200),
    (11,13):(0,255,180),(13,15):(0,255,180),
    (12,14):(255,100,0),(14,16):(255,100,0),
}

SLEEP_DEBUG = os.getenv("SLEEP_DEBUG", "1") == "1"


def _detect_device():
    """Return 0 (GPU) if CUDA is available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[SLEEP-TEST] CUDA available: {torch.cuda.get_device_name(0)}")
            return 0
    except Exception:
        pass
    print("[SLEEP-TEST] No CUDA — running on CPU (slower)")
    return "cpu"


_DEVICE = _detect_device()


def _load_detector():
    """Load detector — tries engine first, falls back to .pt, auto-exports to TensorRT.

    Search order:
      1. yolo26x.engine  (local TRT engine — fastest)
      2. yolo26l.engine  (lighter TRT engine)
      3. yolo26x.pt      (local custom weights → auto-exported to engine)
      4. yolo26l.pt      (lighter custom weights already present → auto-exported)
    """
    from ultralytics import YOLO
    here = os.path.dirname(os.path.abspath(__file__))

    # ── 1. Try existing engine files ──────────────────────────────────────────
    for eng in [os.path.join(here, "yolo26x.engine")]:
        if os.path.exists(eng):
            print(f"[SLEEP-TEST] Loading engine: {eng}")
            m = YOLO(eng, task="detect")
            m.predict([np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)],
                      imgsz=IMG_SIZE, conf=CONF, device=_DEVICE, verbose=False)
            print(f"[SLEEP-TEST] detector ready: {eng}")
            return m

    # ── 2. Find best available .pt (custom first, then hub models) ───────────
    pt_candidates = [
        (os.path.join(here, "yolo26x.pt"), "yolo26x.engine")]
    
    # If a custom local model exists, use it. Otherwise, fallback to downloading yolov8x.pt
    chosen_pt, chosen_engine = next(
        ((p, os.path.join(here, e)) for p, e in pt_candidates if os.path.exists(p)),
        ("yolo26x.pt", os.path.join(here, "yolo26x.engine")) 
    )

    print(f"[SLEEP-TEST] Loading weights: {chosen_pt}")
    m = YOLO(chosen_pt, task="detect")
    
    if _DEVICE != "cpu":
        print(f"[SLEEP-TEST] Exporting to TensorRT → {chosen_engine}  (may take a few minutes)…")
        try:
            m.export(format="engine", device=_DEVICE, half=True, imgsz=IMG_SIZE)
            exported = chosen_pt.replace(".pt", ".engine")
            if not os.path.exists(exported):
                exported = chosen_engine
            if os.path.exists(exported):
                import shutil
                if os.path.abspath(exported) != os.path.abspath(chosen_engine):
                    shutil.move(exported, chosen_engine)
                print(f"[SLEEP-TEST] Engine ready: {chosen_engine}")
                m = YOLO(chosen_engine, task="detect")
            else:
                print("[SLEEP-TEST] Export produced no engine file — running on .pt")
        except Exception as e:
            print(f"[SLEEP-TEST] TRT export failed ({e}) — running on .pt (slower)")
    else:
        print("[SLEEP-TEST] Skipping TRT export (no CUDA) — running on .pt")

    m.predict([np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)],
              imgsz=IMG_SIZE, conf=CONF, device=_DEVICE, verbose=False)
    return m


def _sc(w): return max(0.5, w/1280.0)


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


def draw_progress_bar(frame, bbox, fraction, colour):
    """Draw a thin progress bar just below the person bounding box."""
    h, w = frame.shape[:2]; s = _sc(w)
    x1,y1,x2,y2 = [int(v) for v in bbox]
    bh = max(4, int(6*s))
    bar_y = min(y2+2, h-bh-1)
    bar_w = x2-x1
    cv2.rectangle(frame,(x1,bar_y),(x2,bar_y+bh),(60,60,60),-1)
    filled = max(1, int(bar_w * min(1.0, max(0.0, fraction))))
    cv2.rectangle(frame,(x1,bar_y),(x1+filled,bar_y+bh),colour,-1)


def draw_head_down_arrow(frame, bbox):
    """Draw a downward arrow above the box when head-down is currently detected."""
    h, w = frame.shape[:2]; s = _sc(w)
    x1,y1,x2,y2 = [int(v) for v in bbox]
    cx = (x1+x2)//2; tip_y = max(0,y1-int(8*s)); base_y = max(0,y1-int(28*s))
    cv2.arrowedLine(frame,(cx,base_y),(cx,tip_y),
                    C_SLEEP,max(2,int(2*s)),cv2.LINE_AA,tipLength=0.4)


def draw_skeleton(frame, kpts, confs, kpt_thresh=0.3):
    """Draw COCO-17 skeleton: coloured edges then keypoint dots."""
    h, w = frame.shape[:2]; s = _sc(w)
    r  = max(3, int(5 * s))
    lw = max(1, int(2 * s))
    pts = [(int(kpts[i][0]), int(kpts[i][1])) if confs[i] >= kpt_thresh else None
           for i in range(17)]
    for (a, b), col in _SEG_COLOURS.items():
        if pts[a] and pts[b]:
            cv2.line(frame, pts[a], pts[b], col, lw, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        if pt:
            dot_col = C_YELLOW if i in (5, 6) else (C_RED if i in (0,1,2,3,4) else C_GREEN)
            cv2.circle(frame, pt, r, dot_col, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, r, (0,0,0),  1, cv2.LINE_AA)


def draw_track_debug(frame, bbox, tid, down, detail, frac, n_judged, n_min):
    """Draw compact debug panel beside each person box."""
    h, w = frame.shape[:2]; s = _sc(w)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    fs = max(0.32, 0.40 * s); tk = max(1, int(s)); lh = int(17 * s)

    if down is True:
        hd_text, hd_col = "HEAD-DOWN: YES <<", (0, 60, 255)
    elif down is False:
        hd_text, hd_col = "HEAD-DOWN: NO", (0, 200, 100)
    else:
        hd_text, hd_col = "HEAD-DOWN: NO-POSE", (130, 130, 130)

    hr  = detail.get('head_rise', '?') if detail else '?'
    sw  = detail.get('shoulder_w', '?') if detail else '?'
    rsn = detail.get('reason', '') if detail else ''
    thr = sleeping.HEAD_DROP_RATIO
    frac_pct = int(frac * 100)
    frac_col  = C_SLEEP if frac >= 0.8 else (C_YELLOW if frac >= 0.4 else C_WHITE)

    lines = [
        (f"T{tid}",                         C_CYAN),
        (hd_text,                            hd_col),
        (f"rise={hr}  thresh={thr}",         C_WHITE),
        (f"shoulder_w={sw}px",               C_WHITE),
    ]
    if rsn:
        lines.append((f"SKIP: {rsn}", C_RED))
    lines.append((f"frac={frac_pct}%  n={n_judged}/{n_min}", frac_col))

    panel_x = min(x2 + int(4 * s), w - 2)
    panel_y = y1
    panel_w = int(180 * s)
    panel_h = lh * len(lines) + int(6 * s)
    px2 = min(panel_x + panel_w, w - 1)
    py2 = min(panel_y + panel_h, h - 1)
    if px2 > panel_x and py2 > panel_y:
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (px2, py2), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, (txt, col) in enumerate(lines):
        py = panel_y + int(4 * s) + i * lh + lh - 2
        if py < h and panel_x < w:
            cv2.putText(frame, txt, (panel_x + int(4*s), py),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,0), tk + 2, cv2.LINE_AA)
            cv2.putText(frame, txt, (panel_x + int(4*s), py),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, col,    tk,     cv2.LINE_AA)


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
    cv2.rectangle(ov,(0,h-bh),(w,h),(0,40,60),-1)
    cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
    for i,(text,col) in enumerate(lines):
        cv2.putText(frame,text,(int(12*s),h-bh+int(22*s)+i*step),
                    cv2.FONT_HERSHEY_SIMPLEX,0.75*s,col,
                    max(2,int(2*s)),cv2.LINE_AA)


# Lightweight per-track head-down state tracked for the progress bar
class _HeadDownTracker:
    """Tracks head-down fraction over last SLEEP_SECONDS for the progress bar only."""
    def __init__(self):
        self._samples = {}   # track_id -> deque[(t, down|None)]

    def observe(self, track_id, down, ts):
        win = float(os.getenv("SLEEP_SECONDS","120"))
        q = self._samples.setdefault(track_id, deque())
        q.append((ts, down))
        while q and ts - q[0][0] > win:
            q.popleft()

    def fraction(self, track_id):
        q = self._samples.get(track_id)
        if not q:
            return 0.0
        judged = [s for s in q if s[1] is not None]
        if not judged:
            return 0.0
        return sum(1 for s in judged if s[1]) / len(judged)


def run(args):
    # ── adapter ────────────────────────────────────────────────────────────
    print("[SLEEP-TEST] Building production SleepingAdapter ...")
    pose_l960_runtime.shared_l960.acquire()
    adapter = sleeping.SleepingAdapter(log=print)
    adapter.set_enabled_cameras({CAMERA_ID})
    print("[SLEEP-TEST] Adapter armed.")

    print("[SLEEP-TEST] Loading detector ...")
    det_model = _load_detector()
    tm = tracking.TrackManager(
        person_classes={0}, vehicle_classes=set(),
        class_names={0:"person"}, object_classes=set())

    # ── pose runtime for head-down indicator (separate from adapter) ────────
    import pose_l960_runtime as _l960
    hd_tracker = _HeadDownTracker()
    # We only need pose for the per-frame head-down indicator.
    # The adapter already does this internally for its decision.
    # We re-use the shared runtime directly here for the overlay.

    cap = cv2.VideoCapture(args.input, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        sys.exit(f"[SLEEP-TEST] Cannot open: {args.input}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 1280)
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    print(f"[SLEEP-TEST] Source {src_w}x{src_h} @ {src_fps:.1f} fps")
    sleep_s = float(os.getenv("SLEEP_SECONDS","120"))
    print(f"[SLEEP-TEST] Decision window: {sleep_s:.0f} s  "
          f"(set SLEEP_SECONDS to override)")

    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                 min(TARGET_FPS,src_fps),(min(src_w,1280),min(src_h,720)))
        print(f"[SLEEP-TEST] Output: {args.output}")

    stills_dir = None
    if args.save_stills:
        stills_dir = os.path.join(os.path.dirname(os.path.abspath(args.input)),"sleep_stills")
        os.makedirs(stills_dir, exist_ok=True)
        print(f"[SLEEP-TEST] Stills: {stills_dir}")

    frame_id     = 0
    total_sleep  = 0
    active       = {}      # track_id -> expiry
    fps_win      = deque(maxlen=30)
    interval     = 1.0 / max(0.5, TARGET_FPS)
    next_due     = time.perf_counter()

    try:
        while True:
            now_pc = time.perf_counter()
            if now_pc < next_due:
                time.sleep(min(next_due-now_pc, 0.01)); continue
            next_due += interval

            ok, frame = cap.read()
            if not ok or frame is None:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                break

            frame_id   += 1
            ts          = frame_id / src_fps
            sleeping.SLEEP_SECONDS = 5.0
            sleeping.SLEEP_MIN_SAMPLES = 2
            sleeping.SLEEP_SAMPLE_SECONDS = 0.5
            canvas      = frame.copy()
            h, w        = frame.shape[:2]
            t0          = time.perf_counter()

            # ── detect + track ─────────────────────────────────────────────
            try:
                res = det_model.predict(source=[frame], imgsz=IMG_SIZE,
                                        conf=CONF, device=0, verbose=False)[0]
            except Exception as e:
                print(f"[SLEEP-TEST] detect: {e}"); res = None

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

            # ── per-frame head-down indicator via shared pose ───────────────
            pose_result = _l960.shared_l960.get_or_infer(CAMERA_ID, frame_id, frame)
            current_down   = {}   # track_id -> bool|None
            current_detail = {}   # track_id -> detail dict

            if pose_result is not None and pose_result.boxes is not None \
                    and pose_result.keypoints is not None and len(pose_result.boxes):
                boxes = pose_result.boxes.xyxy.cpu().numpy()
                kxy   = pose_result.keypoints.xy.cpu().numpy()
                kcf   = (pose_result.keypoints.conf.cpu().numpy()
                         if pose_result.keypoints.conf is not None
                         else np.ones(kxy.shape[:2]))

                # Draw ALL raw skeletons from pose model
                for si in range(len(boxes)):
                    draw_skeleton(canvas, kxy[si], kcf[si], kpt_thresh=0.3)

                matched = sleeping.match_skeletons(
                    [(str(t.track_id), list(t.bbox)) for t in persons],
                    boxes, kxy, kcf)
                for tid_s, bbox_t, kpts, confs in matched:
                    down, detail = sleeping.head_down(kpts, confs, bbox_t)
                    # Annotate shoulder width into detail for display
                    if detail is None:
                        detail = {}
                    lsconf = confs[sleeping.L_SHOULDER]
                    rsconf = confs[sleeping.R_SHOULDER]
                    sw = abs(kpts[sleeping.L_SHOULDER][0] - kpts[sleeping.R_SHOULDER][0])
                    detail['shoulder_w'] = round(float(sw), 1)
                    detail['ls_conf']    = round(float(lsconf), 2)
                    detail['rs_conf']    = round(float(rsconf), 2)
                    current_down[tid_s]   = down
                    current_detail[tid_s] = detail
                    hd_tracker.observe(tid_s, down, ts)
                    if SLEEP_DEBUG:
                        hr  = detail.get('head_rise', '?')
                        rsn = detail.get('reason', '')
                        thr = sleeping.HEAD_DROP_RATIO
                        print(f"[DBG] f={frame_id:05d} T{tid_s}  "
                              f"down={str(down):<6}  rise={hr}  "
                              f"thr={thr}  sw={sw:.1f}px  "
                              f"ls={lsconf:.2f} rs={rsconf:.2f}  "
                              f"{rsn}")

            simulated_now = frame_id / src_fps
            # ── sleep adapter ──────────────────────────────────────────────
            adapter.submit(CAMERA_ID, frame, persons, ts, frame_id, w, h, simulated_now=simulated_now)

            new_msgs = []
            for finding in adapter.flush():
                total_sleep += 1
                tid = str(finding.track_id)
                active[tid] = ts + 8.0
                meta = finding.metadata()
                msg = (f"SLEEPING track={tid}  "
                       f"down={finding.down_fraction:.2f}  "
                       f"span={finding.span:.0f}s  "
                       f"movement={finding.movement:.3f}")
                new_msgs.append(msg)
                print(f"[SLEEP-TEST] *** {msg}")
                label_box(canvas, finding.bbox, f"SLEEPING  P{tid}", C_SLEEP, 4)
                if stills_dir:
                    fname = os.path.join(stills_dir,
                                         f"sleep_{time.strftime('%H%M%S')}_{frame_id:06d}_t{tid}.jpg")
                    cv2.imwrite(fname, canvas)
                    print(f"[SLEEP-TEST] Still: {fname}")
                else:
                    # If not saving all stills, let's at least save the crop of the person who triggered
                    crop_dir = os.path.join(os.path.dirname(os.path.abspath(args.input)), "sleep_crops")
                    os.makedirs(crop_dir, exist_ok=True)
                    x1, y1, x2, y2 = [int(v) for v in finding.bbox]
                    # Add margin
                    mx = int((x2 - x1) * 0.2)
                    my = int((y2 - y1) * 0.2)
                    x1 = max(0, x1 - mx)
                    y1 = max(0, y1 - my)
                    x2 = min(w, x2 + mx)
                    y2 = min(h, y2 + my)
                    crop = canvas[y1:y2, x1:x2]
                    crop_fname = os.path.join(crop_dir, f"crop_{frame_id:06d}_t{tid}.jpg")
                    cv2.imwrite(crop_fname, crop)
                    print(f"[SLEEP-TEST] Crop saved: {crop_fname}")

            active = {tid:exp for tid,exp in active.items() if exp > ts}

            # ── draw tracks ────────────────────────────────────────────────
            for t in persons:
                tid = str(t.track_id)
                is_sleeping = tid in active
                down        = current_down.get(tid)
                detail      = current_detail.get(tid, {})
                frac        = hd_tracker.fraction(tid)
                q           = hd_tracker._samples.get(tid)
                n_judged    = len([s for s in q if s[1] is not None]) if q else 0

                if is_sleeping:
                    label_box(canvas, t.bbox, f"SLEEPING P{tid}", C_SLEEP, 3)
                else:
                    col = C_SLEEP if down is True else C_PERSON
                    label_box(canvas, t.bbox, f"P{tid}", col, 2)

                # head-down arrow indicator
                if down is True:
                    draw_head_down_arrow(canvas, t.bbox)

                # progress bar (head-down fraction)
                bar_col = C_SLEEP if frac >= 0.8 else C_YELLOW
                draw_progress_bar(canvas, t.bbox, frac, bar_col)

                # per-track debug panel
                draw_track_debug(canvas, t.bbox, tid, down, detail,
                                 frac, n_judged, sleeping.SLEEP_MIN_SAMPLES)

            # ── HUD ────────────────────────────────────────────────────────
            ms = (time.perf_counter()-t0)*1000
            fps_win.append(time.time())
            fps = ((len(fps_win)-1)/max(1e-6,fps_win[-1]-fps_win[0])
                   if len(fps_win)>1 else 0.0)
            st = adapter.stats()
            sc = C_SLEEP if active else C_GREEN
            draw_hud(canvas,[
                (f"[SLEEP-DETECTION TEST]  cam={CAMERA_ID}  {w}x{h}  "
                 f"{fps:.1f} fps  {ms:.0f} ms", C_WHITE),
                # (f"persons={len(persons)}  frame={frame_id}", C_WHITE),
                # (f"adapter  submitted={st.get('submitted',0)}  "
                #  f"processed={st.get('processed',0)}  "
                #  f"raised={st.get('raised',0)}  "
                #  f"queue={st.get('queue_depth',0)}", C_WHITE),
                (f"pose  judged={st.get('judged',0)}  "
                 f"head_down={st.get('head_down_samples',0)}  "
                 f"no_pose={st.get('no_pose',0)}", C_WHITE),
                (f"window={sleep_s:.0f}s  TOTAL SLEEPING: {total_sleep}  "
                 + ("*** SLEEPING ACTIVE ***" if active else "no active alert"), sc),
            ])

            if active:
                draw_banner(canvas,[("*** SLEEPING DETECTED ***",C_SLEEP)]
                            + [(m,C_SLEEP) for m in new_msgs[:2]])

            out = canvas
            if w > 1280:
                out = cv2.resize(canvas,(1280,int(h*1280/w)),interpolation=cv2.INTER_AREA)
            if writer: writer.write(out)
            if not args.headless:
                cv2.imshow("Sleep Detection Test", out)
                if cv2.waitKey(1)&0xFF in (ord("q"),27): break

    finally:
        cap.release()
        if writer: writer.release()
        if not args.headless: cv2.destroyAllWindows()
        adapter.close()
        print(f"\n[SLEEP-TEST] Done. frames={frame_id}  events={total_sleep}")
        print("[SLEEP-TEST] Stats:", adapter.stats())


def main():
    ap = argparse.ArgumentParser(
        description="Standalone Sleep Detection – production SleepingAdapter")
    ap.add_argument("--input",       required=True)
    ap.add_argument("--output",      default="", help="Save annotated MP4")
    ap.add_argument("--headless",    action="store_true")
    ap.add_argument("--loop",        action="store_true")
    ap.add_argument("--save-stills", action="store_true")
    run(ap.parse_args())

if __name__ == "__main__":
    main()
