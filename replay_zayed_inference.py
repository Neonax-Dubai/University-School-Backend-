#!/usr/bin/env python3
"""
replay_zayed_inference.py
==========================
Run the COMPLETE Zayed University AI inference pipeline against local video
files (or an NVR-fetched clip) - identical analytics to zayed_inference.py,
but driven by files instead of live RTSP streams.

  fall_detection  fight_detection  sleeping_detection  phone_use
  face_id  reid  people_counting  occupancy  zones  abandoned_object
  camera_health  evacuation  floor_plan  camera_tamper  object_logger

Everything is the same production code.  The only differences from the live
process are:

  * CameraStream is given a local video path instead of an RTSP URL.
    (CameraStream already supports this natively via ffmpeg -stream_loop.)
  * Camera configuration is a local JSON file instead of a dashboard fetch.
    (A minimal default config is generated automatically when --config is omitted.)
  * The event outbox points at the REAL dashboard when credentials are set,
    or falls back to a console-only null sink when they are not.
  * A live annotated display shows all cameras in a grid, with per-camera
    detection boxes, HUD, and alert banners.
  * The watchdog, SIGTERM handler and identity supervisor are omitted (no
    need for container restart semantics in a replay).

Usage
-----
    # Quickstart - replay three test clips, all analytics on, display window
    python replay_zayed_inference.py \\
        --clips cam_01:zayed_test_recordings/camera_01_*.mp4 \\
                cam_02:zayed_test_recordings/camera_02_*.mp4

    # With a config file (enables specific features per camera)
    python replay_zayed_inference.py \\
        --config replay_config.json \\
        --clips cam_01:video1.mp4 cam_02:video2.mp4

    # Headless, save annotated output, send real events to dashboard
    DASHBOARD_URL=http://10.x.x.x:8000 DASHBOARD_TOKEN=xxx \\
    python replay_zayed_inference.py \\
        --clips cam_01:video.mp4 --headless --output-dir /tmp/replay_out/

    # Loop all clips forever (useful for long soak tests)
    python replay_zayed_inference.py --clips cam_01:video.mp4 --loop

    # NVR window via nvr_replay.py (produces a single clip, then run this)
    python nvr_replay.py --start "..." --end "..." --channel 101 --analytic full
        (nvr_replay.py with --analytic full calls this script automatically)

Config file format (JSON)
--------------------------
{
  "cameras": [
    {
      "camera_id": "cam_01",
      "name": "Classroom A",
      "features": {
        "fall_detection": true,
        "violence_detection": true,
        "sleeping_detection": true,
        "mobile_phone_detection": true,
        "face_detection": true,
        "people_counting": true,
        "crowd_detection": true,
        "occupancy_heatmap": true,
        "camera_tamper": true
      },
      "zones": [],
      "classroom": {"classroom_id": "CL01"}
    }
  ]
}

Keyboard controls (display mode)
    q / Esc   quit
    p         pause / resume
    s         step one frame (while paused)

Frame annotations (per camera tile)
    Cyan box   + "Px"           confirmed person track
    Identity   above person box if face/reid recognised
    RED box    + "FALL"         fall event (5 s linger)
    MAGENTA    + "FIGHT"        fight/violence event
    ORANGE     + "SLEEPING"     sleeping event
    YELLOW     + "PHONE"        phone-use event
    HUD top-left                cam / res / fps / infer-ms
    HUD bottom                  active analytics, persons, event counters
    Alert banner                large coloured text when an event fires
"""
import argparse, json, os, sys, time, threading, queue
from collections import defaultdict, deque

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── production modules ───────────────────────────────────────────────────────
import abandoned
import camera_health
import evacuation
import events
import evidence
import face_id_adapter
import fall_pose_adapter
import fight_adapter
import floor_plan
import identity_supervisor
import object_logger
import occupancy
import people_counter
import phone_use
import pose_l960_runtime
import reid_adapter
import sleeping
import tracking
import zones
from cctv import CameraStream
import dashboard

HERE = os.path.dirname(os.path.abspath(__file__))

# ── configuration ────────────────────────────────────────────────────────────
ENGINE_DIR   = os.getenv("ZAYED_ENGINE_DIR", "/engines/zayed/current")
ENGINE_PATH  = os.path.join(ENGINE_DIR, "yolo26l.engine")
CONF         = float(os.getenv("DETECTION_CONFIDENCE", "0.45"))
IMG_SIZE     = 640
TARGET_FPS   = float(os.getenv("TARGET_INFERENCE_FPS", "10.0"))
MAX_BATCH    = int(os.getenv("REPLAY_MAX_BATCH", "4"))
SMOOTH_ALPHA = float(os.getenv("SMOOTH_ALPHA", "0.45"))
MAX_FPC      = int(os.getenv("MAX_FRAMES_PER_CAMERA", "3"))
DISPLAY_W    = int(os.getenv("REPLAY_DISPLAY_W", "1280"))   # grid output width
DEVICE       = "cuda:0"

PERSON_CLASSES  = {0}
CARRIED_CLASSES = set(abandoned.CANDIDATE_CLASS_IDS)
PHONE_CLASSES   = {67}
CLASS_NAMES     = dict(enumerate(object_logger.COCO_NAMES))

# Analytics that arm if no config is provided
ALL_FEATURES = {
    "fall_detection": True, "violence_detection": True,
    "sleeping_detection": True, "mobile_phone_detection": True,
    "face_detection": True, "people_counting": True,
    "crowd_detection": True, "occupancy_heatmap": True,
    # Replaying footage through a camera id changes the view relative to whatever
    # baseline was learned, which is exactly what the tamper detector is built to catch.
    # Arming it during replay therefore manufactures "Camera Tampering" alarms for the
    # operator. Pass a config file with camera_tamper: true if you are testing tampering.
    "camera_tamper": False, "abandoned_object": True,
}
PERSON_FEATURES = ("person_detection", "face_detection", "people_counting",
                   "fall_detection", "violence_detection", "crowd_detection",
                   "sleeping_detection", "evacuation_monitoring", "occupancy_heatmap")

# Alert colours
C = {
    "person":  (0, 200, 255),   # cyan
    "fall":    (0,   0, 255),   # red
    "fight":   (255,  0, 200),  # magenta
    "sleep":   (0, 165, 255),   # orange
    "phone":   (0, 220, 220),   # yellow
    "face":    (0, 255, 80),    # green
    "object":  (200, 200, 0),   # teal
    "alert":   (0,   0, 255),
    "warn":    (0, 180, 255),
    "ok":      (0, 220, 80),
    "white":   (255,255,255),
}


# ── helpers ──────────────────────────────────────────────────────────────────
def log(msg):
    print(msg, flush=True)


def _sc(w):
    return max(0.5, w / 960.0)


def label_box(frame, bbox, text, colour, lw=2):
    h, w = frame.shape[:2]; s = _sc(w)
    x1,y1,x2,y2 = [int(v) for v in bbox]
    cv2.rectangle(frame,(x1,y1),(x2,y2),colour,max(1,int(lw*s)))
    if not text: return
    fs=0.5*s; tk=max(1,int(s))
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,fs,tk)
    p=int(5*s); top=max(0,y1-th-p)
    cv2.rectangle(frame,(x1,top),(x1+tw+p+2,top+th+p),colour,-1)
    cv2.putText(frame,text,(x1+int(3*s),top+th+int(s)),
                cv2.FONT_HERSHEY_SIMPLEX,fs,(0,0,0),tk,cv2.LINE_AA)


def draw_hud(frame, lines):
    h,w=frame.shape[:2]; s=_sc(w); step=int(22*s)
    for i,(text,col) in enumerate(lines):
        orig=(int(8*s),int(22*s)+i*step)
        cv2.putText(frame,text,orig,cv2.FONT_HERSHEY_SIMPLEX,0.55*s,
                    (0,0,0),max(3,int(4*s)),cv2.LINE_AA)
        cv2.putText(frame,text,orig,cv2.FONT_HERSHEY_SIMPLEX,0.55*s,
                    col,max(1,int(s)),cv2.LINE_AA)


def draw_banner(frame, lines):
    if not lines: return
    h,w=frame.shape[:2]; s=_sc(w); step=int(26*s)
    bh=step*len(lines)+int(8*s)
    ov=frame.copy()
    cv2.rectangle(ov,(0,h-bh),(w,h),(0,0,40),-1)
    cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
    for i,(text,col) in enumerate(lines):
        cv2.putText(frame,text,(int(8*s),h-bh+int(18*s)+i*step),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65*s,col,max(2,int(2*s)),cv2.LINE_AA)


# ── camera config ─────────────────────────────────────────────────────────────
def _default_config(camera_id):
    """Generate a minimal CameraConfig with all features on."""
    return dashboard.CameraConfig(
        camera_id=camera_id,
        name=camera_id,
        source="",                   # filled in from --clips
        features=dict(ALL_FEATURES),
        zones=[],
        classroom={"classroom_id": camera_id},
        detection_confidence=CONF,
        intrusion_interval=30,
        loitering_interval=30,
        object_classes=[],
    )


def load_configs(config_path, clip_map):
    """Load camera configs from JSON or generate defaults for each clip."""
    if config_path and os.path.exists(config_path):
        raw = json.load(open(config_path))
        cfgs = []
        for entry in raw.get("cameras", []):
            features = entry.get("features", dict(ALL_FEATURES))
            cfgs.append(dashboard.CameraConfig(
                camera_id=entry["camera_id"],
                name=entry.get("name", entry["camera_id"]),
                source=clip_map.get(entry["camera_id"], ""),
                features=features,
                zones=entry.get("zones", []),
                classroom=entry.get("classroom", {"classroom_id": entry["camera_id"]}),
                detection_confidence=float(entry.get("detection_confidence", CONF)),
                intrusion_interval=float(entry.get("intrusion_interval", 30)),
                loitering_interval=float(entry.get("loitering_interval", 30)),
                object_classes=entry.get("object_classes", []),
            ))
        # Fill in sources from --clips for any matching camera
        for c in cfgs:
            if c.camera_id in clip_map:
                c.source = clip_map[c.camera_id]
        return cfgs
    # Auto-generate one config per clip
    cfgs = []
    for camera_id in sorted(clip_map):
        c = _default_config(camera_id)
        c.source = clip_map[camera_id]
        cfgs.append(c)
    return cfgs


def classes_for(camera):
    allowed = set()
    if any(camera.features.get(k) for k in PERSON_FEATURES):
        allowed |= PERSON_CLASSES
    if camera.features.get("abandoned_object"):
        allowed |= CARRIED_CLASSES | PERSON_CLASSES
    if camera.features.get("mobile_phone_detection"):
        allowed |= PHONE_CLASSES | PERSON_CLASSES
    if camera.features.get(object_logger.OBJECT_DETECTION_FEATURE):
        allowed |= object_logger.class_ids_for(getattr(camera, "object_classes", None))
    return allowed


# ── null event sink (when no dashboard credentials) ──────────────────────────
class _ConsoleEventSink:
    """Prints events to stdout instead of posting to the dashboard."""
    sent = failed = dropped = duplicates = dead = retries = enqueued = 0

    def submit(self, event, notable=False):
        self.enqueued += 1
        self.sent += 1
        etype = event.get("event_type", "?")
        cam   = event.get("camera_id", "?")
        log(f"[EVENT-CONSOLE] {etype}  cam={cam}  "
            f"{json.dumps({k:v for k,v in event.items() if k not in ('camera_id','event_type','event_id')}, default=str)[:120]}")

    def start(self): pass
    def stop(self, timeout=0): pass
    def pending(self): return 0
    def dead_letters(self): return 0
    def stats(self):
        return {"sent": self.sent, "enqueued": self.enqueued, "pending": 0, "dead_letters": 0,
                "path": "(console only)"}


# ── per-camera display state ──────────────────────────────────────────────────
class _CamDisplay:
    def __init__(self, camera_id):
        self.camera_id  = camera_id
        self.canvas     = None      # latest annotated frame
        self.fps_win    = deque(maxlen=30)
        self.last_frame = None
        self.active_alerts = {}     # label -> expiry
        self.total_events  = defaultdict(int)
        self.lock = threading.Lock()

    def record_event(self, label, linger=5.0):
        with self.lock:
            self.active_alerts[label] = time.time() + linger
            self.total_events[label] += 1

    def prune_alerts(self, now):
        with self.lock:
            self.active_alerts = {k:v for k,v in self.active_alerts.items() if v > now}

    def alerts(self):
        with self.lock:
            return list(self.active_alerts.keys())


# ── main replay engine ────────────────────────────────────────────────────────
class ZayedReplay:

    def __init__(self, args, camera_configs):
        self.args           = args
        self.camera_configs = camera_configs
        self.cam_display    = {c.camera_id: _CamDisplay(c.camera_id) for c in camera_configs}

        # ── GPU detector ───────────────────────────────────────────────────
        log(f"[REPLAY] Loading {ENGINE_PATH} ...")
        from ultralytics import YOLO
        self.model = YOLO(ENGINE_PATH, task="detect")
        self.model.predict([np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)],
                           imgsz=IMG_SIZE, conf=CONF, device=DEVICE, verbose=False)
        log("[REPLAY] YOLO26-L engine ready.")

        # ── tracking ───────────────────────────────────────────────────────
        self.track_manager = tracking.TrackManager(
            person_classes=PERSON_CLASSES, vehicle_classes=set(),
            class_names=CLASS_NAMES,
            object_classes=CARRIED_CLASSES | PHONE_CLASSES | object_logger.LOGGABLE_CLASS_IDS)

        self.smooth_bboxes  = {}
        self.camera_classes = {c.camera_id: classes_for(c) for c in camera_configs}
        self.camera_conf    = {c.camera_id: c.detection_confidence for c in camera_configs}

        # ── event pipeline ─────────────────────────────────────────────────
        dash_url   = os.getenv("DASHBOARD_URL",   getattr(dashboard, "DASHBOARD_URL",   ""))
        dash_token = os.getenv("DASHBOARD_TOKEN", getattr(dashboard, "DASHBOARD_TOKEN", ""))
        if dash_url and dash_token:
            import outbox as _outbox
            import events as _events_mod
            sender = _outbox.OutboxEventSender(
                dash_url, dash_token,
                path=os.path.join(HERE, "runtime", "replay_outbox", "events.sqlite3"),
                enrich=_events_mod.zayed_enrich)
            log(f"[REPLAY] Events -> {dash_url}")
        else:
            sender = _ConsoleEventSink()
            log("[REPLAY] No DASHBOARD_URL/TOKEN - events printed to console.")

        self.event_pipeline = events.EventPipeline(
            dashboard_url=dash_url or "http://localhost",
            token=dash_token or "none")
        if hasattr(self.event_pipeline, "sender"):
            self.event_pipeline.sender = sender
        sender.start()
        self.event_pipeline.update_camera_intervals(
            {c.camera_id: {"intrusion": c.intrusion_interval,
                           "loitering": c.loitering_interval}
             for c in camera_configs})

        # ── analytics adapters ─────────────────────────────────────────────
        self.reid_inst   = reid_adapter.ReIDAdapter()
        events.set_stable_id_resolver(self.reid_inst.stable_id_for)

        self.face_inst   = face_id_adapter.build_production_adapter()
        self.fall_inst   = fall_pose_adapter.build_production_adapter()
        self.fight_inst  = fight_adapter.build_production_adapter()

        self.face_inst.set_enabled_cameras(self._armed(face_id_adapter.FACE_ID_CAMERA_FEATURE))
        self.fall_inst.set_enabled_cameras(self._armed(fall_pose_adapter.FALL_CAMERA_FEATURE))
        self.fight_inst.set_enabled_cameras(self._armed(fight_adapter.FIGHT_CAMERA_FEATURE))

        self.cam_health    = camera_health.CameraHealthManager(shadow=False,
                               event_sink=self._handle_cam_health)
        self.cam_health.set_enabled_cameras(
            {c.camera_id: {k: bool(c.features.get(k)) for k in camera_health.CAMERA_HEALTH_FEATURES}
             for c in camera_configs})

        self.people_ctr    = people_counter.PeopleCounter()
        self.people_ctr.set_enabled_cameras(
            {c.camera_id: c.zones for c in camera_configs
             if c.features.get(people_counter.PEOPLE_COUNTING_FEATURE)})

        classroom_by_cam   = {c.camera_id: (c.classroom or {}).get("classroom_id")
                              for c in camera_configs}
        events.set_classroom_resolver(classroom_by_cam.get)

        self.occ_inst      = occupancy.ClassroomOccupancy()
        self.occ_inst.configure(camera_configs)
        self.obj_logger    = object_logger.ObjectLogger()
        self.obj_logger.set_enabled_cameras(
            {c.camera_id: object_logger.class_ids_for(c.object_classes)
             for c in camera_configs
             if c.features.get(object_logger.OBJECT_DETECTION_FEATURE)})

        self.zone_eval     = zones.ZoneEvaluator(
            camera_zones={c.camera_id: c.zones for c in camera_configs},
            camera_features={c.camera_id: c.features for c in camera_configs})

        self.abandon_det   = abandoned.AbandonedObjectDetector(
            camera_features={c.camera_id: c.features for c in camera_configs})

        self.phone_det     = phone_use.PhoneUseDetector()
        self.phone_det.set_enabled_cameras(self._armed(phone_use.PHONE_FEATURE))

        self.sleep_adapt   = sleeping.SleepingAdapter(log=log)
        self.sleep_adapt.set_enabled_cameras(self._armed(sleeping.SLEEP_FEATURE))

        self.evac_mon      = evacuation.EvacuationMonitor(
            dash_url or "http://localhost", dash_token or "none", log=log)
        self.evac_mon.configure(camera_configs)

        self.floor_mapper  = floor_plan.FloorPlanMapper(log=log)
        self.floor_mapper.configure(camera_configs)
        self.occ_inst.set_fusion(self.floor_mapper)

        # ── video writers (optional) ───────────────────────────────────────
        self.writers = {}
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

        # ── per-camera streams ─────────────────────────────────────────────
        self.streams = {}
        self.cam_stats = {}
        for cfg in camera_configs:
            cs = CameraStream(camera_id=cfg.camera_id,
                              source=cfg.source,
                              queue_size=3,
                              loop_video=args.loop)
            cs.start()
            self.streams[cfg.camera_id] = cs
            self.cam_stats[cfg.camera_id] = {"frames": 0, "errors": 0}
        log(f"[REPLAY] {len(self.streams)} camera stream(s) started.")

        self._last_inferred = {}
        self._paused = False
        self._step   = False

    def _armed(self, feature):
        return sorted(c.camera_id for c in self.camera_configs if c.features.get(feature))

    # ── frame collection (mirrors zayed_inference._frames_to_infer) ──────────
    def _frames_to_infer(self, camera):
        packets  = camera.drain_frames(MAX_FPC)
        if not packets:
            return []
        spacing  = 1.0 / TARGET_FPS
        last     = self._last_inferred.get(camera.camera_id)
        keep     = []
        for p in packets:
            if last is not None and 0.0 <= (p["timestamp"] - last) < spacing:
                continue
            keep.append(p)
            last = p["timestamp"]
        if not keep:
            keep = packets[-1:]
        self._last_inferred[camera.camera_id] = keep[-1]["timestamp"]
        return keep

    # ── camera health sink ────────────────────────────────────────────────────
    def _handle_cam_health(self, camera_id, event_type, timestamp, metadata,
                           action, frame_width, frame_height):
        self.event_pipeline.handle_camera_event(
            camera_id, event_type, metadata,
            frame_width=frame_width or 1920,
            frame_height=frame_height or 1080,
            observed_at=timestamp,
            bbox=[0, 0, frame_width or 1920, frame_height or 1080],
            scope=action)
        cd = self.cam_display.get(camera_id)
        if cd:
            cd.record_event(f"TAMPER:{event_type}", 8.0)

    # ── emit one finding event ────────────────────────────────────────────────
    def _emit(self, event_type, finding, frame=None, label=""):
        meta   = finding.metadata()
        loc    = self.floor_mapper.location_for(finding.camera_id, finding.bbox)
        if loc:
            meta["location"] = loc
        stream = self.streams.get(finding.camera_id)
        w = getattr(finding, "frame_width", None) or getattr(stream, "width", 0) or 2592
        h = getattr(finding, "frame_height", None) or getattr(stream, "height", 0) or 1944
        result = self.event_pipeline.handle_camera_event(
            finding.camera_id, event_type, meta, frame_width=w, frame_height=h,
            observed_at=finding.observed_at, bbox=finding.bbox,
            frame=frame if frame is not None else getattr(finding, "frame", None),
            scope=finding.scope())
        if hasattr(finding, "frame"):
            finding.frame = None
        if result == "sent":
            log(f"[{label or event_type}] {finding.camera_id} {finding.scope()}")
        return result

    # ── per-frame processing (mirrors zayed_inference.process_result) ─────────
    def process_result(self, result, meta, src):
        camera_id   = meta["camera_id"]
        fw, fh      = meta["frame_width"], meta["frame_height"]
        observed_at = meta["timestamp"]
        frame_id    = meta["frame_id"]
        cd          = self.cam_display[camera_id]
        now         = time.time()

        # ── detection parse ────────────────────────────────────────────────
        dets, phones = [], []
        if result.boxes is not None and len(result.boxes):
            allowed  = self.camera_classes.get(camera_id, set())
            min_conf = self.camera_conf.get(camera_id, CONF)
            ab_armed = self.abandon_det.enabled_for(camera_id)
            ph_armed = self.phone_det.enabled_for(camera_id)
            cls_ids  = result.boxes.cls.detach().cpu().numpy().astype(int)
            confs    = result.boxes.conf.detach().cpu().numpy()
            xyxy     = result.boxes.xyxy.detach().cpu().numpy()
            for cid, cf, box in zip(cls_ids, confs, xyxy):
                if cid not in allowed: continue
                if ph_armed and cid == phone_use.PHONE_CLASS_ID and cf >= phone_use.PHONE_CONFIDENCE:
                    phones.append((float(cf), [float(v) for v in box])); continue
                if cf < min_conf and not (ab_armed and cid in abandoned.CANDIDATE_CLASS_IDS
                                          and cf >= abandoned.OBJECT_CONFIDENCE): continue
                dets.append((int(cid), float(cf),
                             (int(box[0]),int(box[1]),int(box[2]),int(box[3]))))

        tracked = self.track_manager.update(camera_id, dets, now=observed_at)

        # ── bbox smoothing ─────────────────────────────────────────────────
        if SMOOTH_ALPHA < 1.0:
            for t in tracked:
                key = (camera_id, t.track_id)
                rx1,ry1,rx2,ry2 = t.bbox
                prev = self.smooth_bboxes.get(key)
                if prev is None:
                    s = [float(rx1),float(ry1),float(rx2),float(ry2)]
                else:
                    a = SMOOTH_ALPHA
                    s = [a*rx1+(1-a)*prev[0], a*ry1+(1-a)*prev[1],
                         a*rx2+(1-a)*prev[2], a*ry2+(1-a)*prev[3]]
                self.smooth_bboxes[key] = s
                t.bbox = [int(s[0]),int(s[1]),int(s[2]),int(s[3])]
            active = {(camera_id, t.track_id) for t in tracked}
            for k in [k for k in self.smooth_bboxes if k[0]==camera_id and k not in active]:
                del self.smooth_bboxes[k]

        # ── adapters ───────────────────────────────────────────────────────
        self.reid_inst.process_tracks(camera_id, tracked, src, observed_at)
        self.face_inst.submit(camera_id, tracked, src, observed_at)
        self.fight_inst.submit(camera_id, src, tracked, observed_at, frame_id=frame_id)
        self.fall_inst.submit(camera_id, src, tracked, observed_at, frame_id=frame_id)

        obj_tracks  = [t for t in tracked if t.group == tracking.OBJECT_GROUP]
        motion      = [t for t in tracked if t.group != tracking.OBJECT_GROUP]
        persons     = [t for t in tracked if t.group == tracking.PERSON_GROUP]

        for t in motion:
            for zt, zm in self.zone_eval.evaluate(t, fw, fh):
                self.event_pipeline.handle_zone_event(t, zt, zm, frame_width=fw,
                    frame_height=fh, observed_at=observed_at, frame=src)

        pb = [t.bbox for t in persons]
        for ot, am in self.abandon_det.update(camera_id, [t for t in obj_tracks
                                              if t.class_id in abandoned.CANDIDATE_CLASS_IDS],
                                              pb, fw, fh):
            if self.event_pipeline.handle_abandoned_event(ot, am, frame_width=fw,
                    frame_height=fh, observed_at=observed_at, frame=src) == "sent":
                log(f"[UNATTENDED] {camera_id} {ot.class_name} track={ot.track_id}")
            cd.record_event("UNATTENDED", 5.0)

        for ot, om, cb in self.obj_logger.observe(camera_id, obj_tracks, fw, fh, observed_at):
            self.event_pipeline.handle_object_event(ot, om, frame_width=fw,
                frame_height=fh, observed_at=observed_at, frame=src, crop_box=cb)

        for ct, cm, cbbox, cp in self.zone_eval.evaluate_crowd(camera_id, tracked, fw, fh):
            self.event_pipeline.handle_camera_event(camera_id, ct, cm, frame_width=fw,
                frame_height=fh, observed_at=observed_at, bbox=cbbox,
                frame=src, contributors=cp, scope=cm.get("zone_id"))
            cd.record_event("OVERCROWD", 5.0)

        self.people_ctr.observe(camera_id, tracked, fw, fh)
        self.occ_inst.observe(camera_id, tracked, observed_at)
        self.floor_mapper.observe(camera_id, persons, observed_at)

        for finding in self.phone_det.update(camera_id, persons, phones, observed_at):
            self._emit(phone_use.EVENT_TYPE, finding, frame=src, label="PHONE")
            cd.record_event("PHONE", 5.0)

        self.sleep_adapt.submit(camera_id, src, persons, observed_at, frame_id, fw, fh)

        for finding in self.evac_mon.observe(camera_id, persons, src, fw, fh, observed_at):
            self._emit(evacuation.EVENT_TYPE, finding, label="STRANDED")
            cd.record_event("STRANDED", 8.0)

        self.cam_stats[camera_id]["frames"] += 1

        # ── annotate frame for display ─────────────────────────────────────
        if not self.args.headless or self.args.output_dir:
            canvas = src.copy()
            now_ts = time.time()
            cd.prune_alerts(now_ts)
            alerts = cd.alerts()

            active_tids = set(str(t.track_id) for t in persons)

            # person boxes
            for t in persons:
                tid   = str(t.track_id)
                label = f"P{tid}"
                col   = C["person"]
                # check if we should colour it with an active event
                for ev_key, ev_col in (("FALL","fall"),("FIGHT","fight"),
                                       ("SLEEP","sleep"),("PHONE","phone")):
                    if any(ev_key in a for a in alerts):
                        col = C[ev_col]; label = f"{ev_key} P{tid}"
                label_box(canvas, t.bbox, label, col, 2)

            # phone boxes
            if phones:
                for cf_ph, box_ph in phones:
                    label_box(canvas, box_ph, f"PHONE {cf_ph:.2f}", C["phone"], 2)

            # HUD
            fps_w = cd.fps_win
            fps_w.append(time.time())
            fps_live = ((len(fps_w)-1)/max(1e-6,fps_w[-1]-fps_w[0])
                        if len(fps_w)>1 else 0.0)
            armed = [k for k,v in self.camera_configs[0].features.items() if v][:4]

            draw_hud(canvas, [
                (f"{camera_id}  {fw}x{fh}  {fps_live:.1f}fps  f={frame_id}", C["white"]),
                (f"persons={len(persons)}  objects={len(obj_tracks)}", C["white"]),
                (f"events  fall={cd.total_events.get('FALL',0)}  "
                 f"fight={cd.total_events.get('FIGHT',0)}  "
                 f"sleep={cd.total_events.get('SLEEP',0)}  "
                 f"phone={cd.total_events.get('PHONE',0)}", C["white"]),
            ])

            # alert banner
            if alerts:
                banner_lines = []
                col_map = {"FALL":C["fall"],"FIGHT":C["fight"],"SLEEP":C["sleep"],
                           "PHONE":C["phone"],"STRANDED":C["warn"],"UNATTENDED":C["warn"]}
                for a in alerts[:3]:
                    col = next((v for k,v in col_map.items() if k in a), C["alert"])
                    banner_lines.append((f"!!! {a} !!!", col))
                draw_banner(canvas, banner_lines)

            with cd.lock:
                cd.canvas = canvas

            # write output video
            if self.args.output_dir:
                if camera_id not in self.writers:
                    out_path = os.path.join(self.args.output_dir, f"{camera_id}_replay.mp4")
                    fps_out  = min(TARGET_FPS, 25.0)
                    tw = min(fw, 1280); th = int(fh * tw / fw)
                    self.writers[camera_id] = cv2.VideoWriter(
                        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (tw, th))
                    log(f"[REPLAY] Output: {out_path}")
                tw = min(fw, 1280); th = int(fh * tw / fw)
                out_frame = cv2.resize(canvas, (tw, th), interpolation=cv2.INTER_AREA)
                self.writers[camera_id].write(out_frame)

    # ── grid display compositor ───────────────────────────────────────────────
    def _build_grid(self, tiles_w, tile_h):
        cols = max(1, DISPLAY_W // tiles_w)
        rows = (len(self.cam_display) + cols - 1) // cols
        grid_h = rows * tile_h
        grid   = np.zeros((grid_h, DISPLAY_W, 3), dtype=np.uint8)
        for idx, (cid, cd) in enumerate(self.cam_display.items()):
            row, col = divmod(idx, cols)
            x = col * tiles_w; y = row * tile_h
            with cd.lock:
                frame = cd.canvas
            if frame is None:
                continue
            thumb = cv2.resize(frame, (tiles_w, tile_h), interpolation=cv2.INTER_AREA)
            grid[y:y+tile_h, x:x+tiles_w] = thumb
        return grid

    # ── flush loop (mirrors zayed_inference main flush section) ──────────────
    def _flush(self):
        now = time.time()
        try:
            self.reid_inst.flush()
        except Exception as e:
            log(f"[FLUSH] reid: {e}")

        try:
            for fight in (self.fight_inst.flush() or []):
                tid_a, tid_b = str(fight.track_a), str(fight.track_b)
                self.event_pipeline.handle_camera_event(
                    fight.camera_id, fight_adapter.EVENT_TYPE, fight.metadata(),
                    frame_width=fight.frame_width, frame_height=fight.frame_height,
                    observed_at=now, bbox=fight.bbox,
                    frame=getattr(fight,"frame",None), scope=fight.scope())
                fight.frame = None
                cd = self.cam_display.get(fight.camera_id)
                if cd:
                    cd.record_event(f"FIGHT P{tid_a}~P{tid_b}", 6.0)
                log(f"[FIGHT] {fight.camera_id}  {tid_a} vs {tid_b}  "
                    f"strikes={fight.strikes_a}/{fight.strikes_b}")
        except Exception as e:
            log(f"[FLUSH] fight: {e}")

        try:
            for fall in (self.fall_inst.flush() or []):
                m = fall.metadata()
                self.event_pipeline.handle_camera_event(
                    fall.camera_id, fall_pose_adapter.EVENT_TYPE, m,
                    frame_width=fall.frame_width, frame_height=fall.frame_height,
                    observed_at=now, bbox=fall.bbox,
                    frame=getattr(fall,"frame",None), scope=fall.scope())
                fall.frame = None
                cd = self.cam_display.get(fall.camera_id)
                if cd:
                    cd.record_event(f"FALL P{fall.track_id}", 6.0)
                log(f"[FALL] {fall.camera_id} track={fall.track_id}  "
                    f"torso={m.get('torso_angle')}deg  held={m.get('pose_duration')}s")
        except Exception as e:
            log(f"[FLUSH] fall: {e}")

        try:
            for event_type, classroom_id, meta in self.occ_inst.decisions(now):
                cams = self.occ_inst.classroom_cameras(classroom_id)
                if cams:
                    stream = self.streams.get(cams[0])
                    w = getattr(stream,"width",0) or 2592
                    h = getattr(stream,"height",0) or 1944
                    self.event_pipeline.handle_camera_event(
                        cams[0], event_type, dict(meta, classroom_id=classroom_id),
                        frame_width=w, frame_height=h, observed_at=now,
                        bbox=[0,0,w,h], scope=classroom_id)
        except Exception as e:
            log(f"[FLUSH] occupancy: {e}")

        try:
            for finding in self.sleep_adapt.flush():
                self._emit(sleeping.EVENT_TYPE, finding, label="SLEEPING")
                cd = self.cam_display.get(finding.camera_id)
                if cd:
                    cd.record_event(f"SLEEP P{finding.track_id}", 8.0)
        except Exception as e:
            log(f"[FLUSH] sleep: {e}")

        try:
            self.floor_mapper.tick(now)
        except Exception as e:
            log(f"[FLUSH] floor: {e}")

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        interval      = 1.0 / TARGET_FPS
        last_infer    = 0.0
        last_status   = time.perf_counter()
        frames_window = 0
        n_cams        = len(self.camera_configs)
        tile_w        = DISPLAY_W // max(1, n_cams)
        tile_h        = max(360, tile_w * 9 // 16)
        infer_ms_buf  = deque(maxlen=300)

        log(f"\n[REPLAY] Pipeline running.  {n_cams} camera(s).  "
            f"Press q/Esc to quit, p to pause.")

        try:
            while True:
                now_pc = time.perf_counter()
                if now_pc - last_infer < interval:
                    time.sleep(0.002); continue
                last_infer = now_pc

                # ── pause / step ───────────────────────────────────────────
                if self._paused and not self._step:
                    time.sleep(0.05); continue
                self._step = False

                # ── collect frames from all cameras ────────────────────────
                batch_frames, batch_meta = [], []
                for cam in self.streams.values():
                    if not self.camera_classes.get(cam.camera_id):
                        continue
                    for pkt in self._frames_to_infer(cam):
                        f = pkt["frame"]
                        if f is None or getattr(f,"ndim",0)!=3: continue
                        h2,w2 = f.shape[:2]
                        if h2 < 2 or w2 < 2: continue
                        batch_frames.append(f)
                        batch_meta.append({
                            "camera_id":   pkt["camera_id"],
                            "frame_id":    pkt["frame_id"],
                            "timestamp":   pkt["timestamp"],
                            "frame_width": w2,
                            "frame_height": h2,
                        })

                # ── YOLO batch predict ─────────────────────────────────────
                if batch_frames:
                    results = []
                    t0 = time.perf_counter()
                    try:
                        for ci in range(0, len(batch_frames), MAX_BATCH):
                            results.extend(self.model.predict(
                                source=batch_frames[ci:ci+MAX_BATCH],
                                imgsz=IMG_SIZE, conf=CONF,
                                device=DEVICE, verbose=False))
                    except Exception as e:
                        log(f"[REPLAY] predict error: {e}"); results=[]
                    if results:
                        infer_ms_buf.append((time.perf_counter()-t0)*1000)
                        frames_window += len(results)
                    for res, meta, src in zip(results, batch_meta, batch_frames):
                        try:
                            self.process_result(res, meta, src)
                        except Exception as e:
                            log(f"[REPLAY] process_result {meta['camera_id']}: {e}")

                # ── flush adapters ─────────────────────────────────────────
                self._flush()

                # ── status ─────────────────────────────────────────────────
                elapsed = time.perf_counter() - last_status
                if elapsed >= 10.0:
                    fps = frames_window / elapsed if elapsed else 0
                    p50 = sorted(infer_ms_buf)[len(infer_ms_buf)//2] if infer_ms_buf else 0
                    ev  = self.event_pipeline.stats()
                    log(f"[STATUS] fps={fps:.1f}  infer_p50={p50:.0f}ms  "
                        f"events_sent={ev.get('sent',0)}  pending={ev.get('pending',0)}")
                    frames_window = 0
                    last_status   = time.perf_counter()

                # ── display ────────────────────────────────────────────────
                if not self.args.headless:
                    grid = self._build_grid(tile_w, tile_h)
                    cv2.imshow("Zayed Inference Replay", grid)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    elif key == ord("p"):
                        self._paused = not self._paused
                        log(f"[REPLAY] {'PAUSED' if self._paused else 'RESUMED'}")
                    elif key == ord("s"):
                        self._step = True

        except KeyboardInterrupt:
            log("\n[REPLAY] Interrupted.")
        finally:
            self._shutdown()

    def _shutdown(self):
        log("[REPLAY] Shutting down ...")
        for cs in self.streams.values():
            cs.stop()
        for w in self.writers.values():
            w.release()
        if not self.args.headless:
            cv2.destroyAllWindows()
        for closer in (self.reid_inst.close, self.face_inst.close,
                       self.fight_inst.close, self.fall_inst.close,
                       self.sleep_adapt.close, pose_l960_runtime.shared_l960.close):
            try: closer()
            except Exception as e: log(f"[SHUTDOWN] {e}")
        try: self.event_pipeline.stop()
        except Exception: pass
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log("[REPLAY] Done.")


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Replay full Zayed inference pipeline on video files or NVR clips")
    ap.add_argument("--clips", nargs="+", metavar="CAM_ID:PATH",
                    required=True,
                    help='One or more "camera_id:/path/to/clip.mp4" entries')
    ap.add_argument("--config", default="",
                    help="Optional JSON config file (camera features, zones, classroom)")
    ap.add_argument("--headless", action="store_true",
                    help="No display window")
    ap.add_argument("--loop", action="store_true",
                    help="Loop all video files indefinitely")
    ap.add_argument("--output-dir", default="",
                    help="Directory to save per-camera annotated MP4 files")
    args = ap.parse_args()

    # parse clip map
    clip_map = {}
    for item in args.clips:
        if ":" not in item:
            ap.error(f'--clips: expected "camera_id:/path/to/file", got "{item}"')
        cam_id, path = item.split(":", 1)
        if not os.path.exists(path):
            ap.error(f"Clip not found: {path}")
        clip_map[cam_id] = path

    camera_configs = load_configs(args.config, clip_map)
    if not camera_configs:
        ap.error("No camera configs resolved.")

    log(f"[REPLAY] Cameras: {[c.camera_id for c in camera_configs]}")
    for c in camera_configs:
        log(f"  {c.camera_id}  source={c.source}")
        log(f"    features: {[k for k,v in c.features.items() if v]}")

    engine = ZayedReplay(args, camera_configs)
    engine.run()


if __name__ == "__main__":
    main()
