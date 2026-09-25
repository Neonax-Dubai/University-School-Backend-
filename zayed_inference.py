"""
Zayed University classroom CCTV inference - the main loop.

Derived from the Dubai multicam_inf.py (baseline 5b192cef). The per-tick shape is the proven
Dubai one - collect frames -> ONE batched TensorRT YOLO26-L predict -> per-camera parse ->
JeztSort tracking -> analytics -> batched flushes -> events - and every analytic module is the
Dubai module copied verbatim (see docs/DUBAI_BASELINE_CHECKSUMS.txt). Functions reused from
multicam_inf.py are marked "(Dubai, verbatim)".

WHAT CHANGED FOR ZAYED
  * Three classroom cameras via MediaMTX; configuration and classroom placement come from the
    Zayed dashboard backend (dashboard.py), with a secret-free last-known-good cache. Startup
    retries instead of exiting when the backend is unreachable.
  * Dubai-only analytics are not loaded (ANPR, PPE/uniform, fence, fire/smoke, distancing,
    detainee behaviour, vehicles, weapons). No anonymous per-track events: presence is identity
    based and decided by the backend from known-person sightings.
  * CONTAINMENT (the Dubai loop exited on any unexpected exception): each camera's analytics,
    each flush and each predict chunk are guarded. A predict failure skips the tick; only
    MAX_PREDICT_FAILURES consecutive failures exit the process, so the supervisor/container
    restart policy restarts it with a fresh CUDA context.
  * TensorRT engines are REQUIRED and must have been built on this machine by
    tools/bootstrap_models.sh; there is no silent .pt fallback.
  * Durable event outbox (outbox.py) and the Zayed event vocabulary (events.zayed_enrich).
  * An INTERNAL, GPU-only face API (internal_api.py) so the dashboard's enrollment and photo
    search compute their embeddings on this GPU instead of on its own CPU. ML only: every
    business rule, Qdrant write and person record stays in the dashboard.
  * Classroom occupancy de-duplicated across cameras (occupancy.py), Zayed analytics for mobile
    phones (phone_use.py), sleeping (sleeping.py), evacuation / stranded people (evacuation.py)
    and floor-plan coordinates + heatmap (floor_plan.py).
"""
import json
import os
import signal
import sys
import threading
import time
from collections import defaultdict, deque

import torch
from ultralytics import YOLO

import abandoned
import camera_health
import dashboard
import evacuation
import events
import evidence
import face_id_adapter
import fall_pose_adapter
import fight_adapter
import floor_plan
import identity_supervisor
import internal_api
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

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# CONFIGURATION
# ============================================================
ENGINE_DIR = os.getenv("ZAYED_ENGINE_DIR", "/engines/zayed/current")
ENGINE_PATH = os.path.join(ENGINE_DIR, "yolo26l.engine")
CONF = float(os.getenv("DETECTION_CONFIDENCE", "0.45"))
IMG_SIZE = 640
TARGET_INFERENCE_FPS = float(os.getenv("TARGET_INFERENCE_FPS", "10.0"))
SMOOTH_ALPHA = float(os.getenv("SMOOTH_ALPHA", "0.45"))
MAX_FRAMES_PER_CAMERA = int(os.getenv("MAX_FRAMES_PER_CAMERA", "3"))
MAX_PREDICT_FAILURES = int(os.getenv("MAX_PREDICT_FAILURES", "20"))
STATUS_SECONDS = float(os.getenv("STATUS_SECONDS", "10"))
#: A main loop that stops ticking for this long is hung (a stuck CUDA call, a deadlock): the
#: process exits so the container restart policy starts a fresh one. Docker restarts on EXIT
#: only, so without this a hang would be permanent.
WATCHDOG_SECONDS = float(os.getenv("WATCHDOG_SECONDS", "120"))
METRICS_PATH = os.getenv("METRICS_PATH", os.path.join(HERE, "runtime", "metrics.json"))
DEVICE = "cuda:0"

PERSON_CLASSES = {0}
#: Carried items for the unattended-object analytic: backpack, handbag, suitcase.
CARRIED_CLASSES = set(abandoned.CANDIDATE_CLASS_IDS)
PHONE_CLASSES = {67}                                   # COCO "cell phone"
CLASS_NAMES = dict(enumerate(object_logger.COCO_NAMES))
PERSON_FEATURES = ("person_detection", "face_detection", "people_counting", "fall_detection",
                   "violence_detection", "crowd_detection", "sleeping_detection",
                   "evacuation_monitoring", "occupancy_heatmap")


def classes_for(camera):
    """COCO ids this camera's analytics need (Zayed version of the Dubai function)."""
    allowed = set()
    if any(camera.features.get(key) for key in PERSON_FEATURES):
        allowed |= PERSON_CLASSES
    if camera.features.get("abandoned_object"):
        allowed |= CARRIED_CLASSES | PERSON_CLASSES
    if camera.features.get("mobile_phone_detection"):
        allowed |= PHONE_CLASSES | PERSON_CLASSES
    if camera.features.get(object_logger.OBJECT_DETECTION_FEATURE):
        allowed |= object_logger.class_ids_for(getattr(camera, "object_classes", None))
    return allowed


def log(message):
    print(message, flush=True)


# ============================================================
# GPU - no CPU fallback
# ============================================================
log("======================================")
log("Zayed University CCTV AI - YOLO26-L TensorRT FP16")
log("======================================")
if not torch.cuda.is_available():
    log("FATAL: CUDA is not available - refusing to run inference on the CPU")
    raise SystemExit(2)
log(f"Torch {torch.__version__} | GPU {torch.cuda.get_device_name(0)}")


def check_engine():
    """The engine must exist and have been built by the bootstrap for THIS TensorRT/GPU."""
    import tensorrt as trt
    sidecar = ENGINE_PATH + ".json"
    if not os.path.exists(ENGINE_PATH) or not os.path.exists(sidecar):
        log(f"FATAL: {ENGINE_PATH} (or its build record) is missing - run tools/bootstrap_models.sh")
        raise SystemExit(2)
    with open(sidecar) as handle:
        built = json.load(handle)
    gpu = torch.cuda.get_device_name(0)
    if built.get("tensorrt") != trt.__version__ or built.get("gpu") != gpu:
        log(f"FATAL: {ENGINE_PATH} was built for TensorRT {built.get('tensorrt')} on {built.get('gpu')}, "
            f"this is TensorRT {trt.__version__} on {gpu} - rebuild with tools/bootstrap_models.sh")
        raise SystemExit(2)
    return int(built["batch"])


MAX_BATCH = check_engine()


# ============================================================
# CAMERA CONFIGURATION (backend, with retry)
# ============================================================
def fetch_initial_config():
    delay = 5.0
    while True:
        try:
            configs = dashboard.fetch_cameras()
            if configs:
                return configs
            log("[CONFIG] the backend returned no AI-enabled cameras - retrying")
        except dashboard.DashboardError as exc:
            log(f"[CONFIG] cannot load the camera configuration: {exc} - retrying in {delay:.0f}s")
        time.sleep(delay)
        delay = min(delay * 2, 60.0)


log(f"\nFetching camera configuration from {dashboard.DASHBOARD_URL}/api/ai/cameras/ ...")
camera_configs = fetch_initial_config()
for config in camera_configs:
    names = ", ".join(CLASS_NAMES[c] for c in sorted(classes_for(config)))
    log(f"  {config.camera_id}  {config.name}  classroom={(config.classroom or {}).get('classroom_id')}")
    log(f"      source  : {config.source}")
    log(f"      detects : {names or 'nothing'}")

classroom_by_camera = {}


def rebuild_classroom_map():
    classroom_by_camera.clear()
    classroom_by_camera.update({c.camera_id: (c.classroom or {}).get("classroom_id") for c in camera_configs})


rebuild_classroom_map()
events.set_classroom_resolver(classroom_by_camera.get)


# ============================================================
# LOAD YOLO26-L (TensorRT FP16, built on this GB10)
# ============================================================
log(f"\nLoading {ENGINE_PATH} (dynamic batch 1..{MAX_BATCH}) ...")
model = YOLO(ENGINE_PATH, task="detect")
import numpy as _np  # noqa: E402
model.predict(source=[_np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=_np.uint8)], imgsz=IMG_SIZE, conf=CONF,
              device=DEVICE, verbose=False)
if tuple(dict(model.names).get(i) for i in range(len(object_logger.COCO_NAMES))) != object_logger.COCO_NAMES:
    log("WARNING: engine class names differ from object_logger.COCO_NAMES")
log("YOLO26-L engine loaded")


# ============================================================
# CAMERAS  (Dubai, verbatim - trimmed of Dubai-only state)
# ============================================================
camera_streams = {}
camera_classes = {}
camera_confidence = {}
camera_stats = {}
smooth_bboxes = {}


def predict_confidence():
    floor = CONF
    if camera_confidence:
        floor = min(min(camera_confidence.values()), CONF)
    if abandoned_detector.cameras_enabled():
        floor = min(floor, abandoned.OBJECT_CONFIDENCE)
    if phone_detector.any_enabled():
        floor = min(floor, phone_use.PHONE_CONFIDENCE)
    return floor


def new_stats_entry():
    return {"frames": 0, "last_frame": 0, "frame_width": None, "frame_height": None}


def start_camera(config):
    camera = CameraStream(camera_id=config.camera_id, source=config.source, queue_size=3, loop_video=False)
    camera.start()
    camera_streams[config.camera_id] = camera
    camera_classes[config.camera_id] = classes_for(config)
    camera_confidence[config.camera_id] = config.detection_confidence
    camera_stats[config.camera_id] = new_stats_entry()
    return camera


def stop_camera(camera_id):
    camera = camera_streams.pop(camera_id, None)
    if camera is not None:
        camera.stop()
    camera_classes.pop(camera_id, None)
    camera_confidence.pop(camera_id, None)
    camera_stats.pop(camera_id, None)
    track_manager.reset_camera(camera_id)
    if event_pipeline.evidence is not None:
        event_pipeline.evidence.forget_camera(camera_id)
    abandoned_detector.forget_camera(camera_id)
    for analytic in (phone_detector, sleeping_adapter, evacuation_monitor, floor_mapper):
        analytic.forget_camera(camera_id)
    for key in [k for k in smooth_bboxes if k[0] == camera_id]:
        del smooth_bboxes[key]


# ============================================================
# TRACKING, IDENTITY, ANALYTICS, EVENTS
# ============================================================
track_manager = tracking.TrackManager(
    person_classes=PERSON_CLASSES, vehicle_classes=set(), class_names=CLASS_NAMES,
    object_classes=CARRIED_CLASSES | PHONE_CLASSES | object_logger.LOGGABLE_CLASS_IDS)

reid_adapter_instance = reid_adapter.ReIDAdapter()
events.set_stable_id_resolver(reid_adapter_instance.stable_id_for)
face_id_adapter_instance = face_id_adapter.build_production_adapter()
fall_pose_adapter_instance = fall_pose_adapter.build_production_adapter()
fight_adapter_instance = fight_adapter.build_production_adapter()

event_pipeline = events.EventPipeline(dashboard_url=dashboard.DASHBOARD_URL, token=dashboard.DASHBOARD_TOKEN)
event_pipeline.update_camera_intervals({c.camera_id: {"intrusion": c.intrusion_interval,
                                                      "loitering": c.loitering_interval} for c in camera_configs})


def handle_camera_health_event(camera_id, event_type, timestamp, metadata,
                               action, frame_width, frame_height):
    """One confirmed camera-health state transition onto the EXISTING pipeline. (Dubai, verbatim)

    No bbox and no frame of its own: the condition is a property of the whole image, so the
    whole frame is its extent (validate_event requires four values). Scope separates a raise
    from its recovery in the debouncer. Dimensions come from the event's own camera.
    """
    stream = camera_streams.get(camera_id)
    width = frame_width or getattr(stream, "width", 0) or 1920
    height = frame_height or getattr(stream, "height", 0) or 1080

    result = event_pipeline.handle_camera_event(
        camera_id,
        event_type,
        metadata,
        frame_width=width,
        frame_height=height,
        observed_at=timestamp,
        bbox=[0, 0, width, height],
        scope=action,
    )
    if result == "sent":
        print(f"[CAM_HEALTH] {camera_id} {event_type} {action} -> event sent")
    else:
        print(f"[CAM_HEALTH] {camera_id} {event_type} {action} -> {result}")
    return result


camera_health_manager = camera_health.CameraHealthManager(shadow=False, event_sink=handle_camera_health_event)
camera_health_publisher = camera_health.HealthTelemetryPublisher(camera_health_manager, dashboard.DASHBOARD_URL,
                                                                 dashboard.DASHBOARD_TOKEN)
people_counter_instance = people_counter.PeopleCounter()
people_count_publisher = people_counter.PeopleCountPublisher(people_counter_instance, dashboard.DASHBOARD_URL,
                                                             dashboard.DASHBOARD_TOKEN)
occupancy_instance = occupancy.ClassroomOccupancy()
occupancy_publisher = occupancy.OccupancyPublisher(occupancy_instance, dashboard.DASHBOARD_URL,
                                                   dashboard.DASHBOARD_TOKEN)
object_logger_instance = object_logger.ObjectLogger()
zone_evaluator = zones.ZoneEvaluator(camera_zones={c.camera_id: c.zones for c in camera_configs},
                                     camera_features={c.camera_id: c.features for c in camera_configs})
abandoned_detector = abandoned.AbandonedObjectDetector(camera_features={c.camera_id: c.features
                                                                        for c in camera_configs})

# ---- Zayed classroom analytics ----
phone_detector = phone_use.PhoneUseDetector()
sleeping_adapter = sleeping.SleepingAdapter(log=log)
evacuation_monitor = evacuation.EvacuationMonitor(dashboard.DASHBOARD_URL, dashboard.DASHBOARD_TOKEN, log=log)
floor_mapper = floor_plan.FloorPlanMapper(log=log)
heatmap_publisher = floor_plan.HeatmapPublisher(floor_mapper, dashboard.DASHBOARD_URL, dashboard.DASHBOARD_TOKEN,
                                                log=log)
occupancy_instance.set_fusion(floor_mapper)
# Evidence stills for the Zayed event types too (a mutable set in evidence.py - no patch needed).
evidence.ALARM_EVIDENCE_TYPES.update({phone_use.EVENT_TYPE, sleeping.EVENT_TYPE, evacuation.EVENT_TYPE,
                                      "OVERCROWDING_DETECTED"})


# ---- per-analytic arming, from the backend alone (Dubai refresh_* shape) ----
def _armed(feature):
    return sorted(c.camera_id for c in camera_configs if c.features.get(feature))


def refresh_analytics():
    face_id_adapter_instance.set_enabled_cameras(_armed(face_id_adapter.FACE_ID_CAMERA_FEATURE))
    fall_pose_adapter_instance.set_enabled_cameras(_armed(fall_pose_adapter.FALL_CAMERA_FEATURE))
    fight_adapter_instance.set_enabled_cameras(_armed(fight_adapter.FIGHT_CAMERA_FEATURE))
    camera_health_manager.set_enabled_cameras({
        c.camera_id: {key: bool(c.features.get(key)) for key in camera_health.CAMERA_HEALTH_FEATURES}
        for c in camera_configs})
    people_counter_instance.set_enabled_cameras({c.camera_id: c.zones for c in camera_configs
                                                 if c.features.get(people_counter.PEOPLE_COUNTING_FEATURE)})
    object_logger_instance.set_enabled_cameras({c.camera_id: object_logger.class_ids_for(c.object_classes)
                                                for c in camera_configs
                                                if c.features.get(object_logger.OBJECT_DETECTION_FEATURE)})
    occupancy_instance.configure(camera_configs)
    phone_detector.set_enabled_cameras(_armed(phone_use.PHONE_FEATURE))
    sleeping_adapter.set_enabled_cameras(_armed(sleeping.SLEEP_FEATURE))
    evacuation_monitor.configure(camera_configs)
    floor_mapper.configure(camera_configs)


def apply_config(new_configs):
    """Diff a freshly fetched config against the running one. (Dubai shape, trimmed)"""
    global camera_configs
    new_by_id = {c.camera_id: c for c in new_configs}
    old_by_id = {c.camera_id: c for c in camera_configs}
    changes = 0
    for camera_id in sorted(set(old_by_id) - set(new_by_id)):
        log(f"[CONFIG] camera removed: {camera_id}")
        stop_camera(camera_id)
        changes += 1
    for camera_id in sorted(set(new_by_id) - set(old_by_id)):
        log(f"[CONFIG] camera added: {camera_id} -> {new_by_id[camera_id].source}")
        start_camera(new_by_id[camera_id])
        changes += 1
    for camera_id in sorted(set(new_by_id) & set(old_by_id)):
        new, old = new_by_id[camera_id], old_by_id[camera_id]
        if new.source != old.source:
            log(f"[CONFIG] {camera_id} source changed - restarting its stream")
            stop_camera(camera_id)
            start_camera(new)
            changes += 1
            continue
        if new.features != old.features or new.object_classes != old.object_classes:
            camera_classes[camera_id] = classes_for(new)
            if not camera_classes[camera_id]:
                track_manager.reset_camera(camera_id)
            changes += 1
        if new.zones != old.zones or new.classroom != old.classroom:
            changes += 1
    camera_configs = new_configs
    camera_confidence.clear()
    camera_confidence.update({c.camera_id: c.detection_confidence for c in camera_configs})
    event_pipeline.update_camera_intervals({c.camera_id: {"intrusion": c.intrusion_interval,
                                                          "loitering": c.loitering_interval} for c in camera_configs})
    zone_evaluator.update_config(camera_zones={c.camera_id: c.zones for c in camera_configs},
                                 camera_features={c.camera_id: c.features for c in camera_configs})
    abandoned_detector.update_config(camera_features={c.camera_id: c.features for c in camera_configs})
    rebuild_classroom_map()
    refresh_analytics()
    return changes


for config in camera_configs:
    log(f"Starting {config.camera_id} - {config.name}")
    start_camera(config)
refresh_analytics()


# ---- identity recovery: Face-ID / Re-ID that failed to START are rebuilt in the background ----
def _set_face_id(new):
    global face_id_adapter_instance
    face_id_adapter_instance = new


def _set_reid(new):
    global reid_adapter_instance
    reid_adapter_instance = new
    events.set_stable_id_resolver(new.stable_id_for)


identity = identity_supervisor.IdentitySupervisor(log=log)
identity.watch("Face-ID", lambda: face_id_adapter_instance, _set_face_id, face_id_adapter.build_production_adapter,
               failed=lambda a: (identity_supervisor.face_failed(a)
                                 and bool(_armed(face_id_adapter.FACE_ID_CAMERA_FEATURE))),
               arm=lambda a: a.set_enabled_cameras(_armed(face_id_adapter.FACE_ID_CAMERA_FEATURE)),
               ready=identity_supervisor.face_ready)
identity.watch("Re-ID", lambda: reid_adapter_instance, _set_reid, reid_adapter.ReIDAdapter,
               failed=identity_supervisor.reid_failed)
identity.start()
camera_health_manager.start()
camera_health_publisher.start()
people_count_publisher.start()
occupancy_publisher.start()
evacuation_monitor.start()
heatmap_publisher.start()
# Internal, token-authenticated face API for the dashboard (container network only, GPU only).
# Independent of the CCTV face path: if its model cannot load, it answers 503 and inference runs on.
internal_face_api = internal_api.InternalFaceAPI(log=log)
internal_face_api.start()
config_watcher = dashboard.ConfigWatcher()
config_watcher.start()

_last_tick = time.monotonic()


def _watchdog():
    while True:
        time.sleep(10)
        stalled = time.monotonic() - _last_tick
        if stalled > WATCHDOG_SECONDS:
            log(f"[WATCHDOG] FATAL: the main loop has not ticked for {stalled:.0f}s - exiting so the "
                f"restart policy starts a fresh process")
            os._exit(4)


threading.Thread(target=_watchdog, name="watchdog", daemon=True).start()


# ============================================================
# FRAME SELECTION  (Dubai, verbatim)
# ============================================================
_last_inferred_at = {}
_bad_frame_logged = {}


def _frames_to_infer(camera):
    """Frames spaced by TARGET_INFERENCE_FPS on their CAPTURE times, oldest first. (Dubai, verbatim)"""
    packets = camera.drain_frames(MAX_FRAMES_PER_CAMERA)
    if not packets:
        return []
    spacing = 1.0 / TARGET_INFERENCE_FPS
    last = _last_inferred_at.get(camera.camera_id)
    keep = []
    for packet in packets:
        stamp = packet["timestamp"]
        if last is not None and 0.0 <= (stamp - last) < spacing:
            continue
        keep.append(packet)
        last = stamp
    if not keep:
        keep = packets[-1:]
    _last_inferred_at[camera.camera_id] = keep[-1]["timestamp"]
    return keep


def _usable_frame(frame, camera_id):
    """False for a frame no detector can be given, with a throttled warning. (Dubai, verbatim)"""
    if frame is None:
        shape = "none"
    elif getattr(frame, "ndim", 0) != 3 or frame.shape[0] < 2 or frame.shape[1] < 2 or frame.shape[2] != 3:
        shape = "x".join(str(n) for n in frame.shape)
    else:
        return True
    now = time.time()
    if now - _bad_frame_logged.get(camera_id, 0.0) >= 60.0:
        _bad_frame_logged[camera_id] = now
        print(f"[{camera_id}] skipping unusable frame ({shape}) - not inferable")
    return False


# ============================================================
# EVENT HANDLERS  (Dubai, verbatim)
# ============================================================
def handle_fall_pose_events(falls, observed_at):
    for fall in falls:
        result = event_pipeline.handle_camera_event(
            fall.camera_id, fall_pose_adapter.EVENT_TYPE, fall.metadata(),
            frame_width=fall.frame_width, frame_height=fall.frame_height, observed_at=observed_at,
            bbox=fall.bbox, frame=getattr(fall, "frame", None), scope=fall.scope())
        fall.frame = None
        if result == "sent":
            m = fall.metadata()
            log(f"[FALL] {fall.camera_id} track={fall.track_id} torso={m['torso_angle']}deg "
                f"for {m['pose_duration']}s (pose)")


def handle_fight_events(fights, observed_at):
    for fight in fights:
        result = event_pipeline.handle_camera_event(
            fight.camera_id, fight_adapter.EVENT_TYPE, fight.metadata(),
            frame_width=fight.frame_width, frame_height=fight.frame_height, observed_at=observed_at,
            bbox=fight.bbox, frame=getattr(fight, "frame", None), scope=fight.scope())
        fight.frame = None
        if result == "sent":
            log(f"[FIGHT] {fight.camera_id} {fight.track_a}~{fight.track_b} "
                f"strikes={fight.strikes_a}/{fight.strikes_b}")


def handle_occupancy_decisions(now):
    for event_type, classroom_id, metadata in occupancy_instance.decisions(now):
        cameras = occupancy_instance.classroom_cameras(classroom_id)
        if not cameras:
            continue
        camera_id = cameras[0]
        stream = camera_streams.get(camera_id)
        width = getattr(stream, "width", 0) or 2592
        height = getattr(stream, "height", 0) or 1944
        result = event_pipeline.handle_camera_event(
            camera_id, event_type, dict(metadata, classroom_id=classroom_id), frame_width=width,
            frame_height=height, observed_at=now, bbox=[0, 0, width, height], scope=classroom_id)
        if result == "sent":
            log(f"[OCCUPANCY] {classroom_id} {event_type} occupancy={metadata['occupancy']} "
                f"capacity={metadata['capacity']}")


def emit_finding(event_type, finding, frame=None, label=""):
    """One per-person / per-camera Zayed finding onto the existing pipeline, with its floor
    location when the camera is calibrated and its evidence still when a frame is available."""
    metadata = finding.metadata()
    location = floor_mapper.location_for(finding.camera_id, finding.bbox)
    if location:
        metadata["location"] = location
    stream = camera_streams.get(finding.camera_id)
    width = getattr(finding, "frame_width", None) or getattr(stream, "width", 0) or 2592
    height = getattr(finding, "frame_height", None) or getattr(stream, "height", 0) or 1944
    result = event_pipeline.handle_camera_event(
        finding.camera_id, event_type, metadata, frame_width=width, frame_height=height,
        observed_at=finding.observed_at, bbox=finding.bbox,
        frame=frame if frame is not None else getattr(finding, "frame", None), scope=finding.scope())
    if hasattr(finding, "frame"):
        finding.frame = None
    if result == "sent":
        log(f"[{label or event_type}] {finding.camera_id} {finding.scope()} -> event sent")
    return result


# ============================================================
# PER-CAMERA PROCESSING  (Dubai per-result body, Zayed analytics only)
# ============================================================
latency_ms = defaultdict(lambda: deque(maxlen=600))
camera_errors = defaultdict(int)
_error_logged = {}


def _throttled_error(key, message):
    now = time.monotonic()
    if now - _error_logged.get(key, 0.0) >= 30.0:
        _error_logged[key] = now
        log(message)


def process_result(result, metadata, source_frame):
    camera_id = metadata["camera_id"]
    frame_width, frame_height = metadata["frame_width"], metadata["frame_height"]
    observed_at = metadata["timestamp"]
    detections = []
    phones = []
    if result.boxes is not None and len(result.boxes):
        boxes = result.boxes
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()
        xyxy = boxes.xyxy.detach().cpu().numpy()
        allowed = camera_classes.get(camera_id, set())
        min_confidence = camera_confidence.get(camera_id, CONF)
        abandon_armed = abandoned_detector.enabled_for(camera_id)
        phone_armed = phone_detector.enabled_for(camera_id)
        for class_id, confidence, box in zip(class_ids, confidences, xyxy):
            if class_id not in allowed:
                continue
            if phone_armed and class_id == phone_use.PHONE_CLASS_ID and confidence >= phone_use.PHONE_CONFIDENCE:
                # Phones go to the phone analytic at their own floor and stay out of the tracker.
                phones.append((float(confidence), [float(v) for v in box]))
                continue
            if confidence < min_confidence and not (
                    abandon_armed and int(class_id) in abandoned.CANDIDATE_CLASS_IDS
                    and confidence >= abandoned.OBJECT_CONFIDENCE):
                continue
            detections.append((int(class_id), float(confidence),
                               (int(box[0]), int(box[1]), int(box[2]), int(box[3]))))

    tracked_objects = track_manager.update(camera_id, detections, now=observed_at)

    if SMOOTH_ALPHA < 1.0:                              # (Dubai, verbatim) EMA box smoothing
        for tracked in tracked_objects:
            key = (camera_id, tracked.track_id)
            rx1, ry1, rx2, ry2 = tracked.bbox
            prev = smooth_bboxes.get(key)
            if prev is None:
                s = [float(rx1), float(ry1), float(rx2), float(ry2)]
            else:
                a = SMOOTH_ALPHA
                s = [a * rx1 + (1 - a) * prev[0], a * ry1 + (1 - a) * prev[1],
                     a * rx2 + (1 - a) * prev[2], a * ry2 + (1 - a) * prev[3]]
            smooth_bboxes[key] = s
            tracked.bbox = [int(s[0]), int(s[1]), int(s[2]), int(s[3])]
        active = {(camera_id, t.track_id) for t in tracked_objects}
        for gone in [k for k in smooth_bboxes if k[0] == camera_id and k not in active]:
            del smooth_bboxes[gone]

    reid_adapter_instance.process_tracks(camera_id, tracked_objects, source_frame, observed_at)
    face_id_adapter_instance.submit(camera_id, tracked_objects, source_frame, observed_at)
    fight_adapter_instance.submit(camera_id, source_frame, tracked_objects, observed_at,
                                  frame_id=metadata["frame_id"])
    fall_pose_adapter_instance.submit(camera_id, source_frame, tracked_objects, observed_at,
                                      frame_id=metadata["frame_id"])

    object_tracks = [t for t in tracked_objects if t.group == tracking.OBJECT_GROUP]
    motion_tracks = [t for t in tracked_objects if t.group != tracking.OBJECT_GROUP]
    for tracked in motion_tracks:
        for zone_event_type, zone_metadata in zone_evaluator.evaluate(tracked, frame_width, frame_height):
            event_pipeline.handle_zone_event(tracked, zone_event_type, zone_metadata, frame_width=frame_width,
                                             frame_height=frame_height, observed_at=observed_at,
                                             frame=source_frame)

    person_boxes = [t.bbox for t in motion_tracks if t.group == tracking.PERSON_GROUP]
    carried_tracks = [t for t in object_tracks if t.class_id in abandoned.CANDIDATE_CLASS_IDS]
    for obj_track, abandon_meta in abandoned_detector.update(camera_id, carried_tracks, person_boxes,
                                                             frame_width, frame_height):
        if event_pipeline.handle_abandoned_event(obj_track, abandon_meta, frame_width=frame_width,
                                                 frame_height=frame_height, observed_at=observed_at,
                                                 frame=source_frame) == "sent":
            log(f"[UNATTENDED] {camera_id} {obj_track.class_name} track={obj_track.track_id} "
                f"dwell={abandon_meta['dwell_seconds']}s")

    for obj_track, object_meta, crop_box in object_logger_instance.observe(camera_id, object_tracks, frame_width,
                                                                          frame_height, observed_at):
        event_pipeline.handle_object_event(obj_track, object_meta, frame_width=frame_width,
                                           frame_height=frame_height, observed_at=observed_at,
                                           frame=source_frame, crop_box=crop_box)

    for crowd_type, crowd_metadata, crowd_bbox, crowd_people in zone_evaluator.evaluate_crowd(
            camera_id, tracked_objects, frame_width, frame_height):
        event_pipeline.handle_camera_event(camera_id, crowd_type, crowd_metadata, frame_width=frame_width,
                                           frame_height=frame_height, observed_at=observed_at, bbox=crowd_bbox,
                                           frame=source_frame, contributors=crowd_people,
                                           scope=crowd_metadata.get("zone_id"))

    people_counter_instance.observe(camera_id, tracked_objects, frame_width, frame_height)
    occupancy_instance.observe(camera_id, tracked_objects, observed_at)

    persons = [t for t in tracked_objects if t.group == tracking.PERSON_GROUP]
    floor_mapper.observe(camera_id, persons, observed_at)
    for finding in phone_detector.update(camera_id, persons, phones, observed_at):
        emit_finding(phone_use.EVENT_TYPE, finding, frame=source_frame, label="PHONE")
    sleeping_adapter.submit(camera_id, source_frame, persons, observed_at, metadata["frame_id"],
                            frame_width, frame_height)
    for finding in evacuation_monitor.observe(camera_id, persons, source_frame, frame_width, frame_height,
                                              observed_at):
        emit_finding(evacuation.EVENT_TYPE, finding, label="STRANDED")

    stats = camera_stats.get(camera_id)
    if stats is not None:
        stats["frames"] += 1
        stats["last_frame"] = metadata["frame_id"]
        stats["frame_width"], stats["frame_height"] = frame_width, frame_height
    latency_ms[camera_id].append((time.time() - observed_at) * 1000.0)


# ============================================================
# STATUS / METRICS
# ============================================================
inference_ms = deque(maxlen=600)


def _pct(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(p / 100 * (len(ordered) - 1)))], 1)


def write_status(elapsed, frames_in_window, predict_failures_total):
    per_camera = {}
    for camera_id, camera in camera_streams.items():
        stats = camera_stats.get(camera_id, {})
        per_camera[camera_id] = {
            "inferred_frames": stats.get("frames", 0),
            "frames_read": getattr(camera, "frames_read", None),
            "frames_dropped": getattr(camera, "frames_dropped", None),
            "resolution": f"{getattr(camera, 'width', None)}x{getattr(camera, 'height', None)}",
            "decoder": getattr(camera, "decoder", None),
            "e2e_latency_ms_p50": _pct(latency_ms[camera_id], 50),
            "e2e_latency_ms_p95": _pct(latency_ms[camera_id], 95),
            "errors": camera_errors.get(camera_id, 0),
        }
    snapshot = {
        "at": time.time(), "window_seconds": round(elapsed, 1),
        "inferred_fps_total": round(frames_in_window / elapsed, 2) if elapsed else None,
        "inference_ms_p50": _pct(inference_ms, 50), "inference_ms_p95": _pct(inference_ms, 95),
        "predict_failures_total": predict_failures_total, "cameras": per_camera,
        "events": event_pipeline.stats() | {"outbox": event_pipeline.sender.stats() if event_pipeline.sender else None},
        "face_id": face_id_adapter_instance.stats(), "fall_pose": fall_pose_adapter_instance.stats(),
        "fight": fight_adapter_instance.stats(), "camera_health": camera_health_manager.stats(),
        "identity": identity.stats(), "internal_face_api": internal_face_api.stats(),
        "phone_use": phone_detector.stats(), "sleeping": sleeping_adapter.stats(),
        "evacuation": evacuation_monitor.stats(), "floor_plan": floor_mapper.stats(),
        "torch_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
    }
    try:
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        tmp = METRICS_PATH + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(snapshot, handle, default=str)
        os.replace(tmp, METRICS_PATH)
    except OSError:
        pass
    cams = " ".join(f"{c}:{v['inferred_frames']}f/{v['e2e_latency_ms_p50']}ms" for c, v in per_camera.items())
    log(f"[STATUS] fps={snapshot['inferred_fps_total']} infer_p50={snapshot['inference_ms_p50']}ms {cams} "
        f"events_sent={snapshot['events']['sent']} pending={snapshot['events']['pending']} "
        f"predict_failures={predict_failures_total}")
    log(camera_health_manager.status_line())
    for camera_id in camera_stats:
        camera_stats[camera_id]["frames"] = 0


# ============================================================
# MAIN LOOP
# ============================================================
_shutting_down = False


def _handle_sigterm(signum, frame):
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, _handle_sigterm)
inference_interval = 1.0 / TARGET_INFERENCE_FPS
last_inference_time = 0.0
stats_start = time.perf_counter()
frames_in_window = 0
consecutive_predict_failures = 0
predict_failures_total = 0
log("\nInference running. Configuration changes in the backend are picked up live.")

try:
    while True:
        _last_tick = time.monotonic()
        try:
            now = time.perf_counter()
            if now - last_inference_time < inference_interval:
                time.sleep(0.002)
                continue
            last_inference_time = now

            pending_config = config_watcher.take()
            if pending_config is not None:
                applied = apply_config(pending_config)
                if applied:
                    log(f"[CONFIG] refresh #{config_watcher.refreshes}: applied {applied} change(s)")

            batch_frames, batch_metadata = [], []
            camera_health_manager.note_streams(camera_streams)
            for camera in list(camera_streams.values()):
                if not camera_classes.get(camera.camera_id):
                    continue
                try:
                    for packet in _frames_to_infer(camera):
                        if not _usable_frame(packet["frame"], packet["camera_id"]):
                            continue
                        height, width = packet["frame"].shape[:2]
                        camera_health_manager.observe(packet["camera_id"], packet["frame"], packet["timestamp"])
                        batch_frames.append(packet["frame"])
                        batch_metadata.append({"camera_id": packet["camera_id"], "frame_id": packet["frame_id"],
                                               "timestamp": packet["timestamp"], "frame_width": width,
                                               "frame_height": height})
                except Exception as exc:                          # noqa: BLE001 - one camera, not all
                    camera_errors[camera.camera_id] += 1
                    _throttled_error(("collect", camera.camera_id),
                                     f"[{camera.camera_id}] frame collection error: {type(exc).__name__}: {exc}")

            if batch_frames:
                results = []
                started = time.perf_counter()
                try:
                    for chunk_start in range(0, len(batch_frames), MAX_BATCH):
                        results.extend(model.predict(source=batch_frames[chunk_start:chunk_start + MAX_BATCH],
                                                     imgsz=IMG_SIZE, conf=predict_confidence(), device=DEVICE,
                                                     verbose=False))
                    consecutive_predict_failures = 0
                except Exception as exc:                          # noqa: BLE001
                    consecutive_predict_failures += 1
                    predict_failures_total += 1
                    _throttled_error("predict", f"[INFER] predict failed ({consecutive_predict_failures} in a row): "
                                                f"{type(exc).__name__}: {exc}")
                    if consecutive_predict_failures >= MAX_PREDICT_FAILURES:
                        log(f"[INFER] FATAL: {consecutive_predict_failures} consecutive predict failures - exiting "
                            f"so the restart policy starts a fresh process")
                        raise SystemExit(3)
                    results = []
                if results:
                    inference_ms.append((time.perf_counter() - started) * 1000.0)
                    frames_in_window += len(results)
                for result, metadata, source_frame in zip(results, batch_metadata, batch_frames):
                    try:
                        process_result(result, metadata, source_frame)
                    except Exception as exc:                      # noqa: BLE001 - one camera, not all
                        camera_errors[metadata["camera_id"]] += 1
                        _throttled_error(("process", metadata["camera_id"]),
                                         f"[{metadata['camera_id']}] analytics error: {type(exc).__name__}: {exc}")

            for name, action in (("reid", reid_adapter_instance.flush),
                                 ("fight", lambda: handle_fight_events(fight_adapter_instance.flush() or [], time.time())),
                                 ("fall", lambda: handle_fall_pose_events(fall_pose_adapter_instance.flush() or [],
                                                                          time.time())),
                                 ("occupancy", lambda: handle_occupancy_decisions(time.time())),
                                 ("sleeping", lambda: [emit_finding(sleeping.EVENT_TYPE, f, label="SLEEPING")
                                                       for f in sleeping_adapter.flush()]),
                                 ("floor", lambda: floor_mapper.tick(time.time()))):
                try:
                    action()
                except Exception as exc:                          # noqa: BLE001
                    _throttled_error(("flush", name), f"[{name.upper()}] flush error: {type(exc).__name__}: {exc}")

            elapsed = time.perf_counter() - stats_start
            if elapsed >= STATUS_SECONDS:
                try:
                    write_status(elapsed, frames_in_window, predict_failures_total)
                    event_pipeline.prune()
                    if event_pipeline.evidence is not None:
                        event_pipeline.evidence.prune()
                    phone_detector.prune(time.time())
                except Exception as exc:                          # noqa: BLE001
                    _throttled_error("status", f"[STATUS] error: {type(exc).__name__}: {exc}")
                frames_in_window = 0
                stats_start = time.perf_counter()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:                                  # noqa: BLE001 - the loop itself survives
            _throttled_error("loop", f"[LOOP] unexpected error, continuing: {type(exc).__name__}: {exc}")
            time.sleep(0.1)
except KeyboardInterrupt:
    log("\nStopping...")
finally:
    log("Flushing pending events (undelivered events stay in the durable outbox)...")
    for step in (identity.stop, internal_face_api.stop, evacuation_monitor.stop, heatmap_publisher.stop,
                 event_pipeline.stop, config_watcher.stop):
        try:
            step()
        except Exception as exc:                                  # noqa: BLE001
            log(f"shutdown step failed: {exc}")
    for camera in list(camera_streams.values()):
        camera.stop()
    for closer in (reid_adapter_instance.close, face_id_adapter_instance.close, fight_adapter_instance.close,
                   fall_pose_adapter_instance.close, sleeping_adapter.close, camera_health_publisher.stop,
                   people_count_publisher.stop,
                   occupancy_publisher.stop, pose_l960_runtime.shared_l960.close):
        try:
            closer()
        except Exception as exc:                                  # noqa: BLE001
            log(f"close failed: {exc}")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    log("Inference stopped")
