# Zayed University CCTV AI: Phases 2–11 status

As of 2026-09-22, 20:22 (UTC+4). Baselines: Dubai inference `5b192cef` and Dubai dashboard `6c451ed`, both left untouched.
Cameras: exactly three, `camera_01`, `camera_02` and `camera_03`, all in classroom C101.

## Running now

```
NVR ch 101/201/301 ──RTSP──▶ zayed-relay-camera-0N (video-only remux) ──▶ zayed-mediamtx (127.0.0.1:8554, record: false)
      ──▶ zayed-inference (GPU: NVDEC → YOLO26-L TRT → JeztSort → analytics) ──REST──▶ zayed-dashboard (127.0.0.1:8000)
      ──▶ PostgreSQL (university) · Qdrant · SeaweedFS (evidence stills only)        zayed-presence-monitor (absence sweeps)
```

## 1. Camera health

- `camera_health/`: `manager.py`, `detectors.py`, `metrics.py`, `state.py`. It exports `CameraHealthManager`, `HealthTelemetryPublisher`, `CAMERA_HEALTH_FEATURES` and `PUBLISH_INTERVAL_SECONDS`.
- It runs on the CPU, on frames the inference loop has already decoded.
- Signal loss, obstruction, defocus and scene change are all detected. Tamper and obstruction form one `camera_tamper` incident. Signal loss and defocus are diagnostic only, as in Dubai production. Telemetry reports `offline` with `critical` severity.
- **Acceptance: 163/163 pass** (`tests/camera_health/run_acceptance.sh`, exit 0):
  - `test_camera_health`: 60
  - `test_camera_health_recovery`: 8
  - `test_camera_health_umbrella`: 10
  - `test_camera_tamper_incident`: 26
  - `test_camera_tamper_lighting`: 14, using the real CAM-R25 frames, extracted from Dubai history into a temporary directory and never stored in Zayed
  - `test_camera_health_telemetry`: 40
  - `test_camera_health_event`: 5, running the real handler out of `zayed_inference.py`
- **Live:** telemetry for all 3 cameras reaches `CameraHealth` every 15 s: healthy, 30.0 fps, 100% availability.
  - A real camera_02 stream loss was detected at 5.1 s and recovered.
  - Tamper and obstruction have **not** been exercised live, because that needs someone to physically cover or move a camera.

## 2. MediaMTX

- MediaMTX 1.21.0 has three paths fed by one relay per camera.
- Recording, playback, HLS, WebRTC, SRT, RTMP and MoQ are all disabled.
- Ports are bound to `127.0.0.1` only.
- Only the relay account may publish. Its password is random per render and stored as a SHA-256 hash.
- No credential appears in process arguments.
- The NVR's AAC track has an SDP without fmtp, which MediaMTX rejects, hence the video-only relay. The NVR audio setting is unchanged.
- Measured on each camera: H.264 2592×1944 at 30 fps, GOP 8.5 s, NVDEC decoding at 30.0 fps.

## 3. Model bootstrap

`tools/bootstrap_models.sh` checks, downloads, SHA-256-pins, verifies and GPU-smoke-tests these models, and refuses to run without CUDA:

- `yolo26l`, `yolo26l-pose`, `yolo26m-pose` (Ultralytics assets, AGPL-3.0)
- InsightFace `buffalo_l` (non-commercial licence)
- OSNet `osnet_ain_x1_0` (torchreid, MIT)

Nothing was copied from Spark or Dubai. No ANPR, weapon, PPE, fire or vehicle models were downloaded.

## 4. TensorRT engines

- Engines were built locally on the GB10 with TensorRT 10.14.1.48 (sm121, driver 580.178.04), FP16, dynamic batch.
  - `yolo26l.engine`: 640 px, batch 1–6
  - `yolo26l-pose.engine`: 960 px, batch 1–4
  - `yolo26m-pose.engine`: 640 px, batch 1–3
- Location: `/mnt/cctv/cache/tensorrt/zayed/current` → `trt10.14.1.48_NVIDIA-GB10_sm121_drv580.178.04/`.
- `zayed_inference.py` refuses an engine whose build record names a different TensorRT version or GPU.
- The pose runtimes' `.pt` fallback variables point at the engines, so a missing engine is a hard failure, never a silent `.pt` run.
- Face ID refuses the CPU provider.

## 5. Dashboard backend changes (backend only, no pages)

The Zayed backend (`zayed_dashboard/`) is a copy of the Dubai dashboard with the following changes:

- A new `zayed` app: classrooms, placements, profiles, assignments, presence, locations, occupancy, evacuation and heatmaps.
- Classroom-level presence. The Dubai state machine now runs per classroom (PRESENT, PENDING, ABSENT). Three cameras resolve to one presence, and absence sweeps run in `zayed-presence-monitor`.
- Idempotent event ingest keyed on `event_id`.
- Zayed event vocabulary and alarm rules.
- `person_id` is resolved from a track's recent face sighting.
- Stranded-person events are counted on their evacuation session.
- Recording can't be enabled: `pre_save` forces it off, the API refuses it, and video clips are off.
- `cameras.cache.json` dependency removed: the camera config comes from the database.
- `seed_zayed`:
  - arms the Zayed features on the three cameras, with generic object logging OFF (its Dubai class list contained knife, scissors and baseball bat);
  - sets the Re-ID group C101 to the validated `cross_camera_only` mode;
  - creates the service token, written to a 0600 file and never printed.
- Enrollment stays in the dashboard. There is no enrollment CLI.

## 6. API endpoints

Modified (the eight integration points):

| Endpoint | What changed |
|---|---|
| `GET /api/ai/cameras/` | Classroom and placement blocks added; `stream_url` comes from the placement. |
| `POST /api/events/` | Idempotent ingest; Zayed fields, severity, classroom and `person_id`; stranded hook. |
| `POST /api/ai/people-counts/` | Takes an `occupancy` list; the backend applies capacity, percentage and the overcrowding flag. |
| `POST /api/ai/face-observations/` | Stores classroom and identity fields. |
| `POST /api/ai/known-person-sightings/` | Drives classroom presence and `PersonLocation`. |
| `GET /api/ai/reid-groups/` | Serves group C101 with camera_01..03. |
| `POST /api/cameras/health-telemetry/` | Stores the full state. |

Added:

- `GET /api/zayed/classrooms/`
- `GET /api/zayed/classrooms/<id>/presence/`
- `GET /api/zayed/people/search/?q=`
- `POST /api/zayed/evacuation/start/`
- `POST /api/zayed/evacuation/<uid>/end/`
- `GET /api/ai/evacuation/active/`
- `POST /api/ai/heatmap/`
- `GET /api/zayed/classrooms/<id>/heatmap/?minutes=`

Backend tests: `manage.py test zayed` passes 16/16. The inherited `events_log` suite has 31 failures and 20 errors, every one in a known category:
- video clips and recording are deliberately off;
- the event vocabulary and alarm rules grew by the Zayed additions;
- the Dubai-only `anpr_consensus` module isn't in this tree.

## 7. PostgreSQL

New tables (migrations `zayed` 0001 and 0002):

- `Classroom`
- `CameraPlacement`
- `PersonProfile`
- `ClassroomAssignment`
- `ClassroomPresence`
- `PersonLocation`
- `ClassroomOccupancySample`
- `EvacuationSession`
- `ClassroomHeatmapSample`

Modified models:

| Model | Change | Migration |
|---|---|---|
| `Camera` | 4 new feature keys; recording forced off | `cameras` 0013 |
| `CameraHealth` | New fields: signal, severity, telemetry, conditions, incident, scene distance, metrics, published_at | `cameras` 0013 |
| `Event` | New fields: classroom_id, person_id, location, severity, status, evidence_reference; 15 Zayed event types | `events_log` 0024 |
| `FaceObservation` | New fields: classroom and identity | `investigation` 0012 |
| `KnownPersonSighting` | New field: classroom_id | `investigation` 0012 |

The temporary `CREATEDB` privilege (needed for test databases) has been revoked.

## 8. Qdrant

| Collection | Created by | State |
|---|---|---|
| `zayed_face_observations` | Inference (512-d, cosine) | Present, 0 points |
| `zayed_person_reid` | Inference (512-d, cosine) | Present, 0 points |
| `zayed_face_known_persons` | Dashboard, on the first enrollment | Not created yet: nobody is enrolled |

## 9. SeaweedFS evidence

- Stills only, no video: `zayed-evidence/YYYY/MM/DD/<camera_id>/<track-or-scope>/<uid>/{crop|face}.jpg`, with a 30-day TTL, via `http://seaweed-filer:8888`.
- The still is uploaded **before** its event is sent, and the event carries `evidence_reference`. If storage is down, the event is still sent, just without the image.
- Evidence is produced for these event types:
  - fall, fight, unattended object and camera tamper;
  - phone use, sleeping, stranded person and overcrowding.
- No live evidence exists yet, because no qualifying event has occurred.

## 10. Migrated from Dubai

32 files: 27 modules plus `reid_poc/` (5). 26 are byte-identical to `5b192cef`. Six carry marked Zayed patches (`docs/ZAYED_PATCHES.md`): `events.py`, `dashboard.py`, `evidence.py`, `known_person.py`, `face_id_manager.py` and `face_id_adapter.py`.

The `face_id_adapter.py` patch fixes a real bug found live. It upper-cased the armed camera IDs but compared the raw ID, so face ID ignored every `camera_0N` camera.

## 11. Zayed-specific components

- `zayed_inference.py`: the main loop, with per-camera, per-predict and per-flush containment, a hang watchdog, engine checks and `runtime/metrics.json`.
- `outbox.py`: durable SQLite event outbox with whole-outbox backoff and dead letters.
- `occupancy.py`: `max_camera`, switching to `floor_fusion` once the cameras are calibrated.
- `identity_supervisor.py`: rebuilds face ID and Re-ID that failed at start-up.
- `phone_use.py`, `sleeping.py`, `evacuation.py`, `floor_plan.py`.
- `camera_health/`, `mediamtx/`, `docker/`, `tools/bootstrap_models.*`.
- Tests:
  - `tests/unit`: 34, passing
  - `tests/resilience`: 13 loopback tests plus a live identity-recovery test, all passing

## 12. Real-camera validation, measured on the three C101 cameras

| Area | Status |
|---|---|
| Streams → MediaMTX → NVDEC | ✅ Live: 3 × 2592×1944 at 28–30 fps decoded, `h264_cuvid` |
| YOLO26-L TensorRT inference | ✅ Live: 10.0 fps per camera (29.7 fps total). Batch p50 15.6 ms, p95 23.5 ms. End-to-end frame latency p50 40–45 ms, p95 67–84 ms. 0 predict failures |
| Event delivery | ✅ Live: `OCCUPANCY_UPDATED` stored in 22–283 ms, 0 invalid, 0 dead letters |
| Camera-health telemetry | ✅ Live on all 3 cameras. Signal loss detected and recovered live. Tamper and obstruction not exercised |
| Occupancy samples | ✅ Live: one every 10 s. Value 0, and capacity is not set |
| Per-camera isolation | ✅ Live: while camera_02 was down, camera_01 and camera_03 stayed at 10 fps |
| Person detection, tracking, face ID → Qdrant → sighting → presence, Re-ID | ⏳ **Not validated.** The room was empty (evening) and nobody is enrolled |
| Unattended object, fall, fight, sleeping, phone use, stranded person | ⏳ **Not validated.** They need people and staged scenarios |
| Heatmap and floor-fused occupancy | ⏳ Blocked on floor-plan calibration |

Other measurements:
- GPU: SM about 10–11%, NVDEC about 8–9%, 15 W.
- GPU memory: inference process 1.6 GiB, plus 3 × 336 MiB for the decoders.
- CPU: inference container about 3.4 of 20 cores. Each ffmpeg decoder uses about 0.8 of a core, mostly converting 30 fps to BGR while only 10 fps are analysed. Candidate optimisation: decimate to about 20 fps at the decoder.
- RAM: 2.8 GiB.
- Reconnect: 10.9 s from the relay coming back to decoding again, dominated by the 8.5 s GOP.
- All of the above was measured with the room empty, so the face, Re-ID and pose load is still to be measured with the class seated.
- The "6.9%" figure from Phase 1 is not used anywhere.

Resilience tests, none of which touched a live service:
- **Outbox:**
  - 100 events held through a 6 s outage with 12 attempts or fewer, then delivered exactly once;
  - a backlog survived a process restart;
  - duplicates were acknowledged;
  - an invalid event was dead-lettered without blocking the rest;
  - 5xx and 401/403 responses were retried.
- **Evidence:** with the filer down or frozen, the event was still sent within the timeout.
- **Identity recovery:** in an isolated network, the real face-ID and Re-ID modules recovered 28 s after Qdrant appeared, without a process restart.

Container resilience: `restart: unless-stopped`. The watchdog exits after 120 s without a loop tick. After 20 consecutive predict failures the process exits with code 3 so it restarts with a fresh CUDA context.

Not yet run, pending the owner's go-ahead because they disrupt live services:
- camera or relay outage and MediaMTX restart as planned tests;
- dashboard, Postgres, Qdrant and SeaweedFS outages against the running pipeline.

The one unplanned camera_02 relay outage (15:58:00–15:59:51Z, about 111 s, caused by an interrupted test command) is covered in the rows above.

## 13. Remaining blockers

1. **People in C101.** Every person-based use case needs real occupants. Unattended object, fall, fight, sleeping and phone use also need staged scenarios.
2. **Enrollment.** At least one consenting person must be enrolled through the dashboard (photo, ID, role, C101 assignment) before face ID → presence can be validated. This creates `zayed_face_known_persons`.
3. **Classroom capacity.** C101 capacity is not set, so there is no occupancy percentage and no overcrowding alarm.
4. **Floor-plan calibration.** C101 needs a floor plan (width, height, grid) and at least 4 image↔floor point pairs per camera before the heatmap, floor-fused occupancy and event floor locations can work.
5. **Owner go-ahead for disruptive outage tests** (see §12).
6. **New rules need live tuning.** The sleeping, phone-use and stranded-person thresholds are defaults. They must be tuned on real C101 footage.
7. **Licensing before production.** InsightFace `buffalo_l` is non-commercial. Ultralytics is AGPL-3.0.

## 14. Next commands

```bash
# health / metrics
docker ps --filter name=zayed- ; docker logs -f zayed-inference
python3 -m json.tool zayed_ai_inferencing/runtime/metrics.json

# test suites
zayed_ai_inferencing/tests/camera_health/run_acceptance.sh
zayed_ai_inferencing/tests/unit/run_unit.sh
zayed_ai_inferencing/tests/resilience/run_resilience.sh
zayed_ai_inferencing/tests/resilience/run_identity_recovery.sh     # GPU, isolated network

# capacity (replace 40 with the real number of seats)
docker exec zayed-dashboard python manage.py seed_zayed --capacity 40

# enrollment: dashboard -> Known People (photo + student/employee ID + role), then assign to C101
#   (admin: /admin/zayed/classroomassignment/), then walk into C101 and watch:
docker logs -f zayed-inference | grep -E "FACE-ID|KNOWN|PRESENCE"
auth() { printf 'Authorization: Token %s\n' "$(cat zayed_ai_inferencing/runtime/dashboard_token)"; }  # builtin: token never in argv
curl -s -H @<(auth) http://127.0.0.1:8000/api/zayed/classrooms/C101/presence/

# mock evacuation drill (creates a real EVACUATION_STARTED event and alarm - with the owner's OK)
curl -s -X POST -H "Content-Type: application/json" -H @<(auth) \
     -d '{"classroom_id":"C101","evacuation_seconds":120,"trigger_source":"mock"}' \
     http://127.0.0.1:8000/api/zayed/evacuation/start/

# floor-plan calibration: admin -> Classroom C101 floor_plan {"width_m":..,"height_m":..,"grid_m":0.5}
#   and each CameraPlacement floor_plan_homography {"image_points":[[x,y]x4+],"floor_points_m":[[x,y]x4+]}
```

