# Zayed University CCTV AI Inferencing — Migration Audit (Phase 1)

| | |
|---|---|
| **Date** | 2026-09-22 |
| **Baseline audited** | `/home/stack/Zayed_University/dubai_ai_inferncing` at `5b192cef` ("cleaned version without camera health", branch `cleaned`) |
| **Target host** | Dell Pro Max GB10 `promaxgb10-5fae` · NVIDIA GB10 · aarch64 · driver 580 / CUDA 13.0 · NGC PyTorch 25.11 |
| **Scope** | Phase 1 audit only. **Read-only**: no production file, container, database, stream, service or host setting was changed. The only file written is this document. |
| **Method** | Imports traced with an AST walk from the real entry points, including the directories production adds to `sys.path` at runtime. Module code and docstrings read. Host, containers and data services inspected with read-only commands. Dubai's own validation reports read from git history (`9a511c37`). Filenames were never used as evidence on their own. |

### How to read the statuses

| Status | Meaning in this document |
|---|---|
| **READY** | Proven in Dubai production. Reusable for Zayed with configuration only. |
| **ADAPTABLE** | Proven mechanism that needs work for Zayed: a code change (classroom semantics, a local config source instead of the Dubai dashboard, a new data sink), or re-validation on classroom cameras before it can be trusted. |
| **PROTOTYPE** | Experimental or superseded code. Reference only, never copied as production. |
| **MISSING DEPENDENCY** | Required, but not present anywhere this audit could reach. |
| **NOT REQUIRED** | Serves no Zayed requirement. Not to be copied. |

These statuses describe **migration readiness, not use-case completion. No Zayed use case is complete.** Nothing has run on a Zayed classroom camera yet, and GPU containers cannot start on this host today (§2.1).

Requirement IDs used throughout: **R1** student/faculty monitoring · **R2** fighting · **R3** fall · **R4** distraction (sleeping, phone) · **R5** faculty/invigilator presence · **R6** evacuation / stranded person · **R7** occupancy & overcrowding · **R8** backend data · **R9** unattended object · **R10** camera tampering · **R11** floor-plan heatmap · **P** platform (ingest, config, resilience).

---

## 1. Summary

The Dubai stack contains a proven core that Zayed can reuse: NVDEC camera ingest with automatic reconnect, batched TensorRT detection, stable per-camera tracking, InsightFace face identity with a Qdrant gallery, OSNet Re-ID, pose-based fall and fight detection, unattended-object detection, zones and people counting, ground-plane calibration, and an event/evidence pipeline that already uploads to the same SeaweedFS filer this host runs.

Of the **41 root production modules**: **9 READY, 22 ADAPTABLE, 10 NOT REQUIRED**. One required package, **`camera_health`, is a MISSING DEPENDENCY**. Outside the root, the `reid_poc/` library (4 files) is READY, six `distance_measuring/modules/` files are ADAPTABLE, and ANPR, weapon, door and fence code is NOT REQUIRED.

Seven findings shape everything after Phase 1:

1. **GPU containers cannot start on this host right now.** NVIDIA driver packages were upgraded from 580.173.02 to 580.178.04 on 2026-09-21 at 19:42 without a reboot. The loaded kernel module is still 580.173.02, `nvidia-smi` fails with a driver/library mismatch, `nvidia-persistenced` has been stopped since the upgrade, and the CDI spec points at libraries that no longer exist. A test `docker run --gpus all` fails at container creation. The validated GB10 results were measured on 580.173.02. **Every GPU validation is blocked until the owner plans a reboot** (§2.1).
2. **`camera_health` cannot be recovered from anything on this host.** It is absent from all 10 commits, all refs, all 27,055 git objects and the whole filesystem. Its interface and behaviour are fully reconstructed from its callers, 7 test files, the dashboard and a design document. The proven copy should be on `spark-9a32`, which is **offline** (§3).
3. **No model weights exist on this host.** Every Zayed model is a public pretrained model; none of Dubai's custom-trained weights are needed. All must be downloaded, and every TensorRT engine must be built on this machine (§7).
4. **Dubai inference is wired to the MEERANA dashboard at 8 points**: camera config, events, people counts, face observations, known-person sightings, Re-ID groups, camera-health telemetry, and the absence sweep. **Enrollment and the presence/absence state machine live inside the Dubai dashboard, not the inference code.** Replacing these with Zayed backend storage is the largest piece of Phase 3 (§4.3).
5. **The validated GB10 performance does not cover the Dubai model stack.** The four-camera result (GPU 6.9%, NVDEC 10.2%, decode→detection p50 21 ms) was measured with a YOLOX-S detector and a ResNet-50 stand-in, on a zero-copy PyNvVideoCodec ingest path. Dubai uses YOLO26-L, two pose engines, InsightFace and OSNet, fed by FFmpeg subprocesses that copy full-resolution BGR frames to CPU memory. Phase 6 has to measure the real stack (§2.2, §5.2).
6. **Resilience gaps for Phase 7.** Any exception the main loop doesn't expect (for example a GPU error in `model.predict`) exits the whole inference process and stops every camera; one such outage is documented in the code (2026-09-16). The supervisor never restarts it. Events are dropped, not retried, when their sink is down (§5.6).
7. **Secrets and personal data were found in six places**, three of them pushed to GitHub. None were copied, and no values are reproduced in this document (§2.7).

---

## 2. Zayed host environment (inspected read-only)

### 2.1 GPU and driver — BLOCKER

| Item | At validation (host report, 2026-09-15) | Now (2026-09-22) |
|---|---|---|
| Kernel module | 580.173.02 | 580.173.02 (still loaded; last boot 2026-09-14 18:14) |
| Userspace driver packages | 580.173.02 | **580.178.04** (`nvidia-driver-580-open`, `libnvidia-compute-580`, `libnvidia-decode-580`) |
| `nvidia-smi` | works | **fails: "Failed to initialize NVML: Driver/library version mismatch"** |
| `nvidia-persistenced` | active | **inactive since 2026-09-21 19:42:49** (stopped by the package upgrade) |
| CDI spec `/run/cdi/nvidia.yaml` | valid | **stale**: references `libcuda.so.580.173.02`, `libnvcuvid.so.580.173.02`, `libnvidia-ml.so.580.173.02`, which no longer exist |
| `docker run --rm --gpus all … nvidia-smi` | exit 0 | **fails at create**: `open /run/nvidia-persistenced/socket: no such file or directory` |

**Timeline** (from `/var/log/dpkg.log`): 580.142 → 580.173.02 on 2026-09-14 15:35, rebooted 18:14; validation ran 2026-09-15; **580.173.02 → 580.178.04 on 2026-09-21 19:42, with no reboot since.**

**Impact:** no GPU container, and therefore no TensorRT, NVDEC or PyTorch GPU work, can run until this is resolved. That blocks Phases 4–7. The vLLM containers in the workspace compose file would fail for the same reason.

**Owner action (not taken by this audit, per "do not change the host driver setup"):** a planned reboot loads the 580.178.04 kernel module and regenerates the CDI spec at boot (`nvidia-cdi-refresh`). Afterwards, re-run the Phase 2/6 smoke tests from the host report (`docker run --gpus all … nvidia-smi`, TensorRT build, NVDEC decode). 580.178 is in the same 580 branch NGC 25.11 targets, so compatibility is expected, but it must be re-verified rather than assumed.

### 2.2 Container baseline

| Image | Contents relevant here | Status |
|---|---|---|
| `nvcr.io/nvidia/pytorch:25.11-py3` | Python 3.12.3, torch 2.10.0a0, TensorRT 10.14.1 + `trtexec`, cuDNN 9.15, ModelOpt 0.37 | Validated base (host report §9) |
| `cctv/analytics-probe:25.11` | The base plus FFmpeg 6.1 (`h264_cuvid`/`hevc_cuvid`), GStreamer 1.24 `nvcodec`, PyNvVideoCodec 2.2.3, OpenCV-headless 4.13, onnxruntime-gpu 1.29 (CUDA EP; **no TensorRT EP on aarch64**) | Validation image, labelled "not production" |

**Missing for the Dubai code:** `ultralytics`, `insightface`, `torchreid`, `qdrant-client` (plus a check that `requests` and `scipy` are present). They must be added without letting pip replace NGC's pinned torch (use `--no-deps` plus NGC's constraints, as the probe image already does). **InsightFace and torchreid have not been verified on aarch64/NGC**: InsightFace compiles a native extension, and torchreid's PyPI package is old. Verify both in Phase 3.

Both Dubai FFmpeg-cuvid decode (what `cctv.py` uses) and PyNvVideoCodec zero-copy decode were validated in this image, on synthetic streams and on the real NVR (host report §13).

### 2.3 Data services

| Service | Defined in `/home/stack/Zayed_University/docker-compose.yml` | Running | Endpoint | State found |
|---|---|---|---|---|
| PostgreSQL 15-alpine | yes | **yes** (healthy) | 127.0.0.1:5432 | Databases `litellm`, `postgres`, **`university` (exists, no tables)** |
| Qdrant v1.19.1 | yes | **yes** (healthy) | 127.0.0.1:6333 / 6334 | **No collections.** No API key configured. |
| SeaweedFS 4.47 (master, volume, filer, S3) | yes | **yes** (healthy) | filer 127.0.0.1:8888 · S3 127.0.0.1:8333 | Only `/buckets/.system`, no application buckets. The filer URL equals Dubai `evidence.py`'s default. |
| Redis 7 | yes | **no** | (127.0.0.1:6379) | Nothing listening |
| nginx, pgAdmin, LiteLLM, vLLM ×2 | yes | no | — | `/data/models` is empty, so vLLM could not start anyway |

Findings:

- **The compose file currently fails validation** (`services.pgadmin.environment must be a mapping`). Compose validates the whole file, so **no service in it, including Redis, can be started or recreated until that block is fixed.** The running containers predate the pgAdmin edit.
- **Qdrant port conflict.** Dubai `face_id_manager.py` and `known_person.py`, and the dashboard's `face_enrollment.py`, **refuse to connect to port 6333** (`_BLOCKED_QDRANT_PORTS = {6333}`); on `spark-9a32` that port was a shared multi-tenant instance. On this host 6333 is the only Qdrant, so the guard must become configurable (§5.4).
- **Two diverging definitions of the data services exist.** `/mnt/cctv/configs/data-services/` (host-setup Phase 13, never deployed) specifies Postgres 18.6, Redis 8.10.1 and Qdrant 1.19.1 with **no published ports**, an internal-only network and secrets as files. What actually runs uses published 127.0.0.1 ports, secrets from `.env`, and Postgres 15. Choose one before Phase 3 writes schemas.
- **The vLLM services share the GPU.** When running they reserve about 25% of the unified memory pool (0.15 + 0.10) and compete for compute. Include them in the Phase 6 budget, or schedule them.

### 2.4 NVR and cameras (from the host report of 2026-09-15; not re-probed in Phase 1)

| Item | Finding | Consequence |
|---|---|---|
| NVR | Hikvision iDS-7616NXI-I2/8F, firmware V4.1.62 (March 2020) | Vendor security review is an open item |
| Channels | **3 online (101/201/301)**; channel 401 returns 404 | camera_04 must be configured but disabled |
| Streams | **Main streams only.** H.264 2592×1944 (5 MP), 30 fps nominal, ~2 Mbps VBR, audio track. No sub-streams (102/202/302 → 404). | Detection runs on 5 MP frames (see phone detection in §6, bandwidth in §5.2) |
| Bitstream | GOP **255 frames (~8.5 s)**; SPS/PPS only out-of-band in the SDP | First frame took 5.5–18.8 s; a reconnect costs at least one GOP |
| Recorded playback | `/Streaming/tracks/<ch>?starttime=…&endtime=…` works, paced at about real time | The replay/evaluation source, and the future clip source (instead of local recording) |
| Cameras | Bosch units on a separate subnet, not directly routable from this host (host report addendum) | Streams come through the NVR |
| NVR account | Administrator-level account in `gb10_host_setup/nvr.env` | Create a view/playback-only user and rotate the admin password (host report blocker 2) |

Discrepancy to settle in Phase 5: `Ai_inferencing/test.py` targets an address on the camera subnet rather than `NVR_HOST`.

### 2.5 MediaMTX

| Question from the brief | Finding |
|---|---|
| Where is it installed? | **Not installed.** No binary on `PATH`, no systemd unit, no container, no `mediamtx.yml` anywhere on the filesystem. Only the image `bluenviron/mediamtx:1.21.0` is present; it was pulled for the synthetic RTSP tests in host-setup Phase 11. |
| Current configuration | None |
| Docker / container setup | None running or stopped |
| Ports | None listening (no 8554 / 18554 / 8889 / 9996) |
| RTSP paths | None on this host. Dubai served `:18554/<camera_id lowercased>` for AI, `:8889/<id>-web` (WebRTC) for the operator wall, and playback on `:9996`. |
| Authentication | Not applicable |
| Restart policy | Not applicable |
| Required by the current inference architecture? | **Dubai code assumes it.** `dashboard.py` always sets the inference source to `MEDIAMTX_BASE_URL/<camera_id>`, ignoring the camera's direct `rtsp_url`. The Dubai dashboard's clip feature (`events_log/video.py`) reads clips from **MediaMTX's own continuous recording** via its playback API. |

**Assessment for Zayed.** A single inference process does not strictly need a broker, but two facts favour a minimal MediaMTX:

1. `cctv.py` passes the full RTSP URL to FFmpeg as a command-line argument. Pointed straight at the NVR, that exposes the NVR password in `ps`, which the host report explicitly prohibits. A MediaMTX restream on 127.0.0.1 keeps NVR credentials in one 0600 config file. The alternative is the validated in-process demux path, where credentials never enter argv.
2. The future dashboard live view, snapshots and clip extraction would each open extra NVR connections without a fan-out.

If MediaMTX is used for Zayed: **recording must be off** (Zayed forbids local recording, so the Dubai clip design cannot be reused), listen on 127.0.0.1 only, keep a restart policy, and keep its config in a clean `mediamtx/` directory in the Zayed project. **Decision deferred to Phase 5** on measured criteria: first-frame and reconnect time (GOP 8.5 s), CPU cost, and whether any second consumer exists yet.

### 2.6 Storage layout

`/mnt/cctv` (host-setup Phase 3/4) provides `models/` and `configs/` (read-only for the app), `evidence/`, `snapshots/`, `thumbnails/`, `cache/` (holding only TensorRT validation engines), `logs/` and `benchmarks/`. A `cctv` service account (uid 996, gid 983) exists, and **there is no `recordings/` directory**. `evidence/`, `snapshots/` and `thumbnails/` are empty.

`/home/stack/Zayed_University/Ai_inferencing/project_frames/` holds 3 NVR playback clips (99 MB) downloaded for evaluation. That is allowed for debugging and evaluation, but keep them out of the Zayed repository and delete them after use.

### 2.7 Security and privacy findings

None of these files or values were copied; no secret is reproduced here.

| # | Where | What | Action |
|---|---|---|---|
| S1 | `dubai_ai_inferncing/cameras.cache.json` (committed, **pushed**) | 11 Dubai camera RTSP URLs with embedded username:password | Do not copy. Rotate those camera credentials; history rewrite is the repo owner's call. |
| S2 | `dubai_ai_inferncing/.env.example` (committed, **pushed**; also in `ca9cb784`) | A real-looking `DASHBOARD_TOKEN` | Do not copy. Rotate if live. Write a new placeholder-only Zayed template. |
| S3 | `Zayed_University/docker-compose.yml` | Hard-coded pgAdmin default email and password; pgAdmin published on **0.0.0.0:5050** on a host with no firewall | Move to `.env`, bind to 127.0.0.1, fix the invalid block |
| S4 | `Ai_inferencing/test.py` | One RTSP URL with an embedded camera username:password | Move to an env file (0600); rotate if it is a shared service account |
| S5 | `gb10_host_setup/nvr.env` (0600) | Administrator-level NVR account | View/playback-only account, rotate admin (host report blocker 2) |
| S6 | `dubai_ai_inferncing/face_detction/known_collection/` and `data/` (committed) | Face images and embeddings of real people | Do not copy. Zayed enrollment of students and faculty needs its own consent and retention basis (UAE PDPL). |

Also still open from the host report: no host firewall, and SSH password authentication reachable on every interface (blocker 3, draft policy prepared but not applied).

---

## 3. `camera_health` — recovery status

### 3.1 Search performed (in the order the brief requires)

| Location | Method | Result |
|---|---|---|
| Working tree | `find` for `camera_health*` | Only the 5 `testing/test_camera_health*.py` files |
| Every commit (10: 9 Dubai + `5b192cef`) | `git ls-tree -r` on every commit; `git grep` for `class CameraHealthManager` / `class HealthTelemetryPublisher` / `CAMERA_HEALTH_FEATURES =` across `git rev-list --all` | Only test files; no source |
| Branches and refs | `git for-each-ref` | `main`, `cleaned`, `origin/main`, `origin/cleaned`; no tags, no stash. The remote cannot be listed further (no GitHub credentials on this host). |
| Git object store | Every blob from `git cat-file --batch-all-objects` (**27,055 blobs**, including 8 dangling) grepped for the class definitions | **0 hits** |
| History of deletions | `git log --diff-filter=D` | Never deleted, because it was never committed: `camera_health/` was in `.gitignore` from `fdf23a7a` (2026-09-15) until `5b192cef` |
| Whole host | Content grep for the class definitions over `/home/stack`, `/mnt/cctv`, `/tmp`, `/opt`, `/srv`; `find` for `camera_health*` directories | Not present (only this session's own logs matched) |
| Sibling repositories | `dubai_pic_dash` (Dubai dashboard), `Ai_inferencing` | The dashboard *consumes* camera health; no inference-side source |

### 3.2 What the proven package is

Reconstructed from `multicam_inf.py`, the 7 test files in `testing/`, the dashboard, and `camera_tampering_merge/CAMERA_TAMPERING_SINGLE_EVENT.md` (recoverable from `9a511c37`):

- **Layout:** package `camera_health/` with `manager.py`, `detectors.py`, `metrics.py` and `state.py`. The package exports `CameraHealthManager`, `HealthTelemetryPublisher`, `CAMERA_HEALTH_FEATURES` and `PUBLISH_INTERVAL_SECONDS`.
- **Interface used by `multicam_inf.py`:**
  - `CameraHealthManager(shadow=False, event_sink=fn)`, with `.start()`, `.set_enabled_cameras(mapping)`, `.enabled_cameras()`, `.note_streams(camera_streams)`, `.observe(…)` and `.status_line()`.
  - `event_sink(camera_id, event_type, timestamp, metadata, action, frame_width, frame_height)`, routed into `events.EventPipeline.handle_camera_event`.
  - `HealthTelemetryPublisher(manager, dashboard_url, token)`, with `.start()`, `.stop()` and `.status_line()`. It posts current state about every 15 s to `POST /api/cameras/health-telemetry/` (`{"cameras": [...]}`, statuses `healthy | warning | offline`).
- **Behaviour:**
  - A CPU-only side path over frames the loop has already decoded: no model, no stream of its own, no GPU.
  - Four detectors (signal loss, obstruction, defocus, scene change) are **folded into one `camera_tamper` incident**: it raises on the first condition and recovers when the last one clears.
  - The event metadata carries `reason`, `reasons`, `conditions` and `duration_seconds`.
  - Video loss keeps CRITICAL severity.
- **Test-visible names:** `FEATURE_SIGNAL_LOSS`, `FEATURE_OBSTRUCTION`, `FEATURE_DEFOCUS`, `FEATURE_TAMPER`, `UMBRELLA_EVENT_TYPE`, `CameraState`, `Condition`, `PHASE_READY`, `PHASE_STARTING`, `metrics.SAMPLE_WIDTH/HEIGHT`, and a scene-change threshold of 0.65 "measured from real cameras".
- **Proof in Dubai:**
  - Deployed 2026-09-16.
  - A lens-cover test on real footage produced **1 raise and 1 recovery**, where the previous version produced 4 events and 2 alarms.
  - 60 + 10 unit tests passing; inference at 61–66 FPS after deployment.
- **Probable original location:** `/home/matrix/Dubai_Police/AI_inferencing/camera_health/` on `spark-9a32`, from the path hard-coded in `testing/test_camera_health.py`.

### 3.3 `spark-9a32` status

It is a peer on this host's Tailscale network (`spark-9a32.taild8e203.ts.net`) and is **offline, last seen 1 day ago**. This host has no SSH config or `known_hosts` entry for it. No login was attempted in Phase 1.

### 3.4 Phase 2 entry criteria

1. `spark-9a32` brought online by its owner (see `tailscale status`), with read access granted to this host.
2. Copy `camera_health/` **verbatim** and record SHA-256 sums.
3. Run the 7 existing tests unmodified, except for their hard-coded Dubai path, and replay the lens-cover scenario from the design document.
4. Only then make minimal Zayed changes: the telemetry target (PostgreSQL instead of the Dubai endpoint), and making sure disabled or offline cameras such as camera_04 raise no alarms.

**No replacement implementation before this.**

---

## 4. Production dependency graph (Dubai, as built)

### 4.1 Processes

```
start_production.sh                       setsid nohup, appends to multicam_inf.consensus.log, refuses a 2nd stack
└── run_inference.py                      supervisor: PID-only; stops children together; never restarts inference
    ├── multicam_inf.py                   THE inference process (everything below)
    ├── ANPR/anpr_worker.py               NOT REQUIRED
    ├── python -m weapon_detection        NOT REQUIRED (opens its own cctv.CameraStream per camera)
    └── <dashboard venv> manage.py monitor_known_person_absence
                                          Dubai DASHBOARD command: the absence half of presence
```

`door_detection/` is a fourth, separately launched worker (`door_detection/start_door_worker.sh`): NOT REQUIRED.

### 4.2 Data flow inside `multicam_inf.py`

```
Camera input   dashboard.fetch_cameras()  ->  GET /api/ai/cameras/ (cache fallback: cameras.cache.json)
               CameraConfig.source = MEDIAMTX_BASE_URL/<camera_id>   (always MediaMTX)
Decoder        cctv.CameraStream: ffprobe -> ffmpeg -hwaccel cuda -c:v h264_cuvid|hevc_cuvid
               -> raw BGR24 on a stdout pipe -> NumPy -> bounded queue (3 frames)
Preprocessing  _frames_to_infer(): capture-time spacing to TARGET_INFERENCE_FPS = 10
               _usable_frame(): rejects malformed frames (the 2026-09-16 outage guard)
               letterbox to 640 inside Ultralytics (CPU)
Model          YOLO26-L TensorRT FP16 (Ultralytics), COCO-80, dynamic batch <= 19, imgsz 640,
               one batched predict() per tick; per-camera class and confidence filters after it
Tracker        tracking.TrackManager -> jeztsort.ImprovedTracker per (camera, group)
               -> TrackedObject(camera_id, track_id "P-RUN-0017", group, class, confidence, bbox)
               EMA bbox smoothing (SMOOTH_ALPHA) before analytics
Identity       face_id_adapter -> face_id_manager (InsightFace buffalo_l, ORT CUDA EP)
                 -> Qdrant cctv_face_reid_test (F-####) + known_person -> cctv_face_known_persons
               reid_adapter -> reid_poc (OSNet osnet_ain_x1_0) -> Qdrant cctv_person_reid_production
                 -> events.set_stable_id_resolver() stamps metadata.stable_id
Business rule  inline:  zones, line_crossing, people_counter, abandoned, object_logger, distancing, fall_policy
               async:   fall_pose_adapter + fall_pose_policy  (pose_l960_runtime: yolo26l-pose @960)
                        fight_adapter + fight_fall_prototype  (pose_runtime: yolo26m-pose @640)
                        behaviour_adapter + behaviour_policy  (pose_l960_runtime)
                        ppe_adapter, fence_adapter, fire_smoke_adapter
               side:    camera_health (CPU, missing)
Event          events.EventPipeline: build -> debounce -> validate -> bounded queue (2000)
               -> EventSender thread -> POST /api/events/  (failure: counted, logged, DROPPED)
Evidence       evidence.EvidencePipeline: JPEG crop on the inference thread (new track / alarm only)
               -> upload to the SeaweedFS filer (TTL) -> THEN the event is POSTed with its reference
Storage        Dubai dashboard (Django + PostgreSQL) over HTTP; Qdrant (faces, Re-ID); SeaweedFS (crops);
               event clips pulled later by the dashboard from MediaMTX recordings
```

### 4.3 Coupling to the Dubai dashboard

| Coupling | Caller | Purpose | Zayed replacement |
|---|---|---|---|
| `GET /api/ai/cameras/` (+ `DASHBOARD_TOKEN` required before the cache fallback is even tried) | `dashboard.py` | Camera list, features, zones, lines, confidences | Local config file (camera_01–04, classroom, capacity, zones, invigilators, floor plan) |
| `POST /api/events/` | `events.py` | Every event | Write to the `university` PostgreSQL DB (with a durable outbox) |
| `POST /api/ai/people-counts/` | `people_counter.py` | Occupancy samples | PostgreSQL occupancy table |
| `POST /api/ai/face-observations/` | `face_id_sink.py` | Face observations | PostgreSQL |
| `POST /api/ai/known-person-sightings/` | `face_id_manager.py` (`KNOWN_PERSON_SINK_PATH`) | Enrolled-person sightings, which feed presence | Zayed presence service |
| `GET /api/ai/reid-groups/` | `reid_config_provider.py` | Re-ID camera groups (static fallback already exists) | Local config: one classroom = one group |
| `POST /api/cameras/health-telemetry/` | `camera_health.HealthTelemetryPublisher` | Current camera state | PostgreSQL camera-health table |
| `manage.py monitor_known_person_absence` | `run_inference.py` | Absence sweep | Zayed presence sweeper |
| MediaMTX playback `:9996` | Dubai dashboard `events_log/video.py` | Event video clips | NVR playback, when the dashboard phase needs clips |

Enrollment (`investigation/face_enrollment.py`) and the presence state machine (`investigation/presence.py`) are **dashboard code** in `dubai_pic_dash`. They have to be ported off Django for Zayed (§5.4).

---

## 5. Component audit

Columns follow the brief: component · location · purpose · Zayed requirement · status · dependencies · model · recommended action.

### 5.1 Orchestration and configuration

| Component | Location | Purpose | Req | Status | Dependencies | Model | Recommended action |
|---|---|---|---|---|---|---|---|
| `start_production.sh` | root | Starts the supervisor with setsid/nohup; appends to the log (`>>`, required by logrotate `copytruncate`); refuses a second stack | P | ADAPTABLE | bash, `aienv` venv | — | Replace with a container or systemd service that has a restart policy; keep the single-instance guard; drop the weapon-worker wait logic |
| `run_inference.py` | root | Supervises inference, ANPR, weapon and the dashboard absence monitor; stops them together; does not restart a crashed inference | P, R5 | ADAPTABLE | `dashboard.py` (.env loader), Dubai dashboard venv | — | Keep PID supervision for inference plus the Zayed presence sweeper; remove ANPR/weapon/door; add restart-on-failure |
| `multicam_inf.py` | root | Core loop (3,586 lines, runs at import time): config → cameras → batched detection → tracking → all analytics → events/evidence | P, all | ADAPTABLE | 26 local modules incl. **`camera_health` (missing, unguarded import at line 12)**; ultralytics; torch | yolo26l | Derive a Zayed main loop keeping ingest → detect → track → dispatch; remove PPE, uniform, ANPR, fence, distancing, fire and detainee-behaviour wiring; contain exceptions per iteration around `predict()` and each analytic; config from the local file |
| `dashboard.py` | root | Camera config from `GET /api/ai/cameras/`, 30 s refresh, cache fallback; forces the MediaMTX source; requires `DASHBOARD_TOKEN` | P | ADAPTABLE | requests | — | Replace with a file-based provider that builds the same `CameraConfig` objects, supports direct-NVR or MediaMTX sources, and needs no token |
| `cameras.cache.json` | root | Cached Dubai camera config | — | NOT REQUIRED | — | — | **Do not copy** (S1) |
| `.env.example` | root | Dubai env template | — | NOT REQUIRED | — | — | **Do not copy** (S2); write a Zayed placeholder-only template |
| `logrotate.conf` | root | `copytruncate` rotation | P | ADAPTABLE | logrotate | — | Only needed outside containers; otherwise use Docker json-file log limits |

### 5.2 Camera ingest and decode

| Component | Location | Purpose | Req | Status | Dependencies | Model | Recommended action |
|---|---|---|---|---|---|---|---|
| `cctv.py` | root | One FFmpeg NVDEC process per camera → BGR24 pipe → bounded queue. `start()` never blocks (connects inside the reader thread); reconnects forever every 2 s; throttled logs. An offline camera cannot stall startup. | P, R10 | ADAPTABLE | ffmpeg/ffprobe with cuvid (present in `analytics-probe`), numpy | — | Reuse as the baseline ingest. Before any direct-NVR use: **keep credentials out of argv** (MediaMTX or in-process demux). Cap the load: **5 MP BGR at 30 fps ≈ 450 MB/s per camera** is piped while inference uses 10 fps; add a GPU scale/fps filter. Compare against the validated PyNvVideoCodec path in Phase 6. |
| GB10 Phase 11/12 harness | `gb10_host_setup/benchmarks/` (not Dubai) | Validated ingest: OpenCV raw-packet demux → PyNvVideoCodec NVDEC → zero-copy CUDA NV12 → 1-slot buffers → round-robin batching → one TensorRT engine | P | PROTOTYPE | PyNvVideoCodec 2.2.3, TensorRT 10.14 | YOLOX-S (validation only) | Reference architecture for Phase 6. Labelled "not production code". Note: PyNvVideoCodec's own RTSP demuxer segfaults, so keep the callback-demuxer pattern. |

### 5.3 Detection and tracking

| Component | Location | Purpose | Req | Status | Dependencies | Model | Recommended action |
|---|---|---|---|---|---|---|---|
| YOLO26-L detector (inside `multicam_inf.py`) | root | COCO-80 detection, TensorRT FP16, dynamic batch 19, imgsz 640; per-camera class and confidence filters | R1 R4 R7 R9 R11 | ADAPTABLE | ultralytics, TensorRT | yolo26l | Build the engine on this host (batch 4); classroom class set: person, cell phone, backpack, handbag, suitcase, laptop (book optional) |
| `tracking.py` + `jeztsort.py` | root | Stable P-/V-/O- labels per (camera, group) on JeztSort; wall-clock ageing; `TRACK_*` tuning | P, all | READY | scipy, numpy | — | Copy unchanged. Derive center_x/center_y, classroom_id and person_id in a Zayed track-record layer rather than editing `tracking.py`. |
| `object_logger.py` | root | One `object_detected` event per physical object for the armed COCO classes (defaults include cell phone and laptop) | R4 R9 | READY | tracking | yolo26l | Copy; arm "cell phone" as the phone-presence signal (usage attribution is §6 R4) |
| `track_confirmation.py` | root | Temporal confirmation windows for PPE and uniform | — | NOT REQUIRED | — | — | Leave; reuse the pattern only if phone/sleeping need confirmation windows |

### 5.4 Identity (face, known persons, Re-ID, presence)

| Component | Location | Purpose | Req | Status | Dependencies | Model | Recommended action |
|---|---|---|---|---|---|---|---|
| `face_id_adapter.py` | root | Thread boundary: queues person crops per track with cooldown; never blocks cameras | R1 R5 R6 | READY | `face_id_manager` | — | Copy unchanged |
| `face_id_manager.py` | root | InsightFace buffalo_l (ORT CUDA EP **with silent CPU fallback**), quality gates, best-vs-runner-up match, gallery `cctv_face_reid_test` (auto F-####), async upsert, known-person hook, sightings sink | R1 R5 R6 | ADAPTABLE | insightface, onnxruntime-gpu, qdrant-client, `known_person`, `face_evidence`, `face_id_sink`, `reid_poc` path | buffalo_l | **Blocks Qdrant port 6333**, the only Qdrant here: make the guard configurable. Zayed collection names. Verify InsightFace on aarch64 and assert the CUDA EP is really used. |
| `known_person.py` | root | "Is this an enrolled person?" search in `cctv_face_known_persons`; opt-in; circuit breaker; **uncalibrated** threshold | R1 R5 R6 | ADAPTABLE | qdrant-client | buffalo_l embeddings | Same port guard. Zayed enrolled-person collection with student/faculty ID and name in the payload. Calibrate on classroom footage; live recognition was never validated in Dubai. |
| `face_id_sink.py` | root | POSTs face observations to the dashboard | R8 | ADAPTABLE | requests | — | Retarget to the Zayed PostgreSQL store |
| `face_evidence.py` | root | One face crop per identity to SeaweedFS | R1 R8 | READY | `evidence.py` | — | Copy |
| `reid_adapter.py`, `reid_config_provider.py` | root | Fail-safe Re-ID wrapper; camera groups from the dashboard with a static fallback | R1 R6 R11 | ADAPTABLE | `reid_poc`, dashboard (optional) | osnet_ain_x1_0 | One classroom = one group (`same_and_cross`); Zayed collection; groups from the local config |
| `reid_observability.py` | root | Opt-in JSONL log of Re-ID decisions for calibration | R1 | READY | — | — | Copy |
| `reid_poc/` library (`config.py`, `qdrant_reid.py`, `reid.py`, `reid_manager.py`, `config/`) | `reid_poc/` | OSNet embeddings (torchreid `FeatureExtractor`), Qdrant store, global identity manager | R1 R6 R11 | READY | torchreid, qdrant-client | osnet_ain_x1_0 | Copy the library only; leave its analysis and test scripts. `FeatureExtractor` is built with no `model_path`, so weights are fetched on first use: confirm which weights, and pre-stage them. |
| Enrollment: `investigation/face_enrollment.py` | `dubai_pic_dash/` | Portrait → InsightFace embedding → Qdrant known-person point | R1 R5 | ADAPTABLE | Django, insightface, qdrant-client | buffalo_l | Port off Django as a Zayed enrollment CLI (ID, name, role, photos) |
| Presence: `investigation/presence.py` + `monitor_known_person_absence` | `dubai_pic_dash/` | Persistent PRESENT / PENDING / ABSENT_ALERTED state per (person, camera) in PostgreSQL; `min_hits`; absence timeout; recognition-health gate; `CameraPersonMonitoring` assignment | R1 R5 R6 R8 | ADAPTABLE | Django ORM, PostgreSQL | — | Port the state machine to a Zayed presence service on the `university` DB, keyed per **classroom** (several cameras) rather than per camera |
| `face_detction/` | `face_detction/` | Original prototypes: `room_alarm.py` (in-memory presence), `register_known.py`, `recognize_faces.py`, `yolo_face.py` | R1 R5 | PROTOTYPE | — | YOLO face, buffalo_l | Reference only. **Do not copy the data** (S6). |

### 5.5 Use-case analytics

| Component | Location | Purpose | Req | Status | Dependencies | Model | Recommended action |
|---|---|---|---|---|---|---|---|
| `zones.py` | root | Normalised polygon zones: intrusion, perimeter, loitering, `crowd_detected` (per-zone threshold) | R6 R7 R11 | ADAPTABLE | — | — | Copy; overcrowding threshold derived from capacity; zones from the Zayed config |
| `people_counter.py` | root | Per-camera and per-zone person counts in short windows → dashboard | R7 R8 | ADAPTABLE | `zones`, requests | — | Retarget the sink; add occupancy % against capacity. **3–4 cameras in one room will over-count**, so de-duplicate (see calibration below). |
| `distance_measuring/modules/calibration.py` | `distance_measuring/modules/` | `GroundPlaneCalibration`: image foot point → metric floor coordinates via homography | R7 R11 | ADAPTABLE | cv2 | — | The basis for floor-plan heatmap coordinates and multi-camera fusion; one calibration per camera |
| `line_crossing.py` | root | Tripwires on track trajectories | R7 (optional door counting) | READY | — | — | Copy if door entry/exit counting is wanted |
| `abandoned.py` | root | Stationary, ownerless bag beyond a dwell time | R9 | ADAPTABLE | tracking | yolo26l | Copy; classroom class list, dwell and proximity from config; validate with students seated next to their bags |
| `fall_pose_adapter.py` + `fall_pose_policy.py` + `pose_l960_runtime.py` | root | Asynchronous pose fall: torso angle on COCO-17 keypoints; shared yolo26l-pose at 960 | R3 (geometry reusable for R4) | ADAPTABLE | ultralytics, TensorRT | yolo26l-pose | Copy. Dubai status: "validated offline, NOT YET LIVE" (1 positive, 16 negatives). Torso angle is camera-geometry dependent, so re-validate with controlled falls on classroom cameras. |
| `fall_policy.py` | root | Bounding-box aspect-ratio fall rule (the one live in Dubai) | R3 | NOT REQUIRED | — | — | Superseded: 16 of 16 candidates were false positives in Dubai's review. Keep only as a fallback if the pose path fails validation. |
| `fight_adapter.py` + `fight_fall_prototype.py` + `pose_runtime.py` | root | Asynchronous pose fight: limb velocity, strike, mutual confirmation; shared yolo26m-pose at 640 | R2 | ADAPTABLE | ultralytics, TensorRT, cv2 | yolo26m-pose | Copy. Dubai status: "ready for controlled pilot, **no live scenario ever run**". Staged classroom tests needed. Optionally extract the 3 symbols used from the 1,132-line prototype. |
| `behaviour_adapter.py` + `behaviour_policy.py` + `distance_measuring/modules/{pose_landmarks, consumption, face_touch, proximity, detection}.py` | root, `distance_measuring/modules/` | Pose plus held-object rules: face touch, eating, drinking | R4 | ADAPTABLE | `pose_l960_runtime` | yolo26l-pose + yolo26l | The pattern for phone use (phone held near face or hands) and for sleeping (a new head-down / torso rule). Copy only the modules used. |
| `camera_health` | — (missing) | Signal loss, obstruction, defocus, scene change → one `camera_tamper` incident; state telemetry | R10 R8 | MISSING DEPENDENCY | CPU only | — | §3 |
| `fire_smoke_adapter.py` | root | Vision fire/smoke detection | — | NOT REQUIRED | — | fire_smok.pt | Zayed uses a software or mock alarm trigger |
| `distancing.py` | root | Physical distancing | — | NOT REQUIRED | — | — | — |
| `fence_adapter.py` + `fence_detection/` | root, `fence_detection/` | Fence climbing | — | NOT REQUIRED | — | yolo26m-pose | Leave (it shares `pose_runtime` with fight) |
| `ppe_adapter.py`, `ppe_policy.py`, `ppe_episode_continuity.py`, `ppe_zone_episode.py` | root | PPE and police uniform | — | NOT REQUIRED | — | ppe_v6 | — |
| `anpr_observations.py` + `ANPR/` | root, `ANPR/` | ANPR | — | NOT REQUIRED | — | best_uae, number_plate, emirate_classifier_best, best.keras | — |
| `weapon_detection/` | `weapon_detection/` | Weapon worker (separate process) | — | NOT REQUIRED | — | yolov8x-worldv2 (+weapons6), CLIP ViT-B-32 | Reference only: a clean template for an isolated worker |
| `door_detection/` | `door_detection/` | Door-state worker | — | NOT REQUIRED | — | — | — |

### 5.6 Events, evidence and storage

| Component | Location | Purpose | Req | Status | Dependencies | Model | Recommended action |
|---|---|---|---|---|---|---|---|
| `events.py` | root | Build, validate and debounce events; bounded queue; POST `/api/events/`. **Failures are counted and dropped: no retry, no outbox.** | R8, all | ADAPTABLE | requests, `evidence` | — | Keep the builder, debouncer and validator. Replace the HTTP sink with a PostgreSQL writer behind a durable outbox. Extend the schema (§8). |
| `evidence.py` | root | One JPEG crop per new track or alarm; uploads to the SeaweedFS filer **before** the event is sent; TTL | R8, all | READY | requests, cv2, SeaweedFS filer | — | Copy; the filer URL already matches this host. Set `EVIDENCE_TTL` explicitly (the code default is 30d; its docstring says 7d). |
| `evidence_overlay.py` + `bbox.py` | root | Which boxes to draw per event type; the renderer | R8 | ADAPTABLE / READY | cv2 | — | Add the Zayed event types to the overlay; the renderer copies as is |
| Clip pipeline: `events_log/video.py`, `video_policy.py` | `dubai_pic_dash/` | Event clip from **MediaMTX recording** playback → ffmpeg → SeaweedFS | R8 | NOT REQUIRED (as built) | MediaMTX recording | — | Depends on continuous local recording, which is prohibited. Rebuild on NVR playback when the dashboard phase needs clips. |
| Main-loop exception handling | `multicam_inf.py` | `try: while True … except KeyboardInterrupt … finally` | P | ADAPTABLE | — | — | Contain exceptions per iteration and per analytic; add a supervisor/container restart policy (Phase 7) |

### 5.7 Tests, tools and prototypes

| Component | Location | Purpose | Req | Status | Recommended action |
|---|---|---|---|---|---|
| `testing/` (65 files) | `testing/` | Unit and validation tests; about 40 cover modules Zayed reuses: camera health ×7, fall ×9, fight ×4, face/known person ×7, Re-ID ×6, plus zones, crowd, tracking, abandoned, bbox | all | ADAPTABLE | Copy only the tests for copied modules; fix the hard-coded Dubai paths (for example `/home/matrix/Dubai_Police/AI_inferencing`). Use the 7 camera-health tests as the Phase 2 acceptance suite. |
| `scripts/` (79 files) | `scripts/` | ANPR audits and replays, plus core unit tests (event sender, evidence capture, track identity, zero-size frames, production lifecycle) | P, R8 | ADAPTABLE (core tests) / NOT REQUIRED (ANPR) | Copy the core unit tests only |
| `tools/` (17 files) | `tools/` | Benchmarks and dataset builders | — | NOT REQUIRED | `measure_l960_gpu.py` is optional for Phase 6 |
| `supervision_poc/` (97 files) | `supervision_poc/` | ByteTrack, zones, dwell and heatmap experiments (isolated venv) | R11 | PROTOTYPE | Reference for heatmap aggregation (`heatmap_trace_test.py`) |
| `Fall_detection/`, `Abandoned_object_detection/` | same | Original prototypes of the production rules | R3 R9 | PROTOTYPE | Reference only |
| `UAE-License-Plate-Detection-Recognition/` (9,772 files) | same | Plate-model training project | — | NOT REQUIRED | Do not copy |
| `reference_frames/` | same | 19 Dubai camera stills; no code references them | — | NOT REQUIRED | Do not copy |

---

## 6. Requirement mapping

| Req | Dubai components | Status | Gap / new work |
|---|---|---|---|
| **R1** Student/faculty monitoring | `face_id_adapter`, `face_id_manager`, `known_person`, `face_evidence`, `reid_*`, tracking; dashboard `face_enrollment.py` and `presence.py` | ADAPTABLE | Zayed person registry (student/faculty ID, name, role) and enrollment CLI; Zayed Qdrant collections; presence store answering "which classroom is X in now" and search by name or ID; threshold calibration |
| **R2** Fighting | `fight_adapter`, `fight_fall_prototype`, `pose_runtime` (yolo26m-pose @640) | ADAPTABLE | Never run on a live fight in Dubai. Staged classroom scenarios; check false positives from lively but non-violent seated rows. |
| **R3** Fall | `fall_pose_adapter`, `fall_pose_policy`, `pose_l960_runtime` (yolo26l-pose @960) | ADAPTABLE | Offline-validated only. Controlled falls on each classroom camera; torso-angle thresholds may need per-camera geometry. |
| **R4** Sleeping / lying on desk | Pose infrastructure; torso geometry from `fall_pose_policy`; `behaviour_*` pattern | New policy on an ADAPTABLE base | No existing rule. A seated person with head at or below shoulder level, or resting on the desk, with low motion for N seconds, must be separated from reading or writing. |
| **R4** Mobile phone | yolo26l COCO "cell phone" + `object_logger`; `behaviour_policy` held-object pattern | ADAPTABLE, validation-critical | **Small objects:** letterboxing a 5 MP frame to 640 shrinks it about 4×, leaving a hand-held phone only tens of pixels. Likely needs a person-crop second pass or a larger input size. Attributing a phone to a student uses pose wrists and face. |
| **R5** Invigilator presence | `known_person` + presence state machine (`CameraPersonMonitoring`) + absence sweep | ADAPTABLE | Port presence off Django; invigilator assignment per classroom (and timetable, if needed) from config; INVIGILATOR_PRESENT/ABSENT events |
| **R6** Evacuation / stranded | None for the alarm. Inputs: `people_counter`, `zones`, face ID, Re-ID, presence | NEW | Software/mock alarm trigger (API/CLI); per-classroom evacuation timer; STRANDED_PERSON_DETECTED with track, location and identity when known. `fire_smoke_adapter` is not needed. |
| **R7** Occupancy & overcrowding | `people_counter`, `zones` (`crowd_detected`), `calibration` | ADAPTABLE | Occupancy % against capacity; overcrowding at capacity; **multi-camera de-duplication** by fusing foot points on the floor plan (or a designated counting camera) |
| **R8** Backend data | `events`, `evidence`, `face_id_sink`, people-count publisher, camera-health telemetry | ADAPTABLE | Replace the 8 dashboard couplings (§4.3) with tables in `university`; SeaweedFS evidence is already compatible |
| **R9** Unattended object | `abandoned`, object tracks | ADAPTABLE | Classroom classes; dwell and owner-proximity validation |
| **R10** Camera tampering | `camera_health` | MISSING DEPENDENCY | Recover (§3); validate obstruction, defocus and scene change on classroom cameras; no alarms for disabled camera_04 |
| **R11** Floor-plan heatmap | Track foot points, `zones`, `calibration.GroundPlaneCalibration`; `supervision_poc` heatmap experiment | NEW on an ADAPTABLE base | Per-camera homography to the single classroom floor plan; aggregate an occupancy/activity grid; store coordinates for later rendering |

### Proposed event types mapped to Dubai equivalents

| Zayed event (brief) | Nearest Dubai source | Reuse |
|---|---|---|
| PERSON_PRESENT / PERSON_ABSENT | `person_detected` (track) · dashboard presence PRESENT / `known_person_absent` | Port the presence state machine |
| FACE_IDENTIFIED | Face observation / known-person sighting (`face_watchlist_hit` in the dashboard) | Adapt |
| FIGHT_DETECTED | `violence_detected` | Rename |
| FALL_DETECTED | `fall_detected` | Rename |
| SLEEPING_DETECTED | — | New |
| MOBILE_PHONE_DETECTED | `object_detected` (cell phone) | Adapt; add a usage rule |
| INVIGILATOR_PRESENT / ABSENT | Presence state machine with `CameraPersonMonitoring` | Port |
| OCCUPANCY_UPDATED | People-count telemetry (a periodic sample, not an event) | Adapt |
| OVERCROWDING_DETECTED | `crowd_detected` | Adapt (threshold = capacity) |
| UNATTENDED_OBJECT_DETECTED | `abandoned_object` | Rename |
| CAMERA_TAMPERING_DETECTED | `camera_tamper` (single folded incident) | Recover and rename |
| EVACUATION_STARTED / STRANDED_PERSON_DETECTED | — | New |

---

## 7. Model audit

**None of these weights are on this host** (a full filesystem search found no YOLO26, pose, PPE, fire, weapon, CLIP, ANPR, InsightFace or OSNet files, and no InsightFace or torch cache). TensorRT engines are version- and GPU-specific: build every engine **on this host**, inside the deployment container (TensorRT 10.14.1), and cache it keyed by TensorRT version, GPU and driver. Never copy `.engine` files from `spark-9a32`. Note that the Dubai pose runtimes silently fall back to the PyTorch `.pt` file when an engine is missing, and `behaviour_adapter` refuses to run without one: check both at startup.

| Model | Dubai use | Zayed requirement | Category | Action |
|---|---|---|---|---|
| `yolo26l.pt` → `yolo26l.engine` | Main detector (COCO-80, class names asserted at startup) | R1 R4 R7 R9 R11 (person, bags, laptop, cell phone) | **1 · directly reusable** for person and bags; **2 · classroom adaptation** for cell phone (small-object recall) | Download the public weights, pin the hash, export FP16 with dynamic batch 1–4 on this host |
| `yolo26l-pose.pt` → engine (@960) | Behaviour + Fall (`pose_l960_runtime`) | R3, R4 (sleeping, phone attribution) | **1** for fall mechanics; **2** for the new sleeping rule | Download and export on this host |
| `yolo26m-pose.pt` → engine (@640) | Fight + Fence (`pose_runtime`) | R2 | **1** | Download and export on this host. The Dubai audit found Fight cannot share the L@960 engine without changing its decisions, so keep two pose engines unless Phase 6 re-validates a single one. |
| InsightFace `buffalo_l` (det_10g, w600k_r50, genderage) | Face ID + known persons + enrollment | R1 R5 R6 | **1** as a model; **2** for the threshold (portrait vs classroom CCTV) | Pre-stage the pack; verify the ORT CUDA EP on aarch64 |
| `osnet_ain_x1_0` | Re-ID | R1 R6 R11 | **1** | Pre-stage; confirm which weights `FeatureExtractor` loads |
| Phone / sleeping specialist model | — | R4 | **3 · missing, obtain only if needed** | Decide after the Phase 4 validation of COCO + pose |
| `ppe_v6` | PPE / uniform | — | **4 · not required** | — |
| `fire_smok.pt` | Fire/smoke | — (R6 uses a software trigger) | **4** | — |
| `yolov8x-worldv2.pt`, `weapons6`, CLIP `ViT-B-32.pt` | Weapon worker | — | **4** | — |
| `best_uae.pt`, `number_plate.pt`, `emirate_classifier_best.pt`, `best.keras` | ANPR | — | **4** | — |

**Licensing — an owner decision before production** (host report item 10 recommends avoiding AGPL unless licensed):

- Ultralytics YOLO26 is **AGPL-3.0** unless an enterprise licence is purchased; the GB10 validation deliberately avoided it.
- InsightFace's published pretrained packs, including `buffalo_l`, are released for **non-commercial research use**.
- torchreid is MIT.

Fine for a POC decision; must be settled before any commercial deployment.

---

## 8. Reuse plan

- **Copy unchanged:** `tracking.py`, `jeztsort.py`, `bbox.py`, `evidence.py`, `face_evidence.py`, `face_id_adapter.py`, `object_logger.py`, `line_crossing.py`, `reid_observability.py`, and the `reid_poc/` library. Once recovered and verified, `camera_health/` too. Bring their existing tests with them.
- **Copy, then adapt:**
  - Config and orchestration: `multicam_inf.py` → Zayed main loop; `dashboard.py` → file config provider; `run_inference.py`.
  - Ingest: `cctv.py` (credentials, fps/scale).
  - Identity: `face_id_manager.py` and `known_person.py` (port guard, collections); `face_id_sink.py`; `reid_adapter.py` and `reid_config_provider.py`.
  - Analytics: `zones.py`, `people_counter.py`, `abandoned.py`, the fall-pose trio, the fight trio, `behaviour_*` with the pose modules, `calibration.py`.
  - Events: `events.py` (sink, schema), `evidence_overlay.py`.
- **Port from the Dubai dashboard (off Django):** `face_enrollment.py` → enrollment CLI; `presence.py` + `monitor_known_person_absence` → presence service and sweeper.
- **Build new:** Zayed config (camera_01–04, classroom, capacity, zones, invigilators, floor plan, enabled use cases), PostgreSQL event/occupancy/presence/camera-health/heatmap schema with an outbox, sleeping rule, phone-use rule, occupancy % and multi-camera de-duplication, evacuation/stranded state machine, heatmap aggregator, and a placeholder-only `.env.example`.
- **Leave behind:** ANPR, weapon, door, fence, distancing, PPE/uniform, fire/smoke, `track_confirmation.py`, `fall_policy.py`, the UAE training project, `reference_frames/`, `face_detction/` data, `cameras.cache.json`, `.env.example`, and all logs, `__pycache__`, `.pyc`, `.pre-*` files, spool images and old evidence.

A **common event record** for Phase 3, built on Dubai's validated `build_event()` fields (`camera_id`, `event_type`, `timestamp`, `track_id`, `confidence`, `bbox`, `frame_width`, `frame_height`, `metadata`, `stable_id`), extended with `event_id`, `classroom_id`, `person_id`, `location` (floor-plan x/y), `severity`, `status` and `evidence_reference`.

---

## 9. Blockers, risks and owner decisions

| # | Item | Impact | Owner / next step |
|---|---|---|---|
| B1 | GPU driver mismatch: userspace 580.178.04, kernel module 580.173.02, persistenced stopped, stale CDI spec | No GPU container can start; blocks Phases 4–7 | **Owner:** planned reboot, then re-run the GPU/TensorRT/NVDEC smoke tests |
| B2 | `camera_health` missing; `spark-9a32` offline | Blocks Phase 2 and R10; `multicam_inf.py` cannot import | **Owner:** bring `spark-9a32` online and grant read access |
| B3 | No model weights on this host | Blocks Phase 4 | Download the public weights (Wi-Fi internet), pin hashes, build engines on this host |
| B4 | Compose file invalid; Redis not running; pgAdmin password and 0.0.0.0:5050 | No compose service can be (re)started | **Owner:** fix the pgAdmin block, move secrets to `.env`, bind to 127.0.0.1 |
| B5 | Dubai code blocks Qdrant port 6333, the only Qdrant here | Face ID and known persons refuse to connect | Phase 3: make the guard configurable |
| D1 | Which data-services definition is authoritative (running workspace compose vs the prepared isolated one) | Schema and network design | **Owner** decision |
| D2 | Licensing: Ultralytics AGPL-3.0, InsightFace non-commercial packs | Production and commercial use | **Owner** decision |
| D3 | MediaMTX: direct NVR or a local restream (recording off) | Credentials in argv, NVR connection count | Phase 5, on measured criteria (§2.5) |
| D4 | Enrollment consent, and evidence and biometric retention (UAE PDPL) | Legal basis for R1/R5 | **Owner** / university |
| R-1 | Main loop exits on any unexpected exception; no automatic restart | One GPU error stops every camera | Phase 3 containment + restart policy; proven in Phase 7 |
| R-2 | Events dropped when the sink is down | Lost events during PostgreSQL outages | Durable outbox (Phase 3) |
| R-3 | 5 MP BGR piping at 30 fps; GB10 numbers were measured on YOLOX-S | Performance unknown for the real stack | Phase 6 profiling; GPU scale/fps cap or the zero-copy path |
| R-4 | Phone small-object recall at imgsz 640 | R4 accuracy | Phase 4 validation; crop second pass |
| R-5 | Fall and fight thresholds tuned on Dubai cameras; fight never tested live; fall pose only offline | False positives or negatives in classrooms | Staged classroom scenarios |
| R-6 | Multiple cameras in one room | Occupancy over-count | Floor-plan fusion / de-duplication |
| R-7 | GOP 8.5 s, main stream only, NVR firmware from 2020 | Slow reconnect, heavy frames | Camera/NVR settings (I-frame interval 1–2 s, sub-streams) owned by the NVR owner |
| R-8 | vLLM services share the GPU (about 25% of unified memory when running) | Resource contention | Phase 6 budget |
| R-9 | Secrets S1–S5 already exist outside this project | Credential exposure | Rotate; never copy |

## 10. Next steps

**Phase 1 is complete** and this document is its deliverable.

Before Phase 2, the owner actions **B1** (reboot), **B2** (`spark-9a32` access) and **B4** (compose fix) unblock the work, and **D1/D2** should be decided before Phase 3 writes schemas and pulls models.

Phase 2 then starts with recovering `camera_health` exactly as in §3.4. No replacement will be written before the original has been checked.

---

### Appendix A — evidence sources

- Host setup report and raw results: `/mnt/cctv/benchmarks/GB10_HOST_SETUP_REPORT.md`, `phase11_video_pipeline/`, `phase12_four_camera/`.
- Validated harness and container: `gb10_host_setup/benchmarks/phase12_four_camera_arch_test.py`, `/mnt/cctv/benchmarks/scripts/dockerfiles/analytics-probe/Dockerfile`.
- Prepared data services: `/mnt/cctv/configs/data-services/` (README + compose).
- Workspace services: `/home/stack/Zayed_University/docker-compose.yml`.
- Dubai validation reports, recoverable with `git show 9a511c37:<file>`:
  - `camera_tampering_merge/CAMERA_TAMPERING_SINGLE_EVENT.md`
  - `fall_pose_production.md`, `fall_pose_validation_report.md`, `fall_production_integration_report.md`
  - `fight_fence_production_readiness.md`, `fight_fence_integration_report.md`, `fight_fence_preproduction_report.md`
  - `shared_pose_audit_report.md`, `shared_pose_yolo26l_unification_report.md`
  - `full_round_test_report.md`
  - `FACE_ID_KNOWN_PERSON_DEPLOYMENT_REPORT.md`, `FACE_ID_KNOWN_PERSON_PHASE2_REPORT.md`, `FACE_ID_QDRANT_MIGRATION_REPORT.md`, `FACE_ID_THRESHOLD_040_CALIBRATION_REPORT.md`
  - `AI_INFERENCING_PRODUCTION_INVENTORY.md`
- Dubai dashboard (presence, enrollment, clips, telemetry ingest): `dubai_pic_dash/investigation/{presence.py, face_enrollment.py}`, `investigation/management/commands/monitor_known_person_absence.py`, `events_log/{video.py, video_policy.py}`, `cameras/api_urls.py`.
