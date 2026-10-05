# Zayed changes to modules copied from the Dubai baseline

Baseline: `dubai_ai_inferncing` at commit `5b192cef` (frozen, never modified).
`DUBAI_BASELINE_CHECKSUMS.txt` holds the SHA-256 prefix of every copied file **as it is in
Dubai**. 23 of the 32 copied files are still byte-identical to it. The nine below were changed
on purpose. Each change is marked `ZAYED` in the source, except `cctv.py`'s stall watchdog
(see its row).

Verify at any time:

```bash
cd zayed_ai_inferencing
while read -r sum f; do [ "$(sha256sum "$f" | cut -c1-16)" = "$sum" ] || echo "changed: $f"; done \
  < <(grep -E '^[0-9a-f]{16}  ' docs/DUBAI_BASELINE_CHECKSUMS.txt)
# expected output: exactly the nine files listed below
```

| File | Change | Why |
|---|---|---|
| `events.py` | Uses the durable `outbox.OutboxEventSender` instead of the in-memory `EventSender`. Adds `zayed_enrich()`, which maps Dubai event names to the Zayed vocabulary (`violence_detected`→`FIGHT_DETECTED`, `fall_detected`→`FALL_DETECTED`, `abandoned_object`→`UNATTENDED_OBJECT_DETECTED`, `camera_tamper`→`CAMERA_TAMPERING_DETECTED`; zone `crowd_detected` stays as it is since Phase 4), and `set_classroom_resolver()`, which stamps `classroom_id`, `location`, `person_id` and `evidence_reference`. Unattended objects (2026-10-02): the `abandoned_object` debounce is keyed by `episode_id`, and `handle_abandoned_resolution()` reports an alerted episode's end. | Dubai dropped events whenever the dashboard was down. The Zayed backend expects classroom-level fields. A track id is not an object identity (see `abandoned.py`). |
| `dashboard.py` | Default URLs are `zayed-dashboard:8000` and `mediamtx:8554`. The config cache lives under `runtime/`. Zayed feature keys added. `CameraConfig` carries `classroom` and `placement`. The stream source is the backend's `stream_url`. | The backend is the system of record for camera placement. |
| `evidence.py` | `EVIDENCE_KEY_PREFIX` is configurable (default `zayed-evidence`), and so is `EVIDENCE_TTL`. | Keep Zayed evidence separate from any Dubai key space. |
| `known_person.py` | `KNOWN_PERSON_ALLOW_SHARED_QDRANT` opt-in for port 6333. | On the Zayed GB10, 6333 is the project's own Qdrant (the same opt-in face ID and Re-ID already had). |
| `face_id_manager.py` | `_require_gpu()` after `FaceAnalysis.prepare()` (`FACE_ID_REQUIRE_GPU`, default on). | ONNX Runtime silently falls back to CPU when CUDA fails. Zayed refuses that. |
| `face_id_adapter.py` | `camera_armed()` and `submit()` upper-case the camera id before the membership test (as `fall_pose_adapter` and `fight_adapter` already do). | `set_enabled_cameras()` stores upper-case ids. Dubai ids (`CAM-R25`) were already upper case, so the bug never showed. With Zayed's `camera_01`, face ID observed nothing. Found live on 2026-09-22. |
| `cctv.py` | Per-camera stall watchdog: an ffmpeg that is alive but has produced no frame for `CCTV_STALL_TIMEOUT_SECONDS` (20 s) is killed, so the reader's existing reconnect path runs; `stalls_recovered` in `get_stats()`. Tests: `tests/unit/test_cctv_stall.py`. Not marked `ZAYED` in the source. | A decoder that goes quiet without exiting parked the reader for hours (2026-09-24, twice). Committed in `ba1a578` ("before remediation"); this row was added in the fall remediation because the log had not recorded it. |
| `abandoned.py` | Each resting object is an EPISODE (OBSERVED → CANDIDATE → UNATTENDED → ALERTED → RESOLVED) that survives tracker-id changes: a new track continues an unseen episode nearby (distance, size, class compatibility, recovery window). One alert per episode; brief proximity no longer re-arms an alerted one; deterministic resolution (removed / moved / attended) reported once through `drain_resolutions()`; `update(..., now=)` takes the frame time. Configurable, validated classes (`ABANDON_CLASSES`). Tests: `tests/unit/test_unattended_episodes.py`, plus the Dubai suite copied as `tests/unit/test_abandoned_spot_continuity.py`. | 2026-09-30: 24 "Unattended Object" alarms in 1 h 40 min for a few bags (one track alarmed 7 times; one bag passed through 99 track ids). `docs/ZAYED_UNATTENDED_OBJECT.md`. |
| `fall_pose_policy.py` | Lower-body evidence for every "down" rule: the ankle axis, else the knee->shoulder axis, else hips and shoulders must both have dropped towards the floor from the person's last upright pose (`HIP_DROP_MIN_RATIO` 1.0, `SHOULDER_DROP_MIN_RATIO` 2.0 upright torso lengths). `CONFIRM_SECONDS` 0.3 -> 1.0. New event metadata (`down_rule`, `lower_body_evidence`, drops, `policy_revision`). | Event 32 (2026-09-28): a person bending over a desk raised a critical "Person Fell". The desk hid the ankles, so Dubai fell back to the torso alone and the ground rule trusted a desk-truncated box. `docs/ZAYED_FALL_REMEDIATION.md`. |

New Zayed files (not in Dubai): `zayed_inference.py`, `outbox.py`, `occupancy.py`,
`identity_supervisor.py`, `internal_api.py`, `phone_use.py`, `sleeping.py`, `evacuation.py`, `floor_plan.py`,
`camera_health/`, `reid_poc/config/camera_groups.json`, `mediamtx/`, `docker/`,
`docker-compose.yml`, `tools/bootstrap_models.*`, `tests/camera_health/run_acceptance.sh`,
`tests/unit/`, `tests/resilience/`, `tests/fall/` (fall scenarios, Dubai regression runner,
isolated clip replay), `tests/occupancy/` and `tests/unattended/` (isolated replays).

Zayed behaviour that is set from `zayed_inference.py` rather than by patching a module:
`evidence.ALARM_EVIDENCE_TYPES` gains `MOBILE_PHONE_DETECTED`, `SLEEPING_DETECTED`,
`STRANDED_PERSON_DETECTED` and `OVERCROWDING_DETECTED` at startup. Face-ID and Re-ID that
fail to start are rebuilt by `identity_supervisor.py` and swapped in through the main loop's
globals, so `face_id_adapter`'s and `reid_adapter`'s own start-up latches are left untouched.

`line_crossing.py` is copied but not used by `zayed_inference.py` (line crossing is not a
Zayed use case).

## 2026-09-23: GPU-only internal face API

`internal_api.py` (new, inference) serves `POST /internal/face/embed` and `GET
/internal/face/health` on the container network only, with bearer-token auth. It loads ONE extra
buffalo_l instance with `providers=[CUDAExecutionProvider]` alone and verifies every session's
effective provider, so it can never compute on the CPU. The CCTV `FaceIDManager` instance is not
shared (its det_thresh is baked in at prepare time), and API requests are serialised by a
semaphore of one with a bounded wait, answering 429 rather than delaying camera processing.

The dashboard no longer loads InsightFace at all: `investigation/face_client.py` (new) posts the
photo to that API, and `face_enrollment.py` / `face_search.py` keep every business rule, the
Qdrant write and the Qdrant read. `face_id_manager.py` is unchanged (still `a3746fd23c5cc2dc`).
