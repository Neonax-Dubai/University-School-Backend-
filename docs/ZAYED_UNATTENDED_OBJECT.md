# Unattended objects: one alert per physical episode

2026-10-02.

| Part | Files |
|---|---|
| Inference code | `abandoned.py`, `events.py` (`handle_abandoned_event`, `handle_abandoned_resolution`), `zayed_inference.py` (call site) |
| Dashboard code | `zayed/unattended.py`, `events_log/api_views.py` (dispatch), `events_log/models.py` (detail line) |
| Tests | `tests/unit/test_unattended_episodes.py`, `tests/unit/test_abandoned_spot_continuity.py`; dashboard `zayed.test_unattended_episodes` |
| Isolated replay | `tests/unattended/` |

## 1. The failure

On 2026-09-30, 10:26–12:05Z, C101 raised **24 "Unattended Object" alarms** for a few bags (inference log, `[UNATTENDED]` lines). None of them ever closed.

| Pattern in the log | Example | Cause |
|---|---|---|
| The same track id alarms again and again | camera_02 backpack `O-101W-0877` alarmed **7 times** (11:44–12:04Z); camera_01 backpack `O-101W-1064` 3 times | A person within the proximity radius, often just passing or at the back of the room, reset the resting spot, `alerted` flag included. The next 32 s with nobody near alarmed again. |
| A new track id alarms again | camera_01 handbag `0940 → 0956 → 0975 → 0994 → 1011` | The bag was detected at confidence 0.25–0.57 and missed for up to 70 s at a time. The spot was forgotten after 30 s unseen, and the returning detection started over. |
| Nothing downstream folds them | every row | `events.py` debounced per **track id** (30 s), and a new id bypassed it. The dashboard raised one Alarm per event and had no recovery. |

The isolated replay of the recorded footage (§8) reproduces **22 of the 24** alarms: every one inside the replayed windows, 21 of them within 11 s of the log (one at 22 s) after the ~45 s NVR clock offset. The other two (camera_02 at 10:26 and 11:26Z) fall just before the replayed windows.

## 2. Where the old code made those decisions (production `d32af38`)

1. **Track id as identity.** `abandoned.py` already kept its clock per resting **spot** (Dubai, 2026-09-15), not per track. But:
   - `events.EventDebouncer` keys on `(camera, track_id, type, scope)`, so the abandoned-object cooldown was per track (`events.py:784-812`);
   - the dashboard creates `Alarm(ALM-{event_id})` per event (`events_log/api_views.py`, the `ALARM_RULES` branch).
2. **A new track creates a new candidate.** `AbandonedObjectDetector._spot_for` (`abandoned.py:215-244`) creates a new spot when no spot lies within the move tolerance, or when the nearby spot was last seen more than 30 s ago.
3. **The event.** `zayed_inference.process_result` passes the carried tracks and person boxes to `abandoned_detector.update` (`zayed_inference.py:631-639`). For each object that crossed the dwell, `event_pipeline.handle_abandoned_event` calls `_emit`, which crops the alert evidence and queues the event to the outbox. The outbox renames it `UNATTENDED_OBJECT_DETECTED` and posts it to `/api/events/`, and the dashboard stores an Event and an Alarm.
4. **Deduplication.**
   - The detector fired once per spot, but `_reset` cleared `alerted` whenever a person came near (`abandoned.py:298-300`, `335-342`).
   - The event layer's per-track 30 s cooldown is shorter than the 32 s re-arm, so it never folded a repeat.
   - The dashboard had none.
5. **Disappearance.** Nothing was resolved:
   - a spot was simply no longer matched after 30 s;
   - `prune()` was never called, so state grew without bound;
   - no event said the object had gone, so the alarm stayed open forever.

## 3. Track id versus physical object

A tracker id (`O-RUN-NNNN`) is one continuous run of matched detections. `tracking.py` retires an object's label after 1.5 s of coasting.

At C101:
- A still bag is detected at confidence 0.25–0.57 and drops out for up to **70 s with nobody near it**.
- The camera_01 bag of 15:44–16:18 passed through **99 track ids**, alternating "handbag" and "backpack" (65 tracks flipped class).
- The camera_02 backpack kept one id the whole time and still alarmed 7 times.

A left bag does not move unless someone handles it, so its place, size and time, together with who was near it, are the identity that survives.

## 4. Episode lifecycle

```
            person near (held)                   dwell 30 s and observed ≥ 30 % of it
 OBSERVED ─────────────────────▶ CANDIDATE ──▶ UNATTENDED ─────────────────────────────▶ ALERTED
    ▲   nobody near: grace 2 s      │              │                                      │
    └───────────────────────────────┴──────────────┘ person near (pre-alert rule)         │ (one alert)
                                                                                          ▼
                    RESOLVED: removed │ moved │ attended (alerted episodes report it once)
```

**Association.** Each detection (after the same-frame cross-class dedupe) joins an episode **on its own camera**, in this order:
1. the episode its track id already belongs to;
2. else an unresolved episode not seen this frame that meets all of:
   - centre within `ASSOCIATION_DISTANCE` (0.75) × the larger box diagonal of its resting position;
   - box area within `ASSOCIATION_SIZE_RATIO` (4);
   - a compatible class (all bag classes by default);
   - still inside `RECOVERY_WINDOW`;
3. else a new episode.

A detection that overlaps an episode already seen in the same frame is a fragment of it, not a new object. No appearance model is used: on the recordings, place, size and time were enough (§8).

**Occlusion.** An unseen episode is not ended by a missed frame. Its "unseen" clock counts only time when the resting spot is in view: time with a person at the spot does not count, since they may simply be in front of it. A camera stall (no frames) counts at most 2 s.

**Resolution** is deterministic. Only ALERTED episodes report it; pre-alert episodes end silently.

| Reason | Rule (default) |
|---|---|
| `removed` | unseen for `RECOVERY_WINDOW` = 120 s with the spot in view, or 600 s in all |
| `moved` | the centre stays outside the resting tolerance (0.5 × box diagonal) for 1 s, **and** a person was at the spot within the preceding 10 s. A resting object only moves when someone handles it, so a box that wanders with nobody there is an artefact: no move and no dwell progress. A new episode starts where the object now is. |
| `attended` | a person stays within the proximity radius for 300 s continuously (gaps up to the 2 s grace) |

The resolution is sent once, as the **same event type** with `metadata.recovered = true` and the same `episode_id`. This is the camera-tampering incident pattern, so it needs no new event type and no migration. The event carries no crop and no clip (`video.required = false`).

**Proximity hysteresis.**
- Before the alert, the original rule stands: a person near the object holds its clock (`PREALERT_ATTEND_SECONDS` = 0).
- After the alert, a person passing, standing in front of the object, or nudging it within its resting tolerance neither re-raises nor closes it. Only 300 s of continuous attendance counts as someone taking charge.

**After a resolution** the next unattended spell is a new episode with its own alert. That is the only way a second alert for the same object happens. There is no global cooldown.

## 5. Alert deduplication end to end

| Layer | Rule |
|---|---|
| `abandoned.py` | one alert per episode, guaranteed by the state machine |
| `events.py` | backstop cooldown keyed by `episode_id`, not track id |
| dashboard `zayed/unattended.py` | One open alarm per `(camera, episode_id)`. A resolution closes exactly that episode's alarm, new or acknowledged, and never raises one. |
| dashboard, after an inference **restart** | An episode re-created for an object still lying there joins the open alarm at the same place (box IoU ≥ 0.3), and the alarm follows the live episode. |

An event without an `episode_id` (an older inference build) still raises an alarm (fail-open, as zone alarms do).

**Evidence.** There is one crop per episode, the alert's own. Before, every repeat alert cut and uploaded another crop and opened another alarm. The resolution event carries none.

## 6. Configuration

Environment variables of the inference container. Each value is checked against the 2026-09-30 recordings (§8).

| Variable | Default | Evidence for the default |
|---|---|---|
| `ABANDON_DWELL_SECONDS` | 30 | unchanged |
| `ABANDON_OWNER_GRACE_SECONDS` | 2 | unchanged; also the gap allowed inside one attendance run |
| `ABANDON_PROXIMITY` | 0.12 of the frame diagonal (≈ 389 px) | unchanged; see limitations |
| `ABANDON_RECOVERY_WINDOW_SECONDS` (old name `ABANDON_SPOT_MEMORY_SECONDS`) | **120** (was 30) | The longest in-view dropout of a bag that was still there was 69.7 s (camera_01). 60 s split that bag into 4 alarms; 90 s and above gave 1. |
| `ABANDON_ATTENDED_RESOLVE_SECONDS` | **300** | Transient attendance runs reached 50 s. 60–120 s closed the camera_01 alarm on a neighbour at the next desk; 300 s closed it when the owner took the bag (`moved`, 16:18:04). Alarm counts were the same from 30 s to ∞. |
| `ABANDON_MOVE_TOLERANCE` / `ABANDON_MOVE_CONFIRM_SECONDS` | 0.5 × diagonal / 1.0 s | The tolerance is unchanged. The confirmation stops one bad box from counting as a move. |
| `ABANDON_MOVE_HANDLING_SECONDS` | 10 | Both real moves seen (the owner taking a laptop out; the owner collecting the bag) had the owner at the spot. |
| `ABANDON_ASSOCIATION_DISTANCE` | **0.75** × diagonal | The owner put a bag back 0.52 diagonals from where it was. At 0.5 the old alarm stayed open 88 s beside the new one. Hand-offs from detector dropouts were 0.00–0.01 apart. |
| `ABANDON_ASSOCIATION_SIZE_RATIO` | 4.0 (area) | partial occlusion halves the box; a phone-sized box does not continue a bag |
| `ABANDON_ASSOCIATION_STRICT_CLASS` | 0 | one bag was reported as handbag and backpack alternately |
| `ABANDON_MAX_UNSEEN_SECONDS` | 600 | caps an episode hidden behind people |
| `ABANDON_PREALERT_ATTEND_SECONDS` | 0 | original rule (Dubai tests) |
| `ABANDON_CLASSES` (old name `ABANDON_CLASS_IDS`) | `backpack,handbag,suitcase` | see below |
| `ABANDON_MIN_OBSERVED_FRACTION`, `ABANDON_OBJECT_CONFIDENCE` | 0.3, 0.25 | unchanged |

**Classes.**
- **Supported:** only COCO `backpack` (24), `handbag` (26) and `suitcase` (28, alias `luggage`). These are what the deployed YOLO26-L reports and what the use case was validated with.
- **Refused:** anything else is refused with a log line and never enabled. The COCO model has **no box/package class**, so that needs a custom-trained detector.
- **Fallback:** if nothing valid is configured, the validated default applies.
- **Scope:** classes are global, not per camera.

## 7. Tests

| Suite | Result |
|---|---|
| `tests/unit/test_unattended_episodes.py` (new): brief items 1–17, moves (handled / wandering / nudged / relocated), fragments, size, strict class, stalls, retention, class config, event layer | 37/37 |
| `tests/unit/test_abandoned_spot_continuity.py`: the Dubai suite, test bodies unchanged | 12/12, and 12/12 on the untouched Dubai file |
| All inference unit suites (`tests/unit/run_unit.sh`) | 116/116 |
| Resilience, camera-health acceptance | pass, as on production `d32af38` |
| Mutation checks (`abandoned.py`, `events.py`) | 19/19 caught |
| Dashboard `zayed.test_unattended_episodes` | 13/13; 11/11 mutants caught |
| Dashboard full suite / zayed app / `makemigrations --check` | same 93 pre-existing failures as base `5d0ddd3` / 205/205 / no changes |

## 8. Replay validation (isolated)

Method:
- **Stage 1 (GPU, `--network none`):** recorded clips go through the production detector, filters, tracker and box smoothing, and the dump keeps exactly what the analytic receives. NVR clips were pulled read-only through the recordings viewer.
- **Stage 2 (CPU):** the dump goes through production `abandoned.py` (with its per-track debounce and one alarm per event) and through the new detector (with the dashboard's episode handling), using identical inputs on the video clock.
- **Ground truth:** the reviewer looked at the rendered frames at every alarm and at every resolution.

| Window (NVR, 2026-09-30) | What happened (frames reviewed) | Raw track ids | Baseline alarms | New alarms | Episodes: tracker ids merged → resolution |
|---|---|---|---|---|---|
| camera_02 14:30–15:05 | backpack on a chair; its owner sits beside it from about 14:33 | 3 | 1 | 1 | `UA-…0001`: 2 ids → `attended` 14:38:24 |
| camera_02 15:30–16:07 | the same backpack, left; people pass or sit at the back of the room | 5 | **7** (all `O-0SDA-0001`) | **1** | 1 id → still open at the end (bag still there) |
| camera_01 14:25–14:57 | owner leaves a bag; returns, takes the laptop out (bag moved); sits with it 5 min; leaves; returns 45 s and nudges it; leaves | 277 | **7** (7 ids) | **3** | 1 id → `moved` 14:33:50 · 11 ids → `attended` 14:41:27 · 33 ids → still open |
| camera_01 15:44–16:23 | one bag left on a chair before 15:44; owner collects it at 16:18 | 570 | **8** (7 ids) | **1** | **99 ids** `O-021J-0003 … 0606` → `moved` 16:18:04 |
| camera_03 16:00–16:17 | no bag | 0 | 0 | 0 | – |
| 09-26 17:48–17:56, three cameras | bags carried, never left | 105 | 0 | 0 | – |

**Totals:** 23 alarms before and 6 after for 6 physical unattended episodes, so **17 false duplicates become 0**. The new code raises no false alarm on the negative sequences.

**Latency.** The first alert of every episode comes at the same moment as the baseline's. The episode ends are reported:
- `moved` 1 s after the object is moved;
- `attended` after 300 s of attendance;
- `removed` after 120 s of seeing the spot empty.

The alarm used never to close.

**Performance:**
- `update()` costs p50 0.8–8 µs and p95 ≤ 11 µs per camera frame (production: p50 0.6–2.8, p95 ≤ 4), against a 100 ms frame budget at 10 fps.
- There is no GPU work: the detector, tracker and model are untouched.
- Memory is bounded: resolved episodes are dropped after 300 s, where the old spots were never pruned.

## 9. Known limitations

1. **Proximity is in pixels.** It is 12% of the frame diagonal, about 389 px. With C101's perspective, people at the back of the room, or at the next desk, count as "near". Before an alert this holds the clock, which can delay an alert. After an alert, 300 s of such presence closes it as `attended`. A floor-plane radius needs calibration (homographies are empty).
2. **What ends a spell is a rule, not intent.**
   - A 45 s visit that leaves the bag within its resting tolerance continues the episode (camera_01, 14:51).
   - Picking the bag up and putting it back elsewhere starts a new one (14:33).
   - An operator may read either as the owner "returning".
3. **Episode state is in memory.**
   - After an inference restart, an object still lying there gets a new episode, which the dashboard folds into the open alarm at the same place.
   - An object removed while inference was down leaves its alarm open, because nothing observed the removal. An operator closes it.
   - The same applies to a camera dropped from inference (`forget_camera`).
4. **No appearance identity.**
   - A different bag put in the same place, of similar size, within 120 s of the first one's last sighting, continues the first episode.
   - A second bag set down within 0.75 diagonals by a person, while the first is momentarily undetected, ends the first as `moved`.
5. **Long dropouts.** A dropout longer than the recovery window, with the spot in view, ends an episode as `removed`. If the bag is detected again afterwards, it raises a new alert. The longest dropout seen was 69.7 s.
6. **Configuration is global** (environment), not per camera, and only the three COCO bag classes are supported.
7. **Only the alert crop is kept.** There is no separate "first unattended" still, and no "latest" still.
8. **Recovery rows in the Events list.** A resolution appears as an "Unattended Object Detected" row reading "Resolved: removed after Ns …", as camera-tampering recoveries do.
9. **Ground truth is a reviewer's reading of the frames,** not operator labels. Replays use one of the three 10 fps frame phases, and each window starts with empty state.
