# Unattended objects: one alert per physical episode

2026-10-02. Extended 2026-10-05 to laptops and cell phones (§10).

| Part | Files |
|---|---|
| Inference code | `abandoned.py`, `events.py` (`handle_abandoned_event`, `handle_abandoned_resolution`), `zayed_inference.py` (call site), `detection_routing.py` (which detections reach the tracker and the phone analytic) |
| Dashboard code | `zayed/unattended.py`, `events_log/api_views.py` (dispatch), `events_log/models.py` (detail line) |
| Tests | `tests/unit/test_unattended_episodes.py`, `tests/unit/test_abandoned_spot_continuity.py`, `tests/unit/test_detection_routing.py`; dashboard `zayed.test_unattended_episodes` |
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
| `ABANDON_ASSOCIATION_STRICT_CLASS` | 0 = same class **group** (bags together; laptop alone; phone alone) | one bag was reported as handbag and backpack alternately; a laptop or a phone beside a bag is another object (§10) |
| `ABANDON_MAX_UNSEEN_SECONDS` | 600 | caps an episode hidden behind people |
| `ABANDON_PREALERT_ATTEND_SECONDS` | 0 | original rule (Dubai tests) |
| `ABANDON_CLASSES` (old name `ABANDON_CLASS_IDS`) | `backpack,handbag,suitcase,laptop,cell phone` | see §10 |
| `ABANDON_MIN_OBSERVED_FRACTION`, `ABANDON_OBJECT_CONFIDENCE` | 0.3, 0.25 | unchanged |

**Classes:** see §10. Only supported classes can be enabled; anything else is refused with a log line and never enabled. If nothing valid is configured, the validated default applies. Classes are global, not per camera.

## 7. Tests

| Suite | Result |
|---|---|
| `tests/unit/test_unattended_episodes.py`: brief items 1–17, moves (handled / wandering / nudged / relocated), fragments, size, strict class, stalls, retention, class config, event layer; from 2026-10-05 every supported class, laptop A–H, phone A–F, class groups (§10) | 62/62 (37 on 2026-10-02) |
| `tests/unit/test_detection_routing.py` (2026-10-05): the phone analytic's input equals the f4e16a9 routing for every combination; phones reach the tracker only as an unattended class | 6/6 |
| `tests/unit/test_abandoned_spot_continuity.py`: the Dubai suite, test bodies unchanged | 12/12, and 12/12 on the untouched Dubai file |
| All inference unit suites (`tests/unit/run_unit.sh`) | 147/147 (116 on 2026-10-02) |
| Resilience, camera-health acceptance | pass, as on deployed `f4e16a9` |
| Mutation checks (`abandoned.py`, `events.py`, `detection_routing.py`) | 30/30 caught (19 on 2026-10-02) |
| Dashboard `zayed.test_unattended_episodes` | 16/16 (13 on 2026-10-02); 13/13 mutants caught |
| Dashboard full suite / zayed app / `makemigrations --check` | same 93 pre-existing failures as base `c7883e2` / 221/221 / no changes |

The 2026-10-05 change edited three existing `ClassConfigTests` because they pinned the old supported set: the refused list no longer contains 67, the fallback is the five classes, and the default test is renamed. Their intent (refuse unsupported classes, never enable them, fall back to the validated default) is unchanged.

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
6. **Configuration is global** (environment), not per camera. Only the five COCO classes of §10 are supported.
7. **Only the alert crop is kept.** There is no separate "first unattended" still, and no "latest" still.
8. **Recovery rows in the Events list.** A resolution appears as an "Unattended Object Detected" row reading "Resolved: removed after Ns …", as camera-tampering recoveries do.
9. **Ground truth is a reviewer's reading of the frames,** not operator labels. Replays use one of the three 10 fps frame phases, and each window starts with empty state.

## 10. Supported classes (2026-10-05)

### 10.1 What is supported, and why exactly these

The deployed detector is **YOLO26-L TensorRT FP16** (`/engines/zayed/current/yolo26l.engine`), trained on **COCO, 80 classes**. The class map is read from the engine's own metadata header, not from documentation.

| Class | COCO id | Class group | Dashboard label | Status |
|---|---|---|---|---|
| backpack | 24 | bag | "Bag" | supported (since 2026-09) |
| handbag | 26 | bag | "Bag" | supported (since 2026-09) |
| suitcase | 28 | bag | "Suitcase" | supported (since 2026-09) |
| laptop | 63 | laptop | "Laptop" | **supported from 2026-10-05** |
| cell phone | 67 | phone | "Cell phone" | **supported from 2026-10-05** |

The dashboard shows backpack and handbag as "Bag" by an existing, deliberate rule: the detector flips between the two on one bag. The stored `metadata.object_type` keeps the exact class.

**Not supported:**

| Item | Why |
|---|---|
| wallet | Not a COCO class. The model has no wallet output, and no class is mapped to it (`wallet` is refused by `ABANDON_CLASSES`). A wallet may sometimes be detected as "cell phone"; it is then reported as a cell phone, never as a wallet. |
| generic package, box | Not COCO classes. |
| generic "personal belongings" | Not a class of any deployed model. Approximating it by enabling other COCO classes (book 73, bottle 39, cup 41, chair 56, …) would alarm on room furniture and on what students normally keep on desks. These are refused, not enabled. |

**Future custom classes** (each needs a detector that outputs it; none is started): wallet; laptop bag as its own class (today it is a "handbag" or "laptop"); package; box; notebook; other valuable item. Options for a wallet, in order of effort:

1. **Fine-tune YOLO26 with an added wallet class.** This means collecting and labelling wallet images (public datasets plus Zayed CCTV at C101/C201/C301 angles), training on the 80 COCO classes plus the new class, re-exporting TensorRT and re-validating every existing analytic (person, bag and phone accuracy must not drop).
2. **A second, small personal-belongings detector**, run only on cameras with unattended-object armed. It costs extra GPU time per frame and must be measured against the 10 fps per camera budget.
3. **Another validated model**, for example an open-vocabulary detector. It needs a latency and false-alarm validation on these cameras before any use.

### 10.2 The same lifecycle for every class

Laptops and phones go through the unchanged episode lifecycle of §4: OBSERVED → CANDIDATE → UNATTENDED → ALERTED → RESOLVED (`removed`, `moved`, `attended`). They use the same dwell (30 s), owner grace (2 s), proximity radius (12 % of the frame diagonal), recovery window (120 s), attended resolution (300 s), object confidence floor (0.25), one alert per episode, one evidence crop per episode, and the same dashboard alarm folding by `(camera, episode_id)`.

There are **no per-class thresholds**: the replay (§10.6) showed no need for one.

A phone in a hand lies inside its holder's person box, at distance 0, so it is always attended. A phone on a desk in front of a seated student is within the proximity radius. The phone alerts only when nobody has been near it for the dwell.

### 10.3 MOBILE_PHONE_DETECTED and UNATTENDED_OBJECT_DETECTED (cell phone) are different use cases

| | `MOBILE_PHONE_DETECTED` (`phone_use.py`) | `UNATTENDED_OBJECT_DETECTED`, `object_type = "cell phone"` (`abandoned.py`) |
|---|---|---|
| Question | Is a person **using** a phone? | Has a phone been **left** with nobody near it? |
| Input | Raw phone boxes ≥ `PHONE_CONFIDENCE` 0.30, plus person tracks | Tracked phone boxes ≥ 0.25 (on cameras with `abandoned_object` armed), plus person boxes |
| Fires when | A phone box stays attached to a person (centre in the upper 80 % of the person box) for ≥ 5 s within 15 s, over ≥ 6 frames; one event per person per 300 s | A phone has rested with nobody within the radius for 30 s; one alert per physical episode |
| A phone in a hand | the event | never an alert (attended) |
| Camera flag | `mobile_phone_detection` | `abandoned_object` |
| Changed on 2026-10-05 | **No.** The same boxes, at the same threshold, reach it (pinned by `test_detection_routing.test_the_phone_analytic_receives_exactly_what_it_did_before`) | Phones added as a class |

### 10.4 The routing change (`detection_routing.py`)

Before, a phone ≥ 0.30 on a phone-armed camera went **only** to `phone_use`: the tracker, and therefore the unattended analytic, never saw it. Adding 67 to the allowlist alone would have been a silent half-feature. `detection_routing.route()` is now the one rule for the live loop and the replay:

- A phone goes to `phone_use` exactly as before.
- **Additionally**, it goes to the tracker, but only when "cell phone" is an unattended class and `abandoned_object` is armed on that camera.
- With either condition off, a phone is routed exactly as before.
- Every other class routes exactly as before.

YOLO inference itself is unchanged: the predict floor was already 0.25 on unattended-armed cameras.

### 10.5 One physical object, one episode, across classes

With the extra classes, three rules decide whether two detections are one object:

| Rule | Where | Across class groups? |
|---|---|---|
| Two near-identical boxes in **one frame** (IoU ≥ 0.5) are one object | `_dedupe` | **Yes.** A laptop bag reported as "laptop" and "handbag" in the same frame alerts once. |
| A detection inside an object already seen this frame is a fragment of it | the fragment rule in `_associate` | **No.** A phone lying on a laptop is a second object. |
| A detection continues an episode: through its track id, or as a nearby unseen episode of similar size within the recovery window | `_associate`, `_same_object_class` | **Only if the box coincides** with the episode's latest box (IoU ≥ 0.5). This covers one object labelled differently on different frames. Bags continue each other as before. |

The track-id rule was added after the replay showed that the object tracker is class-agnostic:
- JeztSort's second matching pass is by centroid distance (≤ 150 px).
- While a laptop was missed, a phone held 140 px away was handed the laptop's track id (camera_01, `abandoned_object_04`, t = 22.3 s).
- Without the rule, the phone would have continued (and could have "moved") the laptop's episode.
- The same replay showed one laptop shoulder bag called "handbag" and "laptop" on alternate frames, on an identical box (20 flips in 90 s). The coinciding-box exception keeps that as one episode.

For bags only (the deployed configuration), these rules are exactly the old ones: every bag class is in one group.

### 10.6 Replay (isolated, 2026-10-05)

**Footage.** 76 clips, 5.15 h in all, over three cameras. The laptop and phone content was verified on frames before any claim was made.
- 57 POC clips of 50–215 s (`Ai_inferencing/poc_app/clips*/{abandoned_object,phone_use}`):
  - laptops (open and closed) and bags left on desks at C101 and C201;
  - people holding and using phones;
  - phones lying on desks.
- 19 recordings of 40 s to 67 min (`zayed_ai_inferencing/zayed_test_recordings/`, `zayed_test_recordings/2026-09-12`, `2026-09-28`). These include a student working at a laptop at C101 (09-12), a 29-minute C201 session full of phone use (09-12), and bags carried through the rooms (09-26).
- There is no left-suitcase scene.

**Method.**
1. **Stage 1** (GPU, `--network none`). Every clip was run twice on the same frames: with the new routing, and with the f4e16a9 routing (`dump_unattended_inputs.py --separate --legacy-routing`).
2. **Stage 2** (CPU, `batch_eval.py`). The new `abandoned.py` ran on the new dumps, and the deployed f4e16a9 `abandoned.py` on the legacy dumps.

Each clip starts from empty state. An object already in place therefore alerts 32 s into the clip, and successive POC clips of one desk each alert once.

| Class | Detections | Raw track ids | Episodes | Reached UNATTENDED | Alerts | Physical objects alerted | Duplicate alerts | False alerts (frames reviewed) | Resolutions | Track ids per alerted episode |
|---|---|---|---|---|---|---|---|---|---|---|
| laptop | 68,266 | 394 | 105 | 23 | **13** | 3 (C101 desk in 11 POC clips; C101 09-12; C201) | **0** | **0** | 1 `attended`: the owner sat back down at it (09-12, alert 13:30:42, resolved 21 min later) | 1–6 |
| cell phone | 18,234 | 397 | 326 | 10 | **2** | 2 | **0** | **0**: one phone on a desk; one small dark folding item, a wallet or a phone in a folio case, reported as "Cell phone" | none inside the clips | 1 |
| backpack | 22,623 | 353 | 85 | 28 | 11 | – | 0 | as f4e16a9 | 1 `moved`, as f4e16a9 | 1–8 |
| handbag | 15,664 | 316 | 105 | 20 | 9 | – | 0 | as f4e16a9 | – | 1–7 |
| suitcase | 92 | 22 | 3 | 0 | 0 | – | – | – | – | – |

- **Phones in hands never alerted.** 11,134 phone boxes were attached to a person (the `MOBILE_PHONE_DETECTED` input). Of the 326 phone episodes, 10 had nobody near for the 2 s grace, and 2 stayed unattended for the 30 s dwell: both were items left on an empty desk.
- **Missed phone.** A black phone lying face-down behind the C101 laptop (15:52–16:06) was never detected as "cell phone": 0 detections in about 3,400 frames. It raised no alert. This is a limit of the detector's recall, not of the episode logic.
- **Bags unchanged.** In all 76 clips, the bag alerts and resolutions of the new code on the new routing equal those of f4e16a9 on the old routing (time ±2 s, class group, resolution): 20 alerts and 1 `moved` each.
- **Phone analytic input unchanged.** Over 184,612 frames:
  - The `phone_use` input is identical except for 48 frames.
  - In 36 of these the confidence differs by ≤ 0.011.
  - In 12, a box at 0.300–0.324 moves by 1 px or crosses the 0.30 floor, in both directions.
  - Two runs of the same legacy code on the same clips differ in the same way. This is TensorRT FP16 run-to-run variation, not the routing change.
- **Track handovers across classes.** Over the 37 C101 POC and 09-30 clips:
  - At the committed `TRACK_DIST_THRESH` 150 px, 24 track-class changes were a jump to another object.
  - At 250 px (an uncommitted edit in the production checkout), there were 69.
  - The alerts were the same with and without the §10.5 rule at both settings. The rule keeps each episode to one object.
- **No per-class threshold is needed.**
  - Phone boxes: short side median 62 px; 17 of 18,234 below the 24 px evidence minimum.
  - Laptop boxes: short side median 114 px.
  - Below 0.30 confidence: phones 13 %, handbags 10 %, laptops 9 %, backpacks 7 %.
  - No false alert was found at the shared 0.25 floor.

**Performance** (the same 184,612 frames per run, both runs concurrently on one GPU):

| Camera | predict ms p50 / p95, new vs f4e16a9 | tracker ms p50 / p95 | `update()` µs p50 / p95 | prediction failures |
|---|---|---|---|---|
| camera_01 | 11.01 / 12.38 vs 11.04 / 12.42 | 0.58 / 1.04 vs 0.58 / 1.02 | 6.0 / 13.0 vs 1.0 / 4.8 | 0 / 0 |
| camera_02 | 10.97 / 12.32 vs 10.97 / 12.29 | 0.59 / 1.03 vs 0.60 / 1.04 | 3.8 / 9.7 vs 1.7 / 4.4 | 0 / 0 |
| camera_03 | 10.90 / 12.23 vs 10.91 / 12.25 | 0.54 / 1.00 vs 0.52 / 0.97 | 2.6 / 28.6 vs 1.0 / 7.6 | 0 / 0 |

- YOLO inference is unchanged: the same engine, input size and predict floor, so the GPU load is the same. During the camera_01 run, each replay container used about 190 % CPU (mostly H.264 decoding), 1.31 GiB of RAM and 1,221 MiB of GPU memory, the same for both.
- The added per-frame cost is under 0.1 ms per camera against a 100 ms budget at 10 fps. The production figure of 29.7 fps in total (about 10 per camera, 0 failures) is not expected to change. It was not re-measured live, because production was not restarted.

### 10.7 Limitations specific to laptops and phones

1. **Detector recall.** A phone lying face-down or edge-on, or partly hidden by a laptop lid, may never be detected, and then it never alerts (§10.6).
2. **Class confusion.**
   - A wallet, a phone in a folio case or a power bank can be detected as "cell phone".
   - A laptop sleeve can be detected as "laptop" or "handbag". It still raises one alert (§10.5), named by its majority class.
3. **A phone on a desk counts as attended** while anyone is within the proximity radius (about 389 px at C101), the same rule as for bags.
4. **Phone evidence crops are small:** a median short side of 62 px before padding.
