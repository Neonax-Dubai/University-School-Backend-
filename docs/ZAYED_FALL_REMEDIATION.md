# Fall detection: remediation of the Event 32 false positive

| Check | Status |
|---|---|
| Algorithm validation | **PASS**: unit scenarios, the Dubai regression suite, and an isolated replay of the real Event 32 footage |
| C101 staged fall validation | **PENDING**: no genuine fall has ever been recorded in C101 (procedure in §9) |
| Production deployment | **DEPLOYED** 2026-09-29 at 14:00:43Z, with the operator's approval. Branch `zayed-poc-remediation` @ `7e0512b` is checked out in the production checkout, and a 48 h observation is running (§10). |

Branch `zayed-poc-remediation`. The only code change is in `fall_pose_policy.py`; `fall_pose_adapter.py`, `zayed_inference.py` and the dashboard are untouched.

## 1. The defect

Event 32 is `ZEV-0E13033A6A964072`:
- camera_01, at 2026-09-28 03:15:14.896Z (07:15:14 Asia/Dubai);
- track `P-L76B-0113`;
- it raised the critical alarm **"Person Fell"**, which is still open and left for operator review.

The person was standing and bending over a desk. Production recorded:

| Quantity | Value | Rule it passed |
|---|---|---|
| torso | 66.35° | ground rule needs ≥ 65° |
| box width/height | 1.045 | ground rule needs ≥ 1.0 |
| trunk axis | unavailable | — |
| confident keypoints | 12 | — |
| time held down | 0.323 s | needs 0.3 s |
| last upright | 1.115 s earlier | needs within 3 s |

## 2. Reproduction: offline and isolated

**Footage.** It was extracted read-only with `recordings_viewer/tools/extract_clip.py`, run inside the `recordings-viewer` container. The NVR credentials never left that container. Clips are stored in `zayed_test_recordings/2026-09-28/camera_01/` (mode 0700/0600, outside every repository):

| Clip (NVR time) | SHA-256 | Contents |
|---|---|---|
| 07:14:55–07:15:35 | `7cff60c2…991f` | the person walking in, still upright when the clip ends |
| 07:15:25–07:16:05 | `ad155a04…a482` | the bend |

**The NVR clock runs about 32 s ahead of the server.** The first clip, cut at the server's 07:15, ended before the bend. Three independent sources line up:
- The replayed event matches production's box to the pixel at NVR 07:15:47.18, against server 07:15:14.90.
- camera_01's occupancy count went from 0 to 1 at server 03:14:50–03:15:00Z.
- Re-ID first logged the track at 03:14:53Z.

This offset affects any correlation between dashboard events and NVR footage, and is recorded as a separate item.

**Method.** `tests/fall/run_fall_replay.sh` runs `tests/fall/replay_fall_clip.py`:
- **Isolation:** a `--network none` container with the checkout, engines and clip mounted read-only. The output directory is the only thing written.
- **Same chain as production:** YOLO26-L TensorRT at 640 (confidence floor 0.25, person ≥ 0.45), then the same `TrackManager` and EMA smoothing (0.45), then `FallPoseAdapter` (candidate gate, 2 s hot window, one-to-one association) on the shared YOLO26L-pose runtime at 960.
- **Sampling:** frames at the live 10/s, in each of the three phases of the 30 fps stream.
- **Comparison:** every pose observation goes to **two** policies with identical arguments. The baseline is `git show main:fall_pose_policy.py`, the file production runs; the other is this branch's.
- **Nothing reaches production:** nothing is published to MediaMTX, and nothing touches PostgreSQL, Qdrant, Redis, SeaweedFS, the dashboard or the outbox.

**Result with the baseline policy.** It fires once in each phase, at NVR 07:15:47.11–47.18:

| | Replay | Production |
|---|---|---|
| Box | `[2222,204,2409,382]` | `[2222,204,2408,382]` |
| Torso | 66.5–66.9° | 66.35° |
| Width/height | 1.051–1.056 | 1.045 |
| Trunk axis | none | none |

The defect is reproduced, not merely resembled.

## 3. Root cause, measured on the bend (all three phases)

| Quantity | Measured during the bend | What a person lying down shows |
|---|---|---|
| Knee confidence | **0.64–0.97: knees visible** | — |
| Ankle confidence | 0.05–0.27: hidden by the desk, so no trunk axis | — |
| Knee→shoulder axis | **16.6–28.2°** from vertical | 70–90° |
| Hip drop vs. last upright pose | **−0.074 to +0.028** torso lengths | ~1.6 |
| Shoulder drop | +0.32 to +0.36 torso lengths | ~2.4 |
| Old rule's continuous "down" run | **0.9 s, 1.3 s, 1.2 s** | seconds |

Two gaps in the Dubai policy combined:

1. **The `horizontal` rule fell back to the torso alone** when the ankle-based trunk axis was missing. Dubai test 32 pinned this on purpose: "requiring an axis that cannot be measured would suppress real falls behind desks".
2. **The `ground` rule never looked below the hips.** It relied on the box being wider than tall, reasoning that "a bend keeps an upright-shaped box". The box, however, is the *track's* box, and a desk that hides the lower legs cuts it off at the knees. Every bend over a desk becomes a wide box.

**A longer hold alone would not have fixed it.** The bend held the old down state for up to 1.3 s continuously, so a 1.0 s hold would still have fired in two of the three phases.

## 4. The change

| Rule | Before (Dubai) | After (Zayed) |
|---|---|---|
| horizontal | torso ≥ 75° and (trunk ≥ 75° **or trunk unavailable**) | torso ≥ 75° and **lower body down**, judged against 75° |
| ground | (torso ≥ 65° or trunk ≥ 75°) and box w/h ≥ 1.0 | the same **and lower body down**, judged against 65° |
| lower body down | — | the **ankle** axis ≥ threshold. If the ankles are hidden, the **knee→shoulder** axis ≥ threshold. With neither visible, the **hips have dropped ≥ 1.0** and the **shoulders ≥ 2.0** upright torso lengths since the last upright pose. With no upright reference: no. |
| hold | 0.3 s | **1.0 s** |
| transition (upright within 3 s) | required | required, unchanged |

Where the ankles are visible, `horizontal` is unchanged, apart from the hold. `ground` now also needs the ankle axis ≥ 65°, which it never checked before. As a result a reach to the floor with the arms out no longer passes: the box is wide but the legs are upright, with an ankle axis around 31° (test 06, which production would flag).

**Why each threshold has its value:**
- **Knee axis (65°/75°).** It is the same horizontality the rule already asks of the torso, so no new angle is introduced.
- **Hip drop 1.0.** On the floor, the hips drop about 1.6 torso lengths. A bend keeps the hips at standing height; Event 32's moved by about 0.
- **Shoulder drop 2.0.** On the floor, the shoulders drop about 2.4. Sitting down and slumping onto the desk gives about 1.2, kneeling about 1.6, and a deep squat leaning forward 70° about 1.8. A first draft used 1.5, which that squat would have passed.
- **Hold 1.0 s.** That is ten samples at the live 10/s, and it fits inside the adapter's 2 s candidate hold. Longer is not free: a single non-down sample restarts the hold, and the Event 32 footage shows such a one-sample dip within a sustained posture.

**Other details:**
- All four values can be overridden by environment variable: `FALL_POSE_CONFIRM_SECONDS`, `FALL_POSE_HIP_DROP_RATIO`, `FALL_POSE_SHOULDER_DROP_RATIO` and the existing angle variables. None is set in production.
- Event metadata gains `down_rule`, `lower_body_evidence` (ankle, knee or none), `lower_body_axis_angle`, `knee_axis_angle`, `hip_drop_ratio`, `shoulder_drop_ratio`, both thresholds and `policy_revision`. All existing fields are unchanged.
- The rejection reasons say which evidence was missing.

## 5. Tests: before and after

| Suite | Before (main) | After (this branch) |
|---|---|---|
| `tests/fall/run_fall_tests.sh`: Zayed scenarios | — (new) | **29/29** |
| Dubai `test_fall_pose.py`, unmodified | 36/38 (2 need missing data) | 29/38 as shipped; **35/38 with the hold set back to 0.3 s** |
| `tests/unit`, `camera_health`, `resilience`, `internal_face` | 245/245 | **245/245** |

**Zayed scenarios** (`tests/fall/test_fall_pose_zayed.py`, at 10 samples/s):

| Group | Must not fire | Must fire |
|---|---|---|
| Classroom postures | standing; seated; **Event 32's exact numbers** (knees visible); Event 32's numbers with knees hidden; **the replayed Event 32 sequence** (30 real samples); reaching down in a wide box; sitting on the floor; kneeling bent over; a seated student slumping onto the desk; a deep squat behind a desk | — |
| Genuine falls | — | a full-body fall (once, after exactly the hold); knees visible with ankles hidden; no legs visible, on the landmark drop; prone and propped on the arms (Dubai's fall3 geometry, ground rule) |
| Boundaries | just under each threshold | just over it |

The boundary group also checks that a hold shorter than 1.0 s does not fire, and that one non-down sample restarts the hold. The lifecycle group checks that there is no event:
- without an upright phase;
- when the upright pose was too long ago;
- more than once per fall (then a cooldown, then a second fall fires);
- for a bending track sitting next to a falling one (tracks are independent);
- without the metadata fields.

**Before checks.** With `FALL_BASELINE_POLICY` set, six tests also run the production policy and assert that it **still raises** each false alarm. They cover Event 32 (both variants and the replayed sequence), the reach, the kneel and the seated slump, so the fixtures are proven to reproduce the defect.

**Dubai regression.** The runner classifies every failure and fails on anything unclassified:

| Tests | Why they fail |
|---|---|
| 2, 10 | Missing test data: `pose_capture.json` is in no repository. Unchanged from baseline. |
| **32** | **Intended change.** It pins the torso-only fallback, the Event 32 mechanism. |
| 4, 9, 18, 19, 22, 34 | Pinned to the 0.3 s hold. They observe for only 0.76–0.96 s, or assert `CONFIRM_SECONDS == 0.3`. With the hold at 0.3 s all six pass, and test 34 shows the fall still firing at 1.0 s. |

Every other Dubai behaviour still passes:
- true falls, and the prone ground state;
- the waist bend, the seated posture and the reclined chair;
- the transition requirement and expiry, cooldown, and pose association.

## 6. Isolated replay of Event 32: before and after

| Phase | Pose observations | Baseline (production) | This branch |
|---|---|---|---|
| 0 | 77 | 1 event at 07:15:47.178 | **0 events**; 77/77 samples not down |
| 1 | 81 | 1 event at 07:15:47.112 | **0 events**; 81/81 not down |
| 2 | 77 | 1 event at 07:15:47.145 | **0 events**; 77/77 not down |

- The earlier clip (07:14:55–07:15:35) has 0 events with both policies.
- At the sample where production fired, this branch recorded: *"torso 66.9 deg, but the knee->shoulder axis is 27.4 deg: the body below the hips is upright - bent or seated, not lying"*.
- During the bend the lower-body axis stayed between 16.6° and 28.2°. The rule needs 65°, a margin of about 37°.
- The raw per-sample records are kept beside the clips as `replay_after_*/observations_phase*.jsonl`. They are not committed.

## 7. Performance

- **Policy cost.** `observe()` takes 4.0 µs per sample, against 3.5 µs before (median of 400 runs over the replayed sequence). Pose samples exist only for candidate tracks, at most about 10/s per camera, which makes the added load on the order of 15 µs of CPU per second.
- **Nothing else changes.** Detection, decoding, pose inference, the adapter and the main loop are untouched.
- **Production during the offline replays** stayed at 29.66–29.72 fps. Inference p50 rose briefly from about 16 ms to 18–21 ms while the replay container shared the GPU.
- **Pre-deployment baseline** for the post-deployment comparison:

  | Metric | Value |
  |---|---|
  | Throughput | 29.7 fps total, 9.9 per camera |
  | Inference p50 / end-to-end p50 | 16 ms / 41–50 ms |
  | Predict failures | 0 |
  | GPU | ~21%; 4.3 GiB inference plus 3 × 336 MiB NVDEC |
  | CPU / RAM | about 3.6 cores / 4.5 GiB |

- **The post-deployment measurement is pending** until deployment is approved.

## 8. Remaining uncertainty

- **No real fall has ever been observed in C101.** Recall rests on synthetic sequences and Dubai's fixtures, and the thresholds on body geometry plus one real bend.
- **Falls with no legs visible** rely on image-space drops:
  - A fall straight away from the camera shows less drop than one across its view.
  - A collapse in which the torso stays upright while the knees give way can update the upright reference part-way down. Both can reduce recall on this path only.
  - A fall straight away from a steep camera is hard for any image-space rule, because the body is foreshortened along the line of sight.
- **Pose interruptions.** A single non-down sample restarts the 1.0 s hold, so a genuine fall with flickering pose may confirm later than it would have at 0.3 s.
- **Real footage exercised only the knee path.** In Event 32 the knees were visible, so the replay tested the knee rule; the no-legs path has been tested only on synthetic sequences.
- **Adapter gate unchanged.** The candidate gate (box width/height ≥ 0.75, 2 s hold) is unchanged, so poses are still only evaluated for wide-ish boxes.

## 9. C101 staged fall validation: PENDING

This is not claimed until it has been done. It needs:
- a consenting adult volunteer;
- a gym mat;
- a spotter;
- a supervisor who can stop the test.

Run it with the change deployed, with fall detection in **shadow mode** (`FALL_POSE_SHADOW_MODE=1`) so no real alarm reaches the dashboard, or with an operator warned in advance.

| # | Scenario | Expected |
|---|---|---|
| 1 | stand, then walk the aisle | no event |
| 2 | sit at a desk, then slump onto it for 2 minutes | no fall |
| 3 | bend over a desk as in Event 32, for 5 s, at several desks | no fall |
| 4 | pick a bag up off the floor, legs visible and legs behind a desk | no fall |
| 5 | kneel, squat, sit on the floor | no fall |
| 6 | controlled fall onto the mat across camera_01's view, legs visible | one fall within about 1–2 s of landing |
| 7 | the same between desk rows, legs hidden | one fall (no-legs path) |
| 8 | the same, falling away from and towards camera_01 | record the result: this is the known weak direction |

For each run, record:
- the event, or its absence;
- the `down_rule`, `lower_body_evidence`, drops and axis from the metadata or the `[FALL-POSE]` shadow log;
- the latency from landing.

## 10. Deployment (done 2026-09-29, with the operator's approval)

The chosen workflow is **branches in the production checkout**, not separate worktrees and not a merge. `main` stays at `ba1a578`, and nothing is pushed.

1. **Checked before starting**
   - MediaMTX publishers were exactly the three relays (IPs matched), with no publisher change in the previous 90 min.
   - `zayed-inference` had last restarted at 12:02Z.
   - The production checkout was clean apart from the owner's untracked `tst.txt`.
   - The only other active session was running Django tests in an isolated network (`ztest-net-*`), mounting `zayed_dashboard/` read-only. It was unaffected by an inference restart.
2. **Recorded before** (`ops/phase1_fall_observation/pre_deploy_snapshot.json`, 13:59:38Z):
   - `main` @ `ba1a578`; 29.69 fps, 9.9 per camera; inference p50/p95 16.1/25.0 ms; end-to-end p50/p95 41.5–45.6 / 71–84 ms; 0 predict failures.
   - GPU 13%, 4,661 MiB plus 3 × 336 MiB; CPU 3.35 cores; RAM 4.41 GiB.
   - One open tamper incident on camera_02, since 13:01:06Z, raised during a lighting change that camera_01 and camera_03 correctly absorbed.
3. **Switched the branch.** The remediation worktree was detached first, to free the branch.

   ```bash
   git -C /home/stack/Zayed_University/zayed_ai_inferencing checkout zayed-poc-remediation   # HEAD 7e0512b
   ```

   `git diff main..zayed-poc-remediation` changes one runtime Python file, `fall_pose_policy.py`; the rest is `docs/` and `tests/fall/`.
4. **Restarted inference only**, with `docker restart zayed-inference`:

   | Time (Z) | Event |
   |---|---|
   | 14:00:43.4 | Restart issued |
   | 14:00:47.3 | Container running (graceful stop about 4 s, no SIGKILL) |
   | 14:00:55 | All three cameras connected |
   | 14:01:14.7 | Inference running |
   | 14:01:25 | First analysed frames |
   | 14:02:15.9 | Full rate, 29.67 fps, 99 frames per 10 s on every camera |

   Analytics were fully back **about 92 s** after the restart command.

   During warm-up, MediaMTX dropped the camera_01 reader with `write … i/o timeout` at 14:01:32. It reconnected in about 7 s, and camera health logged signal loss raise and recover as a diagnostic, with no event. The 12:02Z restart on the old code showed the same transient on all three cameras, so it is **pre-existing** and logged as a separate item.
5. **Verified after** (`post_deploy_snapshot.json`, 14:04:31Z):
   - The container is healthy with 0 errors in 200 log lines.
   - The deployed file sets hold 1.0 s, hip 1.0, shoulder 2.0. It was checked out 40 s before the process started.
   - 29.60–29.67 fps, 9.9 per camera; inference p50/p95 16.4/25.9 ms; end-to-end p50/p95 41.9–46.6 / 75–82 ms; predict, camera and fall-worker errors 0.
   - CPU 3.36 cores.
   - GPU 7%, 2,055 MiB, and RAM 3.16 GiB are lower only because the pose and face models load lazily on the first person; they are to be re-compared during the observation.
   - 0 FALL events since deployment. Event 32 and its critical alarm are untouched.
6. **Observation.** `ops/phase1_fall_observation/observe.sh` takes a read-only snapshot every 5 min until 2026-10-01T14:00:43Z; `summarize.py --db` gives the report. The camera_02 tamper incident's in-memory state was lost with the restart, so its dashboard alarm stays open for operator review.

## 11. Rollback

```bash
git -C /home/stack/Zayed_University/zayed_ai_inferencing checkout main    # back to ba1a578
docker restart zayed-inference
```

A partial runtime rollback, without code, is `FALL_POSE_CONFIRM_SECONDS=0.3` in the compose environment. It restores the 0.3 s hold only; the lower-body rules stay.
