"""
Zayed fall-policy tests: classroom postures, genuine falls, boundaries and lifecycle - and Event 32
reproduced from its own numbers.

    PYTHONPATH=<repo> python -m unittest -v test_fall_pose_zayed     (tests/fall/run_fall_tests.sh)

THE "BEFORE". With FALL_BASELINE_POLICY=<path to the policy production runs>, every scenario that
production gets wrong is ALSO run through that file, and the test asserts that production still
raises its false alarm there. A fixture that no longer reproduces the defect therefore fails
instead of passing quietly. run_fall_tests.sh sets it from `git show main:fall_pose_policy.py`.

Geometry is IMAGE pixels and time runs at 10 samples/s, the live pose rate. Synthetic fixtures are
labelled with the rule path they exercise; the Event 32 fixtures are the measured numbers.
"""
import importlib.util
import math
import os
import unittest

import fall_pose_policy as P

BASELINE_PATH = os.environ.get("FALL_BASELINE_POLICY")
B = None
if BASELINE_PATH:
    _spec = importlib.util.spec_from_file_location("fall_pose_policy_baseline", BASELINE_PATH)
    B = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(B)
needs_baseline = unittest.skipUnless(B, "FALL_BASELINE_POLICY not set - the 'before' is not checked")

DT = 0.1
SEEN, HIDDEN = 0.9, 0.05
HEAD_ARMS = (0, 1, 2, 3, 4, 7, 8, 9, 10)      # nose, eyes, ears, elbows, wrists


def body(shoulder, hip, knee=None, ankle=None, extra_visible=5):
    """COCO-17 keypoints from body centres. knee/ankle None = hidden (a desk, a chair).

    Each pair is placed symmetrically about its centre, so the policy's midpoints are exactly
    the centres given. `extra_visible` head/arm points are confident, the rest are not."""
    sx, sy = shoulder
    hx, hy = hip
    kx, ky = knee or (hx, hy + 80.0)
    ax, ay = ankle or (hx, hy + 160.0)
    k = [(sx, sy - 30.0)] * 5
    k += [(sx - 20.0, sy), (sx + 20.0, sy), (sx - 30.0, sy + 30.0), (sx + 30.0, sy + 30.0),
          (sx - 40.0, sy + 60.0), (sx + 40.0, sy + 60.0)]
    k += [(hx - 15.0, hy), (hx + 15.0, hy), (kx - 15.0, ky), (kx + 15.0, ky),
          (ax - 15.0, ay), (ax + 15.0, ay)]
    c = [HIDDEN] * 17
    for i in (5, 6, 11, 12):
        c[i] = SEEN
    if knee is not None:
        c[13] = c[14] = SEEN
    if ankle is not None:
        c[15] = c[16] = SEEN
    for i in HEAD_ARMS[:extra_visible]:
        c[i] = SEEN
    return k, c


def box_of(k, c):
    pts = [p for p, conf in zip(k, c) if conf >= P.KPT_CONF]
    return (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))


def frame(t, pose, bbox=None, track="T1"):
    k, c = pose
    return (t, track, k, c, bbox or box_of(k, c))


def run(module, frames, camera="camera_01"):
    """Feed frames [(t, track, k, c, bbox)] to a fresh policy -> (policy, events, last results)."""
    pol = module.FallPosePolicy()
    events, last = [], {}
    for t, track, k, c, bbox in frames:
        r = pol.observe(camera, track, k, c, bbox, observed_at=t, frame_width=2592, frame_height=1944)
        last[track] = r
        if r.event is not None:
            events.append((round(t, 3), track, r.event))
    return pol, events, last


def hold(start, seconds, pose, bbox=None, track="T1"):
    n = int(round(seconds / DT))
    return [frame(start + i * DT, pose, bbox, track) for i in range(n + 1)]


# ------------------------------------------------------------------ reference postures
#: Standing, whole body visible: 90 px torso, feet at y 638.
STANDING = body((500.0, 300.0), (500.0, 390.0), knee=(500.0, 470.0), ankle=(500.0, 550.0))
#: Standing behind a desk: the same body with knees and ankles hidden.
STANDING_BEHIND_DESK = body((500.0, 300.0), (500.0, 390.0), extra_visible=7)
#: Lying on the floor across the view, whole body visible (Dubai test 29's geometry, lowered).
LYING = body((400.0, 560.0), (560.0, 565.0), knee=(660.0, 568.0), ankle=(760.0, 570.0))


def standing_then(t_up, down_frames, up_pose=STANDING, track="T1"):
    """Two upright samples, then the given down frames."""
    return [frame(t_up - DT, up_pose, track=track), frame(t_up, up_pose, track=track)] + down_frames


# ------------------------------------------------------------------ Event 32 fixtures
#: Event 32 as production recorded it (event metadata, 2026-09-28 03:15:14.896Z, camera_01):
#: torso 66.35 deg, box [2222, 204, 2408, 382] (w/h 1.045), trunk axis unavailable, 12 confident
#: keypoints, down 0.323 s, last upright 1.115 s before. Knees at the position the NVR replay
#: measured for this person (visible, 0.64-0.94) and ankles hidden by the desk.
E32_BOX = (2222.0, 204.0, 2408.0, 382.0)
E32_SHOULDER = (2281.0, 248.0)
E32_HIP = (E32_SHOULDER[0] + 40.0 * math.tan(math.radians(66.35)), 288.0)     # torso 66.35 deg
E32_KNEE = (2356.0, 392.0)
#: The last upright sample the replay recorded before the bend (torso 38.3 deg, box w/h 0.862).
E32_UPRIGHT = body((2284.2, 211.9), (2347.0, 291.3), knee=(2324.7, 373.1), extra_visible=8)
E32_UPRIGHT_BOX = (2219.0, 174.0, 2382.0, 363.0)


def event32_frames(knees_visible=True, seconds=5.0):
    pose = (body(E32_SHOULDER, E32_HIP, knee=E32_KNEE, extra_visible=6) if knees_visible
            else body(E32_SHOULDER, E32_HIP, extra_visible=8))          # 12 confident either way
    return ([frame(10.0, E32_UPRIGHT, E32_UPRIGHT_BOX, "P-L76B-0113")]
            + hold(11.115, seconds, pose, E32_BOX, "P-L76B-0113"))


#: Event 32 as the offline replay of the NVR clip measured it, phase 0, every pose sample from
#: 07:15:45.38 to 07:15:48.28 (NVR time): (t, track box, shoulder, hip, knee, ankle | None,
#: (knee_l, knee_r, ankle_l, ankle_r) confidence, confident keypoints). The unmodified production
#: policy fires on this sequence at t = 1.799, exactly as it did live. Numbers only - no imagery.
REPLAY_BEND = [
    (0.000, (2223, 164, 2377, 363), (2288.2, 201.2), (2340.9, 291.6), (2317.3, 372.6), (2322.0, 442.1), (0.96, 0.75, 0.41, 0.12), 14),
    (0.100, (2221, 169, 2378, 363), (2285.6, 207.6), (2341.6, 291.3), (2318.0, 370.2), (2324.7, 439.4), (0.96, 0.77, 0.44, 0.15), 14),
    (0.200, (2219, 174, 2382, 363), (2284.2, 211.9), (2347.0, 291.3), (2324.7, 373.1), (2331.4, 443.1), (0.96, 0.71, 0.34, 0.1), 14),
    (0.300, (2218, 179, 2385, 363), (2282.2, 216.8), (2349.7, 290.1), (2330.1, 376.0), (2336.9, 446.2), (0.96, 0.69, 0.32, 0.08), 13),
    (0.400, (2217, 183, 2388, 364), (2280.2, 219.1), (2356.4, 288.6), (2336.2, 377.7), (2339.6, 449.6), (0.96, 0.73, 0.39, 0.12), 14),
    (0.500, (2215, 187, 2391, 367), (2282.9, 222.8), (2361.2, 285.5), (2345.6, 381.0), (2346.3, 453.6), (0.95, 0.78, 0.3, 0.13), 12),
    (0.600, (2214, 191, 2393, 372), (2280.2, 227.5), (2361.8, 286.7), (2352.4, 384.8), None, (0.95, 0.8, 0.3, 0.14), 11),
    (0.700, (2213, 193, 2395, 376), (2281.5, 230.2), (2365.2, 288.4), (2353.7, 388.6), None, (0.94, 0.75, 0.24, 0.09), 12),
    (0.800, (2213, 196, 2397, 377), (2280.2, 235.0), (2366.6, 284.8), (2353.1, 388.0), None, (0.95, 0.79, 0.25, 0.1), 11),
    (0.900, (2215, 198, 2399, 379), (2280.8, 238.3), (2368.6, 283.8), (2355.1, 385.6), None, (0.94, 0.75, 0.25, 0.1), 11),
    (1.000, (2215, 200, 2401, 380), (2280.2, 240.8), (2369.3, 285.1), (2356.4, 388.6), None, (0.94, 0.73, 0.24, 0.08), 11),
    (1.099, (2216, 201, 2403, 381), (2278.1, 243.1), (2368.6, 288.9), (2355.1, 389.6), None, (0.94, 0.64, 0.22, 0.06), 12),
    (1.199, (2218, 202, 2405, 381), (2280.8, 244.4), (2369.9, 288.8), (2356.4, 390.2), None, (0.94, 0.67, 0.23, 0.07), 11),
    (1.299, (2219, 203, 2405, 382), (2281.5, 246.0), (2372.0, 288.3), (2357.1, 391.5), None, (0.92, 0.66, 0.19, 0.06), 11),
    (1.399, (2220, 203, 2406, 382), (2282.2, 248.5), (2372.0, 288.0), (2357.1, 392.5), None, (0.92, 0.66, 0.19, 0.06), 11),
    (1.499, (2221, 203, 2408, 382), (2280.2, 248.1), (2372.6, 287.0), (2356.4, 392.9), None, (0.93, 0.67, 0.21, 0.06), 10),
    (1.599, (2221, 203, 2408, 382), (2280.8, 248.1), (2372.0, 286.9), (2356.4, 392.5), None, (0.94, 0.67, 0.21, 0.06), 11),
    (1.699, (2221, 204, 2409, 382), (2280.8, 246.5), (2372.6, 287.0), (2355.8, 392.5), None, (0.94, 0.67, 0.21, 0.05), 11),
    (1.799, (2222, 204, 2409, 382), (2280.8, 247.7), (2372.0, 286.5), (2355.8, 392.3), None, (0.94, 0.68, 0.21, 0.06), 10),
    (1.899, (2222, 204, 2409, 382), (2279.5, 247.1), (2371.3, 286.2), (2355.1, 391.8), None, (0.94, 0.69, 0.2, 0.05), 9),
    (1.999, (2222, 204, 2409, 382), (2280.2, 246.0), (2372.0, 286.5), (2355.1, 391.5), None, (0.94, 0.69, 0.21, 0.06), 11),
    (2.098, (2221, 204, 2410, 382), (2280.2, 247.1), (2372.0, 285.9), (2355.1, 390.7), None, (0.95, 0.71, 0.22, 0.06), 9),
    (2.198, (2221, 204, 2410, 382), (2280.2, 247.6), (2371.3, 285.8), (2355.1, 390.5), None, (0.95, 0.73, 0.22, 0.06), 9),
    (2.298, (2220, 204, 2409, 382), (2280.2, 247.6), (2371.3, 285.3), (2354.4, 390.3), None, (0.94, 0.72, 0.2, 0.06), 9),
    (2.398, (2239, 204, 2409, 382), (2280.2, 247.4), (2371.3, 285.6), (2353.1, 390.3), None, (0.94, 0.71, 0.2, 0.06), 9),
    (2.498, (2231, 204, 2409, 382), (2280.2, 247.3), (2370.6, 285.5), (2353.1, 389.6), None, (0.94, 0.71, 0.2, 0.06), 9),
    (2.598, (2227, 204, 2409, 380), (2282.2, 245.3), (2371.3, 285.2), (2351.7, 387.5), None, (0.95, 0.72, 0.22, 0.07), 11),
    (2.698, (2226, 205, 2410, 379), (2284.9, 244.1), (2369.3, 287.9), (2350.3, 385.3), None, (0.95, 0.71, 0.27, 0.08), 11),
    (2.798, (2227, 204, 2410, 378), (2292.3, 240.3), (2372.6, 288.4), (2352.4, 384.6), None, (0.93, 0.7, 0.23, 0.08), 11),
    (2.898, (2231, 202, 2412, 377), (2294.3, 240.4), (2377.3, 291.9), (2352.4, 380.7), (2361.2, 459.0), (0.97, 0.82, 0.52, 0.18), 14),
]


def replay_frames(t0=100.0):
    out = []
    for t, bbox, shoulder, hip, knee, ankle, _conf, n in REPLAY_BEND:
        base = 6 + (2 if ankle else 0)
        pose = body(shoulder, hip, knee=knee, ankle=ankle, extra_visible=max(0, n - base))
        out.append(frame(t0 + t, pose, tuple(float(v) for v in bbox), "P-NDQE-0001"))
    return out


# ================================================================== classroom postures
class ClassroomPostures(unittest.TestCase):

    def test_01_standing_never_falls(self):
        _, events, last = run(P, hold(0.0, 20.0, STANDING))
        self.assertEqual(events, [])
        self.assertEqual(last["T1"].state, P.POSE_CANDIDATE)

    def test_02_seated_upright_never_falls(self):
        # Seated: thigh forward to the knee, lower legs under the desk.
        seated = body((500.0, 400.0), (505.0, 500.0), knee=(575.0, 505.0))
        _, events, _ = run(P, standing_then(1.0, hold(2.0, 20.0, seated)))
        self.assertEqual(events, [])

    def test_03_event32_exact_numbers_bending_over_desk_is_not_a_fall(self):
        frames = event32_frames(knees_visible=True)
        m = P.pose_measurements(frames[1][2], frames[1][3], E32_BOX)
        self.assertAlmostEqual(m["torso_angle"], 66.35, places=2)
        self.assertAlmostEqual(m["body_aspect_ratio"], 1.045, places=3)
        self.assertIsNone(m["trunk_angle"])
        self.assertEqual(m["confident_keypoints"], 12)
        _, events, last = run(P, frames)
        self.assertEqual(events, [], "Event 32 must not raise a fall")
        self.assertEqual(last["P-L76B-0113"].state, P.POSE_CANDIDATE)
        self.assertIn("knee->shoulder axis", last["P-L76B-0113"].reason)

    @needs_baseline
    def test_03_before_production_policy_raises_event32(self):
        _, events, _ = run(B, event32_frames(knees_visible=True))
        self.assertEqual(len(events), 1, "the fixture must reproduce production's false alarm")
        t, _, ev = events[0]
        md = ev.metadata()
        self.assertEqual((md["torso_angle"], md["aspect_ratio"], md["trunk_angle"]), (66.35, 1.045, None))
        self.assertEqual(md["confident_keypoints"], 12)
        self.assertAlmostEqual(md["seconds_since_upright"], 1.115, places=3)
        self.assertGreaterEqual(md["pose_duration"], 0.3)
        self.assertLess(md["pose_duration"], 0.4)

    def test_04_event32_numbers_with_knees_hidden_too_is_not_a_fall(self):
        _, events, last = run(P, event32_frames(knees_visible=False))
        self.assertEqual(events, [])
        r = last["P-L76B-0113"]
        self.assertIn("no knee or ankle visible", r.reason)
        self.assertLess(r.measurements["hip_drop_ratio"], 0.1)

    @needs_baseline
    def test_04_before_production_policy_raises_it_too(self):
        _, events, _ = run(B, event32_frames(knees_visible=False))
        self.assertEqual(len(events), 1)

    def test_05_event32_replayed_sequence_is_not_a_fall(self):
        _, events, last = run(P, replay_frames())
        self.assertEqual(events, [])
        self.assertEqual(last["P-NDQE-0001"].state, P.POSE_CANDIDATE)

    @needs_baseline
    def test_05_before_production_policy_fires_where_it_fired_live(self):
        # Live and in the replay it confirmed at t = 1.799 (NVR 07:15:47.178), the fourth or
        # fifth sample of a down run that began at 1.399 - which of the two depends on 0.3 s
        # against sample spacing to the millisecond.
        _, events, _ = run(B, replay_frames())
        self.assertEqual(len(events), 1)
        self.assertTrue(101.69 <= events[0][0] <= 101.81, events[0][0])

    def test_06_reaching_down_in_a_wide_box_is_not_a_fall(self):
        # Waist bend to the floor with the arms out: torso ~86 deg in a WIDER-than-tall box, but
        # the ankle axis stays upright (31 deg). Dubai's ground rule had no lower-body check.
        reach = body((400.0, 420.0), (560.0, 410.0), knee=(566.0, 560.0), ankle=(570.0, 700.0))
        wide = (280.0, 380.0, 640.0, 700.0)
        _, events, last = run(P, standing_then(1.0, hold(1.5, 5.0, reach, wide)))
        self.assertEqual(events, [])
        self.assertIn("ankle->shoulder axis", last["T1"].reason)

    @needs_baseline
    def test_06_before_production_policy_raises_it(self):
        reach = body((400.0, 420.0), (560.0, 410.0), knee=(566.0, 560.0), ankle=(570.0, 700.0))
        _, events, _ = run(B, standing_then(1.0, hold(1.5, 5.0, reach, (280.0, 380.0, 640.0, 700.0))))
        self.assertEqual(len(events), 1)

    def test_07_sitting_on_the_floor_is_not_a_fall(self):
        floor_sit = body((500.0, 470.0), (505.0, 560.0), knee=(590.0, 570.0), ankle=(680.0, 575.0))
        _, events, _ = run(P, standing_then(1.0, hold(1.5, 10.0, floor_sit)))
        self.assertEqual(events, [])

    def test_08_kneeling_bent_over_is_not_a_fall(self):
        # Kneeling to look under a desk: torso horizontal, knees on the floor (visible), feet
        # hidden behind. Knee axis 60 deg - the body below the hips is not lying down.
        kneel = body((400.0, 520.0), (540.0, 520.0), knee=(540.0, 600.0))
        _, events, last = run(P, standing_then(1.0, hold(1.5, 5.0, kneel)))
        self.assertEqual(events, [])
        self.assertIn("knee->shoulder axis", last["T1"].reason)

    @needs_baseline
    def test_08_before_production_policy_raises_it(self):
        kneel = body((400.0, 520.0), (540.0, 520.0), knee=(540.0, 600.0))
        _, events, _ = run(B, standing_then(1.0, hold(1.5, 5.0, kneel)))
        self.assertEqual(len(events), 1)

    def test_09_seated_student_slumping_onto_the_desk_is_not_a_fall(self):
        # Legs under the desk, so no knee and no ankle: the torso tips to 72 deg in a wide box
        # (arms on the desk), but the hips stay on the chair.
        seated = body((500.0, 400.0), (505.0, 500.0), extra_visible=7)
        slump = body((410.0, 470.0), (505.0, 500.0), extra_visible=7)
        frames = [frame(1.0, seated, (440.0, 330.0, 570.0, 510.0))] + hold(
            1.5, 10.0, slump, (330.0, 420.0, 570.0, 520.0))
        _, events, last = run(P, frames)
        self.assertEqual(events, [])
        self.assertIn("leaning over, not on the floor", last["T1"].reason)

    @needs_baseline
    def test_09_before_production_policy_raises_it(self):
        seated = body((500.0, 400.0), (505.0, 500.0), extra_visible=7)
        slump = body((410.0, 470.0), (505.0, 500.0), extra_visible=7)
        frames = [frame(1.0, seated, (440.0, 330.0, 570.0, 510.0))] + hold(
            1.5, 10.0, slump, (330.0, 420.0, 570.0, 520.0))
        _, events, _ = run(B, frames)
        self.assertEqual(len(events), 1)

    def test_10_squat_behind_a_desk_drops_the_hips_but_not_the_shoulders(self):
        # Side view at 180 px/m, floor at y 552 (STANDING_BEHIND_DESK: shoulders 1.4 m, hips
        # 0.9 m). A deep squat leaning 70 deg forward: hips at 0.35 m (down 1.1 torsos),
        # shoulders at 0.52 m (down 1.76) - the upper body stays well above the floor.
        squat = body((415.4, 458.2), (500.0, 489.0), extra_visible=7)
        frames = standing_then(1.0, hold(1.5, 5.0, squat, (370.0, 420.0, 560.0, 520.0)),
                               up_pose=STANDING_BEHIND_DESK)
        _, events, last = run(P, frames)
        self.assertEqual(events, [])
        self.assertGreaterEqual(last["T1"].measurements["hip_drop_ratio"], P.HIP_DROP_MIN_RATIO)
        self.assertLess(last["T1"].measurements["shoulder_drop_ratio"], P.SHOULDER_DROP_MIN_RATIO)


# ================================================================== genuine falls
class GenuineFalls(unittest.TestCase):

    def test_11_full_body_fall_fires_once_after_the_hold(self):
        pol, events, _ = run(P, standing_then(1.0, hold(1.5, 6.0, LYING)))
        self.assertEqual(len(events), 1)
        t, _, ev = events[0]
        self.assertGreaterEqual(t - 1.5, P.CONFIRM_SECONDS - 1e-9)
        self.assertLess(t - 1.5, P.CONFIRM_SECONDS + DT + 1e-9)
        self.assertEqual(ev.metadata()["lower_body_evidence"], "ankle")

    def test_12_fall_with_the_feet_hidden_but_the_knees_visible_fires(self):
        lying = body((400.0, 560.0), (560.0, 565.0), knee=(660.0, 568.0))
        _, events, _ = run(P, standing_then(1.0, hold(1.5, 5.0, lying)))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][2].metadata()["lower_body_evidence"], "knee")

    def test_13_fall_behind_a_desk_with_no_legs_visible_fires_on_the_landmark_drop(self):
        # From standing behind the desk (hips y 390, shoulders 300, torso 90 px) to the floor:
        # hips drop 1.8 torsos, shoulders 2.7; the upper body lies across the view.
        down = body((420.0, 543.0), (520.0, 552.0), extra_visible=7)
        frames = standing_then(1.0, hold(1.5, 5.0, down, (380.0, 500.0, 560.0, 580.0)),
                               up_pose=STANDING_BEHIND_DESK)
        _, events, _ = run(P, frames)
        self.assertEqual(len(events), 1)
        md = events[0][2].metadata()
        self.assertEqual(md["lower_body_evidence"], "none")
        self.assertGreaterEqual(md["hip_drop_ratio"], P.HIP_DROP_MIN_RATIO)
        self.assertGreaterEqual(md["shoulder_drop_ratio"], P.SHOULDER_DROP_MIN_RATIO)

    def test_14_prone_on_the_arms_ground_state_still_fires(self):
        # Dubai test 37's fall3 geometry (torso 72, trunk 75, box wider than tall): the ground
        # rule's reason for existing must survive the lower-body check.
        prone = body((400.0, 480.0), (520.0, 520.0), knee=(610.0, 545.0), ankle=(700.0, 560.0))
        _, events, _ = run(P, standing_then(1.0, hold(1.5, 5.0, prone)))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][2].metadata()["down_rule"], "ground")


# ================================================================== boundaries
class Boundaries(unittest.TestCase):

    def _landmark_drop(self, hip_drop, shoulder_drop):
        """No legs visible; hips and shoulders placed at exact drops from STANDING_BEHIND_DESK,
        torso 80 deg."""
        hip_y = 390.0 + hip_drop * 90.0
        shoulder_y = 300.0 + shoulder_drop * 90.0
        hip = (520.0, hip_y)
        shoulder = (520.0 - abs(hip_y - shoulder_y) * math.tan(math.radians(80.0)), shoulder_y)
        down = body(shoulder, hip, extra_visible=7)
        frames = standing_then(1.0, hold(1.5, 3.0, down, (shoulder[0] - 40, min(hip_y, shoulder_y) - 60,
                                                          hip[0] + 40, max(hip_y, shoulder_y) + 40)),
                               up_pose=STANDING_BEHIND_DESK)
        return run(P, frames)[1]

    def test_15_hip_drop_threshold(self):
        s = P.SHOULDER_DROP_MIN_RATIO + 0.5
        self.assertEqual(self._landmark_drop(P.HIP_DROP_MIN_RATIO - 0.05, s), [])
        self.assertEqual(len(self._landmark_drop(P.HIP_DROP_MIN_RATIO + 0.05, s)), 1)

    def test_16_shoulder_drop_threshold(self):
        h = P.HIP_DROP_MIN_RATIO + 0.3
        self.assertEqual(self._landmark_drop(h, P.SHOULDER_DROP_MIN_RATIO - 0.05), [])
        self.assertEqual(len(self._landmark_drop(h, P.SHOULDER_DROP_MIN_RATIO + 0.05)), 1)

    def test_17_confirmation_needs_the_full_hold(self):
        short = standing_then(1.0, hold(1.5, P.CONFIRM_SECONDS - 0.1, LYING)
                              + hold(1.5 + P.CONFIRM_SECONDS + 0.1, 3.0, STANDING))
        self.assertEqual(run(P, short)[1], [], "down for less than the hold, then up")
        _, events, _ = run(P, standing_then(1.0, hold(1.5, P.CONFIRM_SECONDS, LYING)))
        self.assertEqual(len(events), 1)

    def test_18_one_non_down_sample_restarts_the_hold(self):
        half = P.CONFIRM_SECONDS / 2.0
        frames = standing_then(1.0, hold(1.5, half, LYING) + [frame(1.5 + half + DT, STANDING)]
                               + hold(1.5 + half + 2 * DT, half, LYING))
        self.assertEqual(run(P, frames)[1], [])


# ================================================================== lifecycle
class Lifecycle(unittest.TestCase):

    def test_19_lying_from_first_sight_is_not_a_fall(self):
        self.assertEqual(run(P, hold(1.0, 10.0, LYING))[1], [])

    def test_20_upright_too_long_ago_is_not_a_fall(self):
        frames = standing_then(1.0, hold(1.0 + P.UPRIGHT_WINDOW_SECONDS + 0.5, 5.0, LYING))
        self.assertEqual(run(P, frames)[1], [])

    def test_21_one_event_per_fall_then_cooldown_then_a_later_fall(self):
        first = standing_then(1.0, hold(1.5, 70.0, LYING))
        t = 1.5 + 70.0 + DT
        second = [frame(t + i * DT, STANDING) for i in range(5)] + hold(t + 5 * DT, 5.0, LYING)
        _, events, _ = run(P, first + second)
        self.assertEqual(len(events), 2, [e[0] for e in events])

    def test_22_tracks_are_independent(self):
        bender = event32_frames(knees_visible=True)
        faller = [(t, "T2", k, c, b) for t, _, k, c, b in standing_then(10.0, hold(10.5, 5.0, LYING))]
        frames = sorted(bender + faller, key=lambda f: f[0])
        _, events, _ = run(P, frames)
        self.assertEqual([track for _, track, _ in events], ["T2"])

    def test_23_event_metadata_explains_the_decision(self):
        _, events, _ = run(P, standing_then(1.0, hold(1.5, 3.0, LYING)))
        md = events[0][2].metadata()
        for key in ("detection_method", "torso_angle", "pose_duration", "confident_keypoints",
                    "trunk_axis_available", "down_rule", "lower_body_evidence", "lower_body_axis_angle",
                    "knee_axis_angle", "hip_drop_ratio", "shoulder_drop_ratio", "hip_drop_threshold",
                    "shoulder_drop_threshold", "policy_revision", "pose_confirm_seconds"):
            self.assertIn(key, md)
        self.assertEqual(md["pose_confirm_seconds"], P.CONFIRM_SECONDS)
        self.assertEqual(events[0][2].scope(), "fall:T1")


if __name__ == "__main__":
    unittest.main()
