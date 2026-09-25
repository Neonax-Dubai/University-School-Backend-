import unittest

import numpy as np

import helpers  # noqa: F401

import sleeping as S


def skeleton(head_rise, shoulders_y=200.0, width=60.0, face=True, conf=0.9):
    """17 keypoints: shoulders at shoulders_y, head `head_rise` shoulder-widths above them."""
    k = np.zeros((17, 2))
    c = np.zeros(17)
    k[S.L_SHOULDER] = (100 + width, shoulders_y)
    k[S.R_SHOULDER] = (100, shoulders_y)
    c[S.L_SHOULDER] = c[S.R_SHOULDER] = conf
    if face:
        k[S.NOSE] = (100 + width / 2, shoulders_y - head_rise * width)
        c[S.NOSE] = conf
    return k, c


class HeadDownTests(unittest.TestCase):
    def test_upright_is_not_head_down(self):
        k, c = skeleton(0.7)
        self.assertEqual(S.head_down(k, c, [80, 100, 190, 400])[0], False)

    def test_head_at_shoulder_level_is_head_down(self):
        k, c = skeleton(0.05)
        self.assertEqual(S.head_down(k, c, [80, 150, 190, 400])[0], True)

    def test_face_hidden_uses_the_top_of_the_box(self):
        k, c = skeleton(0.0, face=False)
        self.assertEqual(S.head_down(k, c, [80, 195, 190, 400])[0], True, "nothing above the shoulders")
        self.assertEqual(S.head_down(k, c, [80, 140, 190, 400])[0], False, "a head is above them")

    def test_invisible_shoulders_cannot_be_judged(self):
        k, c = skeleton(0.05, conf=0.1)
        self.assertIsNone(S.head_down(k, c, [80, 150, 190, 400])[0])


class TrackerTests(unittest.TestCase):
    def run_track(self, down_pattern, seconds=130, step=2.0, jitter=0.0):
        t = S.SleepTracker()
        out = None
        for i in range(int(seconds / step)):
            box = [100 + (jitter * (i % 2)), 100, 200 + (jitter * (i % 2)), 400]
            r = t.observe("camera_01", "P-1", box, down_pattern(i), 1000.0 + i * step)
            out = out or r
        return out, t

    def test_sustained_head_down_and_still_raises(self):
        decision, _ = self.run_track(lambda i: True)
        self.assertIsNotNone(decision)
        self.assertGreaterEqual(decision["down_fraction"], S.HEAD_DOWN_FRACTION)

    def test_mostly_upright_does_not_raise(self):
        decision, _ = self.run_track(lambda i: i % 3 == 0)
        self.assertIsNone(decision)

    def test_moving_student_does_not_raise(self):
        decision, _ = self.run_track(lambda i: True, jitter=120.0)
        self.assertIsNone(decision, "writing / reaching is not sleeping")

    def test_short_episode_does_not_raise(self):
        decision, _ = self.run_track(lambda i: True, seconds=60)
        self.assertIsNone(decision)

    def test_unjudgeable_samples_do_not_count(self):
        decision, _ = self.run_track(lambda i: None)
        self.assertIsNone(decision)


class FakeTensor:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeResult:
    def __init__(self, boxes, kxy, kcf):
        self.boxes = type("B", (), {"xyxy": FakeTensor(boxes), "__len__": lambda s: len(boxes)})()
        self.keypoints = type("K", (), {"xy": FakeTensor(kxy), "conf": FakeTensor(kcf)})()


class FakePose:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def acquire(self):
        return self

    def get_or_infer(self, camera_id, frame_id, frame):
        self.calls += 1
        return self.result


class AdapterTests(unittest.TestCase):
    def test_worker_matches_skeletons_and_parks_findings(self):
        k, c = skeleton(0.02, shoulders_y=200)
        pose = FakePose(FakeResult([[98, 150, 202, 402]], [k], [c]))
        adapter = S.SleepingAdapter(pose=pose, log=lambda *_: None)
        adapter.set_enabled_cameras(["camera_01"])
        track = helpers.Track("P-9", [100, 150, 200, 400])
        for i in range(70):                                  # drive the worker synchronously
            adapter._process("camera_01", None, [(track.track_id, track.bbox)], 1000.0 + i * 2.0, i, 2592, 1944)
        found = adapter.flush()
        adapter.close()
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].track_id, "P-9")
        self.assertEqual(found[0].metadata()["method"], "pose_head_at_shoulder_level_and_still")

    def test_submit_is_throttled_per_camera(self):
        adapter = S.SleepingAdapter(pose=FakePose(None), log=lambda *_: None)
        adapter.set_enabled_cameras(["camera_01"])
        track = helpers.Track("P-1", [0, 0, 10, 10])
        sent = [adapter.submit("camera_01", None, [track], 1.0, i, 10, 10) for i in range(5)]
        adapter.close()
        self.assertEqual(sent, [True, False, False, False, False])


if __name__ == "__main__":
    unittest.main(verbosity=2)
