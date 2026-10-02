"""
Unit tests for abandoned.py - resting-spot continuity (Dubai, 2026-09-15).

Taken unchanged from dubai_ai_inferncing@5b192cef testing/test_abandoned_spot_continuity.py (only this
header and the import differ) so the Dubai behaviour stays guarded in the Zayed suite: the episode
rework (test_unattended_episodes.py) must keep every one of these passing.

Deterministic: a fake monotonic clock, synthetic tracks, no model, no camera.

The CAM-R25 regression: a backpack left against a wall dropped out of detection for 2.5-10 s at a
time, and tracking.py relabels a track that coasted > 1.5 s, so every dropout arrived here as a NEW
track id and restarted the 30 s dwell.
"""
import unittest
from unittest import mock

import helpers  # noqa: F401

import abandoned as ab

CAM = "CAM-R25"
W, H = 1920, 1080
BAG = [1143, 645, 1271, 867]            # the real box from the CAM-R25 test
FAR_PERSON = [100, 100, 200, 400]       # well outside the 264 px radius
NEAR_PERSON = [1000, 500, 1100, 900]    # inside it


class Track:
    def __init__(self, track_id, bbox=BAG, class_id=24, class_name="backpack", confidence=0.6):
        self.track_id, self.bbox, self.class_id, self.class_name = track_id, list(bbox), class_id, class_name
        self.confidence = confidence


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class SpotContinuityTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        patcher = mock.patch.object(ab.time, "monotonic", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.det = ab.AbandonedObjectDetector({CAM: {"abandoned_object": True}}, dwell_seconds=30.0)

    def run_frames(self, seconds, tracks_fn, persons_fn=lambda t: [], fps=10):
        fired = []
        for _ in range(int(seconds * fps)):
            self.clock.t += 1.0 / fps
            tracks = tracks_fn(self.clock.t)
            fired += self.det.update(CAM, tracks, persons_fn(self.clock.t), W, H)
        return fired

    # -------------------------------------------------------------- the bug
    def test_detector_dropouts_with_new_track_ids_still_fire(self):
        """Bag visible 8 s, missed 4 s, reappears under a new id - repeatedly.
        The old per-track clock restarted at every new id and never fired."""
        t0 = self.clock.t
        def tracks(t):
            phase = (t - t0) % 12.0
            generation = int((t - t0) // 12.0)
            return [Track(f"O-{generation}")] if phase < 8.0 else []
        fired = self.run_frames(45, tracks)
        self.assertEqual(len(fired), 1, "a bag left 45 s with detector dropouts must alarm once")

    def test_clock_starts_when_grace_ends_not_at_next_detection(self):
        """Seen, then missed for 10 s with nobody near, then seen again: the
        dwell counts from the end of grace, so it fires ~32 s after first sight."""
        t0 = self.clock.t
        def tracks(t):
            e = t - t0
            return [] if 2.5 <= e < 12.5 else [Track("O-1" if e < 2.5 else "O-2")]
        fired = []
        for _ in range(360):
            self.clock.t += 0.1
            f = self.det.update(CAM, tracks(self.clock.t), [], W, H)
            if f:
                fired.append(self.clock.t - t0)
        self.assertEqual(len(fired), 1)
        self.assertLess(fired[0], 33.0)

    # ------------------------------------------------------------ safeguards
    def test_owner_returning_during_a_dropout_resets_the_clock(self):
        t0 = self.clock.t
        tracks = lambda t: [] if 10 <= t - t0 < 20 else [Track("O-1")]
        persons = lambda t: [NEAR_PERSON] if 14 <= t - t0 < 16 else [FAR_PERSON]
        fired = self.run_frames(44, tracks, persons)   # 16 s + 2 s grace + 30 s > 44 s
        self.assertEqual(fired, [], "owner came back at 14-16 s while the bag was missed")

    def test_carried_bag_never_fires(self):
        fired = self.run_frames(60, lambda t: [Track("O-1")], lambda t: [NEAR_PERSON])
        self.assertEqual(fired, [])

    def test_moved_bag_restarts_its_dwell(self):
        t0 = self.clock.t
        moved = [1143 - 400, 645, 1271 - 400, 867]
        tracks = lambda t: [Track("O-1", moved if t - t0 >= 20 else BAG)]
        fired = self.run_frames(45, tracks)
        self.assertEqual(fired, [], "moved at 20 s: new resting spot, needs 32 s more")

    def test_barely_observed_spot_does_not_fire(self):
        """Two brief blips 25 s apart are not 30 s of observed bag."""
        t0 = self.clock.t
        tracks = lambda t: [Track("O-1")] if (t - t0) % 25.0 < 0.5 else []
        fired = self.run_frames(90, tracks)
        self.assertEqual(fired, [])

    def test_two_separate_bags_keep_separate_clocks(self):
        t0 = self.clock.t
        other = [300, 700, 420, 900]
        tracks = lambda t: [Track("O-1")] + ([Track("O-2", other)] if t - t0 >= 15 else [])
        fired = []
        for _ in range(600):                       # O-2 appears at 15 s, fires ~47 s
            self.clock.t += 0.1
            fired += [(o.track_id, round(self.clock.t - t0)) for o, _ in self.det.update(CAM, tracks(self.clock.t), [], W, H)]
        self.assertEqual([f[0] for f in fired], ["O-1", "O-2"])
        self.assertGreater(fired[1][1] - fired[0][1], 10)

    def test_same_bag_reported_as_backpack_and_handbag_alarms_once(self):
        """Low-confidence detection of one bag under two classes, two tracks."""
        dup = [1140, 646, 1270, 869]
        tracks = lambda t: [Track("O-1", BAG, 24, "backpack", 0.31), Track("O-2", dup, 26, "handbag", 0.28)]
        fired = self.run_frames(60, tracks)
        self.assertEqual(len(fired), 1)

    def test_handbag_is_a_candidate(self):
        self.assertTrue({24, 26, 28} <= set(getattr(ab, "CANDIDATE_CLASS_IDS", set())))

    def test_stale_spot_is_not_reported_on_clock(self):
        self.run_frames(10, lambda t: [Track("O-1")])
        self.assertEqual(self.det.active_count(), 1)
        self.run_frames(8, lambda t: [])
        self.assertEqual(self.det.active_count(), 0, "unseen for 8 s must not read as a running clock")

    def test_disabled_camera_does_nothing(self):
        det = ab.AbandonedObjectDetector({CAM: {"abandoned_object": False}})
        self.assertEqual(det.update(CAM, [Track("O-1")], [], W, H), [])

    def test_alarm_fires_once_per_resting_spot(self):
        fired = self.run_frames(120, lambda t: [Track("O-1")])
        self.assertEqual(len(fired), 1)


if __name__ == "__main__":
    unittest.main(verbosity=1)
