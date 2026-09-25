"""
Identity supervisor: Face-ID / Re-ID that failed to start are rebuilt and swapped in. CPU only.

    python -m unittest test_identity_supervisor
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import identity_supervisor as ids                               # noqa: E402


class FakeFace:
    """Shaped like face_id_adapter.FaceIDAdapter's two private flags the supervisor reads."""

    def __init__(self, qdrant_up):
        self._qdrant_up = qdrant_up
        self._processor = None
        self._processor_failed = False
        self.closed = False

    def set_enabled_cameras(self, cameras):
        if cameras and self._processor is None:
            if self._qdrant_up():
                self._processor = object()
            else:
                self._processor_failed = True

    def close(self):
        self.closed = True


class FakeReID:
    def __init__(self, ok):
        self._authorised = {"CAMERA_01"}
        self.enabled = ok
        self.closed = False

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class IdentitySupervisorTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.up = False
        self.live = {"face": FakeFace(lambda: self.up), "reid": FakeReID(False)}
        self.live["face"].set_enabled_cameras(["camera_01"])       # failed at startup
        self.logs = []
        self.sup = ids.IdentitySupervisor(log=self.logs.append, clock=self.clock, min_delay=30, max_delay=120)
        self.sup.watch("Face-ID", lambda: self.live["face"], lambda n: self.live.__setitem__("face", n),
                       lambda: FakeFace(lambda: self.up), failed=ids.face_failed,
                       arm=lambda a: a.set_enabled_cameras(["camera_01"]), ready=ids.face_ready)
        self.sup.watch("Re-ID", lambda: self.live["reid"], lambda n: self.live.__setitem__("reid", n),
                       lambda: FakeReID(self.up), failed=ids.reid_failed)

    def test_failed_components_are_rebuilt_once_the_dependency_is_back(self):
        old_face, old_reid = self.live["face"], self.live["reid"]
        self.assertEqual(self.sup.check_once(), [])                  # noticed, first retry scheduled
        self.clock.t = 31
        self.assertEqual(self.sup.check_once(), [])                  # still down: backoff doubles
        self.assertEqual(self.sup.stats()["Face-ID"]["next_delay_s"], 60)
        self.up = True                                               # Qdrant comes up
        self.clock.t = 92
        self.assertEqual(sorted(self.sup.check_once()), ["Face-ID", "Re-ID"])
        self.assertIsNot(self.live["face"], old_face)
        self.assertTrue(ids.face_ready(self.live["face"]))
        self.assertTrue(self.live["reid"].enabled)
        self.assertTrue(old_face.closed and old_reid.closed, "the failed instances are closed")
        self.clock.t = 500
        self.assertEqual(self.sup.check_once(), [], "healthy components are left alone")

    def test_backoff_is_capped(self):
        self.sup.check_once()
        for step in range(1, 8):
            self.clock.t += 1000
            self.sup.check_once()
        self.assertEqual(self.sup.stats()["Face-ID"]["next_delay_s"], 120)

    def test_a_rebuild_that_raises_is_contained(self):
        sup = ids.IdentitySupervisor(log=self.logs.append, clock=self.clock, min_delay=1, max_delay=8)

        def boom():
            raise RuntimeError("CUDA out of memory")

        sup.watch("Face-ID", lambda: self.live["face"], lambda n: None, boom, failed=ids.face_failed)
        sup.check_once()
        self.clock.t = 2
        self.assertEqual(sup.check_once(), [])
        self.assertIn("CUDA out of memory", sup.stats()["Face-ID"]["last_error"])

    def test_reid_with_no_authorised_camera_is_not_a_failure(self):
        quiet = FakeReID(False)
        quiet._authorised = set()
        self.assertFalse(ids.reid_failed(quiet))


if __name__ == "__main__":
    unittest.main(verbosity=2)
