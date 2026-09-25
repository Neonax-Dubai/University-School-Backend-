"""
A camera-health recovery must be as patient as the raise that it closes.

    aienv/bin/python test_camera_health_recovery.py

THE DEFECT. Condition.update() required a fault to persist for `persistence`
seconds before announcing it, but announced the RECOVERY on the first good
sample. On CAM-R25's 2026-09-16 tamper the pair landed one second apart
(raised 11:05:46, recovered 11:05:47), so an operator saw two camera-tamper
alerts a second apart for one incident. It also meant a hand waved past a lens
could produce a raise/recover pair per wave.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from camera_health.state import Condition                             # noqa: E402

PERSIST = 5.0
COOLDOWN = 600.0


class RecoveryHoldTests(unittest.TestCase):
    def setUp(self):
        self.condition = Condition("camera_tamper")

    def feed(self, bad, at, **kwargs):
        return self.condition.update(bad, at, PERSIST, COOLDOWN, {}, **kwargs)

    def raise_it(self, t0=0.0):
        """Drive the condition to ACTIVE and return the time it rose."""
        self.feed(True, t0)
        self.assertIsNone(self.feed(True, t0 + PERSIST - 1))
        self.assertEqual(self.feed(True, t0 + PERSIST), "raise")
        return t0 + PERSIST

    def test_raise_still_needs_its_persistence(self):
        self.raise_it()

    def test_recovery_waits_the_same_window(self):
        """THE REGRESSION: this returned "recover" on the first good sample.

        The hold runs from the FIRST good sample, mirroring the raise, which
        runs from the first bad one."""
        raised = self.raise_it()
        first_good = raised + 1
        self.assertIsNone(self.feed(False, first_good))
        self.assertIsNone(self.feed(False, first_good + PERSIST - 0.5))
        self.assertEqual(self.feed(False, first_good + PERSIST), "recover")

    def test_the_two_events_are_at_least_persistence_apart(self):
        raised = self.raise_it()
        recovered = None
        for step in range(1, 40):
            if self.feed(False, raised + step * 0.5) == "recover":
                recovered = raised + step * 0.5
                break
        self.assertIsNotNone(recovered)
        self.assertGreaterEqual(recovered - raised, PERSIST)

    def test_a_fault_returning_during_the_hold_cancels_the_recovery(self):
        """A flickering condition is ONE incident, not a stream of pairs."""
        raised = self.raise_it()
        self.assertIsNone(self.feed(False, raised + 2))
        self.assertIsNone(self.feed(True, raised + 3))      # bad again
        self.assertIsNone(self.feed(False, raised + 4))     # hold restarts here
        self.assertIsNone(self.feed(False, raised + 4 + PERSIST - 0.5))
        self.assertEqual(self.feed(False, raised + 4 + PERSIST), "recover")

    def test_one_raise_per_incident(self):
        raised = self.raise_it()
        for step in range(1, 10):
            self.assertIsNone(self.feed(True, raised + step))

    def test_recovery_can_be_forced_immediately_when_a_caller_asks(self):
        """The rebaseline path clears the condition deliberately."""
        raised = self.raise_it()
        self.assertEqual(self.feed(False, raised + 0.1, recover_persistence=0),
                         "recover")

    def test_good_samples_before_any_raise_do_nothing(self):
        for step in range(5):
            self.assertIsNone(self.feed(False, step))

    def test_a_second_incident_can_still_raise_after_recovery(self):
        raised = self.raise_it()
        self.feed(False, raised + 1)                       # hold starts here
        self.assertEqual(self.feed(False, raised + 1 + PERSIST), "recover")
        later = raised + PERSIST + COOLDOWN + 10
        self.feed(True, later)
        self.assertEqual(self.feed(True, later + PERSIST), "raise")


if __name__ == "__main__":
    unittest.main(verbosity=1)
