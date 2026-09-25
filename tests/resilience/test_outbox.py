"""
Durable event outbox under dashboard / database outages. CPU only, loopback only.

    python -m unittest test_outbox
"""
import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import outbox                                                   # noqa: E402
from fakes import FakeDashboard, free_port                      # noqa: E402

outbox.MAX_BACKOFF_SECONDS = 2.0          # keep the test fast; the policy under test is unchanged


def wait_for(predicate, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def event(i, event_id=None):
    e = {"camera_id": "camera_01", "event_type": "OCCUPANCY_UPDATED", "timestamp": time.time(),
         "metadata": {"i": i}}
    if event_id:
        e["event_id"] = event_id
    return e


class OutboxTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "events.sqlite3")
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.logs = []
        self.senders = []
        self.dash = None

    def tearDown(self):
        for s in self.senders:
            s.stop(1.0)
        if self.dash:
            self.dash.stop()

    def sender(self):
        s = outbox.OutboxEventSender(self.url, "test-token", timeout=1.0, path=self.path,
                                     log=self.logs.append)
        self.senders.append(s)
        return s

    def test_outage_loses_nothing_and_delivers_exactly_once(self):
        s = self.sender()
        s.start()
        for i in range(100):
            s.submit(event(i))
        time.sleep(6.0)                                  # the dashboard is DOWN for 6 s
        self.assertEqual(s.pending(), 100, "every event is held while the dashboard is down")
        self.assertLessEqual(s.failed, 12, f"backs off as a whole, not row by row ({s.failed} attempts)")
        self.dash = FakeDashboard(self.port).start()     # the dashboard comes back
        self.assertTrue(wait_for(lambda: s.pending() == 0), "the backlog drains")
        self.assertEqual(len(self.dash.stored), 100, "all 100 stored")
        self.assertEqual(s.sent, 100)
        self.assertEqual(s.dropped, 0)

    def test_backlog_survives_a_process_restart(self):
        first = self.sender()
        first.start()
        for i in range(50):
            first.submit(event(i))
        time.sleep(1.0)
        first.stop(1.0)                                   # the inference process exits
        second = self.sender()                            # ... and is restarted
        self.assertEqual(second.pending(), 50, "undelivered events restored from disk")
        self.assertTrue(any("restored" in line for line in self.logs))
        self.dash = FakeDashboard(self.port).start()
        second.start()
        self.assertTrue(wait_for(lambda: second.pending() == 0))
        self.assertEqual(len(self.dash.stored), 50)

    def test_a_duplicate_is_acknowledged_not_resent_forever(self):
        self.dash = FakeDashboard(self.port).start()
        self.dash.stored["ZEV-ALREADY"] = {}
        s = self.sender()
        s.start()
        s.submit(event(1, event_id="ZEV-ALREADY"))
        self.assertTrue(wait_for(lambda: s.pending() == 0))
        self.assertEqual(s.duplicates, 1)
        self.assertEqual(len(self.dash.stored), 1, "no second row")

    def test_an_invalid_event_is_dead_lettered_and_does_not_block_the_rest(self):
        self.dash = FakeDashboard(self.port).start()
        self.dash.reject.add("ZEV-BAD")
        s = self.sender()
        s.start()
        s.submit(event(0, event_id="ZEV-BAD"))
        for i in range(5):
            s.submit(event(i + 1))
        self.assertTrue(wait_for(lambda: s.pending() == 0))
        self.assertEqual(s.dead_letters(), 1)
        self.assertEqual(len(self.dash.stored), 5)

    def test_server_errors_are_retried(self):
        self.dash = FakeDashboard(self.port).start()
        self.dash.script = {1: 503, 2: 500, 3: 502}       # database outage behind the dashboard
        s = self.sender()
        s.start()
        s.submit(event(1))
        self.assertTrue(wait_for(lambda: s.pending() == 0, timeout=40))
        self.assertEqual(len(self.dash.stored), 1)
        self.assertGreaterEqual(s.retries, 3)
        self.assertEqual(s.dead_letters(), 0)

    def test_an_auth_failure_is_retried_not_dropped(self):
        self.dash = FakeDashboard(self.port).start()
        self.dash.script = {1: 401, 2: 403}               # token rotated / not yet valid
        s = self.sender()
        s.start()
        s.submit(event(1))
        self.assertTrue(wait_for(lambda: s.pending() == 0, timeout=40))
        self.assertEqual(len(self.dash.stored), 1)
        self.assertEqual(s.dead_letters(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
