"""
Evidence stills under a SeaweedFS outage: the event must still be delivered. CPU, loopback only.

    python -m unittest test_evidence_outage
"""
import os
import sys
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import cv2                                                      # noqa: E402
import numpy as np                                              # noqa: E402

import evidence                                                 # noqa: E402
from fakes import FakeFiler, HangingServer, free_port           # noqa: E402

JPEG = cv2.imencode(".jpg", np.full((64, 48, 3), 127, np.uint8))[1].tobytes()


class RecordingSender:
    def __init__(self):
        self.events = []
        self.cv = threading.Condition()

    def submit(self, event, notable=False):
        with self.cv:
            self.events.append(event)
            self.cv.notify_all()

    def wait(self, n, timeout):
        with self.cv:
            return self.cv.wait_for(lambda: len(self.events) >= n, timeout)


def run(store, timeout=10.0):
    sender = RecordingSender()
    pipeline = evidence.EvidencePipeline(sender, store=store, workers=1, queue_size=8)
    pipeline.start()
    started = time.monotonic()
    pipeline.submit(JPEG, 48, 64, {"camera_id": "camera_01", "event_type": "FALL_DETECTED",
                                   "metadata": {}}, True, "camera_01", "P-0001", time.time())
    delivered = sender.wait(1, timeout)
    elapsed = time.monotonic() - started
    pipeline.stop(2.0)
    return sender, pipeline, delivered, elapsed


class EvidenceOutageTests(unittest.TestCase):
    def test_filer_down_event_still_sent_without_evidence(self):
        store = evidence.FilerStore(base_url=f"http://127.0.0.1:{free_port()}", ttl=None, timeout=1.0)
        sender, pipeline, delivered, elapsed = run(store)
        self.assertTrue(delivered, "the detection is delivered although storage is down")
        self.assertNotIn("evidence", sender.events[0]["metadata"])
        self.assertEqual(pipeline.failed, 1)

    def test_filer_frozen_event_still_sent_after_the_timeout(self):
        port = free_port()
        hang = HangingServer(port).start()
        try:
            store = evidence.FilerStore(base_url=f"http://127.0.0.1:{port}", ttl=None, timeout=1.0)
            sender, pipeline, delivered, elapsed = run(store)
        finally:
            hang.stop()
        self.assertTrue(delivered)
        self.assertLess(elapsed, 5.0, "bounded by the upload timeout, not stuck")
        self.assertNotIn("evidence", sender.events[0]["metadata"])

    def test_filer_up_event_carries_its_zayed_evidence_key(self):
        port = free_port()
        filer = FakeFiler(port).start()
        try:
            store = evidence.FilerStore(base_url=f"http://127.0.0.1:{port}", ttl="30d", timeout=2.0)
            sender, pipeline, delivered, elapsed = run(store)
        finally:
            filer.stop()
        self.assertTrue(delivered)
        ev = sender.events[0]["metadata"]["evidence"]
        self.assertTrue(ev["storage_key"].startswith("zayed-evidence/"), ev["storage_key"])
        self.assertIn("/camera_01/P-0001/", ev["storage_key"])
        self.assertEqual(filer.keys, [ev["storage_key"]], "uploaded BEFORE the event was handed on")


if __name__ == "__main__":
    unittest.main(verbosity=2)
