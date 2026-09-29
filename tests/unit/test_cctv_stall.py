"""
A camera whose decoder goes quiet must recover itself.

    python -m unittest test_cctv_stall

The failure being fixed: CameraStream._read_exact() blocks on the ffmpeg pipe with no timeout,
and the reader's reconnect logic sits at the TOP of its loop - reachable only when ffmpeg
EXITS and closes stdout. An ffmpeg that is alive but emitting nothing parks the reader in
read() for ever. The camera goes silent, inference stays "running" with its own watchdog
satisfied, and only a full container restart recovers it.

Production saw this twice: 2026-09-24 03:34 (two cameras, ~5 h) and 2026-09-24 22:43 (all
three, ~12 h). Both times the relays were healthy and still publishing.

These tests use a REAL child process that produces nothing, because a mock cannot demonstrate
the thing that was broken - a genuinely blocking pipe read.
"""
import subprocess
import sys
import threading
import time
import unittest

import helpers  # noqa: F401

import cctv


def quiet_process():
    """A live child that writes nothing and does not exit - an ffmpeg that has wedged."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


class StallWatchdog(unittest.TestCase):
    def setUp(self):
        self.original = (cctv.STALL_TIMEOUT_SECONDS, cctv.STALL_CHECK_SECONDS)
        cctv.STALL_TIMEOUT_SECONDS = 0.4          # keep the test quick
        cctv.STALL_CHECK_SECONDS = 0.1

    def tearDown(self):
        cctv.STALL_TIMEOUT_SECONDS, cctv.STALL_CHECK_SECONDS = self.original

    def stream(self):
        stream = cctv.CameraStream.__new__(cctv.CameraStream)
        stream.camera_id = "camera_01"
        stream.running = True
        stream.proc = None
        stream.last_frame_at = time.time()
        stalls = []
        stream.stalls_recovered = 0
        return stream, stalls

    def test_a_live_but_silent_decoder_is_killed_so_the_reader_can_reconnect(self):
        stream, _ = self.stream()
        proc = quiet_process()
        stream.proc = proc
        closed = threading.Event()

        def close():
            proc.kill()
            proc.wait(timeout=5)
            stream.proc = None
            closed.set()
        stream._close_proc = close

        watchdog = threading.Thread(target=stream._watchdog, daemon=True)
        watchdog.start()
        try:
            self.assertTrue(closed.wait(5), "the watchdog never killed the stalled decoder")
            self.assertEqual(proc.poll(), -9, "the process was not actually terminated")
            self.assertGreaterEqual(stream.stalls_recovered, 1)
        finally:
            stream.running = False
            watchdog.join(timeout=2)
            if proc.poll() is None:
                proc.kill()

    def test_a_camera_that_is_delivering_frames_is_left_alone(self):
        stream, _ = self.stream()
        proc = quiet_process()
        stream.proc = proc
        stream._close_proc = lambda: self.fail("the watchdog killed a healthy decoder")

        watchdog = threading.Thread(target=stream._watchdog, daemon=True)
        watchdog.start()
        try:
            # Keep reporting frames for well over the stall timeout.
            deadline = time.time() + 1.2
            while time.time() < deadline:
                stream.last_frame_at = time.time()
                time.sleep(0.05)
            self.assertEqual(stream.stalls_recovered, 0)
        finally:
            stream.running = False
            watchdog.join(timeout=2)
            proc.kill()

    def test_a_decoder_that_already_exited_is_left_to_the_reader(self):
        # The reader's own path handles a clean exit; the watchdog must not double-handle it.
        stream, _ = self.stream()
        proc = quiet_process()
        proc.kill()
        proc.wait(timeout=5)
        stream.proc = proc
        stream.last_frame_at = time.time() - 60          # long stalled
        stream._close_proc = lambda: self.fail("the watchdog interfered with an exited process")

        watchdog = threading.Thread(target=stream._watchdog, daemon=True)
        watchdog.start()
        time.sleep(0.5)
        stream.running = False
        watchdog.join(timeout=2)
        self.assertEqual(stream.stalls_recovered, 0)

    def test_it_stops_when_the_stream_stops(self):
        stream, _ = self.stream()
        stream._close_proc = lambda: None
        watchdog = threading.Thread(target=stream._watchdog, daemon=True)
        watchdog.start()
        stream.running = False
        watchdog.join(timeout=3)
        self.assertFalse(watchdog.is_alive(), "the watchdog outlived its stream")

    def test_one_stalled_camera_does_not_touch_another(self):
        stalled, _ = self.stream()
        healthy, _ = self.stream()
        healthy.camera_id = "camera_02"

        stalled_proc, healthy_proc = quiet_process(), quiet_process()
        stalled.proc, healthy.proc = stalled_proc, healthy_proc
        killed = threading.Event()

        def close_stalled():
            stalled_proc.kill(); stalled_proc.wait(timeout=5); stalled.proc = None; killed.set()
        stalled._close_proc = close_stalled
        healthy._close_proc = lambda: self.fail("a stalled camera killed a healthy one")

        threads = [threading.Thread(target=s._watchdog, daemon=True) for s in (stalled, healthy)]
        for t in threads:
            t.start()
        try:
            deadline = time.time() + 1.5
            while time.time() < deadline:
                healthy.last_frame_at = time.time()
                time.sleep(0.05)
            self.assertTrue(killed.is_set(), "the stalled camera was not recovered")
            self.assertEqual(healthy.stalls_recovered, 0)
            self.assertIsNone(healthy_proc.poll(), "the healthy decoder was killed")
        finally:
            stalled.running = healthy.running = False
            for t in threads:
                t.join(timeout=2)
            for p in (stalled_proc, healthy_proc):
                if p.poll() is None:
                    p.kill()


if __name__ == "__main__":
    unittest.main(verbosity=2)
