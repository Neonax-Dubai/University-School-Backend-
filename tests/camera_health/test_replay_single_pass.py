"""A replayed file plays ONCE, at real time, and the harness then stops (ZAYED 2026-10-11).

THE DEFECT. Without --loop, CameraStream still treated a file's end like a lost RTSP stream and
reconnected, so replay_zayed_inference.py replayed the clip indefinitely: 40 passes in 10.5 min
of the camera_03 tamper clip on 2026-10-11. Every pass re-emitted its events (one
OCCUPANCY_UPDATED per pass for a classroom camera), and every restart's jump back to t=0 was
read by camera health as a lighting change ("baselines re-learned"). Together with unpaced
decoding (see test_replay_wallclock_dwell.py) it made a bounded, faithful replay impossible.

THE FIX. An UNLOOPED local file ends after one pass (CameraStream.finished); --loop and live
RTSP keep reconnecting exactly as before. The harness exits through its normal shutdown once
every consumed clip has finished and been drained.

Two layers:
  * StreamEndTests drive the real reader loop with a stub decoder (no ffmpeg, deterministic),
    so the reconnect decision is pinned for files AND RTSP.
  * RealFfmpegPlaybackTests decode a generated clip with the real ffmpeg on the CPU path and
    measure wall time: they fail if playback is unpaced or if the clip plays more than once.
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

import cctv                                                                  # noqa: E402
from cctv import CameraStream                                                # noqa: E402

W, H = 8, 6
FRAME = bytes(W * H * 3)


def _wait(predicate, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class _FakeProc:
    """Stands in for the ffmpeg Popen: a pipe holding `frames` frames, then end of stream."""

    def __init__(self, frames):
        self.stdout = io.BytesIO(FRAME * frames)

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0


class _StubbedStream(CameraStream):
    """The real reader loop, with connect() replaced so no ffmpeg/ffprobe is needed."""

    def __init__(self, source, loop_video, frames=5):
        super().__init__("cam", source, queue_size=1000, loop_video=loop_video)
        self.reconnect_delay = 0.01
        self.frames_per_connect = frames
        self.connects = 0

    def connect(self):
        self._close_proc()
        self.width, self.height = W, H
        self.proc = _FakeProc(self.frames_per_connect)
        self.connects += 1
        return True

    def _watchdog(self):
        return                                 # irrelevant here; keep the test single-threaded


class StreamEndTests(unittest.TestCase):

    def setUp(self):
        fd, self.clip = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        self.streams = []

    def tearDown(self):
        for s in self.streams:
            s.stop()
            if s.thread:
                s.thread.join(timeout=2)
        os.unlink(self.clip)

    def _start(self, source, loop_video):
        s = _StubbedStream(source, loop_video)
        self.streams.append(s)
        s.start()
        return s

    def test_an_unlooped_file_plays_exactly_once(self):
        s = self._start(self.clip, loop_video=False)
        self.assertTrue(_wait(lambda: s.finished, 2), "the file never finished")
        s.thread.join(timeout=2)
        self.assertFalse(s.thread.is_alive(), "the reader must exit after one pass")
        self.assertEqual(s.connects, 1, "the file was reopened - it would replay again")
        self.assertEqual(s.frames_read, 5)
        self.assertFalse(s.running)

    def test_a_looped_file_keeps_playing(self):
        """--loop (soak tests) is unchanged: the file never 'finishes'."""
        s = self._start(self.clip, loop_video=True)
        self.assertTrue(_wait(lambda: s.connects >= 3, 2), "a looped file must restart")
        self.assertFalse(s.finished)

    def test_live_rtsp_still_reconnects_when_the_stream_drops(self):
        """Production builds CameraStream(..., loop_video=False) for RTSP (zayed_inference.py).
        A dropped camera must reconnect, never be treated as a finished replay."""
        for url in ("rtsp://mediamtx:8554/camera_03", "rtsps://mediamtx:8555/camera_03"):
            s = self._start(url, loop_video=False)
            self.assertTrue(_wait(lambda: s.connects >= 3, 2), f"{url} stopped reconnecting")
            self.assertFalse(s.finished, f"{url} must never finish")
            self.assertTrue(s.running)

    def test_a_non_rtsp_source_that_is_not_a_file_still_reconnects(self):
        """Only a real local file can end. Anything else - an http source, a path that is
        not there yet - keeps the old reconnect behaviour."""
        s = self._start("/no/such/clip.mp4", loop_video=False)
        self.assertTrue(_wait(lambda: s.connects >= 3, 2))
        self.assertFalse(s.finished)

    def test_stop_during_playback_is_not_a_finished_pass(self):
        class _Endless:
            def read(self, n):
                time.sleep(0.001)
                return bytes(n)

        s = _StubbedStream(self.clip, loop_video=False)
        real_connect = s.connect

        def connect():
            real_connect()
            s.proc.stdout = _Endless()
            return True

        s.connect = connect
        self.streams.append(s)
        s.start()
        self.assertTrue(_wait(lambda: s.frames_read > 0, 2))
        s.stop()
        s.thread.join(timeout=2)
        self.assertFalse(s.finished, "an operator stop is not the end of the clip")


class HarnessSinglePassTests(unittest.TestCase):
    """replay_zayed_inference.ZayedReplay._all_clips_finished, on stand-in streams."""

    @classmethod
    def setUpClass(cls):
        try:
            import replay_zayed_inference as R
        except Exception as exc:                       # heavy imports (torch, ultralytics)
            raise unittest.SkipTest(f"replay harness not importable here: {exc}")
        cls.check = staticmethod(R.ZayedReplay._all_clips_finished)

    @staticmethod
    def _stream(cam, finished, queued):
        return types.SimpleNamespace(camera_id=cam, finished=finished,
                                     queue_size=lambda: queued)

    def _harness(self, streams, classes):
        return types.SimpleNamespace(streams={s.camera_id: s for s in streams},
                                     camera_classes=classes)

    def test_waits_while_a_clip_is_still_playing(self):
        h = self._harness([self._stream("a", True, 0), self._stream("b", False, 0)],
                          {"a": {0}, "b": {0}})
        self.assertFalse(self.check(h))

    def test_waits_until_the_last_frames_are_drained(self):
        h = self._harness([self._stream("a", True, 2)], {"a": {0}})
        self.assertFalse(self.check(h), "queued frames would be lost unanalysed")

    def test_finishes_when_every_clip_is_done_and_drained(self):
        h = self._harness([self._stream("a", True, 0), self._stream("b", True, 0)],
                          {"a": {0}, "b": {0}})
        self.assertTrue(self.check(h))

    def test_an_undrained_camera_does_not_block_the_finish(self):
        """The loop never drains a camera with no detection classes; waiting on its queue
        would hang the replay forever."""
        h = self._harness([self._stream("a", True, 0), self._stream("b", True, 3)],
                          {"a": {0}, "b": set()})
        self.assertTrue(self.check(h))

    def test_no_streams_is_not_finished(self):
        self.assertFalse(self.check(self._harness([], {})))


@unittest.skipUnless(shutil.which(cctv.FFMPEG) and shutil.which(cctv.FFPROBE),
                     "ffmpeg/ffprobe not on PATH")
class RealFfmpegPlaybackTests(unittest.TestCase):
    """The real ffmpeg, on a generated clip. mpeg4 has no NVDEC decoder in cctv._CUVID, so this
    runs on the CPU path and needs no GPU."""

    SECONDS, FPS = 5, 10

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.clip = os.path.join(cls.tmp, "clip.mp4")
        subprocess.run(
            [cctv.FFMPEG, "-nostdin", "-loglevel", "error", "-f", "lavfi",
             "-i", f"testsrc=size=64x48:rate={cls.FPS}:duration={cls.SECONDS}",
             "-c:v", "mpeg4", "-y", cls.clip],
            check=True, timeout=60)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_one_pass_at_real_time(self):
        s = CameraStream("cam", self.clip, queue_size=1000, loop_video=False)
        connects = []
        real_connect = s.connect
        s.connect = lambda: connects.append(1) or real_connect()
        t0 = time.time()
        s.start()
        try:
            self.assertTrue(_wait(lambda: s.finished, self.SECONDS + 15),
                            "the clip did not finish")
            wall = time.time() - t0
        finally:
            s.stop()
        self.assertEqual(len(connects), 1, "the clip was reopened - it played more than once")
        # One pass is SECONDS*FPS frames, less the one the existing `-fflags nobuffer -flags
        # low_delay` input options drop on this clip (29 of 30 with or without -re, measured
        # 2026-10-11). A second pass would roughly double the count.
        expected = self.SECONDS * self.FPS
        self.assertIn(s.frames_read, (expected - 1, expected),
                      "not exactly one pass of the clip")
        # Unpaced, this clip decodes in well under a second. Paced, ffmpeg 6.1's -re still
        # bursts the first 0.5 s of input (-readrate_initial_burst default) and the last frame
        # needs no wait, so one pass takes about SECONDS - 0.6 s (measured 2.37 s for 3 s).
        self.assertGreaterEqual(wall, self.SECONDS - 1.0,
                                f"played in {wall:.2f} s - playback is not paced")
        self.assertLess(wall, self.SECONDS + 5, f"played in {wall:.2f} s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
