"""Replay file playback is unpaced, so wall-clock dwells cannot be met (ZAYED 2026-10-11).

THE DEFECT. camera_health measures every dwell in the timestamps handed to observe(), and the
replay hands it wall-clock time (cctv.py: "timestamp": time.time()). But CameraStream plays a
local file with -stream_loop and WITHOUT -re, so ffmpeg decodes as fast as it can.

Measured on tamper/camera_03_20260930_160800_160900.mp4:

    clip source duration        60.0 s
    wall per pass               17.80 s and 17.86 s   -> 3.37x FASTER than real time
    obstruction source t=20-34  14.0 s  ->  4.15 s of wall clock
    TAMPER_SECONDS              8.0 (wall)
    => a cover must last 8.0 * 3.37 = 27.0 s of SOURCE to raise in this harness

That is why the harness emitted no transition while the detector raises in every harness that
timestamps samples in source time. Counters from the instrumented run agree: ~18 samples
accepted per 60 s pass, transitions=0, observe_err=0, process_err=0, thread alive.

The first class records the arithmetic. The second exercises the fix - cctv.py paces every
local file with -re - on the real command builder. Single-pass playback (no reconnect at the
end of an unlooped file) is covered in test_replay_single_pass.py.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from camera_health import detectors as D                                      # noqa: E402

CCTV = os.path.join(HERE, "..", "..", "cctv.py")

# Measured 2026-10-11 on the instrumented replay, two consecutive passes.
CLIP_SOURCE_SECONDS = 60.0
CLIP_WALL_SECONDS = 17.80
OBSTRUCTION_SOURCE_SECONDS = 14.0          # source t=20 .. t=34


class CompressedTimelineArithmeticTests(unittest.TestCase):
    """What an unpaced replay costs a wall-clock dwell."""

    @property
    def speed_up(self):
        return CLIP_SOURCE_SECONDS / CLIP_WALL_SECONDS

    def test_the_replay_runs_faster_than_real_time(self):
        self.assertGreater(self.speed_up, 3.0, "measured 3.37x on 2026-10-11")

    def test_the_obstruction_is_compressed_below_the_dwell(self):
        wall = OBSTRUCTION_SOURCE_SECONDS / self.speed_up
        self.assertLess(
            wall, D.TAMPER_SECONDS,
            f"obstruction occupies {wall:.2f} s of wall clock against a "
            f"{D.TAMPER_SECONDS} s dwell - it cannot raise, however good the detector is")

    def test_the_source_duration_a_cover_would_need(self):
        needed = D.TAMPER_SECONDS * self.speed_up
        self.assertGreater(
            needed, 25.0,
            "an unpaced replay demands ~27 s of SOURCE obstruction for an 8 s dwell")


class FilePlaybackPacingTests(unittest.TestCase):
    """The pacing fix, exercised on the REAL command builder rather than the source text.

    Pacing file playback with -re makes wall time track source time, which is what every
    second-counting analytic in this pipeline already assumes - camera health here, and the
    same arithmetic applies to sleeping, phone-use and unattended-object dwells.

    `cctv.py` is shared with live inference, so the RTSP cases below matter as much as the
    file ones: a future production restart loads this file.
    """

    @staticmethod
    def _cmd(source, codec="h264", loop=True):
        from cctv import CameraStream
        stream = CameraStream("cam", source, loop_video=loop)
        stream.width, stream.height, stream.codec = 1920, 1080, codec
        return stream._ffmpeg_cmd()

    # ---------------------------------------------------------------- file
    def test_a_looped_file_is_paced(self):
        cmd = self._cmd("/clips/test.mp4")
        self.assertIn("-re", cmd, "local file playback must be paced at native rate")

    def test_pacing_precedes_the_input(self):
        """-re is an INPUT option; after -i it would be ignored."""
        cmd = self._cmd("/clips/test.mp4")
        self.assertLess(cmd.index("-re"), cmd.index("-i"))

    def test_looping_is_preserved(self):
        cmd = self._cmd("/clips/test.mp4")
        self.assertIn("-stream_loop", cmd)
        self.assertEqual(cmd[cmd.index("-stream_loop") + 1], "-1")

    def test_the_rest_of_the_command_is_unchanged(self):
        cmd = self._cmd("/clips/test.mp4")
        for expected in ("-nostdin", "-hide_banner", "-loglevel", "-fflags", "nobuffer",
                         "-flags", "low_delay", "-an", "-f", "rawvideo", "-pix_fmt",
                         "bgr24", "pipe:1"):
            self.assertIn(expected, cmd, f"{expected} went missing from the ffmpeg command")

    def test_an_UNLOOPED_file_is_paced_too(self):
        """THE TRAP. replay_zayed_inference.py passes loop_video=args.loop, so without --loop
        a file plays unlooped - and that is exactly the configuration every replay validation
        has used. Putting -re inside the loop branch made it dead code: the live ffmpeg process
        carried neither -stream_loop nor -re. Pacing belongs to reading a FILE, not looping it."""
        cmd = self._cmd("/clips/test.mp4", loop=False)
        self.assertIn("-re", cmd, "an unlooped file must still be paced")
        self.assertNotIn("-stream_loop", cmd, "looping must stay off when not requested")

    # ---------------------------------------------------------------- RTSP
    def test_rtsp_is_never_paced(self):
        """A live stream already arrives in real time; -re would throttle it twice."""
        for url in ("rtsp://mediamtx:8554/camera_03", "rtsps://mediamtx:8555/camera_03"):
            cmd = self._cmd(url)
            self.assertNotIn("-re", cmd, f"{url} must not be paced")
            self.assertNotIn("-stream_loop", cmd, f"{url} must not be looped")

    def test_rtsp_keeps_its_transport(self):
        cmd = self._cmd("rtsp://mediamtx:8554/camera_03")
        self.assertIn("-rtsp_transport", cmd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
