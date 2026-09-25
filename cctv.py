"""Camera reader with GPU (NVDEC) decoding.

Replaces the previous OpenCV `cv2.VideoCapture` path, which decoded H.265 on the
CPU and choked on out-of-order network frames — the `Could not find ref with
POC` / `Error constructing the frame RPS` spam and the dropped frames seen in the
benchmark. Here each camera is decoded by a dedicated FFmpeg process using the
GPU's hardware decoder (`hevc_cuvid` / `h264_cuvid`, NVDEC). FFmpeg emits raw
BGR24 frames on its stdout pipe; we wrap them as NumPy arrays — the exact same
format the rest of the pipeline (YOLO, cv2) expects — so nothing downstream
changes.

Why this fixes the drops:
  * NVDEC is a hardened decoder and does the RTSP depacketise + decode in one
    pipeline, so it does not trip on the reference-frame ordering that broke the
    OpenCV path.
  * Decode moves off the CPU onto the GPU's dedicated decode block, so 19 streams
    no longer contend for CPU — freeing the analytic loop to hit its target FPS.
  * One process per camera means no Python GIL contention between decoders.

Env overrides:
  CCTV_NVDEC=0        fall back to FFmpeg software decode (still more robust than
                      the old OpenCV path); default 1 = GPU/NVDEC.
  FFMPEG_BIN          ffmpeg binary (default "ffmpeg")
  FFPROBE_BIN         ffprobe binary (default "ffprobe")
  RTSP_TRANSPORT      tcp | udp for RTSP sources (default tcp)
  CCTV_FFMPEG_LOG=1   surface ffmpeg stderr for debugging (default suppressed)
"""
import json
import os
import subprocess
import threading
import time
from queue import Queue, Full

import numpy as np

FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")
RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp")
# How many frames a camera may hold for the inference loop.
#
# WHY IT IS NOT 3 ANY MORE. A camera losing RTP packets does not deliver frames
# evenly: its decoder stalls and then emits a burst. Measured on CAM-R06, which
# logged 219 packet-loss events in 30 minutes, the reader was discarding 11.85
# frames per second into a 3-deep queue while the inference loop got only 1.75
# frames per second out of it - against 7.1-8.2 on cameras with no loss. A
# queue that holds one burst lets the loop pick frames ACROSS the burst instead
# of seeing whatever single frame survived it. Costs 1920x1080x3 bytes per
# slot per camera (about 6 MB), so 8 slots on 10 cameras is under 500 MB.
QUEUE_SIZE = int(os.getenv("CCTV_QUEUE_SIZE", "8"))

USE_NVDEC = os.getenv("CCTV_NVDEC", "1") == "1"
FFMPEG_LOG = os.getenv("CCTV_FFMPEG_LOG", "0") == "1"

# Codec name (from ffprobe) -> NVDEC/CUVID decoder.
_CUVID = {"hevc": "hevc_cuvid", "h264": "h264_cuvid"}


class CameraStream:

    def __init__(
        self,
        camera_id,
        source,
        queue_size=None,
        loop_video=True,
    ):

        self.camera_id = camera_id
        self.source = source

        self.frame_queue = Queue(
            maxsize=QUEUE_SIZE if queue_size is None else queue_size)

        self.running = False
        self.thread = None

        self.frame_id = 0

        # For testing local video files (loop them forever).
        self.loop_video = loop_video

        # Decode plumbing
        self.proc = None
        self.width = None
        self.height = None
        self.codec = None
        self.decoder = None            # "NVDEC (hevc_cuvid)" etc., for reporting
        self.reconnect_delay = 2.0

        # Statistics
        self.frames_read = 0
        self.frames_dropped = 0

        # Probe-failure log throttle. A camera that is off writes one line every
        # reconnect_delay seconds otherwise - CAM-R16 alone put tens of
        # thousands of identical lines into the inference log while it was down,
        # which is how a real message gets missed.
        self._probe_fail_reason = None
        self._probe_fail_count = 0
        self._probe_fail_logged_at = 0.0

    # ------------------------------------------------------------------ probe
    def _is_rtsp(self):
        return self.source.lower().startswith(("rtsp://", "rtsps://"))

    def _probe(self):
        """Read width/height/codec once so we know the raw frame size."""
        cmd = [FFPROBE, "-v", "error"]
        if self._is_rtsp():
            cmd += ["-rtsp_transport", RTSP_TRANSPORT]
        cmd += [
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name",
            "-of", "json",
            self.source,
        ]
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=20
            ).stdout
            st = json.loads(out)["streams"][0]
            width = int(st["width"])
            height = int(st["height"])

            # A camera that is flapping can answer the probe while reporting no
            # picture size. Accepting 0x0 was a production outage on 2026-09-16:
            # frame_size became 0, _read_exact(0) returned instantly, and every
            # "frame" queued was an empty (0, 0, 3) array. The inference loop
            # fed one to YOLO and the whole process died with
            #   ValueError: Expected a single (H, W, C) image, but got array of
            #   shape (0, 0, 3)
            # after 3.5 h, taking every other camera down with it. The log line
            # that preceded it was "[CAM-R25] Connected (0x0 h264 -> NVDEC)".
            # An unusable size is a failed probe: the reader retries.
            if width <= 0 or height <= 0:
                self._note_probe_failure(f"probe returned {width}x{height} - "
                                         f"stream not ready, will retry")
                return False

            self.width = width
            self.height = height
            self.codec = st.get("codec_name", "hevc")
            self._probe_fail_reason = None
            self._probe_fail_count = 0
            return True
        except Exception as exc:
            self._note_probe_failure(f"probe failed: {exc}")
            return False

    def _note_probe_failure(self, message, every=60.0):
        """Log a probe failure, then repeat the same one at most once a minute."""
        now = time.time()
        if message != self._probe_fail_reason:
            self._probe_fail_reason = message
            self._probe_fail_count = 1
            self._probe_fail_logged_at = now
            print(f"[{self.camera_id}] {message}")
            return

        self._probe_fail_count += 1
        if now - self._probe_fail_logged_at < every:
            return

        repeats = self._probe_fail_count - 1
        since = now - self._probe_fail_logged_at
        self._probe_fail_count = 1
        self._probe_fail_logged_at = now
        print(f"[{self.camera_id}] {message} (still failing: "
              f"{repeats} more in the last {since:.0f}s)")

    # ------------------------------------------------------------------ ffmpeg
    def _ffmpeg_cmd(self):
        cuvid = _CUVID.get(self.codec) if USE_NVDEC else None
        self.decoder = f"NVDEC ({cuvid})" if cuvid else "CPU (ffmpeg)"

        cmd = [FFMPEG, "-nostdin", "-hide_banner",
               "-loglevel", "error",
               "-fflags", "nobuffer", "-flags", "low_delay"]

        # Loop local files for testing; irrelevant for live RTSP.
        if self.loop_video and not self._is_rtsp():
            cmd += ["-stream_loop", "-1"]

        # Input options are demuxer-specific.
        if self._is_rtsp():
            cmd += ["-rtsp_transport", RTSP_TRANSPORT]

        # Hardware decode (must precede -i to bind to this input).
        if cuvid:
            cmd += ["-hwaccel", "cuda", "-c:v", cuvid]

        cmd += [
            "-i", self.source,
            "-an",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]
        return cmd

    def connect(self):
        """Probe the stream and start the FFmpeg NVDEC decoder process."""
        self._close_proc()

        if not self._probe():
            return False

        try:
            self.proc = subprocess.Popen(
                self._ffmpeg_cmd(),
                stdout=subprocess.PIPE,
                stderr=(None if FFMPEG_LOG else subprocess.DEVNULL),
                bufsize=0,
            )
        except FileNotFoundError:
            print(f"[{self.camera_id}] ffmpeg not found on PATH")
            return False

        print(
            f"[{self.camera_id}] Connected "
            f"({self.width}x{self.height} {self.codec} -> {self.decoder})"
        )
        return True

    def _close_proc(self):
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=2)
            except Exception:
                pass
            self.proc = None

    # ------------------------------------------------------------------ read
    def _read_exact(self, n):
        """Read exactly n bytes from the pipe, or None if the stream ended."""
        buf = bytearray()
        stdout = self.proc.stdout
        while len(buf) < n:
            if not self.running:
                return None
            chunk = stdout.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def start(self):
        # Connect INSIDE the reader thread, not here. The probe blocks for the
        # full ffprobe timeout whenever a camera's MediaMTX source is momentarily
        # not ready (these cameras flap), and doing it synchronously would stall
        # the startup of every other camera queued behind it - which is exactly
        # why multicam_inf.py could hang on one flapping camera while the same
        # camera works fine in a standalone single-camera script. The reader
        # already (re)connects and retries, so startup stays instant and every
        # camera comes up independently and self-heals.
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:

            # (Re)establish the decoder if needed.
            if self.proc is None:
                if not self.connect():
                    time.sleep(self.reconnect_delay)
                    continue

            frame_size = (self.width or 0) * (self.height or 0) * 3

            # Belt and braces for the outage above: never pump from a zero-size
            # frame. _read_exact(0) returns an empty buffer without blocking, so
            # this loop would spin a core and flood the queue with empty frames.
            if frame_size <= 0:
                print(f"[{self.camera_id}] refusing to read {self.width}x{self.height} "
                      f"frames - reconnecting")
                self._close_proc()
                time.sleep(self.reconnect_delay)
                continue

            # Pump frames until the process dies.
            while self.running:
                data = self._read_exact(frame_size)

                if data is None:                       # ffmpeg exited / stream lost
                    break

                # bytearray backing => writable array, no extra copy needed.
                try:
                    frame = np.frombuffer(data, np.uint8).reshape(
                        self.height, self.width, 3
                    )
                except ValueError as exc:
                    # Byte count no longer matches the probed size (a camera
                    # that changed profile mid-stream). Reconnect and re-probe
                    # rather than letting this thread die and the camera go
                    # silent until the process restarts.
                    print(f"[{self.camera_id}] frame does not match "
                          f"{self.width}x{self.height} ({exc}) - reconnecting")
                    break

                self.frames_read += 1
                self.frame_id += 1

                packet = {
                    "camera_id": self.camera_id,
                    "frame_id": self.frame_id,
                    "timestamp": time.time(),
                    "frame": frame,
                }

                # Bounded queue: on overflow, drop the oldest and keep newest.
                try:
                    self.frame_queue.put_nowait(packet)
                except Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frames_dropped += 1
                    except Exception:
                        pass
                    try:
                        self.frame_queue.put_nowait(packet)
                    except Full:
                        self.frames_dropped += 1

            # Stream ended — tear down and reconnect (unless we're stopping).
            self._close_proc()
            if self.running:
                if not self._is_rtsp() and self.loop_video:
                    print(f"[{self.camera_id}] Video ended - restarting")
                else:
                    print(f"[{self.camera_id}] Stream lost - reconnecting")
                time.sleep(self.reconnect_delay)

    # ------------------------------------------------------------------ access
    def get_latest_frame(self):
        latest = None
        while True:
            try:
                latest = self.frame_queue.get_nowait()
            except Exception:
                break
        return latest

    def drain_frames(self, max_frames):
        """
        Empty the queue and return up to `max_frames` packets, OLDEST FIRST.

        Same drain as get_latest_frame() - nothing is left behind to go stale -
        but the caller can see more than the single newest frame, which is the
        only way a camera that arrives in bursts can be inferred more than once
        per burst. When more frames are queued than asked for, the NEWEST are
        kept: an old frame is worth less than a recent one to every analytic.
        """
        packets = []

        while True:
            try:
                packets.append(self.frame_queue.get_nowait())
            except Exception:
                break

        if max_frames is not None and len(packets) > max_frames:
            packets = packets[-max_frames:]

        return packets

    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except Exception:
            return None

    def queue_size(self):
        return self.frame_queue.qsize()

    def get_stats(self):
        return {
            "camera_id": self.camera_id,
            "frames_read": self.frames_read,
            "frames_dropped": self.frames_dropped,
            "queue_size": self.queue_size(),
            "decoder": self.decoder,
        }

    def stop(self):
        self.running = False
        self._close_proc()
        print(f"[{self.camera_id}] Stopped")
