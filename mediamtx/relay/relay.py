"""
NVR -> MediaMTX video-only relay (one process per camera).

WHY THIS EXISTS. The Hikvision NVR (firmware V4.1.62) advertises each main
stream's audio track as "MPEG4-GENERIC/32000" with no fmtp line, which RFC
3640 requires ("config"). MediaMTX rejects the WHOLE session description for
that ("invalid SDP: media 2 is invalid: config is missing"), so it cannot pull
these streams itself. FFmpeg's RTSP client tolerates the SDP. This relay pulls
the camera's H.264 track only, REMUXES it (no decode, no re-encode) and
publishes a valid video-only stream into MediaMTX, which then serves every
local reader. The NVR remains the only recorder; nothing is written to disk.

CREDENTIALS. The NVR URL and the MediaMTX publish password come from the
environment (a 0600 env file rendered by render_config.sh). They never appear
in this process's command line, and every URL in the log is redacted.

    CAMERA=camera_01  CAMERA_01_SOURCE=rtsp://...  RELAY_PUBLISH_USER=...  RELAY_PUBLISH_PASS=...
    MEDIAMTX_RTSP=rtsp://mediamtx:8554
"""

import os
import re
import sys
import time
import urllib.parse

import av

CAMERA = os.environ["CAMERA"]
SOURCE = os.environ[f"{CAMERA.upper()}_SOURCE"]
BASE = os.environ.get("MEDIAMTX_RTSP", "rtsp://mediamtx:8554").rstrip("/")
USER = urllib.parse.quote(os.environ["RELAY_PUBLISH_USER"], safe="")
PASS = urllib.parse.quote(os.environ["RELAY_PUBLISH_PASS"], safe="")
TARGET = BASE.replace("rtsp://", f"rtsp://{USER}:{PASS}@", 1) + f"/{CAMERA}"
OPEN_TIMEOUT = float(os.environ.get("RELAY_OPEN_TIMEOUT", "15"))
READ_TIMEOUT = float(os.environ.get("RELAY_READ_TIMEOUT", "15"))
MAX_BACKOFF = float(os.environ.get("RELAY_MAX_BACKOFF", "30"))


def redact(text):
    return re.sub(r"(rtsp://)[^/@\s]+@", r"\1***@", str(text))


def log(message):
    print(f"[relay {CAMERA}] {redact(message)}", flush=True)


def relay_once():
    """One session: returns packets relayed; raises on any failure."""
    source = av.open(SOURCE, options={"rtsp_transport": "tcp", "analyzeduration": "2000000",
                                      "probesize": "2000000"},
                     timeout=(OPEN_TIMEOUT, READ_TIMEOUT))
    try:
        video = source.streams.video[0]
        target = av.open(TARGET, mode="w", format="rtsp", options={"rtsp_transport": "tcp"},
                         timeout=(OPEN_TIMEOUT, READ_TIMEOUT))
        try:
            out = target.add_stream_from_template(video)
            log(f"source open: {video.codec_context.name} {video.codec_context.width}x"
                f"{video.codec_context.height} - waiting for the first keyframe")
            started, last_dts, relayed, dropped = False, None, 0, 0
            for packet in source.demux(video):
                if packet.dts is None or packet.size == 0:
                    continue
                if not started:
                    if not packet.is_keyframe:
                        continue
                    started = True
                    log("first keyframe - publishing to MediaMTX")
                if last_dts is not None and packet.dts <= last_dts:
                    dropped += 1                  # out-of-order after loss; the muxer would abort
                    continue
                last_dts = packet.dts
                packet.stream = out
                target.mux(packet)
                relayed += 1
                if relayed % 9000 == 0:
                    log(f"relayed {relayed} packets ({dropped} out-of-order dropped)")
            return relayed
        finally:
            target.close()
    finally:
        source.close()


def main():
    log(f"starting: {SOURCE} -> {TARGET}")
    backoff = 1.0
    while True:
        started_at = time.monotonic()
        try:
            relayed = relay_once()
            log(f"source ended after {relayed} packets - reconnecting")
        except Exception as exc:                          # noqa: BLE001
            log(f"session failed: {type(exc).__name__}: {exc}")
        if time.monotonic() - started_at > 60:
            backoff = 1.0                                  # it had been running; retry fast
        time.sleep(backoff)
        backoff = min(backoff * 2, MAX_BACKOFF)


if __name__ == "__main__":
    sys.exit(main())
