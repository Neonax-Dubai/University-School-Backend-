#!/usr/bin/env python3
"""
Validate the three MediaMTX streams before inference connects to them.

Runs inside cctv/analytics-probe:25.11 on dell_network with the GPU:

    docker run --rm --gpus all --network dell_network \
        -v $PWD/mediamtx:/v:ro cctv/analytics-probe:25.11 python /v/validate_streams.py

Reads only the credential-free local URLs (rtsp://mediamtx:8554/camera_0N).
Per camera: codec, resolution, frame rate, keyframe interval (from packet
flags over a sample window), time to first decoded frame and sustained NVDEC
decode rate. Prints JSON; nothing is written to disk.
"""
import json
import os
import re
import subprocess
import sys
import time

BASE = os.environ.get("MEDIAMTX_RTSP", "rtsp://mediamtx:8554")
CAMERAS = os.environ.get("CAMERAS", "camera_01,camera_02,camera_03").split(",")
SAMPLE_SECONDS = float(os.environ.get("SAMPLE_SECONDS", "20"))


def probe(url):
    out = subprocess.run(["ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-select_streams", "v:0",
                          "-show_entries", "stream=codec_name,profile,width,height,r_frame_rate,avg_frame_rate,pix_fmt",
                          "-of", "json", url], capture_output=True, text=True, timeout=40)
    streams = json.loads(out.stdout or "{}").get("streams") or [{}]
    return streams[0], out.stderr.strip()[-300:]


def keyframes(url, seconds):
    out = subprocess.run(["ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-select_streams", "v:0",
                          "-read_intervals", f"%+{seconds}", "-show_entries", "packet=pts_time,flags",
                          "-of", "csv=p=0", url], capture_output=True, text=True, timeout=seconds + 40)
    rows = [line.split(",")[:2] for line in out.stdout.splitlines() if "," in line]
    pts = [(float(t), "K" in f) for t, f in rows if t not in ("", "N/A")]
    keys = [t for t, k in pts if k]
    intervals = [round(b - a, 2) for a, b in zip(keys, keys[1:])]
    span = (pts[-1][0] - pts[0][0]) if len(pts) > 1 else 0.0
    return {"packets": len(pts), "span_s": round(span, 2),
            "packet_rate_fps": round((len(pts) - 1) / span, 2) if span > 0 else None,
            "keyframes": len(keys), "keyframe_intervals_s": intervals[:10]}


def nvdec(url, seconds):
    """Decode with NVDEC; report wall time to the first frame and the rate after it."""
    started = time.monotonic()
    proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error",
                             "-rtsp_transport", "tcp", "-hwaccel", "cuda", "-c:v", "h264_cuvid", "-i", url,
                             "-f", "null", "-", "-progress", "pipe:1"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    first_at, first_frames, last_at, last_frames = None, 0, None, 0
    try:
        for line in proc.stdout:
            if not line.startswith("frame="):
                continue
            frames, now = int(line.split("=", 1)[1]), time.monotonic()
            if frames > 0 and first_at is None:
                first_at, first_frames = now, frames
            if first_at is not None:
                last_at, last_frames = now, frames
                if now - first_at >= seconds:
                    break
            if now - started > seconds + 30:
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    if first_at is None:
        return {"ok": False, "error": (proc.stderr.read() or "no frame decoded")[-300:]}
    span = (last_at - first_at) if last_at else 0.0
    return {"ok": True, "time_to_first_frame_s": round(first_at - started, 2),
            "frames_after_first": last_frames - first_frames, "measured_s": round(span, 2),
            "decode_fps_after_first_frame": round((last_frames - first_frames) / span, 2) if span > 0 else None}


def main():
    report = {"base": BASE, "cameras": {}}
    for camera in CAMERAS:
        url = f"{BASE}/{camera}"
        t0 = time.monotonic()
        stream, err = probe(url)
        entry = {"url": url, "probe_s": round(time.monotonic() - t0, 2), "stream": stream}
        if err:
            entry["probe_error"] = err
        entry["gop"] = keyframes(url, SAMPLE_SECONDS)
        entry["nvdec"] = nvdec(url, 15)
        report["cameras"][camera] = entry
    print(json.dumps(report, indent=2))
    return 0 if all(c["nvdec"]["ok"] for c in report["cameras"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
