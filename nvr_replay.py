#!/usr/bin/env python3
"""
NVR Time-Range Replay Script
==============================
Fetch a specific 1-hour (or any duration) time window from a Hikvision NVR
and replay it through one of the three detection test scripts.

How it works
------------
1. Queries the NVR ISAPI ContentMgmt/search to find physical recording files
   covering the requested window.
2. Downloads the covering file(s) once via ContentMgmt/download (identical
   mechanism to extract_clips.py - no RTSP playback session, much kinder to
   the NVR, byte-accurate seek).
3. Cuts each covering file to the exact requested window with ffmpeg -c copy
   (no re-encode, frame-identical output).
4. Runs the chosen detection script against the stitched clip.

Usage
-----
    # Replay the 09:00-10:00 window from channel 101, run fall detection
    python nvr_replay.py \\
        --start "2026-09-28 09:00:00" \\
        --end   "2026-09-28 10:00:00" \\
        --channel 101 \\
        --analytic fall

    # Fight detection, save annotated output, no display
    python nvr_replay.py \\
        --start "2026-09-28 14:30:00" \\
        --end   "2026-09-28 15:30:00" \\
        --channel 201 --analytic fight \\
        --output /tmp/fight_replay.mp4 --headless

    # Sleep detection, keep downloaded NVR file for later
    python nvr_replay.py \\
        --start "2026-09-28 08:00:00" \\
        --end   "2026-09-28 09:00:00" \\
        --channel 101 --analytic sleep \\
        --keep-source

    # NVR credentials via env (preferred; avoids them appearing in process list)
    export NVR_HOST=10.232.7.x NVR_USER=admin NVR_PASS=secret
    python nvr_replay.py --start "..." --end "..." --channel 101 --analytic fall

    # Or point at an nvr.env file
    python nvr_replay.py ... --nvr-env /path/to/nvr.env

NVR credential lookup order
    1. --nvr-env file (KEY=VALUE, one per line)
    2. NVR_HOST / NVR_USER / NVR_PASS environment variables

Requirements: requests, ffmpeg in PATH (same as extract_clips.py)
"""
import argparse, datetime, html, json, os, re, shutil, subprocess, sys, uuid
import requests
from requests.auth import HTTPDigestAuth

HERE = os.path.dirname(os.path.abspath(__file__))
NS   = "http://www.hikvision.com/ver20/XMLSchema"

# How many extra seconds to add to the tail when cutting (NVR MPEG-PS tail loss)
TAIL_SLACK = 15.0

ANALYTICS = {
    "fall":  ("test_fall_detection.py",  "[FALL-TEST]"),
    "fight": ("test_fight_detection.py", "[FIGHT-TEST]"),
    "sleep": ("test_sleep_detection.py", "[SLEEP-TEST]"),
    # full pipeline: all Zayed analytics simultaneously (via replay_zayed_inference.py)
    "full":  ("replay_zayed_inference.py", "[FULL-REPLAY]"),
}


# ── NVR helpers (ported from extract_clips.py) ────────────────────────────

def load_env(path):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def _parse_nvr_time(text):
    """NVR labels timestamps Z but runs on local time."""
    return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def _fmt_nvr(ts):
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def search_files(session, host, auth, channel, start_ts, end_ts):
    """Return physical NVR files overlapping [start_ts, end_ts]."""
    day_s = datetime.datetime.fromtimestamp(start_ts).replace(
        hour=0, minute=0, second=0)
    day_e = datetime.datetime.fromtimestamp(end_ts).replace(
        hour=23, minute=59, second=0)
    body = (
        f'<?xml version="1.0" encoding="utf-8"?>'
        f'<CMSearchDescription version="2.0" xmlns="{NS}">'
        f'<searchID>{uuid.uuid4()}</searchID>'
        f'<trackList><trackID>{channel}</trackID></trackList>'
        f'<timeSpanList><timeSpan>'
        f'<startTime>{day_s.strftime("%Y-%m-%dT%H:%M:%SZ")}</startTime>'
        f'<endTime>{day_e.strftime("%Y-%m-%dT%H:%M:%SZ")}</endTime>'
        f'</timeSpan></timeSpanList>'
        f'<maxResults>200</maxResults><searchResultPostion>0</searchResultPostion>'
        f'<metadataList><metadataDescriptor>'
        f'//recordType.meta.std-cgi.com'
        f'</metadataDescriptor></metadataList>'
        f'</CMSearchDescription>'
    )
    r = session.post(f"http://{host}/ISAPI/ContentMgmt/search", data=body,
                     auth=auth, headers={"Content-Type":"application/xml"}, timeout=60)
    r.raise_for_status()
    files = []
    for item in re.findall(r"<searchMatchItem>(.*?)</searchMatchItem>", r.text, re.S):
        uri   = html.unescape(re.search(r"<playbackURI>(.*?)</playbackURI>", item).group(1))
        name  = re.search(r"name=([^&]+)", uri)
        fst   = _parse_nvr_time(re.search(r"<startTime>(.*?)</startTime>", item).group(1))
        fen   = _parse_nvr_time(re.search(r"<endTime>(.*?)</endTime>",   item).group(1))
        if fen <= start_ts or fst >= end_ts:
            continue
        files.append({
            "name":  name.group(1) if name else uuid.uuid4().hex,
            "start": fst, "end": fen, "uri": uri,
        })
    return files


def download_file(session, host, auth, entry, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
        print(f"  [NVR] already cached: {os.path.basename(out_path)}")
        return True
    body = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<downloadRequest version="1.0" xmlns="{NS}">\n'
        f'<playbackURI>{entry["uri"].replace("&","&amp;")}</playbackURI>\n'
        '</downloadRequest>'
    )
    span_s = datetime.datetime.fromtimestamp(entry["start"]).strftime("%H:%M:%S")
    span_e = datetime.datetime.fromtimestamp(entry["end"]).strftime("%H:%M:%S")
    print(f"  [NVR] downloading {os.path.basename(out_path)} "
          f"({span_s}-{span_e}) ...", end="", flush=True)
    r = session.post(f"http://{host}/ISAPI/ContentMgmt/download", data=body,
                     auth=auth, headers={"Content-Type":"application/xml"},
                     timeout=900, stream=True)
    if r.status_code != 200:
        print(f" HTTP {r.status_code}")
        return False
    written = 0
    with open(out_path + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk); written += len(chunk)
    os.replace(out_path + ".part", out_path)
    print(f" {written/1e6:.0f} MB")
    return True


def cut_clip(source, offset_s, duration_s, out_path):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{max(0,offset_s):.3f}", "-i", source,
           "-t",  f"{duration_s + TAIL_SLACK:.3f}",
           "-map", "0:v:0", "-c", "copy",
           "-avoid_negative_ts", "make_zero",
           "-movflags", "+faststart", out_path]
    done = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=duration_s * 4 + 300)
    return done.returncode, (done.stderr or "").strip()


def stitch_clips(clips, out_path):
    """Concatenate multiple clips (stream-copy) into one file."""
    if len(clips) == 1:
        shutil.copy2(clips[0], out_path)
        return
    concat_list = out_path + ".ffconcat"
    with open(concat_list, "w") as f:
        f.write("ffconcat version 1.0\n")
        for c in clips:
            f.write(f"file '{c}'\n")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "concat", "-safe", "0", "-i", concat_list,
           "-c", "copy", "-movflags", "+faststart", out_path]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    os.unlink(concat_list)
    return done.returncode, (done.stderr or "").strip()


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Fetch a Hikvision NVR time window and replay it through a detection script")
    ap.add_argument("--start",    required=True,
                    help='Start time  e.g. "2026-09-28 09:00:00"')
    ap.add_argument("--end",      required=True,
                    help='End time    e.g. "2026-09-28 10:00:00"')
    ap.add_argument("--channel",  default="101",
                    help="Primary NVR track/channel number (default 101). "
                         "For --analytic full use --channels instead.")
    ap.add_argument("--channels", default="",
                    help="Comma-separated channels for full-pipeline replay, "
                         'e.g. "101,201,301".  Each maps to cam_01, cam_02 ...  '
                         "Overrides --channel when --analytic full is used.")
    ap.add_argument("--analytic", choices=["fall","fight","sleep","full"], default="fall",
                    help="Which script to run (default: fall).  "
                         "'full' runs the entire Zayed inference pipeline.")
    ap.add_argument("--config",   default="",
                    help="JSON config file for --analytic full "
                         "(camera features, zones, classroom).  Optional.")
    ap.add_argument("--output",   default="",
                    help="Save annotated output MP4 to this path (single-analytic modes)")
    ap.add_argument("--output-dir", default="",
                    help="Directory for per-camera annotated MP4s (--analytic full only)")
    ap.add_argument("--headless", action="store_true",
                    help="No display window")
    ap.add_argument("--save-stills", action="store_true",
                    help="Save JPEG per alert event (single-analytic modes)")
    ap.add_argument("--keep-source", action="store_true",
                    help="Keep downloaded NVR files after cutting")
    ap.add_argument("--nvr-env",  default="",
                    help="Path to nvr.env file with NVR_HOST/NVR_USER/NVR_PASS")
    ap.add_argument("--work-dir", default="",
                    help="Where to write temp files (default: ./nvr_replay_tmp)")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("[NVR-REPLAY] ffmpeg not found in PATH")

    # ── credentials ─────────────────────────────────────────────────────
    if args.nvr_env:
        env = load_env(args.nvr_env)
    else:
        env = {}
    host  = env.get("NVR_HOST",  os.getenv("NVR_HOST",  ""))
    user  = env.get("NVR_USER",  os.getenv("NVR_USER",  ""))
    pw    = env.get("NVR_PASS",  os.getenv("NVR_PASS",  ""))
    if not host:
        sys.exit("[NVR-REPLAY] NVR_HOST not set. "
                 "Use --nvr-env or export NVR_HOST/NVR_USER/NVR_PASS.")

    # ── time window ─────────────────────────────────────────────────────
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        start_ts = datetime.datetime.strptime(args.start, fmt).timestamp()
        end_ts   = datetime.datetime.strptime(args.end,   fmt).timestamp()
    except ValueError:
        sys.exit(f'[NVR-REPLAY] Use time format "YYYY-MM-DD HH:MM:SS"')
    if end_ts <= start_ts:
        sys.exit("[NVR-REPLAY] --end must be after --start")
    duration = end_ts - start_ts
    # resolve channel list
    if args.analytic == "full" and args.channels:
        channel_list = [c.strip() for c in args.channels.split(",") if c.strip()]
    else:
        channel_list = [args.channel]
    print(f"[NVR-REPLAY] Window: {args.start} -> {args.end}  "
          f"({duration/3600:.2f} h)  channels={','.join(channel_list)}  "
          f"analytic={args.analytic}")

    # ── work directory ──────────────────────────────────────────────────
    work = args.work_dir or os.path.join(HERE, "nvr_replay_tmp")
    os.makedirs(work, exist_ok=True)
    src_dir = os.path.join(work, "source")
    os.makedirs(src_dir, exist_ok=True)

    # ── query NVR + download + cut  (one loop per channel) ──────────────
    session = requests.Session()
    auth    = HTTPDigestAuth(user, pw)
    # channel_id -> ready clip path
    channel_clips = {}   # {channel: clip_path}
    sources_used  = []

    for channel in channel_list:
        print(f"\n[NVR-REPLAY] === channel {channel} ===")
        try:
            files = search_files(session, host, auth, channel, start_ts, end_ts)
        except requests.RequestException as exc:
            print(f"  search failed: {exc}")
            continue
        if not files:
            print(f"  no recordings found for this channel.")
            continue
        print(f"  {len(files)} recording file(s) found.")

        downloaded = []
        for entry in files:
            path = os.path.join(src_dir, f"ch{channel}_{entry['name']}.dav")
            if download_file(session, host, auth, entry, path):
                entry["path"] = path
                downloaded.append(entry)
                sources_used.append(path)
            else:
                print(f"  WARN: could not download {entry['name']}, skipping.")

        if not downloaded:
            print(f"  all downloads failed for channel {channel}.")
            continue

        clips = []
        for entry in downloaded:
            clip_start = max(start_ts, entry["start"])
            clip_end   = min(end_ts,   entry["end"])
            offset     = clip_start - entry["start"]
            clip_dur   = clip_end - clip_start
            ts_str     = datetime.datetime.fromtimestamp(clip_start).strftime("%H%M%S")
            clip_path  = os.path.join(work, f"clip_ch{channel}_{ts_str}_{int(clip_dur)}s.mp4")
            print(f"  Cutting {os.path.basename(clip_path)} "
                  f"(offset={offset:.0f}s  dur={clip_dur:.0f}s) ...", end="", flush=True)
            rc_cut, err = cut_clip(entry["path"], offset, clip_dur, clip_path)
            if rc_cut != 0:
                print(f" FAILED: {err[:120]}")
            else:
                print(f" {os.path.getsize(clip_path)/1e6:.0f} MB")
                clips.append(clip_path)

        if not clips:
            print(f"  all cuts failed for channel {channel}.")
            continue

        if len(clips) > 1:
            merged = os.path.join(work, f"merged_ch{channel}.mp4")
            print(f"  Stitching {len(clips)} clips -> {merged} ...")
            stitch_clips(clips, merged)
            channel_clips[channel] = merged
        else:
            channel_clips[channel] = clips[0]

    if not channel_clips:
        sys.exit("[NVR-REPLAY] No clips produced for any channel.")

    # For backward-compat with single-analytic modes, expose a single clip
    clips = list(channel_clips.values())
    replay_input = clips[0]   # primary clip (single-channel modes)

    # ── launch detection script ─────────────────────────────────────────
    print(f"\n[NVR-REPLAY] Launching {args.analytic} detection ...\n")
    for cid, path in channel_clips.items():
        print(f"[NVR-REPLAY] Channel {cid} -> {path}")

    script_name, _prefix = ANALYTICS[args.analytic]
    script_path = os.path.join(HERE, script_name)
    if not os.path.exists(script_path):
        sys.exit(f"[NVR-REPLAY] Detection script not found: {script_path}\n"
                 f"Make sure {script_name} is in the same directory.")

    if args.analytic == "full":
        # replay_zayed_inference.py expects --clips cam_id:path ...
        # Map channel numbers to camera IDs: 101 -> cam_101, etc.
        clip_args = []
        for channel, clip_path in channel_clips.items():
            cam_id = f"cam_{channel}" if str(channel).isdigit() else channel
            clip_args.append(f"{cam_id}:{clip_path}")
        cmd = [sys.executable, script_path, "--clips"] + clip_args
        if args.config:
            cmd += ["--config", args.config]
        if getattr(args, "output_dir", ""):
            cmd += ["--output-dir", args.output_dir]
        if args.headless:
            cmd.append("--headless")
    else:
        # Single-analytic modes: use only the primary (first) clip
        cmd = [sys.executable, script_path, "--input", replay_input]
        if args.output:
            cmd += ["--output", args.output]
        if args.headless:
            cmd.append("--headless")
        if args.save_stills:
            cmd.append("--save-stills")

    print(f"[NVR-REPLAY] Running: {' '.join(cmd)}\n")
    rc = subprocess.call(cmd)

    # ── cleanup ─────────────────────────────────────────────────────────
    if not args.keep_source:
        for p in sources_used:
            if p and os.path.exists(p):
                try: os.unlink(p)
                except OSError: pass
        print("[NVR-REPLAY] Source NVR files removed (--keep-source to keep).")

    print(f"[NVR-REPLAY] Detection script exited with code {rc}.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
