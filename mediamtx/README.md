# MediaMTX — local RTSP broker

MediaMTX takes each classroom camera's stream from the NVR once and serves it locally, without credentials, to the AI inference process. It is **not a recorder**. The NVR stays the only recording system (about 7 days).

```
Hikvision NVR ──RTSP (credentials)──► relay-camera-0N ──RTSP publish──► MediaMTX ──► rtsp://mediamtx:8554/camera_0N
  channels 101/201/301                 (video only, remux)               (127.0.0.1 + dell_network only)
```

| Camera | NVR channel | Local URL (host) | Local URL (containers on `dell_network`) |
|---|---|---|---|
| camera_01 | 101 | `rtsp://127.0.0.1:8554/camera_01` | `rtsp://mediamtx:8554/camera_01` |
| camera_02 | 201 | `rtsp://127.0.0.1:8554/camera_02` | `rtsp://mediamtx:8554/camera_02` |
| camera_03 | 301 | `rtsp://127.0.0.1:8554/camera_03` | `rtsp://mediamtx:8554/camera_03` |

## Start, stop, check

```bash
cd /home/stack/Zayed_University/zayed_ai_inferencing
./mediamtx/render_config.sh                              # writes relay.env + mediamtx.runtime.yml (0600)
docker compose -f mediamtx/docker-compose.yml up -d --build
curl -s http://127.0.0.1:9997/v3/paths/list              # all three "ready": true
docker run --rm --gpus all --network dell_network -v $PWD/mediamtx:/v:ro \
    cctv/analytics-probe:25.11 python /v/validate_streams.py
docker compose -f mediamtx/docker-compose.yml down       # stop
```

`render_config.sh` generates a new relay password every time it runs. After re-rendering, restart the whole stack (`up -d --force-recreate`) so MediaMTX and the relays agree.

## Why there is a relay

MediaMTX cannot pull these streams directly. The NVR (firmware V4.1.62) describes each audio track as `MPEG4-GENERIC/32000` with no `fmtp` line. The `config` parameter that RFC 3640 requires is therefore missing, and MediaMTX rejects the whole session (`invalid SDP: media 2 is invalid: config is missing`).

FFmpeg tolerates this. So `relay/relay.py` (PyAV) pulls only the H.264 track, remuxes it without decoding or re-encoding, and publishes a valid video-only stream.

Two NVR-side changes would each let MediaMTX pull directly and remove the relay:

- disable audio on the three main streams; or
- a firmware update that emits a valid AAC `fmtp`.

Both are owner decisions. Disabling audio also stops the NVR recording audio.

## Security

- NVR credentials exist only in `relay.env` (0600, git-ignored). They never appear in git, in `mediamtx.yml`, or in any process's command line. This is verified: 0 matches in `ps` across the host.
- Only the relay account can publish, and only from `dell_network`. The runtime config stores its password as a SHA-256 hash. `overridePublisher: false`, so no other publisher can take over a path.
- Read access, the API and metrics are allowed from `127.0.0.1` and `172.18.0.0/16` only. Every published port is bound to `127.0.0.1`.
- Containers run non-root, with a read-only root filesystem and `no-new-privileges`. They restart `unless-stopped`.
- **Owner action still open:** the NVR account in `gb10_host_setup/nvr.env` is administrator-level. Create a view-only account and re-run `render_config.sh`.

## Recording policy

Recording (`record: false`) and the playback server (`playback: false`) are off. HLS, WebRTC, SRT, RTMP and MoQ are off too. No recordings directory exists, and the MediaMTX filesystem is read-only. For replay or evaluation, use NVR playback (`/Streaming/tracks/<ch>?starttime=…&endtime=…`), not local copies.

## Measured 2026-09-22 (validate_streams.py and restart tests)

| | camera_01 | camera_02 | camera_03 |
|---|---|---|---|
| Codec / resolution | H.264 Main, 2592×1944 | same | same |
| Frame rate (packets) | 30.0 fps | 30.0 fps | 30.0 fps |
| Keyframe interval | 8.5 s | 8.5 s | 8.5 s |
| NVDEC decode after first frame | 30.02 fps | 30.01 fps | 30.03 fps |
| Time to first decoded frame | 7.9 s | 6.8 s | 2.5 s |

- Restarting a relay: camera ready again in 14.3 s.
- Restarting MediaMTX: all three ready in 9.7 s.
- Both are bounded by the camera's 8.5 s GOP. Shortening the camera I-frame interval to 1–2 s is the biggest reconnect improvement available, and it is an owner/NVR setting.
