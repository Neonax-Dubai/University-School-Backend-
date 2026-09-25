"""
Live GPU check for the internal face API: real buffalo_l, real CUDA, real image.
Read-only - no Qdrant, no dashboard, no production data. Run by run_live_gpu.sh.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, "/app")
import internal_api as api                                      # noqa: E402

IMAGE = sys.argv[1]
BASELINE = sys.argv[2] if len(sys.argv) > 2 else ""
TOKEN = "live-check-token"
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{(' - ' + detail) if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


service = api.FaceService(log=lambda m: print(m, flush=True))
server = api.InternalFaceAPI(log=lambda m: print(m, flush=True), service=service,
                             host="127.0.0.1", port=0, token=TOKEN)
server.start()
port = server._server.server_address[1]


def call(path, data=None, ctype=None, token=TOKEN):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if ctype:
        request.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


deadline = time.time() + 180
status, health = 0, {}
while time.time() < deadline:
    status, health = call("/internal/face/health")
    if status == 200:
        break
    time.sleep(2)

check("health returns 200 once the GPU model is loaded", status == 200, json.dumps(health)[:160])
check("health reports CUDAExecutionProvider", health.get("provider") == "CUDAExecutionProvider",
      str(health.get("providers")))
check("health declares gpu_required and no cpu_fallback",
      health.get("gpu_required") is True and health.get("cpu_fallback") is False)
check("health reports embedding_dim 512", health.get("embedding_dim") == 512)
check("unauthenticated health is refused", call("/internal/face/health", token=None)[0] == 401)


def multipart(path, profile):
    boundary = "----zayedlive"
    with open(path, "rb") as handle:
        blob = handle.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"profile\"\r\n\r\n{profile}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"u.jpg\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + blob + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


body, ctype = multipart(IMAGE, "search")
started = time.perf_counter()
status, payload = call("/internal/face/embed", data=body, ctype=ctype)
elapsed = (time.perf_counter() - started) * 1000
check("embed returns HTTP 200", status == 200, str(payload)[:160])
vector = np.asarray(payload.get("embedding") or [], dtype=np.float32)
check("embedding is 512-dimensional", vector.shape == (512,), str(vector.shape))
check("embedding is L2-normalised", abs(float(np.linalg.norm(vector)) - 1.0) < 1e-4,
      f"norm={float(np.linalg.norm(vector)):.6f}")
check("embedding was computed on CUDA", payload.get("provider") == "CUDAExecutionProvider",
      str(payload.get("provider")))
check("bbox and det_score are reported", bool(payload.get("bbox")) and payload.get("det_score") is not None,
      f"bbox={payload.get('bbox')} det={payload.get('det_score')} sharpness={payload.get('sharpness')}")
print(f"      faces={payload.get('faces_detected')} model={payload.get('model')} "
      f"api_latency_ms={payload.get('latency_ms')} round_trip_ms={elapsed:.0f}", flush=True)

body, ctype = multipart(IMAGE, "enroll")
status, enrolled = call("/internal/face/embed", data=body, ctype=ctype)
check("enroll profile detects at least as many faces as search",
      status == 200 and enrolled.get("faces_detected", 0) >= payload.get("faces_detected", 0),
      f"enroll={enrolled.get('faces_detected')} search={payload.get('faces_detected')}")
check("the two profiles embed the same largest face",
      enrolled.get("bbox") == payload.get("bbox"), f"{enrolled.get('bbox')} vs {payload.get('bbox')}")

if BASELINE and os.path.exists(BASELINE):
    baseline = np.load(BASELINE)
    cosine = float(vector @ baseline)
    check("embedding matches the pre-change inference embedding (cosine >= 0.999)", cosine >= 0.999,
          f"cosine={cosine:.6f}")
else:
    print(f"      (no baseline vector at {BASELINE!r} - compatibility comparison skipped)", flush=True)

server.stop()
print(f"\n{'ALL LIVE GPU CHECKS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}", flush=True)
sys.exit(1 if failures else 0)
