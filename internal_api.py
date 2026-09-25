"""
Internal face API (Zayed) - GPU-only face detection + embedding for the dashboard.

WHY THIS EXISTS. The dashboard owns enrollment and photo search, but it has no GPU: its
InsightFace fell back to CPU (0.9-2.2 s per photo) and cost ~2 GiB RSS in every gunicorn
worker. This moves ONLY the ML computation to the process that already has the GPU. Everything
the dashboard owns stays in the dashboard: face selection rules, quality gates, Qdrant writes
and reads, PersonProfile, enrollment workflow, FaceSearchJob.

    POST /internal/face/embed    multipart: image=<file>, profile=enroll|search
    GET  /internal/face/health

NETWORK. Bound inside the container only and never published to the host, so it is reachable
only from the Docker network the dashboard shares with this process. Every request must carry
`Authorization: Bearer <token>`; the token is read from INTERNAL_FACE_TOKEN_FILE (0600), never
from source, argv or the image.

GPU IS MANDATORY. The session is created with CUDAExecutionProvider ALONE, and every loaded
model's effective provider is then verified. If CUDA is not the provider the API refuses to
serve: /health answers 503 and /embed answers 503. It never computes an embedding on the CPU
and never reports healthy while degraded. The CCTV pipeline is untouched by that failure.

MODEL AND CONCURRENCY. Exactly ONE additional buffalo_l instance is loaded, at det_thresh 0.5
(the enroll profile, the more sensitive of the two); the search profile filters the same
detector output to >= 0.6 in Python, which is equivalent to detecting at 0.6. The CCTV
FaceIDManager instance is deliberately NOT shared: its det_thresh is baked in at prepare() time
(0.6), so serving the enroll profile from it would mean mutating state the camera worker is
using. Requests are serialised through a semaphore of one with a bounded wait
(INTERNAL_FACE_WAIT_SECONDS); a request that cannot get the slot in time is answered 429 rather
than queued, so an operator upload can never stall camera processing.

The uploaded image is decoded in memory and dropped when the response is written. Nothing is
stored on disk, and no Qdrant client is imported here.
"""
import hmac
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

ENABLED = os.getenv("INTERNAL_FACE_API", "1") == "1"
BIND_HOST = os.getenv("INTERNAL_FACE_HOST", "0.0.0.0")          # container-internal; never published
BIND_PORT = int(os.getenv("INTERNAL_FACE_PORT", "8080"))
TOKEN_FILE = os.getenv("INTERNAL_FACE_TOKEN_FILE", "")
MAX_UPLOAD_BYTES = int(os.getenv("INTERNAL_FACE_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
WAIT_SECONDS = float(os.getenv("INTERNAL_FACE_WAIT_SECONDS", "5.0"))
DET_SIZE = int(os.getenv("INTERNAL_FACE_DET_SIZE", "640"))
ENROLL_DET_THRESHOLD = float(os.getenv("INTERNAL_FACE_ENROLL_DET_THRESHOLD", "0.5"))
SEARCH_DET_THRESHOLD = float(os.getenv("INTERNAL_FACE_SEARCH_DET_THRESHOLD", "0.6"))
MODEL_NAME = "buffalo_l"
MODEL_ID = "insightface:buffalo_l:w600k_r50"
EMBEDDING_DIM = 512
CUDA_PROVIDER = "CUDAExecutionProvider"
PROFILES = {"enroll": ENROLL_DET_THRESHOLD, "search": SEARCH_DET_THRESHOLD}


class CudaUnavailable(RuntimeError):
    """The face models are not on the GPU. Never downgrade to CPU - fail instead."""


def read_token(path=None):
    path = TOKEN_FILE if path is None else path
    if not path:
        return ""
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return ""


def verify_cuda(app):
    """Every loaded InsightFace session must actually be on CUDA. Raises CudaUnavailable."""
    models = getattr(app, "models", None) or {}
    if not models:
        raise CudaUnavailable("no InsightFace model sessions were loaded")
    effective = {}
    for task, model in models.items():
        providers = list(model.session.get_providers() or [])
        effective[task] = providers[0] if providers else "none"
        if not providers or providers[0] != CUDA_PROVIDER:
            raise CudaUnavailable(
                f"GPU-only face inference required: {task} model is on {providers} "
                f"({CUDA_PROVIDER} unavailable) - refusing to serve on the CPU")
    return effective


def sharpness_of(image, bbox):
    """Variance of the Laplacian over the face box - the same blur proxy face_id_manager uses.
    Computed HERE because the dashboard no longer opens the image at all."""
    try:
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(image.shape[1], int(bbox[2])), min(image.shape[0], int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image[y1:y2, x1:x2]
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return round(float(cv2.Laplacian(grey, cv2.CV_64F).var()), 4)
    except Exception:                                    # noqa: BLE001 - a metric never breaks a call
        return None


def normalise_gender(raw):
    """InsightFace emits 0=female, 1=male - the same mapping the dashboard already stores."""
    if raw is None:
        return None
    try:
        return {0: "female", 1: "male"}.get(int(raw))
    except (TypeError, ValueError):
        return None


def parse_multipart(body, content_type):
    """Minimal multipart/form-data reader: {field: bytes}. Raises ValueError on malformed input."""
    marker = "boundary="
    if marker not in (content_type or ""):
        raise ValueError("expected multipart/form-data with a boundary")
    boundary = content_type.split(marker, 1)[1].split(";")[0].strip().strip('"')
    if not boundary:
        raise ValueError("empty multipart boundary")
    sep = b"--" + boundary.encode()
    fields = {}
    for part in body.split(sep):
        if not part or part in (b"--\r\n", b"--", b"\r\n"):
            continue
        head, _, payload = part.partition(b"\r\n\r\n")
        if not _:
            continue
        name = None
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-disposition:") and b"name=" in line:
                name = line.split(b'name="', 1)[1].split(b'"', 1)[0].decode("utf-8", "replace")
                break
        if name:
            fields[name] = payload.rstrip(b"\r\n")
    if not fields:
        raise ValueError("no multipart fields found")
    return fields


class FaceService:
    """The one extra buffalo_l instance, loaded on the GPU, serialised by a bounded semaphore."""

    def __init__(self, log=print, app_factory=None):
        self._log = log
        self._app_factory = app_factory or self._build_app
        self._app = None
        self._providers = {}
        self._error = None
        self._lock = threading.Lock()
        self._slot = threading.Semaphore(1)
        self.requests = self.failures = self.busy_rejections = 0
        self.loaded_at = None

    # ------------------------------------------------------------------ model
    @staticmethod
    def _build_app():
        from insightface.app import FaceAnalysis
        # CUDA ALONE - no CPUExecutionProvider in the list. If CUDA cannot initialise, ONNX
        # Runtime falls back internally and verify_cuda() below catches it.
        app = FaceAnalysis(name=MODEL_NAME, providers=[CUDA_PROVIDER],
                           allowed_modules=["detection", "recognition", "genderage"])
        app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE), det_thresh=ENROLL_DET_THRESHOLD)
        return app

    def load(self):
        """Build and GPU-verify the model. Returns True when the API may serve."""
        with self._lock:
            if self._app is not None:
                return True
            try:
                app = self._app_factory()
                self._providers = verify_cuda(app)
                self._app = app
                self._error = None
                self.loaded_at = time.time()
                self._log(f"[FACE-API] ready on {CUDA_PROVIDER}: {MODEL_ID} det_size={DET_SIZE} "
                          f"det_thresh={ENROLL_DET_THRESHOLD} (search filters to {SEARCH_DET_THRESHOLD})")
                return True
            except Exception as exc:                      # noqa: BLE001 - reported, never hidden
                self._error = f"{type(exc).__name__}: {exc}"
                self._app = None
                self._log(f"[FACE-API] UNHEALTHY - face API will refuse every request: {self._error}")
                return False

    def health(self):
        ready = self._app is not None and not self._error
        return {
            "status": "ok" if ready else "unavailable",
            "model": MODEL_NAME,
            "model_id": MODEL_ID,
            "provider": (self._providers.get("recognition") if ready else None),
            "providers": self._providers if ready else {},
            "gpu_required": True,
            "cpu_fallback": False,
            "embedding_dim": EMBEDDING_DIM,
            "profiles": PROFILES,
            "loaded_at": self.loaded_at,
            "requests": self.requests,
            "failures": self.failures,
            "busy_rejections": self.busy_rejections,
            "error": self._error,
        }, (200 if ready else 503)

    # ------------------------------------------------------------------ work
    def embed(self, image_bytes, profile):
        """(payload, status). Detection + embedding only - no business rules, no storage."""
        if profile not in PROFILES:
            return {"error": f"profile must be one of {sorted(PROFILES)}"}, 400
        if self._app is None and not self.load():
            return {"error": "face API unavailable (GPU)", "detail": self._error}, 503
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "could not decode the uploaded image"}, 400

        if not self._slot.acquire(timeout=WAIT_SECONDS):
            self.busy_rejections += 1
            return {"error": "face API busy", "retry_after_seconds": round(WAIT_SECONDS, 1)}, 429
        started = time.perf_counter()
        height, width = frame.shape[:2]
        try:
            faces = self._app.get(frame)
            # Re-verify on every call: an engine reload or a driver problem must never be served
            # from as if it were fine.
            self._providers = verify_cuda(self._app)
            measured = {id(f): sharpness_of(frame, f.bbox) for f in faces}
        except CudaUnavailable as exc:
            self.failures += 1
            self._error = str(exc)
            self._log(f"[FACE-API] {exc}")
            return {"error": "GPU unavailable", "detail": str(exc)}, 503
        except Exception as exc:                          # noqa: BLE001
            self.failures += 1
            self._log(f"[FACE-API] inference failed: {type(exc).__name__}: {exc}")
            return {"error": "face inference failed", "detail": f"{type(exc).__name__}"}, 500
        finally:
            self._slot.release()
            del frame                                     # the image is never retained

        threshold = PROFILES[profile]
        kept = [f for f in faces if float(getattr(f, "det_score", 0.0) or 0.0) >= threshold]
        rows = sorted(
            ({"bbox": [int(v) for v in f.bbox],
              "det_score": round(float(getattr(f, "det_score", 0.0) or 0.0), 4),
              "area": int(max(0.0, f.bbox[2] - f.bbox[0]) * max(0.0, f.bbox[3] - f.bbox[1])),
              "sharpness": measured.get(id(f))}
             for f in kept), key=lambda r: -r["area"])
        self.requests += 1
        payload = {
            "faces_detected": len(kept),
            "faces": rows,                                # every face, so the DASHBOARD can apply
            "embedding": None,                            # its own selection / ambiguity rules
            "bbox": None, "det_score": None, "age": None, "gender": None,
            "embedding_dim": EMBEDDING_DIM, "model": MODEL_NAME, "model_id": MODEL_ID,
            "provider": self._providers.get("recognition"), "profile": profile,
            "det_thresh": threshold, "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "image": {"width": int(width), "height": int(height)},
        }
        if not kept:
            return payload, 200

        biggest = max(kept, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        vector = np.asarray(biggest.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        payload.update({
            "embedding": (vector / (norm + 1e-8)).tolist(),        # 512-d, L2-normalised
            "embedding_norm_raw": round(norm, 6),
            "bbox": [int(v) for v in biggest.bbox],
            "det_score": round(float(getattr(biggest, "det_score", 0.0) or 0.0), 4),
            "age": int(biggest.age) if getattr(biggest, "age", None) is not None else None,
            "gender": normalise_gender(getattr(biggest, "gender", None)),
            "sharpness": measured.get(id(biggest)),
        })
        return payload, 200


class _Handler(BaseHTTPRequestHandler):
    server_version = "ZayedInternalFaceAPI/1.0"
    service = None
    token = ""

    def log_message(self, *args):                          # never log request bodies or tokens
        pass

    def _send(self, payload, status):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self):
        if not self.token:
            return False
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        return header.startswith(prefix) and hmac.compare_digest(header[len(prefix):].strip(), self.token)

    def do_GET(self):
        if self.path.rstrip("/") != "/internal/face/health":
            return self._send({"error": "not found"}, 404)
        if not self._authorised():
            return self._send({"error": "unauthorised"}, 401)
        payload, status = self.service.health()
        self._send(payload, status)

    def do_POST(self):
        if self.path.rstrip("/") != "/internal/face/embed":
            return self._send({"error": "not found"}, 404)
        if not self._authorised():
            return self._send({"error": "unauthorised"}, 401)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._send({"error": "bad Content-Length"}, 400)
        if length <= 0:
            return self._send({"error": "empty request"}, 400)
        if length > MAX_UPLOAD_BYTES:
            return self._send({"error": f"image larger than {MAX_UPLOAD_BYTES} bytes"}, 413)
        body = self.rfile.read(length)
        try:
            fields = parse_multipart(body, self.headers.get("Content-Type", ""))
        except ValueError as exc:
            return self._send({"error": str(exc)}, 400)
        finally:
            del body
        image = fields.get("image")
        if not image:
            return self._send({"error": "missing 'image' field"}, 400)
        profile = (fields.get("profile") or b"search").decode("utf-8", "replace").strip()
        payload, status = self.service.embed(image, profile)
        del fields, image
        self._send(payload, status)


class InternalFaceAPI:
    """Owns the HTTP listener. Started and stopped by zayed_inference.py."""

    def __init__(self, log=print, service=None, host=BIND_HOST, port=BIND_PORT, token=None):
        self._log = log
        self.host, self.port = host, port
        self.service = service or FaceService(log=log)
        self.token = read_token() if token is None else token
        self._server = None
        self._thread = None

    def start(self):
        if not ENABLED:
            self._log("[FACE-API] disabled (INTERNAL_FACE_API=0)")
            return False
        if not self.token:
            self._log("[FACE-API] NOT started: no token (set INTERNAL_FACE_TOKEN_FILE to a 0600 file)")
            return False
        handler = type("_BoundHandler", (_Handler,), {"service": self.service, "token": self.token})
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="internal-face-api",
                                        daemon=True)
        self._thread.start()
        self._log(f"[FACE-API] listening on {self.host}:{self.port} (container network only, token auth)")
        # Load and GPU-verify off the inference thread; /health stays 503 until it succeeds.
        threading.Thread(target=self.service.load, name="internal-face-api-load", daemon=True).start()
        return True

    def stop(self, timeout=5.0):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout)

    def stats(self):
        health, status = self.service.health()
        return {"listening": self._server is not None, "port": self.port, "healthy": status == 200,
                "provider": health["provider"], "requests": health["requests"],
                "failures": health["failures"], "busy_rejections": health["busy_rejections"],
                "error": health["error"]}
