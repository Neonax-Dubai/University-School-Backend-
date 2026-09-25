"""
Internal face API: auth, GPU enforcement, profiles, concurrency and payload shape.
CPU-only and offline - the model is stubbed; the GPU path is covered by live_gpu_check.py.

    python -m unittest test_internal_api
"""
import io
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import internal_api as api                                      # noqa: E402


class FakeSession:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return list(self._providers)


class FakeModel:
    def __init__(self, providers):
        self.session = FakeSession(providers)


class FakeFace:
    def __init__(self, bbox, det_score, seed=1):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.det_score = det_score
        self.age = 24
        self.gender = 1
        rng = np.random.default_rng(seed)
        self.embedding = rng.normal(size=512).astype(np.float32) * 7.0


class FakeApp:
    """Shaped like insightface FaceAnalysis for the two attributes the API touches."""

    def __init__(self, faces, providers=("CUDAExecutionProvider", "CPUExecutionProvider"), delay=0.0):
        self.models = {"detection": FakeModel(providers), "recognition": FakeModel(providers),
                       "genderage": FakeModel(providers)}
        self._faces = faces
        self._delay = delay
        self.calls = 0

    def get(self, image):
        import time
        self.calls += 1
        if self._delay:
            time.sleep(self._delay)
        return list(self._faces)


def png_bytes(width=64, height=48):
    import cv2
    return cv2.imencode(".png", np.full((height, width, 3), 120, np.uint8))[1].tobytes()


def service_with(faces, providers=("CUDAExecutionProvider", "CPUExecutionProvider"), delay=0.0):
    app = FakeApp(faces, providers, delay)
    return api.FaceService(log=lambda *_: None, app_factory=lambda: app), app


class CudaEnforcementTests(unittest.TestCase):
    def test_cuda_provider_is_accepted(self):
        self.assertEqual(api.verify_cuda(FakeApp([])) ["recognition"], "CUDAExecutionProvider")

    def test_cpu_only_session_is_refused(self):
        with self.assertRaises(api.CudaUnavailable) as caught:
            api.verify_cuda(FakeApp([], providers=("CPUExecutionProvider",)))
        self.assertIn("refusing to serve on the CPU", str(caught.exception))

    def test_cpu_first_session_is_refused(self):
        """The exact Dubai/dashboard mistake: CUDA present but not the effective provider."""
        with self.assertRaises(api.CudaUnavailable):
            api.verify_cuda(FakeApp([], providers=("CPUExecutionProvider", "CUDAExecutionProvider")))

    def test_no_sessions_is_refused(self):
        empty = FakeApp([])
        empty.models = {}
        with self.assertRaises(api.CudaUnavailable):
            api.verify_cuda(empty)

    def test_the_real_model_is_built_with_cuda_alone(self):
        source = open(os.path.join(HERE, "..", "..", "internal_api.py")).read()
        self.assertIn("providers=[CUDA_PROVIDER]", source)
        self.assertNotIn('"CPUExecutionProvider"]', source.split("def _build_app")[1].split("def load")[0])

    def test_service_refuses_every_request_when_cuda_is_absent(self):
        service, _ = service_with([FakeFace([0, 0, 90, 110], 0.9)], providers=("CPUExecutionProvider",))
        payload, status = service.embed(png_bytes(), "search")
        self.assertEqual(status, 503)
        self.assertIsNone(payload.get("embedding"))
        health, hstatus = service.health()
        self.assertEqual(hstatus, 503)
        self.assertEqual(health["status"], "unavailable")
        self.assertIn("CUDAExecutionProvider", health["error"])

    def test_health_reports_the_effective_provider(self):
        service, _ = service_with([])
        service.load()
        health, status = service.health()
        self.assertEqual((status, health["provider"], health["gpu_required"], health["cpu_fallback"]),
                         (200, "CUDAExecutionProvider", True, False))
        self.assertEqual(health["embedding_dim"], 512)


class EmbedTests(unittest.TestCase):
    def test_embedding_is_512d_unit_norm_with_face_metadata(self):
        service, _ = service_with([FakeFace([10, 20, 110, 140], 0.87)])
        payload, status = service.embed(png_bytes(200, 200), "enroll")
        self.assertEqual(status, 200)
        vector = np.asarray(payload["embedding"], dtype=np.float32)
        self.assertEqual(vector.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)
        self.assertEqual(payload["bbox"], [10, 20, 110, 140])
        self.assertEqual(payload["det_score"], 0.87)
        self.assertEqual(payload["faces_detected"], 1)
        self.assertEqual(payload["provider"], "CUDAExecutionProvider")
        self.assertEqual((payload["model"], payload["embedding_dim"]), ("buffalo_l", 512))
        self.assertEqual(payload["image"], {"width": 200, "height": 200})
        self.assertEqual(payload["gender"], "male")
        self.assertIsNotNone(payload["faces"][0]["sharpness"])

    def test_search_profile_filters_to_06_while_enroll_keeps_05(self):
        faces = [FakeFace([0, 0, 100, 120], 0.55, seed=2), FakeFace([200, 0, 260, 80], 0.95, seed=3)]
        service, _ = service_with(faces)
        enrolled, _ = service.embed(png_bytes(400, 300), "enroll")
        searched, _ = service.embed(png_bytes(400, 300), "search")
        self.assertEqual(enrolled["faces_detected"], 2)
        self.assertEqual(searched["faces_detected"], 1, "0.55 is below the search threshold")
        self.assertEqual(searched["det_score"], 0.95)

    def test_the_largest_face_is_the_embedded_one_and_all_faces_are_reported(self):
        small, large = FakeFace([0, 0, 40, 40], 0.9, seed=4), FakeFace([50, 50, 200, 250], 0.8, seed=5)
        service, _ = service_with([small, large])
        payload, _ = service.embed(png_bytes(400, 400), "enroll")
        self.assertEqual(payload["bbox"], [50, 50, 200, 250])
        self.assertEqual([row["bbox"] for row in payload["faces"]], [[50, 50, 200, 250], [0, 0, 40, 40]])

    def test_no_face_is_an_empty_but_successful_answer(self):
        service, _ = service_with([])
        payload, status = service.embed(png_bytes(), "search")
        self.assertEqual((status, payload["faces_detected"], payload["embedding"]), (200, 0, None))

    def test_unknown_profile_and_unreadable_image_are_400(self):
        service, _ = service_with([FakeFace([0, 0, 90, 110], 0.9)])
        self.assertEqual(service.embed(png_bytes(), "whatever")[1], 400)
        self.assertEqual(service.embed(b"not-an-image", "search")[1], 400)

    def test_a_busy_service_answers_429_rather_than_queueing(self):
        service, _ = service_with([FakeFace([0, 0, 90, 110], 0.9)], delay=1.5)
        api.WAIT_SECONDS = 0.2                                   # bounded wait for the test
        try:
            first = threading.Thread(target=lambda: service.embed(png_bytes(), "search"))
            first.start()
            import time
            time.sleep(0.2)
            payload, status = service.embed(png_bytes(), "search")
            first.join()
        finally:
            api.WAIT_SECONDS = 5.0
        self.assertEqual(status, 429, "the CCTV pipeline must never wait behind dashboard uploads")
        self.assertEqual(service.busy_rejections, 1)


class MultipartTests(unittest.TestCase):
    def test_fields_are_parsed(self):
        body = (b"--B\r\nContent-Disposition: form-data; name=\"profile\"\r\n\r\nenroll\r\n"
                b"--B\r\nContent-Disposition: form-data; name=\"image\"; filename=\"x.png\"\r\n"
                b"Content-Type: image/png\r\n\r\n\x89PNG-bytes\r\n--B--\r\n")
        fields = api.parse_multipart(body, "multipart/form-data; boundary=B")
        self.assertEqual(fields["profile"], b"enroll")
        self.assertEqual(fields["image"], b"\x89PNG-bytes")

    def test_malformed_bodies_raise(self):
        for body, ctype in ((b"x", "application/json"), (b"x", "multipart/form-data; boundary="),
                            (b"--B--\r\n", "multipart/form-data; boundary=B")):
            with self.assertRaises(ValueError):
                api.parse_multipart(body, ctype)


class HttpTests(unittest.TestCase):
    TOKEN = "test-token-not-a-secret"

    def setUp(self):
        self.service, self.app = service_with([FakeFace([5, 5, 105, 125], 0.91)])
        self.service.load()
        self.api = api.InternalFaceAPI(log=lambda *_: None, service=self.service, host="127.0.0.1",
                                       port=0, token=self.TOKEN)
        self.assertTrue(self.api.start())
        self.port = self.api._server.server_address[1]

    def tearDown(self):
        self.api.stop()

    def call(self, path, token=TOKEN, data=None, ctype=None):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data)
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        if ctype:
            request.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    def multipart(self, image=None, profile="search"):
        boundary = "----zayedtest"
        image = png_bytes() if image is None else image
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"profile\"\r\n\r\n{profile}\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"u.png\"\r\n"
                f"Content-Type: image/png\r\n\r\n").encode() + image + f"\r\n--{boundary}--\r\n".encode()
        return body, f"multipart/form-data; boundary={boundary}"

    def test_health_requires_the_token(self):
        self.assertEqual(self.call("/internal/face/health", token=None)[0], 401)
        self.assertEqual(self.call("/internal/face/health", token="wrong")[0], 401)
        status, payload = self.call("/internal/face/health")
        self.assertEqual((status, payload["provider"], payload["status"]), (200, "CUDAExecutionProvider", "ok"))

    def test_embed_round_trip(self):
        body, ctype = self.multipart()
        status, payload = self.call("/internal/face/embed", data=body, ctype=ctype)
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["embedding"]), 512)
        self.assertEqual(payload["bbox"], [5, 5, 105, 125])
        self.assertEqual(payload["provider"], "CUDAExecutionProvider")

    def test_embed_requires_the_token_and_a_known_path(self):
        body, ctype = self.multipart()
        self.assertEqual(self.call("/internal/face/embed", token="nope", data=body, ctype=ctype)[0], 401)
        self.assertEqual(self.call("/internal/face/elsewhere", data=body, ctype=ctype)[0], 404)
        self.assertEqual(self.call("/internal/anything")[0], 404)

    def test_oversized_upload_is_refused_without_decoding(self):
        api.MAX_UPLOAD_BYTES = 1024
        try:
            body, ctype = self.multipart(image=b"\x00" * 4096)
            status, _ = self.call("/internal/face/embed", data=body, ctype=ctype)
        finally:
            api.MAX_UPLOAD_BYTES = 12 * 1024 * 1024
        self.assertEqual(status, 413)
        self.assertEqual(self.app.calls, 0, "an oversized body never reaches the model")

    def test_the_api_stores_nothing(self):
        body, ctype = self.multipart()
        self.call("/internal/face/embed", data=body, ctype=ctype)
        source = open(os.path.join(HERE, "..", "..", "internal_api.py")).read()
        self.assertEqual(source.count("open("), 1, "the token file is the only thing ever opened")
        for forbidden in ("qdrant_client", "imwrite", "makedirs", "NamedTemporaryFile"):
            self.assertNotIn(forbidden, source, f"{forbidden} must not appear in the face API")


if __name__ == "__main__":
    unittest.main(verbosity=2)
