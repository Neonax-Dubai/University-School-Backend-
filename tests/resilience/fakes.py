"""Fake local servers for the resilience tests. Loopback only; nothing leaves the container."""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeDashboard:
    """POST /api/events/ with the real ingest contract: 201 created, 200 {"duplicate": true} for an
    event_id it already holds. `script` maps a request number (1-based) to a forced status, and
    `reject` names event_ids answered 400."""

    def __init__(self, port):
        self.port = port
        self.stored = {}
        self.requests = 0
        self.script = {}
        self.reject = set()
        self._lock = threading.Lock()
        self._server = None

    def start(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with owner._lock:
                    owner.requests += 1
                    forced = owner.script.get(owner.requests)
                    event_id = body.get("event_id")
                    if forced:
                        code, payload = forced, {"detail": "forced"}
                    elif event_id in owner.reject:
                        code, payload = 400, {"event_type": ["invalid"]}
                    elif event_id in owner.stored:
                        code, payload = 200, {"duplicate": True, "event_id": event_id}
                    else:
                        owner.stored[event_id] = body
                        code, payload = 201, {"event_id": event_id}
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


class FakeFiler:
    """SeaweedFS filer stand-in: 201 for every upload, remembering the keys."""

    def __init__(self, port):
        self.port = port
        self.keys = []
        self._server = None

    def start(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                owner.keys.append(self.path.split("?")[0].lstrip("/"))
                self.send_response(201)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


class HangingServer:
    """Accepts connections and never answers - a frozen filer, dashboard or Qdrant."""

    def __init__(self, port):
        self.port = port
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._held = []

    def start(self):
        self._sock.bind(("127.0.0.1", self.port))
        self._sock.listen(16)

        def accept():
            while True:
                try:
                    conn, _ = self._sock.accept()
                    self._held.append(conn)
                except OSError:
                    return

        threading.Thread(target=accept, daemon=True).start()
        return self

    def stop(self):
        self._sock.close()
        for conn in self._held:
            conn.close()
