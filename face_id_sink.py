"""
Face-ID result sink - carries finished face observations to the dashboard.

FaceIDManager decides identities and knows nothing about HTTP. This is the
piece that knows about HTTP and nothing about faces. It is injected into the
manager as `result_sink`, so the identity logic stays testable with no
network and the transport stays replaceable.

    sink = DashboardFaceSink(dashboard_url, token)
    manager = FaceIDManager(result_sink=sink.record)
    ...
    sink.close()

WHY THIS IS ITS OWN QUEUE
-------------------------
There are already two queues in front of this one - the adapter's bounded
observation queue, and the async Qdrant writer - and adding a third needs a
reason. The reason is that the Face-ID worker is a SINGLE thread that also
runs face detection, embedding and a Qdrant search. A dashboard POST is
tens of milliseconds of network wait; doing it inline would idle the one
worker on I/O and shrink face throughput, exactly the coupling the adapter
exists to prevent one layer up.

Mirrors events.py's own EventSender shape deliberately - bounded queue,
background thread, drop-on-full with a counter, never block the caller -
because that pattern is already proven in this pipeline and a second,
different convention for the same job would be worse than a familiar one.

FAILURE POLICY
--------------
A dropped report costs one row of application metadata. The identity itself
is already durable in Qdrant, and the next observation of the same person
re-reports the consensus. So this drops rather than retries hard, and never
blocks: losing enrichment is always cheaper than stalling recognition.

Env:
  DASHBOARD_URL                 default http://127.0.0.1:8000
  DASHBOARD_TOKEN               API token
  FACE_ID_SINK_QUEUE_SIZE       bounded queue depth (default 256)
  FACE_ID_SINK_BATCH            observations per POST (default 16)
  FACE_ID_SINK_INTERVAL         max seconds a batch waits (default 2.0)
  FACE_ID_SINK_TIMEOUT          HTTP timeout seconds (default 5.0)
"""
import os
import queue
import threading
import time

import requests


DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://127.0.0.1:8000").rstrip("/")
DASHBOARD_TOKEN = os.getenv("DASHBOARD_TOKEN", "")

FACE_ID_SINK_QUEUE_SIZE = int(os.getenv("FACE_ID_SINK_QUEUE_SIZE", "256"))
FACE_ID_SINK_BATCH = int(os.getenv("FACE_ID_SINK_BATCH", "16"))
FACE_ID_SINK_INTERVAL = float(os.getenv("FACE_ID_SINK_INTERVAL", "2.0"))
FACE_ID_SINK_TIMEOUT = float(os.getenv("FACE_ID_SINK_TIMEOUT", "5.0"))
FACE_ID_SINK_PATH = os.getenv("FACE_ID_SINK_PATH", "/api/ai/face-observations/")


class DashboardFaceSink:
    """Bounded, batched, non-blocking delivery of face observations."""

    def __init__(self, dashboard_url=None, token=None, queue_size=None,
                 batch=None, interval=None, timeout=None, session=None,
                 path=None, name="face-id-sink", batch_key="observations",
                 alt_batch_key=None, alt_marker=None):
        # `path`, `name` and `batch_key` exist so a SECOND, independent sink
        # can deliver a different payload kind to a different endpoint on the
        # same terms - bounded queue, batched, drop-on-full, never blocking
        # the caller. Known-person recognition uses that; see
        # face_id_manager._recognise_known_person(). All three default to the
        # face-observation behaviour, so every existing caller is unchanged.
        self.url = ((dashboard_url or DASHBOARD_URL).rstrip("/")
                    + (FACE_ID_SINK_PATH if path is None else path))
        self.token = DASHBOARD_TOKEN if token is None else token
        self.batch = FACE_ID_SINK_BATCH if batch is None else batch
        self.interval = FACE_ID_SINK_INTERVAL if interval is None else interval
        self.timeout = FACE_ID_SINK_TIMEOUT if timeout is None else timeout
        self.batch_key = batch_key
        self.alt_batch_key = alt_batch_key
        self.alt_marker = alt_marker

        self._queue = queue.Queue(maxsize=FACE_ID_SINK_QUEUE_SIZE if queue_size is None
                                  else queue_size)
        self._stop = threading.Event()
        self._session = session if session is not None else requests.Session()

        self.queued = 0
        self.sent = 0
        self.dropped_queue_full = 0
        self.failed_posts = 0

        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=name)
        self._thread.start()

    # ------------------------------------------------------------- producer
    def record(self, payload):
        """Called by FaceIDManager on the Face-ID worker thread. Never blocks."""
        try:
            self._queue.put_nowait(dict(payload))
            self.queued += 1
        except queue.Full:
            self.dropped_queue_full += 1

    # --------------------------------------------------------------- sender
    def _run(self):
        pending = []
        last_flush = time.monotonic()

        while True:
            timeout = max(0.05, self.interval - (time.monotonic() - last_flush))

            try:
                pending.append(self._queue.get(timeout=timeout))
            except queue.Empty:
                pass

            due = (len(pending) >= self.batch
                   or (pending and (time.monotonic() - last_flush) >= self.interval))

            if due:
                self._post(pending)
                pending = []
                last_flush = time.monotonic()

            if self._stop.is_set() and self._queue.empty():
                if pending:
                    self._post(pending)
                return

    def _post(self, observations):
        headers = {"Authorization": f"Token {self.token}"} if self.token else {}
        try:
            response = self._session.post(
                self.url, json=self._body(observations),
                headers=headers, timeout=self.timeout,
            )
            if 200 <= response.status_code < 300:
                self.sent += len(observations)
            else:
                self.failed_posts += 1
                print(f"[FACE-ID] dashboard rejected {len(observations)} observations "
                      f"({response.status_code}) - enrichment lost, identities unaffected")
        except Exception as exc:  # noqa: BLE001 - a POST failure must never kill this thread
            self.failed_posts += 1
            print(f"[FACE-ID] dashboard unreachable ({type(exc).__name__}) - "
                  f"{len(observations)} observations dropped, identities unaffected")

    def _body(self, rows):
        """
        The POST body: one key, or two when this sink carries two row kinds.

        Splitting here rather than running a second sink keeps one queue,
        one thread and one connection for one producer. Both parameters
        default to None, so a sink that was never told about a second kind
        behaves exactly as before - which is every existing caller.
        """
        if not self.alt_batch_key or not self.alt_marker:
            return {self.batch_key: rows}

        primary, alternate = [], []
        for row in rows:
            (alternate if row.get(self.alt_marker) else primary).append(row)

        body = {self.batch_key: primary}
        if alternate:
            body[self.alt_batch_key] = alternate
        return body

    # ------------------------------------------------------------ lifecycle
    def stats(self):
        return {
            "sink_queued": self.queued,
            "sink_sent": self.sent,
            "sink_dropped_queue_full": self.dropped_queue_full,
            "sink_failed_posts": self.failed_posts,
            "sink_queue_depth": self._queue.qsize(),
        }

    def close(self, timeout=5.0):
        """Bounded drain. Shutdown must not hang on a slow dashboard."""
        self._stop.set()
        self._thread.join(timeout=timeout)
        remaining = self._queue.qsize()
        print(f"[FACE-ID] sink stopped  sent={self.sent} "
              f"dropped_queue_full={self.dropped_queue_full} "
              f"failed_posts={self.failed_posts} unsent={remaining}")
        return remaining
