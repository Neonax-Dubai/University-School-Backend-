"""
Face evidence - one representative face-crop image per Face-ID identity,
uploaded to SeaweedFS.

Reuses evidence.py's building blocks (crop_bbox, build_storage_key,
make_store) directly - they are pure/decoupled functions with no coupling
to events.py. Does NOT reuse evidence.EvidencePipeline itself: that
class's submit() is hard-wired to also forward an events.py-shaped event
to an EventSender, which Face-ID has no equivalent of (its own reporting
goes through face_id_sink.py, a completely different endpoint/payload).
This module is a small, independent async uploader following the same
proven shape (bounded queue, dedicated background thread(s), fire-and-
forget on failure) already used three times in this codebase
(evidence.EvidencePipeline, reid_poc.reid_manager.AsyncUpsertStore,
face_id_sink.DashboardFaceSink).

Called from face_id_manager.py's process(), NEW branch only - one capture
per identity, ever, not per observation. The crop is already known-good by
that point (quality_check() has already passed it), so unlike evidence.py's
person-track crops there is no "keep retrying until one lands" policy here
- one attempt, and if it fails the identity still mints fine, just without
an image.

    pipeline = FaceEvidencePipeline()
    pipeline.start()
    pipeline.submit(jpeg_bytes, width, height, storage_key)   # non-blocking
    ...
    pipeline.stop()
"""

import os
import threading
import time
from queue import Empty, Full, Queue

from evidence import make_store

WORKERS = int(os.getenv("FACE_EVIDENCE_WORKERS", "1"))
QUEUE_SIZE = int(os.getenv("FACE_EVIDENCE_QUEUE_SIZE", "64"))
FAILURE_LOG_INTERVAL = 15.0


class FaceEvidencePipeline:
    """
    Uploads a face crop to SeaweedFS on a background thread.

    The caller already knows the storage_key before calling submit() -
    build_storage_key() is pure string formatting, no network - so the
    dashboard-facing payload can carry the key immediately, well before the
    actual upload (which happens here, asynchronously) completes.
    """

    def __init__(self, store=None, workers=WORKERS, queue_size=QUEUE_SIZE):
        self.store = store if store is not None else make_store()

        self._queue = Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.uploaded = 0
        self.upload_failed = 0
        self.queue_dropped = 0

        # Liveness marks, not metrics. Counters alone cannot separate "the
        # worker is fine, nothing has been minted lately" from "the worker
        # is stuck on an upload that never returns" - in both cases uploaded
        # and upload_failed simply stop moving. An attempt that started but
        # never finished shows up as last_attempt_age advancing while
        # last_success_age stands still.
        self._last_attempt = None
        self._last_success = None

        self._last_failure_log = 0.0

        self._threads = [
            threading.Thread(target=self._worker, daemon=True,
                             name=f"face-evidence-{index}")
            for index in range(max(1, workers))
        ]

    def start(self):
        for thread in self._threads:
            thread.start()

    # ------------------------------------------------------------- producer
    def submit(self, jpeg, width, height, storage_key):
        """
        Queue a face crop for upload. Non-blocking. Returns False only if
        the queue is saturated - the identity still mints fine either way,
        it just ends up with no image.
        """
        try:
            self._queue.put_nowait((jpeg, width, height, storage_key))
            return True
        except Full:
            with self._lock:
                self.queue_dropped += 1
            return False

    # ------------------------------------------------------------- consumer
    def _worker(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                continue
            self._process_guarded(item)

        # Best-effort drain so a mint moments before shutdown is not lost.
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                break
            self._process_guarded(item)

    def _process_guarded(self, item):
        """
        Outer net around one queue item.

        _process() already contains the expected failure (the store raising
        on a bad upload). This exists for the unexpected one - a malformed
        item that fails to unpack, a store returning something bizarre, a
        logging call that itself blows up. Any of those escaping into
        _worker()'s loop would kill the thread outright and silently end
        evidence capture for the life of the process, so the loop is never
        allowed to see an exception. One bad item is skipped; the worker
        keeps serving the queue.
        """
        try:
            self._process(item)
        except Exception as exc:                      # noqa: BLE001 - a bad item must never kill the worker
            with self._lock:
                self.upload_failed += 1
            self._log_failure(f"unprocessable item ({type(exc).__name__}: {exc})")

    def _process(self, item):
        jpeg, _width, _height, storage_key = item

        # Marked BEFORE the call, so an upload that never returns still
        # leaves evidence that the worker reached it.
        with self._lock:
            self._last_attempt = time.monotonic()

        try:
            self.store.put(storage_key, jpeg)
            with self._lock:
                self.uploaded += 1
                self._last_success = time.monotonic()
        except Exception as exc:                      # noqa: BLE001 - an upload failure must never propagate
            with self._lock:
                self.upload_failed += 1
            self._log_failure(f"{storage_key}: {exc}")

    def _log_failure(self, message):
        # Throttled so a SeaweedFS outage cannot flood the log at camera
        # frame rate. The counters stay exact regardless - stats() is the
        # source of truth for how many failed, this is only for context.
        now = time.monotonic()
        with self._lock:
            if now - self._last_failure_log < FAILURE_LOG_INTERVAL:
                return
            self._last_failure_log = now
        print(f"[FACE-EVIDENCE-ERROR] upload failed (identity keeps its id, "
              f"just no image): {message}")

    # -------------------------------------------------------------- reporting
    def pending(self):
        return self._queue.qsize()

    def stats(self):
        """
        Counters plus enough liveness for an operator to answer one
        question: is the evidence worker alive and processing, or is the
        queue accumulating behind a worker that is stuck?

        Ages are in seconds and None when the thing has never happened.
        The healthy reading is workers_alive == workers_configured with
        last_success_age small; a wedged worker reads as pending climbing
        while last_success_age grows without bound.
        """
        now = time.monotonic()
        with self._lock:
            last_attempt, last_success = self._last_attempt, self._last_success

        return {
            "uploaded": self.uploaded,
            "upload_failed": self.upload_failed,
            "queue_dropped": self.queue_dropped,
            "pending": self.pending(),
            "workers_configured": len(self._threads),
            "workers_alive": sum(1 for t in self._threads if t.is_alive()),
            "last_attempt_age": None if last_attempt is None else round(now - last_attempt, 3),
            "last_success_age": None if last_success is None else round(now - last_success, 3),
        }

    def stop(self, timeout=5.0):
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=timeout)
