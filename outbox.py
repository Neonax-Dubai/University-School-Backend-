"""
Durable event outbox - the Zayed replacement for events.EventSender's in-memory queue.

THE DUBAI WEAKNESS. EventSender held events in a bounded in-memory queue and POSTed each one
once: a failed POST was counted and DROPPED, a full queue dropped the oldest event, and a
process exit lost whatever was queued. A dashboard or database restart therefore silently
discarded detections.

HERE. submit() writes the event to a local SQLite outbox (WAL, fsync-light) before returning;
a background thread delivers it and deletes the row only after the dashboard acknowledged it
(201, or 200 "duplicate" - the ingest is idempotent on event_id, so a retry can never create a
second row). Failures are retried with exponential backoff (capped at MAX_BACKOFF_SECONDS) for
as long as the outbox has room, across process restarts. A payload the dashboard REJECTS as
invalid (4xx other than auth/rate/timeouts) is kept as a dead letter for inspection rather than
retried forever. The interface is EventSender's (submit/start/stop/pending/sent/failed/dropped),
so events.EventPipeline and evidence.EvidencePipeline use it unchanged.

    EVENT_OUTBOX_PATH             default <repo>/runtime/outbox/events.sqlite3
    EVENT_OUTBOX_MAX_PENDING      pending rows kept before the OLDEST is dropped (200000)
    EVENT_POST_TIMEOUT            HTTP timeout in seconds (5.0)
"""
import json
import os
import sqlite3
import threading
import time
import uuid

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUTBOX_PATH = os.getenv("EVENT_OUTBOX_PATH", os.path.join(HERE, "runtime", "outbox", "events.sqlite3"))
MAX_PENDING = int(os.getenv("EVENT_OUTBOX_MAX_PENDING", "200000"))
POST_TIMEOUT = float(os.getenv("EVENT_POST_TIMEOUT", "5.0"))
BATCH = 50
MAX_BACKOFF_SECONDS = 60.0
DEAD_LETTER_KEEP_DAYS = 14
_RETRY_STATUSES = {401, 403, 408, 409, 425, 429}


def new_event_id():
    return f"ZEV-{uuid.uuid4().hex[:16].upper()}"


class OutboxEventSender:
    """Durable, retrying, idempotent event delivery with EventSender's interface."""

    def __init__(self, dashboard_url, token, timeout=POST_TIMEOUT, path=OUTBOX_PATH, enrich=None,
                 session=None, log=None):
        self.url = f"{dashboard_url.rstrip('/')}/api/events/"
        self.token = token
        self.timeout = timeout
        self.path = path
        self._enrich = enrich
        self._session = session
        self._log = log or (lambda msg: print(msg, flush=True))
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._last_log = 0.0
        self.sent = self.failed = self.dropped = 0
        self.duplicates = self.dead = self.retries = self.enqueued = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=30, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("""CREATE TABLE IF NOT EXISTS outbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL,
            notable INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt REAL NOT NULL, created REAL NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
            last_error TEXT)""")
        self._db.execute("CREATE INDEX IF NOT EXISTS outbox_due ON outbox(state, next_attempt, id)")
        restored = self.pending()
        if restored:
            self._log(f"[OUTBOX] {restored} undelivered event(s) restored from {path}")

    # ---------------------------------------------------------------- producer side
    def submit(self, event, notable=False):
        """Persist one event. Never raises; never blocks on the network."""
        try:
            if self._enrich is not None:
                event = self._enrich(event)
            if not event.get("event_id"):
                event["event_id"] = new_event_id()
            now = time.time()
            with self._lock:
                self._db.execute("INSERT OR IGNORE INTO outbox(event_id, payload, notable, next_attempt, created) "
                                 "VALUES (?, ?, ?, ?, ?)",
                                 (event["event_id"], json.dumps(event, default=str), int(bool(notable)), now, now))
                over = self._db.execute("SELECT COUNT(*) FROM outbox WHERE state='pending'").fetchone()[0] - MAX_PENDING
                if over > 0:
                    self._db.execute("DELETE FROM outbox WHERE id IN (SELECT id FROM outbox WHERE state='pending' "
                                     "ORDER BY id LIMIT ?)", (over,))
                    self.dropped += over
                self.enqueued += 1
            self._wake.set()
        except Exception as exc:                                  # noqa: BLE001
            self.dropped += 1
            self._throttled(f"[OUTBOX] could not persist event: {type(exc).__name__}: {exc}")

    def pending(self):
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM outbox WHERE state='pending'").fetchone()[0]

    def dead_letters(self):
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM outbox WHERE state='dead'").fetchone()[0]

    # ---------------------------------------------------------------- delivery side
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="event-outbox", daemon=True)
        self._thread.start()

    def stop(self, timeout=5.0):
        self._stop.set()
        self._wake.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout)

    def _http(self):
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"Content-Type": "application/json",
                                          "Authorization": f"Token {self.token}"})
        return self._session

    def _due(self):
        with self._lock:
            return self._db.execute("SELECT id, event_id, payload, notable, attempts FROM outbox "
                                    "WHERE state='pending' AND next_attempt <= ? ORDER BY id LIMIT ?",
                                    (time.time(), BATCH)).fetchall()

    def _run(self):
        last_prune = 0.0
        while not self._stop.is_set():
            rows = self._due()
            if not rows:
                self._wake.wait(1.0)
                self._wake.clear()
                if time.time() - last_prune > 3600:
                    last_prune = time.time()
                    self._prune()
                continue
            pause = 0.0
            for row_id, event_id, payload, notable, attempts in rows:
                if self._stop.is_set():
                    break
                outcome, detail = self._post(payload)
                with self._lock:
                    if outcome in ("ok", "duplicate"):
                        self._db.execute("DELETE FROM outbox WHERE id=?", (row_id,))
                        self.sent += 1
                        self.duplicates += outcome == "duplicate"
                        if notable:
                            self._log(f"[EVENT] {event_id} delivered")
                        continue
                    self.failed += 1
                    if outcome == "rejected":
                        self._db.execute("UPDATE outbox SET state='dead', last_error=? WHERE id=?",
                                         (detail[:500], row_id))
                        self.dead += 1
                        self._throttled(f"[OUTBOX] {event_id} REJECTED by the dashboard (kept as dead letter): {detail}")
                        continue
                    backoff = min(MAX_BACKOFF_SECONDS, 2.0 ** min(attempts, 6))
                    self._db.execute("UPDATE outbox SET attempts=attempts+1, next_attempt=?, last_error=? WHERE id=?",
                                     (time.time() + backoff, detail[:500], row_id))
                    self.retries += 1
                self._throttled(f"[OUTBOX] delivery failed ({detail}); {self.pending()} pending, retrying in {backoff:.0f}s")
                pause = backoff                           # the dashboard is unhappy: back off as a WHOLE,
                break                                     # not row by row through the backlog
            if pause:
                self._stop.wait(pause)

    def _post(self, payload):
        try:
            response = self._http().post(self.url, data=payload, timeout=self.timeout)
        except Exception as exc:                                  # noqa: BLE001
            return "retry", f"{type(exc).__name__}: {exc}"
        code = response.status_code
        if code == 201:
            return "ok", ""
        if code == 200:
            try:
                return ("duplicate" if response.json().get("duplicate") else "ok"), ""
            except ValueError:
                return "ok", ""
        text = response.text[:300].replace("\n", " ")
        if code in _RETRY_STATUSES or code >= 500:
            return "retry", f"HTTP {code}: {text}"
        return "rejected", f"HTTP {code}: {text}"

    def _prune(self):
        with self._lock:
            self._db.execute("DELETE FROM outbox WHERE state='dead' AND created < ?",
                             (time.time() - DEAD_LETTER_KEEP_DAYS * 86400,))

    def _throttled(self, message):
        now = time.monotonic()
        if now - self._last_log >= 30.0:
            self._last_log = now
            self._log(message)

    def stats(self):
        return {"sent": self.sent, "failed": self.failed, "dropped": self.dropped, "duplicates": self.duplicates,
                "dead": self.dead, "retries": self.retries, "enqueued": self.enqueued, "pending": self.pending(),
                "dead_letters": self.dead_letters(), "path": self.path}
