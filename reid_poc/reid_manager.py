"""
Global person identity resolution, on top of per-camera JeztSort tracks.

    (camera_id, local_track_id) -- observe(embedding) -->  global_person_id

Two OSNet calls per track's lifetime pattern, never per-frame:

  BOOTSTRAP  a brand-new local track collects config.REID_BOOTSTRAP_OBSERVATIONS
             embeddings, spaced config.REID_BOOTSTRAP_INTERVAL_FRAMES confirmed
             frames apart, averaged into one representative embedding, then
             matched ONCE against every known global identity.

  REFRESH    once resolved, a track's representative embedding (and the
             global identity's gallery) is refreshed at most once every
             config.REID_UPDATE_INTERVAL_SECONDS - time-based, so it behaves
             the same regardless of the caller's inference rate.

needs_observation() is the cheap per-tick gate (no OSNet call); observe() is
called only for tracks a caller actually ran through OSNet this tick, so the
caller can batch every eligible crop into ONE extract_batch() call per tick
before feeding the results back in one at a time.

This module knows nothing about JeztSort, cameras, or Qdrant specifically -
it depends only on a `store` object satisfying the Store interface below, so
swapping InMemoryReIDStore for qdrant_reid.QdrantReIDStore is a config change,
never a code change here.
"""
import itertools
import queue
import re
import threading
import time

import numpy as np

import config


# ============================================================
# IDENTITY NUMBERING
#
# Global ids are "P-" + a zero-padded sequence number (P-0001, ..., P-9999,
# P-10000). A persistent store outlives the process, so a new process must
# continue AFTER the highest id already stored - restarting at P-0001 would
# hand an existing person's id to a new person, and the store would merge the
# two galleries. Gaps are never reused.
# ============================================================

_GLOBAL_ID_RE = re.compile(r"^P-(\d+)$")


def highest_identity_seq(global_ids):
    """Highest sequence number among ids of the form P-<digits>, or 0.

    Anything else - None, "", non-strings, other prefixes, P-12a, p-0003 - is
    ignored, never an error: such a point cannot collide with an id this
    manager will mint.
    """
    highest = 0
    for global_id in global_ids:
        match = _GLOBAL_ID_RE.match(global_id) if isinstance(global_id, str) else None
        if match:
            highest = max(highest, int(match.group(1)))
    return highest


# ============================================================
# STORE INTERFACE
#
# Duck-typed on purpose (no ABC), matching this codebase's style elsewhere.
# Both InMemoryReIDStore below and qdrant_reid.QdrantReIDStore implement it.
# ============================================================

class Store:
    """search / upsert / get_history - the only three operations
    GlobalReIDManager needs from a backing identity store."""

    def search(self, embedding, top_k=1, exclude_camera=None, min_timestamp=None,
               allowed_cameras=None, only_global_id=None):
        """
        Return [(global_id, similarity, payload), ...], best match first - the
        best-scoring POINT per distinct global_id, not an average across its
        gallery, so a person seen from very different angles can still match
        on whichever gallery entry looks like the current crop.

        Candidate filtering (the cross-camera policy, and the experimental
        same-camera policy - see config.py):
          exclude_camera   skip points whose payload camera_id equals this
                            (same-camera candidates must never be offered)
          min_timestamp    skip points whose payload timestamp is older than
                            this (an old sighting is not an immediate transition)
          allowed_cameras  if given, ONLY consider points whose camera_id is
                            in this set (topology restriction, OR the
                            same-camera-only experimental search: pass
                            {camera_id} to search just that one camera)
          only_global_id   if given, restrict the result to just this one
                            identity (used by the possible-ID-swap diagnostic
                            to score an embedding against a SPECIFIC known
                            identity's gallery, not the whole store)
        All filters are applied at the POINT level before per-identity
        aggregation, so an identity with points on multiple cameras can still
        match via its OTHER cameras' points even when its point(s) on the
        current camera are excluded.
        """
        raise NotImplementedError

    def upsert(self, global_id, embedding, payload):
        """Add one more representative-embedding point for global_id."""
        raise NotImplementedError

    def get_history(self, global_id):
        """Return this identity's stored payload dicts (oldest first), or []."""
        raise NotImplementedError


class TimingStore(Store):
    """
    Transparent wrapper around any Store that records how long search()/
    upsert() actually take - e.g. real Qdrant network+search latency, without
    GlobalReIDManager or the store implementation needing to know they are
    being measured. Read .search_ms / .upsert_ms (lists, one entry per call)
    from the caller; this class only records, it never aggregates or prints.
    """

    def __init__(self, inner):
        self.inner = inner
        self.search_ms = []
        self.upsert_ms = []

    def search(self, embedding, top_k=1, exclude_camera=None, min_timestamp=None,
               allowed_cameras=None, only_global_id=None):
        start = time.perf_counter()
        result = self.inner.search(embedding, top_k=top_k, exclude_camera=exclude_camera,
                                    min_timestamp=min_timestamp, allowed_cameras=allowed_cameras,
                                    only_global_id=only_global_id)
        self.search_ms.append((time.perf_counter() - start) * 1000.0)
        return result

    def upsert(self, global_id, embedding, payload):
        start = time.perf_counter()
        result = self.inner.upsert(global_id, embedding, payload)
        self.upsert_ms.append((time.perf_counter() - start) * 1000.0)
        return result

    def get_history(self, global_id):
        return self.inner.get_history(global_id)


class AsyncUpsertStore(Store):
    """
    PRODUCTION-HARDENING P1 FIX: decouples upsert() (persistence - does NOT
    need to block inference) from search() (the candidate lookup a Re-ID
    DECISION actually depends on, which stays fully synchronous here,
    unchanged). Root cause this targets: live testing measured Qdrant-backed
    Re-ID at roughly HALF the FPS of the memory-store/disabled baselines,
    because search() AND upsert() both ran synchronously on the main
    inference thread - and upsert() is the more expensive of the two on this
    store (QdrantReIDStore.upsert() always calls _trim_gallery(), which
    always does a scroll() round-trip even when nothing needs trimming -
    measured ~13ms average, 2+ sequential network round-trips per upsert).

    search()/get_history() pass straight through to the wrapped inner store,
    synchronously, on the CALLING thread, completely unchanged. Only
    upsert() is deferred:

        inference thread: upsert() -> enqueue an IMMUTABLE SNAPSHOT
                           (embedding.copy(), dict(payload)) onto a bounded
                           queue.Queue -> returns immediately, no network
                           wait
        writer thread:     drains the queue FIFO, one item at a time, calls
                           the wrapped inner store's REAL upsert() - the
                           SAME store instance search() already uses.

    Sharing ONE inner store/QdrantClient instance across both the calling
    thread (search) and this class's own writer thread (upsert) was verified
    empirically before choosing this design, not assumed: a throwaway
    concurrent-load probe against the real isolated test Qdrant instance
    (127.0.0.1:6343) ran two threads hammering search() and two hammering
    upsert() simultaneously for 8s (3447 search calls / 1107 upsert calls,
    qdrant-client 1.19.0) with zero errors - its REST transport is safe for
    this usage pattern. A dedicated second connection for the writer thread
    would have been the more conservative design; it was not necessary.

    ONE background writer thread, not a pool - deliberately, so writes to
    the SAME global_id are always applied in the SAME order they were
    enqueued (a queue.Queue is FIFO). A pool could reorder two close-together
    upserts for one identity, letting a STALE embedding overwrite a FRESHER
    one - a correctness property worth keeping, especially since upsert()
    was never the pipeline's throughput bottleneck (one per resolved/
    refreshed track, not per frame). Connection reuse falls out of the same
    choice: the one writer thread calls the SAME already-connected inner
    store repeatedly, never reconnecting per write.

    Bounded queue (config.REID_ASYNC_UPSERT_QUEUE_SIZE): upsert() uses
    put_nowait() and DROPS (counted via dropped_queue_full, never logged
    from this module - see below) rather than blocking the inference thread
    if the queue is full. A dropped upsert only costs one fewer gallery
    refresh point for a FUTURE search - it can never corrupt or block the
    CURRENT decision, since GlobalReIDManager.observe() always calls
    upsert() only AFTER a decision has already been made from search()
    results. Blocking inference to guarantee delivery would reintroduce
    exactly the coupling this fix removes.

    Bounded retry (config.REID_ASYNC_UPSERT_MAX_RETRIES, linear backoff
    config.REID_ASYNC_UPSERT_RETRY_BACKOFF_SECONDS) on the WRITER thread
    only - sleeping there costs nothing on the inference path. After
    retries are exhausted the item is dropped (counted via
    dropped_after_retries) and the writer moves on to the next queued item
    rather than stalling the queue forever on one bad write. A sustained
    Qdrant outage therefore drains into "upserts get dropped once the
    bounded queue fills, searches keep working against whatever was already
    persisted" - never a crash, never inference stalling, never a silent
    switch to some other store.

    No I/O of its own: this module has never printed/logged anything (grep-
    confirmed), and this class does not change that - it exposes counters
    (written, dropped_queue_full, dropped_after_retries, queue_depth()) for
    reid_adapter.py's existing [REID-STATS]/close() logging to read on its
    own cadence, rather than introducing a second, parallel logging
    convention here.

    Never used for the DECISION path: get_history() reads are eventually
    consistent (a read immediately after an in-flight async upsert may not
    yet reflect it) - acceptable because no DECISION method in
    GlobalReIDManager calls get_history() at all (grep-verified) - it exists
    for external reporting/diagnostics only.

    NOTE for anyone reading TimingStore.upsert_ms once this wraps a store:
    upsert_ms will now measure this class's upsert() (a queue put - fast),
    not the real Qdrant write time, when layered as raw_store ->
    AsyncUpsertStore -> TimingStore (see reid_adapter.py). That is the
    intended effect (it reflects what the inference thread actually pays),
    not a bug - written/dropped_*/queue_depth() are the metrics that
    describe the writer thread's real behaviour instead.
    """

    def __init__(self, inner, queue_size=None, max_retries=None,
                 retry_backoff_seconds=None, shutdown_flush_seconds=None):
        self.inner = inner
        self.queue_size = (
            config.REID_ASYNC_UPSERT_QUEUE_SIZE if queue_size is None else queue_size
        )
        self.max_retries = (
            config.REID_ASYNC_UPSERT_MAX_RETRIES if max_retries is None else max_retries
        )
        self.retry_backoff_seconds = (
            config.REID_ASYNC_UPSERT_RETRY_BACKOFF_SECONDS if retry_backoff_seconds is None
            else retry_backoff_seconds
        )
        self.shutdown_flush_seconds = (
            config.REID_ASYNC_UPSERT_SHUTDOWN_FLUSH_SECONDS if shutdown_flush_seconds is None
            else shutdown_flush_seconds
        )

        self._queue = queue.Queue(maxsize=self.queue_size)
        self._stop_event = threading.Event()

        self.written = 0
        self.dropped_queue_full = 0
        self.dropped_after_retries = 0
        # PHASE 3B WORKSTREAM B: the real network-write latency, timed on
        # the writer thread around the ACTUAL self.inner.upsert() call in
        # _write_with_retry() below - never confused with TimingStore's own
        # upsert_ms, which (when this class sits between GlobalReIDManager
        # and the real store, the production-default layering) only times
        # the queue.put_nowait() in upsert() above, a near-instant enqueue,
        # not the real write. Only successful writes are timed, matching
        # self.written's own success-only accounting.
        self.write_ms = []

        self._thread = threading.Thread(target=self._run, daemon=True, name="reid-async-upsert-writer")
        self._thread.start()

    # ------------------------------------------------------------- Store API
    def search(self, embedding, top_k=1, exclude_camera=None, min_timestamp=None,
               allowed_cameras=None, only_global_id=None):
        return self.inner.search(embedding, top_k=top_k, exclude_camera=exclude_camera,
                                  min_timestamp=min_timestamp, allowed_cameras=allowed_cameras,
                                  only_global_id=only_global_id)

    def upsert(self, global_id, embedding, payload):
        # Immutable snapshot - the writer thread must never observe a
        # mutation the inference thread makes to its OWN embedding/payload
        # objects after this call returns (see the module-level requirement:
        # "queue immutable snapshots/events, not mutable references").
        # embedding is a fresh np.ndarray per call in every existing caller
        # (GlobalReIDManager.observe()'s _aggregate() output, or the raw
        # per-frame embedding) and payload is a fresh dict literal - .copy()/
        # dict() here is a deliberate belt-and-suspenders guarantee, not a
        # response to any caller that currently violates this.
        snapshot = (global_id, embedding.copy(), dict(payload))
        try:
            self._queue.put_nowait(snapshot)
        except queue.Full:
            self.dropped_queue_full += 1

    def get_history(self, global_id):
        return self.inner.get_history(global_id)

    # ------------------------------------------------------------ lifecycle
    def stop(self, flush_timeout=None):
        """
        Graceful, BOUNDED shutdown: signal the writer thread to exit once
        the queue is drained, then wait at most flush_timeout for that to
        happen - never blocks shutdown indefinitely on a slow/stuck Qdrant.
        Idempotent: a second call joins the already-finished (daemon)
        thread instantly and reports 0 remaining.

        Returns the number of items still queued (not written) after the
        bounded wait - 0 means everything was flushed in time. This module
        never prints/logs itself (see class docstring) - the caller
        (reid_adapter.py's close()) reports this if it wants to.
        """
        if flush_timeout is None:
            flush_timeout = self.shutdown_flush_seconds

        self._stop_event.set()
        self._thread.join(timeout=flush_timeout)

        return self._queue.qsize()

    def queue_depth(self):
        return self._queue.qsize()

    # ------------------------------------------------------------- writer
    def _run(self):
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    return
                continue

            self._write_with_retry(item)

    def _write_with_retry(self, item):
        """
        Deliberately does NOT bail out early just because self._stop_event
        is set: an in-flight retry sequence is exactly the "drain/flush the
        queue" work shutdown is supposed to give a fair chance to finish,
        not abandon on the spot - bailing out the instant stop() is called
        would turn a transient failure that WOULD have succeeded on its next
        attempt into a needless permanent drop, at the one moment (shutdown)
        there is no later chance to recover it. The retry count alone
        (max_retries + 1 attempts, linear backoff) already bounds how long
        any one item can occupy this thread, independent of stop_event -
        stop()'s own thread.join(timeout=flush_timeout) is the outer bound
        the CALLER waits on; it does not need this method to also self-
        abort early to stay bounded.
        """
        global_id, embedding, payload = item

        for attempt in range(self.max_retries + 1):
            try:
                start = time.perf_counter()
                self.inner.upsert(global_id, embedding, payload)
                self.write_ms.append((time.perf_counter() - start) * 1000.0)
                self.written += 1
                return
            except Exception:  # noqa: BLE001 - a write failure must never kill this thread
                if attempt >= self.max_retries:
                    self.dropped_after_retries += 1
                    return
                time.sleep(self.retry_backoff_seconds * (attempt + 1))


class InMemoryReIDStore(Store):
    """
    Phase 2 store: brute-force cosine similarity over a plain Python dict.
    Gone when the process exits - deliberately, so Phase 2 can validate the
    matching LOGIC before Phase 3 adds persistence via qdrant_reid.py.
    """

    def __init__(self, gallery_size=None):
        self.gallery_size = config.REID_GALLERY_SIZE if gallery_size is None else gallery_size
        # global_id -> list of (embedding: np.ndarray[512], payload: dict)
        self._points = {}

    def search(self, embedding, top_k=1, exclude_camera=None, min_timestamp=None,
               allowed_cameras=None, only_global_id=None):
        if not self._points:
            return []

        def point_allowed(payload):
            cam = payload.get("camera_id")

            if exclude_camera is not None and cam == exclude_camera:
                return False

            if allowed_cameras is not None and cam not in allowed_cameras:
                return False

            if min_timestamp is not None:
                ts = payload.get("timestamp")
                if ts is None or ts < min_timestamp:
                    return False

            return True

        identities = (
            {only_global_id: self._points[only_global_id]} if only_global_id in self._points
            else {}
        ) if only_global_id is not None else self._points

        scored = []
        for global_id, points in identities.items():
            eligible = [p for p in points if point_allowed(p[1])]

            if not eligible:
                continue

            # Best-matching point within this identity's ELIGIBLE gallery
            # entries, not the average - a person seen from two very
            # different angles should still match on whichever gallery entry
            # looks like this crop. An identity with points on several
            # cameras can still match here via its OTHER cameras' points even
            # when its point(s) on the excluded camera are filtered out above.
            best_embedding, best_payload = max(
                eligible, key=lambda point: float(np.dot(embedding, point[0]))
            )
            best_sim = float(np.dot(embedding, best_embedding))
            scored.append((global_id, best_sim, best_payload))

        scored.sort(key=lambda row: row[1], reverse=True)
        return scored[:top_k]

    def upsert(self, global_id, embedding, payload):
        points = self._points.setdefault(global_id, [])
        points.append((embedding, payload))

        if len(points) > self.gallery_size:
            del points[: len(points) - self.gallery_size]

    def get_history(self, global_id):
        return [payload for _, payload in self._points.get(global_id, [])]


# ============================================================
# PER-TRACK BOOTSTRAP STATE
# ============================================================

class _TrackState:
    """Everything needed to decide eligibility and, once bootstrapped, to
    remember which global identity a (camera, local_track) resolved to."""

    __slots__ = (
        "seen_count", "bootstrap_embeddings", "bootstrap_confidences",
        "global_id", "last_update_ts", "last_similarity", "last_status",
        "last_seen_ts", "first_seen_ts",
    )

    def __init__(self):
        self.seen_count = 0
        self.bootstrap_embeddings = []
        self.bootstrap_confidences = []
        self.global_id = None
        self.last_update_ts = 0.0
        self.last_similarity = None
        self.last_status = None
        # Touched every needs_observation() call (every confirmed sighting,
        # NOT just ticks Re-ID/OSNet actually ran) - the cheap "is this local
        # track still being confirmed right now" signal the experimental
        # same-camera active-identity-conflict check needs. Distinct from
        # last_update_ts, which only moves on an actual observe() decision.
        self.last_seen_ts = 0.0
        # PHASE 3A OBSERVABILITY: set ONCE, here, never mutated again - the
        # single source of truth for "how long has this local track existed"
        # (track_age_seconds on ObserveResult - see observe()). Deliberately
        # in __init__() rather than at one particular call site, since
        # _TrackState() is constructed from two places (needs_observation()
        # and observe()'s own setdefault()) and this must be correct from
        # whichever one happens to create it first. Purely observational -
        # nothing in the matching/decision logic reads this field.
        self.first_seen_ts = time.monotonic()


class ObserveResult:
    """
    Returned by observe() exactly when a decision was made this call; None
    the rest of the time (still bootstrapping).

    status - which ones are possible depends on config.REID_CAMERA_MODE (see
    its docstring). "cross_camera_only" (the default) statuses:
      NEW                   no eligible candidate cleared the threshold - a
                             fresh global_person_id was created
      MATCH                 a single candidate clearly won (cleared the
                             threshold AND beat the runner-up by min_match_margin)
      UNCERTAIN              a candidate cleared the threshold but could not be
                             told apart from a close runner-up - global_id is
                             None; nothing was written to the store
      REFRESH                an already-resolved track's periodic gallery update
      SKIPPED_SAME_CAMERA    resolved as NEW, but a same-camera identity would
                             have cleared the threshold had cross-camera-only
                             filtering not excluded it - skipped_candidate
                             carries (global_id, similarity) of what was skipped
      SEARCH_FAILED          PRODUCTION-HARDENING G-1 FIX: the bootstrap
                             decision's search() failed on every one of
                             REID_SEARCH_RETRY_MAX_ATTEMPTS attempts (a
                             transient-looking failure, e.g. Qdrant briefly
                             unavailable - see GlobalReIDManager.
                             _search_with_retry()). Handled identically to
                             UNCERTAIN: global_id is None, nothing is written
                             to the store, and the track's bootstrap buffer is
                             cleared so the NEXT eligible observation starts a
                             fresh bootstrap and gets a real chance to decide -
                             this track is NEVER permanently stranded merely
                             because one decision's search failed.
                             search_retry_events carries the attempt history.

    EXPERIMENTAL "same_camera"/"same_and_cross" statuses (see the module
    docstring above _decide_candidates() for the full same-camera policy):
      SAME_CAM_MATCH / CROSS_CAM_MATCH   like MATCH, labelled by which pool
                             the winning candidate came from ("same_camera"
                             mode only ever produces SAME_CAM_MATCH)
      SAME_CAM_UNCERTAIN     "same_camera" mode's UNCERTAIN - global_id None
      SAME_CAM_BLOCKED        resolved as NEW, but the best same-camera
                             candidate was excluded by the active-identity-
                             conflict rule, not by similarity - skipped_candidate
                             carries what was blocked, same convention as
                             SKIPPED_SAME_CAMERA
      (UNCERTAIN is reused as-is in "same_and_cross" mode when the runner-up
      spans both pools - see reasoning in the module docstring)

    global_id is None only for UNCERTAIN/SAME_CAM_UNCERTAIN - every other
    status has a real one. best_candidate_id / second_candidate_id are the
    raw search() result identities BEFORE the threshold/margin decision was
    applied - unlike global_id, best_candidate_id is populated for the
    UNCERTAIN statuses too, so analyze mode can still log which two
    identities a track was too close to call between. best_candidate_payload
    is the winning candidate's raw stored payload (camera_id, local_track_id,
    timestamp, quality) for diagnostic logging - e.g. "which local track was
    this identity previously seen as". best_candidate_scope is "same" or
    "cross" in "same_and_cross" mode (None otherwise - the mode itself already
    says which pool was searched). candidate_active says whether the best
    candidate was flagged as actively held by another local track on this
    camera at decision time (only meaningful for same-camera candidates).
    possible_id_swap is set only on some REFRESH results - see
    GlobalReIDManager._check_possible_id_swap().
    All of best_candidate_id/second_candidate_id/best_candidate_payload/
    best_candidate_scope/candidate_active are None for REFRESH itself (no
    search runs on a refresh observation) unless possible_id_swap fired.

    group / effective_threshold / effective_min_margin / effective_camera_mode
    are the CAMERA-GROUP policy actually used for this decision (see
    GlobalReIDManager._policy_for() - EXPERIMENTAL, see
    config.REID_CAMERA_GROUPS) - group is None when the track's camera was
    not in any configured group (the global defaults were used instead,
    exactly as before this feature existed); effective_camera_mode is always
    set (falls back to the manager-wide camera_mode when the group did not
    override it). Present on every status, including REFRESH, so a log/print
    line never has to guess which threshold/mode a decision was actually
    judged against.

    search_retry_events (PRODUCTION-HARDENING G-1 fix): None for the
    overwhelming majority of calls (search succeeded on the first attempt -
    zero cost, zero behaviour change when Qdrant is healthy). Otherwise a
    list of dicts, one per failed attempt, each
    {"attempt": int, "max_attempts": int, "outcome": "retry"|"exhausted",
    "error": "ExceptionType: message"} - reid_adapter.py logs from this list
    (reid_manager.py itself never prints - see GlobalReIDManager.
    _search_with_retry()'s own docstring).
    """

    def __init__(self, global_id, similarity, status,
                 second_similarity=None, margin=None, skipped_candidate=None,
                 best_candidate_id=None, second_candidate_id=None,
                 best_candidate_payload=None, best_candidate_scope=None,
                 candidate_active=None, possible_id_swap=None,
                 group=None, effective_threshold=None, effective_min_margin=None,
                 effective_camera_mode=None, search_retry_events=None,
                 candidate_count=None, candidate_camera_ids=None,
                 track_age_seconds=None, bootstrap_observations=None):
        self.global_id = global_id
        self.similarity = similarity              # None only when there was no candidate at all
        self.status = status
        self.second_similarity = second_similarity  # runner-up score, if 2+ candidates existed
        self.margin = margin                        # similarity - second_similarity
        self.skipped_candidate = skipped_candidate   # (global_id, similarity) or None
        self.best_candidate_id = best_candidate_id
        self.second_candidate_id = second_candidate_id
        self.best_candidate_payload = best_candidate_payload
        self.best_candidate_scope = best_candidate_scope   # "same" | "cross" | None
        self.candidate_active = candidate_active
        self.possible_id_swap = possible_id_swap
        self.group = group
        self.effective_threshold = effective_threshold
        self.effective_min_margin = effective_min_margin
        self.effective_camera_mode = effective_camera_mode
        self.search_retry_events = search_retry_events
        # PHASE 3A OBSERVABILITY - purely additive, read by
        # reid_adapter.py's observability sink only; nothing in this class or
        # in observe()'s decision logic reads these back. None/0 for REFRESH
        # (no search runs on a refresh - see observe()'s own REFRESH branch)
        # and for SEARCH_FAILED (the search that would have produced them
        # never completed) - both honest, not fabricated, absences.
        self.candidate_count = candidate_count              # raw candidates BEFORE threshold/margin filtering
        self.candidate_camera_ids = candidate_camera_ids    # camera_id of every raw candidate, for cross-camera analysis
        self.track_age_seconds = track_age_seconds          # now - _TrackState.first_seen_ts
        self.bootstrap_observations = bootstrap_observations  # config value in effect for this decision

    def __repr__(self):
        sim = "-" if self.similarity is None else f"{self.similarity:.3f}"
        gid = self.global_id or "-"
        return f"<ObserveResult {gid} {self.status} sim={sim}>"


# ============================================================
# CAMERA GROUP / LOCATION POLICY (EXPERIMENTAL)
#
# Pure config lookup, no manager state - independently testable, and the
# ONE place group membership is resolved (GlobalReIDManager._policy_for()
# wraps this with the global-default fallback; nothing else in this module
# reads config.REID_CAMERA_GROUPS directly - "centralise it, do not
# duplicate throughout observe()", per the spec this was built to).
# ============================================================

def resolve_camera_group(camera_id):
    """
    (group_name, group_cfg) for the group camera_id belongs to, or
    (None, None) if it is not configured into any group. camera_id is
    normalised (stripped + uppercased) before lookup, matching how
    config._validate_and_normalize_camera_groups() normalised the group
    membership lists themselves - "CAM-r09" and "CAM-R09" must resolve
    identically.
    """
    normalized = camera_id.strip().upper()

    for group_name, group_cfg in config.REID_CAMERA_GROUPS.items():
        if normalized in group_cfg["cameras"]:
            return group_name, group_cfg

    return None, None


def _aggregate(embeddings):
    """Mean of a list of unit-norm embeddings, re-normalised to unit length."""
    mean = np.mean(np.stack(embeddings), axis=0)
    norm = np.linalg.norm(mean)
    return mean if norm == 0 else mean / norm


class _SearchRetryExhausted(Exception):
    """
    PRODUCTION-HARDENING G-1 FIX - internal to this module, never escapes
    GlobalReIDManager.observe(). Raised only by _search_with_retry() after
    every attempt has failed with a TRANSIENT-looking exception (see that
    method's own docstring for exactly what counts as transient). Deliberately
    a distinct type, not a re-raise of the original exception: observe()'s
    decision block catches this ONE type specifically (narrow, precise) to
    fall back to the SEARCH_FAILED status - a non-transient exception
    (ValueError/TypeError - a malformed call, not a flaky backend) is never
    wrapped in this and propagates immediately, unretried, uncaught here,
    exactly as any other programming error already does. Chains the original
    exception via `raise ... from exc` so nothing about the real cause is lost.
    """


# ============================================================
# GLOBAL RE-ID MANAGER
# ============================================================

class GlobalReIDManager:
    def __init__(
        self,
        store,
        bootstrap_observations=None,
        bootstrap_interval_frames=None,
        update_interval_seconds=None,
        similarity_threshold=None,
        min_match_margin=None,
        cross_camera_only=None,
        max_time_gap_seconds=None,
        camera_transitions=None,
        use_camera_topology=None,
        camera_mode=None,
        same_camera_active_conflict=None,
        active_window_seconds=None,
        same_camera_cameras=None,
        first_identity_seq=1,
    ):
        # None -> read config.* fresh here, not as a default-parameter value
        # (a default parameter value is bound once at import time, so
        # reassigning e.g. config.REID_SIMILARITY_THRESHOLD afterwards would
        # silently have no effect on a constructor that used it as a default).
        self.store = store
        self.bootstrap_observations = (
            config.REID_BOOTSTRAP_OBSERVATIONS if bootstrap_observations is None
            else bootstrap_observations
        )
        self.bootstrap_interval_frames = (
            config.REID_BOOTSTRAP_INTERVAL_FRAMES if bootstrap_interval_frames is None
            else bootstrap_interval_frames
        )
        self.update_interval_seconds = (
            config.REID_UPDATE_INTERVAL_SECONDS if update_interval_seconds is None
            else update_interval_seconds
        )
        self.similarity_threshold = (
            config.REID_SIMILARITY_THRESHOLD if similarity_threshold is None
            else similarity_threshold
        )
        # margin required over the runner-up, on top of clearing the
        # threshold, before a MATCH is accepted rather than UNCERTAIN - see
        # config.REID_MIN_MATCH_MARGIN.
        self.min_match_margin = (
            config.REID_MIN_MATCH_MARGIN if min_match_margin is None else min_match_margin
        )
        # SUPERSEDED, kept only for backward-compat callers still passing
        # this kwarg (e.g. test_cross_camera_policy.py) - self.camera_mode
        # is what actually decides same-camera exclusion now (see below and
        # observe()'s cross_camera_only branch, which no longer reads this).
        self.cross_camera_only = (
            config.REID_CROSS_CAMERA_ONLY if cross_camera_only is None else cross_camera_only
        )

        # EXPERIMENTAL same-camera Re-ID mode - see config.py's own docstring
        # for the full policy. Validated here too (not just at config.py
        # import time) so a caller constructing GlobalReIDManager directly
        # with a bad string (e.g. from a test) fails loudly the same way.
        self.camera_mode = config.REID_CAMERA_MODE if camera_mode is None else camera_mode
        if self.camera_mode not in ("cross_camera_only", "same_camera", "same_and_cross"):
            raise ValueError(f"camera_mode={self.camera_mode!r} is not valid")

        self.same_camera_active_conflict = (
            config.REID_SAME_CAMERA_ACTIVE_CONFLICT if same_camera_active_conflict is None
            else same_camera_active_conflict
        )
        self.active_window_seconds = (
            config.REID_ACTIVE_WINDOW_SECONDS if active_window_seconds is None
            else active_window_seconds
        )

        # Cameras whose camera_mode is derived from group membership rather
        # than from self.camera_mode - see config.REID_SAME_CAMERA_CAMERAS
        # and _resolve_camera_mode() below. None -> read config fresh, same
        # convention as every other setting here.
        self.same_camera_cameras = (
            config.REID_SAME_CAMERA_CAMERAS if same_camera_cameras is None
            else frozenset(c.strip().upper() for c in same_camera_cameras)
        )

        self.max_time_gap_seconds = (
            config.REID_MAX_TIME_GAP_SECONDS if max_time_gap_seconds is None else max_time_gap_seconds
        )

        use_topology = (
            config.REID_USE_CAMERA_TOPOLOGY if use_camera_topology is None else use_camera_topology
        )
        transitions = config.CAMERA_TRANSITIONS if camera_transitions is None else camera_transitions
        # None (not {}) means "disabled" - _allowed_cameras_for() treats both
        # None and an empty dict as "no restriction", but keeping the two
        # separate here makes an explicit use_camera_topology=False always
        # win, regardless of what a caller passed as camera_transitions.
        self.camera_transitions = transitions if use_topology else None

        # (camera_id, local_track_id) -> _TrackState
        self._tracks = {}

        # global_id -> {first_seen, last_seen, quality, camera_history: [...]}
        self._identities = {}

        # first_identity_seq: 1 for a store that starts empty every process
        # (InMemoryReIDStore); a persistent store's caller passes its
        # highest_identity_seq() + 1 - see IDENTITY NUMBERING above.
        if not isinstance(first_identity_seq, int) or first_identity_seq < 1:
            raise ValueError(f"first_identity_seq={first_identity_seq!r} must be an int >= 1")
        self.first_identity_seq = first_identity_seq
        self._next_seq = itertools.count(first_identity_seq)
        self._id_lock = threading.Lock()

    # ------------------------------------------------------------- identity
    def _new_global_id(self):
        # Observe runs on the inference thread only today; the lock keeps two
        # allocations distinct even if that ever changes.
        with self._id_lock:
            seq = next(self._next_seq)
        return f"P-{seq:04d}"

    def _allowed_cameras_for(self, camera_id):
        """
        None -> no topology restriction (any camera is a valid cross-camera
        candidate, subject only to the cross_camera_only exclusion). Optional
        and OFF by default - see config.CAMERA_TRANSITIONS. A camera_id
        missing from the map is deliberately treated as unrestricted, not
        blocked, so a partially-filled topology never silently excludes every
        candidate for a camera nobody has mapped yet.
        """
        if not self.camera_transitions:
            return None

        return self.camera_transitions.get(camera_id)

    def _resolve_camera_mode(self, camera_id, group_mode, grouped):
        """
        The effective camera_mode for one camera, in strict precedence order.

          1. group_mode - the camera's GROUP set camera_mode explicitly in
             the dashboard. An explicit operator choice always wins; a
             rollout list must never override one.
          2. config.REID_SAME_CAMERA_CAMERAS names this camera - derive the
             mode from group membership:
                 grouped   -> "same_and_cross"
                 ungrouped -> "same_camera"
             See that setting's own docstring for the production evidence
             behind it.
          3. self.camera_mode - the deployment-wide default, exactly as
             before this rollout mechanism existed.

        THE SAFETY PROPERTY, and why it needs no enforcement code here: an
        UNGROUPED camera resolves "same_camera", and observe()'s same_camera
        branch searches allowed_cameras={camera_id} and never reads
        policy["allowed_cameras"] at all. It is not that cross-camera
        matching is checked and refused for such a camera - the cross-camera
        pool is never searched, so there is nothing to refuse. (Its
        allowed_cameras is separately set() by the tiers below, so even the
        same_and_cross branch could not reach another camera from here.)
        """
        if group_mode:
            return group_mode

        if camera_id.strip().upper() in self.same_camera_cameras:
            return "same_and_cross" if grouped else "same_camera"

        return self.camera_mode

    def _policy_for(self, camera_id):
        """
        THE single place a camera's effective Re-ID policy is resolved -
        every threshold/margin comparison and every cross-camera allowed-set
        computation in observe()/_check_possible_id_swap() goes through this,
        never config.REID_SIMILARITY_THRESHOLD/REID_MIN_MATCH_MARGIN directly,
        so a camera-group override can never be accidentally bypassed by a
        hidden global-constant read somewhere else in the decision logic.

        "authorized" (production-hardening P0 fix) is resolved FIRST and
        independently of everything else below: camera_id must appear in
        config.REID_ENABLED_CAMERAS - the dashboard's explicit Re-ID
        authorization set (Camera.reid_enabled, fetched via reid_config_
        provider.py) - or every OTHER field in the returned policy is
        irrelevant, because GlobalReIDManager.needs_observation() refuses to
        even bootstrap an unauthorized camera's tracks (see that method).
        This is a SEPARATE axis from group membership: a camera can be
        authorized with no group (falls to tier 3 below) or, in principle,
        configured into a group without being independently authorized (an
        inconsistency config._validate_and_normalize_enabled_cameras()
        already rejects at the config-loading boundary, not here).

        Precedence, camera-group config always wins over the global default:
          1. camera_id is in a configured REID_CAMERA_GROUPS group -> that
             group's similarity_threshold/min_match_margin/camera_mode
             (falling back to the global default for just the one field if
             the group left it unset), and allowed_cameras = that group's
             own camera list.
          2. camera_id is NOT in any group, but config.REID_DEFAULT_GROUP
             names one -> that group's threshold/margin/camera_mode ONLY
             (never its allowed_cameras - an unconfigured camera is not
             silently granted candidate access to a group it was never
             placed in). allowed_cameras is EMPTY (set()), not None - see
             the P0 fix note below.
          3. Neither -> self.similarity_threshold/self.min_match_margin/
             self.camera_mode (this manager's own global values, exactly as
             before camera groups existed) and allowed_cameras=set() - see
             the P0 fix note below.

        P0 FIX - allowed_cameras is EMPTY (set()), never None, in tiers 2/3:
        a camera with no group membership of its own must NEVER be able to
        cross-camera-search the ENTIRE store unrestricted merely because it
        was never explicitly configured - that was CONFIRMED, reproducible
        production-hardening finding #1 (group isolation bypass). set() and
        None are handled identically by _intersect_allowed()/Store.search()
        EXCEPT that set() actually excludes every candidate (confirmed
        against both InMemoryReIDStore and the real isolated Qdrant
        instance's MatchAny filter), while None means "no restriction at
        all" - this single character of difference (an empty set literal
        instead of None) IS the fix. Same-camera matching is COMPLETELY
        UNAFFECTED: it never reads this field at all (see observe()'s
        same_camera branch, which always searches allowed_cameras=
        {camera_id} directly) - only the CROSS-camera candidate pool for an
        unconfigured camera changes, from "everyone" to "no one", which is
        the safe, explicitly-chosen fallback per the production-hardening
        brief ("enabled + no group -> safe explicitly-defined fallback").

        camera_mode resolution is the SAME fallback shape as
        threshold/margin, not a separate mechanism - a group only needs to
        set it when its policy genuinely differs from the rest of the
        deployment (see config._validate_and_normalize_camera_groups()).

        Returns {"authorized", "group", "allowed_cameras",
        "similarity_threshold", "min_match_margin", "camera_mode"}.
        allowed_cameras is always a set (never None) since the P0 fix -
        empty means "no cross-camera candidates."
        """
        # Same inline normalisation resolve_camera_group() already uses
        # below (not config._normalize_camera_id - that helper is private to
        # config.py; matching its behaviour inline here avoids a cross-module
        # private-function call for what is a one-line string operation).
        authorized = camera_id.strip().upper() in config.REID_ENABLED_CAMERAS

        group_name, group_cfg = resolve_camera_group(camera_id)

        if group_cfg is not None:
            return {
                "authorized": authorized,
                "group": group_name,
                "allowed_cameras": set(group_cfg["cameras"]),
                "similarity_threshold": (
                    group_cfg["similarity_threshold"] if group_cfg["similarity_threshold"] is not None
                    else self.similarity_threshold
                ),
                "min_match_margin": (
                    group_cfg["min_match_margin"] if group_cfg["min_match_margin"] is not None
                    else self.min_match_margin
                ),
                "camera_mode": self._resolve_camera_mode(
                    camera_id, group_cfg["camera_mode"], grouped=True
                ),
            }

        if config.REID_DEFAULT_GROUP is not None:
            fallback_cfg = config.REID_CAMERA_GROUPS[config.REID_DEFAULT_GROUP]
            return {
                "authorized": authorized,
                "group": None,
                "allowed_cameras": set(),
                "similarity_threshold": (
                    fallback_cfg["similarity_threshold"] if fallback_cfg["similarity_threshold"] is not None
                    else self.similarity_threshold
                ),
                "min_match_margin": (
                    fallback_cfg["min_match_margin"] if fallback_cfg["min_match_margin"] is not None
                    else self.min_match_margin
                ),
                # grouped=False: REID_DEFAULT_GROUP lends this camera a
                # threshold/margin/mode, never MEMBERSHIP (allowed_cameras
                # stays set() just below) - so for mode-resolution purposes
                # it is an ungrouped camera, and the rollout set gives it
                # "same_camera", not "same_and_cross". Resolving it as
                # grouped would hand it a cross-camera branch with an empty
                # candidate pool: strictly wasted searches.
                "camera_mode": self._resolve_camera_mode(
                    camera_id, fallback_cfg["camera_mode"], grouped=False
                ),
            }

        return {
            "authorized": authorized,
            "group": None,
            "allowed_cameras": set(),
            "similarity_threshold": self.similarity_threshold,
            "min_match_margin": self.min_match_margin,
            "camera_mode": self._resolve_camera_mode(camera_id, None, grouped=False),
        }

    @staticmethod
    def _intersect_allowed(*camera_sets):
        """
        Combine any number of optional allowed-camera restrictions (topology,
        camera-group - each independently None meaning "no restriction from
        this source") into one - generalises the existing single-source
        allowed_cameras mechanism rather than adding a second, parallel
        filtering path. None only if EVERY source is unrestricted.
        """
        restrictions = [s for s in camera_sets if s is not None]

        if not restrictions:
            return None

        combined = restrictions[0]
        for s in restrictions[1:]:
            combined = combined & s

        return combined

    def stable_id_for(self, camera_id, local_track_id):
        """
        The resolved global identity for one local track, or None.

        READ-ONLY and side-effect-free: it creates no _TrackState, touches no
        timestamp, and never triggers a decision - unlike needs_observation(),
        which increments a counter and must not be called more than once per
        tick. This is what the event pipeline calls, potentially several times
        per track per tick, to stamp metadata["stable_id"].

        None means "no identity yet" and is the normal case for: a camera
        without Re-ID, a vehicle (Re-ID is person-only), a track still
        collecting bootstrap observations, and one whose last decision was
        UNCERTAIN. Callers must treat None as "omit the field", never as an
        error.
        """
        state = self._tracks.get((camera_id, local_track_id))

        return state.global_id if state is not None else None

    def recovery_provenance(self, camera_id, global_id, exclude_local_track_id, now=None):
        """
        Which OTHER local track on this camera last held global_id, and how
        long ago it was last confirmed - the evidence that a same-camera
        decision actually rejoined a fragmented identity rather than minting
        a fresh one.

        Returns (previous_local_track_id, seconds_since_last_seen), or
        (None, None) when no other local track here holds that identity -
        which is the correct answer for a genuinely new identity, and also
        for a recovery whose predecessor has already aged out of _tracks
        (REID_TRACK_STATE_TTL_SECONDS). The second case is reported honestly
        as "unknown", never guessed at.

        Purely observational. Nothing in the matching or decision logic reads
        this, and it is computed AFTER observe() has already decided.
        """
        if global_id is None:
            return None, None

        now = time.monotonic() if now is None else now
        best_track, best_age = None, None

        for (cid, track_id), state in self._tracks.items():
            if cid != camera_id or track_id == exclude_local_track_id:
                continue
            if state.global_id != global_id:
                continue

            age = now - state.last_seen_ts
            if best_age is None or age < best_age:
                best_track, best_age = track_id, age

        return best_track, best_age

    def _active_identities_on_camera(self, camera_id, exclude_local_track_id, now):
        """
        Global identities currently held by some OTHER local track on this
        camera, "currently" meaning confirmed (needs_observation() called,
        NOT necessarily run through Re-ID) within active_window_seconds - the
        set the same-camera active-identity-conflict rule must never match a
        NEW local track onto (see config.REID_SAME_CAMERA_ACTIVE_CONFLICT).
        """
        return {
            state.global_id
            for (cid, track_id), state in self._tracks.items()
            if cid == camera_id
            and track_id != exclude_local_track_id
            and state.global_id is not None
            and (now - state.last_seen_ts) <= self.active_window_seconds
        }

    def _search_with_retry(self, camera_id, local_track_id, retry_events, **search_kwargs):
        """
        PRODUCTION-HARDENING G-1 FIX. Wraps ONE self.store.search() call
        (search_kwargs are exactly what search() itself takes) with a small,
        bounded retry for TRANSIENT-looking failures - CONFIRMED finding: if
        Qdrant is briefly unavailable at the exact moment a track's bootstrap
        decision runs, the search() failure used to strand that local track
        permanently (see the module docstring's "MOST IMPORTANT" note and
        the validation report). Used ONLY for the searches a bootstrap
        DECISION makes (observe()'s cross_camera_only/same_camera/
        same_and_cross branches) - never for _check_possible_id_swap()'s
        diagnostic searches, which already self-heal on the next REFRESH
        cycle via needs_observation()'s own last_update_ts check and do not
        need this (retrying a pure diagnostic would not close any real gap).

        "Transient" is decided narrowly and store-agnostically, without this
        module needing to know anything about Qdrant-specific exception
        types (see the module docstring - this file knows nothing about
        Qdrant specifically): ValueError/TypeError signal a malformed
        call (bad arguments, wrong types) - retrying an inherently-broken
        request only wastes the retry budget and delays surfacing a real bug,
        so those propagate immediately, unretried, exactly as before this
        fix existed. Every other exception (connection errors, timeouts,
        whatever a real backend outage actually raises) is retried.

        On success (first try or a later one), returns search()'s normal
        result, appending one {"outcome": "retry", ...} record per FAILED
        attempt to retry_events (a plain list the caller owns and passes in -
        this method never prints; see ObserveResult.search_retry_events'
        own docstring for why logging lives in reid_adapter.py, not here).
        On exhausting config.REID_SEARCH_RETRY_MAX_ATTEMPTS attempts, appends
        one final {"outcome": "exhausted", ...} record and raises
        _SearchRetryExhausted (chained from the last real exception) - never
        swallowed here, never turned into an empty/fabricated result, so the
        caller cannot mistake "search could not even run" for "search ran
        and found nothing" (which would wrongly create a brand-new identity).

        Bounded by construction: REID_SEARCH_RETRY_MAX_ATTEMPTS total tries,
        REID_SEARCH_RETRY_BACKOFF_SECONDS linear backoff between them (3
        attempts / 50ms default -> at most 150ms added latency) - runs on
        the calling (inference) thread, so both are deliberately small; this
        is not a background retry queue and never becomes one.
        """
        max_attempts = config.REID_SEARCH_RETRY_MAX_ATTEMPTS
        backoff_seconds = config.REID_SEARCH_RETRY_BACKOFF_SECONDS

        for attempt in range(1, max_attempts + 1):
            try:
                return self.store.search(**search_kwargs)
            except (ValueError, TypeError):
                raise  # non-transient - never retried, propagates immediately
            except Exception as exc:  # noqa: BLE001 - classified above, not blanket-swallowed
                exhausted = attempt >= max_attempts
                retry_events.append({
                    "attempt": attempt, "max_attempts": max_attempts,
                    "outcome": "exhausted" if exhausted else "retry",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                if exhausted:
                    raise _SearchRetryExhausted(
                        f"search() failed after {max_attempts} attempt(s) for "
                        f"camera={camera_id} local_track_id={local_track_id}"
                    ) from exc
                time.sleep(backoff_seconds * attempt)

    @staticmethod
    def _decide_candidates(best, second, threshold, min_match_margin):
        """
        The shared best-vs-second-best threshold+margin rule every camera
        mode uses: MATCH only if best clears the threshold AND (there is no
        runner-up, or best beats it by min_match_margin); UNCERTAIN if best
        clears the threshold but the margin does not; NEW otherwise. best /
        second are (global_id, similarity, payload) tuples or None. Returns
        (decision, best_similarity, second_similarity, margin) - decision is
        "MATCH" | "UNCERTAIN" | "NEW", the caller maps that onto the actual
        mode-specific status string (e.g. SAME_CAM_MATCH vs MATCH).
        """
        best_sim = best[1] if best is not None else None
        second_sim = second[1] if second is not None else None
        margin = (best_sim - second_sim) if second_sim is not None else None

        if best is not None and best_sim >= threshold and (second is None or margin >= min_match_margin):
            return "MATCH", best_sim, second_sim, margin

        if best is not None and best_sim >= threshold:
            return "UNCERTAIN", best_sim, second_sim, margin

        return "NEW", best_sim, second_sim, margin

    def _touch_identity(self, global_id, camera_id, local_track_id, quality, now):
        identity = self._identities.get(global_id)

        if identity is None:
            identity = {
                "first_seen": now,
                "last_seen": now,
                "quality": quality,
                "camera_history": [],
            }
            self._identities[global_id] = identity

        identity["last_seen"] = now

        history = identity["camera_history"]
        current = next(
            (h for h in history
             if h["camera_id"] == camera_id and h["local_track_id"] == local_track_id),
            None,
        )

        if current is None:
            history.append({
                "camera_id": camera_id,
                "local_track_id": local_track_id,
                "first_seen": now,
                "last_seen": now,
            })
        else:
            current["last_seen"] = now

        # Running mean quality across every observation the identity has had,
        # weighted equally per resolution event (bootstrap or refresh).
        identity["quality"] = (identity["quality"] + quality) / 2.0

    # -------------------------------------------------------------- gating
    def needs_observation(self, camera_id, local_track_id):
        """
        Call ONCE per confirmed sighting of a person track, every tick,
        BEFORE deciding whether to crop + run OSNet. Increments this track's
        internal seen-counter as a side effect, so it must not be called more
        than once per tick per track.

        Returns True the ticks this track should actually be cropped and run
        through OSNet this iteration.

        PRODUCTION-HARDENING P0 FIX: always False for a camera not in
        config.REID_ENABLED_CAMERAS (the dashboard's Re-ID authorization set -
        see _policy_for()'s "authorized" field and config.py's own docstring
        for the full reasoning). This is the actual enforcement point: an
        unauthorized camera's local tracks never get a _TrackState entry
        created at all, are never cropped, never reach OSNet, never search,
        never write to the store - "disabled camera never calls Re-ID
        matching / never creates Re-ID identities" holds because this
        function, the ONE gate every caller (reid_adapter.py's
        process_tracks()) already goes through before queuing a crop,
        refuses at the very first line. A direct set-membership check, not
        the full _policy_for() resolution - this runs on the hottest path
        in the whole Re-ID system (every confirmed sighting, every tick),
        and authorization alone does not need group/threshold resolution.
        """
        if camera_id.strip().upper() not in config.REID_ENABLED_CAMERAS:
            return False

        key = (camera_id, local_track_id)
        state = self._tracks.get(key)

        if state is None:
            state = _TrackState()
            self._tracks[key] = state

        state.seen_count += 1
        state.last_seen_ts = time.monotonic()

        if state.global_id is None:
            # Still bootstrapping - eligible every Nth confirmed frame of
            # THIS track, until enough observations are collected.
            if len(state.bootstrap_embeddings) >= self.bootstrap_observations:
                return False   # collected enough; observe() just hasn't run yet
            return state.seen_count % self.bootstrap_interval_frames == 0

        # Already resolved - eligible only after the refresh interval elapses.
        return (time.monotonic() - state.last_update_ts) >= self.update_interval_seconds

    # ------------------------------------------------------------- observe
    def observe(self, camera_id, local_track_id, embedding, confidence, timestamp=None):
        """
        Feed one OSNet embedding for a track that needs_observation() said
        was eligible. Returns an ObserveResult when a decision was made this
        call (NEW / MATCH / UNCERTAIN / SKIPPED_SAME_CAMERA / REFRESH), else
        None (still collecting bootstrap observations).

        DEFENSE IN DEPTH for the P0 authorization fix: needs_observation()
        is the actual enforcement point (an unauthorized camera's tracks
        never get queued for OSNet in the first place, so this call should
        never happen for one in normal operation) - this second check
        exists only so a caller that ever invokes observe() directly,
        bypassing needs_observation(), still cannot bootstrap or write
        anything for an unauthorized camera. Returns None (identical to
        "still collecting bootstrap observations" - the safest, most
        conservative signal: nothing happened, nothing was written) without
        creating or touching any _TrackState.
        """
        if camera_id.strip().upper() not in config.REID_ENABLED_CAMERAS:
            return None

        now = time.monotonic()
        key = (camera_id, local_track_id)
        state = self._tracks.setdefault(key, _TrackState())

        if state.global_id is None:
            state.bootstrap_embeddings.append(embedding)
            state.bootstrap_confidences.append(confidence)

            if len(state.bootstrap_embeddings) < self.bootstrap_observations:
                return None   # still collecting

            representative = _aggregate(state.bootstrap_embeddings)
            quality = float(np.mean(state.bootstrap_confidences))

            min_timestamp = (
                timestamp - self.max_time_gap_seconds
                if (timestamp is not None and self.max_time_gap_seconds is not None)
                else None
            )

            # THE camera-group policy resolution point (EXPERIMENTAL - see
            # config.REID_CAMERA_GROUPS): every threshold/margin comparison
            # and every cross-camera allowed-set computation below reads
            # policy[...], never self.similarity_threshold/self.min_match_margin
            # directly, so a group override can never be silently bypassed.
            policy = self._policy_for(camera_id)
            threshold = policy["similarity_threshold"]
            min_match_margin = policy["min_match_margin"]

            skipped_candidate = None
            best_payload = None
            best_scope = None
            candidate_active = None
            # PRODUCTION-HARDENING G-1 FIX support - see _search_with_retry()'s
            # own docstring. Stays [] (falsy) for the overwhelming majority of
            # calls where every search() succeeds first try; only populated
            # when a retry actually happened, so ObserveResult.
            # search_retry_events is None (not an empty list) in the common
            # case - see its own docstring for why that distinction matters
            # to the caller's logging.
            retry_events = []
            # PHASE 3A OBSERVABILITY: the raw, pre-threshold/margin candidate
            # list actually searched this decision - "matches" in the
            # cross_camera_only branch below, "pool" in the same_camera/
            # same_and_cross branch - captured here under ONE name so it is
            # always defined by the time the return statement below builds
            # candidate_count/candidate_camera_ids, including the
            # _SearchRetryExhausted except branch (empty: the search that
            # would have produced candidates never completed). Purely
            # observational - the actual decision keeps reading best/second/
            # threshold/margin exactly as before; nothing here changes it.
            raw_candidates = []

            try:
                if policy["camera_mode"] == "cross_camera_only":
                    # UNCHANGED from before REID_CAMERA_MODE existed - byte-for-
                    # byte the same decision this branch always made (now routed
                    # through the shared _decide_candidates() helper, verified to
                    # produce identical output - see test_cross_camera_policy.py).
                    #
                    # THE cross-camera policy: this camera's own points are
                    # excluded from the candidate pool entirely - two independent
                    # local tracks on the SAME camera can then never merge
                    # through appearance alone. allowed_cameras is the
                    # intersection of the optional topology restriction and the
                    # optional camera-group restriction - either, both, or
                    # neither may be active; None only if neither is.
                    #
                    # Branches on policy["camera_mode"] (the per-camera-group-
                    # resolved value, falling back to self.camera_mode when the
                    # camera's group does not override it - see _policy_for()),
                    # NOT self.camera_mode directly, so a group's camera_mode
                    # override actually takes effect here. Unconditional, not
                    # gated on self.cross_camera_only: this whole branch already
                    # only runs when the resolved mode == "cross_camera_only", so
                    # that flag is fully redundant here now - and reading it
                    # directly was a real bug: self.cross_camera_only defaults
                    # from config.REID_CROSS_CAMERA_ONLY, a flag documented as
                    # SUPERSEDED-and-unread since REID_CAMERA_MODE was
                    # introduced. Its default was later changed to False on
                    # disk, which silently disabled same-camera exclusion for
                    # any caller not ALSO passing cross_camera_only=True
                    # explicitly (test_cross_camera_policy.py's own new_manager()
                    # happened to still pass it, masking this until
                    # test_camera_groups.py's new_manager() - correctly relying
                    # on camera_mode alone, per the flag's own "superseded"
                    # documentation - exposed it). Found via an earlier task's TEST 8/9.
                    exclude_camera = camera_id
                    allowed_cameras = self._intersect_allowed(
                        self._allowed_cameras_for(camera_id), policy["allowed_cameras"]
                    )

                    matches = self._search_with_retry(
                        camera_id, local_track_id, retry_events,
                        embedding=representative, top_k=2,
                        exclude_camera=exclude_camera,
                        min_timestamp=min_timestamp,
                        allowed_cameras=allowed_cameras,
                    )
                    raw_candidates = matches
                    best = matches[0] if matches else None
                    second = matches[1] if len(matches) > 1 else None

                    decision, best_sim, second_sim, margin = self._decide_candidates(
                        best, second, threshold, min_match_margin
                    )

                    if decision == "MATCH":
                        global_id, similarity, status = best[0], best_sim, "MATCH"
                    elif decision == "UNCERTAIN":
                        global_id, similarity, status = None, best_sim, "UNCERTAIN"
                    else:
                        # Nothing in the allowed candidate pool came close - a
                        # genuinely new identity, UNLESS the only reason the pool
                        # excluded a real match is the same-camera policy itself.
                        global_id, similarity, status = self._new_global_id(), best_sim, "NEW"

                        if exclude_camera is not None:
                            same_camera_matches = self._search_with_retry(
                                camera_id, local_track_id, retry_events,
                                embedding=representative, top_k=1,
                                exclude_camera=None,
                                min_timestamp=min_timestamp,
                                allowed_cameras={exclude_camera},
                            )
                            if same_camera_matches and same_camera_matches[0][1] >= threshold:
                                status = "SKIPPED_SAME_CAMERA"
                                skipped_candidate = (same_camera_matches[0][0], same_camera_matches[0][1])

                    if best is not None:
                        best_payload = best[2]

                else:
                    # EXPERIMENTAL: "same_camera" or "same_and_cross" - see
                    # config.py's REID_CAMERA_MODE docstring for the full policy.
                    same_filtered, same_blocked, cross_matches = [], None, []

                    if policy["camera_mode"] in ("same_camera", "same_and_cross"):
                        # Extra headroom when conflict-filtering is on, so a
                        # blocked top candidate does not silently swallow what
                        # would otherwise have been a legitimate #2/#3 candidate.
                        raw_top_k = 7 if self.same_camera_active_conflict else 2
                        same_raw = self._search_with_retry(
                            camera_id, local_track_id, retry_events,
                            embedding=representative, top_k=raw_top_k,
                            allowed_cameras={camera_id}, min_timestamp=min_timestamp,
                        )

                        if self.same_camera_active_conflict:
                            active_ids = self._active_identities_on_camera(camera_id, local_track_id, now)
                            same_filtered = [c for c in same_raw if c[0] not in active_ids]
                            conflicted = [c for c in same_raw if c[0] in active_ids]
                            same_blocked = conflicted[0] if conflicted else None
                        else:
                            same_filtered = same_raw

                    if policy["camera_mode"] == "same_and_cross":
                        cross_matches = self._search_with_retry(
                            camera_id, local_track_id, retry_events,
                            embedding=representative, top_k=2,
                            exclude_camera=camera_id,
                            min_timestamp=min_timestamp,
                            allowed_cameras=self._intersect_allowed(
                                self._allowed_cameras_for(camera_id), policy["allowed_cameras"]
                            ),
                        )

                    # Combined, re-ranked pool - the single best candidate wins
                    # regardless of which pool it came from; each candidate keeps
                    # a tag saying which pool it is, purely for status labelling.
                    pool = sorted(
                        [(*c, "same") for c in same_filtered] + [(*c, "cross") for c in cross_matches],
                        key=lambda c: c[1], reverse=True,
                    )
                    raw_candidates = pool

                    best = pool[0][:3] if pool else None
                    second = pool[1][:3] if len(pool) > 1 else None
                    best_scope = pool[0][3] if pool else None

                    decision, best_sim, second_sim, margin = self._decide_candidates(
                        best, second, threshold, min_match_margin
                    )

                    if decision == "MATCH":
                        global_id, similarity = best[0], best_sim
                        status = "SAME_CAM_MATCH" if best_scope == "same" else "CROSS_CAM_MATCH"
                    elif decision == "UNCERTAIN":
                        global_id, similarity = None, best_sim
                        # A pure same-camera search's ambiguity is unambiguously
                        # a same-camera one; same_and_cross's runner-up can come
                        # from EITHER pool, so a same/cross-specific label there
                        # would misrepresent what was actually ambiguous.
                        status = "SAME_CAM_UNCERTAIN" if policy["camera_mode"] == "same_camera" else "UNCERTAIN"
                    else:
                        global_id, similarity, status = self._new_global_id(), best_sim, "NEW"

                        if same_blocked is not None and same_blocked[1] >= threshold:
                            status = "SAME_CAM_BLOCKED"
                            skipped_candidate = (same_blocked[0], same_blocked[1])

                    if best is not None:
                        best_payload = best[2]
                        # The winner survived the active-conflict filter (or the
                        # filter is off, in which case we never checked - None,
                        # not a false "not active", is the honest answer there).
                        if self.same_camera_active_conflict and best_scope == "same":
                            candidate_active = False
            except _SearchRetryExhausted:
                # PRODUCTION-HARDENING G-1 FIX: every self._search_with_retry()
                # attempt failed for this decision. Handled IDENTICALLY to the
                # UNCERTAIN branch just below (global_id stays None, nothing is
                # written to the store, bootstrap buffer is cleared) - the
                # MOST IMPORTANT requirement this fix exists for: this track
                # must remain eligible for a fresh bootstrap attempt on a
                # FUTURE observation, never permanently stranded merely
                # because Qdrant was briefly unavailable at this exact moment.
                # best/second/second_sim/margin are never referenced again
                # below when status == "SEARCH_FAILED" except to build the
                # (all-None) ObserveResult - explicit here rather than relying
                # on whatever partial state the try body happened to reach.
                global_id, similarity, status = None, None, "SEARCH_FAILED"
                best, second, second_sim, margin = None, None, None, None

            if status in ("UNCERTAIN", "SAME_CAM_UNCERTAIN", "SEARCH_FAILED"):
                # Not resolved - retry with a FRESH batch rather than reusing
                # embeddings that already produced an unclear result (UNCERTAIN)
                # or that a failed search never got to evaluate at all
                # (SEARCH_FAILED - PRODUCTION-HARDENING G-1 FIX, reusing this
                # exact same "clear the buffer, become eligible again" path
                # deliberately rather than inventing a second one - see
                # _search_with_retry()'s docstring). state.global_id stays
                # None; status_for() still reports this decision (see its own
                # gate) until a retry replaces it.
                state.bootstrap_embeddings = []
                state.bootstrap_confidences = []
                # Nothing is written to the store for an uncertain or failed observation.
            else:
                state.global_id = global_id

                self.store.upsert(global_id, representative, {
                    "camera_id": camera_id,
                    "local_track_id": local_track_id,
                    "timestamp": timestamp,
                    "quality": quality,
                })

                self._touch_identity(global_id, camera_id, local_track_id, quality, now)

            state.last_update_ts = now
            state.last_similarity = similarity
            state.last_status = status

            return ObserveResult(global_id, similarity, status,
                                 second_similarity=second_sim, margin=margin,
                                 skipped_candidate=skipped_candidate,
                                 best_candidate_id=best[0] if best is not None else None,
                                 second_candidate_id=second[0] if second is not None else None,
                                 best_candidate_payload=best_payload,
                                 best_candidate_scope=best_scope,
                                 candidate_active=candidate_active,
                                 group=policy["group"],
                                 effective_threshold=threshold,
                                 effective_min_margin=min_match_margin,
                                 effective_camera_mode=policy["camera_mode"],
                                 search_retry_events=retry_events or None,
                                 candidate_count=len(raw_candidates),
                                 candidate_camera_ids=[c[2].get("camera_id") for c in raw_candidates],
                                 track_age_seconds=now - state.first_seen_ts,
                                 bootstrap_observations=self.bootstrap_observations)

        # Already resolved - this is a periodic refresh observation. It
        # extends the identity's gallery with fresh appearance data (new
        # pose/lighting/camera) but does not re-run matching against other
        # identities - state.global_id is already decided for this track.
        #
        # The possible-ID-swap diagnostic MUST run before the upsert below -
        # otherwise "compare this embedding to its own identity's gallery"
        # would trivially find the point this very call is about to insert
        # (similarity ~1.0 against itself), masking any real swap.
        policy = self._policy_for(camera_id)
        possible_id_swap = self._check_possible_id_swap(
            camera_id, local_track_id, state.global_id, embedding, now,
            policy["similarity_threshold"], policy["min_match_margin"],
            policy["camera_mode"],
        )

        self.store.upsert(state.global_id, embedding, {
            "camera_id": camera_id,
            "local_track_id": local_track_id,
            "timestamp": timestamp,
            "quality": confidence,
        })

        state.last_update_ts = now
        state.last_status = "REFRESH"

        self._touch_identity(state.global_id, camera_id, local_track_id, confidence, now)

        return ObserveResult(state.global_id, state.last_similarity, "REFRESH",
                             possible_id_swap=possible_id_swap,
                             group=policy["group"],
                             effective_threshold=policy["similarity_threshold"],
                             effective_min_margin=policy["min_match_margin"],
                             effective_camera_mode=policy["camera_mode"],
                             # candidate_count/candidate_camera_ids deliberately
                             # left None (not 0/[]) - no search runs on a
                             # REFRESH (see this branch's own docstring above),
                             # so "no candidates" would be a fabricated claim;
                             # None honestly means "not applicable to this
                             # decision type". track_age_seconds/bootstrap_
                             # observations ARE meaningful for REFRESH too.
                             track_age_seconds=now - state.first_seen_ts,
                             bootstrap_observations=self.bootstrap_observations)

    def _check_possible_id_swap(self, camera_id, local_track_id, own_global_id, embedding, now,
                                 threshold, min_match_margin, camera_mode):
        """
        DIAGNOSTIC ONLY - never changes state.global_id, never touches
        TrackManager or JeztSort's own ids. Runs only when camera_mode gives
        the manager same-camera awareness at all, piggybacked on the existing
        REFRESH cadence (REID_UPDATE_INTERVAL_SECONDS) so it costs no extra
        OSNet call and no extra per-frame search.

        camera_mode is the CALLER's resolved policy["camera_mode"] (per-group
        override, falling back to the manager-wide self.camera_mode) - passed
        in rather than read from self, exactly like threshold/min_match_margin
        above, so a group's camera_mode override reaches this diagnostic too.

        Flags when this track's CURRENT embedding resembles some OTHER
        actively-tracked identity on this camera substantially more than it
        resembles its OWN assigned identity - exactly the signature JeztSort
        swapping two nearby people's local track ids would leave behind.
        Returns a dict for ObserveResult.possible_id_swap, or None if nothing
        suspicious was found (the overwhelmingly common case).
        """
        if camera_mode not in ("same_camera", "same_and_cross"):
            return None

        own_matches = self.store.search(embedding, top_k=1, only_global_id=own_global_id)
        own_sim = own_matches[0][1] if own_matches else None

        active_ids = self._active_identities_on_camera(camera_id, local_track_id, now)
        active_ids.discard(own_global_id)

        if not active_ids:
            return None

        alt_matches = self.store.search(
            embedding, top_k=len(active_ids) + 2, allowed_cameras={camera_id},
        )
        alt_candidates = [c for c in alt_matches if c[0] in active_ids]

        if not alt_candidates:
            return None

        alt_global_id, alt_sim, _ = alt_candidates[0]

        if alt_sim < threshold:
            return None

        # "substantially stronger" reuses the resolved min_match_margin
        # deliberately - the same "well clear of a coin-flip" semantic the
        # MATCH/UNCERTAIN decision already uses (camera-group-resolved, same
        # as the threshold above - see _policy_for()), not a new unrequested
        # magic number. If own_sim is unavailable (identity's own gallery
        # empty - should not normally happen), any alt candidate clearing the
        # threshold alone is already noteworthy enough to flag.
        if own_sim is not None and (alt_sim - own_sim) < min_match_margin:
            return None

        return {
            "own_global_id": own_global_id,
            "own_similarity": own_sim,
            "alternative_global_id": alt_global_id,
            "alternative_similarity": alt_sim,
            "margin": (alt_sim - own_sim) if own_sim is not None else None,
        }

    # --------------------------------------------------------------- status
    def known_tracks(self, camera_id):
        """
        Every local track this manager currently holds state for on one
        camera - bootstrapping or resolved - for the display table. Sourced
        from persistent per-track state, NOT "whichever tracks happened to
        get a fresh detection match on the single most recent processed
        frame": that ephemeral snapshot can be empty even for a person who
        has been continuously present for minutes, the instant a single
        frame's detection happens to miss (this is common enough that an
        earlier version of this table went blank for exactly that reason -
        see the PoC test log for CAM-R01/qdrant: two local tracks had already
        resolved to real Qdrant points, yet the frame-snapshot table showed
        nothing).
        """
        return sorted(
            local_track_id
            for (cid, local_track_id) in self._tracks
            if cid == camera_id
        )

    def status_for(self, camera_id, local_track_id):
        """
        Last known (global_id, similarity, status) for the display table, or
        None if this track has no DECISION yet (still bootstrapping - see
        progress_for() for that case instead).

        Gated on last_status, NOT on global_id: an UNCERTAIN decision has no
        global_id (nothing was resolved) but must still be visible in the
        table rather than silently reading as "not yet observed" - that is
        the whole point of reporting UNCERTAIN honestly. A fresh retry (see
        observe()) will overwrite this the next time it resolves.
        """
        state = self._tracks.get((camera_id, local_track_id))

        if state is None or state.last_status is None:
            return None

        return {
            "global_id": state.global_id,
            "similarity": state.last_similarity,
            "status": state.last_status,
        }

    def progress_for(self, camera_id, local_track_id):
        """
        Bootstrap progress for a track that has NOT resolved yet - (collected,
        needed), or None if the track is unknown or already resolved. For a
        display that wants to show "collecting 2/5..." instead of nothing
        while a brand-new track is still being observed.
        """
        state = self._tracks.get((camera_id, local_track_id))

        if state is None or state.global_id is not None:
            return None

        return (len(state.bootstrap_embeddings), self.bootstrap_observations)

    def identity(self, global_id):
        """The full record for one global identity (for reporting/README examples)."""
        return self._identities.get(global_id)

    def identity_count(self):
        return len(self._identities)

    def track_count(self):
        """len(self._tracks) via a public accessor, mirroring identity_count()
        - lets callers (reid_adapter.py's cleanup logging, tests) observe the
        production-hardening P1 TTL fix's effect without reaching into a
        leading-underscore attribute directly."""
        return len(self._tracks)

    # ---------------------------------------------------------------- reset
    def forget_camera(self, camera_id):
        """
        Drop per-track bootstrap/refresh state for one camera - mirrors
        tracking.TrackManager.reset_camera(). Global identities themselves
        are NOT deleted (a person seen again on another camera must still
        resolve to the same global_person_id).
        """
        dropped = [k for k in self._tracks if k[0] == camera_id]

        for key in dropped:
            del self._tracks[key]

        return len(dropped)

    def evict_stale_tracks(self, now=None, ttl_seconds=None):
        """
        PRODUCTION-HARDENING P1 FIX: bound self._tracks, which otherwise
        grows without limit over long uptime - needs_observation() creates a
        _TrackState for every (camera_id, local_track_id) pair it has EVER
        seen a confirmed sighting for, and until this method existed nothing
        ever removed one except forget_camera() (a whole-camera reset, not a
        per-track lifecycle mechanism) - CONFIRMED finding, see test_
        production_hardening.py's G12 test group.

        Evicts every _TrackState whose last_seen_ts (touched on every
        needs_observation() call - see that method) is at least ttl_seconds
        old (default config.REID_TRACK_STATE_TTL_SECONDS, which itself
        defaults to exactly config.REID_MAX_TIME_GAP_SECONDS - see that
        config value's own docstring for why that specific number was
        chosen, not an arbitrary aggressive one).

        Touches ONLY self._tracks. self._identities and self.store (the
        actual global-identity/gallery data) are NEVER read or written here,
        so evicting a track can never lose identity data, never merge two
        identities, and never change a future match decision except by
        requiring a fresh bootstrap - JeztSort local_track_ids are never
        reused within a process (a globally-incrementing counter), so a
        genuinely-returning person always gets a brand-new local_track_id
        regardless of whether their old _TrackState was evicted; appearance-
        based matching against the still-intact gallery (untouched by this
        method) is what correctly recovers their identity either way.

        Safe to call for a camera that has disappeared and later reconnects,
        and independently per camera - this is a plain age check over
        whatever keys currently exist, with no camera-level bookkeeping of
        its own.

        This method itself does the whole O(len(self._tracks)) scan
        unconditionally every time it is called - deliberately simple and
        independently testable. The CALLER is responsible for not calling it
        every frame (see reid_adapter.py's flush(), which gates calls to this
        method behind config.REID_TRACK_CLEANUP_INTERVAL_SECONDS, mirroring
        its own pre-existing _maybe_print_stats() elapsed-time-gate pattern) -
        this manager has no timer/thread of its own.

        now/ttl_seconds are injectable (both default to real
        time.monotonic()/config.REID_TRACK_STATE_TTL_SECONDS) purely so tests
        can simulate long-running behaviour without an actual multi-minute
        sleep. Returns the number of tracks evicted.
        """
        if now is None:
            now = time.monotonic()
        if ttl_seconds is None:
            ttl_seconds = config.REID_TRACK_STATE_TTL_SECONDS

        stale = [
            key for key, state in self._tracks.items()
            if (now - state.last_seen_ts) >= ttl_seconds
        ]

        for key in stale:
            del self._tracks[key]

        return len(stale)
