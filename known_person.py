"""
Known-person recognition - "is this face a REGISTERED person?"

A different question from Face-ID's. Face-ID asks "which local identity is
this?" and answers with an auto-minted F-#### against cctv_face_reid_test.
This module asks "does this face belong to somebody an operator deliberately
enrolled?" and answers with a KnownPerson uid against cctv_face_known_persons.
Two collections, two questions, two thresholds. They must not be merged: a
false merge in the Face-ID gallery already cost this project a discarded
collection (see face_id_manager.py's own header), and a false *name* on a
police watchlist is worse than a false F-####.

WHY THIS IS A NEW COMPONENT AND NOT A CALL INTO AN EXISTING STORE
-----------------------------------------------------------------
Both obvious reuse paths were audited and both fail, one of them silently:

  * reid_poc/qdrant_reid.py's QdrantReIDStore binds ONE collection at
    construction, and its search() drops every point whose payload lacks a
    `global_person_id` key (qdrant_reid.py:138-141). Known-person points
    carry `known_person_id` instead, so it returns [] forever - no
    exception, no log, no counter. Worse, its __init__ calls
    _ensure_collection() unconditionally, which would be a create_collection
    attempt against a collection the dashboard's reset command lists as
    PROTECTED.
  * the dashboard's face_search.search_gallery() has the same payload-key
    problem and is a different process anyway.

qdrant_reid.py is imported UNCHANGED by design and is not ours to patch, so
this module owns its own client, its own collection constant, its own port
guard, its own timeout and its own dedupe. That is deliberate duplication of
four small things, not a missing abstraction.

ISOLATION
---------
This is an OPTIONAL side path, exactly like face_evidence.py. It is called
from face_id_manager.py's process() AFTER the embedding exists and BEFORE the
gallery search, and its whole body is wrapped by the caller. It never raises
into Face-ID, never influences the identity decision (it only READS the
embedding), and returns None for every failure - including a collection that
does not exist yet, which is the normal state of a fresh install before any
operator has enrolled anybody.

THRESHOLD IS UNCALIBRATED
-------------------------
KNOWN_PERSON_SIMILARITY_THRESHOLD is an operational starting point, NOT a
calibrated decision boundary, and it is deliberately its own constant rather
than a reuse of FACE_ID_SIMILARITY_THRESHOLD (0.45, which governs permanent
identity merges) or the dashboard's operator-supplied face-search
min_similarity. The two galleries were built under incompatible admission
rules - enrolled vectors come from operator portraits with no size gate,
live vectors from CCTV crops gated at 50x50px/0.65 against a ~45px estate
median - so a value measured on one does not transfer to the other. Until
paired portrait-vs-CCTV evidence from this estate exists, a RECOGNISED
result is reported and may drive presence monitoring, but must NOT be
presented to an operator as a confirmed identity.

    recogniser = KnownPersonRecogniser()
    result = recogniser.search(embedding)     # None, or a Recognition
    if result and result.status == "RECOGNISED":
        ...
"""

import os
import threading
import time

#: OPT-IN, unlike FACE_EVIDENCE_ENABLED which defaults on. Three reasons:
#: this is an uncalibrated biometric path (see THRESHOLD below) and a named
#: match on a police watchlist is a heavier claim than an anonymous F-####;
#: the next inference restart must not silently begin watchlist matching
#: before a controlled live validation has been authorised; and a default-on
#: switch would have the unit-test harness open a real Qdrant connection and
#: a real dashboard sink. Enabling it is a deliberate deployment step.
RECOGNITION_ENABLED = os.getenv("KNOWN_PERSON_RECOGNITION_ENABLED", "0") == "1"

#: Redeclared, not imported. The name lives in the dashboard's Django settings
#: (core/settings.py KNOWN_PERSON_QDRANT_COLLECTION) which this process cannot
#: import - separate venv, no Django. Same redeclaration convention
#: face_search.py:36-42 and face_enrollment.py already follow. If these two
#: ever disagree the watchlist silently searches the wrong collection, so
#: they are kept identical by test, not by hope.
COLLECTION = os.getenv("KNOWN_PERSON_QDRANT_COLLECTION", "cctv_face_known_persons")

HOST = os.getenv("KNOWN_PERSON_QDRANT_HOST", "127.0.0.1")
PORT = int(os.getenv("KNOWN_PERSON_QDRANT_PORT", "6343"))

#: Deliberately much tighter than Face-ID's own 2.0s. This lookup sits on the
#: single shared face worker thread, whose input queue drops observations when
#: full, so a slow watchlist must degrade into "no hit" quickly rather than
#: into dropped faces. A live local lookup measured p99 ~2ms, so 0.5s is a
#: 200x margin over normal and still bounds the bad case tightly.
TIMEOUT = float(os.getenv("KNOWN_PERSON_QDRANT_TIMEOUT", "0.5"))

#: CIRCUIT BREAKER. Measured, not speculative: with an unreachable Qdrant
#: every lookup stalls for the full timeout, and the Face-ID worker is
#: single-threaded behind a 64-deep queue that DROPS when full. Retrying
#: every observation would therefore turn a watchlist outage into lost
#: FACES - a real regression of the existing feature, which is exactly what
#: this side path is not allowed to cause.
#:
#: After this many consecutive failures the breaker opens and search()
#: returns immediately, touching no socket, until the cooldown expires. That
#: bounds the cost of an outage to FAILURE_THRESHOLD x TIMEOUT per cooldown
#: window instead of TIMEOUT per observation, forever.
FAILURE_THRESHOLD = int(os.getenv("KNOWN_PERSON_FAILURE_THRESHOLD", "3"))
BREAKER_COOLDOWN = float(os.getenv("KNOWN_PERSON_BREAKER_COOLDOWN", "60.0"))

#: UNCALIBRATED - see the module docstring. Its own constant on purpose.
SIMILARITY_THRESHOLD = float(os.getenv("KNOWN_PERSON_SIMILARITY_THRESHOLD", "0.50"))

#: Runner-up gap. Without it a two-enrolled-brothers case resolves to
#: whichever scored 0.0001 higher and reports a confident name.
MIN_MARGIN = float(os.getenv("KNOWN_PERSON_MIN_MARGIN", "0.05"))

#: The shared multi-tenant production Qdrant, which hosts an unrelated
#: product's own face_embeddings collection. Fifth redeclaration of the same
#: rail (face_id_manager.py:100, reid_adapter.py:158, face_search.py:42,
#: face_enrollment.py:34) - kept local for the same reason they are.
_BLOCKED_QDRANT_PORTS = {6333}
# ZAYED: on the Zayed GB10, 6333 IS the project's own isolated Qdrant. Same opt-in as
# face_id_manager (FACE_ID_ALLOW_SHARED_QDRANT) and reid_adapter (REID_PROD_ALLOW_SHARED_QDRANT).
ALLOW_SHARED_QDRANT = os.getenv("KNOWN_PERSON_ALLOW_SHARED_QDRANT", "0") == "1"

FAILURE_LOG_INTERVAL = 30.0

#: Status vocabulary. Kept short: the dashboard stores it in a small column
#: and the values are compared exactly.
RECOGNISED = "RECOGNISED"
AMBIGUOUS = "AMBIGUOUS"
NO_HIT = "NO_HIT"


class Recognition:
    """One watchlist answer. Carries no embedding - see SECURITY below."""

    __slots__ = ("status", "person_uid", "name", "similarity",
                 "runner_up_similarity", "margin")

    def __init__(self, status, person_uid=None, name=None, similarity=None,
                 runner_up_similarity=None, margin=None):
        self.status = status
        self.person_uid = person_uid
        self.name = name
        self.similarity = similarity
        self.runner_up_similarity = runner_up_similarity
        self.margin = margin

    def __repr__(self):
        return (f"Recognition({self.status} {self.person_uid} "
                f"sim={self.similarity})")


def decide_known_person(best, second, threshold=None, min_margin=None):
    """
    (status, similarity, runner_up, margin) from the two best candidates.

    Deliberately the same SHAPE as face_id_manager.decide_face_identity - a
    threshold plus a runner-up margin, with an explicit ambiguous outcome
    rather than a bare argmax - but with its OWN numbers. Copying the shape
    is intentional; sharing the constants is not.

    `best`/`second` are (person_uid, similarity, payload) tuples or None.
    """
    threshold = SIMILARITY_THRESHOLD if threshold is None else threshold
    min_margin = MIN_MARGIN if min_margin is None else min_margin

    if best is None:
        return NO_HIT, None, None, None

    best_sim = float(best[1])
    second_sim = float(second[1]) if second is not None else None
    margin = None if second_sim is None else best_sim - second_sim

    if best_sim < threshold:
        return NO_HIT, best_sim, second_sim, margin

    # Above the bar, but not clearly ahead of the runner-up: claim nothing.
    # An ambiguous watchlist hit is reported as ambiguous, never silently
    # resolved to the higher score.
    if margin is not None and margin < min_margin:
        return AMBIGUOUS, best_sim, second_sim, margin

    return RECOGNISED, best_sim, second_sim, margin


class KnownPersonRecogniser:
    """
    Reads cctv_face_known_persons. Never writes to it.

    The client is built LAZILY on first search, never in __init__. Face-ID's
    construction runs on the main CCTV inference loop (multicam_inf.py applies
    camera config inside the frame loop, which reaches face_id_adapter's
    factory synchronously) and a failure there latches Face-ID off for the
    life of the process. A watchlist connect must not be able to do that.
    """

    def __init__(self, collection=COLLECTION, host=HOST, port=PORT,
                 timeout=TIMEOUT, client=None):
        if port in _BLOCKED_QDRANT_PORTS and not ALLOW_SHARED_QDRANT:
            raise RuntimeError(
                f"KNOWN_PERSON_QDRANT_PORT={port} is the shared, multi-tenant "
                f"production Qdrant - it hosts an UNRELATED product's own face "
                f"collections. Refusing to read known persons from there. Use "
                f"the isolated instance (6343)."
            )

        self.collection = collection
        self.host = host
        self.port = port
        self.timeout = timeout

        self._client = client
        self._client_lock = threading.Lock()
        self._lock = threading.Lock()

        self.searches = 0
        self.recognised = 0
        self.ambiguous = 0
        self.no_hit = 0
        self.failed = 0
        self.collection_absent = 0
        self.breaker_skipped = 0
        self.breaker_opened = 0

        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

        self._last_failure_log = 0.0

    # ------------------------------------------------------------- client
    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                from qdrant_client import QdrantClient  # noqa: local - see __init__
                self._client = QdrantClient(host=self.host, port=self.port,
                                            timeout=self.timeout)
        return self._client

    # ------------------------------------------------------------- search
    def search(self, embedding, top_k=2):
        """
        Best known person for this already-normalised embedding.

        Returns a Recognition, or None if recognition could not run at all
        (disabled, collection absent, or the lookup failed). None means "no
        answer", which is NOT the same as Recognition(NO_HIT) - "asked, and
        this face is nobody enrolled".

        Never raises. The caller wraps this too, belt and braces, because a
        failure here must cost only the watchlist answer.
        """
        if not RECOGNITION_ENABLED:
            return None

        if self._breaker_is_open():
            with self._lock:
                self.breaker_skipped += 1
            return None

        # The query AND the ranking are both inside the net. Ranking parses
        # scores and payloads that came off the wire, so a malformed point is
        # every bit as likely to raise as a dead connection - and this method
        # documents that it never raises.
        try:
            points = self._query(embedding, top_k)
            best, second = self._best_two(points)
            status, sim, runner_up, margin = decide_known_person(best, second)
        except _CollectionAbsent:
            with self._lock:
                self.collection_absent += 1
            # Ordinary, not a failure: on a fresh install nobody has enrolled
            # anyone yet, so the dashboard has never created the collection.
            # Deliberately does NOT trip the breaker - the collection will
            # appear the moment somebody enrols, and this costs nothing
            # because Qdrant answers "no such collection" immediately.
            return None
        except Exception as exc:  # noqa: BLE001 - Face-ID outranks the watchlist
            self._record_failure()
            self._log_failure(f"{type(exc).__name__}: {exc}")
            return None

        with self._lock:
            self._consecutive_failures = 0      # a success closes the breaker
            self.searches += 1
            if status == RECOGNISED:
                self.recognised += 1
            elif status == AMBIGUOUS:
                self.ambiguous += 1
            else:
                self.no_hit += 1

        return Recognition(
            status=status,
            person_uid=(best[0] if best is not None else None),
            name=(best[2].get("name") if best is not None else None),
            similarity=sim,
            runner_up_similarity=runner_up,
            margin=margin,
        )

    def _query(self, embedding, top_k):
        client = self._get_client()

        # Over-fetch: one enrolled person may hold several vectors in future,
        # and dedupe-by-person below would otherwise let one person's own
        # near-duplicates crowd out the true runner-up. Same reasoning as
        # qdrant_reid.py's raw_limit, arrived at independently because that
        # store cannot be reused here.
        raw_limit = max(8, top_k * 4)

        try:
            response = client.query_points(
                collection_name=self.collection,
                query=embedding.tolist(),
                limit=raw_limit,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            if _looks_absent(exc):
                raise _CollectionAbsent(str(exc)) from exc
            raise

        return response.points

    @staticmethod
    def _best_two(points):
        """
        Best point per distinct known person, then the top two.

        The dedupe key is `known_person_id`, which is what the dashboard's
        enrollment actually writes (face_enrollment.py) and what maps to
        KnownPerson.person_uid. Points without it are skipped - the same
        defensive shape qdrant_reid.py uses for global_person_id, which is
        precisely why that store cannot read this collection.
        """
        best_per_person = {}
        for point in points:
            payload = point.payload or {}
            person_uid = payload.get("known_person_id")
            if not person_uid:
                continue
            score = float(point.score)
            current = best_per_person.get(person_uid)
            if current is None or score > current[1]:
                best_per_person[person_uid] = (person_uid, score, payload)

        ranked = sorted(best_per_person.values(), key=lambda row: row[1],
                        reverse=True)
        best = ranked[0] if len(ranked) > 0 else None
        second = ranked[1] if len(ranked) > 1 else None
        return best, second

    def is_available(self):
        """Whether a lookup would be attempted right now.

        False only while the circuit breaker is open - i.e. the watchlist has
        failed repeatedly and is being left alone. Read by the heartbeat the
        dashboard's absence sweep depends on: a heartbeat means "recognition
        is working on this camera", so it must not be sent while every search
        is being skipped.
        """
        return not self._breaker_is_open()

    # ------------------------------------------------------------- breaker
    def _breaker_is_open(self):
        with self._lock:
            if self._breaker_open_until == 0.0:
                return False
            if time.monotonic() < self._breaker_open_until:
                return True
            # Cooldown expired: close it and let ONE request through to test
            # the water. If that fails, _record_failure opens it again.
            self._breaker_open_until = 0.0
            self._consecutive_failures = 0
            return False

    def _record_failure(self):
        with self._lock:
            self.failed += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= FAILURE_THRESHOLD:
                if self._breaker_open_until == 0.0:
                    self.breaker_opened += 1
                    should_log = True
                else:
                    should_log = False
                self._breaker_open_until = time.monotonic() + BREAKER_COOLDOWN
            else:
                should_log = False

        if should_log:
            print(f"[FACE-RECOGNITION] watchlist unreachable after "
                  f"{FAILURE_THRESHOLD} attempts - pausing lookups for "
                  f"{BREAKER_COOLDOWN:.0f}s so a slow watchlist cannot cost "
                  f"dropped faces. Face-ID is unaffected.")

    # ------------------------------------------------------------- logging
    def _log_failure(self, message):
        # Throttled: this runs per observation on armed cameras, and a dead
        # Qdrant must not flood the inference log at frame rate.
        now = time.monotonic()
        with self._lock:
            if now - self._last_failure_log < FAILURE_LOG_INTERVAL:
                return
            self._last_failure_log = now
        print(f"[FACE-RECOGNITION-ERROR] watchlist lookup failed "
              f"(Face-ID unaffected): {message}")

    def stats(self):
        return {
            "searches": self.searches,
            "recognised": self.recognised,
            "ambiguous": self.ambiguous,
            "no_hit": self.no_hit,
            "failed": self.failed,
            "collection_absent": self.collection_absent,
            # breaker_skipped climbing with failed flat means the watchlist is
            # down and being deliberately left alone, not that it is healthy.
            "breaker_skipped": self.breaker_skipped,
            "breaker_opened": self.breaker_opened,
        }

    def close(self):
        """Release the client if one was ever built. Never raises."""
        client = self._client
        self._client = None
        if client is None:
            return
        try:
            client.close()
        except Exception:  # noqa: BLE001 - shutdown must not fail
            pass


class _CollectionAbsent(Exception):
    """The watchlist collection does not exist yet. Ordinary, not an error."""


def _looks_absent(exc):
    """
    Whether this exception means "collection not found".

    Matched on the message rather than an exception class because
    qdrant_client raises UnexpectedResponse for a 404 and the class is not
    part of its stable surface. A false negative here only costs a throttled
    log line and a `failed` count instead of a `collection_absent` count -
    the caller's behaviour is identical either way.
    """
    text = str(exc).lower()
    return ("not found" in text
            or "doesn't exist" in text
            or "does not exist" in text
            or "404" in text)
