"""
Face identity manager - the worker-side half of Face-ID.

face_id_adapter.FaceIDAdapter owns the thread boundary and knows nothing
about faces. This module owns everything about faces and knows nothing
about threads: it is handed one FaceObservation at a time, already off the
inference path, and answers one question - which persistent identity, if
any, does this face belong to?

    manager = FaceIDManager()                       # loads models once
    adapter = FaceIDAdapter(processor=manager)      # adapter calls .process()

    Observation
        -> detect faces in the person crop
        -> pick the largest
        -> QUALITY GATE            -> skip, or
        -> embedding (512-D, L2-normalised)
        -> Qdrant search (top 2)
        -> threshold + runner-up margin
        -> MATCH / UNCERTAIN / NEW -> F-####
        -> attributes (age, gender) recorded against that identity
        -> asynchronous gallery upsert

WHAT IS REUSED, AND FROM WHERE
------------------------------
Deliberately almost nothing here is new. The modelling decisions were
already validated by the test_face.py prototype and are reused as-is:

  * InsightFace buffalo_l (det_10g + w600k_r50 + genderage), models cached
    on disk, loaded ONCE per process.
  * reid_poc/qdrant_reid.QdrantReIDStore - imported UNCHANGED, pointed at
    its own collection. Same sys.path fix reid_adapter.py._init_reid_poc()
    and test_face.py both use.
  * reid_poc/reid_manager.AsyncUpsertStore - imported UNCHANGED, with every
    tuning parameter passed EXPLICITLY so no reid_poc/config.py value can
    silently govern Face-ID behaviour.
  * The best-vs-runner-up decision RULE from GlobalReIDManager.
    _decide_candidates(), which test_face.py already ported verbatim. The
    rule is generic; only the threshold/margin VALUES are face-specific.

What is NOT reused is GlobalReIDManager itself - it carries bootstrap-over-
N-observations, actively-held-conflict tracking and camera-topology gating,
all tuned for OSNet body embeddings. Face-ID needs none of that yet.

WHY IDENTITY MINTING IS NOT A COUNTER
-------------------------------------
The prototype used itertools.count(1), which restarts at 1 every process
while the Qdrant collection persists. Run 2's first person was therefore
minted F-0001 and upserted into run 1's F-0001 gallery - two different
people silently merged into one identity, which is the single worst outcome
this system can produce. That collection has been dropped as contaminated.

Allocation here reads the highest F-#### that actually EXISTS in the store
at the moment of allocation, under a lock, and verifies the candidate is
unused before claiming it. See _allocate_face_id() for the exact guarantee
and its one honest limit.

Env:
  FACE_ID_QDRANT_HOST / _PORT / _COLLECTION / _TIMEOUT
  FACE_ID_SIMILARITY_THRESHOLD    UNCALIBRATED (default 0.45)
  FACE_ID_MIN_MATCH_MARGIN        UNCALIBRATED (default 0.05)
  FACE_ID_GALLERY_SIZE            embeddings kept per identity (default 5)
  FACE_ID_MIN_FACE_WIDTH/_HEIGHT  quality gate, px (default 50/50)
  FACE_ID_MIN_DET_SCORE           quality gate (default 0.65)
  FACE_ID_MIN_SHARPNESS           0 = measured but NOT enforced (default 0)
  FACE_ID_DET_THRESHOLD           InsightFace detector threshold (default 0.6)
"""
import os
import re
import sys
import threading
import time
import uuid

import numpy as np

import evidence
import face_evidence
import known_person


def _bool_env(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ============================================================
# CONFIGURATION
# ============================================================

FACE_ID_QDRANT_HOST = os.getenv("FACE_ID_QDRANT_HOST", "127.0.0.1")
FACE_ID_QDRANT_PORT = int(os.getenv("FACE_ID_QDRANT_PORT", "6343"))
FACE_ID_QDRANT_COLLECTION = os.getenv("FACE_ID_QDRANT_COLLECTION", "cctv_face_reid_test")
FACE_ID_QDRANT_TIMEOUT = float(os.getenv("FACE_ID_QDRANT_TIMEOUT", "2.0"))

#: The shared, multi-tenant production Qdrant. Confirmed during the Face-ID
#: audit to already host a `face_embeddings` collection belonging to an
#: UNRELATED product, alongside `video_faces`, `corporate_memory` and
#: sixteen user_memory_* collections. Writing face vectors there would not
#: merely be untidy - it would land in another system's live data. Same
#: rail reid_adapter.py._BLOCKED_QDRANT_PORTS and test_face.py both enforce.
_BLOCKED_QDRANT_PORTS = {6333}
FACE_ID_ALLOW_SHARED_QDRANT = _bool_env("FACE_ID_ALLOW_SHARED_QDRANT", False)

#: buffalo_l's recognition head (w600k_r50) always emits 512-D. Asserted
#: against the first real embedding at runtime, so a wrong value here fails
#: loudly instead of Qdrant silently rejecting vectors later.
FACE_ID_EMBEDDING_DIM = 512
FACE_ID_QDRANT_DISTANCE = "COSINE"
FACE_ID_GALLERY_SIZE = int(os.getenv("FACE_ID_GALLERY_SIZE", "5"))

#: Identifies WHICH model produced a stored vector. Vectors from different
#: face models are not comparable, so a future model change must be able to
#: find and quarantine everything the old one wrote.
FACE_ID_MODEL_ID = os.getenv("FACE_ID_MODEL_ID", "insightface:buffalo_l:w600k_r50")

#: One representative face-crop image per identity, uploaded to SeaweedFS
#: via face_evidence.py on a NEW decision only (never on MATCH/UNCERTAIN -
#: see the NEW branch below). Independent kill switch from FACE_ID_ENABLED
#: and from evidence.ENABLED (the general event-evidence system), so
#: either can be turned off without disturbing the other.
FACE_EVIDENCE_ENABLED = os.getenv("FACE_EVIDENCE_ENABLED", "1") == "1"

#: Where RECOGNISED known-person sightings are delivered. A different
#: endpoint from the face-observation sink (/api/ai/face-observations/)
#: because it carries a different payload to a different model - see
#: _recognise_known_person() for why the two are not merged.
KNOWN_PERSON_SINK_PATH = os.getenv(
    "KNOWN_PERSON_SINK_PATH", "/api/ai/known-person-sightings/")

#: How often, per camera, to tell the dashboard that recognition is alive.
#: This is what lets the absence sweep distinguish "the person left" from
#: "the gallery stopped answering" - without it, an outage would raise a
#: false absence for every monitored person at once. Must stay comfortably
#: under the dashboard's KNOWN_PERSON_HEALTH_STALENESS (default 180s).
#: Padding around the FACE box for its evidence crop, as a fraction of the
#: face's width/height on each side. Face crops only - every other event crop
#: keeps evidence.PADDING (0.12).
#:
#: 0.12 was the body-crop value inherited by default, and on a face it cuts
#: off the forehead, hair and chin: the picture an operator uses to check a
#: named match showed less of the head than the recogniser saw. Compared on
#: 14 recovered known-person faces from CAM-R25/R26 (2026-09-17) at 0.12,
#: 0.4, 0.6 and 1.0: 0.6 is the smallest that shows the whole head with the
#: shoulders and clothing, 1.0 starts shrinking the face itself.
#:
#: Clamped to the PERSON crop the face was found in (observation.crop), so
#: the headroom above the hair is limited to what the person box holds; the
#: sides and shoulders are not.
FACE_EVIDENCE_PADDING = float(os.getenv("FACE_EVIDENCE_PADDING", "0.6"))

KNOWN_PERSON_HEARTBEAT_INTERVAL = float(
    os.getenv("KNOWN_PERSON_HEARTBEAT_INTERVAL", "30"))

#: Throttle for the recognition side-path's own error line. Same reasoning as
#: face_evidence.FAILURE_LOG_INTERVAL: a broken watchlist must not flood the
#: inference log at camera frame rate.
RECOGNITION_ERROR_LOG_INTERVAL = float(
    os.getenv("KNOWN_PERSON_ERROR_LOG_INTERVAL", "30"))

# --- decision -----------------------------------------------------------
#: UNCALIBRATED STARTING POINTS. Deliberately NOT reid_poc's validated
#: 0.75/0.08, which was tuned for OSNet BODY embeddings and transfers
#: nothing to ArcFace face embeddings. These are the prototype's own
#: experimental values and must not be described as production-calibrated
#: until real paired face observations from a suitable camera exist.
#: Raising the threshold makes the system more conservative (more
#: UNCERTAIN, fewer merges); lowering it to increase MATCH count is
#: explicitly the wrong move.
FACE_ID_SIMILARITY_THRESHOLD = float(os.getenv("FACE_ID_SIMILARITY_THRESHOLD", "0.45"))
FACE_ID_MIN_MATCH_MARGIN = float(os.getenv("FACE_ID_MIN_MATCH_MARGIN", "0.05"))

# --- quality gate -------------------------------------------------------
#: The audit measured real face sizes on this estate: median 45 px wide,
#: p25 38 px, minimum 26 px, from 400 real person crops. w600k_r50 consumes
#: 112x112 ALIGNED crops, so a 45 px face is being upsampled ~2.5x - the
#: detail simply is not there, and no threshold can recover it.
#:
#: 50 px is therefore deliberately conservative and WILL reject a large
#: share of observations on this footage. That is the intended trade: the
#: brief ranks a false identity merge as far worse than a skipped
#: observation, and a skipped face costs one sample of a person who will
#: very likely be seen again. Lower it only with paired-crop evidence, and
#: never to raise the MATCH count.
FACE_ID_MIN_FACE_WIDTH = int(os.getenv("FACE_ID_MIN_FACE_WIDTH", "50"))
FACE_ID_MIN_FACE_HEIGHT = int(os.getenv("FACE_ID_MIN_FACE_HEIGHT", "50"))

#: Detector confidence floor. Measured distribution on real crops: median
#: 0.752, minimum 0.613 at a 0.6 detector threshold.
FACE_ID_MIN_DET_SCORE = float(os.getenv("FACE_ID_MIN_DET_SCORE", "0.65"))

#: Blur/sharpness (variance of the Laplacian) is MEASURED on every face and
#: recorded in the payload, but ENFORCED only if this is set above 0.
#: Deliberate: there is no calibration data for what a "too blurry" face
#: looks like on this estate yet, and gating on an unvalidated threshold
#: would silently discard good observations. Collect first, gate later -
#: the same discipline the Re-ID phases used for their own thresholds.
FACE_ID_MIN_SHARPNESS = float(os.getenv("FACE_ID_MIN_SHARPNESS", "0"))

FACE_ID_DET_THRESHOLD = float(os.getenv("FACE_ID_DET_THRESHOLD", "0.6"))
FACE_ID_DET_SIZE = int(os.getenv("FACE_ID_DET_SIZE", "640"))

#: Age buckets reported to operators. Age is a MODEL PREDICTION, never
#: ground truth, and is reported as a range for exactly that reason - the
#: model's own error is comfortably wider than a single year.
AGE_BUCKETS = (
    (0, 12, "0-12"), (13, 17, "13-17"), (18, 24, "18-24"),
    (25, 34, "25-34"), (35, 44, "35-44"), (45, 54, "45-54"),
    (55, 64, "55-64"), (65, 200, "65+"),
)

_FACE_ID_PATTERN = re.compile(r"^F-(\d+)$")


def age_bucket(age):
    """Map a predicted age in years to its reported range label."""
    try:
        age = int(age)
    except (TypeError, ValueError):
        return None
    for low, high, label in AGE_BUCKETS:
        if low <= age <= high:
            return label
    return None


def normalise_gender(raw):
    """
    InsightFace emits 0 = female, 1 = male as a numpy int. The prototype
    stored str(gender), putting the literal "1" in the data layer - raw
    model output leaking straight through to anything that reads it later.
    Mapped here, once, at the only boundary that knows what the model means.
    """
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        text = str(raw).strip().lower()
        return text if text in ("male", "female") else None
    return {0: "female", 1: "male"}.get(value)


def load_face_app():
    """
    Load buffalo_l once. Returns (app, genderage_available).

    Module-level rather than a FaceIDManager method so the QUALITY_ONLY
    harness can obtain a face detector WITHOUT constructing a manager - and
    therefore without opening any Qdrant connection at all. That is what
    makes "QUALITY_ONLY never touches Qdrant" a structural property rather
    than a promise.

    Age/gender is loaded as a separate concern from identity: if the
    genderage head fails, this falls back to detection + recognition and
    reports it, so Face-ID keeps working without attributes.
    """
    from insightface.app import FaceAnalysis   # never imported at module load

    try:
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            allowed_modules=["detection", "recognition", "genderage"],
        )
        app.prepare(ctx_id=0, det_size=(FACE_ID_DET_SIZE, FACE_ID_DET_SIZE),
                    det_thresh=FACE_ID_DET_THRESHOLD)
        _require_gpu(app)
        return app, True
    except Exception as exc:  # noqa: BLE001
        print(f"[FACE-ID-AGE] genderage unavailable ({type(exc).__name__}: {exc}) "
              f"- retrying without it; identity will still work, attributes will not")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=0, det_size=(FACE_ID_DET_SIZE, FACE_ID_DET_SIZE),
                det_thresh=FACE_ID_DET_THRESHOLD)
    _require_gpu(app)
    return app, False


#: ZAYED: refuse a silent CPU fallback. ONNX Runtime drops to CPUExecutionProvider without an
#: error when CUDA cannot be initialised; at classroom load that would quietly starve Face-ID.
FACE_ID_REQUIRE_GPU = _bool_env("FACE_ID_REQUIRE_GPU", True)


def _require_gpu(app):
    if not FACE_ID_REQUIRE_GPU:
        return
    for task, model in app.models.items():
        providers = model.session.get_providers()
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"Face-ID {task} model is on {providers} - refusing CPU fallback "
                               f"(set FACE_ID_REQUIRE_GPU=0 only for offline tests)")


def decide_face_identity(best, second, threshold, min_match_margin):
    """
    The best-vs-runner-up rule, unchanged from reid_poc/reid_manager.py's
    GlobalReIDManager._decide_candidates() (and from test_face.py, which
    already ported it verbatim). Copied rather than reimplemented: the rule
    is small, self-contained and already proven as a MECHANISM. Only the
    threshold/margin VALUES differ for face embeddings.

    best/second are (face_id, similarity, payload) tuples or None. Returns
    (decision, best_similarity, second_similarity, margin).

    A candidate that clears the threshold but cannot be separated from its
    runner-up returns UNCERTAIN, never MATCH. That is the whole point: an
    ambiguous face is exactly the case where a wrong merge happens, and an
    UNCERTAIN costs nothing but a retry.
    """
    best_sim = best[1] if best is not None else None
    second_sim = second[1] if second is not None else None
    margin = (best_sim - second_sim) if second_sim is not None else None

    if best is not None and best_sim >= threshold and (second is None or margin >= min_match_margin):
        return "MATCH", best_sim, second_sim, margin

    if best is not None and best_sim >= threshold:
        return "UNCERTAIN", best_sim, second_sim, margin

    return "NEW", best_sim, second_sim, margin


# ============================================================
# IDENTITY PROFILE
# ============================================================

class FaceIdentity:
    """
    The persistent profile behind one F-####.

    Attributes are AGGREGATED across observations rather than overwritten,
    so one bad frame cannot rewrite a person's demographics. The aggregation
    is deliberately the simplest thing that is explainable to an operator:
    the modal age bucket and the majority gender, each with the fraction of
    observations that agreed as its confidence.
    """

    __slots__ = ("face_id", "created_at", "last_seen_at", "gallery_count",
                 "observation_count", "_age_buckets", "_genders",
                 "last_age_update", "last_gender_update", "cameras")

    def __init__(self, face_id, created_at=None):
        self.face_id = face_id
        self.created_at = time.time() if created_at is None else created_at
        self.last_seen_at = self.created_at
        #: Points actually held in the store for this identity. Capped by
        #: the caller at gallery_size, because QdrantReIDStore._trim_gallery()
        #: deletes the excess - an uncapped counter here would claim a gallery
        #: far larger than the one that really exists.
        self.gallery_count = 0
        #: Every quality-passing observation ever attributed to this identity,
        #: uncapped. Diagnostic only.
        self.observation_count = 0
        self._age_buckets = []
        self._genders = []
        self.last_age_update = None
        self.last_gender_update = None
        self.cameras = set()

    def record_age(self, age):
        bucket = age_bucket(age)
        if bucket is None:
            return False
        self._age_buckets.append(bucket)
        self.last_age_update = time.time()
        return True

    def record_gender(self, raw):
        gender = normalise_gender(raw)
        if gender is None:
            return False
        self._genders.append(gender)
        self.last_gender_update = time.time()
        return True

    @staticmethod
    def _majority(values):
        if not values:
            return None, None, 0
        counts = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        winner = max(counts, key=lambda k: (counts[k], k))
        return winner, counts[winner] / len(values), len(values)

    @property
    def age_range(self):
        return self._majority(self._age_buckets)[0]

    @property
    def age_confidence(self):
        return self._majority(self._age_buckets)[1]

    @property
    def age_observation_count(self):
        return len(self._age_buckets)

    @property
    def gender(self):
        return self._majority(self._genders)[0]

    @property
    def gender_confidence(self):
        return self._majority(self._genders)[1]

    @property
    def gender_observation_count(self):
        return len(self._genders)

    def as_dict(self):
        return {
            "face_id": self.face_id,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
            "age_range": self.age_range,
            "age_confidence": self.age_confidence,
            "age_observation_count": self.age_observation_count,
            "last_age_update": self.last_age_update,
            "gender": self.gender,
            "gender_confidence": self.gender_confidence,
            "gender_observation_count": self.gender_observation_count,
            "last_gender_update": self.last_gender_update,
            "gallery_count": self.gallery_count,
            "observation_count": self.observation_count,
            "cameras": sorted(self.cameras),
        }

    def __repr__(self):
        return (f"<FaceIdentity {self.face_id} age={self.age_range} "
                f"gender={self.gender} gallery={self.gallery_count}>")


# ============================================================
# MANAGER
# ============================================================

class FaceIDManager:
    """
    The processor the adapter's worker calls. One instance per process;
    models load once in __init__.

    Both heavy collaborators are INJECTABLE (face_app, store). Production
    passes neither and gets the real InsightFace + Qdrant pair; tests pass
    fakes and exercise every decision path with no GPU and no network. That
    is what lets the quality gate, the decision rule, the aggregation and
    the identity allocator all be tested for real rather than by inspection.
    """

    def __init__(self, face_app=None, store=None, raw_store=None,
                 threshold=None, min_match_margin=None, gallery_size=None,
                 collection=None, host=None, port=None, timeout=None,
                 load_existing=True, result_sink=None, recognition_sink=None):
        self.threshold = FACE_ID_SIMILARITY_THRESHOLD if threshold is None else threshold
        self.min_match_margin = (
            FACE_ID_MIN_MATCH_MARGIN if min_match_margin is None else min_match_margin
        )
        self.gallery_size = FACE_ID_GALLERY_SIZE if gallery_size is None else gallery_size

        self.identities = {}
        self._closed = False
        self._lock = threading.Lock()
        self._genderage_available = True
        self._embedding_dim_checked = False

        self._counters = {
            "processed": 0,
            "no_face": 0,
            "skipped_quality": 0,
            "embedding_errors": 0,
            "age_errors": 0,
            "gender_errors": 0,
            "qdrant_searches": 0,
            "qdrant_failures": 0,
            "qdrant_upserts": 0,
            "matches": 0,
            "new_identities": 0,
            "uncertain": 0,
            "id_allocation_collisions": 0,

            # Evidence capture is an optional side channel, so it gets its
            # own counters rather than sharing the Face-ID ones: an operator
            # seeing evidence_prepare_failed climbing while new_identities
            # keeps climbing too knows Face-ID is healthy and only the image
            # capture is broken. The upload-side counters live on the
            # pipeline itself (face_evidence.stats()) - preparation happens
            # here, uploading happens there, and neither borrows the other's
            # numbers.
            "evidence_prepared": 0,
            "evidence_prepare_failed": 0,
            "evidence_queue_dropped": 0,

            # Known-person recognition, same reasoning as the evidence
            # counters: an operator seeing recognition_failed climb while
            # matches/new_identities keep climbing knows Face-ID is healthy
            # and only the watchlist is broken. The per-status counts live on
            # the recogniser itself (known_person.stats()); these are the
            # manager's own view of the hook.
            "recognition_attempted": 0,
            "recognition_recognised": 0,
            "recognition_failed": 0,
        }
        self._skip_reasons = {}

        #: Optional callable taking one payload dict, invoked after a
        #: decision is final. Injected so this class keeps knowing nothing
        #: about HTTP, the dashboard, or persistence beyond its own gallery -
        #: production passes face_id_sink.DashboardFaceSink, the live harness
        #: and every test pass nothing. A sink that raises is contained here:
        #: reporting an identity must never undo one.
        self._result_sink = result_sink

        self._face_app = face_app
        self._store = store
        self._raw_store = raw_store if raw_store is not None else store
        self._async_store = None

        # Async, own thread - never on this class's own critical path (see
        # the NEW branch in process()). None when disabled, so the capture
        # call sites can just check `if self._face_evidence is not None`.
        self._face_evidence = None
        if FACE_EVIDENCE_ENABLED:
            self._face_evidence = face_evidence.FaceEvidencePipeline()
            self._face_evidence.start()

        # Known-person recognition - the second optional side path. Built
        # here but it connects LAZILY on first use: this constructor runs on
        # the main CCTV inference loop, and face_id_adapter latches Face-ID
        # off permanently if the factory raises, so a watchlist connection
        # must never be attempted at construction time. Wrapped anyway,
        # because a bad KNOWN_PERSON_QDRANT_PORT would otherwise take Face-ID
        # down with it.
        self._recogniser = None
        self._recognition_sink = recognition_sink
        #: camera_id -> monotonic time of its last heartbeat. Only ever
        #: touched from the single face worker thread, so no lock is needed.
        self._heartbeat_sent = {}
        self._recognition_error_logged = 0.0
        if known_person.RECOGNITION_ENABLED:
            try:
                self._recogniser = known_person.KnownPersonRecogniser()
                if self._recognition_sink is None:
                    # Its own bounded queue and its own thread, deliberately
                    # NOT the face-observation sink: a watchlist backlog must
                    # not delay identity reporting, and the two payloads go
                    # to different endpoints and different models. Same
                    # class, different path - see DashboardFaceSink's
                    # `path`/`batch_key` parameters.
                    from face_id_sink import DashboardFaceSink  # noqa: local
                    self._recognition_sink = DashboardFaceSink(
                        path=KNOWN_PERSON_SINK_PATH,
                        name="known-person-sink",
                        batch_key="sightings",
                        # Heartbeats ride the same queue and the same POST,
                        # under their own key. One producer, one connection.
                        alt_batch_key="heartbeats",
                        alt_marker="heartbeat",
                    )
            except Exception as exc:  # noqa: BLE001 - Face-ID outranks the watchlist
                self._recogniser = None
                self._recognition_sink = None
                print(f"[FACE-RECOGNITION] disabled - could not initialise "
                      f"({type(exc).__name__}: {exc}); Face-ID continues")

        if self._face_app is None:
            self._face_app = self._load_face_app()

        if self._store is None:
            self._store, self._raw_store, self._async_store = self._build_store(
                collection=collection, host=host, port=port, timeout=timeout,
            )

        if load_existing:
            self._load_existing_identities()

    # ------------------------------------------------------------- loading
    def _load_face_app(self):
        app, genderage = load_face_app()
        self._genderage_available = genderage
        return app

    @staticmethod
    def _init_reid_poc_path():
        """reid_poc/ must win over this directory so `import config` inside
        qdrant_reid.py resolves to reid_poc/config.py. Mirrors
        reid_adapter._init_reid_poc() and test_face.py exactly."""
        here = os.path.dirname(os.path.abspath(__file__))
        reid_poc_dir = os.path.join(here, "reid_poc")
        for path in (here, reid_poc_dir):
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)

    def _build_store(self, collection=None, host=None, port=None, timeout=None):
        """
        Real QdrantReIDStore, wrapped in AsyncUpsertStore.

        Both imported UNCHANGED from reid_poc. Every AsyncUpsertStore tuning
        value is passed explicitly so that reid_poc/config.py's own Re-ID
        numbers cannot silently govern Face-ID's persistence behaviour.

        Why async at all, when this already runs off the inference thread:
        the worker is single-threaded, so a slow upsert stalls the NEXT
        face's search and grows the adapter's queue until observations get
        dropped. Deferring the write keeps the worker searching. The
        DECISION path (search) stays fully synchronous - a decision must
        never be made from a stale gallery.
        """
        port = FACE_ID_QDRANT_PORT if port is None else port

        if port in _BLOCKED_QDRANT_PORTS and not FACE_ID_ALLOW_SHARED_QDRANT:
            raise RuntimeError(
                f"FACE_ID_QDRANT_PORT={port} is the shared, multi-tenant production "
                f"Qdrant - it already hosts an UNRELATED product's face_embeddings "
                f"collection. Refusing to write face vectors there. Use the isolated "
                f"instance (6343), or set FACE_ID_ALLOW_SHARED_QDRANT=1 deliberately."
            )

        self._init_reid_poc_path()
        from qdrant_reid import QdrantReIDStore     # reid_poc/qdrant_reid.py, unchanged
        from reid_manager import AsyncUpsertStore   # reid_poc/reid_manager.py, unchanged

        raw = QdrantReIDStore(
            collection_name=FACE_ID_QDRANT_COLLECTION if collection is None else collection,
            vector_dim=FACE_ID_EMBEDDING_DIM,
            distance=FACE_ID_QDRANT_DISTANCE,
            gallery_size=self.gallery_size,
            host=FACE_ID_QDRANT_HOST if host is None else host,
            port=port,
            timeout=FACE_ID_QDRANT_TIMEOUT if timeout is None else timeout,
        )

        async_store = AsyncUpsertStore(
            raw,
            queue_size=int(os.getenv("FACE_ID_UPSERT_QUEUE_SIZE", "128")),
            max_retries=int(os.getenv("FACE_ID_UPSERT_MAX_RETRIES", "2")),
            retry_backoff_seconds=float(os.getenv("FACE_ID_UPSERT_RETRY_BACKOFF", "0.25")),
            shutdown_flush_seconds=float(os.getenv("FACE_ID_UPSERT_FLUSH_SECONDS", "5.0")),
        )

        print(f"[FACE-ID-QDRANT] {FACE_ID_QDRANT_HOST if host is None else host}:{port} "
              f"collection={FACE_ID_QDRANT_COLLECTION if collection is None else collection} "
              f"gallery={self.gallery_size} async_upsert=on")
        return async_store, raw, async_store

    # ---------------------------------------------------------- identities
    def _scroll_all_payloads(self):
        """Every stored point's payload. Uses the raw store's client
        directly - qdrant_reid.py exposes no list-identities call and is not
        to be modified."""
        payloads = []
        client = getattr(self._raw_store, "client", None)
        collection = getattr(self._raw_store, "collection_name", None)
        if client is None or collection is None:
            return payloads

        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection, limit=256,
                with_payload=True, with_vectors=False, offset=offset,
            )
            payloads.extend(p.payload or {} for p in points)
            if offset is None:
                break
        return payloads

    def _load_existing_identities(self):
        """
        Rebuild identity profiles from what is already stored.

        Without this, a restart would keep the vectors (so matching still
        works) but lose every age/gender aggregate, and the first new
        observation of an existing person would start their demographics
        from scratch. Attributes live in the point payloads precisely so
        they can be recovered here.
        """
        try:
            payloads = self._scroll_all_payloads()
        except Exception as exc:  # noqa: BLE001
            self._counters["qdrant_failures"] += 1
            print(f"[FACE-ID-QDRANT] could not load existing identities "
                  f"({type(exc).__name__}: {exc}) - starting with an empty profile set; "
                  f"stored vectors are unaffected")
            return

        for payload in payloads:
            face_id = payload.get("global_person_id")
            if not face_id or not _FACE_ID_PATTERN.match(str(face_id)):
                continue

            identity = self.identities.get(face_id)
            if identity is None:
                identity = FaceIdentity(face_id, created_at=payload.get("timestamp"))
                self.identities[face_id] = identity

            identity.gallery_count += 1
            identity.observation_count += 1
            if payload.get("camera_id"):
                identity.cameras.add(payload["camera_id"])
            if payload.get("timestamp"):
                identity.last_seen_at = max(identity.last_seen_at or 0,
                                            payload["timestamp"])
            if payload.get("age") is not None:
                identity.record_age(payload["age"])
            if payload.get("gender") is not None:
                identity.record_gender(payload["gender"])

        if self.identities:
            print(f"[FACE-ID] recovered {len(self.identities)} existing identities "
                  f"from the store: {', '.join(sorted(self.identities)[:8])}"
                  f"{' ...' if len(self.identities) > 8 else ''}")

    def _allocate_face_id(self):
        """
        Mint the next F-#### without ever reusing one.

        Reads the highest F-#### that EXISTS IN THE STORE right now - not a
        process counter, not a value cached at startup - takes the next one
        above it, and verifies nothing already holds it. Under a lock, so
        two worker threads in this process can never race.

        Guarantee, stated honestly:
          * Within one process: exact. The lock serialises read-max-then-
            claim, and the local identity map is consulted too, so an id
            allocated moments ago but whose async upsert has not landed yet
            is still counted.
          * Across processes: there is a millisecond-scale window between
            reading the maximum and the claiming upsert. Only one Face-ID
            producer runs in this PoC, so the window is not reachable
            today. It is detected rather than ignored - see
            id_allocation_collisions - and the real fix, if a second
            producer is ever introduced, is a single allocator (or a
            dashboard-side database sequence), not a wider lock here.
        """
        with self._lock:
            highest = 0

            for face_id in self.identities:
                match = _FACE_ID_PATTERN.match(str(face_id))
                if match:
                    highest = max(highest, int(match.group(1)))

            try:
                for payload in self._scroll_all_payloads():
                    match = _FACE_ID_PATTERN.match(str(payload.get("global_person_id") or ""))
                    if match:
                        highest = max(highest, int(match.group(1)))
            except Exception as exc:  # noqa: BLE001
                # A store read failure must not fall back to a low id - that
                # is exactly how the prototype's collision happened. Refuse
                # to allocate instead; the observation is skipped and the
                # person gets an identity on a later, healthier attempt.
                self._counters["qdrant_failures"] += 1
                print(f"[FACE-ID] refusing to mint an id while the store is unreadable "
                      f"({type(exc).__name__}: {exc}) - a guessed id could merge two people")
                return None

            candidate = f"F-{highest + 1:04d}"

            if candidate in self.identities:
                self._counters["id_allocation_collisions"] += 1
                print(f"[FACE-ID] allocation collision on {candidate} - retrying")
                candidate = f"F-{highest + 2:04d}"

            self.identities[candidate] = FaceIdentity(candidate)
            return candidate

    # -------------------------------------------------------- quality gate
    @staticmethod
    def _sharpness(image):
        """Variance of the Laplacian - the standard cheap blur proxy.
        Measured always, enforced only when FACE_ID_MIN_SHARPNESS > 0."""
        try:
            import cv2
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            return float(cv2.Laplacian(grey, cv2.CV_64F).var())
        except Exception:  # noqa: BLE001 - a metric must never break a decision
            return None

    @staticmethod
    def quality_check(face, crop=None, min_width=None, min_height=None,
                      min_det_score=None, min_sharpness=None):
        """
        Decide whether this face is good enough to make an IDENTITY claim
        from. Returns (ok, reason, metrics).

        A staticmethod with overridable limits so the QUALITY_ONLY harness
        can measure the SAME gate this manager enforces, and can sweep
        alternative limits over already-recorded faces, without constructing
        a manager (and therefore without any Qdrant connection). Passing
        None for a limit uses the module default, which is what the manager
        itself always does.

        A face that fails here is SKIPPED - not matched, and emphatically
        not minted as a new person. Minting from an unusable crop is how a
        gallery fills with junk that later matches everybody.
        """
        x1, y1, x2, y2 = (float(v) for v in face.bbox)
        width, height = x2 - x1, y2 - y1
        score = float(getattr(face, "det_score", 0.0) or 0.0)

        face_crop = None
        if crop is not None:
            fx1, fy1 = max(0, int(x1)), max(0, int(y1))
            fx2, fy2 = min(crop.shape[1], int(x2)), min(crop.shape[0], int(y2))
            if fx2 > fx1 and fy2 > fy1:
                face_crop = crop[fy1:fy2, fx1:fx2]

        min_width = FACE_ID_MIN_FACE_WIDTH if min_width is None else min_width
        min_height = FACE_ID_MIN_FACE_HEIGHT if min_height is None else min_height
        min_det_score = FACE_ID_MIN_DET_SCORE if min_det_score is None else min_det_score
        min_sharpness = FACE_ID_MIN_SHARPNESS if min_sharpness is None else min_sharpness

        metrics = {
            "face_width": int(width),
            "face_height": int(height),
            "face_area": int(width * height),
            "det_score": round(score, 4),
            "sharpness": (FaceIDManager._sharpness(face_crop)
                          if face_crop is not None else None),
        }

        if width < min_width:
            return False, f"face_too_narrow({int(width)}px<{min_width})", metrics
        if height < min_height:
            return False, f"face_too_short({int(height)}px<{min_height})", metrics
        if score < min_det_score:
            return False, f"low_det_score({score:.2f}<{min_det_score})", metrics
        if (min_sharpness > 0 and metrics["sharpness"] is not None
                and metrics["sharpness"] < min_sharpness):
            return False, f"blurry({metrics['sharpness']:.0f}<{min_sharpness})", metrics

        return True, None, metrics

    @staticmethod
    def select_best_face(faces):
        """Largest face in the person crop. A person crop should contain one
        face; when a bystander's face intrudes at the edge, the tracked
        subject is overwhelmingly the larger of the two."""
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # ------------------------------------------------------------- process
    def process(self, observation):
        """
        The adapter's worker calls this, one observation at a time, already
        off the inference thread. Returns a small result dict (useful to a
        harness; the adapter ignores it).

        Exceptions are allowed to propagate to the adapter's worker, which
        contains them per observation and keeps going - one place for that
        policy, not two.
        """
        if self._closed:
            # Refused, not silently dropped. Once close() has stopped the
            # async writer, an upsert would queue against a dead thread and
            # never be written - an identity decision that looks successful
            # and persists nothing is worse than no decision at all.
            return {"decision": "CLOSED", "face_id": None}

        self._counters["processed"] += 1

        faces = self._face_app.get(observation.crop)
        face = self.select_best_face(faces)

        if face is None:
            self._counters["no_face"] += 1
            return {"decision": "NO_FACE", "face_id": None}

        ok, reason, metrics = self.quality_check(face, observation.crop)
        if not ok:
            self._counters["skipped_quality"] += 1
            self._skip_reasons[reason.split("(")[0]] = (
                self._skip_reasons.get(reason.split("(")[0], 0) + 1
            )
            return {"decision": "SKIPPED_QUALITY", "face_id": None,
                    "reason": reason, "quality": metrics}

        try:
            embedding = np.asarray(face.embedding, dtype=np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        except Exception as exc:  # noqa: BLE001
            self._counters["embedding_errors"] += 1
            print(f"[FACE-ID] embedding failed ({type(exc).__name__}: {exc}) - observation skipped")
            return {"decision": "EMBEDDING_ERROR", "face_id": None}

        if not self._embedding_dim_checked:
            if embedding.shape[0] != FACE_ID_EMBEDDING_DIM:
                raise RuntimeError(
                    f"face embedding dim {embedding.shape[0]} != FACE_ID_EMBEDDING_DIM "
                    f"{FACE_ID_EMBEDDING_DIM} - fix the constant before continuing; "
                    f"Qdrant would otherwise reject or mismatch every vector"
                )
            self._embedding_dim_checked = True

        # ---- known-person recognition (optional side path) ----
        # Placed HERE, not after the identity decision, for two reasons the
        # audit made concrete:
        #
        #   1. Coverage. Every branch below this point can return early -
        #      UNCERTAIN returns without emitting at all - and UNCERTAIN is
        #      exactly the population a watchlist is most useful for: a
        #      good-quality face the local gallery refuses to attribute. A
        #      hook after the decision silently never runs for them.
        #   2. Failure containment. Between the gallery upsert and _emit()
        #      is the one window where an exception splits persistence from
        #      reporting - the vector lands while the dashboard hears
        #      nothing. Before the search, a failure costs only the
        #      watchlist answer.
        #
        # It reads the embedding and NOTHING else, so it cannot influence the
        # identity decision made below - the same structural guarantee the
        # attribute read already relies on.
        self._recognise_known_person(observation, embedding, face)

        # ---- search (synchronous - the decision depends on it) ----
        try:
            self._counters["qdrant_searches"] += 1
            candidates = self._store.search(embedding, top_k=2)
        except Exception as exc:  # noqa: BLE001
            self._counters["qdrant_failures"] += 1
            print(f"[FACE-ID-QDRANT] search failed ({type(exc).__name__}: {exc}) "
                  f"- no identity claimed for this observation")
            return {"decision": "SEARCH_FAILED", "face_id": None}

        best = candidates[0] if len(candidates) > 0 else None
        second = candidates[1] if len(candidates) > 1 else None
        decision, best_sim, second_sim, margin = decide_face_identity(
            best, second, self.threshold, self.min_match_margin,
        )

        evidence_key = None

        if decision == "MATCH":
            face_id = best[0]
            self._counters["matches"] += 1
        elif decision == "NEW":
            face_id = self._allocate_face_id()
            if face_id is None:
                return {"decision": "ALLOCATION_FAILED", "face_id": None}
            self._counters["new_identities"] += 1

            # One representative face image per identity, captured HERE and
            # only here - never on a MATCH, so an established identity does
            # not accumulate near-duplicate crops over its lifetime.
            #
            # Isolated inside _capture_face_evidence() because everything
            # BELOW this line - the Qdrant upsert and _emit() - is what
            # actually makes the identity real, and the id has ALREADY been
            # allocated and registered by this point. An exception escaping
            # here would leave a phantom: an id reserved in self.identities
            # with no vector and no dashboard row, so the same person mints
            # a second id next time they are seen. The image is optional;
            # the identity is not.
            evidence_key = self._capture_face_evidence(observation, face, face_id)
        else:
            # UNCERTAIN: claim nothing, store nothing. The observation is
            # deliberately NOT written to any gallery - writing an
            # ambiguous face into a candidate's gallery is precisely how a
            # false merge becomes permanent.
            self._counters["uncertain"] += 1
            return {"decision": "UNCERTAIN", "face_id": None,
                    "best_candidate": best[0] if best else None,
                    "best_similarity": best_sim, "second_similarity": second_sim,
                    "margin": margin, "quality": metrics}

        # ---- attributes: recorded AFTER the identity decision, never before ----
        # Structurally guaranteed not to influence matching: the decision
        # above is already final by the time any age or gender value is read.
        age_value, gender_value = self._read_attributes(face)

        identity = self.identities.setdefault(face_id, FaceIdentity(face_id))
        identity.last_seen_at = observation.timestamp or time.time()
        identity.cameras.add(observation.camera_id)
        identity.observation_count += 1
        identity.gallery_count = min(identity.gallery_count + 1, self.gallery_size)
        if age_value is not None:
            identity.record_age(age_value)
        if gender_value is not None:
            identity.record_gender(gender_value)

        payload = {
            "camera_id": observation.camera_id,
            "track_id": observation.local_track_id,
            "timestamp": observation.timestamp,
            "model": FACE_ID_MODEL_ID,
            "face_width": metrics["face_width"],
            "face_height": metrics["face_height"],
            "det_score": metrics["det_score"],
            "sharpness": metrics["sharpness"],
            "age": age_value,
            "gender": normalise_gender(gender_value),
        }

        try:
            self._store.upsert(face_id, embedding, payload)
            self._counters["qdrant_upserts"] += 1
        except Exception as exc:  # noqa: BLE001
            self._counters["qdrant_failures"] += 1
            print(f"[FACE-ID-QDRANT] upsert failed ({type(exc).__name__}: {exc}) "
                  f"- {face_id} keeps its existing gallery")

        print(f"[FACE-ID] {observation.camera_id} track={observation.local_track_id} "
              f"{decision} face_id={face_id} "
              f"sim={'-' if best_sim is None else f'{best_sim:.3f}'} "
              f"margin={'-' if margin is None else f'{margin:.3f}'} "
              f"face={metrics['face_width']}x{metrics['face_height']}px")

        self._emit(observation, face, face_id, identity, decision,
                   best_sim, second_sim, margin, metrics,
                   age_value, gender_value, evidence_key)

        return {"decision": decision, "face_id": face_id,
                "best_similarity": best_sim, "second_similarity": second_sim,
                "margin": margin, "quality": metrics,
                "age": age_value, "gender": normalise_gender(gender_value)}

    def _recognise_known_person(self, observation, embedding, face=None):
        """
        Ask the watchlist whether this face belongs to an enrolled person,
        and report the answer. Returns the Recognition, or None.

        Wrapped whole, for the same reason _capture_face_evidence is: this is
        an OPTIONAL side path and Face-ID must not notice when it breaks. The
        recogniser already contains its own failures and returns None, so this
        second net catches only the unexpected - a malformed embedding, a
        sink that raises, an import that fails. Either way the identity
        decision below runs untouched.

        The return value is deliberately NOT threaded into _emit(). Three
        reasons, all from the audit:

          * _emit() never runs for UNCERTAIN, which is the population this
            hook exists to cover, so the observation payload cannot carry
            every answer this method produces.
          * FaceObservationIngestAPI._store() builds its row from a fixed
            kwarg list and silently ignores unknown keys - a new key there
            would return HTTP 200, increment `stored`, and persist nothing.
          * FaceObservation.face_identity is a non-nullable FK, so a sighting
            that minted no F-#### has nowhere to live on that model.

        So sightings travel their own channel to their own model
        (KnownPersonSighting) and the Face-ID observation contract is left
        byte-for-byte unchanged.
        """
        if self._recogniser is None:
            return None

        try:
            self._counters["recognition_attempted"] += 1
            recognition = self._recogniser.search(embedding)

            if recognition is None:
                # Disabled, collection absent, or the lookup failed - the
                # recogniser has already counted and logged it.
                return None

            if recognition.status == known_person.RECOGNISED:
                self._counters["recognition_recognised"] += 1

            self._report_known_person(observation, recognition, face)

            # A completed lookup - hit or not - is the only evidence the
            # dashboard has that recognition was actually WORKING on this
            # camera. Without it the absence sweep cannot tell a person who
            # left from a gallery that stopped answering, and would raise a
            # false absence for every monitored person during an outage.
            # Throttled per camera; a failed lookup deliberately sends
            # nothing, because silence is precisely the signal.
            self._report_recognition_heartbeat(observation)
            return recognition

        except Exception as exc:  # noqa: BLE001 - the identity outranks the watchlist
            self._counters["recognition_failed"] += 1
            # Throttled: this runs per observation on every armed camera, so
            # an unexpected raise must not flood the inference log at frame
            # rate. The counter above stays exact regardless - stats() is the
            # source of truth for how many failed, this line is for context.
            # No embedding, no crop and no vector in it: a watchlist failure
            # has to be diagnosable without putting biometric data in a log.
            now = time.monotonic()
            if now - self._recognition_error_logged >= RECOGNITION_ERROR_LOG_INTERVAL:
                self._recognition_error_logged = now
                print(f"[FACE-RECOGNITION-ERROR] recognition skipped "
                      f"({type(exc).__name__}: {exc}) - Face-ID continues")
            return None

    def _report_recognition_heartbeat(self, observation):
        """Heartbeat for a camera that just produced a face lookup."""
        self.heartbeat(observation.camera_id, observation.timestamp)

    def heartbeat(self, camera_id, evaluated_at=None):
        """
        Tell the dashboard recognition is alive on this camera.

        WHY THIS IS NOT DRIVEN BY FACES. The dashboard's absence sweep will
        not raise an absence alert for a camera whose recognition cannot be
        shown to be working (investigation/presence.py,
        recognition_is_healthy). While the heartbeat was sent only from a
        completed face lookup, the one case absence exists for - the room is
        now EMPTY - produced no faces, so no heartbeat, so the alert was held
        for ever. The pipeline being healthy and nobody being in front of the
        camera are different things, and only the first is what this reports.

        So the adapter now calls this once per armed camera per interval for
        as long as that camera's frames are reaching the face pipeline, and a
        face lookup still refreshes it. The one thing that stops it is the
        watchlist itself being unreachable (breaker open) - then no heartbeat
        is sent, the sweep holds the alert, and silence is again the signal.

        Throttled to one per camera per KNOWN_PERSON_HEARTBEAT_INTERVAL. Safe
        without a lock: both callers run on the single face worker thread.
        """
        sink = self._recognition_sink
        if sink is None:
            return False

        recogniser = self._recogniser
        if recogniser is None:
            return False
        try:
            if not recogniser.is_available():
                return False
        except Exception:  # noqa: BLE001 - a stub recogniser in a harness
            pass

        now = time.monotonic()
        last = self._heartbeat_sent.get(camera_id, 0.0)

        if now - last < KNOWN_PERSON_HEARTBEAT_INTERVAL:
            return False

        self._heartbeat_sent[camera_id] = now
        sink.record({
            "heartbeat": True,
            "camera_id": camera_id,
            "evaluated_at": evaluated_at or time.time(),
        })
        return True

    def _report_known_person(self, observation, recognition, face=None):
        """
        Hand one sighting to the recognition sink, if one is attached.

        Only RECOGNISED sightings are reported. AMBIGUOUS and NO_HIT are
        counted (see known_person.stats()) but not sent: presence monitoring
        must never be advanced by a match the recogniser itself declined to
        commit to, and a NO_HIT carries no person to attribute.

        THE PICTURE. A sighting used to travel as text alone, so the Known
        Person Detected event on the dashboard had no image and no box - an
        operator was told "Ahmed was recognised on CAM-R26" and shown
        nothing, on a claim made by an UNCALIBRATED threshold. The face that
        produced the match is the one thing that lets a person check it, so
        the same crop-and-upload the F-#### identities already use is run
        here, and the key and the frame-coordinate box travel with the
        sighting. Failure costs the image only, never the sighting.
        """
        sink = self._recognition_sink
        if sink is None or recognition.status != known_person.RECOGNISED:
            return

        payload = {
            "camera_id": observation.camera_id,
            "track_id": observation.local_track_id,
            "timestamp": observation.timestamp,
            "known_person_uid": recognition.person_uid,
            "known_person_name": recognition.name,
            "recognition_similarity": recognition.similarity,
            "recognition_runner_up": recognition.runner_up_similarity,
            "recognition_margin": recognition.margin,
            "recognition_status": recognition.status,
            "model": FACE_ID_MODEL_ID,
            "embedding_collection": known_person.COLLECTION,
        }

        if face is not None:
            payload.update(self._sighting_evidence(observation, face,
                                                   recognition.person_uid))

        sink.record(payload)

    def _sighting_evidence(self, observation, face, person_uid):
        """
        The face crop behind one known-person sighting, as payload keys.

        Returns {} when evidence is disabled or anything went wrong - the
        sighting is then reported exactly as it was before this existed.
        Wrapped whole for the same reason _capture_face_evidence is: an
        image is an enhancement, and a bad bbox or a full upload queue must
        not cost the dashboard a sighting it would otherwise have had.
        """
        if self._face_evidence is None:
            return {}

        try:
            cropped = evidence.crop_bbox(observation.crop, face.bbox,
                                        padding=FACE_EVIDENCE_PADDING)
            if cropped is None:
                return {}

            jpeg, crop_w, crop_h = cropped
            evidence_uid = uuid.uuid4().hex[:12]
            storage_key = evidence.build_storage_key(
                camera_id=observation.camera_id,
                track_id=person_uid,
                observed_at=observation.timestamp or time.time(),
                evidence_uid=evidence_uid,
                kind="face",
            )

            if not self._face_evidence.submit(jpeg, crop_w, crop_h, storage_key):
                self._counters["evidence_queue_dropped"] += 1
                return {}

            self._counters["evidence_prepared"] += 1

            # Face box in FRAME coordinates: the detector returned it
            # relative to the person crop, and the crop's origin is the only
            # thing that can put it back on the frame - the same translation
            # _emit() does for an identity.
            origin_x, origin_y = observation.crop_origin
            fx1, fy1, fx2, fy2 = (float(v) for v in face.bbox)

            return {
                "bbox": [int(origin_x + fx1), int(origin_y + fy1),
                         int(origin_x + fx2), int(origin_y + fy2)],
                "evidence": {
                    "uid": evidence_uid,
                    "type": "crop",
                    "backend": "seaweedfs-filer",
                    "storage_key": storage_key,
                    "mime_type": "image/jpeg",
                    "width": crop_w,
                    "height": crop_h,
                    "bytes": len(jpeg),
                },
            }

        except Exception as exc:  # noqa: BLE001 - the sighting outranks the image
            self._counters["evidence_prepare_failed"] += 1
            # No image bytes, no bbox and no embedding: diagnosable without
            # putting face data in a log.
            print(f"[FACE-EVIDENCE-ERROR] sighting crop skipped on "
                  f"{observation.camera_id} ({type(exc).__name__}: {exc}) "
                  f"- the sighting is still reported")
            return {}

    def _capture_face_evidence(self, observation, face, face_id):
        """
        Prepare and queue the one representative face crop for a newly
        minted identity. Returns the storage key on success, None if
        evidence is disabled or anything at all went wrong.

        Wrapped whole, deliberately. This runs AFTER _allocate_face_id()
        has already registered the id but BEFORE the Qdrant upsert and
        _emit() that make it real, so an exception escaping this method
        would strand a phantom identity - reserved, unvectorised, never
        reported. Both calls below genuinely raise on malformed input
        (crop_bbox does int() over the bbox; build_storage_key does
        datetime.fromtimestamp over the timestamp), so this is not a
        theoretical guard: a single NaN bbox or absurd epoch value from
        upstream would otherwise cost the identity, not just the image.

        Returning None on a queue-full drop is intentional: the key would
        point at an object the uploader has already refused to write, and a
        pointer we know is dead is worse than no pointer at all.
        """
        if self._face_evidence is None:
            return None

        try:
            cropped = evidence.crop_bbox(observation.crop, face.bbox,
                                        padding=FACE_EVIDENCE_PADDING)
            if cropped is None:
                # An unusable box (too small, inverted, off-frame) is an
                # ordinary outcome, not a failure - crop_bbox reports it by
                # returning None rather than raising.
                return None

            jpeg, crop_w, crop_h = cropped
            storage_key = evidence.build_storage_key(
                camera_id=observation.camera_id,
                track_id=face_id,
                observed_at=observation.timestamp or time.time(),
                evidence_uid=uuid.uuid4().hex[:12],
                kind="face",
            )

            # build_storage_key() is pure string formatting - the key is
            # valid immediately, before the upload (async) has happened, so
            # _emit() can carry it to the dashboard right away.
            if not self._face_evidence.submit(jpeg, crop_w, crop_h, storage_key):
                self._counters["evidence_queue_dropped"] += 1
                return None

            self._counters["evidence_prepared"] += 1
            return storage_key

        except Exception as exc:  # noqa: BLE001 - the identity outranks the image
            self._counters["evidence_prepare_failed"] += 1
            # No image bytes, no bbox and no embedding in this line: it has
            # to be diagnosable without putting face data into the logs.
            print(f"[FACE-EVIDENCE-ERROR] capture skipped for {face_id} on "
                  f"{observation.camera_id} ({type(exc).__name__}: {exc}) "
                  f"- identity keeps its id, Face-ID continues")
            return None

    def _emit(self, observation, face, face_id, identity, decision,
              best_sim, second_sim, margin, metrics, age_value, gender_value,
              evidence_key=None):
        """
        Hand the finished result to the injected sink, if there is one.

        Wrapped whole: a sink is an OUTPUT, and an output failing must never
        retract a decision the gallery has already recorded. A dropped report
        costs one row of application metadata; an exception escaping here
        would cost the worker its next observation too.
        """
        if self._result_sink is None:
            return

        try:
            origin_x, origin_y = observation.crop_origin
            fx1, fy1, fx2, fy2 = (float(v) for v in face.bbox)

            payload = {
                "face_id": face_id,
                "camera_id": observation.camera_id,
                "track_id": observation.local_track_id,
                "timestamp": observation.timestamp,
                "decision": decision,

                # Face box in FRAME coordinates - the detector returned it
                # relative to the person crop, and the crop's origin is the
                # only thing that can put it back on the frame.
                "bbox_x": int(origin_x + fx1),
                "bbox_y": int(origin_y + fy1),
                "bbox_width": int(fx2 - fx1),
                "bbox_height": int(fy2 - fy1),

                "similarity": best_sim,
                "runner_up_similarity": second_sim,
                "margin": margin,

                "face_width": metrics["face_width"],
                "face_height": metrics["face_height"],
                "det_score": metrics["det_score"],
                "sharpness": metrics["sharpness"],

                # Per-observation predictions, NOT the consensus.
                "age_prediction": age_value,
                "gender_prediction": normalise_gender(gender_value),

                # The consensus so far, so the application never has to
                # recompute an aggregation this class already owns.
                "age_range": identity.age_range,
                "age_confidence": identity.age_confidence,
                "age_observation_count": identity.age_observation_count,
                "gender": identity.gender,
                "gender_confidence": identity.gender_confidence,
                "gender_observation_count": identity.gender_observation_count,
                "gallery_count": identity.gallery_count,
                "observation_count": identity.observation_count,

                # Where the vector lives. The vector itself is NOT sent:
                # Qdrant is the vector store, and duplicating 512 floats into
                # PostgreSQL would make two sources of truth out of one.
                "model": FACE_ID_MODEL_ID,
                "embedding_collection": getattr(self._raw_store, "collection_name", None),

                # SeaweedFS reference for the one representative face image,
                # set only on the observation that actually captured it
                # (decision == "NEW"). Deliberately None on every MATCH -
                # the dashboard must never let an absent value here clobber
                # an already-stored key.
                "evidence_key": evidence_key,
            }

            self._result_sink(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[FACE-ID] result sink failed (contained): "
                  f"{type(exc).__name__}: {exc}")

    def _read_attributes(self, face):
        """
        Read age and gender off the detected face, each isolated from the
        other and both isolated from identity.

        buffalo_l computes attributes in the same forward pass as detection
        and recognition, so they cannot be throttled separately - they
        inherit the adapter's per-track cooldown, which is the throttle the
        brief asks for. What CAN be isolated is the read: a malformed or
        missing attribute costs that attribute only, never the identity that
        has already been decided, and never the other attribute.
        """
        age_value = gender_value = None

        try:
            raw_age = getattr(face, "age", None)
            if raw_age is not None:
                age_value = int(raw_age)
        except Exception as exc:  # noqa: BLE001
            self._counters["age_errors"] += 1
            print(f"[FACE-ID-AGE] unreadable ({type(exc).__name__}: {exc}) - identity unaffected")

        try:
            raw_gender = getattr(face, "gender", None)
            if raw_gender is not None and normalise_gender(raw_gender) is not None:
                gender_value = raw_gender
        except Exception as exc:  # noqa: BLE001
            self._counters["gender_errors"] += 1
            print(f"[FACE-ID-GENDER] unreadable ({type(exc).__name__}: {exc}) - identity unaffected")

        return age_value, gender_value

    def flush(self, timeout=5.0):
        """
        Wait for queued gallery writes to actually reach the store, WITHOUT
        stopping the writer thread. Returns the number still queued (0 means
        everything landed).

        AsyncUpsertStore offers only stop(), which is terminal: after it the
        writer thread is gone and any later upsert() sits in the queue
        forever, silently unwritten. That makes stop() unusable as a flush
        for anything that intends to keep going - a live harness reading
        back what it just wrote, or a test asserting a search sees the
        previous observation.

        Draining is therefore observed rather than commanded: wait for the
        queue to empty, then wait for the written counter to stop moving,
        which covers the item already in flight on the writer thread at the
        moment the queue hit zero.
        """
        if self._async_store is None:
            return 0

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline and self._async_store.queue_depth() > 0:
            time.sleep(0.02)

        stable = 0
        last_written = -1
        while time.monotonic() < deadline and stable < 2:
            written = self._async_store.written
            stable = stable + 1 if written == last_written else 0
            last_written = written
            time.sleep(0.03)

        return self._async_store.queue_depth()

    # ----------------------------------------------------------- reporting
    def stats(self):
        merged = dict(self._counters)
        merged["identities"] = len(self.identities)
        merged["genderage_available"] = self._genderage_available
        for reason, count in self._skip_reasons.items():
            merged[f"skip_{reason}"] = count
        return merged

    def profiles(self):
        return {fid: identity.as_dict() for fid, identity in sorted(self.identities.items())}

    def close(self):
        self._closed = True

        sink = getattr(self, "_sink", None)
        if sink is not None:
            try:
                sink.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[FACE-ID] sink shutdown error: {type(exc).__name__}: {exc}")

        if self._async_store is not None:
            try:
                remaining = self._async_store.stop()
                print(f"[FACE-ID-QDRANT] writer stopped  written={self._async_store.written} "
                      f"dropped_queue_full={self._async_store.dropped_queue_full} "
                      f"dropped_after_retries={self._async_store.dropped_after_retries} "
                      f"unflushed={remaining}")
            except Exception as exc:  # noqa: BLE001
                print(f"[FACE-ID-QDRANT] writer shutdown error: {type(exc).__name__}: {exc}")

        if self._face_evidence is not None:
            try:
                self._face_evidence.stop()
                stats = self._face_evidence.stats()
                # Both halves in one line: preparation is this class's step,
                # uploading is the pipeline's, and an operator needs to see
                # which of the two is failing without correlating two logs.
                print(f"[FACE-EVIDENCE] uploader stopped  uploaded={stats['uploaded']} "
                      f"upload_failed={stats['upload_failed']} "
                      f"queue_dropped={stats['queue_dropped']} "
                      f"prepared={self._counters['evidence_prepared']} "
                      f"prepare_failed={self._counters['evidence_prepare_failed']} "
                      f"dropped_at_submit={self._counters['evidence_queue_dropped']}")
            except Exception as exc:  # noqa: BLE001
                print(f"[FACE-EVIDENCE] uploader shutdown error: {type(exc).__name__}: {exc}")

        if self._recogniser is not None:
            try:
                stats = self._recogniser.stats()
                self._recogniser.close()
                print(f"[FACE-RECOGNITION] recogniser stopped  "
                      f"searches={stats['searches']} "
                      f"recognised={stats['recognised']} "
                      f"ambiguous={stats['ambiguous']} "
                      f"no_hit={stats['no_hit']} "
                      f"failed={stats['failed']} "
                      f"collection_absent={stats['collection_absent']}")
            except Exception as exc:  # noqa: BLE001
                print(f"[FACE-RECOGNITION] recogniser shutdown error: "
                      f"{type(exc).__name__}: {exc}")

        if self._recognition_sink is not None:
            try:
                self._recognition_sink.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[FACE-RECOGNITION] sink shutdown error: "
                      f"{type(exc).__name__}: {exc}")
