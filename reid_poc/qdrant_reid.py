"""
Qdrant-backed Store (Phase 3) - same interface as reid_manager.InMemoryReIDStore,
so GlobalReIDManager and everything upstream of it (JeztSort, the tracker,
the crop/embedding pipeline) needs ZERO changes to use this instead. Swap via
config.REID_STORE = "qdrant".

Collection: config.QDRANT_COLLECTION, vector size config.EMBEDDING_DIM,
distance config.QDRANT_DISTANCE (COSINE). Represents each global identity as
a SMALL GALLERY of points (capped at config.REID_GALLERY_SIZE), not one point
per frame - every point is payload-tagged with global_person_id so a gallery
can be fetched, trimmed, or deleted as a group.
"""
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

import config
from reid_manager import Store, highest_identity_seq


class QdrantReIDStore(Store):
    def __init__(
        self,
        collection_name=None,
        vector_dim=None,
        distance=None,
        gallery_size=None,
        host=None,
        port=None,
        timeout=None,
    ):
        # None -> read config.* fresh here, not as a default-parameter value
        # (see GlobalReIDManager.__init__ for why: a default bound at import
        # time would not see a later config.X reassignment).
        self.collection_name = config.QDRANT_COLLECTION if collection_name is None else collection_name
        self.gallery_size = config.REID_GALLERY_SIZE if gallery_size is None else gallery_size
        vector_dim = config.QDRANT_VECTOR_DIM if vector_dim is None else vector_dim
        distance = config.QDRANT_DISTANCE if distance is None else distance

        # host/port/timeout overrides - added for the production adapter
        # (AI_inferencing/reid_adapter.py), which must point at its OWN
        # Qdrant endpoint/timeout independently of whatever this PoC's own
        # config.QDRANT_HOST/PORT happen to be set to for its own testing.
        # None (the default, every existing caller) keeps reading config.*
        # exactly as before - fully backward compatible.
        self._host = config.QDRANT_HOST if host is None else host
        self._port = config.QDRANT_PORT if port is None else port
        self._timeout = timeout   # None = qdrant-client's own default timeout

        self.client = self._connect()
        self._ensure_collection(vector_dim, distance)

    # ------------------------------------------------------------- connect
    def _connect(self):
        if config.QDRANT_MODE == "local":
            print(f"Qdrant: embedded local mode at {config.QDRANT_LOCAL_PATH}")
            return QdrantClient(path=config.QDRANT_LOCAL_PATH)

        print(f"Qdrant: server mode at {self._host}:{self._port}"
              f"{f' (timeout={self._timeout}s)' if self._timeout is not None else ''}")
        kwargs = {"host": self._host, "port": self._port}
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        return QdrantClient(**kwargs)

    def _ensure_collection(self, vector_dim, distance):
        if self.client.collection_exists(self.collection_name):
            return

        print(f"Qdrant: creating collection '{self.collection_name}' "
              f"(dim={vector_dim}, distance={distance})")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_dim,
                distance=getattr(models.Distance, distance.upper()),
            ),
        )

    # ------------------------------------------------------------- search
    def search(self, embedding, top_k=1, exclude_camera=None, min_timestamp=None,
               allowed_cameras=None, only_global_id=None):
        """
        Best-scoring point PER DISTINCT global_person_id, best identity
        first - mirrors InMemoryReIDStore's "best gallery entry, not the
        average" semantics. Over-fetches raw points (gallery_size * top_k,
        floored at 20) so a genuinely-best identity is not missed just
        because its top point did not rank in a smaller raw top_k.

        exclude_camera / min_timestamp / allowed_cameras are applied AT THE
        QDRANT QUERY LEVEL via a Filter - filtered-out points never count
        towards an identity's best score, and an identity whose only points
        are all filtered out simply does not appear in the results (it can
        still match via its OTHER cameras' points - see reid_manager.Store).
        only_global_id restricts the whole search to one identity's own
        points (used by the possible-ID-swap diagnostic).
        """
        raw_limit = max(20, self.gallery_size * max(top_k, 1) * 3)

        must = []
        must_not = []

        if exclude_camera is not None:
            must_not.append(models.FieldCondition(
                key="camera_id", match=models.MatchValue(value=exclude_camera),
            ))

        if allowed_cameras is not None:
            must.append(models.FieldCondition(
                key="camera_id", match=models.MatchAny(any=list(allowed_cameras)),
            ))

        if min_timestamp is not None:
            must.append(models.FieldCondition(
                key="timestamp", range=models.Range(gte=min_timestamp),
            ))

        if only_global_id is not None:
            must.append(models.FieldCondition(
                key="global_person_id", match=models.MatchValue(value=only_global_id),
            ))

        query_filter = models.Filter(must=must, must_not=must_not) if (must or must_not) else None

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding.tolist(),
            limit=raw_limit,
            with_payload=True,
            query_filter=query_filter,
        )

        best_per_identity = {}
        for point in response.points:
            global_id = point.payload.get("global_person_id")

            if global_id is None:
                continue

            if global_id not in best_per_identity or point.score > best_per_identity[global_id][0]:
                best_per_identity[global_id] = (point.score, point.payload)

        ranked = sorted(best_per_identity.items(), key=lambda row: row[1][0], reverse=True)

        return [
            (global_id, float(score), payload)
            for global_id, (score, payload) in ranked[:top_k]
        ]

    # ------------------------------------------------------------- upsert
    def upsert(self, global_id, embedding, payload):
        point_payload = dict(payload)
        point_payload["global_person_id"] = global_id

        self.client.upsert(
            collection_name=self.collection_name,
            points=[models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=point_payload,
            )],
        )

        self._trim_gallery(global_id)

    def _trim_gallery(self, global_id):
        """Keep at most gallery_size points for one identity, oldest dropped
        by insertion order (Qdrant point ids here carry no ordering, so this
        uses payload timestamp when present, else deletes arbitrarily among
        the oldest-scrolled - acceptable for a small PoC gallery)."""
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self._identity_filter(global_id),
            limit=self.gallery_size * 4,
            with_payload=True,
        )

        if len(points) <= self.gallery_size:
            return

        points.sort(key=lambda p: p.payload.get("timestamp") or 0)
        excess = points[: len(points) - self.gallery_size]

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[p.id for p in excess]),
        )

    # ------------------------------------------------------------- identities
    def highest_identity_seq(self, page_size=1000):
        """Highest P-<n> sequence number stored in the collection, 0 if none.

        Called ONCE, at startup, so a new process continues numbering after
        the ids already persisted (see reid_manager's IDENTITY NUMBERING).
        Reads every point's global_person_id payload only - no vectors - one
        page at a time. Any Qdrant error PROPAGATES: returning 0 on failure
        would restart numbering at P-0001 and collide with stored identities.
        """
        highest, offset = 0, None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=page_size,
                offset=offset,
                with_payload=["global_person_id"],
                with_vectors=False,
            )
            highest = max(highest, highest_identity_seq(
                (p.payload or {}).get("global_person_id") for p in points))
            if offset is None:
                return highest

    # ------------------------------------------------------------- history
    def get_history(self, global_id):
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self._identity_filter(global_id),
            limit=self.gallery_size * 4,
            with_payload=True,
        )
        points.sort(key=lambda p: p.payload.get("timestamp") or 0)
        return [p.payload for p in points]

    @staticmethod
    def _identity_filter(global_id):
        return models.Filter(
            must=[models.FieldCondition(
                key="global_person_id",
                match=models.MatchValue(value=global_id),
            )]
        )
