"""Qdrant multivector storage and MaxSim search.

Three NAMED vectors in one collection. S3 populates all three but queries only
"original"; the pooled pair exists because Qdrant cannot add a named vector to an
existing collection without recreating it, so the schema is a one-way door and
provisioning now is far cheaper than a re-embed at 21.4 s per page later.
"""

import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from visual_verify.contracts import RetrievedPage
from visual_verify.retrieval.pooling import mean_pool_cols, mean_pool_rows
from visual_verify.retrieval.provenance import EmbedProvenance
from visual_verify.retrieval.types import PageEmbedding

DIM = 128
ORIGINAL = "original"
POOL_ROWS = "mean_pooling_rows"
POOL_COLS = "mean_pooling_cols"

# uuid5(NAMESPACE_DNS, "verifiable-visual-rag.pages"). Fixed forever: changing it
# orphans every existing point. Derived rather than random so it is auditable.
POINT_NS = uuid.UUID("5ee1d73c-35dc-53bb-8bf7-94bd98b0b932")


class SchemaMismatch(RuntimeError):
    """An existing collection's vector schema is not the one this code writes."""


def point_id(doc_sha: str, page_no: int) -> str:
    """Deterministic, so re-embedding overwrites instead of duplicating."""
    return str(uuid.uuid5(POINT_NS, f"{doc_sha}:{page_no}"))


class QdrantIndex:
    def __init__(self, url: str, api_key: str | None, collection: str = "pages") -> None:
        # ":memory:" gives a real local client with no server, which is what
        # makes this class testable in CI.
        if url == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        self.collection = collection

    def _vector_params(self) -> models.VectorParams:
        return models.VectorParams(
            size=DIM,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            # m=0 disables HNSW graph construction. Necessary, not merely
            # acceptable: every candidate comparison in a multivector HNSW build
            # is itself a full MaxSim, which is combinatorially expensive. The
            # evaluation in a later slice also needs exact scores.
            hnsw_config=models.HnswConfigDiff(m=0),
        )

    def _check_schema(self) -> None:
        """Refuse a pre-existing collection whose schema is not the one we write.

        Existence is not the same as compatibility, and the gap is dangerous.
        A collection left over from an earlier schema (a single unnamed vector,
        a different dimension, or a comparator other than MAX_SIM) will still
        accept a connection. Depending on which field differs it then either
        fails an upsert with an opaque Qdrant error, or worse, accepts the
        writes and returns confidently ranked wrong results.

        This was not hypothetical: the project's own cloud collection was
        created during design-time smoke testing with a single unnamed vector,
        before the three-named-vector schema existed.
        """
        cfg = self.client.get_collection(self.collection).config.params.vectors
        if not isinstance(cfg, dict):
            raise SchemaMismatch(
                f"collection {self.collection!r} holds a single unnamed vector, but this "
                f"code writes named vectors {sorted([ORIGINAL, POOL_ROWS, POOL_COLS])}. "
                "It predates the current schema. Recreate it with "
                "ensure_collection(recreate=True), which DELETES its contents."
            )
        missing = {ORIGINAL, POOL_ROWS, POOL_COLS} - set(cfg)
        if missing:
            raise SchemaMismatch(
                f"collection {self.collection!r} is missing named vectors {sorted(missing)}. "
                "Qdrant cannot add one to an existing collection, so this needs "
                "ensure_collection(recreate=True), which DELETES its contents."
            )
        for name in (ORIGINAL, POOL_ROWS, POOL_COLS):
            got = cfg[name]
            comparator = got.multivector_config.comparator if got.multivector_config else None
            if got.size != DIM or comparator != models.MultiVectorComparator.MAX_SIM:
                raise SchemaMismatch(
                    f"named vector {name!r} in {self.collection!r} is "
                    f"size={got.size} comparator={comparator}, expected "
                    f"size={DIM} comparator={models.MultiVectorComparator.MAX_SIM}. "
                    "Scores from this collection would not be MaxSim."
                )

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            # Verify rather than assume. See _check_schema.
            self._check_schema()
            # Idempotent, and an older collection may predate the index.
            self._ensure_payload_index()
            return
        if exists:
            self.client.delete_collection(self.collection)
        params = self._vector_params()
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={ORIGINAL: params, POOL_ROWS: params, POOL_COLS: params},
        )
        self._ensure_payload_index()

    def _ensure_payload_index(self) -> None:
        """Index doc_sha, which resumption filters on.

        A real Qdrant server REFUSES to filter on an unindexed payload field:

            400 Bad Request: Index required but not found for "doc_sha"
            of one of the following types: [keyword]

        The local in-memory client used by the tests happily filters without
        one, so every test passes and `existing_page_nos` fails on the first
        call against the real cluster. Resumption depends entirely on that
        function, so the divergence takes out the whole slice.

        Same shape as the SQLite/Postgres timezone trap recorded in CLAUDE.md:
        the development backend is more permissive than production, so the bug
        can only appear in the environment that matters.
        """
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="doc_sha",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def _stored_provenance(self) -> EmbedProvenance | None:
        """Provenance of whatever is already indexed, or None if empty (or the
        collection does not exist yet)."""
        if not self.client.collection_exists(self.collection):
            return None
        points, _ = self.client.scroll(self.collection, limit=1, with_payload=True)
        if not points:
            return None
        return EmbedProvenance.from_payload(points[0].payload)

    def _require_compatible(self, provenance: EmbedProvenance) -> None:
        stored = self._stored_provenance()
        if stored is not None:
            stored.require_compatible(provenance)

    def upsert_page(
        self,
        doc_sha: str,
        page_no: int,
        image_path: str,
        embedding: PageEmbedding,
        provenance: EmbedProvenance,
    ) -> None:
        # Checked before building the point struct or touching the client's
        # upsert call, so a mismatch never gets as far as a partial write.
        self._require_compatible(provenance)
        grid = embedding.grid
        payload = {
            "doc_sha": doc_sha,
            "page_no": page_no,
            "image_path": image_path,
            # Geometry. Unrecoverable without re-embedding, and the grounding
            # slice cannot place a single box without it.
            "n_patches_x": grid.n_x,
            "n_patches_y": grid.n_y,
            "n_image_patches": grid.n_image_patches,
            "n_special_tokens": grid.n_special,
            "patch_offset": grid.offset,
            **provenance.to_payload(),
        }
        self.client.upsert(
            collection_name=self.collection,
            wait=True,  # per-page durability; noise against a 21.4 s embed
            points=[
                models.PointStruct(
                    id=point_id(doc_sha, page_no),
                    vector={
                        ORIGINAL: embedding.vectors.tolist(),
                        POOL_ROWS: mean_pool_rows(embedding.vectors, grid).tolist(),
                        POOL_COLS: mean_pool_cols(embedding.vectors, grid).tolist(),
                    },
                    payload=payload,
                )
            ],
        )

    def existing_page_nos(self, doc_sha: str) -> set[int]:
        """Which pages are already indexed. Qdrant is the source of truth for
        embedding state, so this cannot desync from what is actually stored."""
        flt = models.Filter(
            must=[models.FieldCondition(key="doc_sha", match=models.MatchValue(value=doc_sha))]
        )
        out: set[int] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, scroll_filter=flt, limit=256, offset=offset, with_payload=True
            )
            out.update(p.payload["page_no"] for p in points)
            if offset is None:
                return out

    def get_payload(self, doc_sha: str, page_no: int) -> dict:
        recs = self.client.retrieve(self.collection, ids=[point_id(doc_sha, page_no)])
        return recs[0].payload

    def get_payload_or_none(self, doc_sha: str, page_no: int) -> dict | None:
        """get_payload, but absence is an answer rather than an IndexError.

        Exists so a caller can ask "is this one page indexed" with a single
        point lookup. The obvious alternative, `page_no in
        existing_page_nos(sha)`, scrolls EVERY point of the document 256 at a
        time with payloads attached, which is the right shape for `vvrag embed`
        deciding what to resume and the wrong shape for the API, where
        prepare_page runs on every question.
        """
        recs = self.client.retrieve(self.collection, ids=[point_id(doc_sha, page_no)])
        return recs[0].payload if recs else None

    def get_vectors(self, doc_sha: str, page_no: int) -> dict[str, np.ndarray]:
        """Read stored vectors back. This is what makes a schema change a
        re-index rather than a re-embed."""
        recs = self.client.retrieve(
            self.collection, ids=[point_id(doc_sha, page_no)], with_vectors=True
        )
        return {k: np.asarray(v, dtype=np.float32) for k, v in recs[0].vector.items()}

    def search(
        self, query_vectors: np.ndarray, provenance: EmbedProvenance, limit: int = 5
    ) -> list[RetrievedPage]:
        self._require_compatible(provenance)
        res = self.client.query_points(
            self.collection,
            query=query_vectors.tolist(),
            using=ORIGINAL,
            limit=limit,
            with_payload=True,
        ).points
        return [
            RetrievedPage(
                doc_id=p.payload["doc_sha"],
                page=p.payload["page_no"],
                image_ref=p.payload["image_path"],
                score=p.score,
            )
            for p in res
        ]
