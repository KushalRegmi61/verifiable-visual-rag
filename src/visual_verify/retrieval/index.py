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

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return
        if exists:
            self.client.delete_collection(self.collection)
        params = self._vector_params()
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={ORIGINAL: params, POOL_ROWS: params, POOL_COLS: params},
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
