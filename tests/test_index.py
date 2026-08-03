import numpy as np
import pytest
from qdrant_client import models

from visual_verify.retrieval.geometry import PatchGrid
from visual_verify.retrieval.index import QdrantIndex, SchemaMismatch
from visual_verify.retrieval.provenance import ProvenanceMismatch
from visual_verify.retrieval.types import FakeEmbedder, PageEmbedding

SHA = "a" * 64
DIM = 128


@pytest.fixture
def index():
    """Local in-memory Qdrant: real client code, no server."""
    return QdrantIndex(url=":memory:", api_key=None, collection="pages_test")


@pytest.fixture
def embedder():
    return FakeEmbedder()


def _add(index, embedder, name, page_no):
    emb = embedder.embed_page(name, (100, 200))
    index.upsert_page(SHA, page_no, f"{SHA}/{name}", emb, embedder.provenance)
    return emb


def test_ensure_collection_is_idempotent(index):
    index.ensure_collection()
    index.ensure_collection()
    assert index.count() == 0


def test_upsert_then_count(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "p0.png", 0)
    assert index.count() == 1


def test_upsert_is_idempotent_on_same_page(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "p0.png", 0)
    _add(index, embedder, "p0.png", 0)
    assert index.count() == 1, "deterministic point id must overwrite, not duplicate"


def test_existing_page_nos_drives_resumption(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "p0.png", 0)
    _add(index, embedder, "p2.png", 2)
    assert index.existing_page_nos(SHA) == {0, 2}


def test_existing_page_nos_is_scoped_per_document(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "p0.png", 0)
    assert index.existing_page_nos("b" * 64) == set()


def test_payload_carries_geometry(index, embedder):
    index.ensure_collection()
    emb = _add(index, embedder, "p0.png", 0)
    payload = index.get_payload(SHA, 0)
    assert payload["n_patches_x"] == emb.grid.n_x
    assert payload["n_patches_y"] == emb.grid.n_y
    assert payload["n_special_tokens"] == emb.grid.n_special
    assert payload["patch_offset"] == emb.grid.offset


def test_search_ranks_the_matching_page_first(index, embedder):
    index.ensure_collection()
    for i, name in enumerate(["alpha.png", "beta.png", "gamma.png"]):
        _add(index, embedder, name, i)
    hits = index.search(embedder.embed_query("beta.png"), embedder.provenance, limit=3)
    assert hits[0].page == 1


def test_search_returns_retrieved_pages(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "alpha.png", 0)
    hit = index.search(embedder.embed_query("alpha.png"), embedder.provenance, limit=1)[0]
    assert hit.doc_id == SHA
    assert hit.image_ref == f"{SHA}/alpha.png"
    assert hit.score > 0


def test_search_refuses_on_provenance_mismatch(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "alpha.png", 0)
    from visual_verify.retrieval.provenance import EmbedProvenance

    other = EmbedProvenance(**{**embedder.provenance.to_payload(), "render_dpi": 300})
    with pytest.raises(ProvenanceMismatch, match="render_dpi"):
        index.search(embedder.embed_query("alpha.png"), other, limit=1)


def test_upsert_refuses_on_provenance_mismatch(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "alpha.png", 0)
    from visual_verify.retrieval.provenance import EmbedProvenance

    other = EmbedProvenance(**{**embedder.provenance.to_payload(), "model_id": "other"})
    emb = embedder.embed_page("beta.png", (100, 200))
    with pytest.raises(ProvenanceMismatch, match="model_id"):
        index.upsert_page(SHA, 1, f"{SHA}/beta.png", emb, other)


def test_qdrant_ranking_matches_local_maxsim(index, embedder):
    """A misconfigured collection still accepts writes and still returns
    results; it just returns the wrong ones. This is the test that notices."""
    index.ensure_collection()
    embs = {}
    for i, name in enumerate(["a.png", "b.png", "c.png", "d.png"]):
        embs[i] = _add(index, embedder, name, i)

    q = embedder.embed_query("c.png")
    hits = index.search(q, embedder.provenance, limit=4)
    local = sorted(embs, key=lambda i: -float((q @ embs[i].vectors.T).max(axis=1).sum()))
    assert [h.page for h in hits] == local


def test_pooled_vectors_are_stored(index, embedder):
    index.ensure_collection()
    emb = _add(index, embedder, "a.png", 0)
    stored = index.get_vectors(SHA, 0)
    assert stored["original"].shape[0] == emb.grid.n_vectors
    assert stored["mean_pooling_rows"].shape[0] == emb.grid.n_y + emb.grid.n_special
    assert stored["mean_pooling_cols"].shape[0] == emb.grid.n_x + emb.grid.n_special


def test_pooled_vectors_agree_with_pooling_module(index, embedder):
    """Written in S3, used later. A silently wrong pooled vector would only
    surface as degraded rerank quality much later."""
    from visual_verify.retrieval.pooling import mean_pool_rows

    index.ensure_collection()
    emb = _add(index, embedder, "a.png", 0)
    stored = index.get_vectors(SHA, 0)["mean_pooling_rows"]
    assert np.allclose(stored, mean_pool_rows(emb.vectors, emb.grid), atol=1e-5)


def test_recreate_clears_points(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "a.png", 0)
    index.ensure_collection(recreate=True)
    assert index.count() == 0


def _basis(k: int) -> np.ndarray:
    """The k-th standard basis vector in R^DIM: unit norm and orthogonal to
    every other basis vector by construction, which is what lets us pick exact
    dot products by hand instead of hoping random noise cooperates."""
    v = np.zeros(DIM, dtype=np.float32)
    v[k] = 1.0
    return v


def _grid_2x2() -> PatchGrid:
    return PatchGrid(n_x=2, n_y=2, offset=0, n_vectors=4)


def _max_score(q: np.ndarray, vectors: np.ndarray) -> float:
    """sum over query tokens of the per-token MAX over patches: real MaxSim."""
    return float((q @ vectors.T).max(axis=1).sum())


def _mean_score(q: np.ndarray, vectors: np.ndarray) -> float:
    """sum over query tokens of the per-token MEAN over patches: the
    aggregation a wrongly-configured (e.g. plain-cosine, no multivector)
    collection would effectively produce instead."""
    return float((q @ vectors.T).mean(axis=1).sum())


def test_ranking_is_maxsim_not_mean_similarity(index, embedder):
    """FakeEmbedder's signal is too easy to discriminate comparators: a query's
    own patch vectors are exact unit matches (dot 1.0) against its page and
    near-orthogonal elsewhere, so the true page wins under max, mean, or nearly
    any other aggregation. That made test_qdrant_ranking_matches_local_maxsim
    pass even when its local reference formula was swapped from max to mean.

    This test builds two pages by hand from an orthonormal basis so the two
    aggregations are forced to disagree: page A has one patch nearly identical
    to the query (high max, low mean over its patches) and page B has many
    patches at a moderate, uniform similarity (lower max, higher mean).

    Step 1 below asserts the two local formulas actually pick different
    winners. That assertion is what keeps this test honest: if the hand-built
    vectors ever stopped disagreeing (a careless edit changing a magnitude),
    step 1 fails loudly instead of the test silently degrading back into a
    tautology that passes regardless of which comparator Qdrant is running.
    """
    q = _basis(0)[None, :]  # one query token, unit vector e0

    # Page A: one patch equal to e0 (dot 1.0), three patches orthogonal to it
    # (dot 0.0). max = 1.0, mean = 0.25.
    a_vectors = np.stack([_basis(0), _basis(1), _basis(2), _basis(3)])
    # Page B: four patches all at a uniform, moderate similarity to e0. Built
    # from two orthonormal basis directions so each row is already unit norm:
    # 0.5**2 + (sqrt(3)/2)**2 == 1.0. max = 0.5, mean = 0.5.
    b_dir = (0.5 * _basis(0) + np.sqrt(3) / 2 * _basis(1)).astype(np.float32)
    b_vectors = np.stack([b_dir, b_dir, b_dir, b_dir])

    grid = _grid_2x2()
    page_a = PageEmbedding(vectors=a_vectors, grid=grid)
    page_b = PageEmbedding(vectors=b_vectors, grid=grid)

    max_a, max_b = _max_score(q, a_vectors), _max_score(q, b_vectors)
    mean_a, mean_b = _mean_score(q, a_vectors), _mean_score(q, b_vectors)
    print(f"max: A={max_a:.4f} B={max_b:.4f}  mean: A={mean_a:.4f} B={mean_b:.4f}")

    # Step 1: test the test. If these ever agreed, every assertion below would
    # pass no matter which aggregation Qdrant actually used.
    assert max_a > max_b, "hand-built page A must win under max, or this test is a tautology"
    assert mean_b > mean_a, "hand-built page B must win under mean, or this test is a tautology"

    index.ensure_collection()
    index.upsert_page(SHA, 0, f"{SHA}/a.png", page_a, embedder.provenance)
    index.upsert_page(SHA, 1, f"{SHA}/b.png", page_b, embedder.provenance)

    hits = index.search(q, embedder.provenance, limit=2)

    # Step 2: Qdrant's real ranking must match the MaxSim answer (A first),
    # not the mean answer (which would rank B first).
    assert [h.page for h in hits] == [0, 1]


def test_ensure_collection_accepts_its_own_schema(index):
    """The happy path must not be broken by the schema check."""
    index.ensure_collection()
    index.ensure_collection()  # second call verifies rather than recreates
    assert index.count() == 0


def test_ensure_collection_refuses_an_unnamed_vector_collection(index):
    """Existence is not compatibility.

    A collection predating the named-vector schema still connects fine. Without
    this check it would fail an upsert with an opaque Qdrant error, or accept
    the writes and return confidently ranked wrong results. The project's own
    cloud collection was in exactly this state.
    """
    from qdrant_client import models

    index.client.create_collection(
        collection_name=index.collection,
        vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE),
    )
    with pytest.raises(SchemaMismatch, match="unnamed"):
        index.ensure_collection()


def test_ensure_collection_refuses_a_missing_named_vector(index):
    from qdrant_client import models

    params = models.VectorParams(
        size=128,
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
        hnsw_config=models.HnswConfigDiff(m=0),
    )
    index.client.create_collection(
        collection_name=index.collection,
        vectors_config={"original": params},  # pooled vectors absent
    )
    with pytest.raises(SchemaMismatch, match="mean_pooling"):
        index.ensure_collection()


def test_ensure_collection_refuses_a_wrong_dimension(index):
    from qdrant_client import models

    params = models.VectorParams(
        size=64,  # not DIM
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
    )
    index.client.create_collection(
        collection_name=index.collection,
        vectors_config={
            "original": params,
            "mean_pooling_rows": params,
            "mean_pooling_cols": params,
        },
    )
    with pytest.raises(SchemaMismatch, match="size=64"):
        index.ensure_collection()


def test_recreate_overrides_a_bad_schema(index):
    """recreate=True is the documented escape hatch, and it must work."""
    from qdrant_client import models

    index.client.create_collection(
        collection_name=index.collection,
        vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE),
    )
    index.ensure_collection(recreate=True)
    index.ensure_collection()  # now verifies clean
    assert index.count() == 0


def test_ensure_collection_requests_a_doc_sha_payload_index(index):
    """A real Qdrant server refuses to filter on an unindexed payload field:

        400 Bad Request: Index required but not found for "doc_sha"

    `existing_page_nos` filters on doc_sha, and resumption depends entirely on
    it, so the whole slice fails on the first call against a real cluster.

    This asserts the CALL rather than its effect, which is unusual but forced:
    local Qdrant warns "Payload indexes have no effect in the local Qdrant" and
    reports an empty payload_schema, so the effect is unobservable here. The
    call is the load-bearing part, and the realistic regression is someone
    deleting it after seeing it do nothing in the tests.
    """
    calls = []
    real = index.client.create_payload_index

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    index.client.create_payload_index = spy
    index.ensure_collection(recreate=True)

    assert calls, "ensure_collection must request a doc_sha payload index"
    assert calls[0]["field_name"] == "doc_sha"
    assert calls[0]["field_schema"] == models.PayloadSchemaType.KEYWORD


def test_verify_pass_also_requests_the_index(index, embedder):
    """A collection created before the index existed must gain it on next open."""
    index.ensure_collection()
    _add(index, embedder, "a.png", 0)

    calls = []
    real = index.client.create_payload_index

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    index.client.create_payload_index = spy
    index.ensure_collection()  # verify path, not recreate

    assert calls, "the verify path must also request the index"
    assert index.existing_page_nos(SHA) == {0}
