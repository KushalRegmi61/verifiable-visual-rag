import numpy as np
import pytest

from visual_verify.retrieval.index import QdrantIndex
from visual_verify.retrieval.provenance import ProvenanceMismatch
from visual_verify.retrieval.types import FakeEmbedder

SHA = "a" * 64


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
