import numpy as np
import pytest

from visual_verify.retrieval.geometry import PatchGrid
from visual_verify.retrieval.types import FakeEmbedder, PageEmbedding


def test_page_embedding_rejects_vector_count_mismatch():
    grid = PatchGrid(n_x=2, n_y=2, offset=1, n_vectors=6)
    with pytest.raises(ValueError, match="does not match"):
        PageEmbedding(vectors=np.zeros((3, 8), dtype=np.float32), grid=grid)


def test_fake_embedder_is_deterministic():
    a = FakeEmbedder().embed_page("x.png", (100, 200))
    b = FakeEmbedder().embed_page("x.png", (100, 200))
    assert np.allclose(a.vectors, b.vectors)


def test_fake_embedder_differs_per_page():
    a = FakeEmbedder().embed_page("a.png", (100, 200))
    b = FakeEmbedder().embed_page("b.png", (100, 200))
    assert not np.allclose(a.vectors, b.vectors)


def test_fake_embedder_grid_matches_aspect_ratio():
    """Portrait and landscape must not produce the same grid, or tests that
    depend on grid variation would silently pass on a square assumption."""
    portrait = FakeEmbedder().embed_page("p.png", (100, 200))
    landscape = FakeEmbedder().embed_page("l.png", (200, 100))
    assert portrait.grid.n_x < portrait.grid.n_y
    assert landscape.grid.n_x > landscape.grid.n_y


def test_fake_embedder_vectors_are_unit_normalized():
    e = FakeEmbedder().embed_page("x.png", (100, 200))
    assert np.allclose(np.linalg.norm(e.vectors, axis=1), 1.0, atol=1e-5)


def test_fake_query_matches_its_own_page():
    """The fake must be retrievable, or pipeline tests prove nothing."""
    emb = FakeEmbedder()
    page = emb.embed_page("a.png", (100, 200))
    q = emb.embed_query("a.png")
    other = emb.embed_page("b.png", (100, 200))

    def score(p: PageEmbedding) -> float:
        return float((q @ p.vectors.T).max(axis=1).sum())

    assert score(page) > score(other)
