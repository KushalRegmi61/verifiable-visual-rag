"""Relevance maps, and the special-token exclusion that keeps them honest."""

import numpy as np
import pytest

from visual_verify.grounding.heatmap import dense_relevance
from visual_verify.retrieval.geometry import PatchGrid


def make_grid(n_x=4, n_y=3, offset=2, n_suffix=1):
    """A small grid with the same prefix/suffix shape as a real page."""
    return PatchGrid(n_x=n_x, n_y=n_y, offset=offset, n_vectors=offset + n_x * n_y + n_suffix)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_dense_relevance_has_one_score_per_image_patch():
    grid = make_grid()
    rng = np.random.default_rng(0)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(5, 8)))

    r = dense_relevance(query, page, grid)

    assert r.shape == (grid.n_image_patches,)


def test_dense_relevance_peaks_on_the_planted_patch():
    """A patch made identical to a query token must win its own relevance."""
    grid = make_grid()
    rng = np.random.default_rng(1)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(3, 8)))

    target_patch = 7
    page[grid.offset + target_patch] = query[1]

    r = dense_relevance(query, page, grid)

    assert int(r.argmax()) == target_patch


def test_dense_relevance_ignores_special_tokens():
    """A special token planted with the query must not shift any patch score.

    Special tokens map to no page region. If the map were built over all
    vectors, a prefix token would surface as a confidently drawn box with no
    causal link to the evidence.
    """
    grid = make_grid()
    rng = np.random.default_rng(2)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(3, 8)))

    before = dense_relevance(query, page, grid)
    page[0] = query[0]  # prefix token, index < offset
    page[-1] = query[2]  # suffix token, past the image patches
    after = dense_relevance(query, page, grid)

    assert np.array_equal(before, after)


def test_dense_relevance_rejects_a_vector_count_mismatch():
    """Silent truncation would shift the whole grid and misplace every box."""
    grid = make_grid()
    page = np.zeros((grid.n_vectors + 3, 8))
    query = np.zeros((2, 8))

    with pytest.raises(ValueError, match="does not match grid"):
        dense_relevance(query, page, grid)


def test_dense_relevance_rejects_a_dimension_mismatch():
    grid = make_grid()
    page = np.zeros((grid.n_vectors, 8))
    query = np.zeros((2, 16))

    with pytest.raises(ValueError, match="dimension"):
        dense_relevance(query, page, grid)
