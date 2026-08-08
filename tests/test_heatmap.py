"""Relevance maps, and the special-token exclusion that keeps them honest."""

import numpy as np
import pytest

from visual_verify.grounding.heatmap import attribution, dense_relevance
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


def test_dense_relevance_rejects_nan_in_page_vectors():
    """NaN compares greater than every real score, so argmax would silently
    pick the corrupted patch as the top-ranked candidate: no exception, no
    shape anomaly, nothing about the output looking wrong.
    """
    grid = make_grid()
    rng = np.random.default_rng(3)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(3, 8)))
    page[grid.offset + 1] = np.nan

    with pytest.raises(ValueError, match="NaN or Inf"):
        dense_relevance(query, page, grid)


def test_dense_relevance_rejects_inf_in_page_vectors():
    """An infinite dot product would dominate every comparison downstream."""
    grid = make_grid()
    rng = np.random.default_rng(4)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(3, 8)))
    page[grid.offset + 2] = np.inf

    with pytest.raises(ValueError, match="NaN or Inf"):
        dense_relevance(query, page, grid)


def test_dense_relevance_rejects_nan_in_query_vectors():
    """A corrupted query token is just as capable of winning a fabricated max."""
    grid = make_grid()
    rng = np.random.default_rng(5)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(3, 8)))
    query[0] = np.nan

    with pytest.raises(ValueError, match="NaN or Inf"):
        dense_relevance(query, page, grid)


def test_dense_relevance_rejects_a_1d_query():
    """A real query never has fewer than 14 tokens (ColQwen2's fixed prompt
    prefix), so a 1-D query is always a caller forgetting the token axis, not
    legitimate input. Unchecked, numpy fails deep inside with
    'too many indices for array' instead of naming the actual mistake.
    """
    grid = make_grid()
    page = np.zeros((grid.n_vectors, 8))
    query = np.zeros(8)

    with pytest.raises(ValueError, match="2-D"):
        dense_relevance(query, page, grid)


def test_dense_relevance_rejects_an_empty_query():
    """Zero query tokens is a caller bug, not a zero-relevance page; unchecked
    it fails inside numpy's max reduction instead of naming the mistake.
    """
    grid = make_grid()
    page = np.zeros((grid.n_vectors, 8))
    query = np.zeros((0, 8))

    with pytest.raises(ValueError, match="2-D"):
        dense_relevance(query, page, grid)


def test_attribution_credits_only_the_winning_patch():
    """One query token's score lands on the one patch that won it, and nowhere else.

    Single token on purpose. With several tokens the top-credited patch is not
    necessarily the best-matching one, because credit accumulates: see
    test_attribution_sums_credit_across_tokens.
    """
    grid = make_grid()
    rng = np.random.default_rng(3)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(1, 8)))

    target_patch = 5
    page[grid.offset + target_patch] = query[0]  # exact match, similarity 1.0

    a = attribution(query, page, grid)

    assert a.shape == (grid.n_image_patches,)
    assert np.count_nonzero(a) == 1
    assert int(a.argmax()) == target_patch
    assert a[target_patch] == pytest.approx(1.0)


def test_attribution_sums_credit_across_tokens():
    """Credit accumulates per patch rather than overwriting.

    So a patch winning two tokens can outrank a patch winning one perfect
    token. That is correct for a decomposition of the page score, and it is a
    second reason this map must not be used to rank candidates: the
    highest-credited patch is not necessarily the best-matching one.
    """
    grid = make_grid()
    rng = np.random.default_rng(7)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    duplicated = unit(rng.normal(size=(8,)))
    # Two identical query tokens must both take their maximum on the same
    # patch, so the planted patch is credited twice.
    query = np.stack([duplicated, duplicated])
    page[grid.offset + 2] = duplicated

    a = attribution(query, page, grid)

    assert a[2] == pytest.approx(2.0), "credit must accumulate, not overwrite"
    assert np.count_nonzero(a) == 1


def test_attribution_is_sparse():
    """At most one patch per query token can be credited.

    This is why attribution cannot rank: a real 19-token query lights 4 of 736
    patches, so nearly every line inside a block would score exactly 0 and
    stage 2 would be breaking ties at random.
    """
    grid = make_grid(n_x=8, n_y=8, offset=2)
    rng = np.random.default_rng(4)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(3, 8)))

    a = attribution(query, page, grid)

    assert np.count_nonzero(a) >= 1, "an all-zero map would satisfy the bounds below"
    assert np.count_nonzero(a) <= query.shape[0]
    assert np.count_nonzero(a) < grid.n_image_patches


def test_attribution_drops_tokens_won_by_special_tokens():
    """A token whose maximum is a prefix vector credits no patch at all."""
    grid = make_grid()
    rng = np.random.default_rng(5)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(2, 8)))

    page[0] = query[0]  # prefix token wins query token 0 outright
    a = attribution(query, page, grid)

    # Only query token 1 can have credited anything.
    assert np.count_nonzero(a) <= 1


def test_attribution_drops_tokens_won_by_a_suffix_token():
    """The mirror of the prefix case, covering the upper boundary.

    Without this, a < versus <= slip on the hi side is untested: the prefix
    test only exercises the lower bound.
    """
    grid = make_grid()
    rng = np.random.default_rng(8)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(2, 8)))

    page[-1] = query[0]  # last vector is a suffix token
    a = attribution(query, page, grid)

    assert np.count_nonzero(a) <= 1
