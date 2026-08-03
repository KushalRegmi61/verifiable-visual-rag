import numpy as np
import pytest

from visual_verify.retrieval.geometry import PatchGrid
from visual_verify.retrieval.pooling import mean_pool_cols, mean_pool_rows

GRID = PatchGrid(n_x=4, n_y=3, offset=2, n_vectors=2 + 12 + 1)
DIM = 8


def _vectors() -> np.ndarray:
    rng = np.random.default_rng(0)
    v = rng.normal(size=(GRID.n_vectors, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_row_pooling_yields_one_vector_per_row_plus_specials():
    out = mean_pool_rows(_vectors(), GRID)
    assert out.shape == (GRID.n_y + GRID.n_special, DIM)


def test_col_pooling_yields_one_vector_per_column_plus_specials():
    out = mean_pool_cols(_vectors(), GRID)
    assert out.shape == (GRID.n_x + GRID.n_special, DIM)


def test_row_pool_is_the_mean_of_that_row():
    v = _vectors()
    out = mean_pool_rows(v, GRID)
    # Row 1 is patches 4..7, which are sequence indices 6..9.
    expected = v[GRID.offset + 4 : GRID.offset + 8].mean(axis=0)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(out[1], expected, atol=1e-6)


def test_col_pool_is_the_mean_of_that_column():
    v = _vectors()
    out = mean_pool_cols(v, GRID)
    # Column 2 is patches 2, 6, 10 -> sequence indices 4, 8, 12.
    expected = v[[GRID.offset + 2, GRID.offset + 6, GRID.offset + 10]].mean(axis=0)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(out[2], expected, atol=1e-6)


def test_special_tokens_are_carried_through_unpooled():
    v = _vectors()
    out = mean_pool_rows(v, GRID)
    # Specials are the vectors outside the image block, in sequence order.
    specials = np.concatenate([v[: GRID.offset], v[GRID.offset + GRID.n_image_patches :]])
    assert np.allclose(out[GRID.n_y :], specials, atol=1e-6)


def test_output_is_unit_normalized():
    out = mean_pool_rows(_vectors(), GRID)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_rejects_wrong_vector_count():
    with pytest.raises(ValueError, match="expected"):
        mean_pool_rows(np.zeros((5, DIM), dtype=np.float32), GRID)
