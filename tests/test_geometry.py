import pytest

from visual_verify.retrieval.geometry import PatchGrid

# The real grid measured for this project's A4 pages at 150 dpi.
A4 = PatchGrid(n_x=23, n_y=32, offset=4, n_vectors=747)


def test_counts():
    assert A4.n_image_patches == 736
    assert A4.n_special == 11


def test_first_patch_is_top_left():
    x0, y0, x1, y1 = A4.patch_bbox(0)
    assert (x0, y0) == (0.0, 0.0)
    assert x1 == pytest.approx(1 / 23)
    assert y1 == pytest.approx(1 / 32)


def test_last_patch_is_bottom_right():
    x0, y0, x1, y1 = A4.patch_bbox(735)
    assert x1 == pytest.approx(1.0)
    assert y1 == pytest.approx(1.0)
    assert x0 == pytest.approx(22 / 23)
    assert y0 == pytest.approx(31 / 32)


def test_row_major_order():
    """Patch n_x is the START of the second row, not the second column."""
    _, y0_first, _, _ = A4.patch_bbox(0)
    _, y0_next, _, _ = A4.patch_bbox(23)
    assert y0_next > y0_first
    x0_a, _, _, _ = A4.patch_bbox(1)
    assert x0_a == pytest.approx(1 / 23)


def test_sequence_index_maps_through_offset():
    """Sequence index 4 is image patch 0. Off-by-four shifts every box."""
    assert A4.seq_to_patch(4) == 0
    assert A4.seq_to_patch(739) == 735


def test_special_tokens_have_no_region():
    for seq_idx in (0, 3, 740, 746):
        assert not A4.is_image_token(seq_idx)
        with pytest.raises(ValueError):
            A4.seq_to_patch(seq_idx)


def test_image_tokens_are_recognised():
    assert A4.is_image_token(4)
    assert A4.is_image_token(739)


def test_landscape_grid_is_not_assumed_square():
    """Grid is aspect-ratio dependent; nothing may assume n_x == n_y."""
    wide = PatchGrid(n_x=32, n_y=18, offset=4, n_vectors=32 * 18 + 11)
    assert wide.n_image_patches == 576
    x0, _, x1, _ = wide.patch_bbox(0)
    assert x1 == pytest.approx(1 / 32)


def test_rejects_inconsistent_vector_count():
    with pytest.raises(ValueError, match="inconsistent"):
        PatchGrid(n_x=23, n_y=32, offset=4, n_vectors=999)


def test_rejects_nonpositive_dims():
    with pytest.raises(ValueError):
        PatchGrid(n_x=0, n_y=32, offset=4, n_vectors=11)


def test_rejects_offset_past_the_vector_count():
    """n_special alone cannot catch this: 747 - 736 = 11 looks fine, but the
    patch block starting at 500 runs off the end of a 747-vector sequence."""
    with pytest.raises(ValueError, match="extend past"):
        PatchGrid(n_x=23, n_y=32, offset=500, n_vectors=747)
