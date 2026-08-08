"""Patch-to-candidate weighting and two-stage selection."""

import numpy as np
import pytest

from visual_verify.grounding.snap import patch_weights, rank_candidates, score_candidate
from visual_verify.ingest.boxes import BoxRecord
from visual_verify.retrieval.geometry import PatchGrid


def make_grid(n_x=4, n_y=4, offset=2, n_suffix=1):
    return PatchGrid(n_x=n_x, n_y=n_y, offset=offset, n_vectors=offset + n_x * n_y + n_suffix)


def test_full_page_box_weights_every_patch_fully():
    grid = make_grid()
    w = patch_weights(grid, (0.0, 0.0, 1.0, 1.0))

    assert w.shape == (grid.n_image_patches,)
    assert np.allclose(w, 1.0)


def test_a_box_over_one_cell_weights_only_that_cell():
    grid = make_grid()  # 4x4, each cell 0.25 x 0.25
    w = patch_weights(grid, (0.25, 0.5, 0.5, 0.75))

    # col 1, row 2  ->  patch index row * n_x + col = 2 * 4 + 1 = 9
    assert w[9] == pytest.approx(1.0)
    assert np.count_nonzero(w) == 1


def test_a_box_thinner_than_a_patch_still_gets_weight():
    """The case that breaks centre-based selection.

    A real line box is 0.0142 tall against a 0.0312 patch cell, so it never
    contains a patch centre. Requiring one would score every line zero.
    """
    grid = make_grid()
    w = patch_weights(grid, (0.30, 0.30, 0.70, 0.31))

    assert np.count_nonzero(w) > 0
    assert w.max() < 1.0, "a sliver must not weight a full cell"


def test_weights_are_the_covered_area_fraction():
    grid = make_grid()  # cells are 0.25 x 0.25
    # Exactly half of cell (0,0) horizontally, all of it vertically.
    w = patch_weights(grid, (0.0, 0.0, 0.125, 0.25))

    assert w[0] == pytest.approx(0.5)


def test_weights_align_with_patch_bbox():
    """patch_weights and PatchGrid.patch_bbox must agree on index order.

    A row/column transposition here reproduces the S2 patch-grid bug: every
    box lands somewhere plausible and nothing raises.
    """
    grid = make_grid(n_x=4, n_y=3)
    for idx in range(grid.n_image_patches):
        w = patch_weights(grid, grid.patch_bbox(idx))
        assert int(w.argmax()) == idx, f"patch {idx} weighted the wrong cell"
        assert w[idx] == pytest.approx(1.0)


def test_a_degenerate_box_raises():
    grid = make_grid()
    with pytest.raises(ValueError, match="positive area"):
        patch_weights(grid, (0.5, 0.5, 0.5, 0.5))


def test_a_real_line_box_covers_no_patch_centre_but_still_scores():
    """The measurement the whole weighting scheme exists for.

    On the real 23x32 grid a line of text is 0.0142 tall against a 0.0312
    patch cell. Placed inside one row it contains no patch centre at all, so
    centre containment would score it zero and stage 2 would rank nothing
    while appearing to work. Area weighting gives it 0.0142 * 32 = 0.4544 of
    each cell it spans.
    """
    grid = PatchGrid(n_x=23, n_y=32, offset=4, n_vectors=4 + 23 * 32 + 7)
    line_height = 0.0142
    y0 = 10 / 32  # start on a row boundary, so the line
    y1 = y0 + line_height  # lies entirely within row 10
    bbox = (0.1, y0, 0.6, y1)

    w = patch_weights(grid, bbox)

    patch_boxes = [grid.patch_bbox(i) for i in range(grid.n_image_patches)]

    # Not one patch centre falls inside the box.
    contains_a_centre = any(
        bbox[0] <= (b[0] + b[2]) / 2 <= bbox[2] and bbox[1] <= (b[1] + b[3]) / 2 <= bbox[3]
        for b in patch_boxes
    )
    assert not contains_a_centre, "the whole point of area weighting is that no centre falls here"

    inside_a_patch = any(
        b[0] <= bbox[0] and bbox[2] <= b[2] and b[1] <= y0 and y1 <= b[3] for b in patch_boxes
    )
    assert not inside_a_patch, "the line must span several patches, not sit in one"

    assert np.count_nonzero(w) > 0
    assert w.max() == pytest.approx(line_height * grid.n_y, abs=1e-9)
    assert w.max() < 0.5, "a line covers under half a patch row, as measured"


def box(x0, y0, x1, y1, text="w", kind="line", block_no=0, line_no=0):
    return BoxRecord(
        kind=kind,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        block_no=block_no,
        line_no=line_no,
        word_no=-1,
    )


def hot_map(grid, hot_index, hot=1.0, cold=0.1):
    r = np.full(grid.n_image_patches, cold)
    r[hot_index] = hot
    return r


def test_score_is_the_weighted_mean_over_covered_patches():
    grid = make_grid()  # 4x4
    r = hot_map(grid, 9)  # col 1, row 2
    assert score_candidate(r, grid, (0.25, 0.5, 0.5, 0.75)) == pytest.approx(1.0)


def test_mean_does_not_reward_a_larger_box():
    """Sum is monotone in area and would hand every contest to the page box."""
    grid = make_grid()
    r = hot_map(grid, 9)

    tight = score_candidate(r, grid, (0.25, 0.5, 0.5, 0.75))
    whole = score_candidate(r, grid, (0.0, 0.0, 1.0, 1.0))

    assert tight > whole


def test_sum_reduce_is_available_as_a_control():
    grid = make_grid()
    r = hot_map(grid, 9)

    tight = score_candidate(r, grid, (0.25, 0.5, 0.5, 0.75), reduce="sum")
    whole = score_candidate(r, grid, (0.0, 0.0, 1.0, 1.0), reduce="sum")

    assert whole > tight, "sum must show the area bias the bake-off measures"


def test_rank_candidates_orders_by_score_descending():
    grid = make_grid()
    r = hot_map(grid, 9)
    cold = box(0.0, 0.0, 0.25, 0.25, text="cold")
    warm = box(0.25, 0.5, 0.5, 0.75, text="warm")

    ranked = rank_candidates(r, grid, [cold, warm])

    assert [b.text for b, _ in ranked] == ["warm", "cold"]
    assert ranked[0][1] > ranked[1][1]


def test_rank_candidates_is_deterministic_on_ties():
    """Ties break by input order, never by set or dict iteration."""
    grid = make_grid()
    r = np.full(grid.n_image_patches, 0.5)
    a = box(0.0, 0.0, 0.25, 0.25, text="a")
    b = box(0.5, 0.5, 0.75, 0.75, text="b")

    first = [t.text for t, _ in rank_candidates(r, grid, [a, b])]
    second = [t.text for t, _ in rank_candidates(r, grid, [a, b])]

    assert first == second == ["a", "b"]


def test_rank_candidates_skips_a_degenerate_box_without_raising():
    """Derived boxes are trusted, but a zero-area one must not kill the query."""
    grid = make_grid()
    r = hot_map(grid, 9)
    good = box(0.25, 0.5, 0.5, 0.75, text="good")
    bad = box(0.5, 0.5, 0.5, 0.5, text="bad")

    ranked = rank_candidates(r, grid, [bad, good])

    assert [t.text for t, _ in ranked] == ["good"]
