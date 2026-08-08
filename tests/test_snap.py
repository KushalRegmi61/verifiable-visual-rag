"""Patch-to-candidate weighting and two-stage selection."""

import numpy as np
import pytest

from visual_verify.grounding.snap import patch_weights
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
