"""Patch-to-candidate weighting and two-stage selection."""

import fitz
import numpy as np
import pytest
from PIL import Image

from visual_verify.derive import block_boxes, line_boxes
from visual_verify.evidence import has_ink, ink_ratio, overlap_fraction, shift
from visual_verify.grounding.snap import (
    patch_weights,
    rank_candidates,
    score_candidate,
    snap_to_box,
)
from visual_verify.ingest.boxes import BoxRecord, extract_boxes, word_boxes
from visual_verify.ingest.render import render_page
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


def test_a_box_entirely_off_the_page_scores_zero_rather_than_dividing_by_zero():
    """Reachable, despite passing the degeneracy check.

    A box outside [0,1] has positive area, so patch_weights accepts it, but
    every cell clips to zero coverage. Without the guard this divides by a
    zero total and returns nan, which then wins an argmax the way NaN always
    does.
    """
    grid = make_grid()
    r = hot_map(grid, 9)

    assert score_candidate(r, grid, (1.5, 1.5, 2.0, 2.0)) == 0.0
    assert score_candidate(r, grid, (1.5, 1.5, 2.0, 2.0), reduce="sum") == 0.0


def test_an_off_page_box_is_ranked_last_rather_than_dropped():
    """Deliberately different from the degenerate case.

    A zero-area box cannot be scored at all, so it is dropped. An off-page box
    can be scored, and scores zero, so it stays and simply loses. Dropping it
    too would make a caller's candidate list silently shrink for two unrelated
    reasons.
    """
    grid = make_grid()
    r = hot_map(grid, 9)
    good = box(0.25, 0.5, 0.5, 0.75, text="good")
    off_page = box(1.5, 1.5, 2.0, 2.0, text="off")

    ranked = rank_candidates(r, grid, [off_page, good])

    assert [t.text for t, _ in ranked] == ["good", "off"]
    assert ranked[1][1] == 0.0


def test_rank_candidates_is_deterministic_on_ties_with_more_than_two():
    """A two-element list is trivially stable; this would catch an unstable sort.

    Two ties (hot patches) interleaved with two distinct scores (cold patches):
    higher scores must still sort first, and each tied group must keep its
    input order.
    """
    grid = make_grid()
    r = hot_map(grid, 9)  # patch (col 1, row 2) is hot, everything else cold
    tied_1 = box(0.25, 0.5, 0.5, 0.75, text="tied_1")  # covers the hot patch
    loser = box(0.0, 0.0, 0.25, 0.25, text="loser")  # cold patch
    tied_2 = box(0.25, 0.5, 0.5, 0.75, text="tied_2")  # covers the hot patch, same score
    other_loser = box(0.75, 0.75, 1.0, 1.0, text="other_loser")  # cold patch

    ranked = rank_candidates(r, grid, [tied_1, loser, tied_2, other_loser])

    assert [t.text for t, _ in ranked] == ["tied_1", "tied_2", "loser", "other_loser"]


def page_boxes():
    """Two blocks, two lines each, laid out top to bottom on a 4x4 grid page."""
    return [
        box(0.05, 0.05, 0.45, 0.20, text="alpha one", block_no=0, line_no=0, kind="word"),
        box(0.05, 0.25, 0.45, 0.40, text="alpha two", block_no=0, line_no=1, kind="word"),
        box(0.05, 0.55, 0.45, 0.70, text="beta one", block_no=1, line_no=0, kind="word"),
        box(0.05, 0.75, 0.45, 0.90, text="beta two", block_no=1, line_no=1, kind="word"),
    ]


def test_snap_returns_a_line_when_one_line_clearly_wins():
    grid = make_grid()
    # Patch row 3 (y 0.75-1.0), col 0-1: the region holding "beta two".
    r = np.full(grid.n_image_patches, 0.05)
    r[12] = 1.0
    r[13] = 1.0

    sel = snap_to_box(r, grid, page_boxes())

    assert sel.resolution == "line"
    assert "beta two" in sel.box.text


def test_snap_falls_back_to_the_block_when_lines_are_indistinguishable():
    """Honest about its own resolution instead of guessing a line."""
    grid = make_grid()
    # Uniform heat over the whole lower half: block 1 wins, its lines tie.
    r = np.full(grid.n_image_patches, 0.05)
    r[8:16] = 1.0

    boxes = page_boxes()
    sel = snap_to_box(r, grid, boxes)

    assert sel.resolution == "block"
    assert "beta" in sel.box.text
    corners = (sel.box.x0, sel.box.y0, sel.box.x1, sel.box.y1)
    allowed = {(b.x0, b.y0, b.x1, b.y1) for b in line_boxes(boxes) + block_boxes(boxes)}
    assert corners in allowed


def test_snap_stays_inside_the_winning_block():
    """A stage-2 miss must still land in the right paragraph.

    This is why selection is two-stage rather than a flat ranking over all
    lines: a flat miss can land anywhere on the page.
    """
    grid = make_grid()
    r = np.full(grid.n_image_patches, 0.05)
    r[0] = 1.0  # top-left, inside block 0

    sel = snap_to_box(r, grid, page_boxes())

    assert "alpha" in sel.box.text
    assert "beta" not in sel.box.text


def test_snap_returns_none_when_there_are_no_candidates():
    grid = make_grid()
    r = np.full(grid.n_image_patches, 0.5)

    assert snap_to_box(r, grid, []) is None


def test_snap_never_invents_a_box():
    """The returned bbox must be one a derived candidate actually has."""
    grid = make_grid()
    r = np.full(grid.n_image_patches, 0.05)
    r[12] = 1.0

    boxes = page_boxes()
    sel = snap_to_box(r, grid, boxes)
    corners = (sel.box.x0, sel.box.y0, sel.box.x1, sel.box.y1)

    allowed = {(b.x0, b.y0, b.x1, b.y1) for b in line_boxes(boxes) + block_boxes(boxes)}
    assert corners in allowed


def test_stage_two_never_leaves_the_winning_block():
    """The bounded-error property that justifies two stages at all.

    The hottest single line here belongs to the block that LOST stage 1. A flat
    ranking over every line would return it. Two-stage selection must not: a
    stage-2 mistake has to stay inside the correct paragraph.
    """
    grid = make_grid()
    boxes = page_boxes()
    r = np.full(grid.n_image_patches, 0.05)
    # Block 0 wins stage 1 on total warmth across both its rows...
    r[0:8] = 0.60
    # ...while one line in block 1 is the single hottest thing on the page.
    r[12] = 1.0
    r[13] = 1.0

    sel = snap_to_box(r, grid, boxes)

    assert "alpha" in sel.box.text, f"selection escaped the winning block: {sel.box.text!r}"
    assert "beta" not in sel.box.text


def test_near_zero_scores_are_ambiguous_rather_than_a_clear_win():
    """A relative margin over two near-zero scores measures nothing.

    Without an absolute floor on the denominator, 1e-18 against 0.9e-18 is a
    10 percent gap and resolves to a line, on an absolute difference of 1e-19.
    """
    grid = make_grid()
    r = np.full(grid.n_image_patches, 1e-18)
    r[12] = 1.1e-18

    sel = snap_to_box(r, grid, page_boxes())

    assert sel.resolution == "block"


def test_single_line_block_resolves_as_line():
    """No runner-up to be ambiguous against, so resolution is line outright."""
    grid = make_grid()
    boxes = [
        box(0.05, 0.05, 0.45, 0.20, text="solo", block_no=0, line_no=0, kind="word"),
        box(0.05, 0.55, 0.45, 0.90, text="other block", block_no=1, line_no=0, kind="word"),
    ]
    r = np.full(grid.n_image_patches, 0.05)
    r[0] = 1.0  # inside block 0, the single-line block

    sel = snap_to_box(r, grid, boxes)

    assert sel.resolution == "line"
    assert sel.box.text == "solo"


def test_patch_rectangles_land_on_the_page_ink(born_digital_pdf, tmp_path):
    """The heaviest-ink patch must sit on the text, and not next to it.

    Ink presence proves the coordinate transform, nothing more. It cannot
    prove a selection is correct, because every word box on a real page
    contains ink. See tests/test_evidence.py.
    """
    doc = fitz.open(born_digital_pdf)
    rendered = render_page(doc[0], tmp_path / "p.png", dpi=150)
    words = word_boxes(extract_boxes(doc[0]))
    doc.close()

    grid = PatchGrid(n_x=23, n_y=32, offset=4, n_vectors=4 + 23 * 32 + 7)
    img = Image.open(rendered.path)

    ratios = [ink_ratio(img, grid.patch_bbox(i)) for i in range(grid.n_image_patches)]
    heaviest = grid.patch_bbox(int(np.argmax(ratios)))

    assert has_ink(img, heaviest)
    # Control: the same rect displaced must miss, or the assertion proves
    # nothing on a page with ink scattered across it.
    assert not has_ink(img, shift(heaviest, dy=0.5))

    # Patch geometry and text-layer geometry must agree, since snap-to-box
    # ranks the second using the first. Nothing else in the suite checks that
    # coupling: the weighting tests only prove patch_weights and patch_bbox
    # agree with each other. An index bound cannot do this job here, because
    # this fixture's text sits near the top-left corner, so a row/col swap
    # satisfies any bound that the true row satisfies.
    text_region = (
        min(w.x0 for w in words),
        min(w.y0 for w in words),
        max(w.x1 for w in words),
        max(w.y1 for w in words),
    )
    assert overlap_fraction(heaviest, text_region) > 0.0, (
        f"heaviest-ink patch {heaviest} does not overlap the text at "
        f"{text_region}; patch coordinates disagree with the text layer"
    )


def test_line_resolution_selection_carries_the_lines_score_not_the_blocks():
    """Selection.score is what S5 thresholds on and S7 reports.

    Constructed so the winning line's own score and the winning block's score
    are measurably different: a caller returning the block's score here would
    still pass every other snap test (the box and resolution are unaffected),
    since only the reported score would be wrong.
    """
    grid = make_grid()
    r = np.full(grid.n_image_patches, 0.05)
    r[12] = 1.0  # inside "beta two", the winning line
    r[13] = 1.0

    sel = snap_to_box(r, grid, page_boxes())

    assert sel.resolution == "line"
    expected_line_score = score_candidate(r, grid, (sel.box.x0, sel.box.y0, sel.box.x1, sel.box.y1))
    block = next(b for b in block_boxes(page_boxes()) if b.block_no == sel.box.block_no)
    block_score = score_candidate(r, grid, (block.x0, block.y0, block.x1, block.y1))

    assert block_score != pytest.approx(expected_line_score), "the fixture must make them differ"
    assert sel.score == pytest.approx(expected_line_score)
    assert sel.score != pytest.approx(block_score)


def test_block_resolution_selection_carries_the_blocks_score():
    """The mirror of the line-resolution case, on the other live return path."""
    grid = make_grid()
    r = np.full(grid.n_image_patches, 0.05)
    r[8:16] = 1.0  # uniform heat over block 1: lines tie, block wins

    boxes = page_boxes()
    sel = snap_to_box(r, grid, boxes)

    assert sel.resolution == "block"
    expected_block_score = score_candidate(
        r, grid, (sel.box.x0, sel.box.y0, sel.box.x1, sel.box.y1)
    )
    assert sel.score == pytest.approx(expected_block_score)


def test_snap_to_box_rejects_a_relevance_shape_mismatch():
    """Without this, a scalar or under-sized array broadcasts silently and
    every candidate ties, returning a confident selection that means nothing.
    heatmap.py validates its inputs this hard; snap_to_box took relevance on
    faith, undefended in the parallel module.
    """
    grid = make_grid()

    with pytest.raises(ValueError, match="shape"):
        snap_to_box(np.array([0.5]), grid, page_boxes())


def test_snap_to_box_rejects_a_nan_in_relevance():
    """A hand-built map containing a NaN must raise, not return a confident
    line selection with score=nan: NaN compares greater than every real
    score, so it would silently win any comparison downstream.
    """
    grid = make_grid()
    r = np.full(grid.n_image_patches, 0.05)
    r[12] = np.nan

    with pytest.raises(ValueError, match="NaN or Inf"):
        snap_to_box(r, grid, page_boxes())


def test_patch_bbox_tiles_the_page_without_gaps_or_overlap():
    grid = make_grid(n_x=4, n_y=3)
    total = sum(
        (b[2] - b[0]) * (b[3] - b[1])
        for b in (grid.patch_bbox(i) for i in range(grid.n_image_patches))
    )
    assert total == pytest.approx(1.0)
