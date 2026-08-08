"""The text path, and ground() routing."""

import numpy as np
import pytest

from visual_verify.derive import block_boxes, line_boxes
from visual_verify.grounding import GroundingError, ground
from visual_verify.grounding.text_span import text_regions
from visual_verify.ingest.boxes import BoxRecord
from visual_verify.retrieval.geometry import PatchGrid


def word(x0, y0, x1, y1, text, block_no=0, line_no=0, word_no=0):
    return BoxRecord(
        kind="word",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        block_no=block_no,
        line_no=line_no,
        word_no=word_no,
    )


def two_line_page():
    """'Revenue grew 42 percent' over 'Margins held steady'."""
    first = ["Revenue", "grew", "42", "percent"]
    second = ["Margins", "held", "steady"]
    boxes = []
    for i, t in enumerate(first):
        boxes.append(word(0.1 + i * 0.15, 0.10, 0.22 + i * 0.15, 0.16, t, line_no=0, word_no=i))
    for i, t in enumerate(second):
        boxes.append(word(0.1 + i * 0.15, 0.30, 0.22 + i * 0.15, 0.36, t, line_no=1, word_no=i))
    return boxes


def test_text_regions_finds_an_exact_phrase():
    regions = text_regions("grew 42", two_line_page(), page=3)

    assert len(regions) == 1
    assert regions[0].modality == "text"
    assert regions[0].page == 3
    assert regions[0].text == "grew 42"


def test_text_regions_is_empty_for_an_absent_phrase():
    assert text_regions("revenue fell", two_line_page(), page=0) == []


def test_a_wrapped_phrase_returns_one_rect_per_line_not_a_union():
    """A union over a wrapped match sweeps in every word between the halves.

    On this fixture that is 5.7x the true ink area. The grounding layer must
    pass derive.span_boxes's split through untouched.

    Checking only the rect count and that each is short is not enough: a bug
    that returned the whole first line and the whole second line (instead of
    just the matched words within them) would also produce two short rects.
    The fixture has known coordinates, so pin the exact matched-word geometry.
    """
    regions = text_regions("percent Margins", two_line_page(), page=0)

    assert len(regions) == 2
    assert regions[0].bbox == pytest.approx((0.55, 0.10, 0.67, 0.16))
    assert regions[1].bbox == pytest.approx((0.10, 0.30, 0.22, 0.36))


def test_text_regions_score_is_exact():
    regions = text_regions("Revenue", two_line_page(), page=0)

    assert regions[0].score == 1.0


def make_grid(n_x=4, n_y=4, offset=2, n_suffix=1):
    return PatchGrid(n_x=n_x, n_y=n_y, offset=offset, n_vectors=offset + n_x * n_y + n_suffix)


def planted_vectors(grid, hot_patch, dim=8):
    """Page and query vectors whose MaxSim maximum is `hot_patch`."""
    rng = np.random.default_rng(11)
    page = rng.normal(size=(grid.n_vectors, dim))
    page /= np.linalg.norm(page, axis=1, keepdims=True)
    query = rng.normal(size=(3, dim))
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    page[grid.offset + hot_patch] = query[0]
    return page, query


def tied_vectors(grid, dim=8):
    """Page and query vectors that make every image patch equally relevant.

    Every patch shares one vector, identical to the query's own first token,
    so score_candidate's mean is the same for every line and every block: the
    top two lines tie inside the margin and snap_to_box must fall back to the
    block instead of guessing a line the grid cannot distinguish.
    """
    rng = np.random.default_rng(7)
    page = rng.normal(size=(grid.n_vectors, dim))
    page /= np.linalg.norm(page, axis=1, keepdims=True)
    query = rng.normal(size=(3, dim))
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    page[grid.offset : grid.offset + grid.n_image_patches] = query[0]
    return page, query


def test_an_exact_match_short_circuits_without_touching_vectors():
    """Text wins by default, and the visual path is not even reachable here."""
    regions = ground("grew 42", two_line_page(), page=0)

    assert len(regions) == 1
    assert regions[0].modality == "text"


def test_force_visual_bypasses_the_text_match():
    """proposal.tex line 440 requires measuring the visual path on
    text-locatable questions, which is impossible without this."""
    grid = make_grid()
    page_v, query_v = planted_vectors(grid, hot_patch=0)
    boxes = two_line_page()

    regions = ground(
        "grew 42",
        boxes,
        page=0,
        page_vectors=page_v,
        query_vectors=query_v,
        grid=grid,
        force="visual",
    )

    assert len(regions) == 1
    assert regions[0].modality == "visual"
    allowed = {(b.x0, b.y0, b.x1, b.y1) for b in line_boxes(boxes) + block_boxes(boxes)}
    assert regions[0].bbox in allowed


def test_no_text_match_falls_back_to_the_visual_path():
    grid = make_grid()
    page_v, query_v = planted_vectors(grid, hot_patch=0)

    regions = ground(
        "not on this page at all",
        two_line_page(),
        page=0,
        page_vectors=page_v,
        query_vectors=query_v,
        grid=grid,
    )

    assert len(regions) == 1
    assert regions[0].modality == "visual"


def test_a_page_with_no_candidates_returns_nothing():
    """Absence of evidence, not weakness of evidence. S4 never abstains."""
    grid = make_grid()
    page_v, query_v = planted_vectors(grid, hot_patch=0)

    regions = ground(
        "anything",
        [],
        page=0,
        page_vectors=page_v,
        query_vectors=query_v,
        grid=grid,
    )

    assert regions == []


def test_the_visual_path_without_vectors_raises():
    """Silently returning nothing would look identical to 'no evidence here'."""
    with pytest.raises(GroundingError, match="vectors"):
        ground("not on this page at all", two_line_page(), page=0)


def test_ground_is_deterministic():
    """Result does not depend on the order candidate boxes arrive in.

    derive.py sorts by (block_no, line_no, word_no) before grouping, so a
    reversed input list must rank identically. Calling ground() twice with
    the same list in the same order cannot fail for any nondeterminism this
    pipeline could produce and would be a tautology.
    """
    grid = make_grid()
    page_v, query_v = planted_vectors(grid, hot_patch=0)
    boxes = two_line_page()
    args = dict(page=0, page_vectors=page_v, query_vectors=query_v, grid=grid)

    first = ground("absent phrase", boxes, **args)
    second = ground("absent phrase", list(reversed(boxes)), **args)

    assert [r.bbox for r in first] == [r.bbox for r in second]


def test_a_visual_region_is_always_an_existing_candidate_box():
    """Snap-to-box, stated as an assertion: never drawn from the heatmap."""
    grid = make_grid()
    page_v, query_v = planted_vectors(grid, hot_patch=0)
    boxes = two_line_page()

    regions = ground(
        "absent phrase",
        boxes,
        page=0,
        page_vectors=page_v,
        query_vectors=query_v,
        grid=grid,
    )

    assert regions[0].resolution == "line"
    allowed = {(b.x0, b.y0, b.x1, b.y1) for b in line_boxes(boxes) + block_boxes(boxes)}
    assert regions[0].bbox in allowed


def test_a_visual_region_at_block_resolution_is_also_an_existing_candidate_box():
    """Same invariant as above, pinned on the other live return site.

    A future change that clipped or intersected a rectangle "for safety"
    would still pass the line-resolution test above but fail here, since a
    clipped block union is not one of derive.py's candidate boxes.
    """
    grid = make_grid()
    page_v, query_v = tied_vectors(grid)
    boxes = two_line_page()

    regions = ground(
        "absent phrase",
        boxes,
        page=0,
        page_vectors=page_v,
        query_vectors=query_v,
        grid=grid,
    )

    assert regions[0].resolution == "block"
    allowed = {(b.x0, b.y0, b.x1, b.y1) for b in line_boxes(boxes) + block_boxes(boxes)}
    assert regions[0].bbox in allowed


def test_a_block_fallback_region_says_so():
    """A coarse region must be distinguishable from a confident line hit."""
    grid = make_grid()
    page_v, query_v = planted_vectors(grid, hot_patch=0)
    boxes = two_line_page()

    regions = ground(
        "absent phrase",
        boxes,
        page=0,
        page_vectors=page_v,
        query_vectors=query_v,
        grid=grid,
    )
    assert regions[0].resolution == "line"

    text_only = ground("grew 42", boxes, page=0)
    assert text_only[0].resolution is None, "the text path never snaps"

    tied_page_v, tied_query_v = tied_vectors(grid)
    forced_block = ground(
        "absent phrase",
        boxes,
        page=0,
        page_vectors=tied_page_v,
        query_vectors=tied_query_v,
        grid=grid,
    )
    assert forced_block[0].resolution == "block"
