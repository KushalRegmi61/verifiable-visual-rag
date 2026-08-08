"""Checks on the evidence checkers themselves.

The most important test here is test_ink_cannot_discriminate_the_wrong_box. It
asserts a WEAKNESS rather than a capability, so that nobody later reaches for
has_ink to validate a grounding result and gets a green suite for it.
"""

from dataclasses import dataclass

import fitz
import pytest
from PIL import Image

from conftest import FIRST_LINE, SECOND_LINE
from visual_verify.evidence import (
    covers_text,
    has_ink,
    ink_ratio,
    iou,
    overlap_fraction,
    shift,
    text_in_bbox,
    to_pixels,
)
from visual_verify.ingest.boxes import extract_boxes, word_boxes
from visual_verify.ingest.render import render_page


@dataclass(frozen=True)
class FakeBox:
    """Minimum surface text_in_bbox duck-types against."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@pytest.fixture
def rendered(born_digital_pdf, tmp_path):
    doc = fitz.open(born_digital_pdf)
    page = doc[0]
    out = render_page(page, tmp_path / "p.png", dpi=150)
    boxes = word_boxes(extract_boxes(page))
    doc.close()
    return Image.open(out.path), boxes


def test_to_pixels_truncates_inward():
    # 0.5 * 101 = 50.5; truncating keeps the crop inside the box rather than
    # reaching outward into a neighbouring glyph.
    assert to_pixels((0.5, 0.5, 1.0, 1.0), 101, 101) == (50, 50, 101, 101)


def test_to_pixels_scales_with_render_size():
    small = to_pixels((0.25, 0.5, 0.75, 1.0), 100, 200)
    large = to_pixels((0.25, 0.5, 0.75, 1.0), 200, 400)
    assert small == (25, 100, 75, 200)
    assert large == tuple(c * 2 for c in small)


def test_has_ink_true_on_a_word_and_false_on_the_margin(rendered):
    img, boxes = rendered
    assert has_ink(img, (boxes[0].x0, boxes[0].y0, boxes[0].x1, boxes[0].y1))
    # The fixture writes two short lines near the top; the bottom third is blank.
    assert not has_ink(img, (0.05, 0.8, 0.95, 0.95))


def test_has_ink_false_on_a_degenerate_crop(rendered):
    img, _ = rendered
    # Sub-pixel box: real after normalization, zero-area once truncated. Must
    # return False rather than raise, since getextrema() on an empty crop does.
    assert not has_ink(img, (0.5, 0.5, 0.5001, 0.5001))


def test_ink_ratio_is_higher_for_a_tight_box_than_a_page_wide_one(rendered):
    img, boxes = rendered
    b = boxes[0]
    tight = ink_ratio(img, (b.x0, b.y0, b.x1, b.y1))
    loose = ink_ratio(img, (0.0, 0.0, 1.0, 1.0))
    assert tight > loose > 0.0


def test_shift_stays_on_the_page_and_keeps_its_size():
    x0, y0, x1, y1 = shift((0.9, 0.9, 0.95, 0.95), dx=0.5, dy=0.5)
    assert (x0, y0, x1, y1) == pytest.approx((0.95, 0.95, 1.0, 1.0))
    assert x1 <= 1.0 and y1 <= 1.0


def test_shifted_boxes_mostly_miss_the_ink(rendered):
    """The negative control that makes the alignment assertion mean something."""
    img, boxes = rendered
    true_hits = sum(has_ink(img, (b.x0, b.y0, b.x1, b.y1)) for b in boxes)
    moved = sum(has_ink(img, shift((b.x0, b.y0, b.x1, b.y1), dy=0.25)) for b in boxes)
    assert true_hits == len(boxes)
    assert moved < len(boxes) // 2


def test_overlap_fraction_is_asymmetric():
    small, big = (0.4, 0.4, 0.5, 0.5), (0.0, 0.0, 1.0, 1.0)
    assert overlap_fraction(small, big) == pytest.approx(1.0)
    assert overlap_fraction(big, small) == pytest.approx(0.01)


def test_iou_is_symmetric_and_zero_when_disjoint():
    a, b = (0.0, 0.0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75)
    assert iou(a, b) == pytest.approx(iou(b, a))
    assert 0.0 < iou(a, b) < 1.0
    assert iou(a, (0.9, 0.9, 1.0, 1.0)) == 0.0


def test_text_in_bbox_reads_in_layout_order():
    boxes = [
        FakeBox(0.5, 0.1, 0.6, 0.2, "second"),
        FakeBox(0.1, 0.1, 0.2, 0.2, "first"),
        FakeBox(0.1, 0.5, 0.2, 0.6, "later"),
    ]
    assert text_in_bbox(boxes, (0.0, 0.0, 1.0, 1.0)) == "first second later"


def test_text_in_bbox_excludes_a_barely_clipped_word():
    boxes = [FakeBox(0.0, 0.0, 1.0, 0.1, "wide")]
    # Only a tenth of the word falls inside, so its text must not be claimed as
    # covered. Otherwise a region that merely grazes a word inherits its string.
    assert text_in_bbox(boxes, (0.0, 0.0, 0.1, 0.1)) == ""
    assert text_in_bbox(boxes, (0.0, 0.0, 0.6, 0.1)) == "wide"


def test_covers_text_ignores_case_and_spacing():
    boxes = [FakeBox(0.1, 0.1, 0.2, 0.2, "Revenue"), FakeBox(0.2, 0.1, 0.3, 0.2, "grew")]
    region = (0.0, 0.0, 1.0, 1.0)
    assert covers_text(boxes, region, "revenue   GREW")
    assert not covers_text(boxes, region, "revenue fell")


def test_covers_text_selects_the_right_line(born_digital_pdf, tmp_path):
    """The assertion S4 needs: which line was chosen, not whether ink exists."""
    doc = fitz.open(born_digital_pdf)
    page = doc[0]
    boxes = word_boxes(extract_boxes(page))
    doc.close()

    first = [b for b in boxes if b.text in FIRST_LINE.split()]
    top = min(b.y0 for b in first), max(b.y1 for b in first)
    region = (0.0, top[0], 1.0, top[1])

    assert covers_text(boxes, region, FIRST_LINE)
    assert not covers_text(boxes, region, SECOND_LINE)


def test_ink_cannot_discriminate_the_wrong_box(rendered):
    """Pins the weakness, so has_ink is never mistaken for a grounding check.

    Every candidate box grounding chooses among is a word box, and every word
    box has ink by construction. Measured on proposal.pdf page 3 at 150 dpi:
    435/435. So a selector returning a random candidate passes has_ink every
    time while being correct once in 435.

    If this test ever fails, the fixture stopped containing text; it does not
    mean has_ink became discriminating.
    """
    img, boxes = rendered
    assert len(boxes) > 4
    assert all(has_ink(img, (b.x0, b.y0, b.x1, b.y1)) for b in boxes)

    # The same set, judged by identity instead of presence: exactly one matches.
    target = boxes[0]
    matched = [
        b for b in boxes if covers_text(boxes, (b.x0, b.y0, b.x1, b.y1), target.text)
    ]
    assert len(matched) < len(boxes), "covers_text must reject boxes has_ink accepts"
