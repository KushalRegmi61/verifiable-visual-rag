"""Normalized boxes must not depend on render DPI.

Every downstream consumer relies on this: S4 maps a patch grid onto boxes, and
the frontend scales boxes to an arbitrary viewport. If normalization drifted
with DPI, both would be silently wrong.
"""

import fitz
import pytest

from conftest import TEXT_ORIGIN
from visual_verify.ingest.boxes import extract_boxes, word_boxes
from visual_verify.ingest.render import render_page


def test_boxes_identical_across_dpi(born_digital_pdf, tmp_path):
    """Repeated extraction is deterministic and unaffected by an intervening render.

    Note this is weaker than the file name suggests: extract_boxes never receives
    dpi, so DPI-independence is currently structural rather than behavioural.
    test_box_pixel_position_scales_exactly_with_dpi is what pins the actual
    scale relationship.
    """
    doc = fitz.open(born_digital_pdf)
    page = doc[0]

    render_page(page, tmp_path / "a.png", dpi=72)
    at_72 = extract_boxes(page)
    render_page(page, tmp_path / "b.png", dpi=150)
    at_150 = extract_boxes(page)
    doc.close()

    assert at_72 == at_150


def test_boxes_identical_across_dpi_when_rotated(rotated_pdf, tmp_path):
    """Repeated extraction on a rotated page is deterministic across renders.

    As above, this pins determinism, not DPI-independence: extract_boxes never
    receives dpi, so it structurally cannot vary with the render in between.
    test_box_pixel_position_scales_exactly_with_dpi pins the actual scale
    relationship; test_box_lands_on_ink_at_both_dpis pins it against ground
    truth pixels.
    """
    doc = fitz.open(rotated_pdf)
    page = doc[0]

    render_page(page, tmp_path / "a.png", dpi=72)
    at_72 = extract_boxes(page)
    render_page(page, tmp_path / "b.png", dpi=200)
    at_200 = extract_boxes(page)
    doc.close()

    assert at_72 == at_200


def test_normalized_box_maps_back_to_pixels(born_digital_pdf, tmp_path):
    """A normalized box scaled by the rendered size must land inside the image."""
    doc = fitz.open(born_digital_pdf)
    page = doc[0]
    rendered = render_page(page, tmp_path / "p.png", dpi=150)
    boxes = extract_boxes(page)
    doc.close()

    first = next(b for b in boxes if b.kind == "word")
    px0 = first.x0 * rendered.width_px
    py1 = first.y1 * rendered.height_px

    assert 0 <= px0 <= rendered.width_px
    assert 0 <= py1 <= rendered.height_px
    # Points scale to pixels by dpi/72, derived from the fixture constant
    # rather than a copied literal.
    assert px0 == pytest.approx(TEXT_ORIGIN[0] * 150 / 72, abs=1.0)


def test_box_pixel_position_scales_exactly_with_dpi(born_digital_pdf, tmp_path):
    """The same box, converted to pixels, must scale linearly with DPI.

    This is the assertion the cross-DPI equality tests cannot make: it ties the
    normalized coordinate to two DIFFERENT rendered sizes, so normalizing
    against pixmap dimensions instead of page.rect would break it.
    """
    doc = fitz.open(born_digital_pdf)
    page = doc[0]

    low = render_page(page, tmp_path / "low.png", dpi=72)
    box_low = word_boxes(extract_boxes(page))[0]
    high = render_page(page, tmp_path / "high.png", dpi=216)
    box_high = word_boxes(extract_boxes(page))[0]
    doc.close()

    ratio = 216 / 72
    assert high.width_px == low.width_px * ratio
    assert box_high.x0 * high.width_px == pytest.approx(box_low.x0 * low.width_px * ratio)
    assert box_high.y1 * high.height_px == pytest.approx(box_low.y1 * low.height_px * ratio)


def test_box_lands_on_ink_at_both_dpis(born_digital_pdf, tmp_path):
    """Ground truth: the box must crop actual glyph pixels at any render size."""
    from PIL import Image

    doc = fitz.open(born_digital_pdf)
    page = doc[0]

    for dpi in (72, 200):
        rendered = render_page(page, tmp_path / f"p{dpi}.png", dpi=dpi)
        box = word_boxes(extract_boxes(page))[0]

        img = Image.open(rendered.path).convert("L")
        crop = img.crop(
            (
                int(box.x0 * rendered.width_px),
                int(box.y0 * rendered.height_px),
                int(box.x1 * rendered.width_px),
                int(box.y1 * rendered.height_px),
            )
        )
        assert crop.getbbox() is not None, f"empty crop at {dpi} dpi"
        assert min(crop.getdata()) < 128, f"no ink in box at {dpi} dpi"

    doc.close()
