"""Normalized boxes must not depend on render DPI.

Every downstream consumer relies on this: S4 maps a patch grid onto boxes, and
the frontend scales boxes to an arbitrary viewport. If normalization drifted
with DPI, both would be silently wrong.
"""

import fitz
import pytest

from conftest import TEXT_ORIGIN
from visual_verify.ingest.boxes import extract_boxes
from visual_verify.ingest.render import render_page


def test_boxes_identical_across_dpi(born_digital_pdf, tmp_path):
    doc = fitz.open(born_digital_pdf)
    page = doc[0]

    render_page(page, tmp_path / "a.png", dpi=72)
    at_72 = extract_boxes(page)
    render_page(page, tmp_path / "b.png", dpi=150)
    at_150 = extract_boxes(page)
    doc.close()

    assert at_72 == at_150


def test_boxes_identical_across_dpi_when_rotated(rotated_pdf, tmp_path):
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
