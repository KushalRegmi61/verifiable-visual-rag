import fitz
import pytest

from conftest import PAGE_W, TEXT_ORIGIN
from visual_verify.ingest.boxes import BoxRecord, extract_boxes


def _words(boxes: list[BoxRecord]) -> list[BoxRecord]:
    return [b for b in boxes if b.kind == "word"]


def test_extracts_every_word(born_digital_pdf):
    doc = fitz.open(born_digital_pdf)
    boxes = extract_boxes(doc[0])
    doc.close()

    texts = [b.text for b in _words(boxes)]
    assert texts == ["Revenue", "grew", "42", "percent", "Margins", "held", "steady"]


def test_all_coordinates_are_normalized(born_digital_pdf):
    doc = fitz.open(born_digital_pdf)
    boxes = extract_boxes(doc[0])
    doc.close()

    for b in boxes:
        assert 0.0 <= b.x0 <= 1.0 and 0.0 <= b.y0 <= 1.0
        assert 0.0 <= b.x1 <= 1.0 and 0.0 <= b.y1 <= 1.0
        assert b.x1 > b.x0 and b.y1 > b.y0


def test_first_word_lands_at_known_position(born_digital_pdf):
    """Text starts at TEXT_ORIGIN x on a PAGE_W-wide page.

    Derived from conftest's constants rather than a copied literal, so that
    moving the fixture text fails loudly here instead of silently asserting
    the wrong number.
    """
    doc = fitz.open(born_digital_pdf)
    boxes = extract_boxes(doc[0])
    doc.close()

    first = _words(boxes)[0]
    assert first.text == "Revenue"
    assert first.x0 == pytest.approx(TEXT_ORIGIN[0] / PAGE_W)


def test_parent_hierarchy_is_recorded(born_digital_pdf):
    doc = fitz.open(born_digital_pdf)
    boxes = extract_boxes(doc[0])
    doc.close()

    words = _words(boxes)
    assert words[0].word_no == 0
    assert words[1].word_no == 1
    # The two inserted lines must not share a line_no.
    assert {w.line_no for w in words if w.text == "Revenue"} != {
        w.line_no for w in words if w.text == "Margins"
    } or {w.block_no for w in words if w.text == "Revenue"} != {
        w.block_no for w in words if w.text == "Margins"
    }


def test_rotated_page_boxes_match_rendered_space(rotated_pdf, born_digital_pdf):
    """The core regression test.

    Without page.rotation_matrix, the first word normalizes to x0=0.1176 on the
    rotated page, exactly as on the unrotated one, which is wrong: after a 90
    degree rotation that word belongs near the right edge.
    """
    rot = fitz.open(rotated_pdf)
    rot_boxes = _words(extract_boxes(rot[0]))
    rot.close()

    flat = fitz.open(born_digital_pdf)
    flat_boxes = _words(extract_boxes(flat[0]))
    flat.close()

    rot_first = rot_boxes[0]
    flat_first = flat_boxes[0]

    assert rot_first.text == flat_first.text == "Revenue"
    assert abs(rot_first.x0 - flat_first.x0) > 0.5, "rotation was not applied"
    assert 0.0 <= rot_first.x0 <= 1.0

    # Exact invariant, no magic numbers. A 90 degree rotation maps x' = W - y,
    # y' = x. Normalizing each against its OWN page rect (612x792 flat,
    # 792x612 rotated) makes the relationship exact:
    assert rot_first.x0 == pytest.approx(1.0 - flat_first.y1)
    assert rot_first.y0 == pytest.approx(flat_first.x0)


def test_drops_whitespace_only_words(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "alpha    beta", fontsize=12)
    boxes = extract_boxes(page)
    doc.close()

    assert [b.text for b in _words(boxes)] == ["alpha", "beta"]


def test_drops_degenerate_and_clamps_out_of_bounds(tmp_path):
    """Zero-area boxes are dropped; boxes outside the rect are clamped to 0-1.

    Text placed at a negative y and past the right edge exercises both paths.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "inside", fontsize=12)
    page.insert_text((600, 100), "overhang", fontsize=12)  # runs past x=612
    boxes = extract_boxes(page)
    doc.close()

    words = _words(boxes)
    assert "inside" in [b.text for b in words]
    for b in words:
        assert 0.0 <= b.x0 < b.x1 <= 1.0
        assert 0.0 <= b.y0 < b.y1 <= 1.0


def test_handles_page_with_no_text(scanned_pdf):
    doc = fitz.open(scanned_pdf)
    boxes = extract_boxes(doc[0])
    doc.close()

    assert _words(boxes) == []


def test_extracts_table_cells(tmp_path):
    """A ruled grid should yield table_cell boxes alongside words."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    left, top, cell_w, cell_h = 72, 100, 120, 30
    for r in range(3):
        for c in range(3):
            x, y = left + c * cell_w, top + r * cell_h
            page.draw_rect(fitz.Rect(x, y, x + cell_w, y + cell_h), color=(0, 0, 0), width=1)
            page.insert_text((x + 5, y + 20), f"r{r}c{c}", fontsize=9)
    path = tmp_path / "table.pdf"
    doc.save(path)
    doc.close()

    doc = fitz.open(path)
    boxes = extract_boxes(doc[0])
    doc.close()

    cells = [b for b in boxes if b.kind == "table_cell"]
    assert len(cells) > 0
    for c in cells:
        assert 0.0 <= c.x0 <= 1.0 and 0.0 <= c.y1 <= 1.0
