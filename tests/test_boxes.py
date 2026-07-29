import fitz
import pytest

from conftest import PAGE_W, TEXT_ORIGIN
from visual_verify.ingest.boxes import extract_boxes, word_boxes


def test_extracts_every_word(born_digital_pdf):
    doc = fitz.open(born_digital_pdf)
    boxes = extract_boxes(doc[0])
    doc.close()

    texts = [b.text for b in word_boxes(boxes)]
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

    first = word_boxes(boxes)[0]
    assert first.text == "Revenue"
    assert first.x0 == pytest.approx(TEXT_ORIGIN[0] / PAGE_W)


def test_parent_hierarchy_is_recorded(born_digital_pdf):
    doc = fitz.open(born_digital_pdf)
    boxes = extract_boxes(doc[0])
    doc.close()

    words = word_boxes(boxes)
    assert words[0].word_no == 0
    assert words[1].word_no == 1
    # Downstream grouping keys on the (block, line) PAIR. PyMuPDF puts each
    # insert_text call in its own block, so line_no alone is not distinguishing.
    assert len({(w.block_no, w.line_no) for w in words}) == 2


def test_rotated_page_boxes_match_rendered_space(rotated_pdf, born_digital_pdf):
    """The core regression test.

    Without page.rotation_matrix, the first word normalizes to x0=0.1176 on the
    rotated page, exactly as on the unrotated one, which is wrong: after a 90
    degree rotation that word belongs near the right edge.
    """
    rot = fitz.open(rotated_pdf)
    rot_boxes = word_boxes(extract_boxes(rot[0]))
    rot.close()

    flat = fitz.open(born_digital_pdf)
    flat_boxes = word_boxes(extract_boxes(flat[0]))
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

    assert [b.text for b in word_boxes(boxes)] == ["alpha", "beta"]


def test_drops_degenerate_and_clamps_out_of_bounds(tmp_path):
    """Zero-area boxes are dropped; boxes outside the rect are clamped to 0-1.

    Text placed past the right edge exercises both paths.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "inside", fontsize=12)
    page.insert_text((600, 100), "overhang", fontsize=12)  # runs past x=612
    boxes = extract_boxes(page)
    doc.close()

    words = word_boxes(boxes)
    assert "inside" in [b.text for b in words]
    for b in words:
        assert 0.0 <= b.x0 < b.x1 <= 1.0
        assert 0.0 <= b.y0 < b.y1 <= 1.0


def test_handles_page_with_no_text(scanned_pdf):
    doc = fitz.open(scanned_pdf)
    boxes = extract_boxes(doc[0])
    doc.close()

    assert word_boxes(boxes) == []


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
    assert len(cells) == 9
    for c in cells:
        assert 0.0 <= c.x0 < c.x1 <= 1.0
        assert 0.0 <= c.y0 < c.y1 <= 1.0
    # The grid spans x 72..432pt and y 100..190pt on a 612x792 page.
    assert min(c.x0 for c in cells) == pytest.approx(72 / 612, abs=0.01)
    assert max(c.x1 for c in cells) == pytest.approx(432 / 612, abs=0.01)
    assert min(c.y0 for c in cells) == pytest.approx(100 / 792, abs=0.01)
    assert max(c.y1 for c in cells) == pytest.approx(190 / 792, abs=0.01)


def test_table_detection_failure_degrades_to_words(born_digital_pdf, monkeypatch):
    """A find_tables explosion must not lose the word boxes."""

    def boom(self, *a, **kw):
        raise RuntimeError("simulated pymupdf failure")

    monkeypatch.setattr(fitz.Page, "find_tables", boom)

    doc = fitz.open(born_digital_pdf)
    boxes = extract_boxes(doc[0])
    doc.close()

    assert [b.text for b in word_boxes(boxes)][0] == "Revenue"
    assert [b for b in boxes if b.kind == "table_cell"] == []


def test_table_cells_are_rotation_corrected(tmp_path):
    """Cells reach _normalize by a different path than words, so cover them too."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for r in range(3):
        for c in range(3):
            x, y = 72 + c * 120, 100 + r * 30
            page.draw_rect(fitz.Rect(x, y, x + 120, y + 30), color=(0, 0, 0), width=1)
            page.insert_text((x + 5, y + 20), f"r{r}c{c}", fontsize=9)
    doc[0].set_rotation(90)
    path = tmp_path / "rotated_table.pdf"
    doc.save(path)
    doc.close()

    doc = fitz.open(path)
    cells = [b for b in extract_boxes(doc[0]) if b.kind == "table_cell"]
    doc.close()

    # Exactly 9, not just "some": reintroducing the double-rotation bug drops
    # the count to 6 at this rotation, so the count is itself a regression signal.
    assert len(cells) == 9
    for c in cells:
        assert 0.0 <= c.x0 < c.x1 <= 1.0
        assert 0.0 <= c.y0 < c.y1 <= 1.0
    # Rotated 90, the grid that spanned x 72..432 now spans y 72..432 of a 612-tall page.
    assert min(c.y0 for c in cells) == pytest.approx(72 / 612, abs=0.02)


def test_words_come_before_table_cells(tmp_path):
    """extract_boxes documents this ordering; pin it unconditionally.

    A lone draw_rect is NOT enough: find_tables reports no table for it, the
    box list is all words, and any guarded assertion silently never runs. The
    3x3 grid is the smallest fixture that actually yields both kinds.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for r in range(3):
        for c in range(3):
            x, y = 72 + c * 120, 100 + r * 30
            page.draw_rect(fitz.Rect(x, y, x + 120, y + 30), color=(0, 0, 0), width=1)
            page.insert_text((x + 5, y + 20), f"r{r}c{c}", fontsize=9)
    path = tmp_path / "ordering.pdf"
    doc.save(path)
    doc.close()

    doc = fitz.open(path)
    kinds = [b.kind for b in extract_boxes(doc[0])]
    doc.close()

    assert "word" in kinds and "table_cell" in kinds
    assert kinds.index("table_cell") > max(i for i, k in enumerate(kinds) if k == "word")


def test_bbox_property_matches_fields(born_digital_pdf):
    doc = fitz.open(born_digital_pdf)
    first = word_boxes(extract_boxes(doc[0]))[0]
    doc.close()

    assert first.bbox == (first.x0, first.y0, first.x1, first.y1)
    assert len(first.bbox) == 4
