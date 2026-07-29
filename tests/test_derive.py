import random

import fitz
import pytest

from visual_verify.derive import block_boxes, line_boxes, span_boxes
from visual_verify.ingest.boxes import BoxRecord, extract_boxes, word_boxes


@pytest.fixture
def words(born_digital_pdf) -> list[BoxRecord]:
    doc = fitz.open(born_digital_pdf)
    boxes = word_boxes(extract_boxes(doc[0]))
    doc.close()
    return boxes


def test_line_boxes_group_words(words):
    lines = line_boxes(words)
    assert len(lines) == 2
    texts = sorted(line.text for line in lines)
    assert texts == ["Margins held steady", "Revenue grew 42 percent"]


def test_line_box_spans_all_its_words(words):
    lines = line_boxes(words)
    first = next(line for line in lines if line.text.startswith("Revenue"))
    # Select members by hierarchy, not by text: matching on text would pick up
    # the wrong words on a page with a repeated token.
    members = [w for w in words if (w.block_no, w.line_no) == (first.block_no, first.line_no)]
    assert members
    assert first.x0 == pytest.approx(min(w.x0 for w in members))
    assert first.x1 == pytest.approx(max(w.x1 for w in members))
    assert first.y0 == pytest.approx(min(w.y0 for w in members))
    assert first.y1 == pytest.approx(max(w.y1 for w in members))


def test_block_boxes_group_lines(words):
    blocks = block_boxes(words)
    assert len(blocks) >= 1
    for b in blocks:
        assert b.x1 > b.x0 and b.y1 > b.y0


def test_span_boxes_cover_a_substring(words):
    rects = span_boxes(words, "grew 42")
    assert len(rects) == 1
    box = rects[0]
    grew = next(w for w in words if w.text == "grew")
    forty_two = next(w for w in words if w.text == "42")
    assert box.x0 == pytest.approx(min(grew.x0, forty_two.x0))
    assert box.x1 == pytest.approx(max(grew.x1, forty_two.x1))


def test_span_boxes_match_single_word(words):
    rects = span_boxes(words, "percent")
    assert len(rects) == 1
    assert rects[0].text == "percent"


def test_span_boxes_are_case_insensitive(words):
    assert span_boxes(words, "REVENUE") != []


def test_empty_input_yields_no_boxes():
    assert line_boxes([]) == []
    assert block_boxes([]) == []
    assert span_boxes([], "anything") == []


def test_span_boxes_split_at_line_boundaries(words):
    """A match wrapping a line break returns one rect per line, never a union."""
    rects = span_boxes(words, "percent Margins")
    assert len(rects) == 2
    assert [r.text for r in rects] == ["percent", "Margins"]
    for r in rects:
        assert r.kind == "span"
    # The whole-page union would be ~5.7x the true area. Each rect must be tight.
    assert all(r.x1 - r.x0 < 0.25 for r in rects)


def test_span_boxes_single_line_match_returns_one_rect(words):
    rects = span_boxes(words, "grew 42")
    assert len(rects) == 1
    assert rects[0].text == "grew 42"


def test_span_boxes_returns_empty_when_absent(words):
    assert span_boxes(words, "nonexistent phrase") == []


def test_span_boxes_handles_needle_longer_than_page(words):
    assert span_boxes(words, " ".join(["x"] * 50)) == []


def test_span_boxes_rejects_empty_needle(words):
    assert span_boxes(words, "") == []
    assert span_boxes(words, "   ") == []


def test_span_boxes_returns_first_occurrence(tmp_path):
    """Documented behavior: first match in reading order wins."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "alpha beta gamma", fontsize=12)
    page.insert_text((72, 140), "delta alpha beta", fontsize=12)
    path = tmp_path / "dup.pdf"
    doc.save(path)
    doc.close()

    doc = fitz.open(path)
    ws = word_boxes(extract_boxes(doc[0]))
    doc.close()

    rects = span_boxes(ws, "alpha beta")
    assert len(rects) == 1
    first_alpha_y0 = min(w.y0 for w in ws if w.text == "alpha")
    assert rects[0].y0 == pytest.approx(first_alpha_y0)


def test_grouping_does_not_depend_on_input_order(words):
    """groupby only groups CONSECUTIVE keys, so the sort in _sorted_words is
    load-bearing. Shuffled input must still produce correct groups."""
    shuffled = words[:]
    random.Random(0).shuffle(shuffled)
    assert line_boxes(shuffled) == line_boxes(words)
    assert block_boxes(shuffled) == block_boxes(words)


def test_derived_boxes_carry_their_granularity(words):
    assert all(b.kind == "line" for b in line_boxes(words))
    assert all(b.kind == "block" for b in block_boxes(words))


def test_derived_boxes_cannot_be_re_derived(words):
    lines = line_boxes(words)
    with pytest.raises(ValueError, match="no word boxes"):
        line_boxes(lines)


def test_union_contract(words):
    """Derived boxes report word_no=-1 and inherit block/line from the first member."""
    lines = line_boxes(words)
    for line in lines:
        assert line.word_no == -1
        assert line.block_no != -1


def test_table_cells_do_not_contaminate_line_grouping(tmp_path):
    """Table cells carry block/line/word of -1. Grouping a mixed list by those
    fields would collapse every cell into one fabricated page-wide line."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    x0, y0, cw, ch = 100.0, 300.0, 100.0, 40.0
    for i in range(4):
        page.draw_line(fitz.Point(x0, y0 + i * ch), fitz.Point(x0 + 3 * cw, y0 + i * ch))
        page.draw_line(fitz.Point(x0 + i * cw, y0), fitz.Point(x0 + i * cw, y0 + 3 * ch))
    for r in range(3):
        for c in range(3):
            page.insert_text((x0 + c * cw + 6, y0 + r * ch + 26), f"c{r}{c}", fontsize=11)
    page.insert_text((72, 100), "Revenue grew 42 percent", fontsize=12)
    page.insert_text((72, 140), "Margins held steady", fontsize=12)
    path = tmp_path / "grid.pdf"
    doc.save(path)
    doc.close()

    doc = fitz.open(path)
    mixed = extract_boxes(doc[0])
    doc.close()

    only_words = word_boxes(mixed)
    assert any(b.kind == "table_cell" for b in mixed), "fixture must contain table cells"

    assert line_boxes(mixed) == line_boxes(only_words)
    assert block_boxes(mixed) == block_boxes(only_words)
    assert all(b.block_no != -1 for b in line_boxes(mixed))
    assert all(b.block_no != -1 for b in block_boxes(mixed))
