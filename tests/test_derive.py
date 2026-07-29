import fitz
import pytest

from visual_verify.derive import block_boxes, line_boxes, span_box
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
    members = [w for w in words if w.text in first.text.split()]
    assert first.x0 == pytest.approx(min(w.x0 for w in members))
    assert first.x1 == pytest.approx(max(w.x1 for w in members))


def test_block_boxes_group_lines(words):
    blocks = block_boxes(words)
    assert len(blocks) >= 1
    for b in blocks:
        assert b.x1 > b.x0 and b.y1 > b.y0


def test_span_box_covers_a_substring(words):
    box = span_box(words, "grew 42")
    assert box is not None
    grew = next(w for w in words if w.text == "grew")
    forty_two = next(w for w in words if w.text == "42")
    assert box.x0 == pytest.approx(min(grew.x0, forty_two.x0))
    assert box.x1 == pytest.approx(max(grew.x1, forty_two.x1))


def test_span_box_matches_single_word(words):
    box = span_box(words, "percent")
    assert box is not None
    assert box.text == "percent"


def test_span_box_returns_none_when_absent(words):
    assert span_box(words, "nonexistent phrase") is None


def test_span_box_is_case_insensitive(words):
    assert span_box(words, "REVENUE") is not None


def test_empty_input_yields_no_boxes():
    assert line_boxes([]) == []
    assert block_boxes([]) == []
    assert span_box([], "anything") is None
