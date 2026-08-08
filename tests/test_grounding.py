"""The text path, and ground() routing."""

from visual_verify.grounding.text_span import text_regions
from visual_verify.ingest.boxes import BoxRecord


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
    """
    regions = text_regions("percent Margins", two_line_page(), page=0)

    assert len(regions) == 2
    for r in regions:
        assert r.bbox[3] - r.bbox[1] < 0.10, "a rect spanning both lines is a union"


def test_text_regions_score_is_exact():
    regions = text_regions("Revenue", two_line_page(), page=0)

    assert regions[0].score == 1.0
