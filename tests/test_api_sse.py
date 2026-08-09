"""SSE framing. Ten lines of string building, and two ways to get it wrong."""

from visual_verify.api.sse import frame


def test_a_frame_is_an_event_line_a_data_line_and_a_blank_line():
    assert frame("claim", {"index": 0}) == 'event: claim\ndata: {"index": 0}\n\n'


def test_a_newline_inside_the_payload_does_not_truncate_the_event():
    """A raw newline in a data field ends the field, so a verifier reason
    spanning two lines would cut the event in half and the browser would parse
    the tail as a separate malformed frame. json.dumps escapes it; this pins
    that the dump is not bypassed.

    Counting ALL newlines is what discriminates, and it took two wrong
    assertions to get here. `count("\\n\\n") == 1` holds either way, because an
    unescaped payload adds a lone newline in the middle and leaves the
    terminator intact. `len(splitlines()) == 2` is simply false for a correct
    frame: a trailing "\\n\\n" splits to three elements, the last empty. Both
    were green against an escaped payload and neither could fail against an
    unescaped one.
    """
    text = frame("claim", {"reason": "line one\nline two"})

    # One after the event line, then the two that terminate the frame. A
    # literal newline in the payload makes it four.
    assert text.count("\n") == 3
    assert text.endswith("\n\n")
    assert "\\n" in text


def test_unicode_survives_unescaped():
    """ensure_ascii=False keeps a page's own characters readable in the
    stream. \\u00e9 would still decode, but nobody can eyeball it."""
    assert "é" in frame("claim", {"text": "café"})


def test_the_event_name_is_used_verbatim():
    assert frame("done", {}).startswith("event: done\n")
