"""answer_stream() is the loop; answer() is a drain over it.

The one property that matters is that they cannot diverge. A second copy of
the read-ground-verify loop would put the GroundingError recovery and the
`score < threshold` comparison in two places, and the copy the product hits
is the one no S5 test covers.
"""

from pathlib import Path

import pytest

from visual_verify.agent import AgentError, answer, answer_stream
from visual_verify.agent.events import AnswerComplete, ClaimsProduced, ClaimVerified, ReadingStarted
from visual_verify.agent.schemas import ClaimList, Verdict
from visual_verify.agent.types import FakeChat
from visual_verify.ingest.boxes import BoxRecord


def page_boxes():
    """Two lines, both findable in the text layer.

    Defined here rather than imported from tests/test_agent.py: `tests/` has no
    __init__.py, so `from tests.test_agent import ...` depends on rootdir being
    on sys.path and breaks under a plain `pytest tests/test_answer_stream.py`.

    Every claim scripted below must be findable verbatim, or ground() falls
    through to the visual path, which needs page vectors and a grid and would
    raise GroundingError instead of exercising the loop.
    """
    first = ["Revenue", "grew", "42", "percent"]
    second = ["Margins", "held", "steady"]
    boxes = [
        BoxRecord(
            kind="word",
            x0=0.1 + i * 0.15,
            y0=0.10,
            x1=0.22 + i * 0.15,
            y1=0.16,
            text=t,
            block_no=0,
            line_no=0,
            word_no=i,
        )
        for i, t in enumerate(first)
    ]
    boxes += [
        BoxRecord(
            kind="word",
            x0=0.1 + i * 0.15,
            y0=0.30,
            x1=0.22 + i * 0.15,
            y1=0.36,
            text=t,
            block_no=0,
            line_no=1,
            word_no=i,
        )
        for i, t in enumerate(second)
    ]
    return boxes


def script():
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent", "Margins held steady"])])
    verifier = FakeChat(
        "v",
        [
            Verdict(label="supported", confidence=0.9, reason="matches"),
            Verdict(label="unsupported", confidence=0.8, reason="contradicted"),
        ],
    )
    return reader, verifier


def run_stream():
    reader, verifier = script()
    return list(
        answer_stream(
            "What happened?",
            Path("p.png"),
            page_boxes(),
            page=0,
            reader_chat=reader,
            verifier_chat=verifier,
        )
    )


def test_the_event_order_is_reading_then_count_then_claims_then_complete():
    events = run_stream()

    assert isinstance(events[0], ReadingStarted)
    assert isinstance(events[1], ClaimsProduced)
    assert events[1].n == 2
    assert isinstance(events[2], ClaimVerified)
    assert isinstance(events[3], ClaimVerified)
    assert isinstance(events[4], AnswerComplete)
    assert len(events) == 5


def test_claim_events_are_indexed_in_order():
    events = [e for e in run_stream() if isinstance(e, ClaimVerified)]

    assert [e.index for e in events] == [0, 1]


def test_every_streamed_claim_already_has_a_verdict():
    """The reason S5 refused to stream the reader: a claim must never reach a
    consumer before the verifier has judged it. If this can fail, the product
    can display something it exists to withhold."""
    for event in run_stream():
        if isinstance(event, ClaimVerified):
            assert event.claim.label is not None


def test_answer_returns_exactly_what_the_stream_finished_with():
    """Pins the drain. If answer() ever grows its own loop, this fails."""
    streamed = [e for e in run_stream() if isinstance(e, AnswerComplete)][0].answer

    reader, verifier = script()
    direct = answer(
        "What happened?",
        Path("p.png"),
        page_boxes(),
        page=0,
        reader_chat=reader,
        verifier_chat=verifier,
    )

    assert direct == streamed


def test_the_same_model_guard_raises_before_anything_is_iterated():
    """Eagerly, at call time, with no list() to drive the generator.

    The earlier version of this test wrapped the call in list(), which drives
    the generator to exhaustion and therefore passed whether the guard fired at
    call time, on first advance, or on the last claim. It could not distinguish
    the property it was named for.

    Eager matters concretely. The API reaches this through ask_events(), which
    yields a `retrieved` event first, so a guard that waited for first advance
    would raise after the 200 and its headers were committed and reach the
    browser as an SSE error frame instead of a refusal to start.
    """
    same = FakeChat("same", [ClaimList(claims=["x"])])
    other = FakeChat("same", [Verdict(label="supported", confidence=0.5, reason="r")])

    with pytest.raises(AgentError):
        answer_stream(
            "q",
            Path("p.png"),
            page_boxes(),
            page=0,
            reader_chat=same,
            verifier_chat=other,
        )


def test_a_reader_that_returns_nothing_reports_a_count_of_zero():
    reader = FakeChat("r", [ClaimList(claims=[])])
    verifier = FakeChat("v", [])

    events = list(
        answer_stream(
            "q",
            Path("p.png"),
            page_boxes(),
            page=0,
            reader_chat=reader,
            verifier_chat=verifier,
        )
    )

    assert isinstance(events[0], ReadingStarted)
    assert isinstance(events[1], ClaimsProduced)
    assert events[1].n == 0
    assert isinstance(events[2], AnswerComplete)
    assert events[2].answer.abstained_overall is True
    assert len(events) == 3


def test_the_paragraph_break_survives_the_trip_from_reader_to_claim():
    """The only place starts_paragraph crosses a layer boundary.

    Everything else about the field is schema-level, so deleting the kwarg in
    core.py, or hardcoding it to False, leaves the rest of the suite green while
    the displayed answer renders as one paragraph forever. That is this repo's
    recurring failure shape: correctly-shaped output, wrong value, nothing
    notices.

    Two claims, not one, and they disagree. A single-claim fixture passes
    against a hardcoded True as easily as against the real copy.
    """
    reader = FakeChat(
        "r",
        [
            ClaimList(
                claims=[
                    {"text": "Revenue grew 42 percent"},
                    {"text": "Margins held steady", "starts_paragraph": True},
                ]
            )
        ],
    )
    verifier = FakeChat(
        "v",
        [
            Verdict(label="supported", confidence=0.9, reason="matches"),
            Verdict(label="supported", confidence=0.9, reason="matches"),
        ],
    )

    out = answer(
        "What happened?",
        Path("p.png"),
        page_boxes(),
        page=0,
        reader_chat=reader,
        verifier_chat=verifier,
    )

    assert out.claims[0].starts_paragraph is False
    assert out.claims[1].starts_paragraph is True
