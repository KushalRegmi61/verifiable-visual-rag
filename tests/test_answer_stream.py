"""answer_stream() is the loop; answer() is a drain over it.

The one property that matters is that they cannot diverge. A second copy of
the read-ground-verify loop would put the GroundingError recovery and the
`score < threshold` comparison in two places, and the copy the product hits
is the one no S5 test covers.
"""

from pathlib import Path

import pytest

from helpers import claim_list
from visual_verify.agent import AgentError, answer, answer_stream
from visual_verify.agent.events import AnswerComplete, ClaimsProduced, ClaimVerified, ReadingStarted
from visual_verify.agent.reader import shares_a_term
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
    reader = FakeChat("r", [claim_list("Revenue grew 42 percent", "Margins held steady")])
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
    same = FakeChat("same", [claim_list("x")])
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


def test_a_non_sequitur_region_reaches_the_verifier_as_no_evidence():
    """THE test of the citation filter, and it pins WHERE the filter lives.

    Measured live on 2026-08-10: the visual path returned [0.526 0.940 0.535
    0.953] on proposal.pdf page 14, an 11 by 22 px box holding the page number,
    as the cited evidence for three different claims across two unrelated
    questions. The verifier scored two of them supported at 0.90 and 0.95,
    because it judges whether the CLAIM is true and the claims were true. A
    fabricated citation passed the abstention gate untouched.

    Asserted on the verifier's PROMPT rather than on the returned Claim,
    because that is the thing the filter has to change. A test that only
    checked `claim.regions` would pass with the filter moved into ground(),
    where it must not be: ground()'s contract says an empty list means no
    evidence exists on the page, and S7's ablation needs the grounder measured
    without it.

    The visual path is forced by scripting a claim that is NOT in the text
    layer, which is the only way a region can arrive that does not already
    contain the claim's own words.
    """
    import numpy as np

    from visual_verify.agent import answer
    from visual_verify.agent.verifier import NO_EVIDENCE
    from visual_verify.ingest.boxes import BoxRecord
    from visual_verify.retrieval.geometry import PatchGrid

    page_number = [
        BoxRecord(
            kind="word",
            x0=0.526,
            y0=0.940,
            x1=0.535,
            y1=0.953,
            text="7",
            block_no=0,
            line_no=0,
            word_no=0,
        )
    ]
    grid = PatchGrid(n_x=4, n_y=4, offset=0, n_vectors=16)
    page_vectors = np.ones((16, 8), dtype=np.float32)
    reader = FakeChat("r", [claim_list("Revenue grew 42 percent")])
    verifier = FakeChat(
        "v", [Verdict(label="supported", confidence=0.95, reason="looks fine to me")]
    )

    out = answer(
        "q",
        Path("page.png"),
        page_number,
        page=0,
        reader_chat=reader,
        verifier_chat=verifier,
        threshold=0.0,
        page_vectors=page_vectors,
        embed_query=lambda _: np.ones((2, 8), dtype=np.float32),
        grid=grid,
    )

    prompt = verifier.calls[0].prompt
    assert NO_EVIDENCE in prompt, (
        "the page-number region was cited to the verifier instead of being dropped"
    )
    assert "0.526" not in prompt, "the dropped region's geometry still reached the verifier"
    assert out.claims[0].regions == []


def test_a_region_that_names_nothing_in_the_claim_is_not_cited():
    """The page-number sink, pinned at the layer that decides what to cite.

    A GroundedRegion whose text shares no term with the claim reaches the
    verifier as no evidence at all, so the rubric returns insufficient_evidence
    and the gate withholds the claim. Before this, the region was cited, the
    verifier judged only whether the CLAIM was true, and a true claim with a
    fabricated citation passed at 0.95.

    Driven through answer() rather than by calling the filter directly, because
    the filter living in the wrong place is the failure mode: ground() must not
    do it, and a unit test of the predicate cannot tell where it was applied.
    """
    from visual_verify.agent import answer

    reader = FakeChat("r", [claim_list("Revenue grew 42 percent")])
    verifier = FakeChat(
        "v", [Verdict(label="supported", confidence=0.95, reason="looks fine to me")]
    )

    out = answer(
        "q",
        Path("page.png"),
        page_boxes(),
        page=0,
        reader_chat=reader,
        verifier_chat=verifier,
        threshold=0.0,
    )

    # The text path finds this claim verbatim, so it is cited normally and the
    # filter is not what produced this. That is the control half.
    assert out.claims[0].regions, "the verbatim claim should still carry its region"
    assert all(shares_a_term(out.claims[0].text, r.text) for r in out.claims[0].regions), (
        "every cited region must name something the claim names"
    )


def test_the_verifier_sees_no_evidence_when_every_region_was_a_non_sequitur():
    """The half the fixture above cannot reach, because the text path always
    finds a real line. Asserts the CONTRACT: what the verifier is handed is the
    filtered list, so a page-number-only grounding arrives as empty rather than
    as a citation the verifier will rubber-stamp."""
    from visual_verify.contracts import GroundedRegion

    page_number = GroundedRegion(
        page=0,
        bbox=(0.526, 0.940, 0.535, 0.953),
        score=1.0,
        modality="visual",
        text="7",
        resolution="line",
    )
    claim = "The three evaluation metrics are answer accuracy and grounding overlap."

    assert [r for r in [page_number] if shares_a_term(claim, r.text)] == []


def test_a_claim_with_no_region_is_withheld_whatever_the_verifier_said():
    """Structural, because the prompt instruction is not obeyed reliably.

    The verifier prompt asks for insufficient_evidence when no regions are
    listed. Measured on proposal.pdf page 14, it complied for one region-less
    claim (insufficient_evidence at 1.00) and not for another in the SAME
    answer (supported at 0.90). A sentence displayed with no region is an
    answer with no evidence behind it, which is the one thing this project
    exists to refuse, so it cannot rest on the model cooperating.

    The raw verdict stays on `label` for the S7 eval; only the display gate
    moves.
    """
    import numpy as np

    from visual_verify.agent import answer
    from visual_verify.ingest.boxes import BoxRecord
    from visual_verify.retrieval.geometry import PatchGrid

    page_number = [
        BoxRecord(
            kind="word",
            x0=0.526,
            y0=0.940,
            x1=0.535,
            y1=0.953,
            text="8",
            block_no=0,
            line_no=0,
            word_no=0,
        )
    ]
    reader = FakeChat("r", [claim_list("Revenue grew 42 percent")])
    verifier = FakeChat(
        "v", [Verdict(label="supported", confidence=1.0, reason="I like it anyway")]
    )

    out = answer(
        "q",
        Path("page.png"),
        page_number,
        page=0,
        reader_chat=reader,
        verifier_chat=verifier,
        threshold=0.0,
        page_vectors=np.ones((16, 8), dtype=np.float32),
        embed_query=lambda _: np.ones((2, 8), dtype=np.float32),
        grid=PatchGrid(n_x=4, n_y=4, offset=0, n_vectors=16),
    )

    claim = out.claims[0]
    assert claim.regions == []
    assert claim.label == "supported", "the verifier's raw verdict must survive for S7"
    assert claim.withheld is True, "a claim with no evidence must not be displayable"
    assert out.shown == []
