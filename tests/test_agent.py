"""The pipeline, and the gate that actually withholds."""

from pathlib import Path

import pytest

from visual_verify.agent import AgentError, answer
from visual_verify.agent.schemas import ClaimList, Verdict
from visual_verify.agent.types import FakeChat
from visual_verify.ingest.boxes import BoxRecord


def word(x0, y0, x1, y1, text, word_no=0):
    return BoxRecord(
        kind="word",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        block_no=0,
        line_no=0,
        word_no=word_no,
    )


def page_boxes():
    """Two lines, so a second claim can also ground through the TEXT path.

    Every claim here must be findable in the text layer. A claim that is not
    falls through to ground()'s visual path, which needs page vectors and a
    grid, and would raise GroundingError rather than exercising the pipeline.
    """
    first = ["Revenue", "grew", "42", "percent"]
    second = ["Margins", "held", "steady"]
    boxes = [word(0.1 + i * 0.15, 0.10, 0.22 + i * 0.15, 0.16, t, i) for i, t in enumerate(first)]
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


def test_a_supported_claim_is_shown_with_its_regions():
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent"])])
    verifier = FakeChat("v", [Verdict(label="supported", confidence=0.9, reason="matches")])

    out = answer(
        "What happened?",
        Path("p.png"),
        page_boxes(),
        page=0,
        reader_chat=reader,
        verifier_chat=verifier,
    )

    assert len(out.claims) == 1
    claim = out.claims[0]
    assert claim.abstained is False
    assert claim.label == "supported"
    assert len(claim.regions) == 1
    assert claim.regions[0].modality == "text"


def test_an_unsupported_claim_is_abstained_on():
    """The point of the project: a wrong answer with a confident box drawn on
    it is worse than no answer."""
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent"])])
    verifier = FakeChat("v", [Verdict(label="unsupported", confidence=0.9, reason="no")])

    out = answer(
        "q", Path("p.png"), page_boxes(), page=0, reader_chat=reader, verifier_chat=verifier
    )

    assert out.claims[0].abstained is True
    assert out.claims[0].label == "unsupported"


def test_a_partially_supported_claim_is_abstained_on_at_the_default_threshold():
    """Even at confidence 1.0. The label decides, not the number."""
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent"])])
    verifier = FakeChat("v", [Verdict(label="partially_supported", confidence=1.0, reason="half")])

    out = answer(
        "q", Path("p.png"), page_boxes(), page=0, reader_chat=reader, verifier_chat=verifier
    )

    assert out.claims[0].abstained is True


def test_lowering_the_threshold_admits_a_partially_supported_claim():
    """The threshold is a parameter because S7 sweeps it to build the
    confident-wrong against coverage curve."""
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent"])])
    verifier = FakeChat("v", [Verdict(label="partially_supported", confidence=0.5, reason="half")])

    out = answer(
        "q",
        Path("p.png"),
        page_boxes(),
        page=0,
        threshold=4.0,
        reader_chat=reader,
        verifier_chat=verifier,
    )

    assert out.claims[0].abstained is False


def test_every_claim_reaching_the_verifier_gets_one_call():
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent", "Margins held steady"])])
    verifier = FakeChat(
        "v",
        [
            Verdict(label="supported", confidence=0.9, reason="a"),
            Verdict(label="unsupported", confidence=0.9, reason="b"),
        ],
    )

    answer("q", Path("p.png"), page_boxes(), page=0, reader_chat=reader, verifier_chat=verifier)

    assert len(verifier.calls) == 2


def test_a_reader_returning_nothing_abstains_overall():
    reader = FakeChat("r", [ClaimList(claims=[])])
    verifier = FakeChat("v", [])

    out = answer(
        "q", Path("p.png"), page_boxes(), page=0, reader_chat=reader, verifier_chat=verifier
    )

    assert out.claims == []
    assert out.abstained_overall is True
    assert len(verifier.calls) == 0, "nothing to verify, so no call should be made"


def test_all_claims_abstained_means_abstained_overall():
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent"])])
    verifier = FakeChat("v", [Verdict(label="unsupported", confidence=0.9, reason="no")])

    out = answer(
        "q", Path("p.png"), page_boxes(), page=0, reader_chat=reader, verifier_chat=verifier
    )

    assert out.abstained_overall is True


def test_a_compound_claim_is_flagged_but_still_shown_and_verified():
    """The reader's conjunction check must actually reach the pipeline. A
    compound claim is advisory information for the eval, never a reason to
    drop or reject the claim: dropping it would lose an answer."""
    boxes = [
        word(0.1 + i * 0.15, 0.10, 0.22 + i * 0.15, 0.16, t, i)
        for i, t in enumerate(["Revenue", "grew", "and", "margins", "fell"])
    ]
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew and margins fell"])])
    verifier = FakeChat("v", [Verdict(label="supported", confidence=0.9, reason="matches")])

    out = answer(
        "What happened?",
        Path("p.png"),
        boxes,
        page=0,
        reader_chat=reader,
        verifier_chat=verifier,
    )

    assert len(out.claims) == 1
    claim = out.claims[0]
    assert claim.compound is True
    assert claim.abstained is False
    assert len(claim.regions) == 1


def test_the_same_model_for_both_roles_is_refused():
    """The separate-judge requirement is the reason this slice exists. A
    misconfiguration pointing both roles at one model would otherwise be
    invisible and would silently invalidate every verification."""
    same = FakeChat("openai:gpt-4o", [ClaimList(claims=["a"])])
    other = FakeChat("openai:gpt-4o", [Verdict(label="supported", confidence=0.9, reason="r")])

    with pytest.raises(AgentError, match="same model"):
        answer("q", Path("p.png"), page_boxes(), page=0, reader_chat=same, verifier_chat=other)


def test_a_claim_that_cannot_be_grounded_does_not_abort_the_whole_answer():
    """A reader paraphrases by default, so a claim not findable verbatim in
    the text layer, with no visual vectors supplied (the CLI's default state
    before this fix), is the EXPECTED case, not a rare one. ground() raises
    GroundingError for exactly that input; answer() must treat it as zero
    regions for this claim and keep verifying the rest, not lose every
    already-verified claim to one unlucky one.
    """
    reader = FakeChat(
        "r",
        [ClaimList(claims=["Revenue grew 42 percent", "The paraphrase is nowhere on this page"])],
    )
    verifier = FakeChat(
        "v",
        [
            Verdict(label="supported", confidence=0.9, reason="matches"),
            Verdict(label="insufficient_evidence", confidence=0.6, reason="no region"),
        ],
    )

    out = answer(
        "q", Path("p.png"), page_boxes(), page=0, reader_chat=reader, verifier_chat=verifier
    )

    assert len(out.claims) == 2
    grounded, ungroundable = out.claims
    assert len(grounded.regions) == 1
    assert ungroundable.regions == []
    assert ungroundable.label == "insufficient_evidence"
