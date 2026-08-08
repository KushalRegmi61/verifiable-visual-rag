"""Verdicts, and the property that a verifier which cannot say no is useless."""

from pathlib import Path

from visual_verify.agent.schemas import Verdict
from visual_verify.agent.types import FakeChat
from visual_verify.agent.verifier import verify
from visual_verify.contracts import GroundedRegion


def region(text="Revenue grew 42 percent"):
    return GroundedRegion(page=0, bbox=(0.1, 0.1, 0.5, 0.2), score=1.0, modality="text", text=text)


def test_verify_returns_the_models_verdict():
    chat = FakeChat("m", [Verdict(label="supported", confidence=0.9, reason="matches")])
    v = verify(chat, Path("p.png"), "Revenue grew 42 percent.", [region()])

    assert v.label == "supported"
    assert v.reason == "matches"


def test_the_verifier_can_return_unsupported():
    """The single most important property in this slice.

    A verifier stuck on 'supported' passes every other test in the suite and
    produces a system that looks like it works and verifies nothing.
    """
    chat = FakeChat(
        "m", [Verdict(label="unsupported", confidence=0.8, reason="region says otherwise")]
    )
    v = verify(chat, Path("p.png"), "Revenue fell.", [region()])

    assert v.label == "unsupported"


def test_the_claim_and_the_region_text_both_reach_the_model():
    """Otherwise the verifier judges a claim against nothing and its verdict
    is about the page in general, not about the evidence offered."""
    chat = FakeChat("m", [Verdict(label="supported", confidence=0.5, reason="r")])
    verify(chat, Path("p.png"), "Revenue grew 42 percent.", [region("Revenue grew 42 percent")])

    prompt = chat.calls[0].prompt
    assert "Revenue grew 42 percent." in prompt
    assert "Revenue grew 42 percent" in prompt


def test_a_claim_with_no_regions_is_still_verified():
    """It must NOT be routed around the verifier. 'insufficient_evidence' is a
    label the rubric already has, and skipping the call would discard exactly
    the signal the project is measuring."""
    chat = FakeChat(
        "m", [Verdict(label="insufficient_evidence", confidence=0.7, reason="no evidence")]
    )
    v = verify(chat, Path("p.png"), "Revenue grew.", [])

    assert len(chat.calls) == 1
    assert v.label == "insufficient_evidence"


def test_the_page_image_reaches_the_verifier():
    """The verifier judges against the page, not only against extracted text,
    because visual regions carry evidence no text layer holds."""
    chat = FakeChat("m", [Verdict(label="supported", confidence=0.5, reason="r")])
    verify(chat, Path("page7.png"), "c", [region()])

    assert chat.calls[0].image_path == Path("page7.png")
