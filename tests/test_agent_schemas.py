"""Schemas the models must return, and the contract field that carries a verdict."""

import pytest
from pydantic import ValidationError

from visual_verify.agent.schemas import ClaimList, Verdict
from visual_verify.contracts import Claim


def test_claim_list_holds_atomic_claims():
    parsed = ClaimList(claims=["Revenue grew 42 percent.", "Margins held steady."])
    assert [c.text for c in parsed.claims] == [
        "Revenue grew 42 percent.",
        "Margins held steady.",
    ]


def test_claim_list_rejects_an_empty_string_claim():
    """An empty claim would ground to nothing and verify as insufficient,
    consuming an API call to learn the model returned junk."""
    with pytest.raises(ValidationError):
        ClaimList(claims=["Revenue grew.", "   "])


def test_verdict_requires_a_known_label():
    with pytest.raises(ValidationError):
        Verdict(label="looks_fine", confidence=0.5, reason="x")


def test_verdict_bounds_confidence():
    with pytest.raises(ValidationError):
        Verdict(label="supported", confidence=1.5, reason="x")


def test_verdict_requires_a_reason():
    """The reason is what makes a wrong verdict debuggable after the fact,
    and it goes in the eval output. An empty one is a silent verdict."""
    with pytest.raises(ValidationError):
        Verdict(label="supported", confidence=0.9, reason="")


def test_claim_carries_an_optional_label():
    """None until the verifier has run, so an unverified claim is
    distinguishable from one judged unsupported."""
    unverified = Claim(text="x", confidence=0.0)
    assert unverified.label is None

    judged = Claim(text="x", confidence=0.9, label="supported")
    assert judged.label == "supported"


def test_claim_rejects_an_unknown_label():
    with pytest.raises(ValidationError):
        Claim(text="x", confidence=0.9, label="looks_fine")


def test_a_claim_carries_a_paragraph_break_flag():
    parsed = ClaimList(
        claims=[
            {"text": "The evaluation compares three variants.", "starts_paragraph": False},
            {"text": "The ablation removes the added layer.", "starts_paragraph": True},
        ]
    )

    assert parsed.claims[1].text == "The ablation removes the added layer."
    assert parsed.claims[1].starts_paragraph is True
    assert parsed.claims[0].starts_paragraph is False


def test_a_bare_string_is_still_accepted_as_a_claim():
    """Coercion, kept deliberately. Roughly forty existing construction sites
    across the test suite pass plain strings, and rewriting them all to
    dictionaries would be a large diff that tests nothing. A model that returns
    strings gets starts_paragraph False, which is the right default."""
    parsed = ClaimList(claims=["Revenue grew 42 percent."])

    assert parsed.claims[0].text == "Revenue grew 42 percent."
    assert parsed.claims[0].starts_paragraph is False
