import pytest
from pydantic import ValidationError

from visual_verify.contracts import Answer, Claim, GroundedRegion, RetrievedPage


def test_grounded_region_accepts_normalized_bbox():
    r = GroundedRegion(
        page=3,
        bbox=(0.1, 0.2, 0.4, 0.25),
        score=0.9,
        modality="text",
        text="Revenue grew 42 percent",
    )
    assert r.bbox == (0.1, 0.2, 0.4, 0.25)
    assert r.modality == "text"


def test_grounded_region_rejects_out_of_range_bbox():
    with pytest.raises(ValidationError):
        GroundedRegion(page=1, bbox=(0.1, 0.2, 1.4, 0.25), score=0.9, modality="text")


def test_grounded_region_rejects_inverted_bbox():
    with pytest.raises(ValidationError):
        GroundedRegion(page=1, bbox=(0.5, 0.2, 0.3, 0.25), score=0.9, modality="text")


def test_grounded_region_rejects_unknown_modality():
    with pytest.raises(ValidationError):
        GroundedRegion(page=1, bbox=(0.1, 0.2, 0.3, 0.25), score=0.9, modality="pixels")


def test_claim_and_answer_nest():
    r = GroundedRegion(
        page=1, bbox=(0.1, 0.2, 0.3, 0.25), score=0.9, modality="visual", crop_ref="crops/p1_0.png"
    )
    c = Claim(text="Revenue grew 42 percent", regions=[r], confidence=0.8, abstained=False)
    a = Answer(question="How much did revenue grow?", claims=[c], abstained_overall=False)
    assert a.claims[0].regions[0].crop_ref == "crops/p1_0.png"


def test_retrieved_page_defaults_text_layer_to_none():
    p = RetrievedPage(doc_id="abc", page=2, image_ref="data/pages/abc/p0002.png", score=0.5)
    assert p.text_layer is None


def test_grounded_region_rejects_negative_page():
    with pytest.raises(ValidationError):
        GroundedRegion(page=-1, bbox=(0.1, 0.2, 0.3, 0.25), score=0.9, modality="text")


def test_grounded_region_rejects_zero_area_bbox():
    with pytest.raises(ValidationError):
        GroundedRegion(page=0, bbox=(0.5, 0.5, 0.5, 0.5), score=0.9, modality="text")


def test_grounded_region_accepts_full_page_bbox():
    r = GroundedRegion(page=0, bbox=(0.0, 0.0, 1.0, 1.0), score=0.9, modality="visual")
    assert r.bbox == (0.0, 0.0, 1.0, 1.0)


def test_claim_rejects_confidence_outside_unit_interval():
    with pytest.raises(ValidationError):
        Claim(text="x", regions=[], confidence=1.5)
    with pytest.raises(ValidationError):
        Claim(text="x", regions=[], confidence=-0.1)


def test_score_may_exceed_one():
    """MaxSim scores are sums over query tokens, not probabilities."""
    r = GroundedRegion(page=0, bbox=(0.1, 0.1, 0.2, 0.2), score=17.4, modality="visual")
    assert r.score == 17.4


def test_collections_default_to_empty():
    c = Claim(text="x", confidence=0.5)
    a = Answer(question="q?")
    assert c.regions == []
    assert a.claims == []
    assert a.abstained_overall is False


def test_answer_shown_excludes_abstained_claims():
    """Iterating `claims` directly would display a claim the verifier refused.

    Withholding those is the whole point of the system, and the guarantee
    otherwise rests on every consumer remembering to check a boolean.
    """
    from visual_verify.contracts import Answer, Claim

    a = Answer(
        question="q",
        claims=[
            Claim(text="kept", confidence=0.9, label="supported", abstained=False),
            Claim(text="withheld", confidence=0.9, label="unsupported", abstained=True),
        ],
    )

    assert [c.text for c in a.shown] == ["kept"]
    assert len(a.claims) == 2, "the rejected claim must still be present for the eval"


def test_answer_shown_excludes_a_claim_that_never_reached_the_verifier():
    """`label` defaults to None and `abstained` defaults to False, so a Claim
    built by hand (or by anything other than answer()) and never judged would
    otherwise read as shown. Absence of a verdict is not a passing verdict.
    """
    a = Answer(claims=[Claim(text="unverified", confidence=0.5)], question="q")

    assert a.shown == []


def test_claim_carries_the_verifier_reason():
    from visual_verify.contracts import Claim

    c = Claim(text="Margins held steady", confidence=0.5, reason="the chart shows margin falling")

    assert c.reason == "the chart shows margin falling"


def test_claim_reason_defaults_to_none():
    """Additive optional field: every existing construction site still works."""
    from visual_verify.contracts import Claim

    assert Claim(text="x", confidence=0.5).reason is None
