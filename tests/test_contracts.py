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


def test_an_answer_with_no_claims_abstains():
    """An answer with nothing in it is a refusal, not a successful empty answer.

    This used to assert False, because it was reading the stored field's
    default. core.py meanwhile computed `not claims or ...`, which is True for
    the same object, so the model and the only code that filled it in disagreed
    about the empty case. Deriving it leaves one answer.
    """
    assert Answer(question="q?").abstained_overall is True


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
    c = Claim(text="Margins held steady", confidence=0.5, reason="the chart shows margin falling")

    assert c.reason == "the chart shows margin falling"


def test_claim_reason_defaults_to_none():
    """Additive optional field: every existing construction site still works."""
    assert Claim(text="x", confidence=0.5).reason is None


def test_an_unverified_claim_is_withheld_even_though_it_never_abstained():
    """A Claim that never reached the verifier defaults to abstained=False.
    Reading that as passing is how an unjudged claim would reach a user."""
    assert Claim(text="never judged", confidence=0.0).withheld is True


def test_a_rejected_claim_is_withheld():
    assert Claim(text="x", confidence=0.8, label="unsupported", abstained=True).withheld is True


def test_a_passing_claim_is_not_withheld():
    assert Claim(text="x", confidence=0.9, label="supported", abstained=False).withheld is False


def test_shown_is_the_complement_of_withheld():
    """The display gate and the API's region strip both read Claim.withheld.
    If shown ever stops agreeing with it, one of the two guards is wrong and
    this is the test that says so."""
    claims = [
        Claim(text="passes", confidence=0.9, label="supported"),
        Claim(text="rejected", confidence=0.8, label="unsupported", abstained=True),
        Claim(text="unjudged", confidence=0.0),
    ]
    answer = Answer(question="q", claims=claims)

    assert answer.shown == [c for c in claims if not c.withheld]
    assert [c.text for c in answer.shown] == ["passes"]


def test_an_answer_with_no_verdicts_abstains():
    """THE test of the abstention flag.

    A Claim that never reached the verifier defaults to abstained=False, so the
    old rule, all(c.abstained for c in claims), called this a non-abstention
    with nothing to show. Answer.shown reads Claim.withheld, which is broader on
    purpose, so the two disagreed about the same set of claims. Nothing on the
    answer_stream path can currently produce it, because every claim there gets
    a verdict, but a contract that is only correct because of a caller's habits
    is one refactor from being wrong.
    """
    unverified = Claim(text="never judged", confidence=0.0)
    answer = Answer(question="q", claims=[unverified])

    assert answer.shown == []
    assert answer.abstained_overall is True


def test_the_abstention_flag_cannot_be_set_against_the_claims():
    """It is derived, not stored, so no caller can record an abstention the
    claims contradict or hide one they imply. Constructing it by hand was how
    the two spellings of the rule drifted in the first place."""
    passes = Claim(text="passes", confidence=0.9, label="supported")

    assert Answer(question="q", claims=[passes]).abstained_overall is False
    assert Answer(question="q", claims=[passes], abstained_overall=True).abstained_overall is False


def test_claim_defaults_to_not_starting_a_paragraph():
    """Additive optional field: every existing construction site still works."""
    assert Claim(text="x", confidence=0.5).starts_paragraph is False


def test_a_withheld_lead_abstains_even_when_detail_survived():
    """THE lead rule.

    The first claim is the one that answers the question; the rest support it.
    Showing surviving detail after the answer itself was withheld presents a
    page of context as though it answered a question it never touched, which is
    the fact dump this design exists to remove. A reviewer is likely to read
    this as a bug, which is why it is stated as a test.
    """
    lead = Claim(text="The evaluation runs on SlideVQA.", confidence=0.4, label="unsupported", abstained=True)
    detail = Claim(text="Ground truth is derived automatically.", confidence=0.9, label="supported")
    answer = Answer(question="q", claims=[lead, detail])

    assert answer.shown == [detail]
    assert answer.abstained_overall is True


def test_a_surviving_lead_does_not_abstain():
    lead = Claim(text="The evaluation runs on SlideVQA.", confidence=0.9, label="supported")
    detail = Claim(text="Ground truth is derived automatically.", confidence=0.4, label="unsupported", abstained=True)
    answer = Answer(question="q", claims=[lead, detail])

    assert [c.text for c in answer.shown] == ["The evaluation runs on SlideVQA."]
    assert answer.abstained_overall is False
