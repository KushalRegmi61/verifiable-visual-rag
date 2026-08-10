"""What leaves the process.

The withheld-region strip lives here rather than in answer(), because S7 needs
the regions of rejected claims to compute confident-wrong against coverage.
Putting the guarantee in the core would break the eval instead of protecting
the user, so it belongs at the boundary the data crosses.
"""

from visual_verify.agent.events import AnswerComplete, ClaimsProduced, ClaimVerified, ReadingStarted
from visual_verify.api.wire import to_frame
from visual_verify.contracts import Answer, Claim, GroundedRegion


def region(score=1.0):
    return GroundedRegion(
        page=0,
        bbox=(0.1, 0.2, 0.3, 0.4),
        score=score,
        modality="text",
        text="Revenue grew 42 percent",
        resolution="line",
    )


def shown_claim():
    return Claim(
        text="Revenue grew 42 percent",
        regions=[region()],
        confidence=0.9,
        label="supported",
        reason="matches the table",
        abstained=False,
    )


def withheld_claim():
    return Claim(
        text="Margins held steady",
        regions=[region()],
        confidence=0.8,
        label="unsupported",
        reason="the chart shows margin falling",
        abstained=True,
    )


def test_a_withheld_claim_carries_no_regions():
    """THE test of this module. A rejected claim's geometry must not reach the
    browser at all: styling it differently is not a guarantee, because the
    frontend would then be trusted not to draw what it was handed."""
    name, payload = to_frame(ClaimVerified(index=1, claim=withheld_claim()))

    assert payload["withheld"] is True
    assert payload["regions"] == []


def test_a_withheld_claim_still_carries_its_label_and_reason():
    """A bare count tells a user nothing. The reason is what S5 built to make
    a wrong verdict debuggable and this is the only surface it reaches."""
    _, payload = to_frame(ClaimVerified(index=1, claim=withheld_claim()))

    assert payload["label"] == "unsupported"
    assert payload["reason"] == "the chart shows margin falling"


def test_a_shown_claim_keeps_its_regions():
    name, payload = to_frame(ClaimVerified(index=0, claim=shown_claim()))

    assert name == "claim"
    assert payload["withheld"] is False
    assert len(payload["regions"]) == 1


def test_a_region_carries_the_fields_the_overlay_needs():
    """resolution and modality exist so a coarse block fallback is
    distinguishable from a confident line hit. Dropping either here makes S4's
    bounded-error property invisible to the only human who ever sees it."""
    _, payload = to_frame(ClaimVerified(index=0, claim=shown_claim()))
    r = payload["regions"][0]

    assert r["bbox"] == [0.1, 0.2, 0.3, 0.4]
    assert r["modality"] == "text"
    assert r["resolution"] == "line"


def test_reading_and_claims_events_map_to_their_names():
    assert to_frame(ReadingStarted())[0] == "reading"
    name, payload = to_frame(ClaimsProduced(n=3))
    assert name == "claims"
    assert payload == {"n": 3}


def test_done_counts_shown_against_withheld():
    complete = AnswerComplete(
        answer=Answer(
            question="q",
            claims=[shown_claim(), withheld_claim()],
            abstained_overall=False,
        )
    )

    name, payload = to_frame(complete)

    assert name == "done"
    assert payload == {"shown": 1, "withheld": 1, "abstained_overall": False}


def test_done_counts_use_shown_not_the_abstained_flag():
    """Answer.shown requires a verdict as well as not-abstained, because a
    Claim that never reached the verifier defaults to abstained=False and
    would otherwise be counted as passing."""
    unverified = Claim(text="never judged", confidence=0.0)
    complete = AnswerComplete(
        answer=Answer(question="q", claims=[unverified], abstained_overall=True)
    )

    _, payload = to_frame(complete)

    assert payload["shown"] == 0


def test_a_retrieved_event_names_the_page_and_the_alternatives():
    from pathlib import Path

    from visual_verify.api.ask import Retrieved
    from visual_verify.contracts import RetrievedPage
    from visual_verify.prepare import PreparedPage

    prepared = PreparedPage(
        doc_sha="abc123",
        doc_name="proposal.pdf",
        page_no=3,
        image_path=Path("p.png"),
        boxes=[],
        page_vectors=None,
        grid=None,
    )
    other = RetrievedPage(doc_id="abc123", page=7, image_ref="p7.png", score=8.1)
    elsewhere = RetrievedPage(doc_id="def456", page=24, image_ref="q24.png", score=7.2)

    name, payload = to_frame(
        Retrieved(
            page=prepared,
            score=9.4,
            candidates=[other, elsewhere],
            doc_names={"abc123": "proposal.pdf", "def456": "reference_proposal.pdf"},
        )
    )

    assert name == "retrieved"
    assert payload["doc_sha"] == "abc123"
    assert payload["page"] == 3
    assert payload["score"] == 9.4
    # doc_name on every candidate, because retrieval is corpus-wide: a chip
    # showing only "page 24" reads as page 24 of the document on screen, and
    # clicking it swaps the document with no indication. Found by opening the
    # UI, not by a test.
    assert payload["candidates"] == [
        {"doc_sha": "abc123", "page": 7, "score": 8.1, "doc_name": "proposal.pdf"},
        {"doc_sha": "def456", "page": 24, "score": 7.2, "doc_name": "reference_proposal.pdf"},
    ]


def test_a_candidate_with_no_name_falls_back_to_a_short_sha():
    """to_frame must not raise on a sha the lookup missed. A KeyError here would
    kill the stream after the 200 was committed, turning a cosmetic gap into a
    dead request."""
    from pathlib import Path

    from visual_verify.api.ask import Retrieved
    from visual_verify.contracts import RetrievedPage
    from visual_verify.prepare import PreparedPage

    prepared = PreparedPage(
        doc_sha="abc123",
        doc_name="proposal.pdf",
        page_no=3,
        image_path=Path("p.png"),
        boxes=[],
        page_vectors=None,
        grid=None,
    )
    orphan = RetrievedPage(doc_id="0123456789abcdef", page=1, image_ref="x.png", score=1.0)

    _, payload = to_frame(Retrieved(page=prepared, score=9.4, candidates=[orphan]))

    assert payload["candidates"][0]["doc_name"] == "0123456789ab"


def test_a_claim_carries_its_paragraph_break():
    """The UI cannot infer a topic turn from the text, and the reader is the
    only thing that knows where one is. Dropping the flag here silently
    collapses every answer back into one block."""
    from visual_verify.agent.events import ClaimVerified

    c = Claim(
        text="The ablation removes the added layer.",
        regions=[region()],
        confidence=0.9,
        label="supported",
        reason="stated in the evidence",
        abstained=False,
        starts_paragraph=True,
    )

    _, payload = to_frame(ClaimVerified(index=2, claim=c))

    assert payload["starts_paragraph"] is True


def test_a_withheld_claim_still_carries_its_paragraph_break():
    """It is stripped of regions, not of everything. The UI reads the flag for
    every claim it lays out, and a withheld claim that reported the wrong
    break would shift the paragraph break onto the wrong sentence."""
    from visual_verify.agent.events import ClaimVerified

    c = withheld_claim()
    c = c.model_copy(update={"starts_paragraph": True})

    _, payload = to_frame(ClaimVerified(index=1, claim=c))

    assert payload["regions"] == []
    assert payload["starts_paragraph"] is True


def test_a_withheld_lead_announces_the_abstention_on_its_own_claim_frame():
    """The browser must learn the answer is abstaining from the first claim,
    not from `done`. _answer_events emits every ClaimVerified before
    AnswerComplete, so a UI that waits for `done` renders the surviving support
    under an "Answer" heading for one verifier call per remaining claim and
    then retracts it, which is exactly the framing the lead rule forbids."""
    _, payload = to_frame(ClaimVerified(index=0, claim=withheld_claim()))

    assert payload["abstains_answer"] is True


def test_a_withheld_claim_after_the_lead_does_not_announce_an_abstention():
    """Support failing verification is not a refusal. If this were true for any
    withheld claim, an answer whose lead survived would still decline, and the
    system would refuse to answer questions it had answered."""
    _, payload = to_frame(ClaimVerified(index=2, claim=withheld_claim()))

    assert payload["abstains_answer"] is False


def test_a_surviving_lead_does_not_announce_an_abstention():
    """The other half of the conjunction. Position alone must not abstain, or
    every answer would decline on its own first claim."""
    _, payload = to_frame(ClaimVerified(index=0, claim=shown_claim()))

    assert payload["abstains_answer"] is False
