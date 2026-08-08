"""Claim extraction, and the compound-claim check the schema cannot do."""

from pathlib import Path

from visual_verify.agent.reader import is_compound, read
from visual_verify.agent.schemas import ClaimList
from visual_verify.agent.types import FakeChat


def test_read_returns_the_models_claims():
    chat = FakeChat("m", [ClaimList(claims=["Revenue grew.", "Margins held."])])
    claims = read(chat, Path("page.png"), "What happened?")
    assert claims == ["Revenue grew.", "Margins held."]


def test_read_sends_the_question_and_the_page_image():
    chat = FakeChat("m", [ClaimList(claims=["a"])])
    read(chat, Path("page.png"), "What is the threshold?")

    call = chat.calls[0]
    assert "What is the threshold?" in call.prompt
    assert call.image_path == Path("page.png")


def test_read_returns_an_empty_list_when_the_page_answers_nothing():
    chat = FakeChat("m", [ClaimList(claims=[])])
    assert read(chat, Path("page.png"), "unrelated question") == []


def test_a_conjunction_joined_claim_is_flagged_as_compound():
    """The schema cannot enforce atomicity. A claim asserting two things
    cannot be grounded to one region, so it must be visible in the eval
    rather than silently accepted."""
    assert is_compound("Revenue grew 42 percent and margins held steady.")


def test_a_single_assertion_is_not_flagged():
    assert not is_compound("Revenue grew 42 percent in the third quarter.")


def test_a_noun_phrase_conjunction_is_not_flagged():
    """'X and Y rose' is one assertion about two subjects, not two claims.
    Flagging it would report a decomposition failure that did not happen."""
    assert not is_compound("Revenue and margin both rose.")
