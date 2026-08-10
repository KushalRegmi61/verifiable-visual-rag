"""The seam that keeps LangChain out of every module except models.py."""

import pytest

from visual_verify.agent.schemas import ClaimList, Verdict
from visual_verify.agent.types import FakeChat


def test_fake_chat_returns_the_scripted_response():
    chat = FakeChat("fake-reader", [ClaimList(claims=["a", "b"])])
    out = chat.structured("prompt", [], ClaimList)
    assert [c.text for c in out.claims] == ["a", "b"]


def test_fake_chat_returns_scripted_responses_in_order():
    chat = FakeChat(
        "fake",
        [
            Verdict(label="supported", confidence=0.9, reason="r1"),
            Verdict(label="unsupported", confidence=0.8, reason="r2"),
        ],
    )
    assert chat.structured("p", [], Verdict).label == "supported"
    assert chat.structured("p", [], Verdict).label == "unsupported"


def test_fake_chat_raises_when_the_script_runs_out():
    """Silently repeating the last response would make a test that calls the
    model more times than expected still pass."""
    chat = FakeChat("fake", [ClaimList(claims=["a"])])
    chat.structured("p", [], ClaimList)
    with pytest.raises(AssertionError, match="script exhausted"):
        chat.structured("p", [], ClaimList)


def test_fake_chat_records_what_it_was_asked():
    """Lets a test assert the claim and the regions actually reached the model,
    rather than only that a call happened."""
    chat = FakeChat("fake", [ClaimList(claims=["a"])])
    chat.structured("the prompt text", [], ClaimList)
    assert chat.calls[0].prompt == "the prompt text"


def test_fake_chat_records_every_image_it_was_given(tmp_path):
    """One call now carries several pages, so a test asserting only that AN
    image arrived would pass while the seam dropped all but the first."""
    chat = FakeChat("m", [ClaimList(claims=["a"])])
    chat.structured("p", [tmp_path / "one.png", tmp_path / "two.png"], ClaimList)

    assert chat.calls[0].image_paths == [tmp_path / "one.png", tmp_path / "two.png"]


def test_fake_chat_reports_a_model_id():
    """The cache keys on it, and the different-models test asserts on it."""
    assert FakeChat("fake-verifier", []).model_id == "fake-verifier"
