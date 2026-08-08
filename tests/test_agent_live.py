"""One real call to each provider. Skipped without keys, so a fresh clone runs.

The fake covers behaviour. This covers the thing a fake cannot: that the
request format, the structured-output schema, and the model names are actually
accepted by the live APIs.
"""

import os
from pathlib import Path

import pytest

from visual_verify.agent.schemas import ClaimList, Verdict
from visual_verify.config import Settings

pytestmark = pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") and os.getenv("GOOGLE_API_KEY")),
    reason="needs OPENAI_API_KEY and GOOGLE_API_KEY",
)

FIXTURE = Path(__file__).parent.parent / "data" / "pages"


def _skip_if_no_quota(exc: Exception) -> None:
    """Turn an unreachable provider into a skip, but never a wrong answer.

    A 429 with `limit: 0` means the key's project has no quota for the model at
    all, which is a billing state and not a defect in this code. Failing on it
    would leave the suite red on any machine whose account is not provisioned,
    including a fresh clone.

    Deliberately narrow. Only transport and quota problems skip. A response
    that arrives and is malformed, or a verdict that is simply wrong, must
    still FAIL: those are the two things this file exists to catch, and
    swallowing them would make a broken verifier look like an unconfigured one.
    """
    text = str(exc)
    unreachable = (
        "RESOURCE_EXHAUSTED" in text
        or "429" in text
        or "insufficient_quota" in text
        or "rate limit" in text.lower()
    )
    if unreachable:
        pytest.skip(f"provider reachable but unprovisioned: {text[:160]}")
    raise exc


def _a_page() -> Path:
    if not FIXTURE.exists():
        pytest.skip("no rendered pages; run `vvrag ingest` first")
    pages = sorted(FIXTURE.rglob("*.png"))
    if not pages:
        pytest.skip("no rendered pages found")
    return pages[0]


def test_the_reader_returns_schema_valid_claims():
    from visual_verify.agent.models import make_chat
    from visual_verify.agent.reader import read

    chat = make_chat("reader", Settings.from_env())
    claims = read(chat, _a_page(), "What is this page about?")

    assert isinstance(claims, list)
    assert all(isinstance(c, str) and c.strip() for c in claims)


def test_the_verifier_returns_a_schema_valid_verdict():
    from visual_verify.agent.models import make_chat
    from visual_verify.agent.verifier import verify

    chat = make_chat("verifier", Settings.from_env())
    try:
        v = verify(chat, _a_page(), "This page is blank.", [])
    except Exception as exc:  # noqa: BLE001 - narrowed inside the helper
        _skip_if_no_quota(exc)

    assert isinstance(v, Verdict)
    assert v.label in {"supported", "partially_supported", "unsupported", "insufficient_evidence"}
    assert v.reason.strip()


def test_the_two_roles_really_are_different_models():
    """Config-level assertion against the live settings, not a fake."""
    from visual_verify.agent.models import make_chat

    s = Settings.from_env()
    assert make_chat("reader", s).model_id != make_chat("verifier", s).model_id


def test_a_blank_claim_about_a_real_page_is_not_supported():
    """The live verifier must be capable of saying no, not only the fake."""
    from visual_verify.agent.models import make_chat
    from visual_verify.agent.verifier import verify

    chat = make_chat("verifier", Settings.from_env())
    try:
        v = verify(chat, _a_page(), "This page is a photograph of a cat.", [])
    except Exception as exc:  # noqa: BLE001 - narrowed inside the helper
        _skip_if_no_quota(exc)

    assert v.label != "supported"


def test_reader_output_parses_as_the_claim_schema():
    """Guards the structured-output path itself: if the provider stops
    honouring the schema, this fails rather than the parse silently
    producing an empty list."""
    from visual_verify.agent.models import make_chat

    chat = make_chat("reader", Settings.from_env())
    out = chat.structured("List two facts about this page.", _a_page(), ClaimList)

    assert isinstance(out, ClaimList)
