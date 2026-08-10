"""One real call to each provider. Skipped without keys, so a fresh clone runs.

The fake covers behaviour. This covers the thing a fake cannot: that the
request format, the structured-output schema, and the model names are actually
accepted by the live APIs.
"""

import os
from pathlib import Path

import pytest

from helpers import skip_if_no_quota
from visual_verify.agent.schemas import ClaimList, Verdict
from visual_verify.config import Settings

pytestmark = pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") and os.getenv("GOOGLE_API_KEY")),
    reason="needs OPENAI_API_KEY and GOOGLE_API_KEY",
)

FIXTURE = Path(__file__).parent.parent / "data" / "pages"


def _a_page() -> Path:
    """Any decodable page image. Content is not asserted on here, which is why
    this is NOT shared with test_verifier_strictness: that file needs a page
    whose text it knows."""
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
    assert all(c.text.strip() for c in claims)


def test_the_verifier_returns_a_schema_valid_verdict():
    from visual_verify.agent.models import make_chat
    from visual_verify.agent.verifier import verify

    chat = make_chat("verifier", Settings.from_env())
    try:
        v = verify(chat, _a_page(), "This page is blank.", [])
    except Exception as exc:  # noqa: BLE001 - narrowed inside the helper
        skip_if_no_quota(exc)

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
        skip_if_no_quota(exc)

    assert v.label != "supported"


def test_reader_output_parses_as_the_claim_schema():
    """Guards the structured-output path itself: if the provider stops
    honouring the schema, this fails rather than the parse silently
    producing an empty list."""
    from visual_verify.agent.models import make_chat

    chat = make_chat("reader", Settings.from_env())
    out = chat.structured("List two facts about this page.", _a_page(), ClaimList)

    assert isinstance(out, ClaimList)


def test_the_drafted_answer_holds_together():
    """The four drafting rules, against a real model on a real page.

    A fake cannot test this: the rules are instructions to a model and the only
    thing that can fail them is a model. Asserted as a floor rather than a
    quality bar, because "reads well" is not something a test can decide.

    A failure here is not automatically the reader's fault. `shares_content_word`
    errs in BOTH directions, and its known UNDERCOUNTS look exactly like a
    reader that stopped chaining: an "-es" or "-ies" plural is not reunited with
    its singular ("analyses" against "analysis", "policies" against "policy"),
    and a genuinely repeated bare number never becomes a token at all, because
    `_WORD` requires a leading letter. So check the reported claims by eye before
    editing the prompt; the stemmer is as likely a cause as the model. The
    threshold is deliberately not loosened to cover that, since a failure is
    information either way.
    """
    from visual_verify.agent.models import make_chat
    from visual_verify.agent.reader import opens_with_anaphora, read, shares_content_word

    chat = make_chat("reader", Settings.from_env())
    claims = read(chat, _a_page(), "What is this page about?")

    if not claims:
        pytest.skip("the reader found no answer on this page, which is a valid outcome")

    dangling = [c.text for c in claims if opens_with_anaphora(c.text)]
    assert not dangling, f"claims that would break if their predecessor were withheld: {dangling}"

    unchained = [
        claims[i].text
        for i in range(1, len(claims))
        if not shares_content_word(claims[i - 1].text, claims[i].text)
    ]
    # A floor, not a bar. One topic turn in a long answer is legitimate; a
    # majority of unchained claims means the reader is listing, not answering.
    assert len(unchained) <= len(claims) // 3, (
        f"claims sharing nothing with their predecessor: {unchained}"
    )
