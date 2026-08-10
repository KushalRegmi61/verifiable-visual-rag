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

    Smoke coverage only, and it can go VACUOUS. "What is this page about?" is
    legitimately answerable in one sentence, and on a one-claim answer there are
    no adjacent pairs, so `unchained` is empty against a floor of `1 // 3 == 0`
    and `dangling` is empty because there is nothing to dangle from. Both
    assertions then pass on any output whatsoever. The non-vacuous version is
    `test_a_multi_part_answer_is_not_collapsed_into_one_claim` below, which asks
    a question one sentence cannot answer.

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


def _methodology_page() -> Path:
    """proposal.pdf page 14, whose true answer is genuinely multi-part.

    Resolved through the store rather than hardcoded, because the page lives
    under its document's sha256 and that hash changes whenever the proposal is
    recompiled. Skips instead of failing when the document is not ingested, so
    a fresh clone still runs.
    """
    import sqlalchemy as sa

    from visual_verify.store.engine import make_engine

    settings = Settings.from_env()
    with make_engine(settings.db_url).connect() as conn:
        rows = conn.execute(sa.text("select sha256, path from documents")).all()

    for sha, path in rows:
        if str(path).endswith("proposal_report/proposal.pdf"):
            page = FIXTURE / sha / "p0014.png"
            if not page.exists():
                pytest.skip(f"{page} is not rendered; run `vvrag ingest` first")
            return page
    pytest.skip("proposal_report/proposal.pdf is not ingested")


def test_a_multi_part_answer_is_not_collapsed_into_one_claim():
    """Rule 1 and rule 2 pull against each other, and this is where it shows.

    Rule 1 is a COMPLETENESS test and rule 2 is an ATOMICITY test, and for a
    question whose true answer has several parts they cannot both hold in one
    sentence. The first version of this prompt let rule 1 win: it returned a
    single 37-word claim asserting which systems are evaluated, which three
    metrics are used, and why the ablation exists, grounded to a region of
    0.311 by 0.012 normalised. That is about 397 by 20 px on the 150 dpi
    render, one line of body text, roughly 10 to 12 words. The verifier then
    judged it `supported` at 0.90, so the system emitted a confident,
    region-cited, three-part claim behind evidence that can carry at most one
    part. Nothing raised and the overlay looked right.

    The word cap is a crude atomicity floor, and it is NOT rule-2 coverage.
    `is_compound` returned False on that 37-word claim and would return False
    however `_VERB` grows, because the sentence is a compound OBJECT plus a
    participial adjunct plus a purpose clause, none of which is a coordinated
    clause. Length is the only cheap signal that saw it, and a short sentence
    can still assert two things, so do not read either assertion as proof of
    atomicity.

    Chaining is deliberately NOT asserted here. Measured on this question with
    the fixed prompt: 4 claims, 1 of 3 adjacent pairs sharing a content word,
    which is over the floor the other test uses. That is a real property of a
    metrics-then-ablation answer rather than a stemmer artifact (the unchained
    pairs share nothing at all), and pinning it would fail this test for a
    reason it does not exist to police.
    """
    from visual_verify.agent.models import make_chat
    from visual_verify.agent.reader import is_compound, opens_with_anaphora, read

    chat = make_chat("reader", Settings.from_env())
    claims = read(chat, _methodology_page(), "What is the evaluation methodology?")
    texts = [c.text for c in claims]

    assert len(claims) >= 3, f"a multi-part answer collapsed into {len(claims)} claim(s): {texts}"

    too_long = [t for t in texts if len(t.split()) > 30]
    assert not too_long, f"claims too long for one region of evidence to carry: {too_long}"

    compound = [t for t in texts if is_compound(t)]
    assert not compound, f"claims asserting more than one thing: {compound}"

    dangling = [t for t in texts if opens_with_anaphora(t)]
    assert not dangling, f"claims that would break if their predecessor were withheld: {dangling}"

    # Rule 1, pinned against the exact before-state this prompt replaced: six
    # claims describing the page, three of them opening "Figure 5.1 includes".
    lead = texts[0].lower()
    assert not lead.startswith(("this page", "the page", "figure")), (
        f"the first claim describes the page instead of answering: {texts[0]}"
    )
