# Answer Composition and Honest Abstention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the displayed answer read as connected prose an assistant wrote, while every displayed word stays text a second model verified against a region of the page, and make the refusal case read as a decision rather than as an empty screen.

**Architecture:** All composition happens in the reader, in the one call it already makes. It drafts a connected answer and returns it one sentence per claim, so each sentence is still verified alone and still maps to one region. Two mechanical checks measure whether the prose holds together. The verifier prompt is tightened first, because if it never rejects anything then none of the display logic downstream ever runs. The refusal case gains a lead-claim rule and fixed copy that only ever describes the system's own process.

**Tech Stack:** Python 3.12, pydantic v2, pytest, uv. Next.js 16, React 19, TypeScript, vitest.

**Spec:** `docs/superpowers/specs/2026-08-10-answer-composition-design.md`

---

## Background an engineer needs before starting

**The pipeline.** A question goes to retrieval, which ranks page images. One page is read by a reader VLM, which returns claims. Each claim is grounded to a rectangle on the page. Each claim is then judged by a *different* VLM, the verifier. A claim whose score falls below a threshold is withheld and never reaches the browser with its geometry.

**Why claims and not prose.** The displayed answer is the claims joined. There is no separate prose answer, so nothing can drift between what is shown and what was verified. Read the module docstring at the top of `src/visual_verify/agent/reader.py`.

**The constraint that shapes everything.** Verification runs *after* drafting, so any sentence can be removed before display. A sentence that opens with "It" or "Each of them" becomes meaningless when its predecessor is withheld. That is why the reader is told to chain sentences by repeating a noun phrase instead of by pronoun.

**Repo rules that will trip you up.**
- Never commit `CLAUDE.md`. It is gitignored.
- Never put Claude or AI attribution in a commit message. No `Co-Authored-By`, no "Generated with".
- No em-dashes anywhere in prose, comments, or UI copy. This is enforced by review.
- Conventional Commits. Scopes in use here: `agent`, `contracts`, `api`, `ui`, `cli`.
- Commit bodies explain WHY. Most bugs in this repo produced output that looked correct, so the body is often the only record of how a failure presented.
- After a same-length edit during mutation testing, clear bytecode caches or you will measure the old code:
  `find . -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +`

**Commands.**
```bash
uv sync --all-extras --group dev
uv run pytest tests/test_reader.py -v
uv run ruff check src tests && uv run ruff format --check src tests
cd frontend && npx vitest run && npx tsc --noEmit && npm run build
```

**Do not run the whole suite after every task.** Three test modules load ColQwen2 onto a 4 GB GPU and take about ten minutes. Run only the files your task touches. The full suite runs once, at the end.

```bash
# the fast suite, used throughout
uv run pytest -q --ignore=tests/test_embedder.py \
  --ignore=tests/test_known_item_retrieval.py \
  --ignore=tests/test_grounding_live.py
```
Baseline before you start: **463 passed, 5 skipped**.

---

## File structure

| File | Responsibility after this plan |
|---|---|
| `src/visual_verify/agent/schemas.py` | `DraftedClaim` (text plus `starts_paragraph`), `ClaimList` holding them |
| `src/visual_verify/agent/reader.py` | the drafting prompt, `is_compound`, `opens_with_anaphora`, `shares_content_word`, `read()` |
| `src/visual_verify/agent/verifier.py` | the strict judging prompt |
| `src/visual_verify/agent/core.py` | unchanged loop, carrying `starts_paragraph` onto `Claim` |
| `src/visual_verify/contracts.py` | `Claim.starts_paragraph`, lead rule inside `Answer.abstained_overall` |
| `src/visual_verify/api/wire.py` | emits `starts_paragraph` |
| `frontend/lib/claims.ts` | `groupIntoParagraphs`, pure and unit tested |
| `frontend/components/AnswerPanel.tsx` | renders paragraphs and the partial-state line |
| `frontend/app/page.tsx` | the three answer states and their copy |
| `tests/test_verifier_strictness.py` | new, live, the four probes |

---

## Task 1: Verifier strictness probe

Do this first. If the verifier labels everything `supported`, the lead rule never fires and every other task in this plan builds display logic nobody will ever see. Across the live runs on 2026-08-10, roughly 30 of 32 claims came back `supported` at 90 to 100 percent confidence, which is equally consistent with an easy corpus and with a judge that rubber-stamps.

**Files:**
- Create: `tests/test_verifier_strictness.py`
- Modify: `src/visual_verify/agent/verifier.py:18-32` (the `PROMPT` constant)

- [ ] **Step 1: Write the probe file**

This is a live test. It calls a real provider and is skipped without keys, matching the pattern already in `tests/test_agent_live.py`.

```python
"""Can the verifier say no?

The rubric has four labels and an abstention threshold built on top of them,
and all of that is decoration if the judge approves whatever it is handed.
Measured on 2026-08-10, roughly 30 of 32 claims in live runs came back
supported at 90 to 100 percent. That is equally consistent with an easy corpus,
where the reader only claims what it can see, and with a lenient judge. These
probes separate the two by handing the verifier claims that are false in a
specific way.

Every probe asserts `label != "supported"` rather than a specific label.
Which of the three non-supported labels fits a given probe is a judgement call,
and pinning one would make the test brittle for no gain.
"""

import os
from pathlib import Path

import pytest

from visual_verify.config import Settings
from visual_verify.contracts import GroundedRegion

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="needs OPENAI_API_KEY for the verifier",
)

FIXTURE = Path(__file__).parent.parent / "data" / "pages"


def _a_page() -> Path:
    if not FIXTURE.exists():
        pytest.skip("no rendered pages; run `vvrag ingest` first")
    pages = sorted(FIXTURE.rglob("*.png"))
    if not pages:
        pytest.skip("no rendered pages found")
    return pages[0]


def _region(text: str) -> GroundedRegion:
    return GroundedRegion(
        page=0,
        bbox=(0.13, 0.30, 0.87, 0.32),
        score=1.0,
        modality="text",
        text=text,
        resolution="line",
    )


def _judge(claim: str, region_text: str) -> str:
    from visual_verify.agent.models import make_chat
    from visual_verify.agent.verifier import verify

    chat = make_chat("verifier", Settings.from_env())
    try:
        verdict = verify(chat, _a_page(), claim, [_region(region_text)])
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is a quota state
        text = str(exc)
        if "429" in text or "RESOURCE_EXHAUSTED" in text or "insufficient_quota" in text:
            pytest.skip(f"provider reachable but unprovisioned: {text[:160]}")
        raise
    return verdict.label


EVIDENCE = "Evaluation on SlideVQA with three metrics"


def test_the_control_probe_is_supported():
    """The floor. If this fails the probes below prove nothing, because a
    verifier that rejects everything passes all four of them."""
    assert _judge("The evaluation uses SlideVQA.", EVIDENCE) == "supported"


def test_a_changed_number_is_not_supported():
    assert _judge("The evaluation reports seven metrics.", EVIDENCE) != "supported"


def test_a_swapped_entity_is_not_supported():
    assert _judge("The evaluation runs on DocVQA.", EVIDENCE) != "supported"


def test_absent_content_is_not_supported():
    assert _judge("The system uses conformal calibration.", EVIDENCE) != "supported"


def test_true_of_the_page_but_absent_from_the_regions_is_not_supported():
    """THE probe. The other three are false anywhere. This one may well be true
    of the page image the verifier is also shown, and it is not established by
    the region it was handed. A verifier reading the image instead of the
    evidence passes the first three and fails this one, which is the exact
    failure mode that would make grounding decorative.
    """
    assert _judge("The evaluation reports exact match and F1.", EVIDENCE) != "supported"
```

- [ ] **Step 2: Run the probes and record what happens**

```bash
set -a && . ./.env && set +a
uv run pytest tests/test_verifier_strictness.py -v 2>&1 | tail -20
```

Two outcomes, and both are information:
- All five pass. The verifier is already strict; skip step 3 and commit the probes as a regression guard.
- The control passes and one or more of the other four returns `supported`. Continue to step 3.

- [ ] **Step 3: Tighten the verifier prompt**

Replace `PROMPT` in `src/visual_verify/agent/verifier.py`:

```python
PROMPT = """You are checking whether a claim is supported by specific evidence
from a document page. You did not write the claim. Your job is to catch claims
that should not be shown to a user.

Claim: {claim}

Evidence regions selected from the page:
{evidence}

Judge the claim against the EVIDENCE REGIONS, not against the page as a whole
and not against what you already know. If the regions do not establish the
claim, the label is not supported, even when the claim looks correct and even
when you can see it elsewhere on the page.

Choose exactly one label:
- supported: the evidence regions establish the whole claim
- partially_supported: the regions establish part of the claim, or establish it
  only with a qualification the claim leaves out
- unsupported: the regions contradict the claim, or are about something else
- insufficient_evidence: there is not enough here to judge

supported is the strongest label and the only one shown by default. Do not use
it for a claim you merely believe to be true. Use it when you can point at the
evidence and say the claim follows from it.

Give a confidence between 0 and 1, and one sentence of reasoning that names the
part of the evidence you relied on."""
```

- [ ] **Step 4: Re-run the probes**

```bash
uv run pytest tests/test_verifier_strictness.py -v 2>&1 | tail -20
```
Expected: 5 passed. If the control now fails, the prompt has overcorrected into rejecting everything; soften "Do not use it for a claim you merely believe to be true" and re-run.

- [ ] **Step 5: Confirm nothing else regressed**

```bash
uv run pytest tests/test_agent.py tests/test_answer_stream.py tests/test_agent_live.py -q
```
Expected: all pass. `tests/test_agent.py` uses `FakeChat` and does not touch the prompt text, so it is unaffected; `tests/test_agent_live.py` calls the real verifier and is the one that could move.

- [ ] **Step 6: Commit**

```bash
git add tests/test_verifier_strictness.py src/visual_verify/agent/verifier.py
git commit -F - <<'EOF'
test(agent): probe whether the verifier can actually say no

The four-label rubric and the abstention threshold built on it are decoration
if the judge approves whatever it is handed. Across live runs on 2026-08-10
roughly 30 of 32 claims came back supported at 90 to 100 percent, which is
equally consistent with an easy corpus and with a judge that rubber-stamps,
and nothing distinguished the two.

Four probes hand it claims that are false in a specific way, plus a control so
that a verifier which rejects everything cannot pass by accident. The probe
that matters is the last one: a claim that may well be true of the page image
the verifier is also shown, and that the region it was handed does not
establish. A judge reading the image instead of the evidence passes the other
three and fails that one, which is the failure that would make grounding
decorative.

Each probe asserts the label is not supported rather than naming which of the
three non-supported labels is right, because that is a judgement call and
pinning it would be brittle for no gain.
EOF
```

---

## Task 2: DraftedClaim, so a claim can carry a paragraph break

**Files:**
- Modify: `src/visual_verify/agent/schemas.py:14-25`
- Modify: `src/visual_verify/agent/reader.py:78-81`
- Modify: `src/visual_verify/agent/core.py:116-157`
- Modify: `src/visual_verify/contracts.py` (class `Claim`)
- Modify: `tests/test_reader.py`, `tests/test_agent_schemas.py`, `tests/test_agent_cache.py`, `tests/test_agent_types.py`, `tests/test_agent_live.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_schemas.py`:

```python
def test_a_claim_carries_a_paragraph_break_flag():
    from visual_verify.agent.schemas import ClaimList

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
    from visual_verify.agent.schemas import ClaimList

    parsed = ClaimList(claims=["Revenue grew 42 percent."])

    assert parsed.claims[0].text == "Revenue grew 42 percent."
    assert parsed.claims[0].starts_paragraph is False
```

Append to `tests/test_contracts.py`:

```python
def test_claim_defaults_to_not_starting_a_paragraph():
    """Additive optional field: every existing construction site still works."""
    assert Claim(text="x", confidence=0.5).starts_paragraph is False
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest tests/test_agent_schemas.py::test_a_claim_carries_a_paragraph_break_flag \
  tests/test_contracts.py::test_claim_defaults_to_not_starting_a_paragraph -v
```
Expected: FAIL. The first with a pydantic `ValidationError` about a dict where a string was expected, the second with `AttributeError: 'Claim' object has no attribute 'starts_paragraph'`.

- [ ] **Step 3: Add `DraftedClaim` to `schemas.py`**

Replace the `ClaimList` class in `src/visual_verify/agent/schemas.py`:

```python
class DraftedClaim(BaseModel):
    """One sentence of the drafted answer.

    `starts_paragraph` is metadata, not text. It adds nothing unverified to the
    screen and leaves every sentence mapped to exactly one region, which is
    what keeps hover-to-region and click-to-evidence working. It exists because
    an answer long enough to cover two topics reads badly as one block.
    """

    text: str = Field(min_length=1)
    starts_paragraph: bool = False


class ClaimList(BaseModel):
    """The reader's output: the drafted answer, one sentence per claim."""

    claims: list[DraftedClaim] = Field(default_factory=list)

    @field_validator("claims", mode="before")
    @classmethod
    def _accept_bare_strings(cls, v):
        """A plain string becomes a claim that starts no paragraph.

        Around forty construction sites across the test suite pass strings, and
        a provider is free to return them too. Rewriting all of those to
        dictionaries would be a large diff that tests nothing, and the default
        is the correct one either way.
        """
        if isinstance(v, list):
            return [{"text": c} if isinstance(c, str) else c for c in v]
        return v

    @field_validator("claims")
    @classmethod
    def _no_blank_claims(cls, v: list[DraftedClaim]) -> list[DraftedClaim]:
        blank = [i for i, c in enumerate(v) if not c.text.strip()]
        if blank:
            raise ValueError(f"claims at {blank} are blank; the reader returned junk")
        return v
```

- [ ] **Step 4: Add the field to `Claim` in `contracts.py`**

Insert after the `reason` field of class `Claim`:

```python
    # True when this claim opens a new paragraph in the displayed answer. Set
    # by the reader, carried through verification untouched, and read only by
    # the UI. Metadata rather than text, so it adds nothing unverified to the
    # screen and every sentence still maps to exactly one region.
    starts_paragraph: bool = False
```

- [ ] **Step 5: Return drafted claims from `read()`**

In `src/visual_verify/agent/reader.py`, change the import and the function:

```python
from visual_verify.agent.schemas import ClaimList, DraftedClaim
```

```python
def read(chat: StructuredChat, image_path: Path, question: str) -> list[DraftedClaim]:
    """The drafted answer for `question`, one sentence per claim, in reading order."""
    out = chat.structured(PROMPT.format(question=question), image_path, ClaimList)
    return list(out.claims)
```

- [ ] **Step 6: Carry it through `core.py`**

In `src/visual_verify/agent/core.py`, in `_stream`, change the loop. `read()` now returns objects, so `texts` becomes `drafted` and the claim text is `d.text`:

```python
    drafted = read(reader_chat, image_path, question)
    yield ClaimsProduced(n=len(drafted))

    claims: list[Claim] = []
    for index, d in enumerate(drafted):
        text = d.text
        query_vectors = embed_query(text) if embed_query is not None else None
```

and in the `Claim(...)` construction inside that loop, add one keyword:

```python
            compound=is_compound(text),
            starts_paragraph=d.starts_paragraph,
```

- [ ] **Step 7: Update the test sites that read claim text**

These assert on the *contents* of a `ClaimList` or of `read()`'s return, so coercion does not save them. Construction sites that only pass strings are unaffected and must not be touched.

`tests/test_reader.py`, three assertions:

```python
def test_read_returns_the_models_claims():
    chat = FakeChat("m", [ClaimList(claims=["Revenue grew.", "Margins held."])])
    claims = read(chat, Path("page.png"), "What happened?")
    assert [c.text for c in claims] == ["Revenue grew.", "Margins held."]
```
```python
def test_read_returns_an_empty_list_when_the_page_answers_nothing():
    chat = FakeChat("m", [ClaimList(claims=[])])
    assert read(chat, Path("page.png"), "unrelated question") == []
```
Leave `test_read_sends_the_question_and_the_page_image` alone; it reads `chat.calls`, not claims.

`tests/test_agent_schemas.py:11`, change:
```python
    parsed = ClaimList(claims=["Revenue grew 42 percent.", "Margins held steady."])
    assert [c.text for c in parsed.claims] == [
        "Revenue grew 42 percent.",
        "Margins held steady.",
    ]
```

`tests/test_agent_cache.py`, seven assertions at lines 23, 24, 33, 34, 44, 57 and 58. Each reads `.claims[0].text` instead of comparing the list to a list of strings:
```python
    assert chat.structured("p1", None, ClaimList).claims[0].text == "a"
    assert chat.structured("p2", None, ClaimList).claims[0].text == "b"
```
```python
    assert CachedChat(a, tmp_path).structured("p", None, ClaimList).claims[0].text == "from-a"
    assert CachedChat(b, tmp_path).structured("p", None, ClaimList).claims[0].text == "from-b"
```
```python
    assert CachedChat(cold, tmp_path).structured("p", None, ClaimList).claims[0].text == "a"
```
```python
    assert chat.structured("p", one, ClaimList).claims[0].text == "a"
    assert chat.structured("p", two, ClaimList).claims[0].text == "b"
```

`tests/test_agent_types.py:11`:
```python
    assert [c.text for c in out.claims] == ["a", "b"]
```

`tests/test_agent_live.py:58-66`:
```python
def test_the_reader_returns_schema_valid_claims():
    from visual_verify.agent.models import make_chat
    from visual_verify.agent.reader import read

    chat = make_chat("reader", Settings.from_env())
    claims = read(chat, _a_page(), "What is this page about?")

    assert isinstance(claims, list)
    assert all(c.text.strip() for c in claims)
```

- [ ] **Step 8: Run the fast suite**

```bash
uv run pytest -q --ignore=tests/test_embedder.py \
  --ignore=tests/test_known_item_retrieval.py \
  --ignore=tests/test_grounding_live.py
```
Expected: 466 passed, 5 skipped. If any test fails with `AttributeError: 'DraftedClaim' object has no attribute 'strip'`, that site reads claim text and was missed in step 7.

- [ ] **Step 9: Commit**

```bash
git add src/visual_verify/agent/schemas.py src/visual_verify/agent/reader.py \
  src/visual_verify/agent/core.py src/visual_verify/contracts.py tests/
git commit -F - <<'EOF'
feat(agent): let a drafted claim carry a paragraph break

An answer long enough to cover two topics reads badly as one block, and the
reader is the only thing that knows where the topic turns. starts_paragraph is
metadata rather than text, so it adds nothing unverified to the screen and
every sentence still maps to exactly one region, which is what keeps
hover-to-region and click-to-evidence working.

ClaimList still accepts a bare string and coerces it. Around forty
construction sites across the suite pass strings, rewriting them all would be
a large diff that tests nothing, and a provider that returns strings gets the
correct default either way. Only the sites that read claim text changed.
EOF
```

---

## Task 3: `opens_with_anaphora`

**Files:**
- Modify: `src/visual_verify/agent/reader.py`
- Modify: `tests/test_reader.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reader.py`:

```python
def test_a_sentence_opening_with_a_pronoun_is_flagged():
    from visual_verify.agent.reader import opens_with_anaphora

    assert opens_with_anaphora("It also isolates the added layer.") is True
    assert opens_with_anaphora("They are scored on three metrics.") is True
    assert opens_with_anaphora("This approach avoids annotation.") is True
    assert opens_with_anaphora("Additionally, the ablation removes the layer.") is True


def test_a_sentence_merely_containing_a_pronoun_is_not_flagged():
    """THE discriminating test. A substring match anywhere in the sentence
    flags almost every English sentence, which would make the check useless
    while looking like it works. Only the OPENING can dangle, because only the
    opening is what a reader resolves against the previous sentence.
    """
    from visual_verify.agent.reader import opens_with_anaphora

    assert opens_with_anaphora("The variant that supports it is Grounded RAG.") is False
    assert opens_with_anaphora("The metrics and their definitions appear below.") is False


def test_a_self_contained_sentence_is_not_flagged():
    from visual_verify.agent.reader import opens_with_anaphora

    assert opens_with_anaphora("The evaluation compares three system variants.") is False
    assert opens_with_anaphora("Each of the three variants is scored separately.") is False


def test_a_word_starting_with_a_pronoun_is_not_flagged():
    """Word boundary, not prefix. "Itemised" and "Thistle" both begin with a
    pronoun's letters and neither is anaphora."""
    from visual_verify.agent.reader import opens_with_anaphora

    assert opens_with_anaphora("Itemised costs appear in Table 2.") is False
    assert opens_with_anaphora("Those results are listed in Table 2.") is True
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest tests/test_reader.py -k anaphora -v
```
Expected: FAIL with `ImportError: cannot import name 'opens_with_anaphora'`.

- [ ] **Step 3: Implement it**

Add to `src/visual_verify/agent/reader.py`, below `_VERB`:

```python
# Sentence-initial only. A pronoun in the MIDDLE of a sentence resolves within
# that sentence and is fine; it is the opening that a reader resolves against
# whatever came before, and whatever came before may have been withheld.
_ANAPHORA = re.compile(
    r"^\s*(?:it|its|they|them|their|this|these|those|such|"
    r"additionally|also|furthermore|moreover|however)\b",
    re.IGNORECASE,
)


def opens_with_anaphora(claim: str) -> bool:
    """Whether this claim depends on the sentence before it to make sense.

    Verification runs after drafting, so any claim may be removed before the
    answer is shown. A claim opening with "It" or "Each of them" is meaningless
    once its predecessor is withheld, and it does not announce that it is
    broken: it stays grammatical and quietly says something other than what was
    verified.

    Flagged, never rejected. The same rule `is_compound` follows: dropping the
    claim would lose verified content with a real region behind it, and the
    useful response is to surface the rate in the eval and fix the prompt.

    `\\b` matters. A prefix match would flag "Itemised costs appear in Table 2",
    where "It" is the first two letters of a word and nothing is dangling.
    """
    return bool(_ANAPHORA.match(claim))
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_reader.py -k anaphora -v
```
Expected: 4 passed.

- [ ] **Step 5: Verify the test discriminates**

Temporarily change `_ANAPHORA.match(claim)` to `_ANAPHORA.search(claim)` and re-run.

```bash
find . -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
uv run pytest tests/test_reader.py -k anaphora -v
```
Expected: `test_a_sentence_merely_containing_a_pronoun_is_not_flagged` FAILS. Revert to `match`, clear caches again, confirm 4 passed. If it does not fail, the test is not discriminating and must be strengthened before you continue.

- [ ] **Step 6: Commit**

```bash
git add src/visual_verify/agent/reader.py tests/test_reader.py
git commit -F - <<'EOF'
feat(agent): flag a claim that opens by referring backwards

Verification runs after drafting, so any claim can be removed before the answer
is shown. A claim opening with "It" or "Each of them" is meaningless once its
predecessor is withheld, and it does not announce that it is broken: it stays
grammatical and quietly says something other than what was verified.

Flagged rather than rejected, following is_compound. Dropping it would lose
verified content with a real region behind it, and the useful response is to
count the rate in the eval and fix the prompt.

Anchored to the sentence opening. A search anywhere in the sentence flags
almost every English sentence, which would look like a working check while
carrying no signal, and a test pins that difference. The word boundary matters
for the same reason: "Itemised costs appear in Table 2" opens with the letters
of a pronoun and dangles nothing.
EOF
```

---

## Task 4: `shares_content_word`

**Files:**
- Modify: `src/visual_verify/agent/reader.py`
- Modify: `tests/test_reader.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reader.py`:

```python
def test_a_chained_pair_shares_a_content_word():
    from visual_verify.agent.reader import shares_content_word

    assert shares_content_word(
        "The evaluation compares three system variants on SlideVQA.",
        "Each of the three variants is scored on answer accuracy.",
    ) is True


def test_two_unrelated_claims_share_nothing():
    from visual_verify.agent.reader import shares_content_word

    assert shares_content_word(
        "The evaluation compares three system variants on SlideVQA.",
        "Ground truth is derived automatically.",
    ) is False


def test_a_stopword_overlap_does_not_count_as_chaining():
    """THE test of this function. Every pair of English sentences shares "the"
    or "is". Without a stopword list the check returns True for everything,
    which is worse than not having it: it would report perfect chaining on a
    reader that had reverted to listing disconnected facts.
    """
    from visual_verify.agent.reader import shares_content_word

    assert shares_content_word("The page is here.", "The chart is there.") is False


def test_a_plural_matches_its_singular():
    """Claims chain through a noun phrase that often changes number across the
    join: "three variants" then "each variant"."""
    from visual_verify.agent.reader import shares_content_word

    assert shares_content_word(
        "The evaluation compares three variants.",
        "Each variant is scored separately.",
    ) is True


def test_punctuation_does_not_block_a_match():
    from visual_verify.agent.reader import shares_content_word

    assert shares_content_word("Scores come from SlideVQA.", "SlideVQA has 2000 slides.") is True
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest tests/test_reader.py -k shares_content -v
```
Expected: FAIL with `ImportError: cannot import name 'shares_content_word'`.

- [ ] **Step 3: Implement it**

Add to `src/visual_verify/agent/reader.py`:

```python
# Closed class only. A long stopword list would start removing the words that
# actually carry the chain; these are the ones that appear in essentially every
# sentence and therefore signal nothing about whether two claims are connected.
_STOPWORDS = frozenset(
    """a an and are as at be been by for from has have in is it its of on or
    that the their there these this those to was were which with""".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def _content_words(text: str) -> set[str]:
    """Lowercased words with stopwords dropped and a trailing s stripped."""
    words = _WORD.findall(text.lower())
    return {w.rstrip("s") or w for w in words if w not in _STOPWORDS}


def shares_content_word(previous: str, claim: str) -> bool:
    """Whether `claim` picks up anything from `previous`.

    A proxy for the chaining rule the reader prompt asks for: connect each
    sentence to the one before it by repeating a noun phrase from its end. That
    is what gives the answer human flow while surviving the removal of any
    sentence, since repeating the noun works where a pronoun would dangle.

    Deliberately crude, and honest about it. A reader can satisfy this while
    writing badly, so it is a floor rather than a measure of quality. What it
    does catch is the failure this project actually saw: a reader reverting to
    a list of disconnected facts, where adjacent claims share no content word
    at all.

    The stopword list is what makes it mean anything. Every pair of English
    sentences shares "the" or "is", so without it the answer is always True.
    """
    return bool(_content_words(previous) & _content_words(claim))
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_reader.py -k shares_content -v
```
Expected: 5 passed.

- [ ] **Step 5: Verify the stopword test discriminates**

Temporarily set `_STOPWORDS = frozenset()` and re-run.

```bash
find . -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
uv run pytest tests/test_reader.py -k shares_content -v
```
Expected: `test_a_stopword_overlap_does_not_count_as_chaining` and `test_two_unrelated_claims_share_nothing` both FAIL. Restore the list, clear caches, confirm 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/visual_verify/agent/reader.py tests/test_reader.py
git commit -F - <<'EOF'
feat(agent): measure whether adjacent claims are actually chained

The reader is asked to connect each sentence to the one before it by repeating
a noun phrase from its end, which is what gives the answer human flow while
surviving the removal of any sentence: repeating the noun works where a pronoun
dangles. Without a check, "semantic flow" is a matter of taste and nothing
would notice the reader reverting to a list of disconnected facts.

Crude on purpose and honest about it. A reader can satisfy this while writing
badly, so it is a floor rather than a measure of quality.

The stopword list is the whole check. Every pair of English sentences shares
"the" or "is", so without it the function returns True for everything, which is
worse than not having it: it would report perfect chaining on exactly the
output it exists to catch. A test pins that, and emptying the list fails it.
EOF
```

---

## Task 5: The reader prompt

**Files:**
- Modify: `src/visual_verify/agent/reader.py:17-26` (the `PROMPT` constant)
- Modify: `tests/test_agent_live.py`

- [ ] **Step 1: Replace the prompt**

```python
PROMPT = """You are answering a question from one page of a document, for
someone who cannot see the page.

Answer only from what is visible on this page. If the page does not answer the
question, return an empty list of claims.

Compose the answer as connected prose, then return it as one sentence per
claim, in the order they should be read. Every sentence is checked separately
against the page, and any sentence may be removed before the answer is shown,
so the sentences must obey four rules.

1. The FIRST sentence answers the question directly. Not background, not what
   the page is about. Someone who read only that sentence should have the
   answer.

2. Each sentence asserts exactly ONE thing. Each sentence is matched to a
   single region of the page as its evidence, and a sentence asserting two
   things cannot be evidenced by one region.

3. Each sentence stands on its own. Never begin a sentence with it, they, them,
   this, these, those, such, or Additionally, and never refer to the previous
   sentence or to anything above. Repeat the noun instead.

4. Connect each sentence to the one before it by repeating a noun phrase from
   the end of that sentence, never by a pronoun. "The evaluation compares three
   variants. Each of the three variants is scored on ..." reads as one answer.
   "The evaluation compares three variants. Each of them is scored on ..."
   becomes nonsense the moment the first sentence is removed.

Write the way you would answer a colleague who asked you out loud. Do not
describe the page. State what it says.

Set starts_paragraph on a sentence that opens a new topic, and leave it false
otherwise. Most answers are a single paragraph.

Question: {question}"""
```

- [ ] **Step 2: Add a live test of the four rules**

Append to `tests/test_agent_live.py`:

```python
def test_the_drafted_answer_holds_together():
    """The four drafting rules, against a real model on a real page.

    A fake cannot test this: the rules are instructions to a model and the only
    thing that can fail them is a model. Asserted as a floor rather than a
    quality bar, because "reads well" is not something a test can decide.
    """
    from visual_verify.agent.models import make_chat
    from visual_verify.agent.reader import opens_with_anaphora, shares_content_word

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
    assert len(unchained) <= len(claims) // 3, f"claims sharing nothing with their predecessor: {unchained}"
```

Add the import at the top of that file, next to the existing ones:

```python
from visual_verify.agent.reader import read
```

- [ ] **Step 3: Run the live test**

```bash
set -a && . ./.env && set +a
uv run pytest tests/test_agent_live.py::test_the_drafted_answer_holds_together -v
```
Expected: PASS. If `dangling` is non-empty, rule 3 is not landing; strengthen it in the prompt by listing the forbidden openings again at the end, and re-run. If `unchained` is over the threshold, rule 4 is not landing.

- [ ] **Step 4: Read one real answer and check it against the old output**

```bash
SHA=$(uv run vvrag search "evaluation methodology" -k 1 | grep -oE '[0-9a-f]{64}' | head -1)
uv run vvrag ask "What is the evaluation methodology?" --doc "$SHA" --page 14
```
Compare against the recorded before-state in the spec, section 2: six claims, three of them beginning "Figure 5.1 includes". The first claim should now state what the methodology is, and no claim should describe the figure rather than the content.

- [ ] **Step 5: Run the fast suite**

```bash
uv run pytest -q --ignore=tests/test_embedder.py \
  --ignore=tests/test_known_item_retrieval.py \
  --ignore=tests/test_grounding_live.py
```
Expected: no change in count from Task 2. `tests/test_reader.py::test_read_sends_the_question_and_the_page_image` asserts only that the question appears in the prompt, which is still true.

- [ ] **Step 6: Commit**

```bash
git add src/visual_verify/agent/reader.py tests/test_agent_live.py
git commit -F - <<'EOF'
feat(agent): have the reader answer the question, not describe the page

The prompt said to answer only from what is visible on this page and never
said to answer the question, and it got what it asked for. Measured on
proposal.pdf page 14 for "What is the evaluation methodology?": six claims, all
true and all grounded, three of them beginning "Figure 5.1 includes", and not
one of them stating what the methodology is. Joined, that is a fact dump.

Four rules replace it. The first sentence answers directly. Each sentence
asserts one thing, which is a grounding requirement rather than a style one.
Each stands alone. Each connects to the one before it by repeating a noun
phrase from its end.

The fourth rule is the load-bearing one. Verification runs after drafting, so
the gate punches holes in finished prose, and anaphora is the normal way prose
coheres. A paragraph missing its first sentence is grammatical and meaningless.
Repeating the noun gives the same given-new movement and survives the removal
of any sentence, at the cost of mild repetitiveness.

A live test holds the floor, because only a model can fail instructions to a
model.
EOF
```

---

## Task 6: The lead rule

**Files:**
- Modify: `src/visual_verify/contracts.py` (`Answer.abstained_overall`)
- Modify: `tests/test_contracts.py`

The first half of this property was already fixed on 2026-08-10: it is a `computed_field` returning `not self.shown`. This adds the lead condition.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_contracts.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest tests/test_contracts.py -k lead -v
```
Expected: `test_a_withheld_lead_abstains_even_when_detail_survived` FAILS with `assert False is True`, because `shown` is non-empty so the current rule returns False. The second test passes already.

- [ ] **Step 3: Add the lead condition**

In `src/visual_verify/contracts.py`, change the return of `abstained_overall` and extend its docstring:

```python
        Two conditions, not one. Nothing survived, OR the LEAD claim did not.
        The lead is `claims[0]`, the sentence the reader drafted to answer the
        question directly; everything after it is supporting detail. Showing
        surviving detail after the answer itself was withheld would present a
        page of context as though it answered a question it never touched.

        `claims` holds every claim in drafted order including withheld ones, so
        the lead is still identifiable after the gate has removed it.
        """
        return not self.shown or self.claims[0].withheld
```

Note that `self.claims[0]` is safe: `not self.shown` short-circuits to True whenever `claims` is empty, so the index is only ever reached when there is at least one claim.

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_contracts.py -v
```
Expected: all pass, including the three abstention tests already there.

- [ ] **Step 5: Check the CLI still reads correctly**

`src/visual_verify/cli.py:506` prints "abstained: no claim on this page met the support threshold" when `result.abstained_overall`. That sentence is now wrong in one case: the lead was withheld but other claims passed. Change it to:

```python
    if result.abstained_overall:
        print("\nabstained: the claim answering the question was not verified")
```

- [ ] **Step 6: Run the affected suites**

```bash
uv run pytest tests/test_contracts.py tests/test_cli_retrieval.py tests/test_agent.py \
  tests/test_answer_stream.py tests/test_api_wire.py -q
```
Expected: all pass once `tests/test_cli_retrieval.py` lines 247 and 248 are updated to the new sentence:
```python
    assert "abstained: the claim answering the question was not verified" in out
    assert out.index("Withheld") < out.index("abstained: the claim")
```
That test builds an Answer whose single claim is withheld, so it is `claims[0]` and the lead rule agrees with the old one there.

- [ ] **Step 7: Commit**

```bash
git add src/visual_verify/contracts.py src/visual_verify/cli.py tests/
git commit -F - <<'EOF'
feat(contracts): abstain when the claim that answers was withheld

The first claim is the one the reader drafted to answer the question; the rest
support it. Showing surviving detail after the answer itself was withheld
presents a page of context as though it answered a question it never touched,
which is the fact dump this whole change exists to remove.

The lead is claims[0], and claims holds every claim in drafted order including
withheld ones, so the lead is still identifiable after the gate has removed it.
The index is safe because `not self.shown` short-circuits to True whenever
there are no claims at all.

The CLI's abstention line said no claim met the threshold, which is now wrong
in exactly the new case, so it says what actually happened instead.
EOF
```

---

## Task 7: Carry the paragraph flag to the browser

**Files:**
- Modify: `src/visual_verify/api/wire.py:35-49` (`_claim`)
- Modify: `frontend/lib/api.ts` (`ClaimEvent`)
- Modify: `tests/test_api_wire.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api_wire.py`:

```python
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
    every claim it lays out, and a withheld claim that lies about it would
    shift the paragraph break onto the wrong sentence."""
    from visual_verify.agent.events import ClaimVerified

    c = withheld_claim()
    c = c.model_copy(update={"starts_paragraph": True})

    _, payload = to_frame(ClaimVerified(index=1, claim=c))

    assert payload["regions"] == []
    assert payload["starts_paragraph"] is True
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest tests/test_api_wire.py -k paragraph -v
```
Expected: FAIL with `KeyError: 'starts_paragraph'`.

- [ ] **Step 3: Emit the field**

In `src/visual_verify/api/wire.py`, add one entry to the dict returned by `_claim`, after `"compound": c.compound,`:

```python
        # Where the answer breaks into a new paragraph. Metadata, not text: it
        # adds nothing unverified to the screen and every claim still maps to
        # exactly one region.
        "starts_paragraph": c.starts_paragraph,
```

- [ ] **Step 4: Add it to the frontend type**

In `frontend/lib/api.ts`, add to `ClaimEvent`:

```ts
  // Where the answer breaks into a new paragraph. Set by the reader, which is
  // the only thing that knows where the topic turns.
  starts_paragraph: boolean;
```

- [ ] **Step 5: Run both sides**

```bash
uv run pytest tests/test_api_wire.py tests/test_api_ask.py tests/test_api.py -q
cd frontend && npx tsc --noEmit
```
Expected: Python all pass. TypeScript clean, because the field is additive and nothing reads it yet.

- [ ] **Step 6: Commit**

```bash
git add src/visual_verify/api/wire.py frontend/lib/api.ts tests/test_api_wire.py
git commit -F - <<'EOF'
feat(api): send the paragraph break with each claim

The UI cannot infer a topic turn from claim text, and the reader is the only
thing that knows where one is, so dropping the flag at this boundary silently
collapses every answer back into a single block.

Sent for withheld claims too. A withheld claim is stripped of its regions, not
of everything, and the UI lays out every claim it receives: a withheld claim
that lied about its break would shift the paragraph onto the wrong sentence.
EOF
```

---

## Task 8: Render paragraphs

**Files:**
- Modify: `frontend/lib/claims.ts`
- Create: `frontend/lib/claims.test.ts`
- Modify: `frontend/components/AnswerPanel.tsx`

The grouping logic goes in `lib/claims.ts` because it is pure and testable there. The frontend has `vitest` but no component testing library, so a component test is not available and the panel itself is verified in the browser.

- [ ] **Step 1: Write the failing test**

Create `frontend/lib/claims.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { groupIntoParagraphs } from "./claims";
import type { ClaimEvent } from "./api";

function claim(index: number, starts_paragraph = false): ClaimEvent {
  return {
    index,
    text: `claim ${index}`,
    label: "supported",
    confidence: 0.9,
    reason: null,
    compound: false,
    withheld: false,
    starts_paragraph,
    regions: [],
  };
}

describe("groupIntoParagraphs", () => {
  it("keeps an unbroken answer as one paragraph", () => {
    const groups = groupIntoParagraphs([claim(0), claim(1), claim(2)]);
    expect(groups.length).toBe(1);
    expect(groups[0].map((c) => c.index)).toEqual([0, 1, 2]);
  });

  it("breaks where a claim says it starts a paragraph", () => {
    const groups = groupIntoParagraphs([claim(0), claim(1), claim(2, true), claim(3)]);
    expect(groups.map((g) => g.map((c) => c.index))).toEqual([
      [0, 1],
      [2, 3],
    ]);
  });

  // THE test of this function. Verification removes claims, so the claim
  // carrying the break is frequently the one withheld. Starting a paragraph on
  // the FIRST survivor regardless keeps the answer from opening with an empty
  // block, and dropping the flag entirely would silently merge two topics.
  it("never produces an empty paragraph when the breaking claim is gone", () => {
    const groups = groupIntoParagraphs([claim(0, true), claim(1)]);
    expect(groups.length).toBe(1);
    expect(groups.every((g) => g.length > 0)).toBe(true);
  });

  it("returns nothing for an empty answer", () => {
    expect(groupIntoParagraphs([])).toEqual([]);
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd frontend && npx vitest run lib/claims.test.ts
```
Expected: FAIL with a resolution error for `groupIntoParagraphs`.

- [ ] **Step 3: Implement it**

Append to `frontend/lib/claims.ts`:

```ts
import type { ClaimEvent } from "./api";

/**
 * Split the shown claims into paragraphs at the breaks the reader marked.
 *
 * The break lives on the claim rather than being inferred from the text,
 * because only the reader knows where the topic turns. Verification removes
 * claims after drafting, so the claim carrying a break is frequently the one
 * withheld; the first survivor always opens a paragraph regardless of its own
 * flag, which is what stops the answer from opening with an empty block.
 */
export function groupIntoParagraphs(claims: ClaimEvent[]): ClaimEvent[][] {
  const groups: ClaimEvent[][] = [];
  for (const claim of claims) {
    if (groups.length === 0 || claim.starts_paragraph) groups.push([claim]);
    else groups[groups.length - 1].push(claim);
  }
  return groups;
}
```

- [ ] **Step 4: Run the test**

```bash
npx vitest run lib/claims.test.ts
```
Expected: 4 passed.

- [ ] **Step 5: Render the groups**

In `frontend/components/AnswerPanel.tsx`, import the helper:

```tsx
import { colourFor, groupIntoParagraphs } from "@/lib/claims";
```

Replace the single `<p>` that maps over `shown` with a map over paragraphs. The inner `<span>` and its handlers are unchanged; only the wrapper differs:

```tsx
      {groupIntoParagraphs(shown).map((paragraph, i) => (
        <p
          key={paragraph[0].index}
          className={`text-[17px] leading-[1.65] tracking-[-0.006em] sm:text-lg ${
            i === 0 ? "mt-3" : "mt-4"
          }`}
        >
          {paragraph.map((c) => {
            const colour = colourFor(c.index);
            const dimmed = hovered !== null && hovered !== c.index;
            return (
              <span
                key={c.index}
                onMouseEnter={() => onHover(c.index)}
                onMouseLeave={() => onHover(null)}
                onClick={() => onSelect(c.index)}
                style={{ textDecorationColor: colour, opacity: dimmed ? 0.38 : 1 }}
                className="cursor-pointer underline decoration-2 underline-offset-[5px] transition-opacity duration-150"
              >
                {c.text}
                <sup className="ml-0.5 text-[10px] font-semibold tnum" style={{ color: colour }}>
                  {c.index + 1}
                </sup>{" "}
              </span>
            );
          })}
        </p>
      ))}
```

- [ ] **Step 6: Check it builds and renders**

```bash
npx tsc --noEmit && npx eslint app lib components && npm run build
```
Expected: all clean.

Then start both services and ask a question whose answer spans two topics, and confirm the break renders where the reader marked it:
```bash
# terminal 1, from the repo root
set -a && . ./.env && set +a
uv run uvicorn --factory visual_verify.api.app:create_app --port 8000
# terminal 2
cd frontend && npm run dev
```

- [ ] **Step 7: Commit**

```bash
git add frontend/lib/claims.ts frontend/lib/claims.test.ts frontend/components/AnswerPanel.tsx
git commit -F - <<'EOF'
feat(ui): break the answer into paragraphs where the reader marked them

An answer long enough to cover two topics reads badly as one block, and the
text alone does not say where the topic turns. The break comes from the reader,
which is the only thing that knows.

The grouping lives in lib rather than in the component because it is pure and
the frontend has no component testing library, so this is the only form in
which it can be tested at all.

The first survivor always opens a paragraph regardless of its own flag.
Verification removes claims after drafting, so the claim carrying a break is
frequently the one withheld, and honouring the flag blindly would open the
answer with an empty block.
EOF
```

---

## Task 9: The three answer states

**Files:**
- Modify: `frontend/components/AnswerPanel.tsx`
- Modify: `frontend/app/page.tsx`

Fixed copy may assert facts about the system's own process. It may never assert facts about the document. "I was not able to confirm an answer from this page" is introspection and is true by construction whenever it is shown. "This page does not mention the metrics" is a claim about the world, it has no region, no model judged it, and the system never says it.

- [ ] **Step 1: Add the partial-state line to `AnswerPanel`**

A new prop rather than reading `done.withheld`, which the panel already
receives. `done` arrives only when the stream finishes, so a count taken from
it would stay at zero while claims are still being verified and then jump at
the end. The live count comes from the claims already in hand.

The header keeps its `N verified, M withheld` chip. That is the technical
reading of the same fact; this line is the conversational one, and the spec
asks for both.

Add a prop to the `Props` type:

```tsx
  withheldCount: number;
```

and destructure it in the signature. Replace the existing footer paragraph with:

```tsx
      {withheldCount > 0 && (
        <p className="mt-4 text-sm italic text-muted">
          I left out {withheldCount} statement{withheldCount === 1 ? "" : "s"} I could not
          confirm against this page.
        </p>
      )}

      <p className="mt-4 border-t border-border pt-3 text-xs leading-relaxed text-faint">
        Each sentence above was checked by a second model and carries its own
        evidence region. Hover one to find it on the page, or click it to open the
        evidence.
      </p>
```

- [ ] **Step 2: Pass the count from `page.tsx`**

At the `<AnswerPanel ... />` call site, add:

```tsx
                  withheldCount={withheld.length}
```

- [ ] **Step 3: Replace the abstained block in `page.tsx`**

Replace the existing `{abstained && (...)}` block with:

```tsx
              {abstained && (
                <div className="mb-4 rounded-2xl border border-border bg-surface p-5">
                  <h2 className="text-base font-semibold">
                    I could not answer this from this page
                  </h2>
                  <p className="mt-2 text-sm leading-relaxed text-muted">
                    {/* Only ever a statement about this system's own process.
                        "I could not confirm" is true by construction whenever
                        this is shown. A sentence about what the page does or
                        does not contain would be a claim with no region behind
                        it, which is precisely what the rest of the system
                        exists to refuse. */}
                    I read {retrieved?.doc_name} page {retrieved?.page}, but I was not able
                    to confirm an answer to your question from what is on it. Rather than
                    give you something I cannot stand behind, I have left it out.
                    {retrieved?.warning &&
                      " Note the warning above: this page has no embeddings, so that is" +
                        " very likely a missing index rather than weak evidence."}
                  </p>
                  {shown.length > 0 && (
                    <p className="mt-3 text-sm italic text-muted">
                      Here is what I was able to confirm from the page, in case it helps.
                    </p>
                  )}
                </div>
              )}
```

- [ ] **Step 4: Show the surviving claims as context, not as the answer**

In `page.tsx`, the `AnswerPanel` currently renders whenever `shown.length > 0`. Under the lead rule that can now be true while the answer abstained, and the panel is headed "Answer". Change its condition so it renders only when the system is actually answering:

```tsx
              {shown.length > 0 && !abstained && (
                <AnswerPanel
                  shown={shown}
                  done={done}
                  withheldCount={withheld.length}
                  hovered={hovered}
                  onHover={setHovered}
                  onSelect={revealClaim}
                />
              )}
```

The surviving claims still appear below, in the Evidence Vault, which is already rendered unconditionally and already labels them as evidence rather than as an answer. That is the reframing the spec asks for, and it needs no new component.

- [ ] **Step 5: Check it builds**

```bash
cd frontend && npx tsc --noEmit && npx eslint app lib components && npm run build
```
Expected: all clean.

- [ ] **Step 6: Verify all three states in the browser**

Start both services as in Task 8 step 6, then:

1. **Answered.** Ask a question the top page answers well. Expect the paragraph, no italic line.
2. **Partial.** Ask a question that produces a withheld claim, or temporarily raise the threshold in the request body. Expect the paragraph plus "I left out N statements I could not confirm against this page."
3. **Abstained.** Force it by asking a question unrelated to the corpus, or by posting `{"question": "...", "threshold": 7.0}` to `/ask`, which admits only a supported claim at full confidence. Expect the decline, and the surviving claims present in the Evidence Vault rather than under an "Answer" heading.

Confirm zero console errors in each state.

- [ ] **Step 7: Commit**

```bash
git add frontend/components/AnswerPanel.tsx frontend/app/page.tsx
git commit -F - <<'EOF'
feat(ui): say plainly when the answer could not be confirmed

The refusal case rendered as "No claims to show", which reads as broken
software rather than as a decision the system made on purpose. It now says what
happened, in the first person, and it stops calling the panel an Answer when
the claim that answered the question was withheld.

Every sentence of that copy is about this system's own process and none of it
is about the document. "I was not able to confirm an answer" is introspection
and is true by construction whenever it is shown. "This page does not mention
the metrics" would be a claim about the world with no region behind it, which
is exactly what the rest of the system exists to refuse, so the copy never says
anything of the kind.

Surviving claims are not thrown away. They stay in the Evidence Vault, which
already presents them as evidence rather than as an answer, so the reframing
needs no new component.
EOF
```

---

## Task 10: Full verification

- [ ] **Step 1: Run the complete Python suite once**

The three GPU modules load ColQwen2 onto a 4 GB card and take around ten minutes. Never let this run in the background; the tool's 600 second cap will move it there and the output becomes unreadable. Redirect it and read the file.

```bash
uv run pytest -q > /tmp/suite.log 2>&1
tail -20 /tmp/suite.log
```
Expected: everything passes. Baseline before this plan was 463 passed, 5 skipped on the fast subset and 465 passed, 9 skipped on the full one.

- [ ] **Step 2: Run the frontend checks**

```bash
cd frontend && npx vitest run && npx tsc --noEmit && npx eslint app lib components && npm run build
```
Expected: all clean.

- [ ] **Step 3: Lint the Python side**

```bash
uv run ruff check src tests && uv run ruff format --check src tests
```
Expected: passes. `tests/test_core_is_light.py` and `tests/test_evidence.py` fail `ruff format --check` and did so before this work; leave them.

- [ ] **Step 4: Confirm the GPU is released**

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```
Expected: around 4 MiB. A leftover process means an orphaned test run; kill it before drawing conclusions from any later measurement.

- [ ] **Step 5: Update the spec's status line**

In `docs/superpowers/specs/2026-08-10-answer-composition-design.md`, change the header:

```
Status: implemented 2026-08-10
```

and record the outcome of the Task 1 probes in section 10, replacing the paragraph that says the strictness probe runs before any tuning with what actually happened: whether the verifier already rejected the four probes, or which prompt change was needed.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-08-10-answer-composition-design.md
git commit -F - <<'EOF'
docs: record the answer composition outcome

The spec asked whether the verifier could reject anything at all before any of
the display work was built on top of it. The result of that probe is the one
finding worth keeping from this slice, and it belongs next to the design that
depended on it.
EOF
```

---

## Self-review notes

Checked against the spec, section by section.

- Section 3, the structural problem: no task, and none needed. It is the reasoning behind Tasks 3, 4 and 5.
- Section 5.1, the reader prompt: Task 5.
- Section 6, schema: Task 2.
- Section 7, the two checks: Tasks 3 and 4. The dangling-reference count the spec assigns to S7 is not implemented here, because S7's eval harness does not exist yet. `opens_with_anaphora` is the piece S7 will call, and Task 3 delivers it.
- Section 8, the lead rule and the three states: Task 6 for the rule, Task 9 for the states.
- Section 9, fixed copy: Task 9.
- Section 10, the verifier: Task 1.
- Section 11, testing: distributed across the tasks that introduce each behaviour.
- Section 12, files: matches the table at the top of this plan.
- Section 14, assumption 1, that `starts_paragraph` earns its schema change: still an assumption after this plan. The string-coercion validator in Task 2 exists so that removing it later is cheap if real answers turn out never to need a break.

Type consistency: `DraftedClaim` in Task 2 is the type `read()` returns in Task 2 step 5, the type Task 5's live test iterates with `.text`, and the source of `Claim.starts_paragraph` used in Tasks 7, 8 and 9. `groupIntoParagraphs` is named identically in Task 8 steps 1, 3 and 5.
