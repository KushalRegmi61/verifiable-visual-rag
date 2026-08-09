# S6 Product UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser page where a typed question returns a verified answer with its supporting region drawn on the page image, or an abstain badge.

**Architecture:** A FastAPI service holds the models resident and streams Server-Sent Events; a Next.js frontend draws normalized boxes over the page PNG. The service contains no grounding, no rubric arithmetic, and no threshold comparison: `agent/core.py` grows `answer_stream()` (the same claim loop, yielding) and `answer()` becomes a drain over it, so the abstention gate stays in exactly one place.

**Tech Stack:** FastAPI, uvicorn, anyio (all via the new `api` extra), Next.js 15 with the App Router, TypeScript, vitest.

**Spec:** `docs/superpowers/specs/2026-08-09-s6-product-ui-design.md`

---

## Before you start

Run this once and keep the output. Every task's "expected: PASS" is relative to it.

```bash
cd /home/pursottam/mine/projects/verifiable-visual-rag
uv run pytest -q 2>&1 | tail -3
```

Baseline as of this plan: **380 passed, 2 skipped**. The 2 skips are live Gemini
tests with no quota.

**Never run the full suite in a foreground tool call.** It takes about 16
minutes and loads ColQwen2 three times. Two subagents have already hit the
600 s tool cap, had the run backgrounded, and stopped, leaving one 2.6 GB
orphan on a 3.63 GB card. Run only the test files your task names. The full
suite runs once, at the end, redirected to a log:

```bash
uv run pytest -q > /tmp/claude-1000/-home-pursottam-mine-projects-verifiable-visual-rag/ab1aea5c-001c-4a82-8a57-90bda6f945db/scratchpad/s6-suite.log 2>&1
```

**Do not `git add CLAUDE.md`.** It is gitignored and must stay that way.
**No Claude or AI attribution in any commit message.** This is a graded project.
**No em-dashes** in any prose you write.

---

## File Structure

**Created:**

| path | responsibility |
| --- | --- |
| `src/visual_verify/agent/events.py` | the four event types `answer_stream` yields. No logic. |
| `src/visual_verify/prepare.py` | `prepare_page()`: doc and page in; boxes, vectors, grid, image path out. The adapter the CLI and the API share. |
| `src/visual_verify/api/__init__.py` | package marker |
| `src/visual_verify/api/sse.py` | SSE wire framing. One pure function. |
| `src/visual_verify/api/wire.py` | event to `(name, payload)`. Owns the withheld-region strip. |
| `src/visual_verify/api/stream.py` | `iter_in_thread()`: run a blocking generator without freezing the event loop. |
| `src/visual_verify/api/resources.py` | lifespan-held models and the startup guards |
| `src/visual_verify/api/ask.py` | `ask_events()`: retrieval, prepare, then `answer_stream`. No HTTP. |
| `src/visual_verify/api/app.py` | the FastAPI app and its four routes |
| `frontend/lib/overlay.ts` | normalized bbox to CSS percentages |
| `frontend/lib/api.ts` | the SSE client |
| `frontend/app/page.tsx` | the one page |
| `tests/test_answer_stream.py`, `tests/test_prepare.py`, `tests/test_api_sse.py`, `tests/test_api_wire.py`, `tests/test_api_stream.py`, `tests/test_api.py` | one test module per unit |

**Modified:** `src/visual_verify/contracts.py`, `src/visual_verify/agent/core.py`,
`src/visual_verify/agent/__init__.py`, `src/visual_verify/cli.py`,
`pyproject.toml`, `tests/test_core_is_light.py`, `docs/ROADMAP.md`, `README.md`.

---

## Task 1: `Claim.reason`

The verifier returns a one-sentence reason and `answer()` drops it
(`agent/core.py:95`). The withheld panel is built entirely around showing it.

**Files:**
- Modify: `src/visual_verify/contracts.py:47-66`
- Modify: `src/visual_verify/agent/core.py:97-106`
- Test: `tests/test_contracts.py`, `tests/test_agent.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_contracts.py`:

```python
def test_claim_carries_the_verifier_reason():
    from visual_verify.contracts import Claim

    c = Claim(text="Margins held steady", confidence=0.5, reason="the chart shows margin falling")

    assert c.reason == "the chart shows margin falling"


def test_claim_reason_defaults_to_none():
    """Additive optional field: every existing construction site still works."""
    from visual_verify.contracts import Claim

    assert Claim(text="x", confidence=0.5).reason is None
```

Append to `tests/test_agent.py`:

```python
def test_answer_carries_the_verifier_reason_onto_the_claim():
    """The reason is the only thing that makes a wrong verdict debuggable, and
    the UI's withheld panel is built around it. Dropping it here made it
    unreachable outside the verifier."""
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent"])])
    verifier = FakeChat(
        "v", [Verdict(label="unsupported", confidence=0.8, reason="the chart shows a fall")]
    )

    out = answer(
        "What happened?",
        Path("p.png"),
        page_boxes(),
        page=0,
        reader_chat=reader,
        verifier_chat=verifier,
    )

    assert out.claims[0].reason == "the chart shows a fall"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_contracts.py -k reason tests/test_agent.py -k reason -v
```

Expected: FAIL. Pydantic raises on the unexpected `reason` keyword.

- [ ] **Step 3: Add the field**

In `src/visual_verify/contracts.py`, inside `class Claim`, after `compound`:

```python
    # The verifier's one-sentence justification. None until the verifier has
    # run, like `label`. The product UI shows this for a claim it withholds:
    # a count alone tells a user nothing, and the region cannot be shown, so
    # the reason is the only thing left that explains the refusal. S7 puts it
    # in the eval output for the same purpose.
    reason: str | None = None
```

- [ ] **Step 4: Carry it through `answer()`**

In `src/visual_verify/agent/core.py`, in the `Claim(...)` construction, add
`reason=verdict.reason,` after `label=verdict.label,`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest tests/test_contracts.py tests/test_agent.py -q
```

Expected: PASS, no failures.

- [ ] **Step 6: Commit**

```bash
git add src/visual_verify/contracts.py src/visual_verify/agent/core.py \
        tests/test_contracts.py tests/test_agent.py
git commit -m "feat(contracts): carry the verifier's reason onto the claim

verify() returns a one-sentence justification and answer() dropped it, so it
existed only inside the verifier call. The UI withholds a rejected claim's
region, which leaves the reason as the only thing that can explain the
refusal to a user, and S7 needs it to make a wrong verdict explicable in the
eval output."
```

---

## Task 2: `answer_stream()`

**Files:**
- Create: `src/visual_verify/agent/events.py`
- Modify: `src/visual_verify/agent/core.py`, `src/visual_verify/agent/__init__.py`
- Test: `tests/test_answer_stream.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_answer_stream.py`:

```python
"""answer_stream() is the loop; answer() is a drain over it.

The one property that matters is that they cannot diverge. A second copy of
the read-ground-verify loop would put the GroundingError recovery and the
`score < threshold` comparison in two places, and the copy the product hits
is the one no S5 test covers.
"""

from pathlib import Path

from visual_verify.agent import answer, answer_stream
from visual_verify.agent.events import AnswerComplete, ClaimsProduced, ClaimVerified, ReadingStarted
from visual_verify.agent.schemas import ClaimList, Verdict
from visual_verify.agent.types import FakeChat
from visual_verify.ingest.boxes import BoxRecord


def page_boxes():
    """Two lines, both findable in the text layer.

    Defined here rather than imported from tests/test_agent.py: `tests/` has no
    __init__.py, so `from tests.test_agent import ...` depends on rootdir being
    on sys.path and breaks under a plain `pytest tests/test_answer_stream.py`.

    Every claim scripted below must be findable verbatim, or ground() falls
    through to the visual path, which needs page vectors and a grid and would
    raise GroundingError instead of exercising the loop.
    """
    first = ["Revenue", "grew", "42", "percent"]
    second = ["Margins", "held", "steady"]
    boxes = [
        BoxRecord(
            kind="word",
            x0=0.1 + i * 0.15,
            y0=0.10,
            x1=0.22 + i * 0.15,
            y1=0.16,
            text=t,
            block_no=0,
            line_no=0,
            word_no=i,
        )
        for i, t in enumerate(first)
    ]
    boxes += [
        BoxRecord(
            kind="word",
            x0=0.1 + i * 0.15,
            y0=0.30,
            x1=0.22 + i * 0.15,
            y1=0.36,
            text=t,
            block_no=0,
            line_no=1,
            word_no=i,
        )
        for i, t in enumerate(second)
    ]
    return boxes


def script():
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent", "Margins held steady"])])
    verifier = FakeChat(
        "v",
        [
            Verdict(label="supported", confidence=0.9, reason="matches"),
            Verdict(label="unsupported", confidence=0.8, reason="contradicted"),
        ],
    )
    return reader, verifier


def run_stream():
    reader, verifier = script()
    return list(
        answer_stream(
            "What happened?",
            Path("p.png"),
            page_boxes(),
            page=0,
            reader_chat=reader,
            verifier_chat=verifier,
        )
    )


def test_the_event_order_is_reading_then_count_then_claims_then_complete():
    events = run_stream()

    assert isinstance(events[0], ReadingStarted)
    assert isinstance(events[1], ClaimsProduced)
    assert events[1].n == 2
    assert isinstance(events[2], ClaimVerified)
    assert isinstance(events[3], ClaimVerified)
    assert isinstance(events[4], AnswerComplete)
    assert len(events) == 5


def test_claim_events_are_indexed_in_order():
    events = [e for e in run_stream() if isinstance(e, ClaimVerified)]

    assert [e.index for e in events] == [0, 1]


def test_every_streamed_claim_already_has_a_verdict():
    """The reason S5 refused to stream the reader: a claim must never reach a
    consumer before the verifier has judged it. If this can fail, the product
    can display something it exists to withhold."""
    for event in run_stream():
        if isinstance(event, ClaimVerified):
            assert event.claim.label is not None


def test_answer_returns_exactly_what_the_stream_finished_with():
    """Pins the drain. If answer() ever grows its own loop, this fails."""
    streamed = [e for e in run_stream() if isinstance(e, AnswerComplete)][0].answer

    reader, verifier = script()
    direct = answer(
        "What happened?",
        Path("p.png"),
        page_boxes(),
        page=0,
        reader_chat=reader,
        verifier_chat=verifier,
    )

    assert direct == streamed


def test_the_same_model_guard_still_raises_from_the_stream():
    """AgentError must surface when the generator is first advanced, not be
    swallowed because nobody iterated it."""
    import pytest

    from visual_verify.agent import AgentError

    same = FakeChat("same", [ClaimList(claims=["x"])])
    other = FakeChat("same", [Verdict(label="supported", confidence=0.5, reason="r")])

    with pytest.raises(AgentError):
        list(
            answer_stream(
                "q",
                Path("p.png"),
                page_boxes(),
                page=0,
                reader_chat=same,
                verifier_chat=other,
            )
        )
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_answer_stream.py -q
```

Expected: FAIL at import, `cannot import name 'answer_stream'`.

- [ ] **Step 3: Create the event types**

Create `src/visual_verify/agent/events.py`:

```python
"""What answer_stream() yields.

Deliberately plain frozen dataclasses with no HTTP, no JSON, and no knowledge
of a transport. The API layer converts them; the eval layer ignores them.

Retrieval is NOT represented here. answer_stream() is handed a page and never
chooses one, so an event about which page won belongs to the caller that ran
the search.
"""

from dataclasses import dataclass

from visual_verify.contracts import Answer, Claim


@dataclass(frozen=True)
class ReadingStarted:
    """The reader has been called. Nothing is known yet."""


@dataclass(frozen=True)
class ClaimsProduced:
    """The reader returned. `n` is how many verdicts are still coming."""

    n: int


@dataclass(frozen=True)
class ClaimVerified:
    """One claim, grounded and judged. `claim.label` is never None here."""

    index: int
    claim: Claim


@dataclass(frozen=True)
class AnswerComplete:
    """The last event, always. Carries the same Answer answer() returns."""

    answer: Answer


AnswerEvent = ReadingStarted | ClaimsProduced | ClaimVerified | AnswerComplete
```

- [ ] **Step 4: Rewrite `core.py` as a generator plus a drain**

In `src/visual_verify/agent/core.py`, add `from collections.abc import Callable, Iterator`
(replacing the existing `Callable` import) and
`from visual_verify.agent.events import AnswerComplete, AnswerEvent, ClaimsProduced, ClaimVerified, ReadingStarted`.

Replace the body of `answer()` from the `if reader_chat.model_id ==` guard to
the final `return` with a new `answer_stream()` carrying that body, then make
`answer()` drain it. The claim-building block is unchanged apart from the two
`yield` lines:

```python
def answer_stream(
    question: str,
    image_path: Path,
    boxes: list[BoxRecord],
    *,
    page: int,
    reader_chat: StructuredChat,
    verifier_chat: StructuredChat,
    threshold: float = DEFAULT_THRESHOLD,
    page_vectors: np.ndarray | None = None,
    embed_query: Callable[[str], np.ndarray] | None = None,
    grid: PatchGrid | None = None,
) -> Iterator[AnswerEvent]:
    """The loop, yielding as each claim is judged. See answer() for the arguments.

    This exists so a consumer can show a verified claim while later claims are
    still being verified. It does NOT weaken the guarantee: a ClaimVerified
    event is yielded only after verify() has returned for that claim, so
    nothing leaves this function unjudged. Streaming the READER's output is a
    different thing and is still ruled out (S5 spec section 9).

    answer() is a drain over this. Keeping one loop is the point: the
    GroundingError recovery and the `score < threshold` comparison must not
    exist twice, because the copy a product hits is the one no test covers.
    """
    if reader_chat.model_id == verifier_chat.model_id:
        raise AgentError(
            f"reader and verifier are the same model ({reader_chat.model_id}); "
            "a model grading its own output is biased toward it, which is the "
            "reason this slice uses two providers"
        )

    yield ReadingStarted()
    texts = read(reader_chat, image_path, question)
    yield ClaimsProduced(n=len(texts))

    claims: list[Claim] = []
    for index, text in enumerate(texts):
        query_vectors = embed_query(text) if embed_query is not None else None
        try:
            regions = ground(
                text,
                boxes,
                page=page,
                page_vectors=page_vectors,
                query_vectors=query_vectors,
                grid=grid,
            )
        except GroundingError:
            # A reader model paraphrases by default, so a claim that is not
            # findable verbatim and has no visual path to fall back on (no
            # vectors supplied) is the EXPECTED case, not an edge one. Losing
            # every already-verified claim, after paying for the API calls
            # that produced them, is worse than one claim arriving with no
            # evidence: the verifier still runs, empty regions in hand, and
            # the rubric's insufficient_evidence label makes the gap visible
            # instead of raising a traceback that discards the whole answer.
            #
            # This is NOT swallowing a configuration error. If grounding is
            # unusable for every claim (no vectors ever supplied and nothing
            # in the text layer matches), the visible result is an answer
            # where every claim is insufficiently evidenced, not a crash.
            regions = []
        verdict = verify(verifier_chat, image_path, text, regions)
        score = abstention_score(verdict.label, verdict.confidence)
        claim = Claim(
            text=text,
            regions=regions,
            confidence=verdict.confidence,
            label=verdict.label,
            reason=verdict.reason,
            abstained=score < threshold,
            compound=is_compound(text),
        )
        claims.append(claim)
        yield ClaimVerified(index=index, claim=claim)

    yield AnswerComplete(
        answer=Answer(
            question=question,
            claims=claims,
            abstained_overall=not claims or all(c.abstained for c in claims),
        )
    )
```

Then replace `answer()`'s body (keep its existing docstring, and append the
sentence below to it) with the drain:

```python
    for event in answer_stream(
        question,
        image_path,
        boxes,
        page=page,
        reader_chat=reader_chat,
        verifier_chat=verifier_chat,
        threshold=threshold,
        page_vectors=page_vectors,
        embed_query=embed_query,
        grid=grid,
    ):
        if isinstance(event, AnswerComplete):
            return event.answer
    raise AgentError("answer_stream ended without an AnswerComplete event")
```

Docstring sentence to append to `answer()`:

```
    A thin drain over answer_stream(). The loop lives there so a streaming
    consumer and this one cannot diverge; this signature and return type are
    unchanged, which is what keeps `vvrag ask` and S7's harness working.
```

`read()` currently returns a list, so `len(texts)` is valid. If it is ever
changed to a generator, `ClaimsProduced` breaks loudly rather than silently,
which is the correct failure.

- [ ] **Step 5: Export it**

In `src/visual_verify/agent/__init__.py`:

```python
from visual_verify.agent.core import AgentError, answer, answer_stream

__all__ = ["AgentError", "answer", "answer_stream"]
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
uv run pytest tests/test_answer_stream.py tests/test_agent.py tests/test_core_is_light.py -q
```

Expected: PASS. `test_core_is_light` must still pass: `events.py` imports only
`dataclasses` and `contracts`.

- [ ] **Step 7: Commit**

```bash
git add src/visual_verify/agent/events.py src/visual_verify/agent/core.py \
        src/visual_verify/agent/__init__.py tests/test_answer_stream.py
git commit -m "feat(agent): yield each claim as it is verified

The UI streams verified claims so a three-claim answer stops being tens of
seconds of dead air, which needs the loop to yield rather than return. answer()
becomes a drain over the same generator so the GroundingError recovery and the
score-against-threshold comparison exist once. A second loop in the service
would have left every S5 test passing while the path users actually hit
diverged, which is how most of this repo's real bugs have presented."
```

---

## Task 3: `prepare_page()`

`cmd_ask` holds about 60 lines the service needs verbatim, plus retrieval on
the front. Extract, then rewire the CLI through the extraction so both callers
exercise one code path.

**Files:**
- Create: `src/visual_verify/prepare.py`
- Modify: `src/visual_verify/cli.py:520-605`
- Test: `tests/test_prepare.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prepare.py`:

```python
"""prepare_page() is the adapter the CLI and the API share."""

import pytest
from sqlalchemy.orm import Session

from visual_verify.cli import main
from visual_verify.config import Settings
from visual_verify.prepare import PageNotFound, prepare_page
from visual_verify.store.engine import make_engine


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VVRAG_QDRANT_URL", ":memory:")
    monkeypatch.delenv("VVRAG_QDRANT_API_KEY", raising=False)
    monkeypatch.setenv("VVRAG_FAKE_EMBEDDER", "1")
    return tmp_path


@pytest.fixture
def indexed(env, born_digital_pdf):
    assert main(["ingest", str(born_digital_pdf)]) == 0
    assert main(["embed", "--all"]) == 0
    return Settings.from_env()


def test_it_returns_boxes_vectors_and_a_grid_that_agree(indexed):
    from visual_verify.cli import _make_index

    settings = indexed
    index = _make_index(settings)
    with Session(make_engine(settings.db_url)) as session:
        page = prepare_page(session, index, settings, doc="born_digital", page_no=0)

    assert page.boxes, "the text layer should have produced word boxes"
    assert page.image_path.exists()
    assert page.page_vectors is not None
    # The grid must describe the vectors it was fetched with. A mismatch here
    # is the failure mode that made grounding place boxes off-page in S3.
    assert page.grid.n_vectors == page.page_vectors.shape[0]
    assert page.doc_name == "born_digital.pdf"


def test_an_unknown_document_raises_rather_than_returning_none(indexed):
    from visual_verify.cli import _make_index

    settings = indexed
    index = _make_index(settings)
    with Session(make_engine(settings.db_url)) as session:
        with pytest.raises(PageNotFound):
            prepare_page(session, index, settings, doc="no-such-doc", page_no=0)


def test_a_page_beyond_the_document_raises(indexed):
    from visual_verify.cli import _make_index

    settings = indexed
    index = _make_index(settings)
    with Session(make_engine(settings.db_url)) as session:
        with pytest.raises(PageNotFound):
            prepare_page(session, index, settings, doc="born_digital", page_no=99)


def test_an_ambiguous_document_needle_raises_instead_of_guessing(indexed, tmp_path):
    """cmd_ask's _resolve_document returns a candidate LIST when a needle
    matches more than one document. Silently taking the first is how
    `inspect proposal` used to pick whichever of proposal.pdf and
    reference_proposal.pdf was inserted first."""
    from visual_verify.cli import _make_index

    import fitz

    second = tmp_path / "born_digital_copy.pdf"
    doc = fitz.open()
    doc.new_page(width=612.0, height=792.0).insert_text((72.0, 100.0), "Other text", fontsize=12)
    doc.save(second)
    doc.close()
    assert main(["ingest", str(second)]) == 0

    settings = indexed
    index = _make_index(settings)
    with Session(make_engine(settings.db_url)) as session:
        with pytest.raises(PageNotFound):
            prepare_page(session, index, settings, doc="born_digital", page_no=0)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_prepare.py -q
```

Expected: FAIL at import, `No module named 'visual_verify.prepare'`.

- [ ] **Step 3: Write the module**

Create `src/visual_verify/prepare.py`:

```python
"""Assemble everything one page's worth of grounding needs.

Lifted out of cli.cmd_ask so the API layer does not reimplement it. This is
the adapter: it talks to SQLAlchemy and Qdrant so that grounding and the agent
never have to, and everything it hands back is a plain array or a value object.
That is what keeps those packages inside the core's four dependencies.

Requires the `store` and `retrieval` extras, like the CLI. Nothing in the core
imports this module.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from visual_verify.config import Settings
from visual_verify.ingest.boxes import BoxRecord
from visual_verify.retrieval.geometry import PatchGrid
from visual_verify.store.models import Box, Document, Page


class PageNotFound(LookupError):
    """No single page matched. Also raised for an ambiguous document needle:
    picking the first of several matches is a wrong answer wearing the costume
    of a right one."""


@dataclass(frozen=True)
class PreparedPage:
    """One page, ready to ground against."""

    doc_sha: str
    doc_name: str
    page_no: int
    image_path: Path
    boxes: list[BoxRecord]
    # None when the page has not been embedded. ground() then has no visual
    # fallback and raises GroundingError for any claim it cannot find in the
    # text layer, which answer_stream turns into insufficient_evidence.
    page_vectors: np.ndarray | None
    grid: PatchGrid | None


def _to_record(b: Box) -> BoxRecord:
    return BoxRecord(
        kind=b.kind,
        x0=b.x0,
        y0=b.y0,
        x1=b.x1,
        y1=b.y1,
        text=b.text,
        block_no=b.block_no,
        line_no=b.line_no,
        word_no=b.word_no,
    )


def resolve_document(session: Session, needle: str) -> Document:
    """Exact sha256 first, then a unique path or sha prefix match."""
    exact = session.get(Document, needle)
    if exact is not None:
        return exact

    matches = list(
        session.scalars(
            select(Document)
            .where(Document.path.contains(needle) | Document.sha256.startswith(needle))
            .order_by(Document.path)
        )
    )
    if not matches:
        raise PageNotFound(f"no document matching {needle!r}")
    if len(matches) > 1:
        names = ", ".join(Path(m.path).name for m in matches)
        raise PageNotFound(f"{needle!r} matches more than one document: {names}")
    return matches[0]


def prepare_page(
    session: Session,
    index,
    settings: Settings,
    *,
    doc: str,
    page_no: int,
) -> PreparedPage:
    """Everything needed to ground a claim against one page.

    `index` is a QdrantIndex, untyped here so this module does not import
    qdrant_client at module scope for the sake of an annotation.

    Vectors are fetched unconditionally. The caller cannot know whether the
    text path will suffice, because the claims come from a reader that has not
    run yet, so a page that grounds entirely through the text layer simply
    never uses them. One Qdrant round trip against two model calls per claim.
    """
    document = resolve_document(session, doc)
    page = session.scalar(
        select(Page).where(Page.doc_sha == document.sha256, Page.page_no == page_no)
    )
    if page is None:
        raise PageNotFound(f"no page {page_no} in {Path(document.path).name}")

    boxes = [
        _to_record(b)
        for b in session.scalars(select(Box).where(Box.page_id == page.id, Box.kind == "word"))
    ]

    from visual_verify.retrieval.index import ORIGINAL

    vectors: np.ndarray | None = None
    grid: PatchGrid | None = None
    if page_no in index.existing_page_nos(document.sha256):
        payload = index.get_payload(document.sha256, page_no)
        vectors = index.get_vectors(document.sha256, page_no)[ORIGINAL]
        grid = PatchGrid(
            n_x=payload["n_patches_x"],
            n_y=payload["n_patches_y"],
            offset=payload["patch_offset"],
            n_vectors=vectors.shape[0],
        )

    return PreparedPage(
        doc_sha=document.sha256,
        doc_name=Path(document.path).name,
        page_no=page_no,
        image_path=settings.pages_dir / page.image_path,
        boxes=boxes,
        page_vectors=vectors,
        grid=grid,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_prepare.py -q
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Rewire `cmd_ask` through it**

In `src/visual_verify/cli.py`, replace the body of `cmd_ask` between the
threshold check and the `try: reader = ...` block with:

```python
    settings = Settings.from_env()
    index = _make_index(settings)
    if index.count() == 0:
        print("no pages indexed; run `vvrag embed` first")
        return 1

    with _session(settings) as session:
        try:
            prepared = prepare_page(session, index, settings, doc=args.doc, page_no=args.page)
        except PageNotFound as exc:
            print(str(exc))
            return 1

    embedder = _make_embedder(settings)
```

and replace the `answer(...)` call's page arguments with the prepared ones:

```python
        result = answer(
            args.question,
            prepared.image_path,
            prepared.boxes,
            page=prepared.page_no,
            reader_chat=reader,
            verifier_chat=verifier,
            threshold=args.threshold,
            page_vectors=prepared.page_vectors,
            embed_query=embedder.embed_query,
            grid=prepared.grid,
        )
```

Add to the imports inside `cmd_ask`:

```python
    from visual_verify.prepare import PageNotFound, prepare_page
```

- [ ] **Step 6: Run the CLI tests to verify nothing regressed**

```bash
uv run pytest tests/test_cli.py tests/test_cli_retrieval.py tests/test_prepare.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/visual_verify/prepare.py src/visual_verify/cli.py tests/test_prepare.py
git commit -m "refactor(prepare): share the page adapter between the CLI and the service

cmd_ask assembled boxes, vectors, and the patch grid inline, and the API needs
the identical assembly plus retrieval in front of it. Copying it would put the
grid construction in two places, and a grid that disagrees with its vectors is
the S3 failure that placed boxes off-page while every shape looked right.

resolve_document now raises on an ambiguous needle instead of returning a
candidate list, so a caller cannot take the first match by accident."
```

---

## Task 4: The `api` extra

**Files:**
- Modify: `pyproject.toml`, `tests/test_core_is_light.py`

- [ ] **Step 1: Extend the guard first**

`fastapi` is already in `FORBIDDEN`. Add its siblings. In
`tests/test_core_is_light.py`:

```python
FORBIDDEN = [
    "sqlalchemy", "alembic", "qdrant_client", "fastapi", "starlette", "uvicorn",
    "torch", "transformers",
    "langchain", "langchain_openai", "langchain_google_genai",
]
```

- [ ] **Step 2: Run it to confirm it still passes**

```bash
uv run pytest tests/test_core_is_light.py -q
```

Expected: PASS, 3 tests. The core does not import these, and this arms the
guard before the code that could break it exists.

- [ ] **Step 3: Add the extra**

In `pyproject.toml`, after the `agent` extra:

```toml
# The product UI's service half. Like `agent` and unlike `retrieval`, these are
# not pinned: a web framework bump changes a signature loudly rather than
# degrading output quality in silence the way the colpali-engine and
# transformers combination did.
api = [
    "fastapi>=0.115",
    "uvicorn>=0.30",
]
```

`anyio` arrives transitively with starlette and is not declared separately.

- [ ] **Step 4: Sync and verify the import works**

```bash
uv sync --all-extras --group dev
uv run python -c "import fastapi, uvicorn; print(fastapi.__version__)"
```

Expected: a version number, no traceback.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/test_core_is_light.py
git commit -m "build(api): add the api extra and arm the core guard first

starlette and uvicorn join FORBIDDEN before any code can import them, so the
four-dependency core boundary is enforced from the moment the service exists
rather than after someone notices."
```

---

## Task 5: SSE framing

**Files:**
- Create: `src/visual_verify/api/__init__.py`, `src/visual_verify/api/sse.py`
- Test: `tests/test_api_sse.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_sse.py`:

```python
"""SSE framing. Ten lines of string building, and two ways to get it wrong."""

from visual_verify.api.sse import frame


def test_a_frame_is_an_event_line_a_data_line_and_a_blank_line():
    assert frame("claim", {"index": 0}) == 'event: claim\ndata: {"index": 0}\n\n'


def test_a_newline_inside_the_payload_does_not_truncate_the_event():
    """A raw newline in a data field ends the field, so a reason string
    containing one would silently cut the event in half and the browser would
    parse the remainder as garbage. json.dumps escapes it; this pins that the
    dump is not bypassed."""
    text = frame("claim", {"reason": "line one\nline two"})

    assert text.count("\n\n") == 1
    assert text.endswith("\n\n")
    assert len(text.splitlines()) == 2


def test_unicode_survives_unescaped():
    """ensure_ascii=False keeps a page's own characters readable in the
    stream. \\u00e9 would still decode, but nobody can eyeball it."""
    assert "é" in frame("claim", {"text": "café"})


def test_the_event_name_is_used_verbatim():
    assert frame("done", {}).startswith("event: done\n")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_api_sse.py -q
```

Expected: FAIL, `No module named 'visual_verify.api'`.

- [ ] **Step 3: Write the module**

Create `src/visual_verify/api/__init__.py`:

```python
"""The HTTP surface. Converts events to frames and serves images.

Contains no grounding, no rubric arithmetic, and no threshold comparison. If
any of those appear here, the guarantee exists in two places and the copy the
product hits is the one no test covers.
"""
```

Create `src/visual_verify/api/sse.py`:

```python
"""Server-Sent Events framing.

Hand-rolled rather than pulling in sse-starlette. The format is three lines,
this is directly unit-testable, and the repository gates everything else behind
extras for a reason.
"""

import json


def frame(event: str, data: dict) -> str:
    """One SSE frame: an event name, a JSON payload, and a blank separator.

    json.dumps is load-bearing beyond serialization. A raw newline inside a
    `data:` field terminates that field, so a verifier reason spanning two
    lines would truncate the event and the browser would parse the tail as a
    separate malformed frame. Escaping it is what stops that.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
```

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run pytest tests/test_api_sse.py -q
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/api/__init__.py src/visual_verify/api/sse.py tests/test_api_sse.py
git commit -m "feat(api): add SSE framing

A verifier reason containing a newline would end the data field early and
truncate the event, so the json.dumps escape is the thing under test, not the
string concatenation around it."
```

---

## Task 6: Event to wire payload, and the region strip

This is the task that carries the guarantee. Read the test before the code.

**Files:**
- Create: `src/visual_verify/api/wire.py`
- Test: `tests/test_api_wire.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_wire.py`:

```python
"""What leaves the process.

The withheld-region strip lives here rather than in answer(), because S7 needs
the regions of rejected claims to compute confident-wrong against coverage.
Putting the guarantee in the core would break the eval instead of protecting
the user, so it belongs at the boundary the data crosses.
"""

from visual_verify.api.wire import to_frame
from visual_verify.agent.events import AnswerComplete, ClaimsProduced, ClaimVerified, ReadingStarted
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
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_api_wire.py -q
```

Expected: FAIL, `No module named 'visual_verify.api.wire'`.

- [ ] **Step 3: Write the module**

Create `src/visual_verify/api/wire.py`:

```python
"""Events to JSON-serializable wire payloads.

This module owns one guarantee: a claim the verifier rejected leaves this
process with no geometry attached. Not styled differently, not flagged for the
frontend to handle. Absent.

It is enforced here and not in answer() on purpose. S7 computes confident-wrong
against coverage, which needs the regions of rejected claims, so stripping them
in the core would break the evaluation rather than protect anybody. The right
place for the rule is where the data crosses out of the process.
"""

from visual_verify.agent.events import (
    AnswerComplete,
    AnswerEvent,
    ClaimsProduced,
    ClaimVerified,
    ReadingStarted,
)
from visual_verify.contracts import Claim, GroundedRegion


def _region(r: GroundedRegion) -> dict:
    return {
        "bbox": list(r.bbox),
        "score": r.score,
        "modality": r.modality,
        # Kept deliberately. "block" means the heatmap could not separate the
        # lines inside the winning block, and the overlay draws that dashed.
        # Without it, a coarse fallback and a confident line hit look the same.
        "resolution": r.resolution,
        "text": r.text,
    }


def _claim(index: int, c: Claim) -> dict:
    withheld = c.abstained or c.label is None
    return {
        "index": index,
        "text": c.text,
        "label": c.label,
        "confidence": c.confidence,
        "reason": c.reason,
        "compound": c.compound,
        "withheld": withheld,
        "regions": [] if withheld else [_region(r) for r in c.regions],
    }


def to_frame(event: AnswerEvent) -> tuple[str, dict]:
    """(event name, payload) for one event."""
    if isinstance(event, ReadingStarted):
        return "reading", {}
    if isinstance(event, ClaimsProduced):
        return "claims", {"n": event.n}
    if isinstance(event, ClaimVerified):
        return "claim", _claim(event.index, event.claim)
    if isinstance(event, AnswerComplete):
        answer = event.answer
        shown = len(answer.shown)
        return "done", {
            "shown": shown,
            "withheld": len(answer.claims) - shown,
            "abstained_overall": answer.abstained_overall,
        }
    raise TypeError(f"no wire mapping for {type(event).__name__}")
```

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run pytest tests/test_api_wire.py -q
```

Expected: PASS, 7 tests.

- [ ] **Step 5: Verify the guarantee test can actually fail**

Temporarily change `_claim`'s regions line to
`"regions": [_region(r) for r in c.regions],` and re-run.

```bash
uv run pytest tests/test_api_wire.py::test_a_withheld_claim_carries_no_regions -q
```

Expected: FAIL. **Then revert the change and re-run to confirm PASS.**

This step is not optional. An assertion that cannot fail is decoration, and
this repository has already shipped one (`inside = line_boxes(boxes)` passed
all 21 snap tests). Clear caches before believing a revert:

```bash
find . -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
```

- [ ] **Step 6: Commit**

```bash
git add src/visual_verify/api/wire.py tests/test_api_wire.py
git commit -m "feat(api): strip a withheld claim's regions at the process boundary

A rejected claim leaves with its label and reason and no geometry. Styling it
differently in the frontend would make the guarantee depend on the frontend
honouring it; absence does not.

Enforced here rather than in answer() because S7 needs those regions to compute
confident-wrong against coverage, so the core stripping them would break the
eval instead of protecting a user."
```

---

## Task 7: Running a blocking generator off the event loop

**Files:**
- Create: `src/visual_verify/api/stream.py`
- Test: `tests/test_api_stream.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_stream.py`:

```python
"""answer_stream() blocks on a GPU and on two hosted providers.

Iterating it inside an async endpoint would freeze the event loop for the whole
answer, so /health would hang and the browser would receive nothing until the
end, which defeats the reason for streaming at all.
"""

import asyncio

import pytest

from visual_verify.api.stream import iter_in_thread


async def collect(make_iter):
    return [item async for item in iter_in_thread(make_iter)]


def test_items_arrive_in_order():
    assert asyncio.run(collect(lambda: iter([1, 2, 3]))) == [1, 2, 3]


def test_an_exception_from_the_generator_propagates_to_the_consumer():
    """A provider failure mid-answer must reach the endpoint so it can emit an
    error frame. Swallowing it would end the stream indistinguishably from a
    successful short answer."""

    def boom():
        yield 1
        raise RuntimeError("provider exploded")

    with pytest.raises(RuntimeError, match="provider exploded"):
        asyncio.run(collect(boom))


def test_items_before_the_exception_are_still_delivered():
    """Claims already verified and paid for must not be discarded because a
    later one failed."""

    def boom():
        yield 1
        yield 2
        raise RuntimeError("later")

    seen = []

    async def run():
        async for item in iter_in_thread(boom):
            seen.append(item)

    with pytest.raises(RuntimeError):
        asyncio.run(run())

    assert seen == [1, 2]


def test_the_event_loop_keeps_running_while_the_generator_blocks():
    """The whole point. If the generator ran inline, the ticker below could
    not advance while it slept."""
    import time

    ticks = 0

    def slow():
        time.sleep(0.3)
        yield "done"

    async def run():
        nonlocal ticks

        async def tick():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        ticker = asyncio.create_task(tick())
        out = [item async for item in iter_in_thread(slow)]
        ticker.cancel()
        return out

    assert asyncio.run(run()) == ["done"]
    assert ticks > 5, f"the loop was blocked; only {ticks} ticks in 0.3 s"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_api_stream.py -q
```

Expected: FAIL, `No module named 'visual_verify.api.stream'`.

- [ ] **Step 3: Write the module**

Create `src/visual_verify/api/stream.py`:

```python
"""Bridge a blocking generator into an async consumer."""

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TypeVar

T = TypeVar("T")

_DONE = object()


async def iter_in_thread(make_iter: Callable[[], Iterator[T]]) -> AsyncIterator[T]:
    """Run `make_iter()` on a worker thread, yielding its items as they arrive.

    The queue is unbounded on purpose. A bounded one would give backpressure,
    but if the consumer stops early (a client disconnects) the producer would
    block forever on a full queue and the awaited task would never finish. An
    answer produces a handful of events, so there is nothing to bound.

    The producer is awaited before returning even when the consumer stops
    early. A hosted model call cannot be cancelled cheaply, so the thread runs
    to completion regardless; awaiting it here is what lets the caller hold a
    GPU semaphore until the work is genuinely over rather than until the
    browser lost interest.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def pump() -> None:
        try:
            for item in make_iter():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer side
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    task = loop.run_in_executor(None, pump)
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        await task
```

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run pytest tests/test_api_stream.py -q
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/api/stream.py tests/test_api_stream.py
git commit -m "feat(api): run the blocking answer generator off the event loop

answer_stream holds the GPU and blocks on two hosted providers. Iterating it
inside the async endpoint would freeze the loop for the full answer, so nothing
would reach the browser until the end and /health would hang, which removes the
only reason to stream. The producer is still awaited on early consumer exit,
because a hosted call cannot be cancelled and the GPU semaphore must not be
released while the thread is still using it."
```

---

## Task 8: Lifespan resources and the startup guards

**Files:**
- Create: `src/visual_verify/api/resources.py`
- Test: `tests/test_api_resources.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_resources.py`:

```python
"""A misconfigured service must refuse to start.

Coming up and failing on the first question looks healthy to /health and to
anyone watching it boot, and the moment that gets discovered is during a demo.
"""

import pytest

from visual_verify.api.resources import StartupRefused, check_configuration
from visual_verify.config import Settings


def test_identical_reader_and_verifier_is_refused():
    """The self-preference argument is the reason S5 is shaped as it is. A
    misconfiguration pointing both at one model would be invisible in output."""
    settings = Settings(
        qdrant_url=":memory:",
        reader_provider="openai",
        reader_model="gpt-4o",
        verifier_provider="openai",
        verifier_model="gpt-4o",
    )

    with pytest.raises(StartupRefused, match="same model"):
        check_configuration(settings)


def test_different_models_pass():
    settings = Settings(
        qdrant_url=":memory:",
        reader_provider="openai",
        reader_model="gpt-4o",
        verifier_provider="google",
        verifier_model="gemini-2.0-flash",
    )

    check_configuration(settings)


def test_a_missing_qdrant_url_is_refused():
    settings = Settings(
        qdrant_url=None,
        reader_provider="openai",
        reader_model="gpt-4o",
        verifier_provider="google",
        verifier_model="gemini-2.0-flash",
    )

    with pytest.raises(StartupRefused, match="VVRAG_QDRANT_URL"):
        check_configuration(settings)


def test_the_error_names_the_environment_variable_to_fix():
    """A refusal that does not say what to set is a refusal nobody can act on."""
    settings = Settings(
        qdrant_url=":memory:",
        reader_provider="openai",
        reader_model="gpt-4o",
        verifier_provider="openai",
        verifier_model="gpt-4o",
    )

    with pytest.raises(StartupRefused) as exc:
        check_configuration(settings)

    assert "VVRAG_VERIFIER_MODEL" in str(exc.value)
```

Note the fixture-free `Settings(...)` construction: `Settings` is a frozen
dataclass with defaults, so keyword construction needs no environment at all.

`qdrant_url` defaults to `None` and `check_configuration` tests it FIRST, so
every case above that expects a different refusal must set it. Omitting it makes
the same-model tests pass for the wrong reason: they would raise
`StartupRefused` about Qdrant and never reach the check they name.

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_api_resources.py -q
```

Expected: FAIL, `No module named 'visual_verify.api.resources'`.

- [ ] **Step 3: Write the module**

Create `src/visual_verify/api/resources.py`:

```python
"""What the service holds for its whole lifetime, and what it refuses to start without.

Loading ColQwen2 costs about 20 seconds and 2.6 GB. Every `vvrag search`
invocation pays that; a request-scoped embedder would make the UI unusable and
would not fit alongside itself. So the models load once, here, and the first
question is fast at the cost of a slow boot.
"""

from dataclasses import dataclass

from visual_verify.config import Settings


class StartupRefused(RuntimeError):
    """The service will not come up in this configuration."""


def check_configuration(settings: Settings) -> None:
    """Fail now, loudly, rather than on the first question.

    answer() carries the same reader-verifier check, but it only fires once a
    question has been asked. By then the service has reported itself healthy,
    the browser is open, and somebody is watching.
    """
    if not settings.qdrant_url:
        raise StartupRefused("VVRAG_QDRANT_URL is not set; the service cannot retrieve anything")

    reader = f"{settings.reader_provider}:{settings.reader_model}"
    verifier = f"{settings.verifier_provider}:{settings.verifier_model}"
    if reader == verifier:
        raise StartupRefused(
            f"reader and verifier are the same model ({reader}); a model grading "
            "its own output is biased toward it, which is the reason this project "
            "uses two providers. Set VVRAG_VERIFIER_PROVIDER and "
            "VVRAG_VERIFIER_MODEL to something else."
        )


@dataclass
class Resources:
    """Held on app.state for the process lifetime."""

    settings: Settings
    engine: object
    index: object
    embedder: object
    reader_chat: object
    verifier_chat: object


def build(settings: Settings) -> Resources:
    """Construct everything once. Raises StartupRefused on misconfiguration.

    Imports are function-local so that importing this module (which the tests
    do, to reach check_configuration) does not drag in torch or LangChain.
    """
    check_configuration(settings)

    from visual_verify.agent.cache import CachedChat
    from visual_verify.agent.models import MissingApiKey, UnknownProvider, make_chat
    from visual_verify.cli import _ensure_schema, _make_embedder, _make_index
    from visual_verify.store.engine import make_engine

    _ensure_schema(settings)
    engine = make_engine(settings.db_url)
    index = _make_index(settings)
    embedder = _make_embedder(settings)

    try:
        reader_chat = CachedChat(make_chat("reader", settings), settings.agent_cache_dir)
        verifier_chat = CachedChat(make_chat("verifier", settings), settings.agent_cache_dir)
    except (MissingApiKey, UnknownProvider) as exc:
        raise StartupRefused(str(exc)) from exc

    return Resources(
        settings=settings,
        engine=engine,
        index=index,
        embedder=embedder,
        reader_chat=reader_chat,
        verifier_chat=verifier_chat,
    )
```

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run pytest tests/test_api_resources.py tests/test_core_is_light.py -q
```

Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/api/resources.py tests/test_api_resources.py
git commit -m "feat(api): hold the models for the process lifetime and refuse a bad config

ColQwen2 costs about 20 s and 2.6 GB per load, so a request-scoped embedder
would make the UI unusable and would not fit alongside itself.

Misconfiguration fails at startup rather than on the first question. answer()
already rejects a reader and verifier that are the same model, but only once
somebody has asked something, by which point the service has reported itself
healthy and a demo is underway."
```

---

## Task 9: `ask_events()`

Retrieval, then prepare, then the answer stream. No HTTP, so it is testable
directly.

**Files:**
- Create: `src/visual_verify/api/ask.py`
- Test: `tests/test_api_ask.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_ask.py`:

```python
"""The service's own generator: search, prepare, then answer_stream."""

import pytest
from sqlalchemy.orm import Session

from visual_verify.agent.events import AnswerComplete, ClaimVerified
from visual_verify.agent.schemas import ClaimList, Verdict
from visual_verify.agent.types import FakeChat
from visual_verify.api.ask import AskRequest, NoPagesIndexed, Retrieved, ask_events
from visual_verify.cli import _make_index, main
from visual_verify.config import Settings
from visual_verify.store.engine import make_engine


@pytest.fixture
def indexed(tmp_path, monkeypatch, born_digital_pdf):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VVRAG_QDRANT_URL", ":memory:")
    monkeypatch.delenv("VVRAG_QDRANT_API_KEY", raising=False)
    monkeypatch.setenv("VVRAG_FAKE_EMBEDDER", "1")
    assert main(["ingest", str(born_digital_pdf)]) == 0
    assert main(["embed", "--all"]) == 0
    return Settings.from_env()


def chats():
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent"])])
    verifier = FakeChat("v", [Verdict(label="supported", confidence=0.9, reason="matches")])
    return reader, verifier


def run(settings, request):
    from visual_verify.retrieval.types import FakeEmbedder

    reader, verifier = chats()
    index = _make_index(settings)
    with Session(make_engine(settings.db_url)) as session:
        return list(
            ask_events(
                request,
                session=session,
                index=index,
                embedder=FakeEmbedder(),
                reader_chat=reader,
                verifier_chat=verifier,
                settings=settings,
            )
        )


def test_retrieval_runs_first_and_names_the_page_it_chose(indexed):
    events = run(indexed, AskRequest(question="What happened?"))

    assert isinstance(events[0], Retrieved)
    assert events[0].page.page_no == 0
    assert events[0].page.doc_name == "born_digital.pdf"


def test_the_answer_stream_follows_retrieval(indexed):
    events = run(indexed, AskRequest(question="What happened?"))

    assert any(isinstance(e, ClaimVerified) for e in events)
    assert isinstance(events[-1], AnswerComplete)


def test_an_explicit_page_skips_retrieval_and_reports_no_candidates(indexed):
    """Clicking a candidate re-asks with doc and page pinned. The event shape
    stays identical so the frontend has one code path."""
    events = run(indexed, AskRequest(question="q", doc="born_digital", page=0))

    assert isinstance(events[0], Retrieved)
    assert events[0].candidates == []
    assert events[0].score is None


def test_an_unindexed_corpus_raises_rather_than_answering_from_nothing(
    tmp_path, monkeypatch, born_digital_pdf
):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VVRAG_QDRANT_URL", ":memory:")
    monkeypatch.setenv("VVRAG_FAKE_EMBEDDER", "1")
    assert main(["ingest", str(born_digital_pdf)]) == 0
    settings = Settings.from_env()

    with pytest.raises(NoPagesIndexed):
        run(settings, AskRequest(question="q"))
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_api_ask.py -q
```

Expected: FAIL, `No module named 'visual_verify.api.ask'`.

- [ ] **Step 3: Write the module**

Create `src/visual_verify/api/ask.py`:

```python
"""One question, start to finish: search, prepare, read, ground, verify.

No HTTP here on purpose. The whole sequence is a plain generator over plain
objects, so it is testable without a client, a server, or a socket.

This is the first place the online pipeline exists end to end. The CLI splits
it: `vvrag search` ranks pages and stops, `vvrag ask` requires a page number.
proposal.tex lines 340 to 342 specify retrieve, read, ground, verify, and this
is where that becomes one call.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from visual_verify.agent.core import answer_stream
from visual_verify.agent.events import AnswerEvent
from visual_verify.agent.types import StructuredChat
from visual_verify.config import Settings
from visual_verify.contracts import RetrievedPage
from visual_verify.prepare import PreparedPage, prepare_page

DEFAULT_K = 5


class NoPagesIndexed(RuntimeError):
    """The corpus has documents but no embeddings."""


class AskRequest(BaseModel):
    """The POST body.

    `threshold` defaults to None and is resolved against Settings at use time,
    so the service honours VVRAG_ABSTAIN_THRESHOLD exactly as the CLI does and
    no literal is repeated here. S7 sweeps this to produce the confident-wrong
    against coverage curve, which is why it is a request field at all.
    """

    question: str = Field(min_length=1)
    doc: str | None = None
    page: int | None = Field(default=None, ge=0)
    k: int = Field(default=DEFAULT_K, ge=1, le=20)
    threshold: float | None = None

    @field_validator("threshold")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        import math

        if v is not None and not math.isfinite(v):
            raise ValueError("threshold must be a finite number")
        return v

    @field_validator("page")
    @classmethod
    def _page_needs_a_doc(cls, v: int | None, info) -> int | None:
        if v is not None and not info.data.get("doc"):
            raise ValueError("page requires doc; supply both or neither")
        return v


@dataclass(frozen=True)
class Retrieved:
    """Which page will be read, and what else was considered.

    `score` is None and `candidates` empty when the caller pinned the page, so
    the frontend gets one event shape either way.
    """

    page: PreparedPage
    score: float | None
    candidates: list[RetrievedPage] = field(default_factory=list)


AskEvent = Retrieved | AnswerEvent


def ask_events(
    request: AskRequest,
    *,
    session: Session,
    index,
    embedder,
    reader_chat: StructuredChat,
    verifier_chat: StructuredChat,
    settings: Settings,
) -> Iterator[AskEvent]:
    """Yield a Retrieved event, then everything answer_stream yields."""
    if index.count() == 0:
        raise NoPagesIndexed("no pages indexed; run `vvrag embed --all` first")

    if request.doc is not None and request.page is not None:
        prepared = prepare_page(
            session, index, settings, doc=request.doc, page_no=request.page
        )
        yield Retrieved(page=prepared, score=None, candidates=[])
    else:
        hits = index.search(
            embedder.embed_query(request.question), embedder.provenance, limit=request.k
        )
        if not hits:
            raise NoPagesIndexed("retrieval returned no pages")
        top = hits[0]
        prepared = prepare_page(
            session, index, settings, doc=top.doc_id, page_no=top.page
        )
        yield Retrieved(page=prepared, score=top.score, candidates=list(hits[1:]))

    threshold = (
        request.threshold if request.threshold is not None else settings.abstain_threshold
    )
    yield from answer_stream(
        request.question,
        prepared.image_path,
        prepared.boxes,
        page=prepared.page_no,
        reader_chat=reader_chat,
        verifier_chat=verifier_chat,
        threshold=threshold,
        page_vectors=prepared.page_vectors,
        embed_query=embedder.embed_query,
        grid=prepared.grid,
    )
```

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run pytest tests/test_api_ask.py -q
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Extend `wire.to_frame` to handle `Retrieved`**

Append to `tests/test_api_wire.py`:

```python
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

    name, payload = to_frame(Retrieved(page=prepared, score=9.4, candidates=[other]))

    assert name == "retrieved"
    assert payload["doc_sha"] == "abc123"
    assert payload["page"] == 3
    assert payload["score"] == 9.4
    assert payload["candidates"] == [
        {"doc_sha": "abc123", "page": 7, "score": 8.1}
    ]
```

Then in `src/visual_verify/api/wire.py`, add the branch at the top of
`to_frame` and the import:

```python
def to_frame(event) -> tuple[str, dict]:
    """(event name, payload) for one event."""
    from visual_verify.api.ask import Retrieved

    if isinstance(event, Retrieved):
        return "retrieved", {
            "doc_sha": event.page.doc_sha,
            "doc_name": event.page.doc_name,
            "page": event.page.page_no,
            "score": event.score,
            "candidates": [
                {"doc_sha": c.doc_id, "page": c.page, "score": c.score}
                for c in event.candidates
            ],
        }
    if isinstance(event, ReadingStarted):
```

The import is function-local because `ask.py` imports `prepare.py`, which
imports SQLAlchemy; keeping it out of `wire.py`'s module scope means
`test_api_wire.py` stays a pure unit test of the strip.

- [ ] **Step 6: Run both wire and ask tests**

```bash
uv run pytest tests/test_api_wire.py tests/test_api_ask.py -q
```

Expected: PASS, 12 tests.

- [ ] **Step 7: Commit**

```bash
git add src/visual_verify/api/ask.py src/visual_verify/api/wire.py \
        tests/test_api_ask.py tests/test_api_wire.py
git commit -m "feat(api): join retrieval to the answer loop

The CLI never joined the seam: vvrag search ranks pages and stops, vvrag ask
requires a page number. A user types a question, so the service has to close it.

The candidate list is returned rather than discarded because retrieval can miss,
and without it a miss is an unexplainable wrong answer with no recourse. Pinning
a page yields the same event shape with an empty candidate list, so the frontend
has one code path."
```

---

## Task 10: The FastAPI app

**Files:**
- Create: `src/visual_verify/api/app.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api.py`:

```python
"""The HTTP surface, end to end, with fakes.

No GPU, no API key, no network. This must NOT become a fourth module that
loads ColQwen2: three already fragment the 3.63 GB card badly enough to need
expandable_segments, and a fourth would need process separation.
"""

import json

import pytest
from fastapi.testclient import TestClient

from visual_verify.agent.schemas import ClaimList, Verdict
from visual_verify.agent.types import FakeChat
from visual_verify.api.app import build_app
from visual_verify.api.resources import Resources
from visual_verify.cli import _make_index, main
from visual_verify.config import Settings
from visual_verify.store.engine import make_engine


@pytest.fixture
def client(tmp_path, monkeypatch, born_digital_pdf):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VVRAG_QDRANT_URL", ":memory:")
    monkeypatch.delenv("VVRAG_QDRANT_API_KEY", raising=False)
    monkeypatch.setenv("VVRAG_FAKE_EMBEDDER", "1")
    assert main(["ingest", str(born_digital_pdf)]) == 0
    assert main(["embed", "--all"]) == 0

    from visual_verify.retrieval.types import FakeEmbedder

    settings = Settings.from_env()
    resources = Resources(
        settings=settings,
        engine=make_engine(settings.db_url),
        index=_make_index(settings),
        embedder=FakeEmbedder(),
        reader_chat=FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent"])]),
        verifier_chat=FakeChat(
            "v", [Verdict(label="supported", confidence=0.9, reason="matches")]
        ),
    )
    app = build_app(resources)
    with TestClient(app) as c:
        yield c


def parse_sse(text):
    """[(event name, payload dict)] from a raw SSE body."""
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        name = lines[0].removeprefix("event: ")
        payload = json.loads(lines[1].removeprefix("data: "))
        out.append((name, payload))
    return out


def test_health_reports_the_two_model_ids(client):
    body = client.get("/health").json()

    assert body["reader_model"] == "r"
    assert body["verifier_model"] == "v"
    assert body["indexed_pages"] == 1


def test_documents_lists_what_was_ingested(client):
    body = client.get("/documents").json()

    assert len(body) == 1
    assert body[0]["name"] == "born_digital.pdf"
    assert body[0]["n_pages"] == 1


def test_the_page_image_is_served_as_png(client):
    sha = client.get("/documents").json()[0]["sha"]

    res = client.get(f"/documents/{sha}/pages/0/image")

    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert res.content.startswith(b"\x89PNG")


def test_an_unknown_page_image_is_404(client):
    sha = client.get("/documents").json()[0]["sha"]

    assert client.get(f"/documents/{sha}/pages/99/image").status_code == 404


def test_ask_streams_retrieved_then_claims_then_done(client):
    res = client.post("/ask", json={"question": "What happened?"})

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    names = [name for name, _ in parse_sse(res.text)]
    assert names == ["retrieved", "reading", "claims", "claim", "done"]


def test_every_streamed_claim_carries_a_verdict(client):
    """A claim reaching the wire unverified is the failure the streaming
    decision exists to avoid."""
    res = client.post("/ask", json={"question": "What happened?"})

    for name, payload in parse_sse(res.text):
        if name == "claim":
            assert payload["label"] is not None


def test_a_non_finite_threshold_is_rejected(client):
    res = client.post("/ask", json={"question": "q", "threshold": float("nan")})

    assert res.status_code == 422


def test_a_page_without_a_doc_is_rejected(client):
    res = client.post("/ask", json={"question": "q", "page": 0})

    assert res.status_code == 422


def test_asking_with_nothing_indexed_is_409(tmp_path, monkeypatch, born_digital_pdf):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'j.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data2"))
    monkeypatch.setenv("VVRAG_QDRANT_URL", ":memory:")
    monkeypatch.setenv("VVRAG_FAKE_EMBEDDER", "1")
    assert main(["ingest", str(born_digital_pdf)]) == 0

    from visual_verify.retrieval.types import FakeEmbedder

    settings = Settings.from_env()
    resources = Resources(
        settings=settings,
        engine=make_engine(settings.db_url),
        index=_make_index(settings),
        embedder=FakeEmbedder(),
        reader_chat=FakeChat("r", []),
        verifier_chat=FakeChat("v", []),
    )
    with TestClient(build_app(resources)) as c:
        assert c.post("/ask", json={"question": "q"}).status_code == 409
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_api.py -q
```

Expected: FAIL, `No module named 'visual_verify.api.app'`.

- [ ] **Step 3: Write the app**

Create `src/visual_verify/api/app.py`:

```python
"""The four routes.

build_app(resources) takes its resources rather than constructing them, which
is what lets the tests run the whole surface against FakeChat and FakeEmbedder
with no GPU and no key. create_app() is the production entry point that builds
them from the environment.
"""

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from visual_verify.api.ask import AskRequest, NoPagesIndexed, ask_events
from visual_verify.api.resources import Resources
from visual_verify.api.sse import frame
from visual_verify.api.stream import iter_in_thread
from visual_verify.api.wire import to_frame
from visual_verify.prepare import PageNotFound
from visual_verify.store.models import Document, Page


def build_app(resources: Resources) -> FastAPI:
    app = FastAPI(title="Verifiable Visual RAG")
    app.state.resources = resources

    # The frontend runs on a different origin in development. Deliberately not
    # "*": this service holds two billable API keys behind it.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # The GPU is single-tenant on a 3.63 GB card. Two concurrent asks would put
    # two queries through one embedder and, under load, OOM it. A second
    # request waits instead. Released in a finally, or one disconnected client
    # deadlocks every later question.
    ask_lock = asyncio.Semaphore(1)

    @app.get("/health")
    def health() -> dict:
        return {
            "reader_model": resources.reader_chat.model_id,
            "verifier_model": resources.verifier_chat.model_id,
            "embedder_resident": resources.embedder is not None,
            "indexed_pages": resources.index.count(),
        }

    @app.get("/documents")
    def documents() -> list[dict]:
        with Session(resources.engine) as session:
            docs = session.scalars(select(Document).order_by(Document.path)).all()
            return [
                {
                    "sha": d.sha256,
                    "name": Path(d.path).name,
                    "n_pages": d.n_pages,
                    "status": d.status,
                }
                for d in docs
            ]

    @app.get("/documents/{sha}/pages/{page_no}/image")
    def page_image(sha: str, page_no: int) -> FileResponse:
        """The filename is resolved through the database, never from the URL,
        so no user-supplied string reaches a filesystem path."""
        with Session(resources.engine) as session:
            page = session.scalar(
                select(Page).where(Page.doc_sha == sha, Page.page_no == page_no)
            )
        if page is None:
            raise HTTPException(404, f"no page {page_no} in {sha}")
        path = resources.settings.pages_dir / page.image_path
        if not path.exists():
            raise HTTPException(404, "the page image is missing from disk")
        return FileResponse(path, media_type="image/png")

    @app.post("/ask")
    async def ask(request: AskRequest) -> StreamingResponse:
        async def body():
            async with ask_lock:
                session = Session(resources.engine)

                def produce():
                    try:
                        yield from ask_events(
                            request,
                            session=session,
                            index=resources.index,
                            embedder=resources.embedder,
                            reader_chat=resources.reader_chat,
                            verifier_chat=resources.verifier_chat,
                            settings=resources.settings,
                        )
                    finally:
                        session.close()

                try:
                    async for event in iter_in_thread(produce):
                        name, payload = to_frame(event)
                        yield frame(name, payload)
                except Exception as exc:  # noqa: BLE001 - a provider or network
                    # failure must reach the browser as an error frame. Claims
                    # already delivered stay on screen; ending the stream
                    # silently would be indistinguishable from a short answer.
                    yield frame("error", {"message": f"{type(exc).__name__}: {exc}"})

        # The corpus check happens before streaming starts, so an unindexed
        # corpus is a status code rather than an error frame inside a 200.
        if resources.index.count() == 0:
            raise HTTPException(409, "no pages indexed; run `vvrag embed --all` first")

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def create_app() -> FastAPI:
    """Production entry point: uvicorn visual_verify.api.app:create_app --factory"""
    from visual_verify.api.resources import build
    from visual_verify.config import Settings

    return build_app(build(Settings.from_env()))
```

Note: `PageNotFound` and `NoPagesIndexed` raised inside `produce()` surface as
`error` frames, which is correct for a stream already in flight. The 409 for an
empty index is checked before the response starts, which is why it can be a
status code.

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run pytest tests/test_api.py -q
```

Expected: PASS, 9 tests.

- [ ] **Step 5: Run the whole Python side of S6**

```bash
uv run pytest tests/test_api.py tests/test_api_ask.py tests/test_api_wire.py \
  tests/test_api_sse.py tests/test_api_stream.py tests/test_api_resources.py \
  tests/test_answer_stream.py tests/test_prepare.py tests/test_agent.py \
  tests/test_contracts.py tests/test_cli.py tests/test_cli_retrieval.py \
  tests/test_core_is_light.py -q
```

Expected: PASS, no failures.

- [ ] **Step 6: Commit**

```bash
git add src/visual_verify/api/app.py tests/test_api.py
git commit -m "feat(api): serve health, documents, page images, and a streaming ask

build_app takes its resources instead of building them, so the whole HTTP
surface is testable against FakeChat and FakeEmbedder. That matters more than
convenience here: three test modules already load ColQwen2 in one pytest
process and fragment the card badly enough to need expandable_segments, and a
fourth would need process separation.

A semaphore serialises /ask because the GPU is single-tenant, and image paths
are resolved through the database so no user string reaches the filesystem."
```

---

## Task 11: Frontend scaffold and the overlay maths

**Files:**
- Create: `frontend/` (Next.js), `frontend/lib/overlay.ts`, `frontend/lib/overlay.test.ts`

- [ ] **Step 1: Scaffold**

```bash
cd /home/pursottam/mine/projects/verifiable-visual-rag
npx --yes create-next-app@latest frontend \
  --typescript --app --tailwind --eslint --no-src-dir --import-alias "@/*" --use-npm
cd frontend && npm install --save-dev vitest
```

Expected: `frontend/app/page.tsx` exists and `npm run build` succeeds.

- [ ] **Step 2: Add the test script**

In `frontend/package.json`, add to `"scripts"`:

```json
    "test": "vitest run"
```

- [ ] **Step 3: Write the failing test**

Create `frontend/lib/overlay.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { toStyle } from "./overlay";

describe("toStyle", () => {
  it("maps a normalized bbox to CSS percentages", () => {
    expect(toStyle([0.1, 0.2, 0.5, 0.6])).toEqual({
      left: "10%",
      top: "20%",
      width: "40%",
      height: "40%",
    });
  });

  // The bbox convention is (x0, y0, x1, y1) with origin top-left. S3 shipped a
  // patch-grid transposition that produced correctly-shaped, plausible output
  // and was wrong; the same bug is one swap away here. This asserts the axes
  // are not interchangeable, so a swapped implementation cannot pass.
  it("is not symmetric under an x/y swap", () => {
    const upright = toStyle([0.1, 0.2, 0.5, 0.6]);
    const swapped = toStyle([0.2, 0.1, 0.6, 0.5]);
    expect(upright).not.toEqual(swapped);
  });

  it("handles a full-page box", () => {
    expect(toStyle([0, 0, 1, 1])).toEqual({
      left: "0%",
      top: "0%",
      width: "100%",
      height: "100%",
    });
  });

  // A line box measured on proposal.pdf is 0.0142 tall. Rounding that to a
  // whole percent would collapse it to zero height and draw nothing.
  it("keeps sub-percent heights", () => {
    const style = toStyle([0.1, 0.5, 0.9, 0.5142]);
    expect(parseFloat(style.height)).toBeGreaterThan(0);
    expect(parseFloat(style.height)).toBeLessThan(2);
  });
});
```

- [ ] **Step 4: Run it to verify it fails**

```bash
cd frontend && npm test
```

Expected: FAIL, cannot resolve `./overlay`.

- [ ] **Step 5: Write the module**

Create `frontend/lib/overlay.ts`:

```ts
// Regions arrive as normalized 0-1 boxes against the displayed page rect,
// origin top-left. Percentages rescale for free at any display size, which is
// why S1 normalized the convention in the first place, and why this draws divs
// rather than a canvas.

export type BBox = [number, number, number, number];

export type Region = {
  bbox: BBox;
  score: number;
  modality: "visual" | "text";
  resolution: "line" | "block" | null;
  text: string | null;
};

export type BoxStyle = {
  left: string;
  top: string;
  width: string;
  height: string;
};

// No rounding. A line box on a real page measures 0.0142 of the page height,
// so rounding to whole percentages would give it zero height and draw nothing.
export function toStyle([x0, y0, x1, y1]: BBox): BoxStyle {
  return {
    left: `${x0 * 100}%`,
    top: `${y0 * 100}%`,
    width: `${(x1 - x0) * 100}%`,
    height: `${(y1 - y0) * 100}%`,
  };
}
```

- [ ] **Step 6: Run it to verify it passes**

```bash
cd frontend && npm test
```

Expected: PASS, 4 tests.

- [ ] **Step 7: Verify the transposition test can fail**

Temporarily swap the implementation to
`left: \`${y0 * 100}%\`, top: \`${x0 * 100}%\`, width: \`${(y1 - y0) * 100}%\`, height: \`${(x1 - x0) * 100}%\``
and re-run. Expected: FAIL on both the first and second tests. **Revert and
re-run to confirm PASS.**

- [ ] **Step 8: Add a frontend gitignore entry and commit**

Confirm `frontend/.gitignore` (created by the scaffold) ignores `node_modules`
and `.next`. Then:

```bash
cd /home/pursottam/mine/projects/verifiable-visual-rag
git add frontend
git status --short | head -20   # confirm no node_modules or .next staged
git commit -m "feat(ui): scaffold the frontend and pin the overlay coordinate maths

Regions are drawn as percentage-positioned divs rather than on a canvas, so
they rescale with whatever size the page is displayed at, which is the reason
S1 normalized the bbox convention.

The x/y swap test is the point of the module. S3 shipped a patch-grid
transposition that produced correctly shaped and entirely plausible output, and
an overlay is one swap away from the same bug with no numeric symptom."
```

---

## Task 12: The page

**Files:**
- Create: `frontend/lib/api.ts`, `frontend/app/page.tsx`
- Modify: `frontend/app/layout.tsx` (title only)

- [ ] **Step 1: Write the SSE client**

Create `frontend/lib/api.ts`:

```ts
import type { Region } from "./overlay";

// EventSource is GET-only, which would force the question into a query string
// and give up a request body for doc, page, k and threshold. Streaming a POST
// response is well supported and needs no extra machinery.

export const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";

export type ClaimEvent = {
  index: number;
  text: string;
  label: "supported" | "partially_supported" | "insufficient_evidence" | "unsupported" | null;
  confidence: number;
  reason: string | null;
  compound: boolean;
  withheld: boolean;
  regions: Region[];
};

export type Candidate = { doc_sha: string; page: number; score: number };

export type RetrievedEvent = {
  doc_sha: string;
  doc_name: string;
  page: number;
  score: number | null;
  candidates: Candidate[];
};

export type DoneEvent = { shown: number; withheld: number; abstained_overall: boolean };

type Handlers = {
  onRetrieved: (e: RetrievedEvent) => void;
  onClaims: (n: number) => void;
  onClaim: (e: ClaimEvent) => void;
  onDone: (e: DoneEvent) => void;
  onError: (message: string) => void;
};

export async function ask(
  body: { question: string; doc?: string; page?: number },
  h: Handlers,
): Promise<void> {
  const res = await fetch(`${API}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    h.onError(`${res.status}: ${(await res.text()) || res.statusText}`);
    return;
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Frames are separated by a blank line. A partial frame stays in the
    // buffer until its terminator arrives; parsing early would drop events.
    let split: number;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const [nameLine, dataLine] = block.split("\n");
      const name = nameLine.replace("event: ", "");
      const data = JSON.parse(dataLine.replace("data: ", ""));
      if (name === "retrieved") h.onRetrieved(data);
      else if (name === "claims") h.onClaims(data.n);
      else if (name === "claim") h.onClaim(data);
      else if (name === "done") h.onDone(data);
      else if (name === "error") h.onError(data.message);
    }
  }
}
```

- [ ] **Step 2: Write the page**

Create `frontend/app/page.tsx`:

```tsx
"use client";

import { useState } from "react";
import { API, ask, type Candidate, type ClaimEvent, type DoneEvent, type RetrievedEvent } from "@/lib/api";
import { toStyle } from "@/lib/overlay";

const CLAIM_COLORS = ["#0083d7", "#f45813", "#785ef0", "#1ab050", "#d81b60"];

export default function Home() {
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [retrieved, setRetrieved] = useState<RetrievedEvent | null>(null);
  const [expected, setExpected] = useState(0);
  const [claims, setClaims] = useState<ClaimEvent[]>([]);
  const [done, setDone] = useState<DoneEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const [showWithheld, setShowWithheld] = useState(false);

  async function submit(body: { question: string; doc?: string; page?: number }) {
    setBusy(true);
    setRetrieved(null);
    setClaims([]);
    setDone(null);
    setError(null);
    setExpected(0);
    setShowWithheld(false);
    await ask(body, {
      onRetrieved: setRetrieved,
      onClaims: setExpected,
      onClaim: (c) => setClaims((prev) => [...prev, c]),
      onDone: setDone,
      onError: setError,
    });
    setBusy(false);
  }

  const shown = claims.filter((c) => !c.withheld);
  const withheld = claims.filter((c) => c.withheld);

  return (
    <main className="mx-auto max-w-6xl p-6">
      <h1 className="text-2xl font-semibold">Verifiable Visual RAG</h1>

      <form
        className="mt-4 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (question.trim()) submit({ question });
        }}
      >
        <input
          className="flex-1 rounded border px-3 py-2"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about the indexed documents"
          disabled={busy}
        />
        <button className="rounded bg-black px-4 py-2 text-white disabled:opacity-50" disabled={busy}>
          {busy ? "Working" : "Ask"}
        </button>
      </form>

      {error && <p className="mt-4 rounded bg-red-50 p-3 text-red-800">{error}</p>}

      <div className="mt-6 grid grid-cols-1 gap-8 md:grid-cols-2">
        <section>
          {retrieved && (
            <p className="text-sm text-gray-600">
              Reading {retrieved.doc_name} page {retrieved.page}
              {retrieved.score !== null && ` (score ${retrieved.score.toFixed(2)})`}
            </p>
          )}
          {busy && claims.length < expected && (
            <p className="mt-2 text-sm text-gray-500">
              Verifying claim {claims.length + 1} of {expected}
            </p>
          )}

          {/* The abstain badge is the primary result, not a footnote on an
              empty list. It must not read as "weak answer". */}
          {done?.abstained_overall && (
            <div className="mt-4 rounded-lg border-2 border-amber-500 bg-amber-50 p-4">
              <p className="font-semibold text-amber-900">No answer given</p>
              <p className="mt-1 text-sm text-amber-800">
                Nothing on this page passed verification. The system is declining rather than
                guessing.
              </p>
            </div>
          )}

          <ul className="mt-4 space-y-3">
            {shown.map((c) => (
              <li
                key={c.index}
                className="rounded border p-3"
                style={{ borderLeft: `4px solid ${CLAIM_COLORS[c.index % CLAIM_COLORS.length]}` }}
                onMouseEnter={() => setHovered(c.index)}
                onMouseLeave={() => setHovered(null)}
              >
                <p>{c.text}</p>
                <p className="mt-1 text-xs text-gray-600">
                  {c.label} · {c.confidence.toFixed(2)}
                  {c.regions.some((r) => r.resolution === "block") && " · coarse region"}
                  {c.compound && " · asserts more than one thing"}
                </p>
              </li>
            ))}
          </ul>

          {withheld.length > 0 && (
            <div className="mt-4">
              <button
                className="text-sm text-gray-700 underline"
                onClick={() => setShowWithheld((v) => !v)}
              >
                {withheld.length} claim{withheld.length > 1 ? "s" : ""} withheld
              </button>
              {showWithheld && (
                <ul className="mt-2 space-y-2">
                  {withheld.map((c) => (
                    /* No region is drawn for these. The label and the reason
                       explain the refusal; the geometry never arrives. */
                    <li key={c.index} className="rounded bg-gray-50 p-3 text-sm">
                      <p className="text-gray-700">{c.text}</p>
                      <p className="mt-1 text-xs text-gray-500">
                        {c.label} · {c.confidence.toFixed(2)}
                      </p>
                      {c.reason && <p className="mt-1 text-xs italic text-gray-600">{c.reason}</p>}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {retrieved && retrieved.candidates.length > 0 && (
            <p className="mt-6 text-xs text-gray-500">
              Also considered:{" "}
              {retrieved.candidates.map((c: Candidate, i) => (
                <button
                  key={`${c.doc_sha}-${c.page}`}
                  className="underline"
                  disabled={busy}
                  onClick={() => submit({ question, doc: c.doc_sha, page: c.page })}
                >
                  page {c.page}
                  {i < retrieved.candidates.length - 1 ? ", " : ""}
                </button>
              ))}
            </p>
          )}
        </section>

        <section>
          {retrieved && (
            <div className="relative inline-block w-full">
              <img
                src={`${API}/documents/${retrieved.doc_sha}/pages/${retrieved.page}/image`}
                alt={`${retrieved.doc_name} page ${retrieved.page}`}
                className="w-full border"
              />
              {shown.flatMap((c) =>
                c.regions.map((r, i) => (
                  <div
                    key={`${c.index}-${i}`}
                    className="pointer-events-none absolute"
                    style={{
                      ...toStyle(r.bbox),
                      // Dashed means the heatmap could not separate the lines
                      // inside the winning block, so the region deliberately
                      // stayed coarse. Solid means a line-level hit.
                      border: `2px ${r.resolution === "block" ? "dashed" : "solid"} ${
                        CLAIM_COLORS[c.index % CLAIM_COLORS.length]
                      }`,
                      background: `${CLAIM_COLORS[c.index % CLAIM_COLORS.length]}22`,
                      opacity: hovered === null || hovered === c.index ? 1 : 0.25,
                    }}
                    title={`${r.modality} · ${r.resolution ?? "exact"} · ${r.score.toFixed(3)}`}
                  />
                )),
              )}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
```

- [ ] **Step 3: Set the title**

In `frontend/app/layout.tsx`, replace the exported `metadata` object with:

```tsx
export const metadata = {
  title: "Verifiable Visual RAG",
  description: "Region-level verifiable evidence for question answering over documents",
};
```

- [ ] **Step 4: Verify it builds and the tests still pass**

```bash
cd frontend && npm run build && npm test
```

Expected: build succeeds, 4 tests pass.

- [ ] **Step 5: Run it against the service and look at it**

Terminal 1:

```bash
cd /home/pursottam/mine/projects/verifiable-visual-rag
uv run uvicorn visual_verify.api.app:create_app --factory --workers 1 --port 8000
```

Terminal 2:

```bash
cd /home/pursottam/mine/projects/verifiable-visual-rag/frontend && npm run dev
```

Open `http://localhost:3000` and ask a question against an indexed document.

**Look at the overlay.** Every coordinate bug in this project was found by
looking at rendered output and none were found by reasoning about the
arithmetic. Confirm the boxes sit on the words they claim to, not near them.

- [ ] **Step 6: Commit**

```bash
cd /home/pursottam/mine/projects/verifiable-visual-rag
git add frontend
git commit -m "feat(ui): render verified claims, their regions, and the abstain badge

A withheld claim shows its label and the verifier's reason and never its
region, so nothing the verifier refused is pointed at on the page. Regions the
frontend does receive are drawn dashed when resolution is block, which is the
only place S4's coarse-fallback distinction becomes visible to a person.

The abstain badge is the primary result when nothing passes, not a footnote on
an empty list: a badge reading as weak answer rather than no answer would
misrepresent the behaviour the project exists to demonstrate."
```

---

## Task 13: Documentation and the roadmap

**Files:**
- Modify: `README.md`, `docs/ROADMAP.md`

- [ ] **Step 1: Document the run sequence**

Append to `README.md`:

```markdown
## Running the product UI (S6)

The service is read-only over a corpus built beforehand, so ingest and embed
first:

```bash
uv sync --all-extras --group dev
uv run vvrag ingest <pdf>
uv run vvrag embed --all
```

Then the service and the frontend, in two terminals:

```bash
uv run uvicorn visual_verify.api.app:create_app --factory --workers 1 --port 8000
cd frontend && npm run dev
```

`--workers 1` is not a default worth changing. Each worker loads its own
ColQwen2 at about 2.6 GB, and the development card has 3.63 GB, so a second
worker OOMs at startup. Startup takes about 20 seconds because the model loads
once for the process lifetime rather than once per request.

The service refuses to start if `VVRAG_QDRANT_URL` is unset, if an API key is
missing, or if the reader and the verifier resolve to the same model. That last
one is the point of the design and a misconfiguration would otherwise be
invisible in the output.
```

- [ ] **Step 2: Tick the roadmap**

In `docs/ROADMAP.md`, change the S6 row in the summary table's `Status` column
from `Not started` to `Done`, and tick every checkbox under `## S6: Product UI`.
Change that heading to `## S6: Product UI (done)`. Add below the checkboxes:

```markdown
Spec: `docs/superpowers/specs/2026-08-09-s6-product-ui-design.md`
Plan: `docs/superpowers/plans/2026-08-09-s6-product-ui.md`

Worth knowing: S6 is where the online pipeline first exists end to end. The CLI
still splits it, and deliberately so: `vvrag ask` takes an explicit page because
it is a debugging surface, while the service retrieves.

The CLI and the UI disagree about withheld claims on purpose. `vvrag ask` prints
every claim including rejected ones, because a diagnostic surface should show
everything. The UI uses `Answer.shown` and strips a rejected claim's regions in
`api/wire.py` before they leave the process. Same data, opposite default.
```

- [ ] **Step 3: Run the full suite once**

```bash
cd /home/pursottam/mine/projects/verifiable-visual-rag
uv run pytest -q > /tmp/claude-1000/-home-pursottam-mine-projects-verifiable-visual-rag/ab1aea5c-001c-4a82-8a57-90bda6f945db/scratchpad/s6-suite.log 2>&1
tail -5 /tmp/claude-1000/-home-pursottam-mine-projects-verifiable-visual-rag/ab1aea5c-001c-4a82-8a57-90bda6f945db/scratchpad/s6-suite.log
```

Expected: at least 380 passed plus the new tests, 2 skipped, about 16 minutes.
**Do not background this.** Redirect and read the log.

Then confirm the card is clear:

```bash
nvidia-smi --query-gpu=memory.used --format=csv
```

Expected: a few MiB. If a process is holding gigabytes, find and kill it:
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.

- [ ] **Step 4: Lint**

```bash
uv run ruff check src tests && uv run ruff format --check src tests
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/ROADMAP.md
git commit -m "docs(roadmap): close out S6

Records why the CLI and the UI disagree about withheld claims: vvrag ask prints
everything because it is a diagnostic surface, and the service strips a rejected
claim's regions before they leave the process. Same data, opposite default, and
somebody reading only one of them would otherwise call the other a bug."
```

---

## Done when

- [ ] `uv run pytest` passes with no new failures and the GPU is back to a few MiB
- [ ] `cd frontend && npm test && npm run build` both pass
- [ ] Asking a question in the browser draws a region on the page image, and the
      box sits on the words it claims to (checked by looking, not by reasoning)
- [ ] A withheld claim shows its reason and draws nothing
- [ ] The service refuses to start with the reader and verifier set to one model
- [ ] `git status` shows `CLAUDE.md` untracked and unstaged
