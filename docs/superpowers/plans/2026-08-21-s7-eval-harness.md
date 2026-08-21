# S7 Eval Harness (Retrieval-Augmented Generation) -- Phase 1: Metrics + Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and prove `visual_verify.eval.metrics` (EM/F1, IoU/hit-rate, confident-wrong-vs-coverage) and `visual_verify.eval.harness` (the three-arm Baseline/Grounded/Verified runner), against a small hand-written fixture, with no real SlideVQA data, no OCR, no GPU, and no network.

**Architecture:** Two new files under `src/visual_verify/eval/`. `metrics.py` is pure functions with no dependency on anything else in the package. `harness.py` composes the existing `read()`, `ground()`, and `verify()` functions (already separately importable from `agent/reader.py`, `grounding/core.py`, `agent/verifier.py`) into the three ablation arms, using the existing `FakeChat` test double so this whole phase runs with no API key. Per the approved design
(`docs/superpowers/specs/2026-08-21-s7-eval-harness-design.md`), the real SlideVQA
download/OCR/PDF-assembly pipeline (`eval/dataset.py`) and the CLI commands are a
**separate, later plan** -- out of scope here.

**Tech Stack:** Python 3.12, pydantic (dataclasses for the new types, matching `contracts.py`'s style), pytest. No new dependencies: everything this plan touches already ships in the core package or the existing `agent`/no-extra grounding code.

---

## Context you need before starting

- `src/visual_verify/contracts.py` defines `BBox = tuple[float, float, float, float]` (normalized 0-1, `(x0, y0, x1, y1)`), `GroundedRegion`, `Claim`, `LEAD_INDEX = 0`.
- `src/visual_verify/agent/reader.py::read(chat, image_paths, question) -> list[DraftedClaim]`. `DraftedClaim` (in `agent/schemas.py`) has `.text: str` and `.starts_paragraph: bool`. `read()` sends `ClaimList` as the requested schema to `chat.structured(...)`, so a `FakeChat` used with `read()` must be scripted with `ClaimList(claims=[DraftedClaim(text=..., starts_paragraph=...)])` responses, not bare `DraftedClaim` lists.
- `src/visual_verify/grounding/core.py::ground(claim, boxes, *, page, page_vectors=None, query_vectors=None, grid=None, force=None, reduce="mean") -> list[GroundedRegion]`. Text-span matching runs first automatically when `force != "visual"`; passing `force="text"` forces it and guarantees `ground()` never raises `GroundingError` even with no vectors. This plan's harness always calls `ground()` with `force="text"` -- the visual (heatmap) path needs real page/query vectors, which only exist once `eval/dataset.py` (a later plan) embeds real pages. The fixture's claims are written to match text on the page, so this is not a limitation for THIS plan's tests, only a known simplification to record in the module docstring.
- `src/visual_verify/agent/verifier.py::verify(chat, image_path, claim, regions) -> Verdict`. `Verdict` (in `agent/schemas.py`) has `.label`, `.confidence`, `.reason`.
- `src/visual_verify/agent/rubric.py::abstention_score(label, confidence) -> float` and `SUPPORTED_FLOOR` (the default abstain threshold, exported from this module).
- `src/visual_verify/agent/types.py::FakeChat(_model_id, responses, calls=[], _next=0)` and `StructuredChat` (the protocol). `FakeChat.structured()` never touches the filesystem, so tests can pass `Path("fake.png")` as an image path without creating a real file.
- `src/visual_verify/ingest/boxes.py::BoxRecord` is a frozen dataclass: `kind: BoxKind, x0, y0, x1, y1: float, text: str, block_no=-1, line_no=-1, word_no=-1`, plus a `.bbox` property returning `(x0, y0, x1, y1)`.
- Test file naming in this repo: `tests/test_<module>.py`, one file per source module (see `tests/test_verifier.py` next to `agent/verifier.py`). Follow that pattern: `tests/test_eval_metrics.py` and `tests/test_eval_harness.py`.
- Per this repo's subagent test-scope rule: each task below runs ONLY its own test file, never the full suite. The full suite is run once, in the final task.

---

### Task 1: `eval` package skeleton + answer-accuracy metrics (EM, F1)

**Files:**
- Create: `src/visual_verify/eval/__init__.py`
- Create: `src/visual_verify/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

- [ ] **Step 1: Create the empty package**

```bash
mkdir -p src/visual_verify/eval
```

`src/visual_verify/eval/__init__.py`:

```python
"""S7 evaluation: pure metrics (this file's siblings) and the ablation
harness that produces the records they score.

Deliberately no re-exports here. Every consumer imports from
visual_verify.eval.metrics or visual_verify.eval.harness directly, so it is
never ambiguous which module a name came from.
"""
```

- [ ] **Step 2: Write the failing test for answer-accuracy metrics**

`tests/test_eval_metrics.py`:

```python
"""Pure scoring functions: no I/O, no model calls, no fixtures beyond literals."""

from visual_verify.eval.metrics import exact_match, normalize_answer, token_f1


def test_normalize_answer_lowercases_strips_punctuation_and_articles():
    assert normalize_answer("The Revenue Grew, 42%!") == "revenue grew 42"


def test_exact_match_ignores_case_articles_and_punctuation():
    assert exact_match("The answer is Paris.", "answer is paris") is True


def test_exact_match_false_on_different_answer():
    assert exact_match("Paris", "London") is False


def test_token_f1_is_one_on_a_perfect_match():
    assert token_f1("Revenue grew 42 percent", "revenue grew 42 percent") == 1.0


def test_token_f1_partial_overlap():
    # prediction: {revenue, grew, 42} ; reference: {revenue, grew, 42, percent}
    # precision = 3/3 = 1.0, recall = 3/4 = 0.75, F1 = 2*1.0*0.75/1.75
    f1 = token_f1("Revenue grew 42", "revenue grew 42 percent")
    assert abs(f1 - (2 * 1.0 * 0.75 / 1.75)) < 1e-9


def test_token_f1_is_zero_on_no_overlap():
    assert token_f1("apples", "oranges") == 0.0


def test_token_f1_handles_empty_prediction():
    assert token_f1("", "revenue grew") == 0.0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'visual_verify.eval.metrics'`

- [ ] **Step 4: Implement the metrics module's answer-accuracy half**

`src/visual_verify/eval/metrics.py`:

```python
"""Pure scoring functions for the S7 ablation. No I/O, no model calls.

Answer accuracy follows the SQuAD/HotpotQA normalize-then-compare convention
that proposal.tex line 424 commits to: lowercase, strip punctuation, drop the
articles a/an/the, collapse whitespace. Both exact_match and token_f1 score
against the SAME normalized strings, so a prediction that would exact-match
after normalization always scores F1 1.0 too -- the two metrics can disagree
on partial credit, never on whether they see the same text.
"""

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower().translate(_PUNCTUATION)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: PASS (7 passed)

- [ ] **Step 6: Commit**

```bash
git add src/visual_verify/eval/__init__.py src/visual_verify/eval/metrics.py tests/test_eval_metrics.py
git commit -m "feat(eval): add exact-match and token-F1 answer scoring"
```

---

### Task 2: Grounding metrics (mean IoU, hit-rate@0.25)

**Files:**
- Modify: `src/visual_verify/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_metrics.py`:

```python
from visual_verify.eval.metrics import GroundingScore, grounding_metrics, iou


def test_iou_of_identical_boxes_is_one():
    box = (0.1, 0.1, 0.5, 0.5)
    assert iou(box, box) == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    assert iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0


def test_iou_of_partially_overlapping_boxes():
    # a: [0,0,0.2,0.2] area 0.04; b: [0.1,0.1,0.3,0.3] area 0.04
    # intersection: [0.1,0.1,0.2,0.2] area 0.01; union = 0.04+0.04-0.01 = 0.07
    v = iou((0.0, 0.0, 0.2, 0.2), (0.1, 0.1, 0.3, 0.3))
    assert abs(v - (0.01 / 0.07)) < 1e-9


def test_grounding_metrics_scores_a_missing_prediction_as_a_miss():
    """predicted_bbox=None (nothing cited, or the claim was withheld) must
    count as IoU 0 against the gold box, not be dropped from the denominator --
    otherwise a system that cites nothing would score a perfect mean IoU over
    an empty set."""
    gold = (0.1, 0.1, 0.5, 0.5)
    result = grounding_metrics([(None, gold)])
    assert result == GroundingScore(mean_iou=0.0, hit_rate=0.0, n_scored=1)


def test_grounding_metrics_hit_rate_uses_the_given_threshold():
    gold = (0.0, 0.0, 0.2, 0.2)
    close_hit = (0.0, 0.0, 0.18, 0.18)  # high IoU, above 0.25
    near_miss = (0.0, 0.0, 0.05, 0.05)  # low IoU, below 0.25
    result = grounding_metrics([(close_hit, gold), (near_miss, gold)], threshold=0.25)
    assert result.hit_rate == 0.5
    assert result.n_scored == 2


def test_grounding_metrics_on_empty_input():
    assert grounding_metrics([]) == GroundingScore(mean_iou=0.0, hit_rate=0.0, n_scored=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: FAIL with `ImportError: cannot import name 'GroundingScore'`

- [ ] **Step 3: Implement grounding metrics**

Add these two imports at the TOP of `src/visual_verify/eval/metrics.py`, alongside the existing `re`/`string`/`collections.Counter` imports (not appended at the bottom -- ruff's E402 flags module-level imports that follow code):

```python
from dataclasses import dataclass
```

Then append to the bottom of `src/visual_verify/eval/metrics.py`:

```python
BBox = tuple[float, float, float, float]


def iou(box_a: BBox, box_b: BBox) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter_w, inter_h = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    intersection = inter_w * inter_h
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class GroundingScore:
    mean_iou: float
    hit_rate: float
    n_scored: int


def grounding_metrics(
    pairs: list[tuple[BBox | None, BBox]], threshold: float = 0.25
) -> GroundingScore:
    """Score predicted citations against auto-derived gold boxes.

    `pairs` holds only questions that HAVE a gold box (the caller filters
    those out -- see the S7 design's Section 3, questions whose reference
    answer never appears in the OCR'd text layer get no gold box and are
    excluded before this function is ever called). A predicted box of None
    (no citation, or the claim was withheld before verification) scores IoU
    0.0 here rather than being dropped: a system that never cites anything
    must not score a perfect mean IoU over an empty set.
    """
    if not pairs:
        return GroundingScore(mean_iou=0.0, hit_rate=0.0, n_scored=0)
    ious = [iou(pred, gold) if pred is not None else 0.0 for pred, gold in pairs]
    hits = sum(1 for v in ious if v >= threshold)
    return GroundingScore(
        mean_iou=sum(ious) / len(ious), hit_rate=hits / len(ious), n_scored=len(pairs)
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/eval/metrics.py tests/test_eval_metrics.py
git commit -m "feat(eval): add IoU and hit-rate grounding metrics"
```

---

### Task 3: Abstention metrics (confident-wrong rate vs. coverage)

**Files:**
- Modify: `src/visual_verify/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_metrics.py`:

```python
from visual_verify.eval.metrics import AbstentionScore, abstention_metrics


def test_abstention_metrics_all_correct_and_answered():
    result = abstention_metrics(["correct", "correct", "correct"])
    assert result == AbstentionScore(
        coverage=1.0, confident_wrong_rate=0.0, n_total=3, n_answered=3, n_wrong=0
    )


def test_abstention_metrics_counts_wrong_answers_shown_as_confident_wrong():
    # 4 total, 1 abstained (not covered), 3 answered, 1 of those wrong
    result = abstention_metrics(["correct", "wrong", "correct", "abstained"])
    assert result.coverage == 0.75
    assert abs(result.confident_wrong_rate - (1 / 3)) < 1e-9
    assert result.n_total == 4
    assert result.n_answered == 3
    assert result.n_wrong == 1


def test_abstention_metrics_all_abstained_has_zero_coverage_and_zero_confident_wrong():
    """Zero coverage, not a division error: nothing was shown, so there is
    nothing to be confidently wrong about."""
    result = abstention_metrics(["abstained", "abstained"])
    assert result.coverage == 0.0
    assert result.confident_wrong_rate == 0.0


def test_abstention_metrics_on_empty_input():
    result = abstention_metrics([])
    assert result == AbstentionScore(
        coverage=0.0, confident_wrong_rate=0.0, n_total=0, n_answered=0, n_wrong=0
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: FAIL with `ImportError: cannot import name 'AbstentionScore'`

- [ ] **Step 3: Implement abstention metrics**

Add this import at the TOP of `src/visual_verify/eval/metrics.py`, alongside the other imports:

```python
from typing import Literal
```

Then append to the bottom of `src/visual_verify/eval/metrics.py`:

```python
AnswerOutcome = Literal["correct", "wrong", "abstained"]


@dataclass(frozen=True)
class AbstentionScore:
    coverage: float
    confident_wrong_rate: float
    n_total: int
    n_answered: int
    n_wrong: int


def abstention_metrics(outcomes: list[AnswerOutcome]) -> AbstentionScore:
    """Coverage and confident-wrong rate, proposal.tex's headline abstention pair.

    coverage: the fraction of questions the system chose to answer at all.
    confident_wrong_rate: of the ones it answered, the fraction that were
    wrong -- "wrong" here means exact_match failed against the reference,
    computed by the caller (this function takes the already-classified
    outcome, not the raw prediction, so it stays a pure aggregation with no
    normalization policy baked in).
    """
    n_total = len(outcomes)
    if n_total == 0:
        return AbstentionScore(
            coverage=0.0, confident_wrong_rate=0.0, n_total=0, n_answered=0, n_wrong=0
        )
    n_answered = sum(1 for o in outcomes if o != "abstained")
    n_wrong = sum(1 for o in outcomes if o == "wrong")
    coverage = n_answered / n_total
    confident_wrong_rate = n_wrong / n_answered if n_answered else 0.0
    return AbstentionScore(
        coverage=coverage,
        confident_wrong_rate=confident_wrong_rate,
        n_total=n_total,
        n_answered=n_answered,
        n_wrong=n_wrong,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: PASS (17 passed)

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/eval/metrics.py tests/test_eval_metrics.py
git commit -m "feat(eval): add confident-wrong-vs-coverage abstention metric"
```

---

### Task 4: Harness types + the Baseline arm

**Files:**
- Create: `src/visual_verify/eval/harness.py`
- Test: `tests/test_eval_harness.py`

- [ ] **Step 1: Write the failing test**

`tests/test_eval_harness.py`:

```python
"""The three ablation arms, run against FakeChat -- no network, no key.

Fixture claims are written to match text already on the fixture page, so
ground() finds them with force="text" and never needs real patch vectors.
That is a deliberate simplification of THIS phase: eval/dataset.py (a later
plan) wires real ColQwen2 vectors so the visual snap-to-box path can be
exercised too. See harness.py's module docstring.
"""

from pathlib import Path

from visual_verify.agent.schemas import ClaimList, DraftedClaim, Verdict
from visual_verify.agent.types import FakeChat
from visual_verify.eval.harness import ArmResult, EvalQuestion, run_baseline
from visual_verify.ingest.boxes import BoxRecord


def revenue_question() -> EvalQuestion:
    boxes = [
        BoxRecord(kind="word", x0=0.1, y0=0.1, x1=0.5, y1=0.15, text="Revenue grew 42 percent"),
    ]
    return EvalQuestion(
        question="How much did revenue grow?",
        # Matches the reader's drafted claim text below verbatim (after
        # normalize_answer's case/punctuation folding), so exact_match is
        # True in Task 7's smoke test. A reference answer like "42 percent"
        # would NOT exact-match "Revenue grew 42 percent" -- exact_match is
        # whole-string equality after normalization, not substring
        # containment, so an answer that pads the reference with extra
        # correct words still scores "wrong".
        reference_answer="Revenue grew 42 percent",
        page=0,
        boxes=boxes,
        image_path=Path("fake_page.png"),
        gold_bbox=(0.1, 0.1, 0.5, 0.15),
    )


def test_run_baseline_returns_the_readers_claims_joined_with_no_grounding():
    q = revenue_question()
    chat = FakeChat(
        "m",
        [ClaimList(claims=[DraftedClaim(text="Revenue grew 42 percent.", starts_paragraph=True)])],
    )

    result = run_baseline(chat, q)

    assert result == ArmResult(
        arm="baseline",
        question=q.question,
        reference_answer=q.reference_answer,
        predicted_answer="Revenue grew 42 percent.",
        abstained=False,
        predicted_bbox=None,
        gold_bbox=q.gold_bbox,
    )
    # The reader saw exactly the one page image this question carries.
    assert chat.calls[0].image_paths == [q.image_path]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_eval_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'visual_verify.eval.harness'`

- [ ] **Step 3: Implement the harness types and `run_baseline`**

`src/visual_verify/eval/harness.py`:

```python
"""The three-arm S7 ablation: Baseline / Grounded / Verified.

All three arms are composed from the SAME already-separable pieces the
product pipeline uses -- read(), ground(), verify() -- rather than a
stage-skip switch added to agent/core.py's answer(). Each arm is a strict
superset of the one before it (proposal.tex line 472's ablation design):
Baseline reads, Grounded adds ground(), Verified adds verify() and the
abstention gate.

ground() is always called with force="text" in this phase. The visual
(heatmap) snap-to-box path needs real ColQwen2 page/query vectors, which only
exist once eval/dataset.py (a later plan) embeds the real deck pool; until
then, a claim whose text is not on the fixture/deck page's text layer simply
gets no region, the same as `ground()`'s documented meaning for "no evidence
found" -- it is never mistaken for a GroundingError.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from visual_verify.agent.reader import read
from visual_verify.agent.rubric import SUPPORTED_FLOOR, abstention_score
from visual_verify.agent.types import StructuredChat
from visual_verify.agent.verifier import verify
from visual_verify.contracts import LEAD_INDEX, BBox
from visual_verify.grounding import ground
from visual_verify.ingest.boxes import BoxRecord

Arm = Literal["baseline", "grounded", "verified"]


@dataclass(frozen=True)
class EvalQuestion:
    """One SlideVQA question, already resolved to a page and its boxes.

    `gold_bbox` is None when the reference answer does not appear literally
    in the page's text layer (S7 design Section 3) -- such a question still
    runs through every arm for answer-accuracy scoring, it is only excluded
    from grounding_metrics by the caller.
    """

    question: str
    reference_answer: str
    page: int
    boxes: list[BoxRecord]
    image_path: Path
    gold_bbox: BBox | None


@dataclass(frozen=True)
class ArmResult:
    arm: Arm
    question: str
    reference_answer: str
    predicted_answer: str
    abstained: bool
    predicted_bbox: BBox | None
    gold_bbox: BBox | None


def run_baseline(chat: StructuredChat, q: EvalQuestion) -> ArmResult:
    """Plain document RAG: read the page, answer, no grounding, no verification."""
    claims = read(chat, [q.image_path], q.question)
    predicted_answer = " ".join(c.text for c in claims)
    return ArmResult(
        arm="baseline",
        question=q.question,
        reference_answer=q.reference_answer,
        predicted_answer=predicted_answer,
        abstained=False,
        predicted_bbox=None,
        gold_bbox=q.gold_bbox,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_eval_harness.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/eval/harness.py tests/test_eval_harness.py
git commit -m "feat(eval): add the Baseline ablation arm"
```

---

### Task 5: The Grounded arm

**Files:**
- Modify: `src/visual_verify/eval/harness.py`
- Test: `tests/test_eval_harness.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_harness.py`:

```python
from visual_verify.eval.harness import run_grounded


def test_run_grounded_cites_the_lead_claims_region():
    q = revenue_question()
    chat = FakeChat(
        "m",
        [ClaimList(claims=[DraftedClaim(text="Revenue grew 42 percent", starts_paragraph=True)])],
    )

    result = run_grounded(chat, q)

    assert result.arm == "grounded"
    assert result.predicted_answer == "Revenue grew 42 percent"
    assert result.abstained is False  # Grounded never abstains; see harness.py
    assert result.predicted_bbox == (0.1, 0.1, 0.5, 0.15)


def test_run_grounded_has_no_citation_when_the_lead_claim_is_not_on_the_page():
    q = revenue_question()
    chat = FakeChat(
        "m",
        [ClaimList(claims=[DraftedClaim(text="Something not on the page", starts_paragraph=True)])],
    )

    result = run_grounded(chat, q)

    assert result.predicted_bbox is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_eval_harness.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_grounded'`

- [ ] **Step 3: Implement `run_grounded`**

Append to `src/visual_verify/eval/harness.py`:

```python
def run_grounded(chat: StructuredChat, q: EvalQuestion) -> ArmResult:
    """Read, then ground the lead claim. No verification, no abstention gate.

    Only the LEAD claim (index LEAD_INDEX, the sentence that answers the
    question) is graded here, matching how confident-wrong-vs-coverage and
    the product's own abstention rule both key off the lead in
    contracts.Answer. Supporting claims after it are joined into the answer
    text but not individually graded for grounding.
    """
    claims = read(chat, [q.image_path], q.question)
    lead = claims[LEAD_INDEX]
    predicted_answer = " ".join(c.text for c in claims)
    regions = ground(lead.text, q.boxes, page=q.page, force="text")
    predicted_bbox = regions[0].bbox if regions else None
    return ArmResult(
        arm="grounded",
        question=q.question,
        reference_answer=q.reference_answer,
        predicted_answer=predicted_answer,
        abstained=False,
        predicted_bbox=predicted_bbox,
        gold_bbox=q.gold_bbox,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_eval_harness.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/eval/harness.py tests/test_eval_harness.py
git commit -m "feat(eval): add the Grounded ablation arm"
```

---

### Task 6: The Verified arm

**Files:**
- Modify: `src/visual_verify/eval/harness.py`
- Test: `tests/test_eval_harness.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_harness.py`:

```python
from visual_verify.eval.harness import run_verified


def test_run_verified_shows_a_supported_claim():
    q = revenue_question()
    chat = FakeChat(
        "m",
        [ClaimList(claims=[DraftedClaim(text="Revenue grew 42 percent", starts_paragraph=True)])],
    )
    verifier_chat = FakeChat("v", [Verdict(label="supported", confidence=0.9, reason="matches")])

    result = run_verified(chat, verifier_chat, q)

    assert result.arm == "verified"
    assert result.abstained is False
    assert result.predicted_bbox == (0.1, 0.1, 0.5, 0.15)
    # The verifier saw the page image and the claim it is judging.
    assert verifier_chat.calls[0].image_paths == [q.image_path]
    assert "Revenue grew 42 percent" in verifier_chat.calls[0].prompt


def test_run_verified_abstains_below_the_threshold():
    q = revenue_question()
    chat = FakeChat(
        "m",
        [ClaimList(claims=[DraftedClaim(text="Revenue grew 42 percent", starts_paragraph=True)])],
    )
    verifier_chat = FakeChat(
        "v", [Verdict(label="partially_supported", confidence=0.5, reason="partial")]
    )

    result = run_verified(chat, verifier_chat, q)

    assert result.abstained is True


def test_run_verified_abstains_when_there_is_no_region_to_judge():
    """Matches the product rule stated in agent/core.py and CLAUDE.md:
    abstained = score < threshold OR no regions, so a region-less claim
    cannot slip through on a lucky verdict."""
    q = revenue_question()
    chat = FakeChat(
        "m",
        [ClaimList(claims=[DraftedClaim(text="Not on the page at all", starts_paragraph=True)])],
    )
    verifier_chat = FakeChat(
        "v", [Verdict(label="insufficient_evidence", confidence=1.0, reason="nothing to check")]
    )

    result = run_verified(chat, verifier_chat, q)

    assert result.abstained is True
    assert result.predicted_bbox is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_eval_harness.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_verified'`

- [ ] **Step 3: Implement `run_verified`**

Append to `src/visual_verify/eval/harness.py`:

```python
def run_verified(
    reader_chat: StructuredChat,
    verifier_chat: StructuredChat,
    q: EvalQuestion,
    threshold: float = SUPPORTED_FLOOR,
) -> ArmResult:
    """The real pipeline's shape: read, ground the lead claim, verify it,
    apply the abstention gate.

    Mirrors Claim.withheld / core.py's rule exactly: abstained = score below
    threshold OR no regions at all. A claim with nothing to cite is never
    shown on the strength of a verdict alone -- see
    test_run_verified_abstains_when_there_is_no_region_to_judge.
    """
    claims = read(reader_chat, [q.image_path], q.question)
    lead = claims[LEAD_INDEX]
    predicted_answer = " ".join(c.text for c in claims)
    regions = ground(lead.text, q.boxes, page=q.page, force="text")
    predicted_bbox = regions[0].bbox if regions else None

    verdict = verify(verifier_chat, q.image_path, lead.text, regions)
    score = abstention_score(verdict.label, verdict.confidence)
    abstained = score < threshold or not regions

    return ArmResult(
        arm="verified",
        question=q.question,
        reference_answer=q.reference_answer,
        predicted_answer=predicted_answer,
        abstained=abstained,
        predicted_bbox=predicted_bbox,
        gold_bbox=q.gold_bbox,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_eval_harness.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/eval/harness.py tests/test_eval_harness.py
git commit -m "feat(eval): add the Verified ablation arm and abstention gate"
```

---

### Task 7: Wire a metrics-over-harness-output smoke test, and run the full suite

**Files:**
- Test: `tests/test_eval_harness.py`

This task proves the two modules actually compose end to end: `ArmResult`s
from Task 4-6 feed `grounding_metrics` and `abstention_metrics` from Tasks
2-3 correctly. No new production code -- if this step needs a code change,
it means a field name or shape drifted between `metrics.py` and
`harness.py`, and that drift must be fixed before this plan is done.

- [ ] **Step 1: Write the smoke test**

Append to `tests/test_eval_harness.py`:

```python
from visual_verify.eval.metrics import (
    AbstentionScore,
    GroundingScore,
    abstention_metrics,
    exact_match,
    grounding_metrics,
)


def test_verified_arm_results_feed_the_metrics_functions():
    """End-to-end: two questions, one supported and grounded, one abstained,
    scored through the exact functions eval/dataset.py's real run will use."""
    supported_q = revenue_question()
    unsupported_q = EvalQuestion(
        question="What was the margin?",
        reference_answer="steady",
        page=0,
        boxes=[BoxRecord(kind="word", x0=0.6, y0=0.6, x1=0.9, y1=0.65, text="Margins held steady")],
        image_path=Path("fake_page.png"),
        gold_bbox=(0.6, 0.6, 0.9, 0.65),
    )

    supported_chat = FakeChat(
        "m",
        [ClaimList(claims=[DraftedClaim(text="Revenue grew 42 percent", starts_paragraph=True)])],
    )
    supported_verifier = FakeChat("v", [Verdict(label="supported", confidence=0.9, reason="ok")])
    weak_chat = FakeChat(
        "m", [ClaimList(claims=[DraftedClaim(text="Not on the page", starts_paragraph=True)])]
    )
    weak_verifier = FakeChat(
        "v", [Verdict(label="insufficient_evidence", confidence=0.9, reason="no evidence")]
    )

    results = [
        run_verified(supported_chat, supported_verifier, supported_q),
        run_verified(weak_chat, weak_verifier, unsupported_q),
    ]

    grounding = grounding_metrics(
        [(r.predicted_bbox, r.gold_bbox) for r in results if r.gold_bbox is not None]
    )
    assert grounding == GroundingScore(mean_iou=0.5, hit_rate=0.5, n_scored=2)

    outcomes = [
        "abstained"
        if r.abstained
        else ("correct" if exact_match(r.predicted_answer, r.reference_answer) else "wrong")
        for r in results
    ]
    abstention = abstention_metrics(outcomes)
    assert abstention == AbstentionScore(
        coverage=0.5, confident_wrong_rate=0.0, n_total=2, n_answered=1, n_wrong=0
    )
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_eval_harness.py -v`
Expected: PASS (7 passed). If it fails, it means `ArmResult.predicted_bbox`/`gold_bbox` or an `AbstentionScore`/`GroundingScore` field does not match what Tasks 2-3 defined -- fix the mismatch in `harness.py` or `metrics.py`, not in this test.

- [ ] **Step 3: Run the full test suite once**

Run: `uv run pytest`
Expected: all tests pass, including the pre-existing suite. This is the ONE full-suite run for this plan, per this repo's subagent-scoping rule.

- [ ] **Step 4: Commit**

```bash
git add tests/test_eval_harness.py
git commit -m "test(eval): pin metrics-and-harness composition with an end-to-end smoke test"
```

---

## What this plan deliberately does not build

Per the approved design (`docs/superpowers/specs/2026-08-21-s7-eval-harness-design.md`), these are separate, later work:

- `eval/dataset.py`: SlideVQA download, OCR, synthetic-PDF assembly, real ingest/embed/index into the `slidevqa_eval` Qdrant collection.
- CLI commands (`vvrag eval prepare`, `vvrag eval run`).
- The real ~15-20 deck / ~100-150 question run against gpt-5-nano (reader) and Groq's `qwen/qwen3.6-27b` (verifier).
- Exercising the visual (heatmap) snap-to-box grounding path, which needs real page/query vectors that only exist once the real deck pool is embedded.
