# S5 Reader and Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `answer(question, ...)`, which reads a page with one hosted model, grounds each claim with S4, has a *different* hosted model judge the evidence, and withholds claims that fail.

**Architecture:** A narrow `StructuredChat` protocol is the seam. LangChain lives behind exactly one implementation of it; every other module takes the protocol. A `FakeChat` implements the same protocol, so the entire pipeline is testable with no network, no API key, and no LangChain import. This mirrors how `Embedder` and `FakeEmbedder` already work in `retrieval/types.py`.

**Tech Stack:** pydantic (already core), LangChain plus `langchain-openai` and `langchain-google-genai` behind a new `agent` extra. No GPU.

**Spec:** `docs/superpowers/specs/2026-08-08-s5-reader-verifier-design.md`. Read it first, especially sections 2 (why hosted), 6 (abstention arithmetic), and 9 (why no streaming).

---

## Ground rules for every task

**Test only your own files.** Run the specific test file(s) your task names. The full suite runs once, in Task 12, and takes 12 minutes.

**Never commit `CLAUDE.md` or `.env`.** Both gitignored. Use explicit paths, never `git add -A`.

**No Claude or AI attribution in commit messages.** Graded college project.

**No em-dashes in any prose.** Explicit, repeatedly enforced project rule.

**Line length 100.** Run `uv run ruff check` and `uv run ruff format --check` on every file you touch.

**The seam rule.** `read()` and `verify()` take a `StructuredChat` as an argument. They never construct one. If you find yourself importing `langchain` anywhere except `models.py`, stop: Task 12 has a subprocess test that fails if you do.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/visual_verify/agent/__init__.py` | Public surface: `answer`, `AgentError` |
| `src/visual_verify/agent/rubric.py` | The 4 labels, their ranks, and the abstention score |
| `src/visual_verify/agent/schemas.py` | Pydantic schemas the models must return |
| `src/visual_verify/agent/types.py` | `StructuredChat` protocol and `FakeChat` |
| `src/visual_verify/agent/cache.py` | Content-addressed response cache, wraps any `StructuredChat` |
| `src/visual_verify/agent/models.py` | The only file that imports LangChain |
| `src/visual_verify/agent/reader.py` | page image + question -> claims |
| `src/visual_verify/agent/verifier.py` | claim + regions + image -> verdict |
| `src/visual_verify/agent/core.py` | `answer()`, the pipeline |
| `tests/test_rubric.py` | Label ordering and score arithmetic |
| `tests/test_agent_cache.py` | Cache keying, hit, miss, model-id isolation |
| `tests/test_reader.py` | Claim extraction and the compound-claim check |
| `tests/test_verifier.py` | Verdicts, and that the verifier can say no |
| `tests/test_agent.py` | `answer()` pipeline and the abstention gate |
| `tests/test_agent_live.py` | One real API call, skipped without a key |

---

### Task 1: The rubric

Pure arithmetic, no dependencies. Build it first so everything downstream has the labels.

**Files:**
- Create: `src/visual_verify/agent/__init__.py`
- Create: `src/visual_verify/agent/rubric.py`
- Create: `tests/test_rubric.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rubric.py`:

```python
"""The four-label rubric and the score an abstention threshold acts on."""

import pytest

from visual_verify.agent.rubric import LABELS, Label, abstention_score


def test_all_four_proposal_labels_exist():
    """Fixed by proposal.tex line 377. Renaming one breaks the report."""
    assert set(LABELS) == {
        "supported",
        "partially_supported",
        "unsupported",
        "insufficient_evidence",
    }


def test_labels_rank_supported_highest_and_unsupported_lowest():
    ranked = sorted(LABELS, key=lambda label: abstention_score(label, 0.0))
    assert ranked[0] == "unsupported"
    assert ranked[-1] == "supported"


def test_confidence_orders_within_a_label_but_never_across_one():
    """The label decides; confidence only breaks ties inside it.

    A confident 'partially supported' must never outrank a hesitant
    'supported', or a self-reported number would be overriding the rubric.
    """
    confident_partial = abstention_score("partially_supported", 1.0)
    hesitant_supported = abstention_score("supported", 0.0)

    assert confident_partial < hesitant_supported


def test_confidence_does_order_within_a_label():
    assert abstention_score("supported", 0.9) > abstention_score("supported", 0.1)


def test_score_rejects_an_out_of_range_confidence():
    with pytest.raises(ValueError, match="0 and 1"):
        abstention_score("supported", 1.5)


def test_score_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="unknown label"):
        abstention_score("looks_fine", 0.5)


def test_label_is_a_plain_string_type():
    """Label must stay JSON-serializable for the cache and the eval output."""
    assert Label.__args__ == (
        "supported",
        "partially_supported",
        "insufficient_evidence",
        "unsupported",
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_rubric.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'visual_verify.agent'`

- [ ] **Step 3: Create the package**

Create `src/visual_verify/agent/__init__.py`:

```python
"""Reader, verifier, and abstention: pillars 2 and 3.

The reader and the verifier are deliberately DIFFERENT hosted models. A model
grading its own output is biased toward it (proposal.tex line 377), so the
separation is the point, not an implementation detail.

Everything here takes a StructuredChat as an argument and never constructs one.
LangChain is imported in exactly one file, models.py, which is what keeps the
whole pipeline testable with no network and no API key.

Task 10 adds the public re-exports here. Nothing else belongs in this file.
"""
```

- [ ] **Step 4: Write the implementation**

Create `src/visual_verify/agent/rubric.py`:

```python
"""The four-label rubric and the score an abstention threshold acts on.

Labels are fixed by proposal.tex line 377 and are part of the deliverable.
Do not rename, reorder the public tuple, or add a fifth.

WHY THE SCORE IS label_rank + confidence, and not either alone:

The label decides whether a claim is shown. Confidence only orders claims
WITHIN a label, so a confident "partially supported" can never outrank a
hesitant "supported": a self-reported number must not override the rubric.

Confidence is there because S7's headline metric is confident-wrong against
coverage, and a curve swept over four labels has four operating points. The
fractional part gives the curve resolution without giving it authority.

Self-reported confidence is NOT calibrated. The report must say so. Conformal
calibration is named future work in proposal.tex line 381.
"""

from typing import Literal

Label = Literal[
    "supported",
    "partially_supported",
    "insufficient_evidence",
    "unsupported",
]

# Ranks are spaced by 1 and confidence is bounded to [0, 1], so a label's band
# can never reach the next label's floor. That is what makes the ordering above
# a guarantee rather than a tendency.
_RANK: dict[str, int] = {
    "supported": 3,
    "partially_supported": 2,
    "insufficient_evidence": 1,
    "unsupported": 0,
}

LABELS: tuple[str, ...] = tuple(_RANK)


def abstention_score(label: str, confidence: float) -> float:
    """Rank the claim for the abstention threshold. Higher means more supported."""
    if label not in _RANK:
        raise ValueError(f"unknown label {label!r}; expected one of {sorted(_RANK)}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    return _RANK[label] + confidence
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_rubric.py -q`
Expected: PASS, 7 passed

- [ ] **Step 6: Commit**

```bash
git add src/visual_verify/agent/ tests/test_rubric.py
git commit -m "feat(agent): add the four-label rubric and its abstention score

Ranks are spaced by 1 and confidence is bounded to [0, 1], so a label's band
cannot reach the next label's floor. That makes 'the label decides, confidence
only orders within it' a guarantee rather than a tendency, which matters
because the confidence is self-reported and uncalibrated."
```

---

### Task 2: Response schemas and the `Claim.label` field

**Files:**
- Create: `src/visual_verify/agent/schemas.py`
- Modify: `src/visual_verify/contracts.py`
- Create: `tests/test_agent_schemas.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_schemas.py`:

```python
"""Schemas the models must return, and the contract field that carries a verdict."""

import pytest
from pydantic import ValidationError

from visual_verify.agent.schemas import ClaimList, Verdict
from visual_verify.contracts import Claim


def test_claim_list_holds_atomic_claims():
    parsed = ClaimList(claims=["Revenue grew 42 percent.", "Margins held steady."])
    assert len(parsed.claims) == 2


def test_claim_list_rejects_an_empty_string_claim():
    """An empty claim would ground to nothing and verify as insufficient,
    consuming an API call to learn the model returned junk."""
    with pytest.raises(ValidationError):
        ClaimList(claims=["Revenue grew.", "   "])


def test_verdict_requires_a_known_label():
    with pytest.raises(ValidationError):
        Verdict(label="looks_fine", confidence=0.5, reason="x")


def test_verdict_bounds_confidence():
    with pytest.raises(ValidationError):
        Verdict(label="supported", confidence=1.5, reason="x")


def test_verdict_requires_a_reason():
    """The reason is what makes a wrong verdict debuggable after the fact,
    and it goes in the eval output. An empty one is a silent verdict."""
    with pytest.raises(ValidationError):
        Verdict(label="supported", confidence=0.9, reason="")


def test_claim_carries_an_optional_label():
    """None until the verifier has run, so an unverified claim is
    distinguishable from one judged unsupported."""
    unverified = Claim(text="x", confidence=0.0)
    assert unverified.label is None

    judged = Claim(text="x", confidence=0.9, label="supported")
    assert judged.label == "supported"


def test_claim_rejects_an_unknown_label():
    with pytest.raises(ValidationError):
        Claim(text="x", confidence=0.9, label="looks_fine")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_agent_schemas.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'visual_verify.agent.schemas'`

- [ ] **Step 3: Write the schemas**

Create `src/visual_verify/agent/schemas.py`:

```python
"""What the models are required to return.

Schema-validated on the way in, so a malformed response raises instead of
parsing into something plausible. That matters more here than usual: a silently
mis-parsed claim list is exactly the correctly-shaped wrong output this
repository keeps getting caught by.
"""

from pydantic import BaseModel, Field, field_validator

from visual_verify.agent.rubric import Label


class ClaimList(BaseModel):
    """The reader's output: atomic claims, one assertion each."""

    claims: list[str] = Field(default_factory=list)

    @field_validator("claims")
    @classmethod
    def _no_blank_claims(cls, v: list[str]) -> list[str]:
        blank = [i for i, c in enumerate(v) if not c.strip()]
        if blank:
            raise ValueError(f"claims at {blank} are blank; the reader returned junk")
        return v


class Verdict(BaseModel):
    """The verifier's output for one claim."""

    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    # Not decoration. A verdict with no stated reason cannot be debugged after
    # the fact, and this string goes into the eval output.
    reason: str = Field(min_length=1)
```

- [ ] **Step 4: Add the contract field**

In `src/visual_verify/contracts.py`, add to `Claim`, after `abstained`:

```python
    # The verifier's rubric label, None until the verifier has run. Optional so
    # existing consumers are unaffected; see this file's docstring. Kept as the
    # label rather than reduced to `confidence` alone because the label is what
    # decides show-or-abstain, and the UI and the eval both need to say WHICH
    # verdict a claim received, not merely how strong it was.
    label: Literal[
        "supported", "partially_supported", "insufficient_evidence", "unsupported"
    ] | None = None
```

`Literal` is already imported in that file.

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_agent_schemas.py tests/test_contracts.py -q`
Expected: PASS. Report both counts.

- [ ] **Step 6: Commit**

```bash
git add src/visual_verify/agent/schemas.py src/visual_verify/contracts.py tests/test_agent_schemas.py
git commit -m "feat(agent): add reader and verifier schemas, and Claim.label

The label rides on Claim rather than being reduced to a confidence float,
for the same reason resolution rides on GroundedRegion: the label is what
decides show-or-abstain, and both the UI and the eval need to say which
verdict a claim received, not only how strong it was."
```

---

### Task 3: The `StructuredChat` seam and `FakeChat`

This task is what makes every later task testable offline. Get the protocol right.

**Files:**
- Create: `src/visual_verify/agent/types.py`
- Create: `tests/test_agent_types.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_types.py`:

```python
"""The seam that keeps LangChain out of every module except models.py."""

import pytest

from visual_verify.agent.schemas import ClaimList, Verdict
from visual_verify.agent.types import FakeChat


def test_fake_chat_returns_the_scripted_response():
    chat = FakeChat("fake-reader", [ClaimList(claims=["a", "b"])])
    out = chat.structured("prompt", None, ClaimList)
    assert out.claims == ["a", "b"]


def test_fake_chat_returns_scripted_responses_in_order():
    chat = FakeChat(
        "fake",
        [
            Verdict(label="supported", confidence=0.9, reason="r1"),
            Verdict(label="unsupported", confidence=0.8, reason="r2"),
        ],
    )
    assert chat.structured("p", None, Verdict).label == "supported"
    assert chat.structured("p", None, Verdict).label == "unsupported"


def test_fake_chat_raises_when_the_script_runs_out():
    """Silently repeating the last response would make a test that calls the
    model more times than expected still pass."""
    chat = FakeChat("fake", [ClaimList(claims=["a"])])
    chat.structured("p", None, ClaimList)
    with pytest.raises(AssertionError, match="script exhausted"):
        chat.structured("p", None, ClaimList)


def test_fake_chat_records_what_it_was_asked():
    """Lets a test assert the claim and the regions actually reached the model,
    rather than only that a call happened."""
    chat = FakeChat("fake", [ClaimList(claims=["a"])])
    chat.structured("the prompt text", None, ClaimList)
    assert chat.calls[0].prompt == "the prompt text"


def test_fake_chat_reports_a_model_id():
    """The cache keys on it, and the different-models test asserts on it."""
    assert FakeChat("fake-verifier", []).model_id == "fake-verifier"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_agent_types.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'visual_verify.agent.types'`

- [ ] **Step 3: Write the implementation**

Create `src/visual_verify/agent/types.py`:

```python
"""The narrow seam every agent module talks through.

Deliberately smaller than LangChain's surface. Modules depend on THIS, not on
LangChain, which is what lets the whole pipeline run in tests with no network,
no API key, and no heavy import. Same pattern as retrieval.types.Embedder and
FakeEmbedder, for the same reason.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

S = TypeVar("S", bound=BaseModel)


class StructuredChat(Protocol):
    """A chat model that returns schema-validated output."""

    @property
    def model_id(self) -> str:
        """Provider-qualified, e.g. 'openai:gpt-4o'. The cache keys on this."""
        ...

    def structured(self, prompt: str, image_path: Path | None, schema: type[S]) -> S:
        """One turn. Raises if the response does not satisfy `schema`."""
        ...


@dataclass(frozen=True)
class RecordedCall:
    prompt: str
    image_path: Path | None


@dataclass
class FakeChat:
    """Scripted stand-in. No network, no key, no LangChain import.

    Responses are returned in order and the script must not be over-consumed:
    repeating the last response would let a test that calls the model more
    times than it expects still pass, which hides a duplicated API call.
    """

    _model_id: str
    responses: list[BaseModel]
    calls: list[RecordedCall] = field(default_factory=list)
    _next: int = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    def structured(self, prompt: str, image_path: Path | None, schema: type[S]) -> S:
        self.calls.append(RecordedCall(prompt=prompt, image_path=image_path))
        assert self._next < len(self.responses), (
            f"script exhausted after {self._next} call(s); the code under test "
            "called the model more times than the test scripted"
        )
        out = self.responses[self._next]
        self._next += 1
        assert isinstance(out, schema), (
            f"scripted response {self._next - 1} is {type(out).__name__}, "
            f"but the caller asked for {schema.__name__}"
        )
        return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_agent_types.py -q`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/agent/types.py tests/test_agent_types.py
git commit -m "feat(agent): add the StructuredChat seam and a scripted fake

Modules depend on this protocol rather than on LangChain, so the whole
pipeline runs in tests with no network, no key, and no heavy import. The fake
refuses to over-consume its script, because repeating the last response would
let a test that triggers a duplicate API call still pass."
```

---

### Task 4: The response cache

**Files:**
- Create: `src/visual_verify/agent/cache.py`
- Create: `tests/test_agent_cache.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_cache.py`:

```python
"""Content-addressed response cache: offline demo, and the reproducibility record."""

from visual_verify.agent.cache import CachedChat
from visual_verify.agent.schemas import ClaimList
from visual_verify.agent.types import FakeChat


def test_a_repeated_call_hits_the_cache_instead_of_the_model(tmp_path):
    inner = FakeChat("m1", [ClaimList(claims=["a"])])
    chat = CachedChat(inner, tmp_path)

    first = chat.structured("p", None, ClaimList)
    second = chat.structured("p", None, ClaimList)

    assert first.claims == second.claims == ["a"]
    assert len(inner.calls) == 1, "the second call should not have reached the model"


def test_a_different_prompt_misses(tmp_path):
    inner = FakeChat("m1", [ClaimList(claims=["a"]), ClaimList(claims=["b"])])
    chat = CachedChat(inner, tmp_path)

    assert chat.structured("p1", None, ClaimList).claims == ["a"]
    assert chat.structured("p2", None, ClaimList).claims == ["b"]


def test_a_different_model_id_misses(tmp_path):
    """Otherwise switching provider silently returns the other model's answer,
    which would make an A/B comparison compare a model against itself."""
    a = FakeChat("openai:gpt-4o", [ClaimList(claims=["from-a"])])
    b = FakeChat("google:gemini", [ClaimList(claims=["from-b"])])

    assert CachedChat(a, tmp_path).structured("p", None, ClaimList).claims == ["from-a"]
    assert CachedChat(b, tmp_path).structured("p", None, ClaimList).claims == ["from-b"]


def test_the_cache_survives_a_new_process(tmp_path):
    """The point of writing to disk: the defense demo runs from a cache built
    on a different day, with no network."""
    inner = FakeChat("m1", [ClaimList(claims=["a"])])
    CachedChat(inner, tmp_path).structured("p", None, ClaimList)

    cold = FakeChat("m1", [])  # empty script: any model call now fails loudly
    assert CachedChat(cold, tmp_path).structured("p", None, ClaimList).claims == ["a"]


def test_a_different_image_misses(tmp_path):
    """Same question, different page, must not reuse the answer."""
    one = tmp_path / "one.png"
    two = tmp_path / "two.png"
    one.write_bytes(b"page-one-bytes")
    two.write_bytes(b"page-two-bytes")

    inner = FakeChat("m1", [ClaimList(claims=["a"]), ClaimList(claims=["b"])])
    chat = CachedChat(inner, tmp_path / "cache")

    assert chat.structured("p", one, ClaimList).claims == ["a"]
    assert chat.structured("p", two, ClaimList).claims == ["b"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_agent_cache.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'visual_verify.agent.cache'`

- [ ] **Step 3: Write the implementation**

Create `src/visual_verify/agent/cache.py`:

```python
"""Content-addressed cache over any StructuredChat.

Three jobs, only one of which is speed:

1. The defense demo runs offline. Pre-run the questions and the room needs no
   network.
2. It is the reproducibility record. Hosted models drift, so a cached raw
   response is the only evidence that a number reported in March was real.
3. Re-running the eval after a code change is free when the prompts did not
   change.

The key includes the MODEL ID on purpose. Without it, switching provider
silently returns the other model's answer, which would make an A/B comparison
compare a model against itself and report a difference of zero.

The key is built from the rendered prompt text, the image bytes, and the schema
name: never from LangChain objects. LangChain's representations are free to
change across versions, and keying on them would silently invalidate every
entry while still appearing to hit.
"""

import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from visual_verify.agent.types import StructuredChat

S = TypeVar("S", bound=BaseModel)


def _digest(model_id: str, prompt: str, image_path: Path | None, schema_name: str) -> str:
    h = hashlib.sha256()
    for part in (model_id, schema_name, prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    if image_path is not None:
        h.update(hashlib.sha256(Path(image_path).read_bytes()).hexdigest().encode())
    return h.hexdigest()


class CachedChat:
    """Wraps a StructuredChat. Same protocol, so it is a drop-in."""

    def __init__(self, inner: StructuredChat, cache_dir: Path) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    def structured(self, prompt: str, image_path: Path | None, schema: type[S]) -> S:
        key = _digest(self.inner.model_id, prompt, image_path, schema.__name__)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return schema.model_validate(json.loads(path.read_text()))
        out = self.inner.structured(prompt, image_path, schema)
        # Written after a successful call, so a failed request leaves no entry
        # that a later run would treat as a real answer.
        path.write_text(out.model_dump_json())
        return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_agent_cache.py -q`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/agent/cache.py tests/test_agent_cache.py
git commit -m "feat(agent): cache responses by model, prompt, and image

Not an optimisation. It is what lets the defense demo run with no network,
and it is the only record that makes a number reported against a drifting
hosted model reproducible later. The key includes the model id, so switching
provider misses rather than silently returning the other model's answer."
```

---

### Task 5: The LangChain-backed model, and the `agent` extra

The only file in the package that may import LangChain.

**Files:**
- Modify: `pyproject.toml`
- Create: `src/visual_verify/agent/models.py`
- Modify: `src/visual_verify/config.py`
- Create: `tests/test_agent_models.py`

- [ ] **Step 1: Add the extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, after the `retrieval` block:

```toml
# Hosted reader and verifier. Deliberately NOT pinned like retrieval: these are
# API clients, not weights, so a version bump changes the request format at
# worst and cannot silently degrade model quality the way the colpali-engine
# and transformers combination did.
agent = [
    "langchain>=0.3",
    "langchain-openai>=0.2",
    "langchain-google-genai>=2.0",
]
```

- [ ] **Step 2: Add settings**

In `src/visual_verify/config.py`, add to the `Settings` dataclass fields:

```python
    reader_provider: str = "openai"
    reader_model: str = "gpt-4o"
    verifier_provider: str = "google"
    verifier_model: str = "gemini-2.0-flash"
    abstain_threshold: float = 3.0
```

and to `from_env()`:

```python
            reader_provider=os.getenv("VVRAG_READER_PROVIDER", "openai"),
            reader_model=os.getenv("VVRAG_READER_MODEL", "gpt-4o"),
            verifier_provider=os.getenv("VVRAG_VERIFIER_PROVIDER", "google"),
            verifier_model=os.getenv("VVRAG_VERIFIER_MODEL", "gemini-2.0-flash"),
            abstain_threshold=float(os.getenv("VVRAG_ABSTAIN_THRESHOLD", "3.0")),
```

and a property alongside `pages_dir`:

```python
    @property
    def agent_cache_dir(self) -> Path:
        return self.data_dir / "agent_cache"
```

The default threshold of 3.0 is the `supported` floor: with ranks spaced by 1
and confidence in [0, 1], only a `supported` claim can reach it.

- [ ] **Step 3: Write the failing test**

Create `tests/test_agent_models.py`:

```python
"""Client construction from environment, without importing LangChain."""

import pytest

from visual_verify.config import Settings


def test_settings_read_both_roles_from_env(monkeypatch):
    monkeypatch.setenv("VVRAG_READER_PROVIDER", "openai")
    monkeypatch.setenv("VVRAG_READER_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("VVRAG_VERIFIER_PROVIDER", "google")
    monkeypatch.setenv("VVRAG_VERIFIER_MODEL", "gemini-2.0-flash")

    s = Settings.from_env()

    assert s.reader_model == "gpt-4o-mini"
    assert s.verifier_model == "gemini-2.0-flash"


def test_the_default_threshold_is_the_supported_floor():
    """Ranks are spaced by 1 and confidence is in [0, 1], so 3.0 admits only
    'supported'. A lower default would show partially-supported claims by
    accident."""
    from visual_verify.agent.rubric import abstention_score

    s = Settings()
    assert abstention_score("supported", 0.0) >= s.abstain_threshold
    assert abstention_score("partially_supported", 1.0) < s.abstain_threshold


def test_an_unknown_provider_names_the_env_var(monkeypatch):
    from visual_verify.agent.models import UnknownProvider, make_chat

    monkeypatch.setenv("VVRAG_READER_PROVIDER", "anthropic-typo")
    with pytest.raises(UnknownProvider, match="VVRAG_READER_PROVIDER"):
        make_chat("reader", Settings.from_env())


def test_a_missing_api_key_names_the_variable(monkeypatch):
    """A key error must say WHICH variable is unset. This is the most common
    first-run failure and a bare KeyError from inside a client is useless."""
    from visual_verify.agent.models import MissingApiKey, make_chat

    monkeypatch.setenv("VVRAG_READER_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingApiKey, match="OPENAI_API_KEY"):
        make_chat("reader", Settings.from_env())
```

- [ ] **Step 4: Run to verify it fails**

Run: `uv run pytest tests/test_agent_models.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'visual_verify.agent.models'`

- [ ] **Step 5: Write the implementation**

Create `src/visual_verify/agent/models.py`:

```python
"""The ONLY file in this package that imports LangChain.

Everything else takes a StructuredChat. That boundary is enforced by a
subprocess test in tests/test_core_is_light.py, and it is what lets the reader,
the verifier, and the whole pipeline be tested with no network and no key.

Imports are function-local so that importing visual_verify.agent does not drag
LangChain in. The boundary test checks exactly that.
"""

import base64
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from visual_verify.config import Settings

S = TypeVar("S", bound=BaseModel)

# Environment variable each provider's client library reads for its key.
_KEY_VAR = {"openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}


class UnknownProvider(RuntimeError):
    """Provider string is not one this project supports."""


class MissingApiKey(RuntimeError):
    """The provider's key variable is unset."""


class LangChainChat:
    """StructuredChat backed by a LangChain chat model."""

    def __init__(self, provider: str, model: str) -> None:
        self._model_id = f"{provider}:{model}"
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(model=model, temperature=0)
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        else:  # pragma: no cover - guarded by make_chat
            raise UnknownProvider(provider)

    @property
    def model_id(self) -> str:
        return self._model_id

    def structured(self, prompt: str, image_path: Path | None, schema: type[S]) -> S:
        content: list[dict] = [{"type": "text", "text": prompt}]
        if image_path is not None:
            encoded = base64.b64encode(Path(image_path).read_bytes()).decode()
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            )
        # with_structured_output is why LangChain earns its weight here: one
        # call gives schema-validated output on both providers, so a malformed
        # response raises instead of parsing into something plausible.
        # with_retry covers a transient schema-invalid response, which spec
        # section 10 requires. It retries and then raises: it never coerces a
        # bad response into a valid-looking object, because a silently
        # mis-parsed claim list is the failure this whole layer exists to stop.
        chain = self._llm.with_structured_output(schema).with_retry(stop_after_attempt=3)
        return chain.invoke([{"role": "user", "content": content}])


def make_chat(role: str, settings: Settings) -> LangChainChat:
    """Build the reader's or the verifier's client. role is 'reader' or 'verifier'."""
    if role == "reader":
        provider, model = settings.reader_provider, settings.reader_model
    elif role == "verifier":
        provider, model = settings.verifier_provider, settings.verifier_model
    else:
        raise ValueError(f"role must be 'reader' or 'verifier', got {role!r}")

    if provider not in _KEY_VAR:
        raise UnknownProvider(
            f"VVRAG_{role.upper()}_PROVIDER is {provider!r}; "
            f"expected one of {sorted(_KEY_VAR)}"
        )
    key_var = _KEY_VAR[provider]
    if not os.getenv(key_var):
        raise MissingApiKey(
            f"{key_var} is not set, which the {role} needs to reach {provider}"
        )
    return LangChainChat(provider, model)
```

- [ ] **Step 6: Install and run**

```bash
uv sync --all-extras --group dev
uv run pytest tests/test_agent_models.py tests/test_config.py -q
```
Expected: PASS. Report both counts.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/visual_verify/agent/models.py src/visual_verify/config.py tests/test_agent_models.py
git commit -m "feat(agent): construct reader and verifier clients from env

The only file importing LangChain, with function-local imports so that
importing visual_verify.agent does not drag it in. A missing key names the
variable that is unset: it is the most common first-run failure and a bare
KeyError from inside a client tells the user nothing."
```

---

### Task 6: The reader

**Files:**
- Create: `src/visual_verify/agent/reader.py`
- Create: `tests/test_reader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reader.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_reader.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'visual_verify.agent.reader'`

- [ ] **Step 3: Write the implementation**

Create `src/visual_verify/agent/reader.py`:

```python
"""The reader: page image plus question, out come atomic claims.

Claims are emitted directly as structured output rather than as prose that a
second call splits. One API call instead of two, and the model that wrote the
answer is the one deciding where it separates.

There is no separate prose answer. The displayed answer is the claims joined,
so nothing can drift between what is shown and what is verified.
"""

import re
from pathlib import Path

from visual_verify.agent.schemas import ClaimList
from visual_verify.agent.types import StructuredChat

PROMPT = """You are reading one page of a document to answer a question.

Answer ONLY from what is visible on this page. If the page does not answer the
question, return an empty list of claims.

Break your answer into atomic claims. Each claim must assert exactly ONE thing,
because each claim will be matched to a single region of the page as its
evidence. A claim asserting two things cannot be evidenced by one region.

Question: {question}"""

# A clause-joining conjunction: " and " followed by something with its own verb.
# Deliberately conservative. "Revenue and margin both rose" is ONE assertion
# about two subjects, and flagging it would report a decomposition failure that
# did not happen.
_COMPOUND = re.compile(
    r"\b(?:and|but|while|whereas)\b\s+\w+\s+(?:is|are|was|were|has|have|had|grew|fell|rose|held|remained|increased|decreased)\b",
    re.IGNORECASE,
)


def is_compound(claim: str) -> bool:
    """Whether a claim appears to assert more than one thing.

    The schema cannot enforce atomicity, and the roadmap requires that a
    sentence asserting two things is not grounded to one region. Flagged, not
    rejected: dropping the claim would lose an answer, and the useful response
    is to surface it in the eval as a decomposition failure.
    """
    return bool(_COMPOUND.search(claim))


def read(chat: StructuredChat, image_path: Path, question: str) -> list[str]:
    """Atomic claims answering `question` from the page at `image_path`."""
    out = chat.structured(PROMPT.format(question=question), image_path, ClaimList)
    return list(out.claims)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_reader.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/agent/reader.py tests/test_reader.py
git commit -m "feat(agent): read a page into atomic claims

Claims come back as structured output rather than prose a second call splits,
so there is one API call and no separate answer string that could drift from
what gets verified. The compound-claim check is deliberately conservative: 'X
and Y both rose' is one assertion about two subjects, and flagging it would
report a decomposition failure that did not happen."
```

---

### Task 7: The verifier

**Files:**
- Create: `src/visual_verify/agent/verifier.py`
- Create: `tests/test_verifier.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_verifier.py`:

```python
"""Verdicts, and the property that a verifier which cannot say no is useless."""

from pathlib import Path

from visual_verify.agent.schemas import Verdict
from visual_verify.agent.types import FakeChat
from visual_verify.agent.verifier import verify
from visual_verify.contracts import GroundedRegion


def region(text="Revenue grew 42 percent"):
    return GroundedRegion(
        page=0, bbox=(0.1, 0.1, 0.5, 0.2), score=1.0, modality="text", text=text
    )


def test_verify_returns_the_models_verdict():
    chat = FakeChat("m", [Verdict(label="supported", confidence=0.9, reason="matches")])
    v = verify(chat, Path("p.png"), "Revenue grew 42 percent.", [region()])

    assert v.label == "supported"
    assert v.reason == "matches"


def test_the_verifier_can_return_unsupported():
    """The single most important property in this slice.

    A verifier stuck on 'supported' passes every other test in the suite and
    produces a system that looks like it works and verifies nothing.
    """
    chat = FakeChat(
        "m", [Verdict(label="unsupported", confidence=0.8, reason="region says otherwise")]
    )
    v = verify(chat, Path("p.png"), "Revenue fell.", [region()])

    assert v.label == "unsupported"


def test_the_claim_and_the_region_text_both_reach_the_model():
    """Otherwise the verifier judges a claim against nothing and its verdict
    is about the page in general, not about the evidence offered."""
    chat = FakeChat("m", [Verdict(label="supported", confidence=0.5, reason="r")])
    verify(chat, Path("p.png"), "Revenue grew 42 percent.", [region("Revenue grew 42 percent")])

    prompt = chat.calls[0].prompt
    assert "Revenue grew 42 percent." in prompt
    assert "Revenue grew 42 percent" in prompt


def test_a_claim_with_no_regions_is_still_verified():
    """It must NOT be routed around the verifier. 'insufficient_evidence' is a
    label the rubric already has, and skipping the call would discard exactly
    the signal the project is measuring."""
    chat = FakeChat(
        "m", [Verdict(label="insufficient_evidence", confidence=0.7, reason="no evidence")]
    )
    v = verify(chat, Path("p.png"), "Revenue grew.", [])

    assert len(chat.calls) == 1
    assert v.label == "insufficient_evidence"


def test_the_page_image_reaches_the_verifier():
    """The verifier judges against the page, not only against extracted text,
    because visual regions carry evidence no text layer holds."""
    chat = FakeChat("m", [Verdict(label="supported", confidence=0.5, reason="r")])
    verify(chat, Path("page7.png"), "c", [region()])

    assert chat.calls[0].image_path == Path("page7.png")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_verifier.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'visual_verify.agent.verifier'`

- [ ] **Step 3: Write the implementation**

Create `src/visual_verify/agent/verifier.py`:

```python
"""The verifier: a DIFFERENT model judges whether the evidence supports a claim.

Different by construction, not by convention. proposal.tex line 377 requires a
separate judge because a model grading its own output is biased toward it, and
the two roles are configured to different providers.

verify() takes data and a chat, never a client handle it built itself. The same
discipline that kept ground() free of Qdrant and a GPU, and it buys the same
thing: the whole rubric path is testable with no network and no key.
"""

from pathlib import Path

from visual_verify.agent.schemas import Verdict
from visual_verify.agent.types import StructuredChat
from visual_verify.contracts import GroundedRegion

PROMPT = """You are checking whether a claim is supported by specific evidence
from a document page. You did not write the claim. Judge it strictly.

Claim: {claim}

Evidence regions selected from the page:
{evidence}

Choose exactly one label:
- supported: the evidence clearly establishes the claim
- partially_supported: the evidence establishes part of the claim
- unsupported: the evidence contradicts the claim, or is about something else
- insufficient_evidence: there is not enough evidence to judge

Give a confidence between 0 and 1, and one sentence of reasoning."""

NO_EVIDENCE = "(no regions were found for this claim)"


def _render(regions: list[GroundedRegion]) -> str:
    if not regions:
        return NO_EVIDENCE
    lines = []
    for r in regions:
        x0, y0, x1, y1 = r.bbox
        where = f"page {r.page}, {r.modality}, box [{x0:.3f} {y0:.3f} {x1:.3f} {y1:.3f}]"
        lines.append(f"- {where}: {r.text or '(no text layer here)'}")
    return "\n".join(lines)


def verify(
    chat: StructuredChat, image_path: Path, claim: str, regions: list[GroundedRegion]
) -> Verdict:
    """Judge one claim against its regions.

    A claim with NO regions is still sent. insufficient_evidence is a label the
    rubric already has, and routing an ungrounded claim around the verifier
    would discard the signal the project exists to measure.
    """
    prompt = PROMPT.format(claim=claim, evidence=_render(regions))
    return chat.structured(prompt, image_path, Verdict)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_verifier.py -q`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/agent/verifier.py tests/test_verifier.py
git commit -m "feat(agent): judge a claim against its regions with a separate model

verify() takes a chat and data, never a client it built itself, so the rubric
path is testable with no network. A claim with no regions is still sent:
insufficient_evidence is a label the rubric already has, and routing an
ungrounded claim around the verifier would discard the signal being measured."
```

---

### Task 8: `answer()`, the pipeline

**Files:**
- Create: `src/visual_verify/agent/core.py`
- Modify: `src/visual_verify/agent/__init__.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent.py`:

```python
"""The pipeline, and the gate that actually withholds."""

from pathlib import Path

import pytest

from visual_verify.agent import AgentError, answer
from visual_verify.agent.schemas import ClaimList, Verdict
from visual_verify.agent.types import FakeChat
from visual_verify.contracts import GroundedRegion
from visual_verify.ingest.boxes import BoxRecord


def word(x0, y0, x1, y1, text, word_no=0):
    return BoxRecord(
        kind="word", x0=x0, y0=y0, x1=x1, y1=y1, text=text,
        block_no=0, line_no=0, word_no=word_no,
    )


def page_boxes():
    """Two lines, so a second claim can also ground through the TEXT path.

    Every claim here must be findable in the text layer. A claim that is not
    falls through to ground()'s visual path, which needs page vectors and a
    grid, and would raise GroundingError rather than exercising the pipeline.
    """
    first = ["Revenue", "grew", "42", "percent"]
    second = ["Margins", "held", "steady"]
    boxes = [
        word(0.1 + i * 0.15, 0.10, 0.22 + i * 0.15, 0.16, t, i) for i, t in enumerate(first)
    ]
    boxes += [
        BoxRecord(
            kind="word", x0=0.1 + i * 0.15, y0=0.30, x1=0.22 + i * 0.15, y1=0.36,
            text=t, block_no=0, line_no=1, word_no=i,
        )
        for i, t in enumerate(second)
    ]
    return boxes


def test_a_supported_claim_is_shown_with_its_regions():
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent"])])
    verifier = FakeChat("v", [Verdict(label="supported", confidence=0.9, reason="matches")])

    out = answer("What happened?", Path("p.png"), page_boxes(), page=0,
                 reader_chat=reader, verifier_chat=verifier)

    assert len(out.claims) == 1
    claim = out.claims[0]
    assert claim.abstained is False
    assert claim.label == "supported"
    assert len(claim.regions) == 1
    assert claim.regions[0].modality == "text"


def test_an_unsupported_claim_is_abstained_on():
    """The point of the project: a wrong answer with a confident box drawn on
    it is worse than no answer."""
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent"])])
    verifier = FakeChat("v", [Verdict(label="unsupported", confidence=0.9, reason="no")])

    out = answer("q", Path("p.png"), page_boxes(), page=0,
                 reader_chat=reader, verifier_chat=verifier)

    assert out.claims[0].abstained is True
    assert out.claims[0].label == "unsupported"


def test_a_partially_supported_claim_is_abstained_on_at_the_default_threshold():
    """Even at confidence 1.0. The label decides, not the number."""
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent"])])
    verifier = FakeChat("v", [Verdict(label="partially_supported", confidence=1.0, reason="half")])

    out = answer("q", Path("p.png"), page_boxes(), page=0,
                 reader_chat=reader, verifier_chat=verifier)

    assert out.claims[0].abstained is True


def test_lowering_the_threshold_admits_a_partially_supported_claim():
    """The threshold is a parameter because S7 sweeps it to build the
    confident-wrong against coverage curve."""
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent"])])
    verifier = FakeChat("v", [Verdict(label="partially_supported", confidence=0.5, reason="half")])

    out = answer("q", Path("p.png"), page_boxes(), page=0, threshold=2.0,
                 reader_chat=reader, verifier_chat=verifier)

    assert out.claims[0].abstained is False


def test_every_claim_reaching_the_verifier_gets_one_call():
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent", "Margins held steady"])])
    verifier = FakeChat("v", [
        Verdict(label="supported", confidence=0.9, reason="a"),
        Verdict(label="unsupported", confidence=0.9, reason="b"),
    ])

    answer("q", Path("p.png"), page_boxes(), page=0,
           reader_chat=reader, verifier_chat=verifier)

    assert len(verifier.calls) == 2


def test_a_reader_returning_nothing_abstains_overall():
    reader = FakeChat("r", [ClaimList(claims=[])])
    verifier = FakeChat("v", [])

    out = answer("q", Path("p.png"), page_boxes(), page=0,
                 reader_chat=reader, verifier_chat=verifier)

    assert out.claims == []
    assert out.abstained_overall is True
    assert len(verifier.calls) == 0, "nothing to verify, so no call should be made"


def test_all_claims_abstained_means_abstained_overall():
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent"])])
    verifier = FakeChat("v", [Verdict(label="unsupported", confidence=0.9, reason="no")])

    out = answer("q", Path("p.png"), page_boxes(), page=0,
                 reader_chat=reader, verifier_chat=verifier)

    assert out.abstained_overall is True


def test_the_same_model_for_both_roles_is_refused():
    """The separate-judge requirement is the reason this slice exists. A
    misconfiguration pointing both roles at one model would otherwise be
    invisible and would silently invalidate every verification."""
    same = FakeChat("openai:gpt-4o", [ClaimList(claims=["a"])])
    other = FakeChat("openai:gpt-4o", [Verdict(label="supported", confidence=0.9, reason="r")])

    with pytest.raises(AgentError, match="same model"):
        answer("q", Path("p.png"), page_boxes(), page=0,
               reader_chat=same, verifier_chat=other)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_agent.py -q`
Expected: FAIL, `ImportError: cannot import name 'answer'`

- [ ] **Step 3: Write the implementation**

Create `src/visual_verify/agent/core.py`:

```python
"""answer(): reader, then grounding, then a different model's judgement.

Order is fixed by proposal.tex lines 340 to 342: retrieve, read, ground,
verify. Grounding runs per claim BETWEEN the reader and the verifier, which is
what gives the text path an exact string to search for.

Nothing is streamed. Showing a claim before the verifier has judged it would
display exactly what the system exists to withhold, and retracting it visibly
is worse than a pause. See spec section 9.
"""

from pathlib import Path

import numpy as np

from visual_verify.agent.reader import read
from visual_verify.agent.rubric import abstention_score
from visual_verify.agent.types import StructuredChat
from visual_verify.agent.verifier import verify
from visual_verify.contracts import Answer, Claim
from visual_verify.grounding import ground
from visual_verify.ingest.boxes import BoxRecord
from visual_verify.retrieval.geometry import PatchGrid

# The 'supported' floor: ranks are spaced by 1 and confidence is in [0, 1], so
# only a supported claim reaches 3.0.
DEFAULT_THRESHOLD = 3.0


class AgentError(RuntimeError):
    """A configuration that would invalidate the verification."""


def answer(
    question: str,
    image_path: Path,
    boxes: list[BoxRecord],
    *,
    page: int,
    reader_chat: StructuredChat,
    verifier_chat: StructuredChat,
    threshold: float = DEFAULT_THRESHOLD,
    page_vectors: np.ndarray | None = None,
    query_vectors: np.ndarray | None = None,
    grid: PatchGrid | None = None,
) -> Answer:
    """Answer `question` from one page, with every claim judged before it shows.

    `threshold` is a parameter, not a constant, because S7 sweeps it to produce
    the confident-wrong against coverage curve. A hardcoded value would make the
    project's headline figure unproducible.
    """
    if reader_chat.model_id == verifier_chat.model_id:
        raise AgentError(
            f"reader and verifier are the same model ({reader_chat.model_id}); "
            "a model grading its own output is biased toward it, which is the "
            "reason this slice uses two providers"
        )

    claims: list[Claim] = []
    for text in read(reader_chat, image_path, question):
        regions = ground(
            text,
            boxes,
            page=page,
            page_vectors=page_vectors,
            query_vectors=query_vectors,
            grid=grid,
        )
        verdict = verify(verifier_chat, image_path, text, regions)
        score = abstention_score(verdict.label, verdict.confidence)
        claims.append(
            Claim(
                text=text,
                regions=regions,
                confidence=verdict.confidence,
                label=verdict.label,
                abstained=score < threshold,
            )
        )

    return Answer(
        question=question,
        claims=claims,
        abstained_overall=not claims or all(c.abstained for c in claims),
    )
```

Then append to `src/visual_verify/agent/__init__.py`:

```python
from visual_verify.agent.core import AgentError, answer

__all__ = ["AgentError", "answer"]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_agent.py -q`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/agent/ tests/test_agent.py
git commit -m "feat(agent): add answer(), reader then grounding then verdict

The threshold is a parameter rather than a constant because S7 sweeps it to
produce the confident-wrong against coverage curve; hardcoding it would make
the project's headline figure unproducible.

Pointing both roles at one model raises. A model grading its own output is
biased toward it, so that misconfiguration would silently invalidate every
verification while every test still passed."
```

---

### Task 9: `vvrag ask`

**Files:**
- Modify: `src/visual_verify/cli.py`
- Modify: `tests/test_cli_retrieval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_retrieval.py`:

```python
def test_ask_command_is_registered():
    from visual_verify.cli import build_parser

    args = build_parser().parse_args(["ask", "what is X?", "--doc", "abc", "--page", "2"])

    assert args.question == "what is X?"
    assert args.doc == "abc"
    assert args.page == 2
    assert args.threshold == 3.0


def test_ask_command_accepts_a_threshold():
    from visual_verify.cli import build_parser

    args = build_parser().parse_args(
        ["ask", "q", "--doc", "abc", "--page", "1", "--threshold", "2.0"]
    )

    assert args.threshold == 2.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cli_retrieval.py -q -k ask`
Expected: FAIL, `SystemExit: 2`

- [ ] **Step 3: Implement**

Add to `src/visual_verify/cli.py` after `cmd_ground`:

```python
def cmd_ask(args: argparse.Namespace) -> int:
    """Answer a question from one page, with every claim verified before it shows."""
    from visual_verify.agent import AgentError, answer
    from visual_verify.agent.cache import CachedChat
    from visual_verify.agent.models import MissingApiKey, UnknownProvider, make_chat

    settings = Settings.from_env()
    with _session(settings) as session:
        found = _resolve_document(session, args.doc)
        if found is None or isinstance(found, list):
            print(f"no unique document matching {args.doc!r}")
            return 1
        doc = found
        page = session.scalar(
            select(Page).where(Page.doc_sha == doc.sha256, Page.page_no == args.page)
        )
        if page is None:
            print(f"no page {args.page} in {Path(doc.path).name}")
            return 1
        boxes = [
            _to_record(b)
            for b in session.scalars(select(Box).where(Box.page_id == page.id, Box.kind == "word"))
        ]
        image_path = settings.pages_dir / page.image_path

    try:
        reader = CachedChat(make_chat("reader", settings), settings.agent_cache_dir)
        verifier = CachedChat(make_chat("verifier", settings), settings.agent_cache_dir)
    except (MissingApiKey, UnknownProvider) as exc:
        print(f"cannot build the models: {exc}")
        return 1

    try:
        result = answer(
            args.question,
            image_path,
            boxes,
            page=args.page,
            reader_chat=reader,
            verifier_chat=verifier,
            threshold=args.threshold,
        )
    except AgentError as exc:
        print(f"cannot answer: {exc}")
        return 1

    shown = [c for c in result.claims if not c.abstained]
    for c in result.claims:
        mark = "abstained" if c.abstained else "shown"
        print(f"[{mark:9}] {c.label:<22} {c.confidence:.2f}  {c.text}")
        for r in c.regions:
            x0, y0, x1, y1 = r.bbox
            print(f"              {r.modality:<6} [{x0:.3f} {y0:.3f} {x1:.3f} {y1:.3f}]")

    if result.abstained_overall:
        print("\nabstained: no claim on this page met the support threshold")
    else:
        print(f"\n{len(shown)} of {len(result.claims)} claim(s) shown")
    return 0
```

Register in `build_parser`, before `return parser`:

```python
    p_ask = sub.add_parser("ask", help="answer a question from a page, with verification")
    p_ask.add_argument("question")
    p_ask.add_argument("--doc", required=True, help="document sha256, prefix, or path substring")
    p_ask.add_argument("--page", type=int, required=True)
    p_ask.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="abstain below this score; 3.0 admits only fully supported claims",
    )
    p_ask.set_defaults(func=cmd_ask)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_cli_retrieval.py -q`
Expected: PASS. Report the count.

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/cli.py tests/test_cli_retrieval.py
git commit -m "feat(cli): add vvrag ask with per-claim verdicts

Prints every claim with its label and whether it was shown or abstained on,
so the abstention behaviour is visible rather than inferred from a shorter
answer. Both roles go through the cache, so re-asking a question costs
nothing and the demo runs offline."
```

---

### Task 10: The live smoke test

**Files:**
- Create: `tests/test_agent_live.py`

- [ ] **Step 1: Write the test**

Create `tests/test_agent_live.py`:

```python
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
    v = verify(chat, _a_page(), "This page is blank.", [])

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
    v = verify(chat, _a_page(), "This page is a photograph of a cat.", [])

    assert v.label != "supported"


def test_reader_output_parses_as_the_claim_schema():
    """Guards the structured-output path itself: if the provider stops
    honouring the schema, this fails rather than the parse silently
    producing an empty list."""
    from visual_verify.agent.models import make_chat

    chat = make_chat("reader", Settings.from_env())
    out = chat.structured("List two facts about this page.", _a_page(), ClaimList)

    assert isinstance(out, ClaimList)
```

- [ ] **Step 2: Run it**

```bash
set -a && . ./.env && set +a && uv run pytest tests/test_agent_live.py -q
```

Expected: 5 passed if keys are present, 5 skipped if not. **If it skips, report which guard skipped it.** A skip is not a pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_agent_live.py
git commit -m "test(agent): one live call per provider, skipped without keys

The fake covers behaviour; this covers what a fake cannot, that the request
format, the schema, and the model names are actually accepted by the live
APIs. Includes a claim the verifier must refuse, since a live verifier stuck
on 'supported' would pass every fake-based test in the suite."
```

---

### Task 11: Boundary, full suite, and documentation

**Files:**
- Modify: `tests/test_core_is_light.py`
- Modify: `docs/ROADMAP.md`
- Modify: `README.md`

- [ ] **Step 1: Extend the boundary guard**

In `tests/test_core_is_light.py`, extend `FORBIDDEN`:

```python
FORBIDDEN = [
    "sqlalchemy", "alembic", "qdrant_client", "fastapi", "torch", "transformers",
    "langchain", "langchain_openai", "langchain_google_genai",
]
```

and append:

```python
def test_agent_pulls_no_client_library_at_import():
    """Importing the agent package must not drag LangChain in.

    models.py imports it inside functions on purpose. If this fails, an import
    moved to module scope and the rest of the package stopped being testable
    without the extra installed.
    """
    loaded = _modules_after_importing("visual_verify.agent")
    leaked = [m for m in FORBIDDEN if m in loaded]
    assert not leaked, f"agent leaked heavy deps: {leaked}"
```

Run: `uv run pytest tests/test_core_is_light.py -q`
Expected: 3 passed. **If it fails, report BLOCKED with the leaked names.**

- [ ] **Step 2: Run the full suite, once**

```bash
find . -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
set -a && . ./.env && set +a && uv run pytest -q > /tmp/s5_suite.log 2>&1
tail -20 /tmp/s5_suite.log
```

**Redirect to a file.** Two agents on the previous slice had a long run moved to the background at the tool's timeout, could not read the output, and once left an orphan holding 2.6 GB on a 3.63 GB card.

Expect roughly 350 tests, about 13 minutes. Report the exact final line.

- [ ] **Step 3: Update the roadmap**

In `docs/ROADMAP.md`, change `## S5: Reader and verifier (not started)` to `(done)`, tick every S5 checkbox, and set the S5 row in the slice table to `Done`.

Replace the "Blocker, unresolved" paragraph with a note that it was resolved by hosting both models, that `proposal.tex` line 369 already specified this, and that local execution was rejected because fitting two VLMs needs 4-bit quantization, which S3 measured dropping known-item retrieval from 1.00 to 0.00 silently, and answer accuracy is the ablation's control variable.

- [ ] **Step 4: Update the README**

Add after the `vvrag ground` lines:

```bash
uv run vvrag ask "<question>" --doc <sha> --page <n>
uv run vvrag ask "<question>" --doc <sha> --page <n> --threshold 2.0   # admit partial support
```

and a short section naming the four env vars (`VVRAG_READER_PROVIDER`, `VVRAG_READER_MODEL`, `VVRAG_VERIFIER_PROVIDER`, `VVRAG_VERIFIER_MODEL`) plus `OPENAI_API_KEY` and `GOOGLE_API_KEY`. Match the surrounding formatting.

- [ ] **Step 5: Commit**

```bash
git add tests/test_core_is_light.py docs/ROADMAP.md README.md
git commit -m "test(agent): enforce the client boundary and close out S5

Importing visual_verify.agent must not pull LangChain in. models.py imports
it inside functions so the rest of the package stays testable without the
extra installed, and this guard is what keeps that true."
```

Do NOT `git add CLAUDE.md`.

---

## Definition of done

- [ ] Reader and verifier are different models, refused at runtime if not
- [ ] The verifier can return `unsupported`, pinned by a fake test and a live one
- [ ] The abstention gate actually withholds; a below-threshold claim is marked abstained
- [ ] The threshold is a parameter and lowering it admits partially-supported claims
- [ ] A claim with no regions is still verified, not routed around
- [ ] Compound claims are flagged rather than silently accepted
- [ ] The cache misses on a different model id
- [ ] The whole pipeline runs with no network and no key, via `FakeChat`
- [ ] `visual_verify.agent` imports no LangChain, enforced in a subprocess
- [ ] Full suite green, run once, output captured to a file
- [ ] No streaming of unverified claims
