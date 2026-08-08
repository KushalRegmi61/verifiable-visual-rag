# S5 Reader and Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `verify(question, reader, verifier, ...)`, which reads a retrieved page, splits the answer into atomic claims, grounds each claim with S4's `ground()`, judges each claim with a model **different from the reader**, and abstains on weak judgements.

**Architecture:** Four small pure modules under `src/visual_verify/verify/` (claims, rubric, evidence, core) and one I/O module (backends) that the CLI wires. The core takes reader and verifier objects as arguments and never constructs a model; the pure modules need nothing beyond pydantic, PIL, and numpy, so the whole pipeline is testable with fakes — no GPU, no network, no 20 s model load. The backends (HuggingFace local, HTTP hosted) are thin and lazy-import their heavy libraries.

**Tech Stack:** pydantic, PIL, numpy, existing `ground()` and `contracts.py` (`Answer`, `Claim`, `GroundedRegion`). No new dependencies: the local VLM backend reuses the `retrieval` extra's transformers/torch; the hosted backend uses stdlib urllib.

**Spec:** `docs/superpowers/specs/2026-08-08-s5-reader-verifier-design.md`. Read it before starting. Section 3 (measured constraints) is why the compute path is a wiring choice rather than a baked decision, and section 4 fixes the rules for making it.

**Status: executed and merged.** The checkboxes below are left unticked on
purpose, following S4's convention: this is the plan as written. Four deviations
from it are recorded here:

1. **The visual-path fetch in `cmd_ask` is eager, not lazy.** Task 6's text said
   "fetch vectors and grid exactly as cmd_ground does, but lazily." That is
   impossible as written: claims only exist after the reader runs, so ask cannot
   know in advance whether one will need the visual path, and `verify()`'s seam
   takes plain arrays, not a lazy provider. cmd_ask fetches eagerly and says so
   in its docstring; per-request model economy belongs to S6's load-once service.
2. **`query_vectors` was removed from `verify()` during planning.** The first
   draft of the signature carried it; it is a footgun (one pre-fetched array
   cannot serve claims that only exist after the reader runs), so the plan and
   the spec were edited before any code was written. `embed` is the only seam.
3. **`test_core_is_light.py` gained a third guard.** `test_verify_pulls_no_store_or_model_dependency`
   asserts `import visual_verify.verify` loads no torch/transformers/sqlalchemy,
   which is the literal encoding of this plan's "the core never builds a model"
   ground rule. The guard would catch a lazy import drifting to module level.

Task 7's measurement is closed. The planned candidate `Qwen/Qwen2.5-VL-2B-Instruct`
does not exist (the 2.5 family starts at 3B), so the default verifier became
`Qwen/Qwen2-VL-2B-Instruct`, which loads and judges on this card: fp16 at 4.2 GB
and ~16-20 s per judgement, nf4 at 1.5 GB. The measurement also required
`torch==2.6.0+cu124` from the pytorch index, because PyPI's Windows torch is
CPU-only. Spec section 3.1 records the numbers; the ROADMAP records the pairing.

**Spec:** `docs/superpowers/specs/2026-08-08-s5-reader-verifier-design.md`. Read it before starting. Section 3 (measured constraints) is why the compute path is a wiring choice rather than a baked decision, and section 4 fixes the rules for making it.

---

## Ground rules for every task

**Test only your own module.** Run the specific test file(s) your task names, never the full suite. The full suite runs once, in Task 8.

**The core never builds a model.** Any import of torch or transformers inside `verify/` outside `backends.py` fails review. `backends.py` may import lazily, inside methods, never at module top.

**The gate is a knob, not a policy.** `verify()` forwards `threshold`; nobody in the core hardcodes a second opinion about what "weak" means. The S7 ablation must be able to tune one number and see confident-wrong versus coverage move.

**A failed judgement is never an answer and never an abstain.** Transport failures and malformed model output raise `VerifierError` with the role named. Only a verifier's actual label can produce `abstained=True`.

**No model, no network, no disk in the core tests.** Every assertion in Tasks 1 through 5 runs with fakes and hand-built arrays. The live measurement lives in Task 7 and skips without the models, like `test_grounding_live.py`.

**Clear bytecode caches when mutation-testing.** `find . -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +`

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/visual_verify/verify/__init__.py` | Public surface: `verify`, `VerifierError` |
| `src/visual_verify/verify/errors.py` | `VerifierError`. Own module because claims, rubric, and core all raise it, and any one of them importing a sibling that imports it back is a cycle. |
| `src/visual_verify/verify/claims.py` | `ReaderOutput` shape and `parse_reader_output()`: JSON contract validation |
| `src/visual_verify/verify/rubric.py` | `LABELS`, `SUFFICIENCY`, `Judgement`, `sufficiency()`, `is_answered()` |
| `src/visual_verify/verify/evidence.py` | `Evidence` shape, `crop_region()`, `build_evidence()`, best-region selection |
| `src/visual_verify/verify/core.py` | `verify()` orchestration |
| `src/visual_verify/verify/backends.py` | `Reader`/`Verifier` protocols, local HF and hosted HTTP implementations, default-pairing wiring |
| `src/visual_verify/cli.py` | New `vvrag ask` subcommand and the fetching adapter |
| `tests/test_claims.py` | Parsing contract: valid, empty, malformed, wrong shape |
| `tests/test_rubric.py` | Label table, unknown label, threshold behaviour |
| `tests/test_verify_evidence.py` | Text vs visual evidence, crop geometry, best-region selection. Named for S5 because `tests/test_verify_evidence.py` already tests S4's evidence checkers. |
| `tests/test_verify.py` | Orchestration with fakes: order, force forwarding, gate, abstain decisions, Answer shape |
| `tests/test_backends.py` | Protocols enforced, pairing differs, hosted request shape |
| `tests/test_verify_live.py` | End-to-end ask against real corpus and models; skips without them |

---

### Task 1: Package skeleton, errors, and the claims contract

**Files:**
- Create: `src/visual_verify/verify/__init__.py`
- Create: `src/visual_verify/verify/errors.py`
- Create: `src/visual_verify/verify/claims.py`
- Create: `tests/test_claims.py`

**Step 1: Write the failing test** — `tests/test_claims.py`:

```python
"""The reader's JSON output contract: parse it or refuse to proceed.

A free-form answer is not an answer. If the reader's output cannot be
parsed into claims, raising is the only honest behaviour: silently
dropping claims would turn an uncheckable answer into a confident one.
"""

import pytest

from visual_verify.verify.claims import ReaderOutput, parse_reader_output
from visual_verify.verify.errors import VerifierError


def test_parse_valid_output():
    out = parse_reader_output('{"answer": "It is 42.", "claims": ["The answer is 42."]}')
    assert out == ReaderOutput(answer="It is 42.", claims=["The answer is 42."])


def test_parse_strips_whitespace():
    out = parse_reader_output('{"answer": "  It is 42.  ", "claims": ["  42.  "]}')
    assert out.answer == "It is 42."
    assert out.claims == ["42."]


def test_parse_empty_claims_is_valid_but_vacuous():
    out = parse_reader_output('{"answer": "No idea.", "claims": []}')
    assert out.claims == []


def test_parse_rejects_non_json():
    with pytest.raises(VerifierError):
        parse_reader_output("It is 42.")


def test_parse_rejects_json_with_wrong_shape():
    with pytest.raises(VerifierError):
        parse_reader_output("[1, 2, 3]")
    with pytest.raises(VerifierError):
        parse_reader_output('{"answer": 42, "claims": []}')
    with pytest.raises(VerifierError):
        parse_reader_output('{"answer": "", "claims": []}')
    with pytest.raises(VerifierError):
        parse_reader_output('{"answer": "x", "claims": "one claim"}')
    with pytest.raises(VerifierError):
        parse_reader_output('{"answer": "x", "claims": [""]}')


def test_parse_rejects_missing_fields():
    with pytest.raises(VerifierError):
        parse_reader_output('{"claims": []}')
```

**Step 2: Make it pass.**

`errors.py`:

```python
class VerifierError(RuntimeError):
    """A contract violation or transport failure in the read-verify pipeline.

    Own module on purpose. claims, rubric, and core all raise this, and
    putting it in core or __init__ makes one of them import a sibling that
    imports it back, a cycle that only works by definition order.
    """
```

`claims.py`:

```python
"""The reader's structured output: one answer, N atomic claims."""

import json

from pydantic import BaseModel

from visual_verify.verify.errors import VerifierError


class ReaderOutput(BaseModel):
    answer: str
    claims: list[str]


def parse_reader_output(raw: str) -> ReaderOutput:
    """Validate the reader's JSON contract, or raise VerifierError."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VerifierError("reader output is not valid JSON") from exc
    if not isinstance(data, dict):
        raise VerifierError("reader output must be a JSON object")
    answer = data.get("answer")
    claims = data.get("claims")
    if not isinstance(answer, str) or not answer.strip():
        raise VerifierError("reader output has no answer text")
    if not isinstance(claims, list) or not all(isinstance(c, str) and c.strip() for c in claims):
        raise VerifierError("reader output claims must be a list of non-empty strings")
    return ReaderOutput(answer=answer.strip(), claims=[c.strip() for c in claims])
```

`__init__.py`:

```python
"""Reader and verifier: pillar 3 of the project.

verify() answers a question from one page, splits the answer into atomic
claims, grounds each claim with S4's ground(), and has a model different
from the reader judge each claim. Weak judgements abstain: a wrong answer
with a confident box drawn on it is worse than no answer.

The core never constructs a model. Reader and verifier arrive as objects,
which is what keeps everything but backends.py testable with fakes.
"""

from visual_verify.verify.core import verify
from visual_verify.verify.errors import VerifierError

__all__ = ["verify", "VerifierError"]
```

**Step 3: Run** `uv run --python 3.13 pytest -q tests/test_claims.py`. All pass.

---

### Task 2: The rubric and the gate

**Files:**
- Create: `src/visual_verify/verify/rubric.py`
- Create: `tests/test_rubric.py`

**Step 1: Write the failing test** — `tests/test_rubric.py`:

```python
"""The four-label rubric and the threshold that acts on it.

The model emits a label; this module owns the mapping label -> number and
the gate number >= threshold. The gate is a knob: S7 tunes one threshold
and measures confident-wrong against coverage.
"""

import pytest

from pydantic import ValidationError

from visual_verify.verify.errors import VerifierError
from visual_verify.verify.rubric import LABELS, Judgement, is_answered, sufficiency


def test_sufficiency_maps_every_label():
    assert {sufficiency(l) for l in LABELS} == {1.0, 0.5, 0.0}


def test_supported_and_partial_pass_default_threshold():
    assert is_answered("supported", 0.5)
    assert is_answered("partial", 0.5)
    assert not is_answered("unsupported", 0.5)
    assert not is_answered("insufficient", 0.5)


def test_threshold_moves_the_gate():
    assert is_answered("partial", 0.4)
    assert not is_answered("partial", 0.6)


def test_unknown_label_raises():
    with pytest.raises(VerifierError):
        sufficiency("maybe")


def test_judgement_rejects_unknown_labels_at_construction():
    with pytest.raises(ValidationError):
        Judgement(label="maybe")
```

**Step 2: Make it pass.** `rubric.py`:

```python
"""The four-label rubric, the sufficiency mapping, and the abstention gate.

The verifier model emits one of LABELS. The number a threshold acts on is
computed HERE, from a table, and pinned by tests: the model never emits
numbers and the gate is auditable.
"""

from typing import Literal

from pydantic import BaseModel

from visual_verify.verify.errors import VerifierError

LABELS = ("supported", "partial", "unsupported", "insufficient")

SUFFICIENCY = {
    "supported": 1.0,
    "partial": 0.5,
    "unsupported": 0.0,
    "insufficient": 0.0,
}

RubricLabel = Literal["supported", "partial", "unsupported", "insufficient"]


class Judgement(BaseModel):
    label: RubricLabel


def sufficiency(label: str) -> float:
    if label not in SUFFICIENCY:
        raise VerifierError(f"unknown rubric label {label!r}")
    return SUFFICIENCY[label]


def is_answered(label: str, threshold: float) -> bool:
    return sufficiency(label) >= threshold
```

**Step 3: Run** `uv run --python 3.13 pytest -q tests/test_rubric.py`.

---

### Task 3: Evidence assembly

**Files:**
- Create: `src/visual_verify/verify/evidence.py`
- Create: `tests/test_verify_evidence.py`

**Step 1: Write the failing test** — `tests/test_verify_evidence.py`:

```python
"""What a claim is judged against: the region S4 actually returned.

Text regions are judged from their text; visual regions get the crop cut
from the page render. A claim is judged against its best region only.
"""

from dataclasses import FrozenInstanceError

import pytest
from PIL import Image

from visual_verify.contracts import GroundedRegion
from visual_verify.verify.evidence import Evidence, best_region, build_evidence, crop_region


def make_region(page=0, bbox=(0.1, 0.2, 0.4, 0.5), score=1.0, modality="text", text="x"):
    return GroundedRegion(page=page, bbox=bbox, score=score, modality=modality, text=text)


def test_text_region_is_judged_from_text_only():
    r = make_region()
    ev = build_evidence(r, None)
    assert ev.text == "x"
    assert ev.image is None


def test_visual_region_crops_the_page_image():
    r = make_region(bbox=(0.25, 0.0, 0.75, 0.5), modality="visual")
    img = Image.new("RGB", (400, 200))
    ev = build_evidence(r, img)
    assert ev.image is not None
    assert ev.image.size == (200, 100)


def test_visual_region_without_image_keeps_text():
    r = make_region(modality="visual", text="chart label")
    ev = build_evidence(r, None)
    assert ev.text == "chart label"
    assert ev.image is None


def test_best_region_is_highest_score_ties_to_first():
    regions = [
        make_region(bbox=(0.0, 0.0, 0.1, 0.1), score=0.4, text="low"),
        make_region(bbox=(0.1, 0.0, 0.2, 0.1), score=0.9, text="high"),
        make_region(bbox=(0.2, 0.0, 0.3, 0.1), score=0.9, text="high tie"),
    ]
    assert best_region(regions).text == "high"


def test_best_region_of_empty_is_none():
    assert best_region([]) is None
```

**Step 2: Make it pass.** `evidence.py`:

```python
"""Evidence assembly: the region S4 returned, shaped for the verifier."""

from dataclasses import dataclass

from PIL import Image

from visual_verify.contracts import GroundedRegion


@dataclass(frozen=True)
class Evidence:
    text: str | None = None
    image: Image.Image | None = None


def crop_region(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    """Cut the normalized 0-1 bbox out of the page render."""
    w, h = image.size
    x0, y0, x1, y1 = bbox
    return image.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


def build_evidence(region: GroundedRegion, image: Image.Image | None) -> Evidence:
    """Text regions need no pixels; visual regions get the crop."""
    if region.modality == "text":
        return Evidence(text=region.text)
    return Evidence(
        text=region.text,
        image=crop_region(image, region.bbox) if image is not None else None,
    )


def best_region(regions: list[GroundedRegion]) -> GroundedRegion | None:
    """The region the verifier judges against: highest score, ties to first.

    Judging against every region would let a weak claim pass on a stray
    match. All regions still travel on the returned Claim; only the
    judgement is per-claim.
    """
    if not regions:
        return None
    return max(regions, key=lambda r: r.score)
```

**Step 3: Run** `uv run --python 3.13 pytest -q tests/test_verify_evidence.py`.

---

### Task 4: Core orchestration

**Files:**
- Create: `src/visual_verify/verify/core.py`
- Create: `tests/test_verify.py`

**Step 1: Write the failing test** — `tests/test_verify.py`. The fakes are the point: the whole pipeline, including the force forwarding and the gate, is asserted without any model.

```python
"""verify() orchestration: order, force forwarding, the gate, abstention."""

import numpy as np
import pytest
from PIL import Image

from visual_verify.contracts import Answer, Claim, GroundedRegion
from visual_verify.verify.backends import Reader, Verifier
from visual_verify.verify.claims import ReaderOutput
from visual_verify.verify.core import verify
from visual_verify.verify.errors import VerifierError
from visual_verify.verify.evidence import Evidence
from visual_verify.verify.rubric import Judgement


class FakeReader:
    def __init__(self, output: ReaderOutput):
        self.output = output
        self.seen: list[tuple] = []

    def read(self, question, image, text_layer):
        self.seen.append((question, text_layer))
        return self.output


class FakeVerifier:
    def __init__(self, label: str = "supported"):
        self.label = label
        self.seen: list[tuple[str, Evidence]] = []

    def judge(self, claim: str, evidence: Evidence) -> Judgement:
        self.seen.append((claim, evidence))
        return Judgement(label=self.label)


def make_page_image():
    img = Image.new("RGB", (100, 100))
    for y in range(10, 30):
        for x in range(20, 60):
            img.putpixel((x, y), (255, 0, 0))
    return img


def test_verify_reads_grounds_judges_and_returns_answer_shape():
    reader = FakeReader(ReaderOutput(answer="42.", claims=["The answer is 42."]))
    verifier = FakeVerifier("supported")
    ans = verify(
        "What is the answer?",
        reader,
        verifier,
        page=3,
        image=make_page_image(),
        text_layer="The answer is 42.",
        boxes=[],
    )
    assert isinstance(ans, Answer)
    assert len(ans.claims) == 1
    c = ans.claims[0]
    assert c.text == "The answer is 42."
    assert c.confidence == 1.0
    assert not c.abstained
    assert not ans.abstained_overall


def test_weak_labels_abstain_the_claim():
    ans = verify(
        "Q", FakeReader(ReaderOutput(answer="a", claims=["c1", "c2"])), FakeVerifier("unsupported"),
        page=0, image=make_page_image(), text_layer="", boxes=[], threshold=0.5,
    )
    assert [c.abstained for c in ans.claims] == [True, True]
    assert ans.abstained_overall
    assert [c.confidence for c in ans.claims] == [0.0, 0.0]


def test_partial_below_threshold_abstains_partial_above_passes():
    reader = FakeReader(ReaderOutput(answer="a", claims=["c"]))
    ans = verify("Q", reader, FakeVerifier("partial"), page=0, image=make_page_image(),
                 text_layer="", boxes=[], threshold=0.6)
    assert ans.claims[0].abstained
    ans2 = verify("Q", reader, FakeVerifier("partial"), page=0, image=make_page_image(),
                  text_layer="", boxes=[], threshold=0.5)
    assert not ans2.claims[0].abstained


def test_zero_claims_abstains_the_whole_answer():
    ans = verify("Q", FakeReader(ReaderOutput(answer="No idea.", claims=[])), FakeVerifier(),
                 page=0, image=make_page_image(), text_layer="", boxes=[])
    assert ans.abstained_overall
    assert ans.claims == []


def test_reader_failure_raises_with_role_named():
    class BrokenReader:
        def read(self, question, image, text_layer):
            raise OSError("no network")

    with pytest.raises(VerifierError, match="reader"):
        verify("Q", BrokenReader(), FakeVerifier(), page=0, image=make_page_image(),
               text_layer="", boxes=[])


def test_verifier_failure_raises_with_role_named():
    class BrokenVerifier:
        def judge(self, claim, evidence):
            raise OSError("oom")

    with pytest.raises(VerifierError, match="verifier"):
        verify("Q", FakeReader(ReaderOutput(answer="a", claims=["c"])), BrokenVerifier(),
               page=0, image=make_page_image(), text_layer="", boxes=[])


def test_visual_path_requires_embed_and_vectors():
    reader = FakeReader(ReaderOutput(answer="a", claims=["c"]))

    class TrackingEmbed:
        def __init__(self):
            self.calls = []

        def __call__(self, claim):
            self.calls.append(claim)
            rng = np.random.default_rng(0)
            return rng.normal(size=(3, 8))

    with pytest.raises(VerifierError, match="embed"):
        verify("Q", reader, FakeVerifier(), page=0, image=make_page_image(),
               text_layer="", boxes=[make_box()], force="visual")


def test_force_visual_embeds_the_claim_and_uses_it():
    reader = FakeReader(ReaderOutput(answer="a", claims=["c"]))
    embed = TrackingEmbed()
    grid = make_grid()
    page_vecs = np.random.default_rng(1).normal(size=(grid.n_vectors, 8))
    ans = verify(
        "Q", reader, FakeVerifier(), page=0, image=make_page_image(),
        text_layer="", boxes=[make_box()], embed=embed, page_vectors=page_vecs,
        grid=grid, force="visual",
    )
    assert embed.calls == ["c"]
```

Helpers live at the bottom of the test file:

```python
def make_box():
    from visual_verify.ingest.boxes import BoxRecord
    return BoxRecord(page=0, bbox=(0.2, 0.3, 0.6, 0.6), text="c", kind="word")


def make_grid():
    from visual_verify.retrieval.geometry import PatchGrid
    return PatchGrid(n_x=4, n_y=3, offset=2, n_vectors=2 + 12 + 1)
```

Note the `ground()` text path needs `boxes` with a matching text span; `make_box`'s text is `"c"` and claims are `"c"`, so text grounding succeeds without vectors in the non-visual tests. In `test_visual_path_requires_embed_and_vectors` the claim text is `"c"` too, but `force="visual"` skips the text path, which is exactly what that test pins.

**Step 2: Make it pass.** `core.py`:

```python
"""verify(): the read-ground-judge-gate pipeline over one page.

Everything except the two model calls is pure and tested with fakes.
"""

from typing import Callable, Literal

import numpy as np
from PIL import Image

from visual_verify.contracts import Answer, Claim, GroundedRegion
from visual_verify.grounding import GroundingError, ground
from visual_verify.ingest.boxes import BoxRecord
from visual_verify.retrieval.geometry import PatchGrid
from visual_verify.verify.backends import Reader, Verifier
from visual_verify.verify.claims import ReaderOutput
from visual_verify.verify.errors import VerifierError
from visual_verify.verify.evidence import Evidence, best_region, build_evidence
from visual_verify.verify.rubric import is_answered, sufficiency


def verify(
    question: str,
    reader: Reader,
    verifier: Verifier,
    *,
    page: int,
    image: Image.Image | None,
    text_layer: str | None,
    boxes: list[BoxRecord],
    embed: Callable[[str], np.ndarray] | None = None,
    page_vectors: np.ndarray | None = None,
    grid: PatchGrid | None = None,
    force: Literal["text", "visual"] | None = None,
    threshold: float = 0.5,
) -> Answer:
    """Answer `question` from one page, per claim, with abstention."""
    output = _read(reader, question, image, text_layer)
    if not output.claims:
        return Answer(question=question, claims=[], abstained_overall=True)

    claims: list[Claim] = []
    for text in output.claims:
        regions = _ground_claim(
            text, boxes, page=page, embed=embed, page_vectors=page_vectors,
            grid=grid, force=force,
        )
        region = best_region(regions)
        evidence = build_evidence(region, image) if region is not None else Evidence()
        label = _judge(verifier, text, evidence)
        score = sufficiency(label)
        claims.append(
            Claim(text=text, regions=regions, confidence=score, abstained=not is_answered(label, threshold))
        )
    return Answer(question=question, claims=claims, abstained_overall=all(c.abstained for c in claims))


def _read(reader: Reader, question: str, image: Image.Image | None, text_layer: str | None) -> ReaderOutput:
    try:
        return reader.read(question, image, text_layer)
    except VerifierError:
        raise
    except Exception as exc:
        raise VerifierError(f"reader failed: {exc}") from exc


def _judge(verifier: Verifier, claim: str, evidence: Evidence) -> str:
    try:
        return verifier.judge(claim, evidence).label
    except VerifierError:
        raise
    except Exception as exc:
        raise VerifierError(f"verifier failed: {exc}") from exc


def _ground_claim(
    claim: str,
    boxes: list[BoxRecord],
    *,
    page: int,
    embed: Callable[[str], np.ndarray] | None,
    page_vectors: np.ndarray | None,
    grid: PatchGrid | None,
    force: Literal["text", "visual"] | None,
) -> list[GroundedRegion]:
    if force != "visual":
        try:
            regions = ground(claim, boxes, page=page)
        except GroundingError:
            regions = []
        if regions:
            return regions

    if embed is None or page_vectors is None or grid is None:
        raise VerifierError(
            "the visual path needs embed, page_vectors, and grid; "
            "without them this claim cannot be grounded"
        )
    return ground(
        claim, boxes, page=page, page_vectors=page_vectors,
        query_vectors=embed(claim), grid=grid, force="visual",
    )
```

There is deliberately no `query_vectors` parameter: claims only exist after the reader
runs, so one pre-fetched vector array cannot serve all claims, and reusing one would
ground every claim against the first claim's query. `embed` is the only seam.

**Step 3: Run** `uv run --python 3.13 pytest -q tests/test_verify.py`.

---

### Task 5: Backends — protocols, hosted HTTP, local HF, and the pairing rule

**Files:**
- Create: `src/visual_verify/verify/backends.py`
- Create: `tests/test_backends.py`

**Step 1: Write the failing test** — `tests/test_backends.py`:

```python
"""The model seam: protocols, hosted request shape, and the pairing rule.

The independence rule (spec 3.2) is enforced here, in wiring, because it
cannot be enforced in the core: reader and verifier arrive as objects and
the core has no way to know their model families.
"""

import json

import pytest

from visual_verify.verify.backends import (
    DEFAULT_READER_MODEL,
    DEFAULT_VERIFIER_MODEL,
    HostedAPIReader,
    HostedAPIVerifier,
)
from visual_verify.verify.claims import ReaderOutput
from visual_verify.verify.evidence import Evidence
from visual_verify.verify.rubric import Judgement


def test_default_pairing_satisfies_the_independence_rule():
    assert DEFAULT_READER_MODEL != DEFAULT_VERIFIER_MODEL


def test_hosted_reader_builds_a_json_request(monkeypatch):
    payload = {}

    def fake_post(url, body):
        payload["url"] = url
        payload["body"] = json.loads(body)
        return json.dumps({"answer": "42.", "claims": ["42."]})

    monkeypatch.setattr("visual_verify.verify.backends._post", fake_post)
    reader = HostedAPIReader(url="https://example.test/read", key="k")
    out = reader.read("Q?", None, None)
    assert isinstance(out, ReaderOutput)
    assert out.claims == ["42."]
    assert payload["url"] == "https://example.test/read"
    assert payload["body"]["question"] == "Q?"
    assert payload["body"]["model"] == DEFAULT_READER_MODEL


def test_hosted_verifier_parses_the_label(monkeypatch):
    def fake_post(url, body):
        return json.dumps({"label": "supported"})

    monkeypatch.setattr("visual_verify.verify.backends._post", fake_post)
    verifier = HostedAPIVerifier(url="https://example.test/judge", key="k")
    j = verifier.judge("c", Evidence(text="evidence"))
    assert j == Judgement(label="supported")


def test_hosted_verifier_rejects_a_garbage_label(monkeypatch):
    def fake_post(url, body):
        return json.dumps({"label": "maybe"})

    monkeypatch.setattr("visual_verify.verify.backends._post", fake_post)
    verifier = HostedAPIVerifier(url="https://example.test/judge", key="k")
    with pytest.raises(Exception):
        verifier.judge("c", Evidence(text="evidence"))
```

**Step 2: Make it pass.** `backends.py`:

```python
"""The model seam.

Two roles, each with a protocol, a hosted HTTP implementation using only
stdlib, and a local HuggingFace implementation that imports torch and
transformers lazily. The CLI wires the default pairing; the independence
rule is a wiring assertion, tested above.

The local VLM fit is a measurement, not a constant: spec 3.1 measured the
card and the plan's live task records which model actually loads. These
IDs are the defaults to try first.
"""

from typing import Protocol

import numpy as np
from PIL import Image

from visual_verify.verify.claims import ReaderOutput, parse_reader_output
from visual_verify.verify.evidence import Evidence
from visual_verify.verify.rubric import Judgement

DEFAULT_READER_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_VERIFIER_MODEL = "Qwen/Qwen2.5-VL-2B-Instruct"


class Reader(Protocol):
    def read(
        self, question: str, image: Image.Image | None, text_layer: str | None
    ) -> ReaderOutput: ...


class Verifier(Protocol):
    def judge(self, claim: str, evidence: Evidence) -> Judgement: ...


def _post(url: str, body: str, key: str | None = None) -> str:
    """One stdlib POST. Kept as a module function so tests can stub it."""
    import urllib.request

    req = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


class HostedAPIReader:
    """Reader through any OpenAI-style chat-completions endpoint."""

    def __init__(self, url: str, key: str | None, model: str = DEFAULT_READER_MODEL):
        self.url = url
        self.key = key
        self.model = model

    def read(self, question: str, image: Image.Image | None, text_layer: str | None) -> ReaderOutput:
        messages = [{"role": "user", "content": _reader_prompt(question, text_layer)}]
        response = json.loads(_post(self.url, json.dumps({"model": self.model, "messages": messages}), self.key))
        raw = response["choices"][0]["message"]["content"]
        return parse_reader_output(raw)


class HostedAPIVerifier:
    def __init__(self, url: str, key: str | None, model: str = DEFAULT_VERIFIER_MODEL):
        self.url = url
        self.key = key
        self.model = model

    def judge(self, claim: str, evidence: Evidence) -> Judgement:
        messages = [{"role": "user", "content": _verifier_prompt(claim, evidence.text)}]
        response = json.loads(_post(self.url, json.dumps({"model": self.model, "messages": messages}), self.key))
        raw = response["choices"][0]["message"]["content"]
        return Judgement(**json.loads(raw))
```

The two prompt builders and the local classes:

```python
_READER_PROMPT = """Answer the question from the page provided. Return ONLY JSON:
{"answer": "the full answer as a sentence", "claims": ["one atomic claim per sentence"]}
Each claim must be a single assertion that can be located on the page by itself."""


def _reader_prompt(question: str, text_layer: str | None) -> str:
    layer = text_layer if text_layer else "(no text layer; use the image)"
    return f"{_READER_PROMPT}\n\nQuestion: {question}\n\nPage text:\n{layer}"


_VERIFIER_PROMPT = """Judge whether the EVIDENCE supports the CLAIM. Return ONLY JSON:
{"label": "supported" | "partial" | "unsupported" | "insufficient"}
supported: the evidence establishes the claim.
partial: the evidence establishes part of the claim.
unsupported: the evidence contradicts the claim or fails to establish it.
insufficient: the evidence does not address the claim at all."""


def _verifier_prompt(claim: str, evidence_text: str | None) -> str:
    ev = evidence_text if evidence_text else "(no text; only an image crop)"
    return f"{_VERIFIER_PROMPT}\n\nCLAIM: {claim}\n\nEVIDENCE: {ev}"
```

Local classes are sketched structurally here; the live task measures the load and completes them:

```python
class LocalVLMReader:
    """HuggingFace generative VLM as reader. Lazy imports: the core must be
    importable without torch, and so must this module."""

    def __init__(self, model_id: str = DEFAULT_READER_MODEL, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def read(self, question, image, text_layer):
        from ._local import generate_json  # lazy, lives outside the core import path

        raw = generate_json(self.model_id, self.device, _reader_prompt(question, text_layer), image)
        return parse_reader_output(raw)
```

If the lazy local loader turns out to be more than a stub, it lives in
`src/visual_verify/verify/_local.py`, which is the ONLY module allowed to import
transformers/torch, and it is never imported by the core.

**Step 3: Run** `uv run --python 3.13 pytest -q tests/test_backends.py`.

---

### Task 6: The CLI command

**Files:**
- Modify: `src/visual_verify/cli.py`
- Modify: `tests/test_cli.py` (or a new `tests/test_cli_verify.py`)

**Step 1: Write the failing test** — follow the existing `cmd_ground` test's shape in `tests/test_cli_ground.py` (see S4's merge for the established pattern): invoke `main(["ask", "--doc", ..., "--page", ...])` against a stub backends layer, assert the printed per-claim verdict lines and the overlay file.

**Step 2: Implement.** `cmd_ask` is the adapter:

- Resolve the document and page exactly as `cmd_ground` does; fetch `boxes` (word kind), the page image path, and the text layer.
- Build backends from config: `VVRAG_READER_BACKEND` and `VVRAG_VERIFIER_BACKEND` in `{local, hosted}`, with `VVRAG_READER_URL`, `VVRAG_READER_KEY`, `VVRAG_VERIFIER_URL`, `VVRAG_VERIFIER_KEY` for hosted, and model IDs for local. Unset or invalid backend names produce a clear error, never a silent default.
- Fetch vectors and grid exactly as `cmd_ground` does, but lazily: only when a claim needs the visual path. The embedder is constructed once (`_make_embedder`) and passed as `embed=embedder.embed_query`.
- Call `verify(...)` and print:

```
answer: It is 42.
  [supported 1.000] "The answer is 42."  [(0.100 0.200) (0.400 0.500)] text
  [ABSTAINED 0.000] "Nothing else matters."  (no evidence)
abstained_overall: False
```

- `--overlay PATH` draws surviving regions like `cmd_ground`.
- Errors print the stage and return 1.

**Step 3: Run** the CLI tests. The full suite runs in Task 8.

---

### Task 7: Live measurement and the end-to-end test

**Files:**
- Create: `tests/test_verify_live.py`

The live test follows `test_grounding_live.py`: skip unless the corpus and the configured models are available. It asserts one end-to-end ask returns an `Answer` with per-claim regions and verdicts.

The measurement the spec 3.1/4 promised happens here and is recorded in the spec's
Status line and in the CLI defaults:

1. Load the local reader candidate on the card with ColQwen2 unloaded; record peak VRAM and load time.
2. Run one verify on a known-text question; check the visual path only when forced.
3. If no local candidate fits the card, the defaults stay hosted for the reader and the verifier picks a second vendor, and the spec records that finding.

**Definition of done for this task:** the spec's Status line names the measured default pairing, the CLI defaults match it, and the live test passes with it.

---

### Task 8: Close out

1. **Full suite:** `uv run --python 3.13 pytest -q` — everything green, nothing skipped that did not skip before this slice.
2. **Lint:** `uv run --python 3.13 ruff check src tests`.
3. **ROADMAP:** mark the S5 checkboxes that are done, record the compute-path decision in the S5 blocker paragraph, and note the measured default pairing.
4. **Spec status:** flip to "implemented and merged" with the measured pairing recorded, and reconcile any section the implementation contradicted, the way S4's merge did.
5. **README:** add `vvrag ask` to the command list if the README documents commands.

---

## Definition of done

- `verify()` runs read → ground → judge → gate over one page with no model constructed in the core, no new core dependency, and every code path asserted with fakes.
- Atomic claims: malformed reader output raises `VerifierError`; empty claims abstain the whole answer.
- The four-label rubric is a pinned table; unknown labels cannot reach the gate.
- The independence rule is asserted in wiring: the default reader and verifier differ.
- `vvrag ask` prints per-claim verdicts with regions and an abstain state, and draws an overlay.
- The spec's Status records the measured compute-path pairing; the ROADMAP blocker paragraph reflects the decision.
- Full suite green; ruff clean.
