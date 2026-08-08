# S5: Reader and Verifier

Date: 2026-08-08
Status: implemented and merged. See docs/superpowers/plans/2026-08-08-s5-reader-verifier.md
Depends on: S3 (retrieved pages), S4 (grounded regions per claim)

## 1. What this slice delivers
`verify(question, page, evidence, reader, verifier)` answers the question from the
retrieved page, splits the answer into atomic claims, grounds each claim to a region,
and has a **different model** judge each claim against that region. Claims whose
judgement is weak are **abstained**, not answered.

The abstention gate is the point of the project. A wrong answer with a confident box
drawn on it is worse than no answer.

## 2. Where S5 sits

`proposal.tex` wires the online pipeline as `retrieve -> reader -> ground -> verify`.
S3 delivered `retrieve`; S4 delivered `ground` per claim; S5 delivers the two model
roles and the gate between them.

S4's spec section 13 fixed what S5 consumes: `list[GroundedRegion]` per claim, uniform
across modalities, so the verifier's rubric applies without knowing whether a region
came from the text path or the visual path. `contracts.py` already carries the output
shapes this slice fills: `Claim(text, regions, confidence, abstained)` and
`Answer(question, claims, abstained_overall)`.

The seam `verify()` must consume **data, an image, and boxes** — never a client handle.
Everything else in this repo obeys that: `ground()` takes plain arrays, the CLI is the
only adapter that fetches. S5 extends the same rule to models: the core calls reader and
verifier **objects passed in**, and the CLI (and later S6) is where those objects get
built from hardware or API configuration.

## 3. Measured constraints

### 3.1 The local card cannot hold a 7B model, ever

Measured on this machine (2026-08-08): NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB
total. The S4 work measured ColQwen2 loading at 2.65 GB against 3.63 GB usable — the
difference is Windows and driver reservation, not headroom we can spend.

**The dev venv currently cannot reach the card at all.** `uv run python -c "import
torch"` reports `2.6.0+cpu`: on Windows, PyPI's torch wheels are CPU-only, and this
lock has no index pointing at download.pytorch.org. Any local-VLM measurement therefore
needs `torch==2.6.0+cu124` installed from the pytorch index first. The S4-era GPU
numbers were measured in an environment that had it; this one does not, and the
difference is recorded rather than assumed away.

**The local verifier was measured on the card (2026-08-08, after installing
`torch==2.6.0+cu124`).** The candidate `Qwen/Qwen2.5-VL-2B-Instruct` does not exist —
the Qwen2.5-VL family starts at 3B — so the default became `Qwen/Qwen2-VL-2B-Instruct`,
and it works:

| variant | VRAM | load time | one judgement |
| --- | --- | --- | --- |
| fp16 | 4.2 GB | ~16-20 s (first: model download) | ~16-20 s text, ~16 s image |
| nf4 4-bit | 1.5 GB | ~16 s | same, ~2x slower per token |

The fp16 verifier cannot coexist with ColQwen2's 2.65 GB on this card; sequential
load/unload (the ROADMAP's first option) or the 4-bit build is required for an ask
that needs both in one process. The end-to-end ask against a synthetic page judged a
true claim "supported" at confidence 1.0 with zero abstention errors.

| model | fp16 footprint (est.) | fits 3.63 GB usable? |
| --- | --- | --- |
| ColQwen2 (retriever, already loaded by S3) | 2.65 GB measured | yes, alone |
| Qwen2.5-VL-7B-Instruct (research/recommended_stack.md) | ~15 GB | **no** |
| Qwen2.5-VL-3B, nf4 | ~2-3 GB | marginal, only one at a time |
| SmolVLM2-2.2B | ~4.4 GB fp16 / ~1.5 GB quantized | marginal, only one at a time |

Two roles and one card. **At least one of reader or verifier cannot be a local VLM on
this hardware**, and whichever role is local must be small enough to coexist with, or
sequentially replace, ColQwen2. This is the ROADMAP blocker, and it is settled here by
measurement rather than coded around.

### 3.2 Independence is a constraint, not a preference

A model grading its own output is biased toward it. The verifier must be a different
model. The repo's research pass (research/recommended_stack.md section 2.3) made the
consequence explicit: independence is chosen by *family or vendor difference*, not by
benchmark, because no benchmark picks a "best verifier".

If both roles run locally they must be different families. If both run hosted they must
be different vendors. The wiring in the CLI encodes and can test this.

### 3.3 The verifier's job is a constrained classification

The rubric is four labels over (claim, region). A small VLM is enough. The corollary
from recommended_stack.md section 2.3: a text-region claim can be judged from the region
text alone; only chart-and-figure crops need the vision input. The evidence assembly in
section 7 makes that a data-shape decision, not a model decision.

## 4. The compute-path decision

The ROADMAP listed three options: sequential load and unload, a hosted API for the
verifier, or the campus GPU. Section 3.1 removes the "both local" branch; the remaining
options are *which* role is local and which is hosted, or neither local on this card.

The seam in section 5 makes this a wiring choice, and the choice is recorded, per
environment, like S3's was. The design rule, fixed here:

1. **The core never builds a model.** Reader and verifier arrive as objects.
2. **Both roles have a local implementation and a hosted implementation.** The local
   ones are thin HuggingFace loaders; the hosted ones are thin HTTP callers. Neither
   lives in the core.
3. **The pairing must satisfy 3.2.** A test over the CLI wiring asserts the chosen
   reader and verifier are not the same model ID, which catches the failure without a
   GPU.
4. **Default pairing for this machine:** reader = hosted (Gemini 2.5 Flash free tier,
   per recommended_stack.md), verifier = local small VLM if one fits the card in
   measurement, else hosted from a second vendor. The plan measures the local fit the
   way S3 measured the retriever, and records what the default actually is.
5. **The campus GPU story survives.** With ~44-50 GB available on a 64 GB card, both
   roles can be local 7B-class there. Nothing in the core changes; the wiring changes.

## 5. The `verify()` contract

```python
class Reader(Protocol):
    def read(
        self, question: str, image: Image.Image | None, text_layer: str | None
    ) -> ReaderOutput: ...

class Verifier(Protocol):
    def judge(self, claim: str, evidence: Evidence) -> Judgement: ...

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
```

`verify()` runs one retrieved page end to end:

1. `reader.read(question, image, text_layer)` returns the answer text and its atomic
   claims. (Section 6.)
2. For each claim: `ground(claim, boxes, ...)` from S4, with the same arguments and the
   same `force` flag, so the S7 ablation can measure the visual path on questions the
   text path would otherwise have answered. (Section 9.)
3. For each grounded claim: assemble evidence (section 7) and call
   `verifier.judge(claim, evidence)`.
4. Map the rubric label through the gate (section 8). Weak claims get
   `abstained=True`; `abstained_overall=True` when no claim survives.

### 5.1 What gets passed in, and why

`image` and `text_layer` are the page's render and text, passed in. `boxes` are the
word records. Vectors and grid arrive the same way they do in `ground()`: None unless
the visual path can be reached. The CLI is the adapter that fetches all of it, exactly
as `cmd_ground` does today.

**The embedding seam.** `ground()`'s visual path needs query vectors for the claim it
is grounding. Claims only exist after the reader runs, so the CLI cannot pre-embed
them; `verify()` receives `embed: Callable[[str], np.ndarray]`, which embeds one claim
string and returns its query vectors, and calls it only when the visual path is
actually reached. The CLI supplies the S3 embedder; tests supply a fake; a core caller
that hits the visual path without `embed` gets `VerifierError` rather than a silent
fallback to nothing.

A claim with no regions keeps its text and is judged against no evidence — the
verifier's `insufficient` label, which the gate abstains. An empty region list from
`ground()` means "no evidence exists on this page" (S4 section 5.1), and the verifier is
the only thing allowed to turn that into a judgement.

## 6. Atomic claims

The reader emits the answer **and** the claims, in one structured response. Splitting
is the model's job; the core's job is to validate the structure and refuse to proceed
on garbage.

**Why atomic:** a sentence asserting two things cannot be grounded to one region. The
reader is instructed to emit one claim per assertion, each a self-contained sentence
the verifier can check and the grounder can locate.

**The parsing contract.** The reader returns JSON of the form

```json
{"answer": "...", "claims": ["...", "..."]}
```

The core validates: claims is a list of non-empty strings. A well-formed response with
zero claims abstains the whole answer (an answer with nothing checkable is not an
answer). A response that is not valid JSON, or is JSON of the wrong shape, raises
`VerifierError`: silently dropping claims would turn an uncheckable answer into a
confident-looking one, which is the failure class this slice exists to prevent.

Reader implementations that cannot produce structured output are rejected at the seam:
`read()` returns `ReaderOutput`, not text, so a free-form reader cannot be wired in by
accident.

## 7. Evidence assembly

Each `GroundedRegion` becomes the input to one `judge()` call:

- **text modality:** `Evidence(text=region.text)` — the exact matched span. No image.
- **visual modality:** `Evidence(text=region.text, image=crop(image, bbox))` — the
  snapped region's text (if any) plus the crop cut from the page render.

The crop is a pure PIL operation on the page image and the region bbox, testable with a
synthetic image and no model. S4's coordinates are normalized 0-1, so the crop is
`image.crop((bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h))`; the one coordinate system
everywhere rule (contracts.py) makes this a two-line function.

**A claim is judged against its best region, not all of them.** `ground()` can return
several regions for one claim (several matching spans, say). The verifier judges the
highest-scoring region; ties break to the first. Judging against every region would let
a weak claim pass on a stray match, and a claim that needs three regions to be true was
split wrongly at step 1. All regions still travel on the returned `Claim`, so the UI
and S7 see everything; only the judgement is per-claim.

This is the shape-level version of 3.3: the verifier model decides how to use its
inputs, but the core decides what a claim may be judged against, and it is always the
region S4 actually returned.

## 8. The rubric and the gate

The verifier returns one of four labels:

| label | meaning | sufficiency |
| --- | --- | --- |
| `supported` | region establishes the claim | 1.0 |
| `partial` | region establishes part of the claim | 0.5 |
| `unsupported` | region contradicts or does not establish it | 0.0 |
| `insufficient` | region does not address the claim | 0.0 |

The mapping label → number is **core code, pinned by tests**. The model never emits
numbers; it emits a label, and a threshold acts on the mapped number. That keeps the
gate auditable: the rubric is a table, not model output.

**The gate.** A claim is answered iff `sufficiency(label) >= threshold`. Otherwise it is
abstained. `abstained_overall` is true when no claim survives.

**Why one number, not two.** The project's headline is confident-wrong rate against
coverage (S7). Raising the threshold answers less and is wrong less; lowering it does
the reverse. A single threshold parameterizes that trade, and S7 tunes it — the gate is
a knob, not a policy, and the ablation can separate the verifier's contribution from the
gate's.

## 9. The `force` flag survives, for the same reason it was added

`ground()`'s `force="visual"` exists so S7 can measure the visual path on questions the
text path would otherwise answer (S4 section 5). `verify()` forwards it. Without that,
the visual path can only be measured on questions with no text match, and the
grounding-overlap metric cannot be computed on text-answerable questions.

The plan's Task for the CLI will default `force` to unset and expose the flag exactly as
`vvrag ground` does.

## 10. Error handling

- Malformed reader output: raise `VerifierError`. (Section 6.)
- Verifier returning an unknown label: raise `VerifierError`. A label table with a
  string not in it is a contract violation, not a judgement; silently mapping it would
  put garbage through the gate.
- Any reader or verifier transport failure (network, OOM, empty response): propagate
  as `VerifierError` with the role named. The CLI catches it, prints the stage, and
  returns nonzero. A failed verification is never an abstain and never an answer: both
  would fake a judgement that did not happen.
- Empty `regions` for a claim is **not** an error; it feeds the verifier with no
  evidence and the gate abstains. (S4 section 5.1: absence is not doubt, and the
  verifier — not the grounder — owns doubt.)

## 11. The core boundary

Pure and testable without models, without a GPU, without the network:

- claim validation and normalization (claims.py)
- rubric mapping and gate (rubric.py)
- evidence assembly, including the crop (evidence.py)
- orchestration over injected fakes: reader returns canned output, verifier returns
  canned labels, and the assertions cover routing, force forwarding, gate behaviour,
  and the Answer shape (core.py)

Thin and wired in the CLI only:

- local model loaders (backends: HuggingFace)
- hosted callers (backends: HTTP)
- the `vvrag ask` command, which fetches pages, vectors, images, and boxes the way
  `cmd_ground` does

The live path is tested in a file that skips without the model and corpus, following
`test_grounding_live.py`'s convention.

## 12. Testing strategy

| file | covers |
| --- | --- |
| `tests/test_claims.py` | parsing contract: valid, empty, malformed, wrong shape |
| `tests/test_rubric.py` | label table, unknown-label error, threshold behavior |
| `tests/test_verify_evidence.py` | text vs visual evidence, crop geometry, absence case |
| `tests/test_verify.py` | orchestration with fakes: pipeline order, force forwarding, gate, abstain decisions, Answer shape |
| `tests/test_backends.py` | wiring assertions: default reader/verifier pairing differs (3.2), hosted caller builds the right request shape |
| `tests/test_verify_live.py` | one end-to-end ask against the real corpus and whichever pairing is the measured default; skips without them |
| `tests/test_cli.py` | `vvrag ask` invocation and error paths with a stub backends layer |

## 13. CLI

`vvrag ask "<question>" --doc DOC --page N [--force visual] [--threshold 0.5]`
replaces the manual claim on `vvrag ground`: it reads the page, grounds each claim,
verifies each, and prints the answer with per-claim verdicts and regions. `--overlay`
draws the surviving regions, as `vvrag ground` does.

## 14. What S6 and S7 get from this

S6 receives `Answer` with per-claim `Claim.confidence`, `Claim.abstained`, and regions
already in normalized coordinates: everything the UI needs for the answer text, the
region overlay, and the abstain badge, with no new model calls.

S7 receives `force="visual"` forwarded through, the gate as a single threshold knob,
and the confident-wrong vs coverage measurement the threshold makes possible. The
three-way ablation — Baseline against Grounded against Verified — maps onto
`ask` with fake reader/verifier, real reader/fake verifier, and the full wiring.

## 15. Out of scope

No retrieval (S3 owns it), no new grounding logic (S4 owns it), no UI (S6), no metrics
harness (S7). No fine-tuning of any model. No new dependencies in the core: pydantic,
PIL, and numpy cover everything in sections 5 through 8; the backends may add the HTTP
client and the model libraries, and those additions are recorded in the plan.
