# S5: Reader and Verifier

Date: 2026-08-08
Status: design approved, not implemented
Depends on: S3 (page retrieval), S4 (grounding)

## 1. What this slice delivers

`answer(question, ...)` returns an `Answer`: a list of `Claim`s, each carrying its
grounded regions, a rubric label, a confidence, and whether it was abstained on.

Three things happen here that nothing earlier does. A reader model produces
claims from a page. Each claim is grounded by S4. A **different** model judges
whether the evidence supports the claim, and claims that fail the judgement are
withheld rather than shown.

This is pillars 2 and 3. S4 made evidence locatable; S5 makes it checkable, and
makes the system able to say it does not know.

## 2. The compute path, and why the blocker was smaller than recorded

The roadmap recorded S5 as blocked on hardware: two different VLMs against a
3.63 GiB single-tenant card, with ColQwen2 alone measuring 2.65 GB idle and 3.5 GB
under load. A 2B-parameter VLM is roughly 4.4 GB at fp16 and 2.2 GB at 4-bit, so
nothing fits alongside the retriever, and the online pipeline needs up to three
models per query.

**Both models are hosted, on two different providers.** That is not a deviation:
`proposal.tex` line 369 already specifies "a multimodal model such as Qwen-VL for
local execution, or GPT-4o or Gemini through an API", and the components table at
line 393 lists the same three.

Two providers also makes the self-preference requirement true by construction
rather than by assertion. `proposal.tex` line 377 requires the verifier to be
"a separate judge model, distinct from the reader"; different vendors cannot
share weights.

Local execution was rejected for a specific reason beyond speed. Fitting two VLMs
on this card requires 4-bit quantization, and S3 measured what blanket
quantization does here: known-item top-1 fell from 1.00 to 0.00 with no warning,
no NaN, and correctly shaped output. The ablation treats answer accuracy as a
**control** variable, so a reader degraded by quantization would move the control
for a reason unrelated to grounding, which is precisely what the ablation must
not allow.

### 2.1 What hosting costs, and how the design pays it

Two costs, both absorbed by the cache in section 6 rather than accepted.

**The defense room may have no internet.** Pre-run the demo questions; the cache
serves them offline.

**Hosted models drift.** A number reported in March is not reproducible in June
unless the raw response was kept. The cache is that record.

## 3. Client layer

LangChain, behind a new `agent` optional-dependency extra, with
`langchain-openai` and `langchain-google-genai` so both providers are reachable.

Provider and model come from the environment, following the pattern
`VVRAG_QDRANT_URL` already establishes: no module hardcodes a model name.

```
VVRAG_READER_PROVIDER    openai | google
VVRAG_READER_MODEL       e.g. gpt-4o
VVRAG_VERIFIER_PROVIDER  openai | google
VVRAG_VERIFIER_MODEL     e.g. gemini-2.0-flash
```

`with_structured_output()` is the reason this layer earns its weight: one call
gives schema-validated output on both providers, so a malformed response raises
instead of parsing into something plausible. That matters more here than usual,
because a silently mis-parsed claim list is exactly the kind of correctly-shaped
wrong output this repository keeps getting caught by.

**`langchain`, `langchain_openai`, and `langchain_google_genai` are added to
`FORBIDDEN` in `tests/test_core_is_light.py`.** The core stays at four packages
(pydantic, pymupdf, pillow, numpy) and the guard runs in a subprocess.

### 3.1 The cost of this choice, stated

LangChain pulls a large transitive dependency tree into a repository that
otherwise gates everything behind extras, and its abstraction hides the literal
request being sent. The second point has a concrete consequence: **the cache must
key on the rendered prompt text and model id, never on LangChain objects**, or a
library upgrade silently invalidates every cached response while appearing to
hit.

## 4. Reader

Input: the page image and the question. Output: a list of atomic claims.

The reader emits claims directly as structured output rather than writing prose
that a second call splits. One call instead of two, and the model that wrote the
answer is the one that decides where it separates.

The displayed answer is the claims joined, not a separate string. There is no
prose answer that the claims could drift from.

**Schema cannot enforce atomicity.** Nothing stops a reader returning "Revenue
grew 42 percent and margins held steady" as one claim, and the roadmap's own
requirement is that a sentence asserting two things cannot be grounded to one
region. So a check flags conjunction-joined claims and the test suite pins it.
Flagged, not rejected: silently dropping a claim would lose an answer, and the
right response is to surface it in the eval as a decomposition failure.

## 5. Verifier and rubric

Input: the page image, one claim, and that claim's grounded regions. Output: a
label, a confidence, and a one-sentence reason.

The four labels are fixed by `proposal.tex` line 377: `supported`,
`partially_supported`, `unsupported`, `insufficient_evidence`.

**`verify()` takes data, never a client handle.** Page image, claim, regions in;
verdict out. The model is injected. This is the same discipline that kept
`ground()` free of Qdrant and a GPU, and it buys the same thing: the entire rubric
path is testable against a fake chat model with no network and no API key.

The reason string is not decoration. It is what makes a wrong verdict debuggable
after the fact, and it goes in the eval output.

## 6. Abstention

```
score = label_rank + confidence

  supported             3 + c
  partially_supported   2 + c
  insufficient_evidence 1 + c
  unsupported           0 + c
```

The **label** decides show or abstain. The confidence only orders claims within a
label.

That split exists because S7's headline metric is confident-wrong against
coverage, and a curve needs more than four operating points. Four labels alone
give four. The confidence adds resolution for the sweep without letting a
self-reported number override the rubric.

**The threshold is a parameter, not a constant.** S7 sweeps it to produce the
coverage curve. A hardcoded threshold would make the project's headline figure
unproducible.

**Self-reported confidence is not calibrated and the report must say so.** The
proposal already names conformal calibration as future work rather than a
committed deliverable (line 381), which is the correct place for it.

## 7. Caching

Content-addressed on `(model_id, prompt_text, image_sha256)`, stored on disk.

Three jobs, only one of which is speed:

1. **Offline demo.** Pre-run the defense questions; the room needs no network.
2. **Reproducibility.** Hosted models change under you. The cached raw response is
   the evidence that a reported number was real.
3. **Free re-runs.** Re-running the eval after a code change costs nothing when
   the prompts did not change.

Cache entries record the model id, so switching provider is a miss rather than a
silent hit against another model's answer.

## 8. Data flow

```
question
  -> S3 retrieval: MaxSim page rank
  -> reader(page image, question) -> claims
  -> for each claim:
       S4 ground(claim, boxes, ...) -> regions
       verify(page image, claim, regions) -> label, confidence, reason
       score >= threshold ? show : abstain
  -> Answer
```

Grounding runs between the reader and the verifier, per claim, as
`proposal.tex` lines 340 to 342 specify.

## 9. No streaming, and the distinction that matters

The reader's output is **not** streamed to the user.

This is not a performance decision. `proposal.tex` line 377 states that no region
is shown without verification. Streaming the reader puts a claim on screen before
the verifier has judged it, and if the verdict is `unsupported` the system has
already displayed the thing it exists to withhold. Retracting it visibly is worse
than a pause. The project's entire argument is that a confident wrong answer is
worse than a slow one.

Streaming also fights structured output, since a partial JSON object cannot be
schema-validated, and it complicates content-addressed caching.

**What S6 may do instead:** stream *verified* claims one at a time, verifying
claim 1 and sending it, then claim 2. The user sees progress and nothing reaches
the screen unverified. That is a real improvement and it costs nothing
architecturally. What is ruled out is passing the reader's tokens straight
through.

## 10. Error handling

| situation | behaviour |
| --- | --- |
| provider returns a schema-invalid response | raise after retry; never coerce |
| reader returns zero claims | `Answer` with no claims, `abstained_overall=True` |
| a claim grounds to no regions | verify anyway with empty regions; the correct verdict is `insufficient_evidence`, which the rubric already has a label for |
| API key missing | raise at client construction with the env var named |
| network failure | raise; do not fall back to an unverified answer |

The third row is deliberate. An ungrounded claim is exactly the case the rubric's
fourth label exists for, so routing it around the verifier would discard the
signal the project is trying to measure.

## 11. Testing strategy

The failure mode here differs from earlier slices. S4's risk was a plausible
wrong box; S5's is a verifier that agrees with everything, which produces a
system that looks like it works and verifies nothing.

1. **The verifier must be able to say no.** A claim paired with deliberately
   wrong regions must return `unsupported`. Without this, a verifier stuck on
   `supported` passes every other test in the suite.
2. **Reader and verifier are different models**, asserted from configuration.
   The self-preference argument is the reason this slice is shaped as it is, and
   a misconfiguration that pointed both at one model would be invisible.
3. **The abstention gate actually withholds.** A claim scoring below threshold
   must not appear in the shown output.
4. **Compound claims are flagged**, since the schema cannot prevent them.
5. **The whole pipeline runs against a `FakeChatModel`** with scripted structured
   output: no network, no key, no cost, and it runs in CI.
6. **Cache round-trip**, including that a different model id misses rather than
   returning another model's answer.
7. **One live smoke test**, marked and skipped without an API key, so a fresh
   clone still runs the suite.

## 12. What S6 and S7 get

S6 receives `Answer` with per-claim labels, regions, and an abstained flag, which
is everything the UI needs to render a highlighted region and an abstention badge.

S7 receives a sweepable threshold, the cached responses that make a reported
number reproducible, and the reason strings that make a wrong verdict
explicable.

## 13. Out of scope

No conformal calibration; the proposal names it future work. No multi-page
reading, one page per claim. No streaming, per section 9. No local model
execution, per section 2.
