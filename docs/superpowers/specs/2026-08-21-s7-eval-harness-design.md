# S7 Evaluation Harness (Retrieval-Augmented Generation) -- Design

**Status:** approved, ready for implementation planning.

## Goal

Build the offline SlideVQA evaluation harness the proposal commits to
(`proposal_report/proposal.tex` lines 23, 406-472): answer accuracy (EM/F1),
grounding overlap (mean IoU, hit-rate@0.25), and confident-wrong rate vs.
coverage, across a 3-way ablation (Baseline / Grounded / Verified). No campus
GPU is available yet, so this first pass runs on a local GTX 1650 against a
deliberately small deck pool, not the full ~14.5k-question SlideVQA split.
Scaling to the full split (or a larger subset) on campus GPUs is later work
and should require config changes only, not a rewrite.

## Non-goals for this pass

- The full 1k-question (or larger) SlideVQA run. This design proves the
  pipeline works end to end on a small, GPU-time-bounded pool.
- Manual gold-box annotation. Gold boxes are auto-derived (see below);
  questions whose answer text is not literally present in the OCR'd text
  layer get no gold box and are excluded from grounding metrics.
- Changing `agent/core.py`'s `answer()`/`answer_stream()`. The ablation is
  built by composing the same already-separable pieces
  (`read()`, `ground()`, `verify()`) from the eval harness, not by adding an
  eval-only stage-skip switch to the production pipeline.

## Build order

Metrics and harness logic are built and proven against a small hand-written
fixture *before* any real SlideVQA data, OCR, or GPU embedding work happens.
This proves the scoring logic is correct in isolation and de-risks the
data-pipeline work, which is the most time-consuming and error-prone part.

1. `eval/metrics.py` -- pure scoring functions, no I/O.
2. `eval/harness.py` -- the three-arm runner, tested against a 2-3-question
   hand-written fixture (known claims/answers/boxes/pages, no network).
3. `eval/dataset.py` -- real SlideVQA download, OCR, synthetic-PDF assembly,
   ingest/embed/index into the eval collection.
4. CLI commands + the real ~15-20 deck / ~100-150 question run.

## Section 1: Data pipeline

SlideVQA ships as slide PNGs per deck plus a QA JSON (question, reference
answer, gold deck/slide index). There is no PDF and no text layer, which
conflicts with this project's reliability floor: the exact text-span path
requires a PyMuPDF text layer (`CLAUDE.md`, "snap-to-box, never
draw-from-pixels").

Resolution: run OCR once per slide (Tesseract, CPU-only, no GPU needed) to
get word boxes and text, then assemble a one-page-per-slide PDF per deck with
the OCR'd words burned in as invisible text via PyMuPDF's `insert_text` at
each box's position. The real `ingest/` module then runs completely
unmodified against these synthetic PDFs -- same word/line/block extraction,
same coordinate handling, same `sink.checkpoint()` -- so the eval harness
exercises the actual product pipeline rather than a parallel eval-only code
path.

Caveat, stated plainly: OCR introduces its own error (misread words, boxes
slightly off) that a born-digital PDF would not have. A failed text-span
grounding on the eval set can mean "OCR missed it," not "the system failed."
This is a known, accepted limitation of this first pass, not something to
solve here.

Indexing goes into a **new, separate Qdrant collection, `slidevqa_eval`**,
via the same `vvrag embed`-equivalent path (ColQwen2, ~21.4 s/page on the
GTX 1650). The existing `pages` collection (dev/demo documents, used for
product-UI testing) is left untouched.

## Section 2: Scope and sampling

Target: **~15-20 decks (~300-400 slide pages), yielding roughly 100-150
questions** (SlideVQA averages ~5-6 questions/deck). This bounds one-time
embedding to roughly 1.8-2.4 hours on the GTX 1650, instead of the many hours
a true random 1k-question sample would require (a random 1k spans several
hundred decks, since decks average ~20 slides).

Sampling is deck-first: pick the deck pool (~15-20 decks), then take every
question SlideVQA attaches to those decks -- no further subsampling within
the pool. This is a deliberate deviation from "random 1k from the dev split"
for this first pass; the harness's deck-pool size is a parameter so a later,
GPU-time-unconstrained run can widen it toward the full random-1k (or larger)
sample without a design change.

## Section 3: Gold regions

No manual annotation. For each question, search the reference answer string
against the deck's OCR'd text layer using the same span-match logic
`ingest/evidence.py` already implements for the product pipeline's own
grounding, and take the matched line(s) as the gold box(es) on SlideVQA's
named gold slide. Questions whose reference answer does not appear literally
in the OCR'd text (paraphrased or numerical-reasoning questions) get no gold
box: they are scored on EM/F1 only and excluded from IoU/hit-rate, rather
than being force-matched to something approximate.

## Section 4: The three ablation arms

All three arms share the same retrieved top page(s) from `slidevqa_eval`
(Qdrant MaxSim search), so retrieval quality is common across arms and the
ablation isolates the grounding/verification layer, per proposal line 472.

- **Baseline**: retrieve -> `read()` -> the reader's drafted claims, joined,
  are the answer text. No grounding, no verification, no abstention. This is
  "plain document RAG": scored on EM/F1 only.
- **Grounded**: retrieve -> `read()` -> `ground()` each claim. No `verify()`,
  no abstention gate -- the answer is shown regardless of a verifier
  judgment. Scored on EM/F1 and grounding (IoU/hit-rate), isolating what
  grounding alone contributes.
- **Verified**: the real `answer()`/`answer_stream()` path, full
  retrieve -> read -> ground -> verify -> abstention-gate chain. The only arm
  that calls the verifier model, and therefore the only arm constrained by
  the verifier's rate limits (see Section 6). Scored on all three metric
  families, including confident-wrong-vs-coverage.

## Section 5: Metrics

`eval/metrics.py`, pure functions:

- **Answer accuracy**: exact match and token-level F1 between generated and
  reference answer, following the SlideVQA/HotpotQA convention already cited
  in the proposal (line 424).
- **Grounding**: mean IoU and hit-rate at IoU >= 0.25 between a claim's cited
  region(s) and the auto-derived gold box(es), over only the questions that
  have a gold box (Section 3).
- **Abstention**: confident-wrong rate vs. coverage -- of the claims the
  Verified arm did NOT withhold, what fraction were wrong, as a function of
  how much of the question set it chose to answer at all.

## Section 6: Models and rate limits

- **Reader**: `VVRAG_READER_PROVIDER=openai`, `VVRAG_READER_MODEL=gpt-5-nano`.
  Called once per question, all three arms.
- **Verifier**: `VVRAG_VERIFIER_PROVIDER=openai_compatible`,
  `VVRAG_VERIFIER_BASE_URL=https://api.groq.com/openai/v1`,
  `VVRAG_VERIFIER_MODEL=qwen/qwen3.6-27b` (Groq's vision-capable preview
  model), `VVRAG_VERIFIER_API_KEY=<groq key>`. Called once per claim, Verified
  arm only. Different provider families from the reader, so the self-preference
  identity guard (`agent/models.py::model_family`) is satisfied automatically.
- Groq's free tier for this model: 30 RPM / 1,000 RPD / 8,000 TPM / 200,000
  TPD. The harness paces verifier calls to stay under 30 RPM and checkpoints
  per-question results to a resumable file, so a run that exhausts the daily
  token or request budget can pause and resume the next day without
  re-judging already-scored claims.
- OpenAI budget: $3.92 credit on the account running the reader. gpt-5-nano
  is priced low enough that ~100-150 single-page reader calls should stay
  well under this, but the harness should log token usage per call so this is
  verifiable rather than assumed.

## Section 7: Package structure and output

New `src/visual_verify/eval/` package, an optional dependency group (like
`store`/`retrieval`) so the core package's four-dependency rule
(`tests/test_core_is_light.py`) is unaffected:

- `eval/metrics.py` -- scoring functions (Section 5).
- `eval/harness.py` -- the three-arm runner (Section 4), built and fixture-
  tested before `eval/dataset.py` exists (see Build order).
- `eval/dataset.py` -- SlideVQA download, OCR, synthetic-PDF assembly,
  ingest/embed/index (Section 1). Built after the harness is proven.

Output: per-question results to a JSONL file (one line per question per arm,
resumable, inspectable without a parser), plus an aggregated metrics table
per arm (matching the proposal's ablation table shape) printed and saved as
JSON.

CLI, following the existing `cmd_*` pattern in `cli.py`:

- `vvrag eval prepare` -- Section 1's one-time deck-pool build (OCR, PDF
  assembly, ingest, embed, index into `slidevqa_eval`).
- `vvrag eval run --arm {baseline,grounded,verified}` (or `--all`) -- runs
  the harness against the prepared pool and writes results.
