# Roadmap

Progress tracker for the seven slices. Each slice gets its own spec, plan, and
implementation cycle; this file is the index over all of them.

Last updated: 2026-08-03.

**Status: 3 of 7 slices complete.** S1, S2, and S3 are merged on `master`. The
written deliverables (proposal PDF, defense deck, research notes) are built. What
remains holds the project's actual research contribution: grounding,
verification, and the evaluation that measures whether they earn their place.

## The argument every slice serves

> A document QA system should show you **which region** of a page supports each
> claim, have that region checked by an **independent** model, and **abstain**
> when the check fails.

Three pillars: region-level evidence, independent verification, abstention.
Every slice exists to make one of those possible, or to measure it.

| # | Slice | Delivers | GPU | Depends on | Status |
|---|---|---|---|---|---|
| S1 | Skeleton + contracts | uv project, package, frozen contracts, test harness | No | - | Done |
| S2 | Ingest pipeline | PDF to page images, text-layer boxes, persistence, CLI | No | S1 | Done |
| S3 | Retrieval index | ColQwen2 embeddings, Qdrant multivector MaxSim | Batch | S2 | Done |
| S4 | Grounding core | `ground()`: text-span path, then visual snap-to-box | Query | S2, S3 | Done |
| S5 | Reader + verifier | Atomic claims, independent judge, abstention gate | No | S4 | Done |
| S6 | Product UI | FastAPI service + Next.js app: answer, regions, abstain badge | Yes | S5 | Done |
| S7 | Eval harness | SlideVQA, auto gold boxes, EM/F1 + IoU + confident-wrong, ablation | Yes | S4, S5 | Not started |

---

## S1: Skeleton and contracts (done)

**Objective.** A Python package with frozen public types and a hard dependency
boundary.

**Why it exists.** It serves no pillar directly. It is the slice that makes the
other six buildable without them tangling into each other.

- [x] uv project, src layout, hatchling build
- [x] Core pinned to four dependencies: pydantic, pymupdf, pillow, numpy.
      Ingest must run on a machine with no GPU; if torch leaks into the core,
      that property is gone.
- [x] Frozen contracts `GroundedRegion`, `Claim`, `Answer`, `RetrievedPage`.
      S4 through S7 all build against these, so freezing them early stops a
      later slice quietly reshaping the interface.
- [x] Normalized 0-1 bbox convention, validated. A box extracted at 150 dpi is
      still correct when the region is cropped at 300 dpi for the reader.
      Integer pixels would not survive that.
- [x] Env-driven `Settings`; no module hardcodes a connection string
- [x] `tests/test_core_is_light.py` enforces the boundary in a subprocess, which
      makes it a test rather than a convention. That is why S3 could add a
      2.5 GB torch stack safely.

Worth knowing: `score` on `GroundedRegion` is deliberately unbounded. MaxSim
sums a per-token maximum over query tokens, so it is not a probability.

## S2: Ingest pipeline (done)

**Objective.** PDF in; page images plus every word's bounding box out, stored
durably and resumably.

**Why it exists.** This is where pillar 1 becomes possible. The boxes extracted
here are the only boxes the system will ever show. Nothing downstream invents
geometry.

- Spec: `docs/superpowers/specs/2026-07-29-s1-s2-ingest-design.md`
- Plan: `docs/superpowers/plans/2026-07-29-s1-s2-ingest.md`

- [x] Born-digital gate. No OCR is in scope, so a scanned PDF has no text layer
      and grounding is impossible on it. That must fail loudly at ingest rather
      than silently produce empty pages.
- [x] Page rendering at configurable DPI
- [x] Word-level boxes carrying block, line, and word indices. Words are the
      finest granularity; lines and blocks are derived at query time, so S4 can
      retune which granularity snap-to-box ranks without re-ingesting anything.
- [x] Table cells, a candidate region that words cannot express
- [x] Derived line, block, and span boxes as pure functions
- [x] SQLAlchemy 2.0 store with Alembic migrations
- [x] Per-page `checkpoint()`. Before it existed a crash rolled back the whole
      document and resumption recovered nothing.
- [x] Content-hash identity, so re-ingesting an unchanged PDF is free
- [x] `vvrag ingest`, `vvrag status`, `vvrag inspect --find --overlay`

The methodology that came out of this slice: verify coordinates against
**rendered ink**, not arithmetic. Crop the PNG and assert dark pixels. Every
coordinate bug was found that way and none by reasoning about the maths.

## S3: Retrieval index (done)

**Objective.** Embed each page with ColQwen2, store the multivectors in Qdrant,
rank pages by MaxSim.

**Why it exists.** Two jobs, and the second matters more. It finds the right
page, and it produces the query-to-patch similarity matrix that S4 uses to rank
boxes. Retrieval here is a means to the heatmap, not an end in itself.

- Spec: `docs/superpowers/specs/2026-08-03-s3-retrieval-design.md`
- Plan: `docs/superpowers/plans/2026-08-03-s3-retrieval.md`

- [x] Retriever chosen by measurement rather than assumption
- [x] `PatchGrid`, the bridge from a model vector index to a page rectangle.
      S4 cannot place a single box without it.
- [x] Patch grid geometry stored per page. The grid is 23x32 and varies with
      page aspect ratio, so not storing it would mean re-embedding at 21 s per
      page to recover it.
- [x] Special tokens tracked. Eleven of the 747 vectors map to no page region,
      and an argmax landing on one would draw a confident box with no causal
      link to the evidence.
- [x] `EmbedProvenance` with refusal on mismatch. Vectors from different models
      are not comparable and nothing about a stored vector reveals which model
      produced it.
- [x] Row and column mean pooling, stored now though unused until S7, because
      Qdrant cannot add a named vector without recreating the collection.
- [x] `ColQwen2Embedder` with the vision tower left unquantized
- [x] `QdrantIndex`: three named vectors, MAX_SIM, HNSW m=0, schema verification
- [x] Resumable per-page embedding pipeline
- [x] `vvrag embed`, `vvrag search`
- [x] Known-item retrieval tests, the only check that caught any of the four
      silent bugs found in this slice
- [x] Full corpus indexed: 74 pages across 4 documents

Measured on the target hardware (GTX 1650, 4 GB): 21.4 s per page to embed,
197 ms to embed a query, 1 ms per page for CPU MaxSim, 187 KB per page stored.

---

## S4: Grounding core (done)

**Objective.** `ground()` returns the specific region of a page that supports a
claim.

**Why it exists.** This is pillar 2 and the project's actual contribution.
Published analysis shows ColPali-style saliency maps are fragile and misleading
as faithful attribution. So the heatmap is used only to **rank candidate boxes
that already exist in the text layer**. It never draws one. Everything before
this slice is infrastructure.

- [x] Brainstorm and write the design spec
- [x] Write the implementation plan
- [x] Text-span path first: exact substring matched to stored word boxes. Exact
      and faithful by construction, and it is the reliability floor. If the
      visual path underperforms, text answers are still grounded correctly.
- [x] Query-to-patch similarity matrix, already computed inside MaxSim but not
      currently retained
- [x] Patch index to page rectangle via the stored `PatchGrid`
- [x] Exclude special tokens before any argmax. Skipping this fabricates
      evidence, the same failure class as the over-covering `span_box` fixed
      in S2. Not precautionary: 4 to 5 of every 19 to 30 query tokens take
      their maximum on a special token, so 16 to 26 percent of a query would
      map onto a rectangle with no page region behind it.
- [x] Rank the candidate boxes by heatmap mass and select one. Select, never
      draw. This is snap-to-box.
- [x] Weight patches by the area a candidate covers, not by centre
      containment. A line is 0.0142 tall against a 0.0312 patch cell, so it
      contains no patch centre and centre selection would score every line
      zero while appearing to rank.
- [x] Scope stage two by `block_no` rather than geometry. A block is the
      bounding envelope of its words, so a wrap-around paragraph encloses
      captions belonging to other blocks, which destroys the bounded-error
      property that is the only reason selection is two-stage.
- [x] `ground()` returns `GroundedRegion` for text and visual evidence
      uniformly, so the agent and the eval harness treat them the same
- [x] Carry the line-or-block resolution on `GroundedRegion`, so a coarse
      fallback is distinguishable from a confident line hit. Without it the UI
      cannot flag a coarse region and the eval cannot separate stage-one from
      stage-two failures.
- [x] Verify a selected region by the text it covers, not by ink. Every
      candidate already contains ink (435/435 word boxes on one measured page),
      so an ink check passes a random selector. `evidence.covers_text` is the
      assertion; `evidence.has_ink` only proves the transform.
- [x] Report a random-candidate baseline next to every grounding number, so a
      selector that beats nothing cannot look like it works. Measured over 193
      trials: the selector picks the queried line 50.3 percent of the time
      against a 5.3 percent random-LINE baseline, 9.5 times the floor. Note
      this is a different quantity from the random-BLOCK floor in spec section
      8, which is a mean IoU under 0.01; both are called a floor and they are
      not comparable.
- [x] The scoring bake-off contradicted this design's own argument. Attribution
      mean leads at 0.593 mean IoU against dense mean's 0.483, while lighting 2
      percent of the grid, which the sparsity argument said should make it the
      weakest ranker. The default is NOT changed: all 193 trials come from one
      homogeneous document and S7 evaluates on SlideVQA, so this corpus can
      retire the argument without installing a replacement. Spec 6.2 and 6.3.
- [x] Flat line ranking beats two-stage selection on IoU for all three rules,
      because a wrong block in stage one is unrecoverable and the ambiguity
      fallback returns block-sized boxes against line-sized gold. Two stages
      buy bounded error and an honest resolution flag instead. The trade is
      kept and is now pinned by a test so a reversal surfaces. Spec 7.1.
- [x] `vvrag ground "<claim>" --overlay`, a picture of the claim working

Needs no new hardware: the embeddings are already in Qdrant and the candidate
boxes are already in SQLite.

Open question: which candidate granularity to rank, word or line or block or
table cell. S2 stores words and derives the rest precisely so this can be
retuned without re-ingesting.

## S5: Reader and verifier (done)

**Objective.** Answer the question, split the answer into atomic claims, have a
different model judge each claim against its evidence, and abstain when the
judgement is weak.

**Why it exists.** Pillars 2 and 3. A model grading its own output is biased
toward it, which is why the verifier must be a different model. The rubric feeds
a threshold that lets the system say it does not know.

- [x] Brainstorm and write the design spec
- [x] Write the implementation plan
- [x] **Decide the compute path** (see the blocker below)
- [x] Reader VLM answers from the retrieved page and its grounded regions
- [x] Decompose the answer into atomic claims. A sentence asserting two things
      cannot be grounded to one region.
- [x] Verifier VLM, deliberately a different model from the reader
- [x] Four-label rubric turning a judgement into a number a threshold can act on
- [x] Abstention gate. This is the point of the project: a wrong answer with a
      confident box drawn on it is worse than no answer.
- [x] `verify()` takes data, an image and boxes, never a client handle
- [x] The reader is proven live. `gpt-4o` read the proposal cover page and
      returned its real contents, the project title and all three submitter
      names, so the model genuinely reads the image rather than producing
      fluent text about nothing. A schema-valid response cannot tell those
      apart on its own, which is why this needed a real call.
- [x] The abstention bands are separated by a gap, not merely ordered. Scores
      are `2 * rank + confidence`, giving `[0,1] [2,3] [4,5] [6,7]`. With ranks
      spaced by 1 a partially supported claim at confidence 1.0 tied the
      supported floor exactly, and the gate admits ties, so it would have been
      shown as fully supported. Found by an implementer refusing to make a
      failing test pass.
- [x] The compound-claim check is wired into the pipeline as `Claim.compound`.
      It sat unused for five commits, so the requirement that a two-part claim
      is never grounded to one region was documented and unenforced. Advisory
      only: flagged, never dropped, because discarding a claim loses an answer.

**Blocker, resolved by hosting both models on different providers.** The card
was never going to fit two VLMs: ColQwen2 alone takes 2.65 GB of 3.63 GB usable
and reaches 3.5 GB under load. `proposal.tex` line 369 already specifies "a
multimodal model such as Qwen-VL for local execution, or GPT-4o or Gemini
through an API", so hosting is what the graded document says rather than a
workaround, and two vendors make the separate-judge requirement true by
construction instead of by assertion.

Local execution was rejected for a reason beyond speed. Fitting two VLMs here
needs 4-bit quantization, and S3 measured what that does on this stack:
known-item top-1 fell from 1.00 to 0.00 with no warning and correctly shaped
output. Answer accuracy is the ablation's CONTROL variable, so a reader degraded
by quantization would move the control for a reason unrelated to grounding,
which is exactly what the ablation exists to rule out.

**Outstanding: the verifier has never run against a real model.** The Google
key's project reports zero Gemini quota (`limit: 0`, a billing state rather than
a rate limit), so the two live verifier tests skip. The component that decides
show-or-abstain is therefore proven only against a scripted fake. Enable billing
on that project and the tests run as written. Until then, treat any claim about
verifier behaviour as untested against reality.

## S6: Product UI (done)

**Stack: FastAPI service plus a Next.js frontend.**

**Objective.** A question goes in; an answer comes back with the supporting
region highlighted on the page image, or an abstain badge.

**Why it exists.** It makes the claim visible. In a defense, the highlighted
region and the abstain badge are the demonstration.

**Why the split, and why it helps.** `proposal.tex` states that the grounding
and verification logic is a dependency-light core module reusable independently
of the user interface, with the product layer and the evaluation layer consuming
it through the same interface. A JavaScript frontend physically cannot import
the Python module, so the API boundary the proposal describes becomes real
rather than asserted. A single-process Python UI could always have bypassed it.

- [x] Brainstorm and write the design spec
- [x] Write the implementation plan
- [x] FastAPI service exposing ask, and page image or crop retrieval
- [x] **Load the models once at service startup and hold them in memory.** Each
      `vvrag search` invocation currently reloads ColQwen2 at about 20 s and
      2.6 GB. A request-scoped embedder would make the demo unusable and would
      OOM under any concurrency.
- [x] Single-worker deployment, because the GPU is single-tenant on this
      hardware. Two workers means two model copies and an immediate OOM.
- [x] Next.js app: question input, answer, page image with the region drawn
- [x] Region overlay rendered from normalized 0-1 coordinates, so the frontend
      scales boxes to whatever size it displays the page at
- [x] Abstain badge, visually distinct from a low-confidence answer
- [x] Per-claim evidence, since one answer can carry several claims and regions

Note: `proposal.tex` does not name a UI framework, so this choice contradicts
nothing in the graded deliverable. Two internal design specs mention Streamlit
and are now superseded by this file.

Spec: `docs/superpowers/specs/2026-08-09-s6-product-ui-design.md`
Plan: `docs/superpowers/plans/2026-08-09-s6-product-ui.md`

Worth knowing, because both look like bugs to someone reading only half of it.

**S6 is where the online pipeline first exists end to end.** The CLI still
splits it, deliberately: `vvrag ask` takes an explicit page because it is a
debugging surface, while the service retrieves. `api/ask.py` is the only place
retrieve, read, ground and verify are one call.

**The CLI and the UI disagree about withheld claims on purpose.** `vvrag ask`
prints every claim including rejected ones, because a diagnostic surface should
show everything. The UI uses `Answer.shown`, and `api/wire.py` strips a rejected
claim's regions before they leave the process, so the browser cannot draw
evidence the verifier refused even by mistake. Same data, opposite default. Do
not "fix" either to match the other.

**S7 still needs the regions of rejected claims**, which is exactly why that
strip lives at the API boundary and not in `answer()`. Confident-wrong against
coverage cannot be computed without them.

## S7: Eval harness (not started)

**Objective.** Measure the system on SlideVQA and show whether the grounding and
verification layers earn their place.

**Why it exists.** It turns claims into evidence. Without it there is a system
that appears to work.

- [x] Brainstorm and write the design spec
- [x] Write the implementation plan
- [ ] SlideVQA loading. These are landscape slides, so a different patch grid
      than the project's A4 pages, which works only because S3 stores the
      geometry per page rather than assuming a constant.
- [ ] Auto-derived gold boxes from `derive.span_boxes`. This is why the
      over-covering union fixed in S2 was a correctness bug rather than an
      imprecision: it would have silently capped the reported IoU.
- [ ] Answer metrics, EM and F1. Accuracy is a control variable here, not the
      quantity being maximized.
- [ ] Grounding metrics: mean IoU, hit rate at IoU 0.25 and at 0.5
- [ ] Report every IoU with its oracle ceiling and the random-candidate floor.
      Granularity caps IoU independently of the selector: a perfect selector
      reaches only 0.195 at line level against a 3-word gold span, so a bare
      0.15 reads as failure when it is in fact near-ceiling work.
- [ ] State in the results chapter that the 0.5 to 0.6 figure in the proposal
      (line 452) comes from BBox-DocVQA, whose gold regions are page areas
      rather than 3-word spans, and is therefore not directly comparable.
      Decided 2026-08-08: the proposal is left unedited and this explanation
      carries it. See the S4 spec, section 8.1.
- [ ] The headline metric: confident-wrong rate against coverage. Answering less
      while being wrong less is the win condition.
- [ ] Three-way ablation, Baseline against Grounded against Verified, isolating
      what each layer contributes
- [ ] Pooled-prefetch rerank as a measured experiment, using the vectors S3
      already stored

---

## Measured baseline

Three paraphrased queries against the 74-page corpus, run 2026-08-03. Recorded
because S7's ablation needs an honest starting point.

- "how is the system evaluated and which metrics are used" returned the correct
  evaluation-metrics page at rank 1 and the confident-wrong formula at rank 3.
- "why is the heatmap not treated as faithful evidence" and "what is snap to box
  grounding" both returned the proposal's SUMMARY page at rank 1, a page that
  mentions every concept once and therefore matches everything moderately.

Retrieval works and is mediocre on conceptual queries. That is the baseline S4
and S5 have to improve on, and it is more useful than a perfect score would have
been, because a perfect baseline leaves an ablation with nothing to show.
