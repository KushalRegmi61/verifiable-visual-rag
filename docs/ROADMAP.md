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
| S4 | Grounding core | `ground()`: text-span path, then visual snap-to-box | Query | S2, S3 | Not started |
| S5 | Reader + verifier | Atomic claims, independent judge, abstention gate | Yes | S4 | Not started |
| S6 | Product UI | FastAPI service + Next.js app: answer, regions, abstain badge | Yes | S5 | Not started |
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

## S4: Grounding core (not started)

**Objective.** `ground()` returns the specific region of a page that supports a
claim.

**Why it exists.** This is pillar 2 and the project's actual contribution.
Published analysis shows ColPali-style saliency maps are fragile and misleading
as faithful attribution. So the heatmap is used only to **rank candidate boxes
that already exist in the text layer**. It never draws one. Everything before
this slice is infrastructure.

- [ ] Brainstorm and write the design spec
- [ ] Write the implementation plan
- [ ] Text-span path first: exact substring matched to stored word boxes. Exact
      and faithful by construction, and it is the reliability floor. If the
      visual path underperforms, text answers are still grounded correctly.
- [ ] Query-to-patch similarity matrix, already computed inside MaxSim but not
      currently retained
- [ ] Patch index to page rectangle via the stored `PatchGrid`
- [ ] Exclude special tokens before any argmax. Skipping this fabricates
      evidence, the same failure class as the over-covering `span_box` fixed
      in S2.
- [ ] Rank the candidate boxes by heatmap mass and select one. Select, never
      draw. This is snap-to-box.
- [ ] `ground()` returns `GroundedRegion` for text and visual evidence
      uniformly, so the agent and the eval harness treat them the same
- [ ] Verify a selected region by the text it covers, not by ink. Every
      candidate already contains ink (435/435 word boxes on one measured page),
      so an ink check passes a random selector. `evidence.covers_text` is the
      assertion; `evidence.has_ink` only proves the transform.
- [ ] Report a random-candidate baseline next to every grounding number, so a
      selector that beats nothing cannot look like it works
- [ ] `vvrag ground "<question>" --overlay`, a picture of the claim working

Needs no new hardware: the embeddings are already in Qdrant and the candidate
boxes are already in SQLite.

Open question: which candidate granularity to rank, word or line or block or
table cell. S2 stores words and derives the rest precisely so this can be
retuned without re-ingesting.

## S5: Reader and verifier (not started)

**Objective.** Answer the question, split the answer into atomic claims, have a
different model judge each claim against its evidence, and abstain when the
judgement is weak.

**Why it exists.** Pillars 2 and 3. A model grading its own output is biased
toward it, which is why the verifier must be a different model. The rubric feeds
a threshold that lets the system say it does not know.

- [ ] Brainstorm and write the design spec
- [ ] Write the implementation plan
- [ ] **Decide the compute path** (see the blocker below)
- [ ] Reader VLM answers from the retrieved page and its grounded regions
- [ ] Decompose the answer into atomic claims. A sentence asserting two things
      cannot be grounded to one region.
- [ ] Verifier VLM, deliberately a different model from the reader
- [ ] Four-label rubric turning a judgement into a number a threshold can act on
- [ ] Abstention gate. This is the point of the project: a wrong answer with a
      confident box drawn on it is worse than no answer.
- [ ] `verify()` takes data, an image and boxes, never a client handle

**Blocker, unresolved.** The design needs two different VLMs. ColQwen2 alone
takes 2.65 GB of the card's 3.63 GB usable VRAM, and two model-loading processes
OOM each other. Options: sequential load and unload, a hosted API for the
verifier, or the campus GPU. This is a decision to settle by measurement the way
the S3 retriever was, not something to code around.

## S6: Product UI (not started)

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

- [ ] Brainstorm and write the design spec
- [ ] Write the implementation plan
- [ ] FastAPI service exposing ask, and page image or crop retrieval
- [ ] **Load the models once at service startup and hold them in memory.** Each
      `vvrag search` invocation currently reloads ColQwen2 at about 20 s and
      2.6 GB. A request-scoped embedder would make the demo unusable and would
      OOM under any concurrency.
- [ ] Single-worker deployment, because the GPU is single-tenant on this
      hardware. Two workers means two model copies and an immediate OOM.
- [ ] Next.js app: question input, answer, page image with the region drawn
- [ ] Region overlay rendered from normalized 0-1 coordinates, so the frontend
      scales boxes to whatever size it displays the page at
- [ ] Abstain badge, visually distinct from a low-confidence answer
- [ ] Per-claim evidence, since one answer can carry several claims and regions

Note: `proposal.tex` does not name a UI framework, so this choice contradicts
nothing in the graded deliverable. Two internal design specs mention Streamlit
and are now superseded by this file.

## S7: Eval harness (not started)

**Objective.** Measure the system on SlideVQA and show whether the grounding and
verification layers earn their place.

**Why it exists.** It turns claims into evidence. Without it there is a system
that appears to work.

- [ ] Brainstorm and write the design spec
- [ ] Write the implementation plan
- [ ] SlideVQA loading. These are landscape slides, so a different patch grid
      than the project's A4 pages, which works only because S3 stores the
      geometry per page rather than assuming a constant.
- [ ] Auto-derived gold boxes from `derive.span_boxes`. This is why the
      over-covering union fixed in S2 was a correctness bug rather than an
      imprecision: it would have silently capped the reported IoU.
- [ ] Answer metrics, EM and F1. Accuracy is a control variable here, not the
      quantity being maximized.
- [ ] Grounding metrics: mean IoU, hit rate at IoU 0.25 and at 0.5
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
