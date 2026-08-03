# Roadmap

Progress tracker for the seven slices. Each slice gets its own spec, plan, and
implementation cycle; this file is the index over all of them.

Last updated: 2026-08-03.

**Status: 3 of 7 slices complete.** S1, S2, and S3 are merged on `master`.
The written deliverables (proposal PDF, defense deck, research notes) are also
built. What remains is the project's actual research contribution: grounding,
verification, and the evaluation that measures whether they earn their place.

| # | Slice | Delivers | GPU | Depends on | Status |
|---|---|---|---|---|---|
| S1 | Skeleton + contracts | uv project, package, frozen contracts, test harness | No | - | Done |
| S2 | Ingest pipeline | PDF to page images, text-layer boxes, persistence, CLI | No | S1 | Done |
| S3 | Retrieval index | ColQwen2 embeddings, Qdrant multivector MaxSim | Batch | S2 | Done |
| S4 | Grounding core | `ground()`: text-span path, then visual snap-to-box | Query | S2, S3 | Not started |
| S5 | Reader + verifier | Atomic claims, independent judge, abstention gate | Yes | S4 | Not started |
| S6 | Product UI | Streamlit: answer, highlighted regions, abstain badges | Yes | S5 | Not started |
| S7 | Eval harness | SlideVQA, auto gold boxes, EM/F1 + IoU + confident-wrong, ablation | Yes | S4, S5 | Not started |

---

## S1: Skeleton and contracts (done)

Spec and plan: covered by the S1+S2 documents below.

- [x] uv project, src layout, hatchling build
- [x] Core package pinned to four dependencies (pydantic, pymupdf, pillow, numpy)
- [x] Frozen public contracts: `GroundedRegion`, `Claim`, `Answer`, `RetrievedPage`
- [x] Normalized 0-1 bbox convention, validated, used everywhere
- [x] Env-driven `Settings`, no hardcoded connection strings
- [x] `tests/test_core_is_light.py` enforces the dependency boundary in a subprocess

## S2: Ingest pipeline (done)

- Spec: `docs/superpowers/specs/2026-07-29-s1-s2-ingest-design.md`
- Plan: `docs/superpowers/plans/2026-07-29-s1-s2-ingest.md`

- [x] Born-digital gate (rejects encrypted, corrupt, empty, and OCR-less PDFs)
- [x] Page rendering at configurable DPI
- [x] Word-level candidate boxes with block/line/word hierarchy, plus table cells
- [x] Derived line, block, and span boxes computed at query time, not stored
- [x] SQLAlchemy 2.0 store with Alembic migrations
- [x] Resumable ingest, checkpointed per page, keyed on content hash
- [x] `vvrag ingest`, `vvrag status`, `vvrag inspect --find --overlay`
- [x] Coordinates verified against rendered ink, not arithmetic

## S3: Retrieval index (done)

- Spec: `docs/superpowers/specs/2026-08-03-s3-retrieval-design.md`
- Plan: `docs/superpowers/plans/2026-08-03-s3-retrieval.md`

- [x] Retriever chosen by measurement, not assumption (see spec section 2)
- [x] `PatchGrid`: model vector index to normalized page rectangle
- [x] `EmbedProvenance` with refusal on mismatch
- [x] Row and column mean pooling, stored now for the S7 rerank experiment
- [x] `ColQwen2Embedder`, vision tower left unquantized
- [x] `QdrantIndex`: three named vectors, MAX_SIM, HNSW m=0, schema verification
- [x] Resumable per-page embedding pipeline
- [x] `vvrag embed`, `vvrag search`
- [x] Known-item retrieval tests against the real corpus
- [x] Full corpus indexed: 74 pages across 4 documents

Measured on the target hardware (GTX 1650, 4 GB): 21.4 s per page to embed,
197 ms to embed a query, 1 ms per page for CPU MaxSim, 187 KB per page stored.

---

## S4: Grounding core (not started)

The project's central claim. Everything above is infrastructure; this is the
contribution. The heatmap RANKS candidate boxes that already exist in the text
layer. It never draws one.

- [ ] Brainstorm and write the design spec
- [ ] Write the implementation plan
- [ ] Text-span path: exact substring match to stored word boxes (the reliability floor)
- [ ] Query-to-patch similarity matrix from the retrieval intermediate
- [ ] Patch index to page rect via the stored `PatchGrid` geometry
- [ ] Exclude special tokens before any argmax (they map to no page region)
- [ ] Rank S2's candidate boxes by heatmap mass, select, do not draw
- [ ] `ground()` returns `GroundedRegion` for both modalities uniformly
- [ ] Verify selected boxes against rendered ink, as S2 does
- [ ] `vvrag ground "<question>"` with an overlay, the visual proof it works

Inputs already in place: the patch grid is stored per page in the Qdrant payload
(`n_patches_x`, `n_patches_y`, `patch_offset`, `n_special_tokens`), and the
candidate boxes are in SQLite. No re-embedding is needed.

Open question: which candidate granularity to rank (word, line, block, or table
cell). S2 stores words and derives the rest precisely so this can be retuned
without re-ingesting.

## S5: Reader and verifier (not started)

- [ ] Brainstorm and write the design spec
- [ ] Write the implementation plan
- [ ] **Decide the compute path** (see the blocker below)
- [ ] Reader VLM: answer from the retrieved page plus grounded regions
- [ ] Decompose the answer into atomic claims
- [ ] Verifier VLM, deliberately a DIFFERENT model (self-preference bias)
- [ ] Four-label rubric feeding a confidence score
- [ ] Abstention gate with a tunable threshold
- [ ] `verify()` takes data (image and boxes), never a client handle

**Blocker, unresolved.** The design needs two different VLMs. ColQwen2 alone
already takes 2.65 GB of the card's 3.63 GB usable VRAM, and two model-loading
processes OOM each other. Options: sequential load and unload, a hosted API for
the verifier, or the campus GPU. This is a decision, not something to code
around, and it should be settled by measurement the way the S3 retriever was.

## S6: Product UI (not started)

- [ ] Brainstorm and write the design spec
- [ ] Write the implementation plan
- [ ] Streamlit app: question in, answer with highlighted regions out
- [ ] Abstain badge when the verifier withholds
- [ ] **Hold the model in memory across requests.** Each `vvrag search` currently
      reloads ColQwen2: about 20 s and 2.6 GB. Constructing an embedder per
      request would make the demo unusable and OOM under any concurrency.

## S7: Eval harness (not started)

- [ ] Brainstorm and write the design spec
- [ ] Write the implementation plan
- [ ] SlideVQA loading (landscape pages, so a different patch grid than A4; the
      geometry is already stored per page, which is why that works)
- [ ] Auto-derived gold boxes from `derive.span_boxes`
- [ ] Answer metrics: EM and F1
- [ ] Grounding metrics: mean IoU, hit rate at IoU 0.25 and 0.5
- [ ] The headline metric: confident-wrong rate against coverage
- [ ] Three-way ablation: Baseline, Grounded, Verified
- [ ] Pooled-prefetch rerank as a measured experiment (vectors already stored in S3)

Note that accuracy is a control variable in this project, not the thing being
maximized. The claim is about confident-wrong rate and coverage.

---

## Baseline measured so far

From three paraphrased queries against the 74-page corpus, run 2026-08-03.
Recorded because S7's ablation needs an honest starting point.

- "how is the system evaluated and which metrics are used" returned the correct
  evaluation-metrics page at rank 1 and the confident-wrong formula at rank 3.
- "why is the heatmap not treated as faithful evidence" and "what is snap to box
  grounding" both returned the proposal's SUMMARY page at rank 1, which mentions
  every concept once and matches everything moderately.

Retrieval works and is mediocre on conceptual queries. That is the baseline S4
and S5 have to improve on, and it is more useful than a perfect score would have
been, because a perfect baseline leaves an ablation with nothing to show.
