# S4: Grounding Core

Date: 2026-08-08
Status: design approved, not implemented
Depends on: S2 (candidate boxes), S3 (page embeddings, patch geometry)

## 1. What this slice delivers

`ground(claim, ...)` returns the specific region of a page that supports a claim,
as a `GroundedRegion` carrying page, bbox, score, and modality.

Two paths produce regions. The **text path** finds the claim string in the text
layer and returns the exact word-union rect, which is faithful by construction.
The **visual path** uses the query-to-patch similarity heatmap to *rank candidate
boxes that already exist in the text layer*, and selects one. It never draws a box
from pixels. That distinction is the project's pillar 2 and the whole reason the
slice exists.

Everything before this slice is infrastructure. A competent visual RAG pipeline
ends at S3. S4 is where the thesis becomes code.

## 2. Where grounding sits, and a correction

`proposal.tex` wires the online pipeline as `retrieve -> reader -> ground ->
verify` (lines 340 to 342), and line 355 states that "each claim is mapped to an
exact text span or a snapped visual region."

**Grounding runs after the reader, per claim, not before it.** An earlier internal
note in CLAUDE.md recorded the order as `box snap -> reader`, which was wrong and
has been corrected. The difference is not cosmetic. Grounding a claim gives the
text path an exact string to search for. Grounding a bare query would reduce that
path to keyword matching and would remove the reliability floor the proposal
depends on (line 288).

A useful consequence: **S4 is not blocked on S5.** `ground()` accepts a claim
string, so the CLI supplies one by hand until the reader exists, and the reader
later drops into the same seam with no contract change.

## 3. Measured constraints

Both findings below were measured on `proposal_report/proposal.pdf` at 150 dpi,
the same render the corpus is indexed at. Both constrain the design in ways that
are not visible from the code.

### 3.1 The patch grid is coarser than the candidate boxes

ColQwen2 gives a 23x32 grid for a portrait page: 736 cells, each 0.0435 by 0.0312
in normalized page coordinates.

The grid is not constant. The indexed corpus already contains a landscape page at
**34x19, 646 cells**, confirming that geometry must be read from the payload per
page rather than assumed. Every number below is for the portrait case; a landscape
page has finer horizontal and coarser vertical resolution, which shifts the
limits in 3.1 but does not remove them.

Measured against page 3:

| granularity | count | size vs one cell | sharing a patch row |
| --- | --- | --- | --- |
| word | 435 | 0.12 x 0.45 cells | 25.6 mean, 58 max |
| line | 61 | 0.92 x 0.45 cells | 3.6 mean, 6 max |
| block | 19 | 14.3 x 0.45 cells | 3.6 mean |

Thirty-two rows of vertical resolution against 61 lines of text. **The heatmap
cannot resolve a word, and cannot cleanly resolve a line.** It can rank blocks.

A design asserting that the heatmap picks the best word box would be proposing
something the signal cannot deliver, and it would have looked correct in code
while returning a plausible box every time. This is the same failure signature as
every other bug in this repository.

### 3.2 Candidate granularity caps IoU regardless of the selector

The proposal derives gold boxes automatically as the bounding box of the answer
string in the text layer (line 412), which is a span of a few words. Measured IoU
of a **perfect** selector, one that always picks the candidate containing the gold
span, against a 3-word gold span over pages 2 to 9:

| granularity | mean IoU | median | hit@0.25 | hit@0.5 |
| --- | --- | --- | --- | --- |
| block | 0.097 | 0.053 | 14.4% | 1.1% |
| line | 0.195 | 0.192 | 28.9% | 2.2% |
| random block | 0.004 to 0.009 | 0.000 | 0.0% | 0.0% |

`proposal.tex` line 452 lists indicative targets of 0.5 to 0.6 mean IoU and about
80% hit@0.25 for the visual path. **Neither granularity reaches that with a
perfect oracle.** This is not selector weakness. It is a units mismatch between
two sections of the proposal: gold is a 3-word span, while the smallest candidate
the heatmap can resolve is much larger.

The proposal is not wrong. Line 440 already labels those targets as "indicative
targets, which are reference points reported by related work on the BBox-DocVQA
setting and may differ on SlideVQA," and the cited 0.569 (arXiv:2512.02660) comes
from BBox-DocVQA, whose gold regions are far larger than a 3-word span. The risk
is presentational: reporting 0.15 against a stated target of 0.5 reads as failure
when it is in fact a near-ceiling result. Section 8 exists to fix that.

## 4. Candidate granularity

The visual path ranks **blocks** at stage 1 and **lines** at stage 2. The text
path returns **spans**, from `derive.span_boxes`.

Each signal ranks only what it can actually resolve. `GroundedRegion.modality`
already distinguishes the two, so the product layer and the eval harness treat
them uniformly without pretending they have equal precision.

Only word boxes are stored; `derive.py` computes lines, blocks, and spans as pure
functions. Granularity therefore stays retunable without re-ingesting, which is
what makes section 8's ceiling reporting cheap to produce for any granularity.

## 5. The `ground()` contract

```python
def ground(
    claim: str,
    boxes: list[BoxRecord],
    *,
    page: int,
    page_vectors: np.ndarray | None = None,
    query_vectors: np.ndarray | None = None,
    grid: PatchGrid | None = None,
    force: Literal["text", "visual"] | None = None,
) -> list[GroundedRegion]:
```

Routing:

| condition | result |
| --- | --- |
| `span_boxes(boxes, claim)` non-empty | text regions, `modality="text"`, no vectors touched |
| no text match, candidates exist | one visual region, `modality="visual"` |
| no text match, no candidates | `[]` |
| `force="visual"` | always the visual path, whatever the text path would have found |

`force="visual"` exists because the proposal requires it: line 440 specifies that
the visual path is "exercised by forcing snap-to-box on text-locatable questions
and scoring it against the same gold boxes, so that visual grounding is measured
rather than only demonstrated." Without this flag the visual path can only be
measured on questions that have no gold box.

Text wins by default because it is exact, and because short-circuiting skips the
Qdrant vector fetch on the common case.

### 5.1 Absence is not doubt

An empty list means **no evidence exists on this page**: no text match and no
candidate boxes, as on a scanned or image-only page. It never means "the evidence
looked weak."

`ground()` applies no confidence threshold. `proposal.tex` line 381 places
abstention on the verifier's output, and S4 owning a second threshold would make
the ablation unable to separate the two contributions. S4 locates evidence; S5
judges sufficiency.

Returning a best-effort box on a page with no text layer was rejected: a region
with no causal link to evidence is fabricated evidence, which is the exact failure
the project argues against.

## 6. Heatmap construction

Two per-patch quantities, with different jobs. The split is forced by measurement
(6.1), not by preference.

**Dense max-sim, `r(p) = max over q of (q . p)`, does the ranking.** Every one of
the 736 patches receives a score, so candidates can be compared at any
granularity. Candidates are scored by the **mean** of `r(p)` over the patches they
cover, not the sum, because the sum is monotone in area and would hand every
contest to the largest candidate.

**Contribution attribution explains the result.** For each query token, the patch
winning its MaxSim maximum receives that token's score, so contributions decompose
the page's own retrieval score and a region's share is the fraction of the ranking
score it accounts for. This is what the UI shows and what gets defended in a viva
("this block accounts for 9 of the 19 query tokens"). It is an honest
decomposition of the real objective rather than a number invented for display.

**Special tokens are excluded before any argmax**, using `grid.offset` and
`grid.n_image_patches` via the existing `PatchGrid.is_image_token`. A special
token maps to no page region. Mapping one anyway draws a confident box with no
causal link to the evidence, the same failure class as the over-covering
`span_box` fixed in S2.

This is not a theoretical precaution. Measured against indexed pages, **4 to 5 of
every 19 to 30 query tokens have their maximum on a special token**, so 16 to 26
percent of the query would map onto a fabricated rectangle if the filter were
missing. Because those tokens are dropped, region scores are a share of the
image-patch total, not of the full page score, and the two do not agree.

### 6.1 Why attribution cannot do the ranking

The first draft of this spec made contribution attribution the ranking rule.
Measurement against indexed pages killed that, and the numbers are the reason the
two quantities are now split:

| query | tokens | tokens landing on patches | **distinct patches lit** | share of grid |
| --- | --- | --- | --- | --- |
| "What is the abstention threshold?" | 19 | 14 | **4** | 0.54% |
| "How many patch vectors ... what grid?" | 30 | 26 | **14** | 1.90% |

Realistic questions run 14 to 30 tokens, and even a three-character query embeds
to 14 because of a fixed prompt prefix. Since at most one patch per query token
receives anything, and winners collide, **a typical query lights 4 to 14 of 736
patches**. At most four blocks can score non-zero, and inside a winning block
nearly every line scores exactly zero, so stage 2 would be picking among ties.

Attribution is therefore an explanation, not a ranking signal. Dense max-sim
scores all 736 patches and is what stage 1 and stage 2 both rank on.

The plan must still confirm the choice empirically on the oracle harness of
section 8, selecting by hit rate among:

1. **Dense mean** (the primary): `mean over patches in B of r(p)`.
2. **Dense sum**: the size-biased variant, as a control that shows the area bias
   is real rather than assumed.
3. **Attribution mass**: retained in the comparison to confirm on data that it
   underperforms, rather than resting on the sparsity argument alone.

Scoring is pure numpy over vectors already in Qdrant, so the whole comparison
costs no GPU time and no re-embedding. Choosing a scoring rule by argument when it
can be chosen by measurement is the mistake this repository has already made
twice, in the retriever quantization and the patch offset. This section is the
third instance, caught during design rather than after implementation.

Scoring is pure numpy over vectors already stored in Qdrant, so the comparison
costs no GPU time and no re-embedding. Choosing a scoring rule by argument when it
can be chosen by measurement is the mistake this repository has already made
twice, in the retriever quantization and the patch offset.

## 7. Two-stage snap

**Stage 1** ranks the page's blocks. This is the discrimination the heatmap
demonstrably supports.

**Stage 2** ranks the lines within the winning block, a choice among roughly 3
rather than among all 61 on the page.

**Ambiguity rule.** If the top two lines in stage 2 are within a configurable
relative margin, return the block instead and mark the region as block-resolution.
The region stays honest about its own resolution rather than committing to a line
the signal cannot distinguish.

Two stages, rather than ranking all 61 lines flat, because the ceiling is the same
(0.195) while the error is bounded. A stage-2 mistake still lands inside the
correct paragraph, whereas a flat miss can land anywhere on the page. The stages
also fail differently and can be reported separately, which section 8 uses.

## 8. Metrics: two numbers, not one

A single IoU conflates two failure modes with different owners:

- **Hit rate.** Did the selector pick the candidate containing the gold span?
  This is what the heatmap controls, and where snap-to-box either works or does
  not. The floor is the random-candidate baseline, measured between 0.004 and
  0.009 mean IoU depending on the sampling seed, with 0.0% hit@0.25 in every
  run. Quote it as "under 0.01", not as a single figure: it is a Monte Carlo
  estimate over which block is drawn, and the ceilings above are not, so only
  the floor moves between runs. `test_the_random_candidate_floor_is_near_zero`
  requires under 0.02, which is loose enough to survive reseeding and still far
  below any working selector.
- **IoU.** How tight is that candidate against gold? This is fixed by granularity
  and is capped at 0.195 for lines no matter how good the selector is.

**Every reported grounding number is accompanied by the oracle ceiling for its
granularity and by the random-candidate baseline.** "0.15 achieved against a 0.195
ceiling at line granularity, over a 0.004 random floor" states that the selector
is working. "0.15 against a target of 0.5" states that it failed. The system is
identical and only the first is accurate.

Hit rate against the random floor is also a cleaner demonstration of pillar 2 than
IoU is, because it isolates the contribution of the heatmap ranking from the
choice of candidate granularity.

### 8.1 Decision on the proposal's stated target

**`proposal.tex` is not edited.** Decided 2026-08-08, after considering a
clarifying sentence and a revision of the numbers.

Line 452's 0.5 to 0.6 target is a cited reference point from BBox-DocVQA
(arXiv:2512.02660), where gold regions are page areas. This project derives gold
automatically as the box of the answer string (line 412), which is a 3-word span.
When the gold box sits inside the predicted box, IoU reduces to the ratio of their
areas, so a tighter gold definition mechanically produces a lower IoU for
identically good grounding. 0.195 here and 0.569 there can describe the same
quality of system.

Nothing in the proposal is false: line 440 already labels those targets as
indicative reference points that "may differ on SlideVQA". The exposure is
presentational, and it is handled in the results chapter rather than by amending a
graded document after the fact.

**The results chapter must therefore state**, wherever visual grounding IoU
appears: the achieved value, the oracle ceiling for its granularity, the
random-candidate floor, and one sentence explaining that the 0.5 to 0.6 reference
comes from a benchmark with larger gold regions and is not directly comparable.
Omitting the ceiling turns a near-ceiling result into an apparent failure.

## 9. The core boundary

`ground()` receives `page_vectors`, `query_vectors`, and `grid` as arguments. It
does not fetch them.

This keeps the grounding module inside the core's four dependencies (pydantic,
pymupdf, pillow, numpy). Qdrant lives behind the retrieval extra and the embedder
needs a GPU, so a module that fetched its own inputs could not be tested without
both. As written, the entire ranking logic is testable with hand-built numpy
arrays: no cluster, no model load, no 21.4 s per page.

A thin adapter in the CLI performs the fetch, calling `QdrantIndex.get_vectors`
and `get_payload` and reconstructing the `PatchGrid` from the stored geometry.
`tests/test_core_is_light.py` enforces the boundary in a subprocess and will fail
if `grounding/` imports `qdrant_client` or `torch`.

This follows `derive.py`, which is pure over `BoxRecord` lists for the same
reason.

## 10. Error handling

| situation | behaviour |
| --- | --- |
| no text layer on the page | `[]` |
| claim absent from the page | `[]`, never a best-effort box |
| `page_vectors` length disagrees with `grid.n_vectors` | raise, following `pooling.py` |
| visual path requested with no vectors supplied | raise, rather than silently returning the text result |
| stage 2 ambiguous | return the block, marked block-resolution |

Silent truncation is the specific thing being avoided. A vector count mismatch
that gets clipped produces a shifted heatmap and a confidently wrong box, which no
assertion downstream would catch.

## 11. Testing strategy

The suite must attack plausible-but-wrong output, because that is the only failure
mode this slice has. A wrong region is a well-formed, normalized, positive-area
four-tuple.

**Selection is asserted with `evidence.covers_text`, never with `has_ink`.** All
435 word boxes on page 3 contain ink, so an ink-based assertion passes a random
selector 200 times out of 200 while it is correct once in 200. This is pinned by
`tests/test_evidence.py::test_ink_cannot_discriminate_the_wrong_box`.

Required tests:

1. **Oracle ceiling as a regression test.** Pins line granularity near 0.195 mean
   IoU. A later change to candidate derivation must not quietly move the ceiling
   out from under the reported numbers.
2. **Selector against the random baseline** over the same candidate set. At or
   below the 0.004 floor, the heatmap contributes nothing and snap-to-box is not
   working, whatever the IoU says.
3. **Special-token exclusion, proven by construction.** A query whose global
   argmax falls on a special token must produce no region from that token. Built
   with synthetic vectors so the condition is guaranteed rather than hoped for.
4. **Patch rectangles land on ink**, with the displaced control from
   `evidence.shift`. `patch_bbox` is new coordinate arithmetic and this repository
   has a history of silent coordinate bugs.
5. **Text path exactness.** A claim spanning a line break returns one rect per
   line, not a union. `derive.span_boxes` already guarantees this; the grounding
   layer must not re-union them, since a union over a wrapped match covers 5.7x
   the true ink area on the two-line fixture.
6. **Determinism.** The same claim, boxes, and vectors produce the same region.
   Ties broken by candidate order, not by dict iteration.
7. **Routing.** Text short-circuits without touching vectors; `force="visual"`
   bypasses the short-circuit.

## 12. CLI

`vvrag ground "<claim>" --doc <sha> --page <n> [--force-visual] [--overlay out.png]`

`--overlay` renders the selected region onto the page PNG. This is the picture of
the claim working, and it is the artifact for the defense.

## 13. What S5 and S7 get from this

S5 receives `list[GroundedRegion]` per claim, uniform across modalities, so the
verifier's 4-label rubric applies without knowing which path produced the region.

S7 receives `force="visual"`, the oracle ceiling harness, and the random-candidate
baseline, which are exactly the three things the grounding-overlap metric needs to
be interpretable.

## 14. Out of scope

No reader and no verifier. No confidence thresholding or abstention. No
free-form box drawing from pixels, in this slice or any later one. No re-embedding
and no schema change, since everything needed is already stored.
