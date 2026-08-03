# S3: Retrieval Index

Design for slice S3 of Verifiable Visual RAG: page embeddings from a
late-interaction visual retriever, stored in Qdrant as multivectors and scored
by MaxSim.

Status: approved, not yet implemented.
Depends on: S1 (package skeleton), S2 (ingest pipeline), both merged.

## 1. What this slice delivers

An indexed corpus and a query path over it.

After S3, a document that has been through `vvrag ingest` can be embedded with
`vvrag embed`, and `vvrag search "<question>"` returns ranked pages. That is the
whole slice. Grounding, reading, and verification are S4 and S5.

Explicitly out of scope: box selection, any reader or verifier model, the
Streamlit UI, and the evaluation harness.

## 2. Hardware findings that constrain the design

Every number here was measured on the target machine (GTX 1650, 4 GB VRAM,
compute capability 7.5; 8 CPU cores; 15 GB RAM) against the 74 real page images
produced by the S2 ingest of `proposal.tex`. The benchmark harness lives outside
the repo; its results are reproduced here because they are the justification for
most decisions below.

| variant | top-1 | MRR | peak VRAM | s/page | patches | KB/page |
|---|---|---|---|---|---|---|
| **colqwen2-4bit-skipvis** | **1.00** | **1.00** | 2.65 GB | 21.4 | 747 | 187 |
| colsmol-500m | 1.00 | 1.00 | 2.56 GB | 10.0 | 871 | 218 |
| colsmol-256m | 1.00 | 1.00 | 2.11 GB | 8.3 | 871 | 218 |
| colqwen2-fp16 | - | - | OOM | - | - | - |
| colqwen2-4bit (blanket) | 0.00 | 0.15 | 1.92 GB | 25.0 | 747 | 187 |

Query-side, for the chosen variant: **197 ms** to embed a query (21 query
tokens), and **1.0 ms per page** for CPU MaxSim in numpy. Note that the CPU
MaxSim figure is a local measurement, not Qdrant server-side latency; any
statement about full-corpus scan time derived from it is an extrapolation and is
labelled as such below.

### 2.1 Patch grid geometry

The 747 vectors per page are **not an opaque count**:

```
747  =  23 x 32 grid  (736 image patches)  +  11 special tokens
```

Measured via `ColQwen2Processor.get_n_patches(image_size, spatial_merge_size=2)`
on the project's own A4 pages rendered at 150 dpi (1241 x 1754 px).

Two properties matter and both are load-bearing:

**The grid is not square, and it is dynamic.** It is derived by `smart_resize`
from the page's aspect ratio, so a landscape slide produces different dimensions
than an A4 portrait page. SlideVQA is landscape and this project's own documents
are portrait, so a single corpus will contain multiple grid shapes. There is no
global constant to hardcode.

**11 of the 747 vectors are not image patches.** They are special tokens
(instruction prefix and similar). They participate in MaxSim scoring correctly,
but they correspond to no region of the page.

Consequently `(n_patches_x, n_patches_y)` and the count of special tokens must
be **stored per page at embed time** (see 6.2). They cannot be recovered later
without re-running the model at 21.4 s/page, and S4's snap-to-box is impossible
without them: mapping a patch index to a page region requires the grid, and
mapping a *special token* to a page region would fabricate a box with no causal
relationship to the evidence. That is the same failure class as the `span_box`
over-union corrected in S2.

Read the last row as a broken configuration, not a model result. See 4.1.

Three consequences shape everything below.

**The online path is cheap and the offline path is not.** Embedding a query
costs 197 ms and scanning 300 pages costs 0.3 ms per query token set; embedding
a page costs 21.4 s. The ratio is about 109x. So the query path can be
synchronous and simple, while the indexing path must be resumable, because a
300-page corpus takes roughly 1.8 hours and will be interrupted.

**Batching makes embedding slower.** 21.4 s/page at batch 1 versus 24.6 s/page
at batch 2. The card is memory-bound, so a larger batch adds pressure without
buying parallelism. Batch size 1 is therefore both the fast path and the simple
one, and it removes padding from the picture entirely (see 4.1).

**Brute force is sufficient at this corpus size.** 1 ms/page means exhaustive
MaxSim over 300 pages is about 0.3 s. Approximate search would save nothing
that matters. See section 8.

## 3. Retriever choice

**ColQwen2 (`vidore/colqwen2-v1.0`), 4-bit NF4, vision tower unquantized.**

This is `proposal.tex`'s stated default, so the report needs no substitution
paragraph, and it fits in 2.65 GB of a 3.63 GB usable card with roughly 1 GB of
headroom. The cost is throughput: 21.4 s/page against colSmol-500M's 10.0 s/page,
so a 300-page ingest is about 1.8 hours rather than 50 minutes. That is an
offline one-time cost against a corpus built once, and section 6 makes it
resumable.

colSmol-500M is retained in this document as a measured fallback rather than an
assumed one. If corpus size grows enough that 2x ingest time matters, the
substitution is a config change, and the numbers to justify it in the report are
already in the table above.

Note for anyone revisiting the earlier assumption: colSmol produces **871**
patch vectors per page against ColQwen2's 747. An earlier estimate that a
smaller model would yield a coarser heatmap, and therefore weaken snap-to-box in
S4, was wrong. SmolVLM's image splitting gives it more grid resolution, not less.

## 4. Embedder invariants

These four are correctness requirements. Each has a measured failure behind it,
and none of them announce themselves when violated.

### 4.1 Never blanket-quantize

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["visual", "custom_text_proj"],   # LOAD-BEARING
)
```

Without the skip list, `load_in_4bit` also quantizes the vision tower and the
projection head. Measured effect, on identical code and hardware: known-item
top-1 falls from **1.00 to 0.00** and MRR from 1.00 to 0.15.

The broken configuration emits no warning, raises nothing, produces no NaN or
Inf, returns the correct shape `(747, 128)`, and yields unit-normalized vectors
in a plausible range. It is numerically immaculate and completely useless. NF4
on a ViT destroys patch-level geometry while leaving the model superficially
healthy.

### 4.2 float16, never bfloat16

The GTX 1650 is Turing (sm_75) and has no native bf16. Every ColPali code sample
in the wild uses `torch.bfloat16`. Verified: fp16 embeddings on this card contain
no NaN or Inf and are unit-normalized, so fp16 is safe here; bf16 is not
available.

### 4.3 Batch size 1

Faster (21.4 s vs 24.6 s per page), lower peak VRAM (2.65 GB vs 2.82 GB), and it
means no padding exists, which makes 4.4 unfalsifiable rather than merely
handled.

### 4.4 Select embeddings by attention mask, never by prefix

```python
vecs = out[j][mask[j].bool()]      # correct
vecs = out[j, :n]                  # WRONG for Qwen2-VL
```

Qwen2-VL's processor pads on the **left** (it is built for generation);
Idefics3 pads right. Prefix slicing is therefore silently correct for colSmol
and silently wrong for ColQwen2, which is exactly the kind of asymmetry that
makes a benchmark lie. This applies to query embedding as well, and matters more
there: MaxSim *sums* over query tokens, so an unmasked pad token contributes a
spurious maximum to every page's score.

### 4.5 Assert the adapter loaded

```python
if any("lora" in k or "custom_text_proj" in k for k in missing_keys):
    raise RuntimeError(...)
```

With `colpali-engine` 0.3.17 on `transformers` 5.x, the ColQwen2 adapter keys do
not match the renamed submodule path (`model.layers` became
`language_model.layers`). Every LoRA weight and the projection head are then
**newly initialized at random**, and transformers reports this as an informational
table rather than an error. Measured result: top-1 0.125, indistinguishable from
chance, from a model that loaded "successfully".

## 5. Dependency stack and the core boundary

### 5.1 Pins

```toml
[project.optional-dependencies]
retrieval = [
  "colpali-engine==0.3.10",
  "transformers>=4.51,<4.52",
  "torch==2.6.0",
  "torchvision==0.21.0",
  "qdrant-client>=1.12",
]
```

Exact pins, with a comment in `pyproject.toml` explaining why. Four combinations
were tried and three fail:

| stack | outcome |
|---|---|
| ce 0.3.17 + tf 5.14 | imports fine, **silently randomizes the adapter** |
| ce 0.3.17 + tf 4.57 | `ImportError: ModernVBertModel` |
| ce 0.3.10 + tf 4.51 + torch 2.11 / tv 0.26 | `torchvision::nms does not exist` |
| ce 0.3.10 + tf 4.51 + torch 2.6 / tv 0.21 | works |

Two of the three failures are loud. The dangerous one is not. `uv.lock` is
therefore load-bearing: an unpinned resolve six weeks from now lands on
`transformers` 5 and degrades retrieval to chance while every existing test
still passes.

### 5.2 The boundary

```
src/visual_verify/retrieval/
  embedder.py    model loading, embed_pages(), embed_query()
  index.py       QdrantIndex: ensure_collection, upsert_page, existing_page_nos, search
  pipeline.py    embed_documents(): store -> embedder -> Qdrant, checkpointed per page
```

The core package stays at its four dependencies. `retrieval/` imports from the
core; the core never imports `retrieval/`. `tests/test_core_is_light.py` gains
`qdrant_client` on its forbidden list alongside `torch` and `transformers`.

This is why the boundary was built in S2. `vvrag ingest` must keep running on a
machine with no GPU and no torch, and S3 introduces a stack that pins torch to a
specific patch release.

`contracts.RetrievedPage` already exists with the right shape
(`doc_id`, `page`, `image_ref`, `text_layer`, `score`) and becomes the search
return type. No contract changes in this slice.

## 6. Indexing pipeline

### 6.1 Separate command, not a pipeline stage

`vvrag ingest` is unchanged and stays torch-free. A new `vvrag embed` reads
already-ingested pages from SQLite, embeds them, and upserts to Qdrant.

Folding embedding into ingest would put torch on the ingest path, break the
boundary in 5.2, and make ingest impossible without a GPU. The 109x cost
asymmetry between the two stages is itself the argument for separating them:
they have different hardware requirements, different runtimes, and different
failure modes.

### 6.2 Point identity

```python
# uuid5(NAMESPACE_DNS, "verifiable-visual-rag.pages"), fixed forever.
# Derived rather than random so it is reproducible and auditable. Changing it
# orphans every existing point, so it is a constant, not a setting.
POINT_NS = uuid.UUID("5ee1d73c-35dc-53bb-8bf7-94bd98b0b932")
point_id = uuid.uuid5(POINT_NS, f"{doc_sha}:{page_no}")
```

Deterministic, so re-embedding a page overwrites its point instead of
duplicating it.

### 6.2.1 Payload

The payload is not incidental metadata. Two groups of fields are required for
correctness, and omitting either forces a full 21.4 s/page re-embed to recover.

```python
{
  # identity
  "doc_sha": str, "page_no": int, "image_path": str,

  # GEOMETRY - required by S4 snap-to-box, unrecoverable without re-embedding
  "n_patches_x": int,        # 23 for this project's A4 pages
  "n_patches_y": int,        # 32
  "n_image_patches": int,    # 736 = x * y
  "n_special_tokens": int,   # 11; these map to NO page region

  # PROVENANCE - required to detect index/query embedder drift
  "model_id": str,           # "vidore/colqwen2-v1.0"
  "model_revision": str,     # resolved commit hash, not a branch name
  "quantization": str,       # "nf4-skipvis" | "none"
  "dtype": str,              # "float16"
  "render_dpi": int,         # the DPI the page PNG was rendered at
  "embed_version": int,      # bumped on any change to the embedding path
}
```

**Geometry.** See 2.1. The grid is dynamic per page, so it cannot be a constant
in code.

**Provenance.** Section 3 deliberately retains colSmol-500M as a swappable
fallback and 6.5 permits differing DPI across documents. Vectors from different
models, quantizations, or render DPIs are not comparable, but nothing about a
stored vector reveals which produced it. Mixing them yields silently wrong
rankings with no error. This is the single most commonly reported production
failure for retrieval systems: the indexing embedder changes, the query embedder
does not, and recall degrades unnoticed.

Therefore `search()` records the same provenance for its query embedder and
**refuses to run** when it does not match what the collection holds, rather than
returning plausible wrong results. `vvrag embed` refuses to add a page whose
provenance differs from the collection's existing points.

### 6.3 Qdrant is the single source of truth for embedding state

Resumption asks Qdrant which pages a document already has, rather than tracking
an `embedded` flag in SQLite.

A flag in SQLite would be a second source of truth that can disagree with the
vector store: a crash between the Qdrant upsert and the SQLite commit leaves the
two permanently inconsistent, and neither is obviously authoritative. Asking
Qdrant is one round trip per document, needs no Alembic migration, and cannot
desync from what is actually indexed.

This mirrors S2's rule that identity is content rather than bookkeeping.

### 6.4 Checkpointing

Upsert with `wait=True` after each page. At 21.4 s of embedding per page, a
sub-second network round trip to Qdrant Cloud is noise, so per-page durability
is effectively free, and a 1.8-hour ingest that dies at page 200 resumes at 201.

Same reasoning as S2's `sink.checkpoint()`, which was added after a crash was
found to roll back an entire document.

### 6.5 DPI coupling

Embeddings are only comparable if all pages were rendered at the same DPI. S2
already refuses to mix DPI within a document (`Sink.page_dpi`). S3 extends this:
`vvrag embed` refuses to index a corpus whose pages span multiple DPIs, because
the resulting patch grids would not be commensurable across documents.

## 7. Query path

```
query text
  -> embed_query()            197 ms, ~21 token vectors
  -> QdrantIndex.search(k)    MAX_SIM, brute force
  -> list[RetrievedPage]
```

Synchronous. At 197 ms plus a sub-second Qdrant round trip, the whole path is
well under a second before any reader model, so S6 needs no async worker, no
precomputation, and no caching layer.

## 8. Qdrant collection

Three **named** vectors in one collection, following the established production
pattern for ColPali-family retrieval:

```
collection: pages
  named vector "original"            128-dim, Cosine, MAX_SIM, hnsw m=0
  named vector "mean_pooling_rows"   128-dim, Cosine, MAX_SIM, hnsw m=0
  named vector "mean_pooling_cols"   128-dim, Cosine, MAX_SIM, hnsw m=0
```

`m=0` disables HNSW graph construction. This is necessary rather than merely
acceptable: building an HNSW graph over multivectors is combinatorially
expensive, on the order of tens of millions of vector comparisons per page
insertion at 20k-page scale, because every candidate comparison is itself a full
MaxSim. Brute force is the correct choice here, and S7's evaluation needs exact
scores rather than approximate ones in any case.

### 8.1 Why three vectors now, when only one is used in S3

**S3 writes all three but queries only `original`.** The pooled vectors are
computed at embed time and stored unused until S7.

This is deliberate. Qdrant cannot add a named vector to an existing collection
without recreating it, so the schema is a one-way door. The pooled
representations are derived from `original` by averaging along each grid axis,
which requires the grid geometry from 2.1. Provisioning them now costs one
cheap arithmetic step per page against a 21.4 s embed, and avoids any
possibility of a later re-embed.

`mean_pooling_rows` and `mean_pooling_cols` reduce roughly 736 image patch
vectors to about 23 and 32 respectively, plus retained special tokens. Published
results for this pooling scheme report **NDCG@20 of 0.952 and about 13x faster
retrieval** versus full-resolution scoring. Max pooling was measured at NDCG@20
0.759 and is not viable, so mean pooling is specified explicitly rather than left
as an implementation choice.

### 8.2 Pooled-prefetch rerank as a query strategy is still deferred to S7

S3 stores the pooled vectors but does not use them at query time. The two-stage
prefetch-then-rerank query path lands in S7 as a measured experiment.

The reason is corpus size, not doubt about the technique. Extrapolating the
measured 1 ms/page CPU MaxSim, a 300-page exhaustive scan is on the order of
0.3 s, so a 13x speedup on a sub-second operation is not worth spending slice
budget on before there is something to measure it against. Treating it as an
S7 experiment with a real before and after is a stronger claim for the report
than an unmeasured optimization asserted to help.

What section 8.1 buys is that this remains a *query-path* change in S7, not a
re-index.

### 8.3 Capacity and why quantization is not used

187 KB/page against Qdrant Cloud's free 1 GB is roughly **5,600 pages** for the
`original` vectors, plus about 8 percent for the pooled ones. Quantization is
therefore unnecessary for storage and is not implemented.

Worth recording explicitly, because it is a common misconception: binary or
scalar quantization reduces memory but **does not reduce the number of vector
comparisons**, so it does not address multivector scaling. Pooling reduces
comparisons; quantization reduces bytes. They solve different problems, and only
the second one is a constraint this project has.

An earlier estimate in the S1/S2 spec put capacity at about 300 pages. That was
pessimistic by roughly 18x because it assumed ColQwen2's full patch budget rather
than the 747 vectors actually produced at this project's render DPI.

### 8.4 Configuration

```
VVRAG_QDRANT_URL=https://<cluster>.qdrant.io:6333
VVRAG_QDRANT_API_KEY=<key>
```

`VVRAG_QDRANT_URL` already exists in `config.py`; S3 adds the API key. Both come
from the environment, loaded from a gitignored `.env`. No module hardcodes a
connection or a credential. Local Docker Qdrant and Qdrant Cloud differ only by
these two variables.

## 9. Testing

**Known-item retrieval (marked `slow`).** A verbatim sentence taken from a page
must retrieve that page. This is the floor, not a quality measure: it is the
easiest possible retrieval task. It is also the only check that caught any of
the four bugs found while designing this slice, every one of which produced
correctly-shaped, numerically-healthy, entirely plausible output.

**Qdrant agrees with local numpy MaxSim.** Ranking must match exactly. A
collection with the wrong comparator or distance still accepts writes and still
returns results; it just returns the wrong ones. Measured agreement on the live
cluster: 1.00 across 7 queries.

**Fake embedder.** A deterministic stub implementing the embedder interface, so
pipeline logic, resumption, and CLI behaviour are testable in CI without a GPU
and without downloading 4 GB of weights. The GPU tests are `slow`; the logic
tests are not.

**Boundary test extended.** `qdrant_client` joins `torch` and `transformers` on
the forbidden list in `test_core_is_light.py`.

**Adapter-load assertion.** Unit test that a missing-key report raises.

**Grid geometry round-trip.** For a page whose stored payload says the grid is
`n_x by n_y`, assert `n_x * n_y + n_special == len(vectors)`. This is the cheap
invariant that catches a grid recorded from the wrong image, a changed
`spatial_merge_size`, or special tokens miscounted. Test on both a portrait and a
landscape page, since the grid is aspect-ratio dependent and a square-grid
assumption would pass on neither by accident.

**Provenance mismatch refuses.** Assert that querying a collection with an
embedder whose `model_id`, `quantization`, `dtype`, or `render_dpi` differs from
the stored points raises rather than returning results. This is the guard against
the silent index/query drift described in 6.2.1, and it must fail loudly because
its whole purpose is to prevent a plausible wrong answer.

**Pooled vectors are consistent with originals.** Assert that
`mean_pooling_rows` equals the row-wise mean of the image patches in `original`,
excluding special tokens. Written in S3 even though pooling is unused until S7,
because a silently wrong pooled vector would only surface as degraded rerank
quality much later and would be misattributed to the technique.

## 10. Operational note

The `pages` collection currently holds 8 points from design-time smoke testing.
`ensure_collection` must expose an explicit recreate path, and the collection
must be cleared before the first real corpus ingest, so that test points cannot
be mistaken for corpus data.

## 11. Definition of done

- `vvrag embed --all` indexes every ingested document, resumably, and is
  idempotent on a second run.
- `vvrag search "<question>" -k 5` returns ranked `RetrievedPage` values.
- Known-item retrieval passes on the real corpus.
- Qdrant ranking matches local numpy MaxSim exactly.
- Every stored point carries the full geometry and provenance payload of 6.2.1,
  verified by the round-trip invariant `n_x * n_y + n_special == len(vectors)` on
  both a portrait and a landscape page.
- All three named vectors are populated, and the pooled ones are verified against
  the originals even though S3 does not query them.
- A provenance mismatch between the query embedder and the collection raises
  rather than returning results.
- `test_core_is_light.py` passes with `qdrant_client` added.
- The full suite passes; `ruff check` and `ruff format --check` are clean.
- `uv.lock` pins the retrieval stack.

## 12. What S4 gets from this

A ranked page list with scores, the query token vectors used to produce it, and
the per-page patch grid geometry.

S4's snap-to-box needs the query-to-patch similarity matrix, which is
`query_vectors @ page_vectors.T`, exactly the intermediate that MaxSim already
computes. Turning a column of that matrix into a page region requires three
things from this slice, all of them stored per page in 6.2.1:

1. `n_patches_x`, `n_patches_y`, so patch index `i` maps to grid cell
   `(i % n_x, i // n_x)` and then to a normalized page rectangle. The grid is
   per page, not a global constant.
2. `n_special_tokens`, so the 11 non-image vectors are **excluded** before any
   argmax. A special token has no page region, and mapping one anyway would
   produce a confidently-drawn box with no causal link to the evidence.
3. `render_dpi`, so patch geometry and the S2 word boxes are expressed against
   the same page rect.

With those, the heatmap ranks the candidate boxes S2 already stores, which is
snap-to-box. Note what S4 does *not* need: any ability to draw a box from pixels.
The grid only ever selects among existing text-layer boxes, which is the
distinction the proposal's gap argument rests on.
