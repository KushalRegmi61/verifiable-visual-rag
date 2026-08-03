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
tokens), and **1.0 ms per page** for CPU MaxSim.

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
duplicating it. Payload carries `doc_sha`, `page_no`, and `image_path`.

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

```
collection: pages
  size        : 128
  distance    : Cosine
  multivector : comparator = MAX_SIM
  hnsw_config : m = 0          (brute force, exact)
```

Verified server-side against the live cluster, not assumed from the create call.

`m=0` disables HNSW graph construction. This is deliberate: the corpus is small,
1 ms/page means exhaustive scoring is fast enough, and the evaluation in S7 needs
exact scores rather than approximate ones. It matches the proposal's description.

### 8.1 Pooled-prefetch rerank is deferred to S7

The proposal describes PLAID's compression and pruning stages mapped onto
Qdrant's pooled prefetch. S3 does not implement it, because at 1 ms/page a
300-page brute-force scan is about 0.3 s and the optimization would buy nothing
measurable.

Deferring it is the stronger position for the report: in S7 it becomes an
experiment with a measured before and after, rather than an unmeasured
optimization asserted to help.

### 8.2 Capacity

187 KB/page against Qdrant Cloud's free 1 GB is roughly **5,600 pages**.
Quantization is therefore unnecessary and is not implemented.

An earlier estimate in the S1/S2 spec put this at about 300 pages. That was
pessimistic by roughly 18x because it assumed ColQwen2's full patch budget rather
than the 747 vectors actually produced at the project's render DPI.

### 8.3 Configuration

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
- `test_core_is_light.py` passes with `qdrant_client` added.
- The full suite passes; `ruff check` and `ruff format --check` are clean.
- `uv.lock` pins the retrieval stack.

## 12. What S4 gets from this

A ranked page list with scores, and the query token vectors used to produce it.
S4's snap-to-box needs the query-to-patch similarity matrix, which is
`query_vectors @ page_vectors.T`, exactly the intermediate MaxSim already
computes. The patch grid geometry (747 vectors per page at a known render DPI)
is what maps a patch index back to a page region, so that the heatmap can rank
the candidate boxes S2 already stores.
