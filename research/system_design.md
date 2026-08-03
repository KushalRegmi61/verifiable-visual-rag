# System Design: Verifiable Visual RAG (End-to-End Implementation)

Scope: how we actually build and run the product. Two pipelines (indexing, retrieval),
every component (frontend, backend, stores, model servers), and every LLM/VLM in the loop,
with self-host compute vs API and free-tier options called out. Evaluation and benchmarking
are deliberately out of scope here; this document is the running system only.

---

## 0. Design Principles (the rules we build against)

1. **Offline vs online split is hard.** Anything expensive and reusable (page rendering, box extraction, embeddings, indexing) happens once at ingest. The query path only does retrieval, reading, grounding, verification. This is the standard RAG separation and it keeps per-query latency and cost bounded.
2. **The grounding + verification + abstention core is a pure library**, UI-agnostic and store-agnostic. The web backend calls it; a CLI can call it; the eval harness (out of scope here) calls the same code. No business logic leaks into the API layer.
3. **Every model sits behind a thin interface** (`Retriever`, `Reader`, `Verifier`) so we can swap self-hosted for API without touching pipeline code. Reader and Verifier must be *different* models (self-preference bias), so this indirection is not optional.
4. **Born-digital only.** Text layer via PyMuPDF, no OCR engine. This is a hard scope boundary that removes a whole failure surface.
5. **Free-tier first, GPU only where it pays.** The only components that genuinely want a GPU are the visual retriever (embedding) and, optionally, a self-hosted reader/verifier. Everything else runs on a free-tier CPU box.

---

## 1. Component Inventory (the whole system on one page)

| Layer | Component | Job | Default choice |
|---|---|---|---|
| Frontend | Web UI | Upload PDF, ask question, render page image + highlighted box + answer/abstain | Next.js (or Streamlit for the fast prototype) |
| Backend | API service | Orchestrates ingest + query, exposes REST/WebSocket | FastAPI (Python) |
| Backend | Core library | Grounding, verification, abstention logic | Pure Python package, no web deps |
| Backend | Task/queue (ingest) | Async ingest so upload does not block | FastAPI `BackgroundTasks` → Celery/RQ only if needed |
| Model server | Visual retriever | Embed page images, produce multi-vector + heatmap | ColQwen2 on vLLM (GPU) |
| Model server | Reader VLM | Read pages, draft answer, atomic claims | API (Gemini free tier) or self-host Qwen2.5-VL |
| Model server | Verifier LLM/VLM | Judge claim vs region, 4-label rubric | Different model from reader (API or self-host) |
| Data store | Vector DB | Store page multi-vectors, MaxSim search | Qdrant (multivector native) |
| Data store | Blob store | Page images (PNG/WebP) + original PDFs | Local disk / S3-compatible (Cloudflare R2 free tier) |
| Data store | Metadata DB | Doc/page/box records, job status | Postgres (Supabase/Neon free tier) or SQLite for prototype |

Nothing here requires a paid tier to stand up a working demo. The single real cost is GPU time for retriever embedding, addressed in §7.

---

## 2. Models and VLMs Required (self-host vs API, free tiers)

We need exactly **three model roles**. Do not conflate them.

### 2.1 Visual Retriever (embedding model) — REQUIRED, GPU

- **Model:** ColQwen2 (ColPali family, Qwen2-VL backbone). Produces ~700 patch vectors per page (dynamic with image size) plus the query-to-patch similarity used later for snap-to-box.
- **Why self-host:** there is effectively no free hosted API that returns *multi-vector* ColPali embeddings; you must run the model to get the patch vectors and the heatmap. This is the one component you cannot API your way out of.
- **Compute:** ColQwen2 is a ~2-3B-class VL model for embedding. Fits in **8-12 GB VRAM** at fp16 for batch-1 to small-batch embedding; comfortable on a 16 GB card, easy on 24 GB. This is much lighter than serving a 7B *generative* VLM.
- **Serving:** `vllm serve` or a small FastAPI wrapper around the HF model. Batch pages during ingest; at query time you embed one short text query (cheap).
- **Free/cheap GPU:** Google Colab (free T4 16 GB) or Kaggle (free T4 x2, 30 hrs/week) for batch ingest of the corpus; a serverless GPU (Modal, RunPod, Beam) for on-demand query-time query-embedding, or just keep it warm on Kaggle during a demo. For a fixed demo corpus you can even pre-embed everything offline and never run the retriever GPU live.

### 2.2 Reader VLM — REQUIRED, API-first

- **Job:** look at the top retrieved page images, draft the answer, split into atomic claims.
- **Recommended default: API, Gemini free tier.** Gemini Flash / Flash-Lite free tier as of 2026 is ~10-15 RPM, ~250k-1M TPM, ~1,500 requests/day, and **accepts image input on the free tier**. That is more than enough for a prototype and demo. No card required to start.
- **Alternate APIs:** OpenRouter exposes several `:free` VLM endpoints (rate-limited) for redundancy; keep the interface generic.
- **Self-host option:** Qwen2.5-VL-7B-Instruct on vLLM. Needs a **24 GB GPU** (RTX 4090 / A100), or ~10-12 GB with AWQ/GPTQ 4-bit quant. Use this only if we want a fully offline, no-API demo, or hit free-tier limits.

### 2.3 Verifier (Judge) — REQUIRED, must differ from reader

- **Job:** given (claim, evidence region text/crop), return supported / partially supported / unsupported / insufficient evidence, plus a confidence used by the abstention gate.
- **Text-region claims** (the reliability floor) only need the span text, so the verifier can be a **text LLM** here: a small self-hosted model (Qwen2.5-7B / Llama-3.1-8B on vLLM, or even a 3B) is plenty for a rubric-constrained judgment, and runs on the same 24 GB box or quantized on 12 GB.
- **Visual-region claims** (chart/figure crops) want a VLM judge: use the *other* API vendor from the reader (e.g., reader = Gemini, verifier = an OpenRouter free VLM, or a self-hosted Qwen2-VL) so the judge is genuinely independent.
- **Free tier:** same Gemini/OpenRouter free tiers apply; just pin a different model id than the reader.

**Rule of thumb for the defense:** three roles, two of them can be free-tier APIs, one (retriever) needs a GPU but a small one, and even that can be pre-computed offline for a fixed demo corpus.

---

## 3. Indexing Pipeline (offline, run once per document)

Triggered on upload or as a batch job over the demo corpus. Output is a searchable, grounded index. Idempotent and resumable per page.

```
PDF in
  |
  v
[1] Validate + fingerprint      -> reject scanned/no-text-layer PDFs (born-digital gate)
  |                                 hash = dedup + cache key
  v
[2] Render pages -> images      -> PyMuPDF page.get_pixmap() at fixed DPI (e.g. 150)
  |                                 store PNG/WebP in blob store, path in metadata DB
  v
[3] Extract text-layer boxes    -> PyMuPDF words/lines/blocks + table/layout boxes
  |                                 each box: page, bbox (normalized), text, type(word/line/table)
  |                                 THIS is the candidate-box set that snap-to-box will pick from
  v
[4] Embed pages (ColQwen2)      -> ~700 multi-vectors per page (GPU)
  |                                 also cache the patch grid geometry for heatmap->box mapping
  v
[5] Upsert to vector DB         -> Qdrant multivector collection (MaxSim)
  |                                 payload: doc_id, page_no, image_path
  v
[6] Mark page indexed           -> metadata DB status; resume skips done pages
```

Key engineering decisions:

- **Fixed render DPI** so pixel bboxes and heatmap patch geometry stay consistent between ingest and display. Store boxes **normalized** (0-1) so the frontend can scale to any display size.
- **Boxes and embeddings are computed together** but stored separately: boxes in the metadata DB (queried by page), multi-vectors in the vector DB. Grounding needs boxes, retrieval needs vectors; do not stuff boxes into vector payloads beyond `image_path` + ids.
- **Batching for the GPU:** page embedding is the expensive step. Batch across pages, run on Colab/Kaggle/Modal, checkpoint after each doc. For a small fixed corpus this whole pipeline runs once and is done.
- **Two-vector-per-page optimization (scale trick, optional):** alongside the full multi-vector, store a single *mean-pooled* vector per page. At query time, prefetch top-K pages by the cheap pooled vector, then rerank only those K with full MaxSim. This is the standard ColPali-at-scale pattern and cuts query latency and memory a lot. For a minor-project-size corpus you can skip it and MaxSim everything; keep it in the back pocket.

---

## 4. Retrieval / Query Pipeline (online, per question)

This is the live request path. Target: one question in, one grounded-and-verified answer (or abstention) out.

```
User question (+ doc scope)
  |
  v
[1] Embed query (ColQwen2)      -> query token vectors  (cheap; 1 short text)
  |
  v
[2] Retrieve pages (Qdrant)     -> MaxSim over multivectors, top-N pages
  |                                 (optional: pooled-vector prefetch -> multivector rerank)
  v
[3] Read (Reader VLM)           -> fetch top-N page images from blob store,
  |                                 draft answer, split into ATOMIC CLAIMS
  v
[4] Ground each claim           -> for each claim:
  |     text path:  find exact span in page text layer -> its bbox (IoU~1.0, faithful)
  |     visual path: use query-to-patch heatmap to RANK candidate boxes from step [3] of
  |                  indexing, SNAP to best box. Never draw a box from pixels.
  v
[5] Verify each (claim, region) -> independent Judge, 4-label rubric + confidence
  |
  v
[6] Abstention gate             -> confidence >= threshold ? show claim + highlight
  |                                                         : withhold / "insufficient evidence"
  v
Answer with per-claim highlighted regions, or abstention
```

Latency budget (rough, prototype): query-embed <100 ms, retrieval <100 ms on small corpus, reader 1-3 s (API), grounding <50 ms (local box math), verify 0.5-2 s per claim (API), so a few seconds end-to-end. Stream partial results over WebSocket so the UI shows the answer, then boxes, then verification badges as they land.

The heatmap-to-box mapping (step 4 visual path) is the one piece of custom math: map the retriever's patch grid coordinates back to page pixel coordinates (using the geometry cached at ingest step [4]), score each candidate box by the patch similarity mass falling inside it, pick the argmax. This is snap-to-box; it is training-free and lives entirely in the core library.

---

## 5. Data Stores

### 5.1 Vector DB — Qdrant

- **Why:** native multivector collections with a `MAX_SIM` comparator, which is exactly ColBERT/ColPali late interaction. No pre/post-processing needed.
- **Config that matters:** multivector config with MaxSim; **HNSW is disabled for the multivector** (proximity graphs do not work with MaxSim), so multivector search is effectively brute-force over the candidate set; use `on_disk=True` to keep RAM down; add the optional pooled single-vector as a *separate* named vector *with* HNSW for the prefetch stage.
- **Free tier:** Qdrant Cloud has a free 1 GB cluster (fine for a demo corpus), or run the Qdrant Docker container locally for zero cost. Local Docker is the recommended prototype default.
- **Scale note for honesty:** ~700 vectors/page means a few hundred pages is ~hundreds of thousands of vectors. Fine locally; the pooled-prefetch trick is what you would reach for beyond that.

### 5.2 Blob store — page images + PDFs

- Local filesystem for the prototype; Cloudflare R2 or Backblaze B2 (S3-compatible, free tier) if we deploy. Frontend loads the page image and overlays normalized boxes.

### 5.3 Metadata DB — Postgres or SQLite

- Tables: `documents` (id, hash, status), `pages` (doc_id, page_no, image_path, patch_geometry), `boxes` (page_id, bbox, text, type), `jobs` (ingest status).
- **Free tier:** Supabase or Neon (managed Postgres, free tier) for a deployed app; SQLite file for local. The box table is the candidate set snap-to-box reads at query time.

---

## 6. Backend Architecture

```
                 +------------------------------------------+
                 |            FastAPI service               |
                 |                                          |
  Frontend  <--> |  /ingest  (async, BackgroundTasks)       |
  (REST +        |  /query   (WebSocket stream)             |
  WebSocket)     |  /status                                 |
                 |                                          |
                 |   calls -->  core library (pure Python)  |
                 |               - Retriever iface          |
                 |               - Reader iface             |
                 |               - Grounder (snap-to-box)   |
                 |               - Verifier iface           |
                 |               - Abstention gate          |
                 +------------------------------------------+
                     |            |             |
                     v            v             v
                  Qdrant     Blob store   Model servers
                             + Metadata   (retriever GPU,
                                DB         reader/verifier API)
```

- **FastAPI** because the whole ML stack is Python; no cross-language bridge, async support for streaming, trivial to containerize.
- **Interfaces over implementations:** `Retriever`, `Reader`, `Verifier` are protocols. A `.env` flag picks `GeminiReader` vs `VllmReader`, `QdrantRetriever`, etc. Swapping a self-hosted model for an API is a config change.
- **Async ingest:** upload returns a job id immediately; `BackgroundTasks` runs the indexing pipeline; the UI polls `/status`. Move to Celery/RQ + Redis only if we batch large corpora (probably unnecessary at project scale).
- **The core library never imports FastAPI, Qdrant, or any vendor SDK directly** beyond the interface adapters. That is what lets the same code back the app and (later, out of scope) the eval harness.

---

## 7. Deployment Topology and Compute Envelope

### 7.0 Available Compute: college 64 GB GPU (shared, time-limited)

We have access to a **college GPU with 64 GB VRAM**, but the allocation is **shared and time-limited**: other students also use it, so we cannot assume it is available on demand or as an always-on serving host. Treat it as a **batch window**, not a live service.

What 64 GB buys us when we do have the window:
- It comfortably co-locates **all three models at once** with room to spare: ColQwen2 retriever (~8-12 GB) + Qwen2.5-VL-7B reader at fp16 (~18-20 GB) + a 7B text/VL verifier (~16 GB) still fits inside 64 GB without quantization. So the "fully self-hosted, no external API" story (Stack B) is genuinely feasible here, not just a quantized squeeze.
- Because access is bursty, the **right use of the window is batch work, not hosting**: run corpus ingest (page embedding, the one heavy step) and any offline generation there, checkpoint per document, and push results to Qdrant. Do not architect the live demo around the GPU being up.

Design consequence: keep the system **decoupled from GPU availability**. Pre-compute embeddings during an allocated slot; the live query path then needs only a light query-embed (CPU-tolerable or a short GPU touch) plus the reader/verifier, which fall back to free APIs when the GPU window is closed. The `Vllm*` adapters point at the college GPU when it is up; the API adapters cover every other time.

Two viable stacks follow. Both keep a working demo at (near) zero recurring cost, and both can borrow the college GPU for the batch ingest step.

### Stack A — Zero-GPU-at-runtime (recommended for the demo)
- Pre-embed the fixed demo corpus offline on the **college 64 GB GPU during an allocated slot** (or Colab/Kaggle free GPU as fallback); push vectors to Qdrant.
- Runtime: FastAPI + Qdrant (Docker) + SQLite/Postgres on a **free CPU host** (Render/Railway/Fly free tier, or a laptop).
- Reader + Verifier = **Gemini free tier + one OpenRouter free VLM** (two vendors = independent judge for free).
- Query-time retriever: embed the short *query* only. This is light; run it on the CPU host (slow-ish but fine) or a tiny serverless GPU call. For a truly fixed demo, you can even restrict to pre-computable query sets.
- **Cost: ~$0.** Limits: API RPM/RPD caps, fine for a live demo.

### Stack B — Fully self-hosted (no external API), on the college 64 GB GPU
- During an allocated GPU slot, one **64 GB college GPU** runs all three at once via vLLM, no quantization needed:
  - ColQwen2 retriever (~8-12 GB), plus
  - Qwen2.5-VL-7B reader at fp16 (~18-20 GB), plus
  - a 7B text/VL verifier (~16 GB).
  - Total well under 64 GB, leaving headroom for KV cache and batching.
- This is the "runs with zero external calls" demo. **Caveat:** the GPU is shared and time-limited, so treat Stack B as something we can *show during a booked slot*, not a service that is up 24/7. Outside the slot, the same interfaces fall back to the free-API path.
- If we ever need this off-campus, a rented 24 GB GPU (RunPod/Vast, ~$0.3-0.8/hr spot) runs the same stack with 4-bit AWQ quantization instead.

**Recommendation:** build on Stack A as the default runtime, and use the college GPU for two things: (1) batch corpus embedding during allocated slots, and (2) demonstrating the fully-offline Stack B during a booked slot. The reader/verifier-must-differ constraint is satisfied for free by two API vendors in Stack A, and by two distinct local models in Stack B. Keep the `Vllm*` adapters implemented so switching between them is a config flip and never depends on the GPU being available at demo time.

---

## 8. Frontend

- **Prototype:** Streamlit. Upload widget, text box, and an image canvas that draws the returned normalized boxes over the page image. Ships in a day, good enough to demo grounding + abstention visually.
- **Product:** Next.js + a canvas/SVG overlay. Two-pane reader (page left, answer right), per-claim highlight-on-hover, a clear "insufficient evidence" state, and streamed verification badges (verifying -> supported/abstained). This is the version that sells the "verifiable" story visually.
- Frontend only ever consumes normalized bboxes + image URLs + answer JSON; it holds no ML logic.

---

## 9. Recommended Concrete Stack (what I would build Monday)

- **Frontend:** Streamlit now, Next.js if time allows.
- **Backend:** FastAPI + a `core/` Python package (interfaces + snap-to-box + abstention).
- **Vector DB:** Qdrant in Docker (local), multivector MaxSim + optional pooled prefetch.
- **Blob + metadata:** local disk + SQLite for the prototype; R2 + Neon if deployed.
- **Retriever:** ColQwen2 on Kaggle/Colab for batch ingest; pre-embed the demo corpus.
- **Reader:** Gemini Flash free tier (image input, ~1,500 req/day).
- **Verifier:** a *different* free VLM (OpenRouter `:free`) for visual claims; a small local/text model for text-span claims.
- **Fallback path:** every model interface also has a vLLM self-host adapter that targets the **college 64 GB GPU** for the fully-offline demo.
- **College GPU usage:** batch corpus embedding during allocated slots, plus one booked slot to demo the fully-offline Stack B. Do not make the live runtime depend on it, since access is shared and time-limited.

Bottom line: exactly one component (the visual retriever) truly needs a GPU, and even that is a small model whose output can be pre-computed offline during a college-GPU slot. The two generative roles run on free API tiers while still satisfying the independent-judge constraint. The whole running system fits in free tiers for a minor-project demo, and the 64 GB college GPU gives us a clean fully-self-hosted path (all three models at once, no quantization) to show it running with zero external calls, as long as we schedule around the shared allocation rather than assuming the GPU is always up.

---

## Sources

- Qdrant multivector / ColPali / MaxSim, ~700 vectors per ColQwen page, HNSW-disabled multivector, pooled-prefetch pattern: [Qdrant multivectors](https://qdrant.tech/documentation/tutorials-search-engineering/using-multivector-representations/), [Qdrant PDF retrieval at scale](https://qdrant.tech/documentation/tutorials-search-engineering/pdf-retrieval-at-scale/), [Qdrant + ColPali blog](https://qdrant.tech/blog/qdrant-colpali/)
- Gemini API free tier limits (Flash/Flash-Lite, RPM/RPD, image input on free tier, Pro removed 2026): [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits), [free tier guide](https://tokenmix.ai/blog/gemini-api-free-tier-limits)
- Qwen2-VL / Qwen2.5-VL 7B VRAM and vLLM quantization: [Qwen2.5-VL-7B HW discussion](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/discussions/18), [Qwen VRAM requirements](https://gigagpu.com/qwen-vram-requirements/), [vLLM Qwen deployment](https://qwen.readthedocs.io/en/v2.5/deployment/vllm.html)
