# Recommended Stack: Optimal Models, Pricing, and Campus-GPU Compute Plan

Decision record for the Verifiable Visual RAG minor project. Model choices, live 2026 pricing,
and a compute plan that fully uses the shared 64 GB campus GPU with free-tier APIs as fallback.
Derived from a fact-checked deep-research pass (architecture claims 3-0 verified) plus direct
pricing/VRAM verification. Evaluation is out of scope; this is the running system only.

Confidence tags: [V] = verified with citation this pass; [A] = arithmetic/engineering estimate; [R] = risk/unverified, flagged.

---

## 1. The Optimal Named Stack (TL;DR)

| Role | Recommended model | Why this one | Where it runs |
|---|---|---|---|
| Visual retriever | **ColQwen2.5-v0.2** (~3B, ColPali family) | Current open-weight late-interaction retriever, strong ViDoRe scores, small enough for cheap batch embedding. Fallback: ColQwen2-v1.0 [V] | Campus GPU (batch) or free Colab/Kaggle |
| Reader VLM | **Gemini 2.5 Flash** (API, free tier) primary; **Qwen2.5-VL-7B-Instruct** self-host alt | Free-tier vision, generous limits; reference stacks even pair the retriever with a 3B reader, so 7B is comfortable headroom [V] | Free API, or campus GPU |
| Verifier (must differ from reader) | **Qwen2.5-VL-7B-Instruct** self-host, OR a *second* API vendor (GPT-5.4-mini / OpenRouter free) | Independent judge = different family/vendor from reader; 4-label rubric is a light task a 7B handles [A] | Campus GPU, or free/cheap API |
| Vector DB | **Qdrant** (multivector, MAX_SIM, HNSW m=0) | Native late-interaction MaxSim; brute-force exact; pooled-prefetch trick for scale [V] | Docker, local or free cloud tier |
| Ingestion | **PyMuPDF** (page render + text-layer boxes, no OCR) | Exact born-digital boxes; these ARE the snap-to-box candidate set | CPU, free |

**One-line pick:** ColQwen2.5 retriever + Gemini 2.5 Flash reader (free) + self-hosted Qwen2.5-VL-7B verifier + Qdrant. Reader and verifier are different families, so the independent-judge rule holds at ~$0.

---

## 2. Model Selection, Justified

### 2.1 Visual Retriever — ColQwen2.5-v0.2
- The open-weight late-interaction family on the ViDoRe leaderboard: ColPali v1.3, ColQwen2 v1.0, **ColQwen2.5 v0.2**, ColSmol, ModernVBERT. No OCR needed. [V]
- **Optimal for us: ColQwen2.5-v0.2** (~3B). Best quality-per-VRAM in the family for a small team; produces ~1024 patch vectors/page plus the query-to-patch similarity we reuse for snap-to-box. [V]
- **If we want max quality and have GPU room:** NVIDIA **nemotron-colembed-vl-8b-v2** is #1 on ViDoRe V3 (NDCG@10 63.42, open weights, 3B/4B/8B variants). Overkill for a minor project; note it as an upgrade path, default to ColQwen2.5. [V]
- **Reality check:** there is effectively no hosted API that returns multi-vector ColPali embeddings, so the retriever must run on our own GPU. This is the single unavoidable GPU dependency. [A]

### 2.2 Reader VLM — Gemini 2.5 Flash (free) primary
- Reference Vidore stacks pair the retriever with a **Qwen2.5-VL-3B-Instruct** reader, which tells us a small reader is sufficient; 7B is headroom, not a requirement. [V]
- **Primary: Gemini 2.5 Flash on the free tier** (vision input included, generous RPM/RPD). Zero cost, no GPU. [V]
- **Self-host alt: Qwen2.5-VL-7B-Instruct** on the campus GPU for the fully-offline demo. [A]

### 2.3 Verifier — different family from the reader
- Hard rule from the proposal: verifier must differ from reader to avoid self-preference bias. The research pass found no single "best verifier" citation [R], so we choose by the independence constraint, not a benchmark.
- **If reader = Gemini (API): verifier = self-hosted Qwen2.5-VL-7B** on campus GPU (different family, and gives us a local judge for the offline story).
- **If reader = self-hosted Qwen2.5-VL: verifier = an API vendor** (Gemini free tier, or GPT-5.4-mini) so the two are genuinely independent.
- The 4-label rubric (supported / partial / unsupported / insufficient) is a constrained classification; a 7B VLM is comfortably enough. Text-span claims can be judged by a text-only 7B; only chart/figure crops need the VLM. [A]

### 2.4 Vector DB — Qdrant
- Native multivector with **MAX_SIM** comparator; **HNSW disabled on the raw multivector (m=0)** because MaxSim does not work with proximity graphs, and m=0 also saves RAM; search is brute-force exact. [V]
- **Scale trick (built in, use if corpus grows):** mean-pool the ~1024 vectors/page down to ~32 pooled vectors, prefetch top K≈200-256 pages by the cheap pooled vector (HNSW-indexed), then rerank those exactly with full MaxSim server-side. Training-free; retrieval quality within ~0.01 NDCG/Recall while giving ~4x QPS and up to ~13x lower latency. [V]
- Weaviate (v1.30+) is a viable alternative with the same MaxSim support if we prefer it. [V]

### 2.5 Grounding note (design risk to own)
- Snap-to-box is validated: "Snappy" (arXiv 2512.02660) ranks existing candidate boxes by patch-to-region similarity and hits **59.7% @IoU 0.5, 84.4% @IoU 0.25, mean IoU 0.569** vs ~6.7% random, training-free at inference. [V]
- **Caveat [R]:** Snappy snaps to *OCR* boxes; we snap to *PyMuPDF text-layer* boxes. These are functionally the same candidate-set idea, and ours are arguably cleaner (exact, no OCR error), but the transfer to PyMuPDF boxes is not empirically published. That transfer is genuinely part of our contribution, so state it as our result, not a cited given.

---

## 3. API Pricing (verified 2026) and Cost Breakdown

### 3.1 Live per-1M-token pricing [V]

| Model | Free tier? | Paid input /1M | Paid output /1M | Vision |
|---|---|---|---|---|
| Gemini 2.5 Flash | **Yes** | $0.30 (text/image/video) | $2.50 | Yes |
| Gemini 3.1 Flash-Lite | **Yes** | $0.25 | $1.50 | Yes |
| Gemini 3.5 Flash | **Yes** | $1.50 | $9.00 | Yes |
| Gemini 2.5 Pro | Yes (limited) | $1.25 (≤200k) | $10.00 | Yes |
| Gemini 3.1 Pro Preview | No | $2.00 (≤200k) | $12.00 | Yes |
| GPT-5.4-mini | No | $0.75 | $4.50 | Yes |
| GPT-5.4 | No | $2.50 | $15.00 | Yes |
| GPT-5.5 | No | $5.00 | $30.00 | Yes |
| OpenRouter Qwen2.5-VL-7B `:free` | Yes (rate-limited) | ~$0 | ~$0 | Yes |

Gemini free tier also has Batch/Flex modes at ~half standard price if we ever go paid. [V]

### 3.2 Demo-scale monthly cost [A]

Workload assumption (one query): reader reads ~3 page images (~1k tokens each) + prompt ≈ 3.5k input, ~300 output. Verifier judges ~3 claims ≈ 4.5k input, ~300 output. So ≈ **8k input / 0.6k output per query**.

| Scenario | Reader | Verifier | Retriever | Monthly cost |
|---|---|---|---|---|
| **Demo (recommended)** — a few hundred queries, all free tier | Gemini 2.5 Flash (free) | OpenRouter Qwen-VL `:free` or self-host | Campus GPU batch (one-time) | **$0** |
| Heavier dev, ~10k queries/mo, paid | Gemini 2.5 Flash | Gemini 3.1 Flash-Lite | Campus GPU batch | ~**$34/mo** (reader ~$18 + verifier ~$16) |
| Premium (if ever needed) | GPT-5.4-mini | Gemini 2.5 Flash | Campus GPU batch | ~**$70-90/mo** at 10k queries |

**Bottom line:** the demo runs at **$0**. Free-tier RPD (~1,500 req/day on Gemini Flash) far exceeds any live demo. Paid only matters if we run large offline batches, and even then it is tens of dollars, not hundreds.

---

## 4. Campus GPU Compute Plan (64 GB, shared, time-limited)

Principle: the GPU is a **batch resource booked in slots**, not an always-on server. Architect so nothing on the live path *requires* it.

### 4.1 fp16 VRAM budget (does it all fit in 64 GB?) [A]

fp16 weights ≈ 2 bytes/param. Estimated footprints:

| Component | Params | Weights (fp16) | With activations/KV |
|---|---|---|---|
| ColQwen2.5 retriever | ~3B | ~6-7 GB | ~8-10 GB |
| Qwen2.5-VL-7B reader | 7B | ~15 GB | ~18-20 GB |
| Qwen2.5-VL-7B verifier | 7B | ~15 GB | ~18-20 GB |
| **All three co-located** | | **~36 GB weights** | **~44-50 GB** |

**Yes: all three fit on one 64 GB GPU at fp16 with no quantization**, leaving ~15-20 GB for KV cache and batching. (A single 7B-VL alone is ~16-20 GB in practice, consistent with community reports.) [V for the single-7B figure; A for the sum]

### 4.2 How to fully utilize the campus GPU, end to end

**Phase 1 — Ingest (the heavy, reusable step). Book a slot, run batch.**
- Run ColQwen2.5 embedding over the whole corpus, batched across pages, checkpoint per document, push multi-vectors to Qdrant. This is the ONLY compute-heavy stage and it is one-time per corpus. A few hundred to a few thousand pages finishes in one slot.
- Also pre-compute the pooled (mean-pooled ~32-vector) index here for the prefetch stage.

**Phase 2 — Offline demo (optional, book a slot). Serve all three via vLLM.**
- During a booked slot, stand up three vLLM servers on the one GPU (retriever + reader + verifier), split with `--gpu-memory-utilization`, and run the fully-self-hosted, zero-external-call demo. ~44-50 GB used, comfortable on 64 GB. [A]
- This is the "runs with no internet, no API keys" story for the defense.

**Phase 3 — Live runtime (no GPU slot needed). Free APIs.**
- Query-time only needs: embed the short query (light; a quick GPU touch or CPU-tolerant), Qdrant search, then reader + verifier via **free-tier APIs** (Gemini Flash reader + a different free vendor as verifier). No GPU booking required, so demos never block on GPU availability.

### 4.3 Decision rule
- **GPU available (booked slot):** batch-embed the corpus, and optionally run Stack B (all-local) for the offline demo.
- **GPU unavailable (default day-to-day):** corpus is already embedded; reader + verifier run on free APIs; system is fully live at $0.

This gives us both claims honestly: a **$0 free-tier runtime** and a **fully-self-hosted no-external-calls demo**, without ever assuming the shared GPU is up at demo time.

---

## 5. Final Recommendation

Build on: **ColQwen2.5-v0.2 (Qdrant multivector + pooled prefetch) → Gemini 2.5 Flash reader (free) → self-hosted Qwen2.5-VL-7B verifier**, with PyMuPDF ingestion and every model behind a swappable interface. Use the campus GPU for one thing that matters (batch corpus embedding) and one thing that impresses (a booked all-local demo), and let free-tier APIs carry the live runtime so nothing blocks on the shared GPU. Cost for the project as scoped: **$0**, with a ~$34/mo ceiling only if we choose to run heavy paid batches.

Open items to own as our own results, not cited facts [R]: (a) snap-to-box transfer from OCR boxes to PyMuPDF text-layer boxes; (b) the exact verifier model, chosen by independence rather than a benchmark; (c) re-check ViDoRe V3 rankings and API prices near submission, both move fast.

---

## Sources

- Qdrant multivector MAX_SIM, HNSW m=0, brute-force; pooled prefetch-rerank (~4x QPS, ~13x latency): [Qdrant PDF retrieval at scale](https://qdrant.tech/documentation/tutorials-search-engineering/pdf-retrieval-at-scale/), [Qdrant ColPali optimization](https://qdrant.tech/blog/colpali-qdrant-optimization/), [Weaviate multi-vector ColiPali](https://docs.weaviate.io/weaviate/recipes/multi-vector-colipali-rag)
- Late-interaction retriever family + ViDoRe V3 (nemotron-colembed #1), ColQwen2/2.5, ~1024 vec/page, reference 3B reader pairing: [ViDoRe / vidore models](https://huggingface.co/vidore), [ColPali arXiv 2407.01449](https://arxiv.org/pdf/2602.03992), [pooling paper arXiv 2602.12510](https://arxiv.org/pdf/2602.12510)
- Snap-to-box grounding numbers (Snappy): [arXiv 2512.02660](https://arxiv.org/pdf/2512.02660)
- Gemini API pricing + free tier: [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing), [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- OpenAI GPT-5.x vision pricing: [OpenAI pricing](https://developers.openai.com/api/docs/pricing)
- Qwen2.5-VL-7B VRAM (single-model ~16-20 GB): [HF Qwen2.5-VL-7B HW discussion](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/discussions/18)
