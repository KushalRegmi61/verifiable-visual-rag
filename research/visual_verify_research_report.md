# Verifiable Visual RAG — Research & Background Report
**For:** IOE Pulchowk Campus (TU), BCT Computer Engineering — Minor Project proposal
**Purpose:** evidence base for the proposal (Background, Related Work, Gap, Methodology, Evaluation, Feasibility). Proposal document to be drafted separately from this.
**Method:** deep-research harness — 6 search angles → 27 sources → 117 candidate claims → top 25 adversarially verified by 3 independent voters (≥2/3 to confirm). 25/25 confirmed, 0 refuted.

> **Confidence convention used below:**
> **[Verified]** = passed 3-vote adversarial check (cited).
> **[Source, unverified]** = from a fetched source but *not* in the verified-25 set — treat as a lead to confirm, especially the commercial-tool claims.
> **[Inference]** = my reasoning connecting verified facts; not itself a citation.

---

## 0. The one finding that should change your design (read this first)

Your core visual-grounding mechanism — *turn ColPali/ColQwen2 query↔patch cosine-similarity into a heatmap, then draw a bounding box* — has a **documented faithfulness problem**, and it's not a minor footnote:

- ColPali/ColQwen2 similarity-based saliency maps are **"fragile and often misleading"** as evidence for which region a vision-LLM actually used; the authors **explicitly caution against relying solely on these visualizations**. **[Verified — AAAI/AIES, ref 4]**
- *Why:* raw cosine similarity is **"agnostic to the output"** — closeness in embedding space does **not** measure a patch's causal contribution to the answer. Faithful attribution needs **gradient-based or perturbation/occlusion** methods instead. **[Verified — ref 4]**
- In controlled grid tests the top-scoring patch **never coincided** with the ground-truth patch, with systematic spatial artifacts (an "O-shaped anomaly" from a faulty vision-embedding region). **[Verified — ref 4]**
- Mechanistically: query tokens **don't map to a single correct patch** — they match patches with *visually similar tokens or their neighbors*, smearing the heatmap. **[Verified — ref 3]**

**This does NOT kill the project. It sharpens it — and actually makes your design *more* defensible, not less.** Here's the reframe:

1. **The hybrid design is the fix, not a nice-to-have.** For prose/tables you already cite the **exact text-layer span** (PyMuPDF) — that path is unaffected by the heatmap problem and is *exact*. Lean on it.
2. **Snap the heatmap to real boxes instead of free-drawing.** A Dec-2025 paper does exactly this: it uses ColPali patch similarity as a **"spatial relevance filter" over extracted regions** rather than as raw pixels, and reports **59.7% hit-rate @ IoU 0.5, 84.4% @ IoU 0.25, mean IoU 0.569** vs ~6.7% random on BBox-DocVQA (ColQwen3-4B). **[Verified — ref 1]** Translation: *patch-similarity → rank/select among candidate boxes* works far better than *patch-similarity → draw a box*. Your candidate boxes come **free** from the PyMuPDF word/line boxes (born-digital PDFs) — no OCR engine needed.
3. **`verify()` stops being optional.** The judge-LLM verification step + calibrated abstention is precisely the mechanism that catches a wrong-but-confident heatmap. The literature says heatmaps lie; your architecture's answer is "then don't trust them un-verified." That's a *strength* to state out loud in the proposal.

**Net:** make the **text-span path first-class** (not a week-3 fast-follow), treat visual grounding as **select-from-text-layer-boxes** rather than draw-from-pixels, and frame `verify()`+abstention as the faithfulness guard the heatmap literature demands. This is the single most important takeaway in this report.

---

## 1. Background & Motivation

**The trust/attribution problem is real and measured — this is your strongest motivation material.**

- **Citation correctness ≠ citation faithfulness.** Current RAG citation evaluation only checks whether a cited doc *supports* a statement. But a citation can be *correct yet unfaithful* — post-rationalized to fit the model's pre-existing knowledge rather than genuinely relied upon. **Up to 57% of RAG citations were found to lack faithfulness.** **[Verified — ref 9]** → This single statistic justifies your *measured faithfulness ablation* better than any hand-waving about "hallucination."
- **Even the best LLMs don't fully ground their claims.** On ALCE, the first benchmark for automatic citation evaluation, **even the best models lack complete citation support ~50% of the time** (ELI5). **[Verified — ref 8]**
- **Frontier models over-trust their own memory.** GPT-4 and Claude-3.5 **over-rely on parametric knowledge in RAG and fail to ground answers exclusively in retrieved documents.** **[Verified — ref 11]** → Motivates an *explicit* grounding+abstention layer rather than "just prompt a good model."
- **Visually-rich documents are genuinely hard.** On MMLongBench-Doc (long PDFs needing text+charts+tables+layout), the **best model, GPT-4o, scores only 42.7% F1; GPT-4V only 31.4%.** **[Verified — ref 6]** → Visual document understanding is *far from solved*; there is real headroom for a contribution.

**Framing for the proposal:** Document AI today answers fluently but (a) cannot reliably prove *where* an answer came from, (b) routinely post-rationalizes citations, and (c) collapses on chart/table/figure-heavy pages. Region-level citation + calibrated abstention is the missing *trust layer*.

---

## 2. Existing Systems / Related Work

### 2A. Academic

| System / Work | What it does | Relevance to us | Confidence |
|---|---|---|---|
| **ColPali** (ref 2) | VLM producing multi-vector embeddings **directly from page images**, late-interaction matching; **no OCR pipeline**; outperforms modern doc-retrieval pipelines while simpler/faster/end-to-end trainable | Our L1 retrieval backbone | [Verified] |
| **ColQwen2 / ColQwen3** (refs 1, 12) | ColPali family on a Qwen2-VL backbone; ColQwen3-4B used in the patch→bbox grounding study | Our retrieval + grounding model | [Verified ref1; Source ref12] |
| **Patch→Region grounding** (ref 1, Dec 2025) | Uses ColPali patch similarity as a **spatial relevance filter over extracted regions** → bounding boxes; **mean IoU 0.569, 84.4% hit@IoU0.25** | **Direct evidence your visual grounding is feasible** — if done as *select-from-boxes* | [Verified] |
| **Saliency-map critique** (ref 4, AAAI/AIES) | Shows ColPali similarity heatmaps are **fragile/misleading**; recommends gradient/perturbation attribution | **The risk you must design around** (see §0) | [Verified] |
| **Late-interaction matching analysis** (ref 3) | Query tokens match *neighboring/visually-similar* patches, not one true patch | Explains *why* raw heatmaps smear | [Verified] |
| **SlideVQA** (ref 5) | 2,600+ decks / 52,000+ slide images / 14,500 questions; **multi-image reasoning** | Your **primary benchmark** | [Verified] |
| **MMLongBench-Doc** (ref 6) | 1,062 Qs over 130 long PDFs (avg 49.4 pp); 33.2% cross-page; **22.8% unanswerable (hallucination probes)** | Your **abstention** benchmark — built-in unanswerables | [Verified] |
| **ALCE** (ref 8) | First benchmark for automatic citation evaluation: fluency / correctness / citation quality | How citation quality is measured in the literature | [Verified] |
| **Citation-faithfulness** (ref 9) | Distinguishes correctness vs faithfulness; 57% unfaithful | Theoretical basis for your faithfulness metric | [Verified] |
| **Conformal abstention** (ref 7) | Abstention with **rigorous guarantees bounding the hallucination rate** below a user-set level | Makes your abstention *calibrated*, not heuristic | [Verified] |
| **LLMs-as-judges** (ref 10) | Survey: LLM evaluators are flexible/scalable/reproducible **but** sensitive to prompt wording and inherit biases | Justifies `verify()` *and* warns how to do it carefully | [Verified] |
| **Trust-Score / Trust-Align** (ref 11) | Metric = ⅓(grounded-refusal F1 + calibrated EM recall + citation-groundedness F1) | A ready-made scoring scheme for your eval dashboard | [Verified] |
| **M3DocRAG / VisRAG** | Multimodal/vision-based RAG pipelines (named in scope) | Closest "visual RAG" prior art | *Not in verified-25 — confirm details before citing* |

### 2B. Commercial

> ⚠️ **Lower confidence.** Commercial-capability claims came from secondary/blog sources and did **not** clear the 3-vote bar. Verify hands-on before putting specific claims in the proposal — these tools change monthly.

| Product | Citation behavior (as reported) | Region-level visual grounding? | Calibrated abstention? | Confidence |
|---|---|---|---|---|
| **ChatGPT (file/PDF Q&A)** | Conversational answers; weak/no persistent source pinning | No region-level box | No principled abstention | [Inference — verify] |
| **Google NotebookLM** | Inline citations back to source passages (text); documented limitations on doc size/handling (ref 18) | Text passage, **not** chart-region boxes | Conservative but not calibrated | [Source ref18, unverified] |
| **Adobe Acrobat AI Assistant** | Cites/links to source location in the PDF | Page/section level, not chart-region | Usage-guideline gated (ref 16 returned no usable claims) | [Inference — verify] |
| **Perplexity (PDF)** | Retrieval-based parsing + citation grounding (ref 19) | Source-level citations, not pixel regions | No | [Source ref19, unverified] |

**The honest read:** commercial tools increasingly do **text-passage citation**, but none of them is documented to do **region-level *visual* grounding on charts/figures with a calibrated, measured abstention rate**. That is a defensible gap — *if you state it as "to our knowledge / based on public documentation as of mid-2026" and don't overclaim.*

---

## 3. The Gap / Problem Statement

Synthesizing §1–§2, the gap your project occupies:

1. **Region-level *visual* citation is largely absent.** Academic work shows patch→region grounding is *possible* (ref 1) but also *fragile if done naively* (ref 4); commercial tools stop at text-passage citation. **No widely available system points at the exact chart bar / table cell and proves it.**
2. **Citation faithfulness is unmeasured in practice.** 57% of RAG citations are post-rationalized (ref 9); standard eval only checks correctness (ref 8). **Almost nobody runs the faithfulness ablation you're proposing.**
3. **Abstention is usually heuristic, not calibrated.** Conformal methods with guarantees exist (ref 7) but aren't standard in doc-QA products. MMLongBench-Doc *bakes in* 22.8% unanswerables (ref 6) precisely because models hallucinate on them.
4. **Hybrid text+visual citation per claim is unaddressed.** Routing each claim to its best evidence form (exact text span for prose/tables; visual region for charts/figures) is your specific novelty.

**One-sentence problem statement (draft):** *Existing document-QA systems — both research prototypes and shipping products — cannot reliably attribute each answer-claim to the exact region of a visually-rich page it came from, cannot distinguish genuine grounding from post-rationalized citation, and lack a calibrated mechanism to abstain when grounding fails; we build and measure exactly that layer.*

---

## 4. Methodology & Architecture — evidence each choice is viable

| Design choice | Evidence | Verdict |
|---|---|---|
| **ColPali/ColQwen2 retrieval over page images** | VLM multi-vector embeddings from images, late-interaction, no OCR, beats prior pipelines (ref 2) | ✅ Solid |
| **Patch-heatmap → bbox visual grounding** | Feasible **as select-over-regions**: IoU 0.569, 84.4% hit@0.25 (ref 1). **But** raw saliency is fragile/misleading (refs 3,4) | ⚠️ Viable **only** as snap-to-text-layer-boxes + verify; not raw pixels (see §0) |
| **PyMuPDF embedded text-layer extraction (no OCR)** | Standard for born-digital PDFs; supplies exact word/line boxes that double as your candidate regions | ✅ [Inference — well-established; not separately verified here] |
| **LangGraph agent orchestration** | Agentic-RAG with LangGraph is an established pattern (ref 20) | ✅ [Source, unverified] |
| **Judge-LLM `verify()`** | LLMs-as-judges is a recognized paradigm (ref 10) — **with caveats**: prompt-sensitivity + inherited bias | ✅ Viable, design carefully (fixed rubric, separate model) |
| **Calibrated abstention** | Conformal prediction bounds hallucination rate with guarantees (ref 7) | ✅ Strong; gives you a *principled* threshold |
| **Faithfulness / trust metrics** | ALCE (ref 8), Trust-Score (ref 11), correctness-vs-faithfulness (ref 9) | ✅ Ready-made metrics |
| **pgvector for embeddings / VLM readers (Qwen2-VL, GPT-4o, Gemini)** | Standard infra; GPT-4o is the *strongest* MMLongBench baseline at 42.7% (ref 6) | ✅ [Infra is conventional; reader choice evidence-backed] |

**Architecture note the research forces:** insert an explicit **faithfulness guard** between "draw region" and "show region" — i.e., `ground()` proposes a region, `verify()` (judge LLM, fixed rubric) confirms the claim is actually supported by that exact region, and **calibrated abstention** (conformal threshold, ref 7) drops anything below bar. This is the citation-invariant ("every claim round-trips to a verified region or it isn't said") made concrete.

---

## 5. Benchmarks & Evaluation

**SlideVQA (primary)** — 2,600+ decks, 52,000+ images, 14,500 questions; **multi-image reasoning** across a deck (not single-image). **[Verified against the primary paper — arXiv 2301.04883 / AAAI 2023, ref 5; confirmed June 2026]**
→ Use for: answer accuracy + your grounding-faithfulness ablation.
→ **Official metric (confirmed):** *"Following HotpotQA, we used exact match (EM) and F1 on each question answering and evidence selection task."* The paper also defines **Joint EM / Joint F1 (JEM/JF1)** scoring QA *and* evidence-selection together — i.e. right-answer-AND-right-source, which rhymes with our grounding thesis. So **EM/F1 is the native metric** (no stale-figure risk), and JEM/JF1 is an optional ready-made "answer + source" metric we can cite.

**MMLongBench-Doc (abstention stress test)** — 1,062 expert Qs over 130 long PDFs (avg 49.4 pages / ~21k tokens); **33.2% cross-page**, **22.8% deliberately unanswerable**. Best model **GPT-4o = 42.7% F1**, GPT-4V = 31.4%. **[Verified — ref 6]**
→ Use for: abstention calibration (the unanswerables are a built-in hallucination probe) and to show a *low SOTA ceiling* = real headroom.

**How grounding-faithfulness is measured in the literature (cite these):**
- **ALCE** — fluency / correctness / **citation quality**, metrics correlated with human judgment. **[Verified — ref 8]**
- **Citation faithfulness** — distinct from correctness; up to 57% unfaithful. **[Verified — ref 9]**
- **Trust-Score** = ⅓(grounded-refusal F1 + calibrated EM recall + citation-groundedness F1) — a concrete composite you can adopt for your dashboard. **[Verified — ref 11]**

**Your ablation spine (with vs without the grounding+abstention layer):** report answer accuracy *and* a grounding/faithfulness number, plus confident-wrong reduction. The 42.7% SOTA ceiling means even a modest, well-measured result is publishable-grade for a minor project.

---

### 5.1 LOCKED eval scope (what the project actually commits to)

> The above is the full evidence base. The **build commits to a minimal, time-boxed subset** — three metrics + one ablation, **all on SlideVQA, with no manual annotation.** This is what the proposal's Evaluation section should promise; everything else is future work.

1. **Answer accuracy** — SlideVQA **EM + F1** (F1 headline; accuracy is the *control variable*, not a SOTA target).
2. **Grounding IoU (text + visual)** — scored against **auto-derived gold boxes**: for any question whose answer string is locatable on the gold slide, its **PyMuPDF word box is free ground truth**. Forcing the *visual* path (heatmap → snap-to-box) on these same questions puts visual grounding on the eval path with **zero annotation**. Report **mean IoU + hit@IoU≥0.25**, the **~6.7% random-box floor**, and the **coverage %** of text-locatable answers. (Pure chart-only answers with no text label fall outside the auto-gold set — state this honestly; an optional ~30-item hand-checked supplement covers the tail.)
3. **Confident-wrong rate + coverage** — answered-but-wrong fraction with abstention **on vs. off**, always reported *with* coverage so it isn't gameable by always-abstaining. Abstention uses a **simple confidence threshold** in scope.

**The ablation (the spine):** with vs. without the ground+verify+abstain layer → accuracy ~flat, grounding appears, confident-wrong drops. One before/after table.

**Explicitly future work (off the required path):** MMLongBench-Doc, visual IoU on BBox-DocVQA, **conformal** calibration, the full ALCE/Trust-Score suite. They appear in this report as evidence/lineage, **not** as committed deliverables.

---

## 6. Feasibility (3 people, ~1 month, part-time)

- **Models are open-source & self-hostable.** ColPali/ColQwen2 are public (illuin-tech repo, HF) and run on a single modern GPU; multiple hosted/blog walkthroughs exist (refs 12–17). **[Source, mostly unverified — but well-established]**
- **No training required.** You *use* pretrained ColQwen2 + a VLM reader + a judge LLM; the work is the grounding/verify/abstain layer + eval. This fits a 1-month part-time budget.
- **Cost lever:** reader/judge can be API (GPT-4o/Gemini) for quality or local Qwen2-VL for zero marginal cost; swappable behind one interface.
- **Main technical risk is §0** (heatmap faithfulness), already mitigated by the hybrid + snap-to-boxes + verify design. The benchmark+ablation is your floor: even if grounding underperforms, a rigorous *measured study* stands on its own.

---

## 7. IOE Pulchowk BCT Minor Project — proposal structure (to feed later)

The DoECE (Dept. of Electronics & Computer Engineering, Pulchowk) publishes project templates, timeline, and evaluation docs (refs 21–23). **[Source — official dept site; details not individually 3-vote verified, confirm against the current-year PDF].** A standard IOE/DoECE proposal report is expected to contain, in order:

1. Title page / certificate
2. **Abstract**
3. **Introduction** (background + problem statement)
4. **Objectives**
5. **Motivation / Significance / Scope of project**
6. **Literature Review / Related Work** ← §1, §2 of this report
7. **Methodology** (system block diagram, algorithms, tools) ← §4
8. **System Requirements** (SW/HW)
9. **Feasibility study** (technical / operational / economic / schedule) ← §6
10. **Expected Output / Deliverables**
11. **Project Timeline / Gantt chart**
12. **References**

Evaluation is via proposal defense + mid-term + final, with marks split across report, presentation, and demonstration (per the DoECE timeline/evaluation PDF, ref 22). **Confirm the exact section list and rubric against your batch's official template before drafting.**

---

## 8. Reference list

1. *From Retrieval to Grounding* (patch→region, BBox-DocVQA) — arXiv 2512.02660
2. *ColPali: Efficient Document Retrieval with Vision Language Models* — arXiv 2407.01449
3. Late-interaction query–patch matching analysis — arXiv 2505.07730
4. *Fragile/misleading ColPali saliency maps* — AAAI/AIES (ojs.aaai.org/.../36763)
5. *SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images* — arXiv 2301.04883 (AAAI 2023). Scale (2.6k+ decks / 52k+ images / 14.5k Qs) and metric (EM + F1, plus Joint EM/F1) **confirmed against the paper, June 2026.**
6. *MMLongBench-Doc* — arXiv 2407.01523
7. *Conformal abstention with hallucination-rate guarantees* — arXiv 2405.01563
8. *ALCE: Automatic LLM Citation Evaluation* — arXiv 2305.14627
9. *Citation correctness vs faithfulness (57% unfaithful)* — Wallat et al. 2025 (UvA)
10. *LLMs-as-Judges survey* — arXiv 2412.05579
11. *Trust-Score / Trust-Align* — arXiv 2409.11242
12. Together.ai — Multimodal Document RAG with Llama-3.2-Vision & ColQwen2 (blog)
13. Qdrant — ColPali blog
14. Spheron — ColPali GPU cloud (blog)
15. Colivara — self-hosting docs
16. illuin-tech/colpali — GitHub
17. HuggingFace — manu/colpali
18. XDA — NotebookLM limitations
19. DataStudios — Perplexity PDF retrieval/citation grounding
20. Medium — Agentic RAG with LangGraph (2026)
21. DoECE Pulchowk — project/thesis LaTeX templates
22. DoECE Pulchowk — BE Project Timeline & Evaluation (PDF)
23. DoECE Pulchowk — Project Timeline (Minor/Major)

---

### Caveats (honesty section)
- **Commercial claims (§2B) are the weakest part** — blog/secondary sourced, not adversarially verified. Verify hands-on; phrase as "based on public documentation as of mid-2026."
- **M3DocRAG / VisRAG / DocVQA** were in scope but didn't surface verified claims; confirm their specifics before citing.
- **IOE proposal structure** is the standard DoECE format but **must be checked against your batch's current template** — don't submit against this list blind.
- **The §0 heatmap-faithfulness finding is the highest-value, highest-confidence result** (3-0 verified across two independent papers) and should reshape both your build and how you pitch the contribution.
