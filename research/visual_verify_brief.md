# Verifiable Visual RAG — Project Brief (2-min read)

## The idea (one line)
A document-Q&A agent that answers questions over **visual PDFs** (charts, tables, figures) and **pins every claim to the exact pixel region it read it from** — and **refuses to answer when it can't ground a claim.** ChatGPT reads documents but can't prove *where* an answer came from or admit when it's unsure. We build exactly that missing trust layer.

## Why it's not "just another RAG demo"
- **Region-level visual citation** — we highlight the exact chart bar / table cell, not "see page 7."
- **Hybrid grounding (text + visual)** — prose/table claims cite the **exact text span** (crisp sentence highlight, pulled free from the PDF's text layer — exact and faithful); chart/figure claims cite the **visual region**, where the ColPali heatmap *selects among those text-layer boxes* (snap-to-box) rather than free-drawing — because raw similarity heatmaps are known to be unreliable on their own. Best citation for each kind of claim. *No separate OCR engine — we use the embedded text layer, not a scanned-image pipeline.*
- **Verification before display** — every cited region is checked by a separate judge LLM; nothing is shown ungrounded. This is the faithfulness guard the visual-grounding literature says you need.
- **Abstention** — it shuts up when it can't ground a claim instead of hallucinating (confidence-threshold in scope; *conformal* calibration is the principled future-work upgrade).
- **Anchored to a public benchmark (SlideVQA)** → three locked metrics, no manual annotation: **(1) answer accuracy** (EM/F1), **(2) grounding IoU** for text *and* visual (scored against gold boxes auto-derived from the answer string's PyMuPDF location — so visual grounding is *measured*, not just demoed), **(3) confident-wrong rate + coverage** (abstention on vs. off). An **ablation** (with/without our layer) proves it *caused* the improvement: accuracy ~flat, grounding appears, confident-wrong drops. That's science, not a demo. *(BBox-DocVQA, MMLongBench-Doc, conformal calibration = future work.)*

---

## INPUT
1. A set of **visually-rich PDFs** (financial reports, research papers, slide decks) — answers live in charts/tables *and* prose. The page *image* is the unit for retrieval; the embedded **text layer** is pulled (free, via PyMuPDF) for precise text grounding. **No OCR engine** — scanned image-only PDFs are out of scope.
2. A **natural-language question**, e.g. *"What was Q3 revenue and how did it trend?"*

## OUTPUT (per question)
- The **answer** in plain English.
- **Region-level citations, per claim** — chart/figure claims → `(page, bbox)` + cropped image (**visual**); prose/table claims → the **exact text span** highlighted (**text**). Each claim routes to whichever grounds it best.
- A **grounding-confidence score** per claim.
- An **abstain / ⚠ flag** when a claim can't be grounded — routed out, not guessed.

> Contract: **every sentence round-trips to a region — a pixel box or a text span — or it doesn't get said.**

---

## THE FINAL PRODUCT (what we demo)
A simple web app:
1. Upload a PDF, ask a question.
2. Answer on the left.
3. Source page on the right with **colored highlights** — chart/table claims light up a **box on the figure**, prose claims light up the **exact sentence** — click a claim, its region lights up.
4. Some claims show **"⚠ couldn't ground — abstained."**

That's the **live product (L3)**. Separately, the **offline eval (L4)** produces a static **results view** of the 3 locked numbers (SlideVQA accuracy · grounding IoU · confident-wrong+coverage) + the ablation table — it renders pre-computed harness output and is *not* part of the live app.

Plus a clean, dependency-light **core module** (`ground` + `verify`) that any RAG pipeline could reuse — published at the end as a small open-source package.

---

## THE LIBRARY WE SHIP (the open-source artifact)

The hard, reusable part — carved out as a **`pip install`-able package** (e.g. `visual-verify`). **Not** the whole app; just the one thing the ecosystem lacks: region-level grounding + verification. Two verbs:

```python
from visual_verify import ground, verify

# 1. Page (image + optional text layer) + question -> the exact region(s) that answer it
regions = ground(page, query)
#   text:   text-layer span match     -> exact span + box   (modality="text",   exact/faithful)
#   visual: heatmap RANKS text boxes  -> selected region    (modality="visual", snap-to-box, not free-drawn)
#   -> [GroundedRegion(bbox=(x,y,w,h), score=0.88, modality="text",   text="Q3 revenue was $4.2M"),
#       GroundedRegion(bbox=(x,y,w,h), score=0.83, modality="visual", crop=<img>), ...]

# 2. A claim + its cited region -> does the region actually support it? abstain if not
result = verify(claim="Q3 revenue was $4.2M", region=regions[0])
#   -> VerifyResult(grounded=True, confidence=0.91, abstain=False)
```

- **One API, both modalities** — a `GroundedRegion` carries a `modality` flag (`"visual"` or `"text"`); callers treat text spans and pixel boxes uniformly.
- **Model-agnostic, framework-agnostic** — no LangGraph/UI dependency. Anyone on LlamaIndex or a plain script bolts on grounded, self-abstaining citations in 5 lines.
- **Ships with a benchmark number** (*"X% grounding accuracy on SlideVQA"*) → that's what makes it a *tool*, not a repo-dump.
- **Near-zero extra cost** because we design L2's code as this standalone module from **day 1**; "publishing it" at week 8 is one afternoon, not a refactor.
- **One rule:** nobody treats "ship the package / get stars" as a task. The deliverables are the demo + the benchmark numbers; the library *falls out of them for free* because we drew the module boundary on week 1.

**Résumé value:** moves the story from *"I built a RAG demo"* → *"I found that visual RAG can't do region-level attribution, built a model-agnostic library that does, and benchmarked it."* That's an engineer who **productizes a capability** — the 1% signal.

---

## THE WORK — 4 lanes (1 month, part-time · 3 people; L4 eval is the lightest, pairs with L3's owner)

| Lane | Owner builds | Skills |
|---|---|---|
| **L1 — Visual Retrieval** | Run ColPali/ColQwen over page images, index in pgvector, retrieve right page(s) for a query | Multimodal retrieval |
| **L2 — Grounding + Agent** | The novel core: text-layer span lookup (**text**, exact) + ColPali heatmap ranking those boxes (**visual**, snap-to-box); judge-LLM `verify()` gates every region; LangGraph loop *retrieve → answer → ground-check each claim → verify → abstain*. Lives in the reusable module. | Agentic + the differentiator |
| **L3 — Product** | The live app: Next.js — answer pane, highlight UI (box *and* span), abstention badges. Calls the package API **live, per request**. The only layer a user touches. | Product |
| **L4 — Eval** *(separate, offline)* | Batch harness over the package → 3 locked metrics on SlideVQA (accuracy EM/F1 · grounding IoU vs auto-gold-boxes, text+visual · confident-wrong+coverage) + the with/without ablation + failure analysis. **No UI, nothing shipped** — never runs in the product. | Evaluation |

**Seam (freeze week 1):** L2 consumes L1's retrieval output; **L3 (product) and L4 (eval) each consume L2's module via its public API, and never touch each other.** Clean contracts → nobody blocks anybody.

---

## What each of us walks away with (résumé payload)
Multimodal document RAG (ColPali) · **hybrid text-and-visual region-level grounding** · agentic LangGraph orchestration · **citation verification + calibrated abstention** · a **real benchmark number with an ablation** · an open-source core module. The "trust layer for visual AI" story — exactly what frontier chatbots don't give you.

> **Scope note (phasing):** Build the **text-span path first** (weeks 1–2) — it's exact, faithful, and produces the candidate boxes the visual path snaps to. Then add the **visual heatmap→box ranking** (weeks 2–3) — the differentiator, but the risky path (saliency maps are fragile, so it goes through `verify()`). The `modality`-aware contract means both share one shape. **If visual underperforms, ship text-only** — still a complete, faithful, measured citation system — and visual region-grounding becomes the headline future-work + failure-analysis result. Reliability floor first, differentiator second.
