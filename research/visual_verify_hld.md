# Verifiable Visual RAG — Lean HLD (minor project)

**Team of 3 · ~1 month · part-time · English-first.** A document-Q&A agent over visually-rich PDFs
that pins every claim to the exact pixel region it read it from, abstains when it can't ground, and
is measured on a public benchmark with an ablation. Ships a small reusable core as `visual-verify`.

> Design rule for the whole build: **the grounding+verify code is a standalone, dependency-light
> module from day 1.** The app imports it; the eval tests it; week-4 we publish it. No bolt-on.

---

## 1. Week-1 scope lock (decide these first, then freeze)

| Decision | Recommended default | Why |
|---|---|---|
| **Benchmark** | **SlideVQA** (primary) + a handful of real financial reports (demo set) | Slide decks are visually rich, manageable size, free ground-truth Q&A |
| **Doc types** | Slides + financial-report pages only | Charts/tables/figures = where region-grounding shines; bounded variety |
| **Grounding modalities** | **Both first-class** — text-span (exact) + visual region (snap-to-box) | Research finding: raw ColPali heatmaps are *fragile/misleading* as faithful attribution. The exact text-span path is immune to that and is our reliability floor; visual is the differentiator. Build both early. |
| **Visual grounding method** | Patch-similarity **selects/ranks among text-layer candidate boxes** — *not* free-drawn from pixels | Snap-to-box hits ~0.57 mean IoU / 84% hit@IoU0.25 (BBox-DocVQA); raw saliency-map→bbox is unfaithful (AAAI/AIES). Heatmap proposes, text-layer boxes constrain, `verify()` confirms. |
| **Text source** | PDF **embedded text layer** (PyMuPDF), **no OCR engine** | Free precise spans **and the candidate boxes the visual path snaps to**; born-digital PDFs only, scanned out of scope |
| **Retrieval/grounding model** | **ColQwen2** (ColPali family) | Late-interaction → patch heatmaps for grounding come almost free |
| **Reader VLM** | Qwen2-VL (local) or GPT-4o/Gemini (API) | Reading the retrieved page; swappable behind one interface |
| **Judge LLM** | A *separate* LLM (not the reader) | Verifies claim↔region support without grading itself |
| **Languages** | English only | Spark stays off; no Devanagari risk |

**Eval scope is LOCKED to three metrics + one ablation** (minimal set to pass; everything else is future work):
1. **Answer accuracy** — SlideVQA, **EM + F1** (F1 is the headline; accuracy is the *control variable*, not a SOTA target).
2. **Grounding IoU** — text *and* visual, scored against **auto-derived gold boxes** (the answer string's PyMuPDF box = free ground truth, **no manual annotation**). Report **mean IoU + hit@IoU≥0.25**.
3. **Confident-wrong rate + coverage** — abstention on vs. off (answered-but-wrong fraction; reported *with* coverage so it isn't gameable).

The **ablation** (with vs. without our ground+verify+abstain layer) is the spine: accuracy stays ~flat, grounding appears, confident-wrong drops. **Future work (off the required path):** visual IoU on BBox-DocVQA, MMLongBench-Doc, conformal calibration, full ALCE suite.

---

## 2. Frozen contracts (week 1 — Pydantic v2)

```python
GroundedRegion(page: int, bbox: tuple[int,int,int,int], score: float,
               modality: Literal["visual","text"],     # the seam that makes text an *add*, not a refactor
               crop_ref: str | None = None,             # set for modality="visual"
               text: str | None = None)                 # set for modality="text" (the cited span)
Claim(text: str, regions: list[GroundedRegion], confidence: float, abstained: bool)
Answer(question: str, claims: list[Claim], abstained_overall: bool)
RetrievedPage(doc_id: str, page: int, image_ref: str, text_layer: str | None, score: float)
```

The **package public API** (freeze this hardest — it's the seam *and* the artifact):

```python
# package: visual_verify  (the shippable core)
def ground(page, query) -> list[GroundedRegion]: ...  # page = image + text layer (boxes)
#   text:   text-layer span match -> exact span + its bbox  (exact, faithful)
#   visual: ColPali patch-heatmap -> SELECT/RANK among text-layer candidate boxes  (snap-to-box, not free-drawn)
def verify(
    claim: str, region: GroundedRegion
) -> VerifyResult: ...  # judge LLM: supported? confidence? abstain?


#   verify() is the faithfulness guard: heatmaps are unreliable alone, so no region is shown un-verified.
```

> **Why snap-to-box, not raw heatmap:** ColPali cosine-similarity saliency maps are documented as *fragile/misleading*
> for faithful attribution (query tokens smear across neighboring patches; similarity is output-agnostic). So the visual
> path uses the heatmap to **rank candidate boxes pulled from the text layer**, never to free-draw pixels — then `verify()`
> + calibrated abstention drops anything the judge can't confirm. Both modalities return the same `GroundedRegion`, so the
> agent and eval harness are modality-blind.

---

## 3. Architecture — 4 lanes, clean seams

**Eval (L4) is its own lane, separate from the product (L3).** L4 imports the package and runs against
SlideVQA in a batch harness — it shares **no code or runtime** with the shipped app. The product and the
eval are two independent consumers of the same `visual_verify` public API.

```
[ L1 Retrieval ]            [ L2 Core module + Agent ]          ┌─> [ L3 Product — the shipped demo ]
 page images -> ColQwen  ->  visual_verify.ground / .verify  ──┤     Next.js app: ask -> answer -> highlight UI
 -> pgvector -> top pages    + LangGraph: retrieve->read->     │     -> abstention badges   (LIVE, per request)
                             answer->ground-check->abstain     │
                                                               └─> [ L4 Eval — OFFLINE, not shipped ]
                                                                     SlideVQA batch harness + 3 metrics + ablation
                                                                     + failure analysis   (no UI, no server)
```

> Two consumers, one API. **L3 (product)** calls `ground()/verify()` live per user request and renders regions;
> **L4 (eval)** calls the same functions in a loop over a dataset and emits numbers. Neither depends on the
> other — you can ship the app without the harness, or run the harness with no app.

### L1 — Retrieval index
Ingest page images, embed with ColQwen2, store in pgvector, return top-k `RetrievedPage` for a query.
Owns: the index + retrieval quality (recall@k). Hands L2 a contract, nothing more.

### L2 — Core module `visual_verify` + the agent (the differentiator)
- **`ground()` — text path (the reliability floor, wks 1–2):** match the claim against the page's **text layer** (PyMuPDF) → exact span + its bbox, returned as a `modality="text"` region. Exact and faithful; no OCR; born-digital text layer only. *This also produces the candidate boxes the visual path needs.*
- **`ground()` — visual path (the differentiator, wks 2–3):** use ColQwen's query↔patch similarity heatmap to **rank/select among the text-layer candidate boxes** (snap-to-box), *not* to free-draw from pixels — because raw saliency maps are unfaithful. Returns a `modality="visual"` region.
- **Per-claim routing:** chart/figure claims → visual region; prose/table-text claims → text span. Same `GroundedRegion` either way.
- **`verify()` — the faithfulness guard:** judge LLM (separate model, fixed rubric) checks the claim is actually supported by the cited region (crop *or* span); low → abstain. Because heatmaps lie, **nothing is shown un-verified.**
- **Agent (LangGraph):** `retrieve → read (VLM) → draft claims → ground each claim → verify each → abstain/assemble`.
  Bounded loop, every claim round-trips to a region or it's dropped. (Your citation-invariant discipline.)
- Owns: the package (standalone, tested) + the agent that uses it.

### L3 — Product (the shipped demo, the live app)
Next.js — upload/ask → answer pane → source page with **highlight boxes** (click a claim → region lights up)
→ ⚠ abstention badges. Calls the same `ground()/verify()` **live, per request**. This is the *only* layer a
user touches; it carries none of the eval/dataset code. Owns: the app + the live UX.

### L4 — Eval (OFFLINE — its own lane, separate from the product)
A standalone batch harness: imports the `visual_verify` package, runs it over the SlideVQA dataset, emits
metrics + the ablation table. **No UI, no server, nothing shipped** — it never executes in the product runtime.
Three locked metrics (see §1), all on SlideVQA, **no manual annotation**:
- **1. Answer accuracy:** SlideVQA **EM + F1** (F1 headline).
- **2. Grounding IoU:** auto-derive a **gold box** from each answer string's PyMuPDF location, then score **both** paths against it → `mean IoU`, `hit@IoU≥0.25`. Forcing the *visual* path here (heatmap → snap-to-box) puts visual grounding on the eval path for free; report the **coverage %** of text-locatable answers, and the **random-box baseline (~7%)** as the floor.

  | Path | mean IoU | hit@0.25 |
  |---|---|---|
  | Text-span (exact) | ~1.0 | ~100% |
  | **Visual (snap-to-box)** | target ~0.5–0.6 | target ~80% |
  | Random box | ~0.07 | ~7% |

- **3. Confident-wrong rate + coverage:** answered-but-wrong fraction, abstention **on vs. off**, always paired with coverage.
- **The ablation (the spine):** with ground+verify+abstain vs. without → accuracy ~flat, grounding appears, confident-wrong drops. One before/after table.
- **Failure analysis:** categorize residual errors (retrieval miss / reader misread / snap miss / judge error) — signals seniority louder than the score.
- Owns: the offline harness + the numbers + the failure analysis. Output is a static results table/dashboard that only renders pre-computed harness output.

**Seam rule:** L2 consumes L1's `RetrievedPage`; **L3 (product) and L4 (eval) each touch L2 only through the
package's public API** — and never touch each other. Nobody edits a shared file.

> **Lanes vs. headcount:** 4 lanes (L1–L4), 3 people. L4 (eval) is the lightest lane — a batch script, no UI —
> so it pairs naturally with whoever owns L3, but it's a **separate lane/deliverable with its own seam**, not
> folded into the product.

---

## 4. 4-week timeline (part-time)

| Wk | L1 Retrieval | L2 Core+Agent | L3 Product | L4 Eval (offline) |
|---|---|---|---|---|
| 1 | Scope lock · **freeze contracts** · ColQwen index skeleton | **Text-span path** (PyMuPDF) → exact spans **+ candidate boxes** · `verify()` skeleton | UI skeleton (answer pane, page view) | SlideVQA loader · auto-gold-box extractor |
| 2 | Retrieval working · recall@k | **2-day heatmap spike** → visual path (snap-to-box over candidate boxes) | Highlight UI (box *and* span) wired | Answer-accuracy (EM/F1) + grounding-IoU harness live |
| 3 | Tuning · real-report demo set | Agent loop + abstention + citation invariant · end-to-end | Abstention badges · click-to-highlight | Confident-wrong/coverage + **the ablation** |
| 4 | Polish/index docs | Package cleanup + tests | Demo polish | Failure analysis · results dashboard · **publish `visual-verify`** |

> If the **visual** heatmap spike underperforms (it's the risky path — saliency maps are fragile), ship **text-span-only** grounding — still a complete, faithful, measured citation system — and list visual region-grounding as the headline future-work + failure-analysis result. The reliability floor is the text path, which is built first.

---

## 5. Risks & honest mitigations
- **Visual heatmap is unfaithful** (the real headline risk — ColPali saliency maps are documented as fragile/misleading) → mitigated three ways: (1) **snap-to-box** instead of free-draw (rank text-layer candidate boxes, ~0.57 IoU vs 6.7% random), (2) **`verify()` gates every region** (nothing shown un-confirmed), (3) the **text-span path is the reliability floor** and is built first — if visual fails entirely, text-only still ships a complete faithful system.
- **Grounding underperforms** → still have a rigorous *measured study* + ablation (random-box floor stays ~7% on the IoU table). The benchmark is the safety net; visual doc-QA has a low SOTA ceiling generally (e.g. best MMLongBench-Doc model = 42.7% F1 — a *future-work* benchmark, cited here for context), so even modest measured results are defensible.
- **Judge LLM is unreliable** (prompt-sensitive, biased) → separate model from the reader, fixed rubric, and a **simple confidence-threshold** abstention cutoff for the locked scope. **Conformal calibration** (rigorous hallucination-rate guarantee) is the principled upgrade — listed as future work, not on the required path.
- **Real-time/cost** → none; this is offline doc Q&A, frame-sampling N/A.
- **Scope creep** → slides + financial pages only, English only, **no OCR engine** (text layer only). Abstention covers degraded inputs by design.
- **Package distraction** → it's a *publishing* step, not a workstream; the seam is drawn week 1.

## 6. Résumé payload
Multimodal document RAG (ColQwen) · **hybrid text-and-visual region-level grounding** · agentic LangGraph · citation
verification + calibrated abstention · **a real benchmark number with an ablation + failure analysis** ·
a published open-source core. The clean text→multimodal upgrade — and it pairs with **Breaker** (your
solo agent-security flex) for a portfolio that reads as both *rigorous* and *dangerous*.
