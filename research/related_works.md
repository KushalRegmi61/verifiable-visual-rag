# Related Work: Verifiable Visual RAG for Region-Level Evidence

Working notes surveying systems adjacent to our proposal. Organized by the three
pillars our system combines: **(1)** region/bounding-box-level evidence on the page
image, **(2)** an *independent* verifier that checks the region actually supports the
claim, and **(3)** abstention when it does not. The recurring finding is that existing
work occupies **one** of these camps at a time; the union of all three is unfilled.

> Citation status: IDs marked ✅ are established and safe to cite. IDs marked ⚠️
> surfaced in search with future/late-2025 stamps and **must be verified on arxiv.org
> before inclusion** in the proposal.

---

## 1. Visual Document Retrieval (retrieve over page *images*, no OCR pipeline)

These define the retrieval layer we build on. All ground evidence at **page level**;
sub-page signal (if any) is a late-interaction heatmap used for ranking, never emitted
as validated region evidence.

| System | Arch / backbone | Evidence granularity | Verify | Abstain |
|---|---|---|---|---|
| **ColPali** ✅ | Late-interaction multi-vector (ColBERT MaxSim), PaliGemma | Page; per-token patch heatmap = interpretability only | No | No |
| **ColQwen2 / ColQwen2.5** ✅ | Same, Qwen2-VL backbone (our default retriever) | Page; heatmap for ranking | No | No |
| **DSE** (Document Screenshot Embedding) ✅ | Single-vector bi-encoder, Phi-3-V / Qwen2-VL | Page only (pooling destroys locality) | No | No |

- ColPali: *ColPali: Efficient Document Retrieval with Vision Language Models*, Faysse
  et al., 2024 — arXiv:2407.01449 (introduces the ViDoRe benchmark). ✅
- ColQwen2: ColPali family on Qwen2-VL; `vidore/colqwen2` checkpoints, now in HF
  Transformers. ✅
- DSE: *Unifying Multimodal Retrieval via Document Screenshot Embedding*, Ma et al.,
  2024 — arXiv:2406.11251. ✅

**Takeaway:** the heatmap our retriever produces is a *by-product for ranking*. No
retriever validates it as faithful localization — which is precisely why we treat it
only as a ranking signal over independently-derived text-layer boxes (snap-to-box),
never as evidence drawn from pixels.

---

## 2. Visual RAG Pipelines (retriever + VLM generator)

End-to-end systems closest to our overall shape. All feed **whole page images** to the
generator and ground only to page identity; none verify the answer against the region
or abstain.

| System | What it adds | Evidence | Verify | Abstain |
|---|---|---|---|---|
| **VisRAG** ✅ | VLM page-embedding retriever + VLM generator; 20-40% gain over text RAG | Page image | No | No |
| **M3DocRAG** ✅ | ColPali-style retrieval across 1000s of PDFs / 40k+ pages | Top-K pages | No | No |
| **VDocRAG** ✅ | Unified-image format, self-supervised page compression | Page image | No | No |

- VisRAG: *VisRAG: Vision-based RAG on Multi-modality Documents*, Yu et al., 2024 —
  arXiv:2410.10594. ✅
- M3DocRAG: *M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page
  Multi-document Understanding*, Cho et al., 2024 — arXiv:2411.04952 (introduces
  M3DocVQA). ✅
- VDocRAG: *VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents*,
  Tanaka et al., 2025 — arXiv:2504.09795 (introduces OpenDocVQA). ✅

**Takeaway:** evidence is a whole page; the generator can attend anywhere, so there is
no localized, checkable evidence link. This is the gap our grounding + verification
stages close.

---

## 3. Region-Level Attribution on Document Images (closest prior art)

The only systems that emit a region/box tied to the answer on a document page image.
Critically, they obtain boxes by having the **VLM predict pixel coordinates** — the
mechanism our design deliberately avoids.

- **VISA** — *Retrieval-Augmented Generation with Visual Source Attribution*, 2024 —
  arXiv:2412.14457. ✅ VLM is trained to emit the supporting bounding box (~89% region
  match). **Boxes predicted from pixels**, not snapped to a text layer; degrades on
  long/multi-doc.
- **LAT** — *Look as You Think: Unifying Reasoning and Visual Evidence Attribution for
  Verifiable Document RAG via Reinforcement Learning*, Liu et al., 2025 —
  arXiv:2511.12003. ✅ **Verified.** Chain-of-Evidence (CoE) grounds each reasoning step
  to a region via bbox + page index, trained with RL. Still **pixel-coordinate boxes**;
  needs cold-start annotations.
- **RegionRAG** — *Region-level Retrieval-Augmented Generation for Visual Document
  Understanding*, Li et al., 2025 — arXiv:2510.27261. ✅ **Verified.** Groups salient
  patches into semantic regions at inference (+10% R@1 retrieval, +3.6% QA, 71% of the
  visual tokens). Region-level **retrieval**, but no independent answer verification or
  abstention.
- **Patch-to-Region Relevance Propagation** — *Spatially-Grounded Document Retrieval via
  Patch-to-Region Relevance Propagation*, Georgiou, 2025 — arXiv:2512.02660. ✅
  **Verified.** **The nearest neighbor to our snap-to-box method.** Propagates ColPali
  patch-heatmap scores onto OCR boxes via IoU (open-source tool "Snappy"; ColQwen3-4B
  hits 59.7% at IoU@0.5, cuts context tokens ~29-52%). Training-free. **But it is
  retrieval-only: it ranks regions to feed RAG and never verifies a generated answer
  against the chosen region.** Our loop closes where theirs stops — this paper must be
  cited and out-differentiated directly.
- **SciEGQA** — *A Dataset for Scientific Evidence-Grounded Question Answering and
  Reasoning*, Yu et al., 2025 — arXiv:2511.15090. ✅ **Verified** (this ID was
  previously mislabeled "BBox-DocVQA"). Directly on our scope: scientific documents with
  **semantic evidence regions annotated by bounding boxes**; 1,623 curated QA pairs +
  30K auto-generated training pairs; reports VLMs struggle at evidence grounding but
  improve when trained on it. A candidate evaluation/comparison dataset for us.
- **BBox-DocVQA** — dataset of semantic-region bboxes (benchmarked by Patch-to-Region
  above). ⚠️ **Real name, arXiv ID not yet located** — the ID 2511.15090 first attached
  to it is actually SciEGQA. Look up its true ID before citing.

**Takeaway:** region attribution on document images is new (late-2024 onward) and
almost entirely pixel-coordinate-based. The one text-layer-snap system is retrieval-only.

---

## 4. Answer Verification, LLM-as-Judge, Hallucination Detection

Mature for **text** RAG; none of it is wired to visual/region evidence, and the
self-checking variants are known to be biased.

- **SelfCheckGPT** ✅ (arXiv:2303.08896, 2023) — samples the *same* model; consistency
  != correctness, no external grounding.
- **Chain-of-Verification (CoVe)** ✅ (arXiv:2309.11495, 2023) — same model plans and
  answers its own verification questions; inherits the generator's overconfidence.
- **FActScore** ✅ (arXiv:2305.14251, EMNLP 2023) — canonical **atomic-fact
  decomposition** + separate checker; binary supported/unsupported, text-only.
- **RefChecker** ✅ (arXiv:2405.14486, Amazon 2024) — claim-triplet extraction +
  separate checker over three context regimes; text references only.
- **MiniCheck / Bespoke-MiniCheck** ✅ (arXiv:2404.10774, EMNLP 2024) — small separate
  fact-checker, GPT-4-level at ~400x lower cost; sentence-vs-document, no "insufficient
  evidence" class.
- **RAGAS** ✅ (EACL 2024) and **ARES** ✅ (arXiv:2311.09476, 2023) — RAG faithfulness
  evaluation suites; text-only, no per-answer abstention gate.

**On why the judge must differ from the reader:**
- **Self-Preference Bias in LLM-as-a-Judge** ✅ (arXiv:2410.21819, 2024) — judges favor
  their own outputs.
- **MLLM-as-a-Judge** ✅ (arXiv:2402.04788, ICML 2024) — multimodal judges (GPT-4V)
  suffer bias, hallucination, inconsistency.

**Takeaway:** verify-decompose exists, but only over text and often via same-model
self-check. Our **separate judge** + **4-way rubric with an explicit "insufficient
evidence" class** (richer than binary FActScore / triplet RefChecker) is the documented
mitigation applied, for the first time, to region-grounded visual evidence.

---

## 5. Abstention / Selective Prediction / Conformal Calibration

- **Risk-coverage tradeoff** — El-Yaniv & Wiener, 2010 (foundational selective
  prediction). ✅
- **Selective VQA** ✅ (arXiv:2306.08751, CVPR 2023) — abstains on images but at the
  **whole-answer** level with a confidence score; no claim-level, region-grounded check.
- **Conformal Abstention for LLMs** ✅ (arXiv:2405.01563, 2024) — distribution-free
  bound P(hallucinate) <= alpha via a calibration set; **text-only**.
- **RefusalBench** — *Generative Evaluation of Selective Refusal in Grounded Language
  Models*, Muhamed et al., 2025 — arXiv:2510.10390. ✅ **Verified.** Closest to our
  verify-then-abstain idea; finds even strong models score <50% on multi-document
  refusal. But **text-only and evaluation-only** (176 perturbation strategies over
  NQ/GaRAGe), not a live gate on a visual pipeline.

**Takeaway:** abstention is well-studied for text and whole-answer visual QA, but
distribution-free calibration of *region-grounded claim verification* is unaddressed —
which is exactly why we position conformal calibration as **future work**, a defensible
extension rather than a solved baseline.

---

## 6. Attention/Heatmap Faithfulness (theoretical justification for snap-to-box)

- **Jain & Wallace, "Attention is not Explanation"** ✅ (arXiv:1902.10186, NAACL 2019) —
  attention weights are not faithful; permuted attention yields the same prediction.
- **Wiegreffe & Pinter, "Attention is not not Explanation"** ✅ (arXiv:1908.04626,
  EMNLP 2019) — rebuttal: attention can be explanation under stricter tests; not a free
  pass.

**Takeaway:** raw attention/heatmaps are not faithful evidence. This is the direct
theoretical grounding for our core design choice: **never draw a box from pixels;** use
the heatmap only to *rank* independently-derived PyMuPDF text-layer boxes.

---

## 7. Production / Commercial Tools ("chat with your PDF")

None combine region highlighting + verification + abstention. Citations are pointers,
and mismatch is documented.

| Product | Citation granularity | Region box on page image | Verify claim↔region | Abstain |
|---|---|---|---|---|
| NotebookLM (Google) | Text passage (inline, hover-to-quote) | No | No (source-grounded only) | Refuses out-of-source only |
| ChatPDF / Humata | Passage/page, scroll-to-source | No (text-only) | No | No |
| SciSpace / Elicit | Passage / claim-to-quote | No | No (recommends human check) | No |
| Consensus / Scite | Study / citation-statement | No | Scite classifies support/contrast *in the literature*, not your answer | Displays weakness, no gate |
| Adobe Acrobat AI | Clickable source anchor | No | No (warns of "incorrect attributions") | No |
| Perplexity | Inline URL citations | No | No (measured citation-answer mismatch) | No |
| Azure DI / AWS Textract+Bedrock / Mistral OCR | Bbox coords + confidence | Coords available, dev draws | No (provenance only) | No |

**Takeaway:** consumer QA tools give fluent answers + text-passage citations with **no
box, no verifier, no abstention** (and documented attribution errors). Cloud/OCR infra
exposes bounding boxes but only as *provenance of a retrieved chunk*, never as verified
support. The strongest partial precedents to cite and out-differentiate: **Scite**
(support classification, but over literature not your own answer), **Consensus**
(evidence-strength display, no abstention), and **Azure DI / AWS Bedrock / Mistral OCR**
(bbox provenance, no verification).

---

## 8. Positioning: The Empty Cell

Prior work assembles our pillars only pairwise:

- **Text RAG** has verify → decompose → abstain, but no images (FActScore, RefChecker,
  conformal abstention).
- **Visual RAG** has grounding, but self-verifies (unfaithfully) with the generator and
  never abstains (VisRAG, M3DocRAG, VISA, LAT).
- **Products** give citations-as-pointers that are known to mismatch (Adobe, Perplexity).

**No system — research or commercial — combines all three:** (i) region-level
bounding-box evidence on the page image, obtained *without* pixel-coordinate prediction
(snap-to-box over the PDF text layer, answering the "attention is not faithful"
critique); (ii) an *independent* verifier gate confirming the region entails the answer
(mitigating self-preference bias); and (iii) principled abstention when it does not.
The single system doing text-layer snap-to-box (Patch-to-Region, arXiv:2512.02660) is
retrieval-only and closes no verification loop. **That intersection is our contribution.**

---

## 9. Engineering View: Current Systems, Gaps, and Where We Grow

An implementation-oriented read of the same landscape, mapped onto our own stack. The
key strategic fact: **every component we need exists in isolation; nobody has wired them
into one live loop — and the loop is buildable without training a model.**

### 9.1 State of each layer of our stack

| Layer | Current systems | Maturity | What they lack (technical) |
|---|---|---|---|
| **Retrieval** | ColPali, ColQwen2 (late-interaction), DSE (single-vector), RegionRAG | Mature — works, in HF Transformers, gives page-level results + a patch heatmap for free | Stops at the page. Heatmap is a ~32x32 grid, post-hoc, unvalidated. RegionRAG makes regions only to *retrieve better*, not to cite. |
| **Grounding** | VISA, LAT, Patch-to-Region | New (2024-25), hottest and least-settled part of the stack | VISA/LAT make the **VLM predict pixel coords** — needs training + annotations, and IoU is poor even when the answer is right (SciEGQA, BBox-DocVQA). Patch-to-Region does our snap-to-box but is **retrieval-only**, never checks the answer. |
| **Verification** | FActScore, RefChecker, MiniCheck (text); SelfCheckGPT, CoVe (self-check) | Mature for text, **absent for visual/region evidence** | All text-only. Self-check variants reuse the generator model → self-preference bias, fails where it matters most. |
| **Abstention** | RefusalBench, conformal abstention, Selective VQA | Studied but siloed | Text-only or whole-answer-only. Nobody abstains at **claim x region** granularity. RefusalBench is closest and is evaluation-only, text-only. |

### 9.2 The core technical gaps (what nobody has shipped)

1. **A faithful region signal.** Every region method either trusts a raw heatmap
   (unfaithful) or trains a VLM to draw boxes (costly + inaccurate). Nobody derives
   boxes from the PDF text layer and uses the heatmap only to *rank* them.
2. **Verification wired to visual evidence.** Text RAG has verify-decompose-abstain; it
   has never been ported onto region-grounded visual claims.
3. **A separate judge.** Visual pipelines self-verify with the generator; no visual-RAG
   system uses an independent verifier.
4. **A claim x region abstention gate on the live path.** Does not exist anywhere.

### 9.3 Growth / improvement opportunities, ranked by leverage

1. **(Highest — the moat) The grounding + verification loop as one training-free
   pipeline.** Patch-to-Region proved snap-to-box works for *retrieval*; our delta is
   closing the loop — snapped box -> atomic claim -> independent judge -> abstain. It is
   the empty cell, and because it is training-free (PyMuPDF boxes + heatmap ranking +
   judge prompt), it is buildable at minor-project scale. Most other gaps here need GPU
   training we do not have.
2. **(Second) Snap-to-box engineering itself** — where real effort goes and where we can
   measurably beat Patch-to-Region:
   - IoU aggregation of heatmap -> text-layer boxes. Their reported 85%-of-failures came
     from **OCR** box mismatch; we sidestep it by using the **born-digital text layer**
     (PyMuPDF), not OCR — a genuine accuracy edge.
   - The sub-patch resolution ceiling (regions smaller than one patch) — an open failure
     mode we can at least characterize.
   - Tables/figures with no text span — the honest hard case; our fallback logic here is
     itself a contribution.
3. **(Third) Abstention calibration.** Ship a simple confidence threshold now; frame
   conformal calibration as future work (genuinely unaddressed for region-grounded
   claims). Do not over-invest for the minor project.

### 9.4 Recommended build order (each stage de-risks the next)

1. **Ingestion + snap-to-box first** (PyMuPDF text-layer boxes, heatmap ranking) — the
   novel, verifiable core, with zero model-training risk.
2. **Verifier as a prompt-driven separate model** (different from the reader) — cheap to
   prototype, high conceptual payoff.
3. **Abstention as a threshold**, measured by the locked confident-wrong + coverage
   metrics.
4. **Evaluate on SlideVQA**; consider **SciEGQA** (arXiv:2511.15090) as a second dataset
   since it is scientific-document + bbox-native, matching our domain better than
   SlideVQA.

**Bottom line for the engineer:** the competitive edge is not any single component — it
is being first to wire them into one live loop, achievable *without* training a model.
"Novel" + "actually buildable by one student" is the sweet spot this project occupies.

---

## Appendix: Citation Verification Checklist

All arXiv IDs below were fetched and confirmed against their live abstract pages.

Established / safe (✅): 2407.01449, 2406.11251, 2410.10594, 2411.04952, 2504.09795,
2412.14457, 2305.14627 (ALCE), 2210.08726 (RARR), 2305.14251, 2405.14486, 2404.10774,
2303.08896, 2309.11495, 2311.09476, 2410.21819, 2402.04788, 2405.01563, 2306.08751,
1902.10186, 1908.04626, 2301.04883 (SlideVQA).

Verified this pass (✅): 2511.12003 (LAT — full title confirmed), 2512.02660
(Patch-to-Region, Georgiou), 2510.10390 (RefusalBench), 2510.27261 (RegionRAG),
2511.15090 (**SciEGQA** — NOT BBox-DocVQA; ID corrected).

Still open (⚠️): **BBox-DocVQA** — the dataset name is real (benchmarked by
Patch-to-Region) but its own arXiv ID is not yet located; do not attach 2511.15090 to
it. Look up before citing.
