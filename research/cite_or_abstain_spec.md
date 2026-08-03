# Cite-or-Abstain — Minor Project Spec

**A browser extension that fact-checks what you're reading — and refuses to assert anything it can't
ground in a source.** On any article, blog, or social post, it pulls out the factual claims, verifies
each with grounded web search, and labels them ✅ verified (with the source) / ⚠️ unverified /
❌ contradicted. When it can't find evidence, it says **"can't verify"** instead of guessing.

> The identity — and the whole pitch — is *refusal to answer without grounding.* Every other AI
> extension summarizes and asserts; this one's defining feature is **calibrated abstention.** That's
> the builder's differentiating skill (eval / verification / abstention) turned into a shippable,
> free, one-click product.

Team of 3 · ~1 month part-time · English-first.

---

## 0. Two honest truths this plan is built around (from the research)

1. **Distribution is the real project, harder than the build.** Verified repeatedly: products ship and
   get *zero* users because the team keeps coding instead of doing user acquisition. → We budget **equal
   effort for getting users** (Lane 3 owns it), and the form factor is deliberately **free + frictionless
   + one-click shareable** (a browser extension) so there's nothing blocking adoption.
2. **This is NOT a proven FAANG résumé cheat code.** The claims "shipped products beat papers for hiring"
   and "labs value LangGraph multi-agent" were both refuted in research. We build this for *genuine
   shipped-product evidence + real skill in verification systems* — not because a recruiter screens for it.
   Honest framing only.

---

## 1. Input → Output

**INPUT:** the page the user is on (article / blog / Reddit / X / news), via a one-click "Check this page."
Optionally a single highlighted sentence to check just that.

**OUTPUT (inline, on the page):**
- Each factual claim gets a colored badge: ✅ **verified** (hover → the supporting source + quote),
  ⚠️ **unverified** (no sufficient evidence found → *abstained*), ❌ **contradicted** (a source disputes it).
- A page-level **trust summary**: N claims · X verified · Y unverified · Z contradicted.
- A confidence score per claim, and an explicit **abstain** state — never a confident guess.

The contract: **no claim is marked "verified" unless it round-trips to a real, quotable source.**
Everything else is honestly labeled unverified.

---

## 2. The engine (agentic verify-or-abstain — your moat, training-free)

A bounded per-claim pipeline (the research confirmed all of these are deployable with no fine-tuning):

1. **Extract claims** — from page text, pull discrete checkable factual assertions (drop opinions/fluff).
2. **Plan verification** — Chain-of-Verification style: generate targeted search queries per claim.
3. **Retrieve** — web search (Tavily/Brave/SerpAPI) → candidate evidence passages.
4. **Judge groundedness** — LLM-as-judge: does the evidence *support*, *contradict*, or *not address* the claim?
5. **Calibrate + abstain** — threshold on judge confidence / evidence agreement; below threshold → **⚠️ abstain**,
   not a guess. (Conformal-style thresholding on a small labeled calibration set.)
6. **Attach citation** — verified claims carry the source URL + the exact supporting quote.

Bounded concurrency, typed results, deliver-or-fail per claim. This is your CareGene citation-invariant +
abstention discipline, productized.

---

## 3. Architecture & the optional artifact

```
[ Browser Extension (UI) ]  ⇄  [ Backend API ]  ⇄  [ verify-or-abstain engine ]  ⇄  [ web search ]
  content script: claims                FastAPI            extract→plan→retrieve→
  on page, badges, summary                                 judge→calibrate→cite
```

- **Extension (frontend):** content script that finds claims' text spans, injects badges, renders the
  trust summary; popup for settings. Manifest V3.
- **Backend:** FastAPI endpoint `POST /check` (page text → list of `ClaimResult`). Caches by claim hash.
- **Engine = a standalone module** (same day-1-seam discipline as before). Carve it so it can be published
  as a small open-source **`groundedness`/`cite-or-abstain` library** at the end — *this is the merge of
  idea ① + ②: ship the product AND a dev artifact from one engine.* (Optional; only if the seam stays clean.)

**Frozen contracts (week 1, Pydantic v2):**
```python
Claim(text: str, span: tuple[int,int])
Evidence(url: str, quote: str, stance: Literal["support","contradict","neutral"])
ClaimResult(claim: Claim, verdict: Literal["verified","unverified","contradicted"],
            confidence: float, abstained: bool, evidence: list[Evidence])
```

---

## 4. Team split — 3 lanes, clean seams

| Lane | Owns | Scope |
|---|---|---|
| **L1 — Engine** | The verify-or-abstain module | claim extraction · CoVe query planning · search · LLM-judge · abstention calibration · the publishable library |
| **L2 — Extension + Backend** | Product surface | Manifest-V3 extension (content script, badges, trust summary, popup) · FastAPI `/check` · caching · the demo |
| **L3 — Eval + Distribution** | Credibility + USERS | accuracy/abstention eval on a labeled claim set · **AND the user-acquisition plan (this is half the project)** |

Seam: L2 calls L1 only through `POST /check` → `ClaimResult`. L3 measures L1 via the same contract.

---

## 5. Eval (so "verified" means something)
- Build a small labeled set: ~150–300 claims tagged true / false / unverifiable (mix of news + known
  misinformation + things genuinely not on the web).
- **Metrics:** verification accuracy on checkable claims · **abstention calibration** (does it abstain on the
  unverifiable ones instead of guessing?) · citation faithfulness (does the quote actually support the verdict?)
  · false-"verified" rate (the cardinal sin — keep near zero).
- This eval doubles as the open-source library's benchmark number.

---

## 6. Distribution plan — Lane 3's real job (budget = build effort)

Evidence-backed, ranked for an English-first team with no audience:
1. **Chrome Web Store, SEO-optimized listing** — the store is itself a discovery engine (0→10k installs
   documented, no ads). Keyword-first title/description; good screenshots; a demo GIF.
2. **One-shot launch:** Product Hunt + Show HN + an X thread with a 20-second demo video.
3. **Anchor communities, hands-on:** personally walk a few specific communities (a journalism/edu group,
   r/skeptic-type subs, NTK's NCIT Discord) through using it → word-of-mouth. The #1 tactic in the research.
4. **Measure real adoption:** weekly active installs, checks run, retention — these numbers ARE the deliverable.

Honest caveat baked in: cold-start English launch from Nepal is uncertain; the two big case studies had
warm audiences / non-English localization. Treat any single launch as a coin flip; the durable channel is
store-SEO + steady community onboarding.

---

## 7. 4-week timeline (part-time)

| Wk | L1 Engine | L2 Extension+Backend | L3 Eval+Distribution |
|---|---|---|---|
| 1 | **freeze contracts** · claim extraction + search prototype | extension skeleton · `/check` stub · badge render | label-set v1 · draft store listing + demo script |
| 2 | judge + abstention calibration | claims → badges end-to-end on real pages | accuracy/abstention harness live |
| 3 | tighten false-verified rate · package the module | trust summary · popup · caching · polish | **Web Store submission** · launch assets · onboard 1–2 anchor groups |
| 4 | publish `cite-or-abstain` lib (if seam clean) | demo polish · bug bash | **launch (PH/HN/X)** · track installs/retention · writeup |

---

## 8. Résumé payload (honest)
Agentic verification pipeline (CoVe + LLM-as-judge) · **calibrated abstention** · citation-faithfulness eval ·
a **shipped product with real, measurable installs** · an optional open-source groundedness library.
Pairs with **Breaker** (your solo agent-security flex): Breaker = unique/badass, this = a real product people
actually use. Don't oversell it as a hiring guarantee — sell it as *"I ship trustworthy AI systems and got
strangers to use one."*
