# Multi-page reading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** The reader sees the top 3 retrieved pages instead of 1, each claim is grounded against all 3, and the page viewer follows whichever page a claim's evidence landed on.

**Architecture:** One reader call carrying 3 page images. Provenance comes from the EVIDENCE, not from the model: each claim is grounded against every prepared page and keeps the best region, so a misattribution by the reader cannot send grounding to the wrong page. Verification runs against the page the winning region came from.

**Tech Stack:** Python 3.12, pydantic v2, pytest, uv. Next.js 16, React 19, TypeScript, vitest.

---

## Decisions already made, do not relitigate

- **k = 3**, from retrieval's existing k=5.
- **Same document only.** The 3 pages are the top-ranked pages belonging to the TOP HIT's document. Retrieval is corpus-wide, so the raw top 3 can span documents, and `GroundedRegion` carries `page` but no document identity, so a cross-document region could not be rendered. Cross-document is a later slice and needs `GroundedRegion.doc_sha`.
- **The reader is never asked which page a claim came from.** A model that misattributes sends grounding to the wrong page's boxes and fabricates a citation, which is the defect fixed in `e7b12e6`. Grounding decides provenance.
- **The viewer follows the claim.** Hovering or selecting a sentence switches the page image to that claim's page.

## Two things that will bite

**Score scales are not comparable.** A text-path region's `score` comes from span matching; a visual region's comes from MaxSim. Taking `max()` across both is meaningless. The rule is: if ANY page yields a text-path region, use text-path regions and break ties by retrieval rank; only when no page yields one do we compare visual scores, which are the same quantity across pages.

**The cache key currently covers one image.** `_digest(model_id, prompt, image_path, schema_name)`. With three images, two different page sets produce the same key and one silently returns the other's answer. The digest must cover every image, in order, length-prefixed for the reason its own docstring already gives.

## Baseline

`uv run pytest -q --ignore=tests/test_embedder.py --ignore=tests/test_known_item_retrieval.py --ignore=tests/test_grounding_live.py` is **506 passed, 12 skipped** without keys and **518 passed** with them. Frontend `npx vitest run` is 30 passed.

Run only the files your task touches. Never the whole suite; three modules load a vision model onto a 4 GB GPU and exceed the tool timeout.

---

## Task 1: Let the chat seam carry several images

**Files:** `src/visual_verify/agent/types.py`, `src/visual_verify/agent/models.py`, `src/visual_verify/agent/cache.py`, `tests/test_agent_types.py`, `tests/test_agent_cache.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_agent_cache.py`:

```python
def test_two_different_page_sets_do_not_share_a_cache_entry(tmp_path):
    """THE test of this change. The digest used to cover ONE image, so a call
    over pages [7, 8, 9] and a call over pages [7, 8] hashed identically and
    the second silently returned the first's answer. A cached wrong answer is
    the worst failure this module can produce, because it is reproducible."""
    from visual_verify.agent.cache import CachedChat

    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    inner = FakeChat("m", [claim_list("from both"), claim_list("from one")])
    chat = CachedChat(inner, tmp_path / "cache")

    assert chat.structured("p", [a, b], ClaimList).claims[0].text == "from both"
    assert chat.structured("p", [a], ClaimList).claims[0].text == "from one"


def test_image_order_changes_the_cache_key(tmp_path):
    """[a, b] and [b, a] are different prompts to a vision model: the pages are
    numbered in the text and the model reads them in order."""
    from visual_verify.agent.cache import CachedChat

    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    inner = FakeChat("m", [claim_list("ab"), claim_list("ba")])
    chat = CachedChat(inner, tmp_path / "cache")

    assert chat.structured("p", [a, b], ClaimList).claims[0].text == "ab"
    assert chat.structured("p", [b, a], ClaimList).claims[0].text == "ba"
```

In `tests/test_agent_types.py`:

```python
def test_fake_chat_records_every_image_it_was_given(tmp_path):
    chat = FakeChat("m", [ClaimList(claims=["a"])])
    chat.structured("p", [tmp_path / "one.png", tmp_path / "two.png"], ClaimList)

    assert chat.calls[0].image_paths == [tmp_path / "one.png", tmp_path / "two.png"]
```

- [ ] **Step 2: Run them, confirm they fail**

`uv run pytest tests/test_agent_cache.py tests/test_agent_types.py -q`

- [ ] **Step 3: Widen the protocol**

In `types.py`, `StructuredChat.structured` takes `image_paths: list[Path]` instead of `image_path: Path | None`. An empty list replaces `None`. `RecordedCall.image_path` becomes `image_paths: list[Path]`. Update `FakeChat` to match.

Do NOT keep a backwards-compatible single-image overload. One spelling, and the type checker then finds every call site.

- [ ] **Step 4: Send them all**

In `models.py`, loop over `image_paths` appending one `image_url` content block each. The block list already supports it.

- [ ] **Step 5: Fix the cache digest**

`_digest(model_id, prompt, image_paths, schema_name)`. Feed every path's BYTES into the hash in order, keeping the existing length-prefixing. Hashing the path string rather than the content would key on a filename, and page images live under a sha-named directory that changes when the document is recompiled.

- [ ] **Step 6: Update every call site**

`reader.py`, `verifier.py` and their tests now pass a one-element list. `uv run pytest tests/test_reader.py tests/test_agent_cache.py tests/test_agent_types.py tests/test_agent.py tests/test_answer_stream.py -q`

- [ ] **Step 7: Mutation check**

Make `_digest` hash only `image_paths[0]`, clear `__pycache__`, confirm `test_two_different_page_sets_do_not_share_a_cache_entry` FAILS. Restore, clear, confirm it passes.

- [ ] **Step 8: Commit**

`feat(agent): let one call carry several page images`

---

## Task 2: Prepare the top pages of one document

**Files:** `src/visual_verify/prepare.py`, `src/visual_verify/api/ask.py`, `tests/test_prepare.py`, `tests/test_api_ask.py`

- [ ] **Step 1: Write the failing test**

```python
def test_only_pages_of_the_top_hits_document_are_prepared():
    """Retrieval is corpus-wide, so the raw top 3 can span documents, and a
    GroundedRegion carries `page` but no document identity: a region from
    another document could not be rendered, and merging two documents into one
    answer hides that it happened. The top hit's document wins and the rest are
    dropped."""
```

Build hits for doc A page 3, doc B page 7, doc A page 9, and assert the prepared pages are A/3 and A/9 in that order.

- [ ] **Step 2: Implement `prepare_pages`**

`prepare_pages(session, index, settings, hits, limit=3) -> list[PreparedPage]`, filtering `hits` to `hits[0].doc_id` and preparing at most `limit`, preserving retrieval order. Reuse `prepare_page` per page.

- [ ] **Step 3: Use it in `ask.py`**

`_choose_page` becomes `_choose_pages`, returning a `Retrieved` whose `page` stays the top page (the wire and the UI's initial view depend on it) plus a new `pages: list[PreparedPage]`. The unembedded-page warning fires when the TOP page has no vectors, unchanged.

- [ ] **Step 4: Tests, then commit**

`uv run pytest tests/test_prepare.py tests/test_api_ask.py -q`
`feat(api): prepare the top pages of the top hit's document`

---

## Task 3: Ground each claim across every prepared page

**Files:** `src/visual_verify/agent/core.py`, `tests/test_answer_stream.py`

This is the heart of the change. Read `ground()`'s docstring first; it must not change.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_text_hit_on_a_later_page_beats_a_visual_hit_on_the_first():
    """Score scales are not comparable. A text-path region's score comes from
    span matching and a visual one's from MaxSim, so max() across both is
    meaningless. An exact match wins outright wherever it is."""


def test_the_claim_is_verified_against_the_page_its_region_came_from():
    """verify() takes ONE image. Handing it the top page while the region is on
    page 3 asks it to check a box it cannot see, which is a fabricated citation
    by a different route."""


def test_visual_scores_are_compared_across_pages_when_no_page_has_a_text_hit():
```

- [ ] **Step 2: Implement `_best_region`**

```python
def _best_region(claim, pages, query_vectors):
    """The strongest region for `claim` across every prepared page.

    Text first, everywhere, because an exact span match is not a score on the
    same scale as MaxSim and beats it categorically. Ties break by retrieval
    order, which is why `pages` must stay ranked. Only when no page yields a
    text region do visual scores compete, and those ARE the same quantity
    across pages.

    Returns (page, regions). Regions is empty when nothing was found, which is
    the same meaning ground() gives it.
    """
```

Call `ground(..., force="text")` per page first, then `ground(...)` unforced per page for the visual pass. Catch `GroundingError` per page so one unembedded page cannot lose the others.

- [ ] **Step 3: Use it in the loop**

`_stream` takes `pages: list[PreparedPage]` instead of `image_path`, `boxes`, `page`, `page_vectors`, `grid`. Per claim: one `embed_query`, then `_best_region`, then `verify(verifier_chat, winning_page.image_path, text, regions)`.

Keep the citation filter and `abstained=score < threshold or not regions` exactly as they are.

- [ ] **Step 4: Tests, then commit**

`feat(agent): ground each claim against every page that was read`

---

## Task 4: Send the read pages to the browser

**Files:** `src/visual_verify/api/wire.py`, `frontend/lib/api.ts`, `tests/test_api_wire.py`

- [ ] **Step 1: Failing test**

The `retrieved` frame gains `pages: [{doc_sha, page}]`, the pages the reader actually saw, in order. The frontend needs it to build an image URL for a claim grounded away from the top page.

- [ ] **Step 2: Implement, test, commit**

`feat(api): tell the browser which pages the reader read`

---

## Task 5: The viewer follows the claim

**Files:** `frontend/lib/claims.ts`, `frontend/lib/claims.test.ts`, `frontend/app/page.tsx`, `frontend/components/PageViewer.tsx`

- [ ] **Step 1: Failing test for the pure part**

```ts
// pageForClaim(claim, pages, fallback) -> {doc_sha, page}
// A claim whose regions are empty has no page of its own and must fall back to
// the page on screen rather than blanking the viewer.
```

- [ ] **Step 2: Wire it up**

`page.tsx` holds `viewing: {doc_sha, page}` defaulting to the top page. Hovering or selecting a claim sets it to `pageForClaim(...)`. `PageViewer` draws only the regions whose `page` matches what is displayed.

Page images are already cached per page by the retrieved-pages work, so switching costs no request.

- [ ] **Step 3: Verify in the browser**

Ask a question whose answer spans two pages. Confirm hovering a sentence switches the image and lights its box, and that a claim with no region leaves the viewer where it was.

- [ ] **Step 4: Commit**

`feat(ui): follow the claim to the page its evidence is on`

---

## Non-goals

- **Cross-document answers.** Needs `GroundedRegion.doc_sha` and a UI that says which document each sentence came from.
- **Re-ranking pages by what the reader used.** Retrieval order stands.
- **Showing all three pages at once.** One viewer, following the claim.
