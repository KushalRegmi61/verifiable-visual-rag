# S4 Grounding Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ground(claim, ...)`, which returns the page region supporting a claim, either as an exact text-layer span or as a candidate box selected (never drawn) by the query-to-patch heatmap.

**Architecture:** Four small modules under `src/visual_verify/grounding/`. Ranking logic is pure numpy over vectors passed in as arguments, so the package stays inside the core's four dependencies and is fully testable without Qdrant, without a GPU, and without a 21.4 s/page model load. A thin adapter in the CLI does the fetching.

**Tech Stack:** numpy, pydantic (`contracts.GroundedRegion`), existing `derive.py`, `evidence.py`, and `retrieval/geometry.py`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-08-s4-grounding-design.md`. Read it before starting. The two measured constraints in section 3 are the reason this design looks the way it does.

---

## Ground rules for every task

**Test only your own module.** Run the specific test file(s) your task names, never the full suite. The full suite runs once, in Task 11.

**The assertion rule.** Selection is asserted with `evidence.covers_text`, never with `evidence.has_ink`. All 435 word boxes on a real page contain ink, so an ink assertion passes a random selector 200/200 times. `has_ink` is only ever used to check coordinate transforms, and only with the `evidence.shift` displaced control alongside it.

**Never draw a box.** Every returned bbox must be, exactly, the bbox of a `BoxRecord` that came out of `derive.py`. If you find yourself computing a rectangle from patch coordinates and returning it, stop: that is the one thing this slice exists to not do.

**Clear bytecode caches when mutation-testing.** `find . -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +`

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/visual_verify/grounding/__init__.py` | Public surface: `ground`, `GroundingError` |
| `src/visual_verify/grounding/heatmap.py` | Query x patch relevance maps. Owns special-token exclusion. |
| `src/visual_verify/grounding/snap.py` | Patch-to-candidate weighting, candidate scoring, two-stage selection |
| `src/visual_verify/grounding/text_span.py` | Exact text path over `derive.span_boxes` |
| `src/visual_verify/grounding/core.py` | `ground()` routing and the `GroundedRegion` contract |
| `src/visual_verify/cli.py` | New `vvrag ground` subcommand and the Qdrant/embedder adapter |
| `tests/test_heatmap.py` | Relevance maps, special-token exclusion, validation |
| `tests/test_snap.py` | Weighting, scoring, two-stage selection, ambiguity rule |
| `tests/test_grounding.py` | `ground()` routing, contract, determinism |
| `tests/test_grounding_oracle.py` | Ceiling regression and random floor. Pure geometry, no GPU. |
| `tests/test_grounding_live.py` | Scoring bake-off and end-to-end. Needs corpus + GPU, skips without. |

---

### Task 1: Package skeleton and dense relevance map

**Files:**
- Create: `src/visual_verify/grounding/__init__.py`
- Create: `src/visual_verify/grounding/heatmap.py`
- Create: `tests/test_heatmap.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_heatmap.py`:

```python
"""Relevance maps, and the special-token exclusion that keeps them honest."""

import numpy as np
import pytest

from visual_verify.grounding.heatmap import dense_relevance
from visual_verify.retrieval.geometry import PatchGrid


def make_grid(n_x=4, n_y=3, offset=2, n_suffix=1):
    """A small grid with the same prefix/suffix shape as a real page."""
    return PatchGrid(n_x=n_x, n_y=n_y, offset=offset, n_vectors=offset + n_x * n_y + n_suffix)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_dense_relevance_has_one_score_per_image_patch():
    grid = make_grid()
    rng = np.random.default_rng(0)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(5, 8)))

    r = dense_relevance(query, page, grid)

    assert r.shape == (grid.n_image_patches,)


def test_dense_relevance_peaks_on_the_planted_patch():
    """A patch made identical to a query token must win its own relevance."""
    grid = make_grid()
    rng = np.random.default_rng(1)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(3, 8)))

    target_patch = 7
    page[grid.offset + target_patch] = query[1]

    r = dense_relevance(query, page, grid)

    assert int(r.argmax()) == target_patch


def test_dense_relevance_ignores_special_tokens():
    """A special token planted with the query must not shift any patch score.

    Special tokens map to no page region. If the map were built over all
    vectors, a prefix token would surface as a confidently drawn box with no
    causal link to the evidence.
    """
    grid = make_grid()
    rng = np.random.default_rng(2)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(3, 8)))

    before = dense_relevance(query, page, grid)
    page[0] = query[0]              # prefix token, index < offset
    page[-1] = query[2]             # suffix token, past the image patches
    after = dense_relevance(query, page, grid)

    assert np.array_equal(before, after)


def test_dense_relevance_rejects_a_vector_count_mismatch():
    """Silent truncation would shift the whole grid and misplace every box."""
    grid = make_grid()
    page = np.zeros((grid.n_vectors + 3, 8))
    query = np.zeros((2, 8))

    with pytest.raises(ValueError, match="does not match grid"):
        dense_relevance(query, page, grid)


def test_dense_relevance_rejects_a_dimension_mismatch():
    grid = make_grid()
    page = np.zeros((grid.n_vectors, 8))
    query = np.zeros((2, 16))

    with pytest.raises(ValueError, match="dimension"):
        dense_relevance(query, page, grid)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_heatmap.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'visual_verify.grounding'`

- [ ] **Step 3: Create the package**

Create `src/visual_verify/grounding/__init__.py`:

```python
"""Region-level grounding: pillar 2 of the project.

The heatmap RANKS candidate boxes that already exist in the document text
layer. It never draws one. Every bbox this package returns is, exactly, the
bbox of a BoxRecord that came out of derive.py.

Pure numpy over arguments: page vectors, query vectors, and the patch grid are
passed in, never fetched. That keeps the package inside the core's four
dependencies and makes the whole ranking path testable with hand-built arrays,
with no Qdrant, no GPU, and no 21.4 s/page model load.

Task 7 adds the public re-exports here. Nothing else belongs in this file:
defining a shared symbol here and importing it back from a submodule creates a
cycle that only works by definition order, which is a trap for whoever edits
it next.
"""
```

- [ ] **Step 4: Write the minimal implementation**

Create `src/visual_verify/grounding/heatmap.py`:

```python
"""Query-to-patch relevance maps.

Two maps with different jobs, and the split is forced by measurement, not
preference. See spec section 6.1.

  dense_relevance   ranks. Scores all n_image_patches, so candidates are
                    comparable at any granularity.
  attribution       explains. Each query token's MaxSim winner receives that
                    token's score, so contributions decompose the page's own
                    retrieval score. Measured on real pages, a 14-30 token query
                    lights only 4-14 distinct patches out of 736, which is
                    0.5-2% of the grid. That is far too sparse to rank lines
                    inside a block, where nearly every candidate would score 0.

BOTH maps exclude special tokens. Measured: 4-5 of every 19-30 query tokens
take their maximum on a special token, so 16-26% of a query would map onto a
fabricated rectangle if the filter were missing.
"""

import numpy as np

from visual_verify.retrieval.geometry import PatchGrid


def _validate(query_vectors: np.ndarray, page_vectors: np.ndarray, grid: PatchGrid) -> None:
    if page_vectors.shape[0] != grid.n_vectors:
        raise ValueError(
            f"page vector count {page_vectors.shape[0]} does not match grid "
            f"n_vectors {grid.n_vectors}; truncating would shift every patch box"
        )
    if query_vectors.shape[-1] != page_vectors.shape[-1]:
        raise ValueError(
            f"dimension mismatch: query is {query_vectors.shape[-1]}-d, "
            f"page is {page_vectors.shape[-1]}-d"
        )


def _image_slice(grid: PatchGrid) -> slice:
    return slice(grid.offset, grid.offset + grid.n_image_patches)


def dense_relevance(
    query_vectors: np.ndarray, page_vectors: np.ndarray, grid: PatchGrid
) -> np.ndarray:
    """Per-patch relevance: max over query tokens of (q . p).

    Shape (grid.n_image_patches,), in patch index order, so entry i pairs with
    grid.patch_bbox(i). This is the map both snap stages rank on.
    """
    _validate(query_vectors, page_vectors, grid)
    sim = np.asarray(query_vectors, dtype=np.float64) @ np.asarray(
        page_vectors, dtype=np.float64
    ).T
    return sim[:, _image_slice(grid)].max(axis=0)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_heatmap.py -q`
Expected: PASS, 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/visual_verify/grounding/ tests/test_heatmap.py
git commit -m "feat(grounding): add dense query-to-patch relevance map

Excludes special tokens via PatchGrid, which is load-bearing rather than
precautionary: measured on real pages, 4-5 of every 19-30 query tokens take
their maximum on a special token, so without the filter 16-26 percent of a
query would map onto a rectangle with no page region behind it."
```

---

### Task 2: Attribution map, for explanation only

**Files:**
- Modify: `src/visual_verify/grounding/heatmap.py`
- Modify: `tests/test_heatmap.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_heatmap.py`:

```python
from visual_verify.grounding.heatmap import attribution


def test_attribution_credits_only_the_winning_patch():
    """One query token's score lands on the one patch that won it, and nowhere else.

    Single token on purpose. With several tokens the top-credited patch is not
    necessarily the best-matching one, because credit accumulates: see
    test_attribution_sums_credit_across_tokens.
    """
    grid = make_grid()
    rng = np.random.default_rng(3)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(1, 8)))

    target_patch = 5
    page[grid.offset + target_patch] = query[0]   # exact match, similarity 1.0

    a = attribution(query, page, grid)

    assert a.shape == (grid.n_image_patches,)
    assert np.count_nonzero(a) == 1
    assert int(a.argmax()) == target_patch
    assert a[target_patch] == pytest.approx(1.0)


def test_attribution_sums_credit_across_tokens():
    """Credit accumulates per patch rather than overwriting.

    So a patch winning two tokens can outrank a patch winning one perfect
    token. That is correct for a decomposition of the page score, and it is a
    second reason this map must not be used to rank candidates: the
    highest-credited patch is not necessarily the best-matching one.
    """
    grid = make_grid()
    rng = np.random.default_rng(7)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    duplicated = unit(rng.normal(size=(8,)))
    # Two identical query tokens must both take their maximum on the same
    # patch, so the planted patch is credited twice.
    query = np.stack([duplicated, duplicated])
    page[grid.offset + 2] = duplicated

    a = attribution(query, page, grid)

    assert a[2] == pytest.approx(2.0), "credit must accumulate, not overwrite"
    assert np.count_nonzero(a) == 1


def test_attribution_is_sparse():
    """At most one patch per query token can be credited.

    This is why attribution cannot rank: a real 19-token query lights 4 of 736
    patches, so nearly every line inside a block would score exactly 0 and
    stage 2 would be breaking ties at random.
    """
    grid = make_grid(n_x=8, n_y=8, offset=2)
    rng = np.random.default_rng(4)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(3, 8)))

    a = attribution(query, page, grid)

    assert np.count_nonzero(a) <= query.shape[0]
    assert np.count_nonzero(a) < grid.n_image_patches


def test_attribution_drops_tokens_won_by_special_tokens():
    """A token whose maximum is a prefix vector credits no patch at all."""
    grid = make_grid()
    rng = np.random.default_rng(5)
    page = unit(rng.normal(size=(grid.n_vectors, 8)))
    query = unit(rng.normal(size=(2, 8)))

    page[0] = query[0]              # prefix token wins query token 0 outright
    a = attribution(query, page, grid)

    # Only query token 1 can have credited anything.
    assert np.count_nonzero(a) <= 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_heatmap.py -q`
Expected: FAIL, `ImportError: cannot import name 'attribution'`

- [ ] **Step 3: Write the minimal implementation**

Append to `src/visual_verify/grounding/heatmap.py`:

```python
def attribution(
    query_vectors: np.ndarray, page_vectors: np.ndarray, grid: PatchGrid
) -> np.ndarray:
    """Per-patch share of the page's MaxSim score. For EXPLANATION, not ranking.

    Each query token's maximum is taken over ALL vectors, exactly as MaxSim
    does, and the winning vector receives that token's score. Tokens won by
    special tokens credit nothing, which is correct: they point at no region.

    Because those tokens are dropped, the total here is a share of the
    image-patch contribution and will NOT equal the full page MaxSim score.

    Do not rank with this. See the module docstring and spec section 6.1.
    """
    _validate(query_vectors, page_vectors, grid)
    sim = np.asarray(query_vectors, dtype=np.float64) @ np.asarray(
        page_vectors, dtype=np.float64
    ).T
    winners = sim.argmax(axis=1)
    out = np.zeros(grid.n_image_patches, dtype=np.float64)
    lo, hi = grid.offset, grid.offset + grid.n_image_patches
    for token, winner in enumerate(winners):
        if lo <= winner < hi:
            out[winner - lo] += sim[token, winner]
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_heatmap.py -q`
Expected: PASS, 14 passed (10 from Task 1, plus these 4)

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/grounding/heatmap.py tests/test_heatmap.py
git commit -m "feat(grounding): add attribution map for explaining a selection

Decomposes the page's own retrieval score, so a region's share is the
fraction of the ranking it accounts for. Kept out of the ranking path
deliberately, for two independent reasons now pinned by tests. At most one
patch per query token is credited, so a real 19-token query lights 4 of 736
patches and nearly every line inside a block scores zero. And credit
accumulates, so the highest-credited patch is not necessarily the
best-matching one: two mediocre tokens on one patch outscore a single
exact match elsewhere."
```

---

### Task 3: Patch weights for a candidate box

**Files:**
- Create: `src/visual_verify/grounding/snap.py`
- Create: `tests/test_snap.py`

The subtlety here is the whole reason this is its own task. A line box is 0.0142 tall against a patch cell 0.0312 tall, so a line is **thinner than one patch row**. Selecting patches by "is the patch centre inside the box" would give many lines zero patches and a zero score. Weighting by the fraction of each patch that the box covers is what makes sub-patch candidates rankable at all.

- [ ] **Step 1: Write the failing test**

Create `tests/test_snap.py`:

```python
"""Patch-to-candidate weighting and two-stage selection."""

import numpy as np
import pytest

from visual_verify.grounding.snap import patch_weights
from visual_verify.retrieval.geometry import PatchGrid


def make_grid(n_x=4, n_y=4, offset=2, n_suffix=1):
    return PatchGrid(n_x=n_x, n_y=n_y, offset=offset, n_vectors=offset + n_x * n_y + n_suffix)


def test_full_page_box_weights_every_patch_fully():
    grid = make_grid()
    w = patch_weights(grid, (0.0, 0.0, 1.0, 1.0))

    assert w.shape == (grid.n_image_patches,)
    assert np.allclose(w, 1.0)


def test_a_box_over_one_cell_weights_only_that_cell():
    grid = make_grid()                      # 4x4, each cell 0.25 x 0.25
    w = patch_weights(grid, (0.25, 0.5, 0.5, 0.75))

    # col 1, row 2  ->  patch index row * n_x + col = 2 * 4 + 1 = 9
    assert w[9] == pytest.approx(1.0)
    assert np.count_nonzero(w) == 1


def test_a_box_thinner_than_a_patch_still_gets_weight():
    """The case that breaks centre-based selection.

    A real line box is 0.0142 tall against a 0.0312 patch cell, so it never
    contains a patch centre. Requiring one would score every line zero.
    """
    grid = make_grid()
    w = patch_weights(grid, (0.30, 0.30, 0.70, 0.31))

    assert np.count_nonzero(w) > 0
    assert w.max() < 1.0, "a sliver must not weight a full cell"


def test_weights_are_the_covered_area_fraction():
    grid = make_grid()                      # cells are 0.25 x 0.25
    # Exactly half of cell (0,0) horizontally, all of it vertically.
    w = patch_weights(grid, (0.0, 0.0, 0.125, 0.25))

    assert w[0] == pytest.approx(0.5)


def test_weights_align_with_patch_bbox():
    """patch_weights and PatchGrid.patch_bbox must agree on index order.

    A row/column transposition here reproduces the S2 patch-grid bug: every
    box lands somewhere plausible and nothing raises.
    """
    grid = make_grid(n_x=4, n_y=3)
    for idx in range(grid.n_image_patches):
        w = patch_weights(grid, grid.patch_bbox(idx))
        assert int(w.argmax()) == idx, f"patch {idx} weighted the wrong cell"
        assert w[idx] == pytest.approx(1.0)


def test_a_degenerate_box_raises():
    grid = make_grid()
    with pytest.raises(ValueError, match="positive area"):
        patch_weights(grid, (0.5, 0.5, 0.5, 0.5))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_snap.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'visual_verify.grounding.snap'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/visual_verify/grounding/snap.py`:

```python
"""Scoring and selecting candidate boxes against a relevance map.

Nothing here creates a rectangle. Every box returned came from derive.py.
"""

import numpy as np

from visual_verify.contracts import BBox
from visual_verify.retrieval.geometry import PatchGrid


def _axis_overlap(n: int, lo: float, hi: float) -> np.ndarray:
    """Fraction of each of n equal cells on [0,1] that [lo,hi] covers."""
    edges = np.arange(n + 1, dtype=np.float64) / n
    left, right = edges[:-1], edges[1:]
    covered = np.minimum(right, hi) - np.maximum(left, lo)
    return np.clip(covered, 0.0, None) * n


def patch_weights(grid: PatchGrid, bbox: BBox) -> np.ndarray:
    """Fraction of each image patch that `bbox` covers, in patch index order.

    Area fraction rather than centre containment. A real line box is 0.0142
    tall against a 0.0312 patch cell, so it contains no patch centre at all;
    centre-based selection would score every line zero and make stage 2
    meaningless.

    Index order matches PatchGrid.patch_bbox: patch i is at column i % n_x,
    row i // n_x. The transposition test in tests/test_snap.py pins this.
    """
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"candidate box must have positive area, got {bbox}")
    wx = _axis_overlap(grid.n_x, x0, x1)
    wy = _axis_overlap(grid.n_y, y0, y1)
    return np.outer(wy, wx).ravel()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_snap.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/grounding/snap.py tests/test_snap.py
git commit -m "feat(grounding): weight patches by covered area, not centre

A line box is 0.0142 tall against a 0.0312 patch cell, so it contains no
patch centre. Centre-based selection would score every line zero and leave
stage 2 with nothing to rank."
```

---

### Task 4: Score and rank candidates

**Files:**
- Modify: `src/visual_verify/grounding/snap.py`
- Modify: `tests/test_snap.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_snap.py`:

```python
from visual_verify.grounding.snap import rank_candidates, score_candidate
from visual_verify.ingest.boxes import BoxRecord


def box(x0, y0, x1, y1, text="w", kind="line", block_no=0, line_no=0):
    return BoxRecord(
        kind=kind, x0=x0, y0=y0, x1=x1, y1=y1, text=text,
        block_no=block_no, line_no=line_no, word_no=-1,
    )


def hot_map(grid, hot_index, hot=1.0, cold=0.1):
    r = np.full(grid.n_image_patches, cold)
    r[hot_index] = hot
    return r


def test_score_is_the_weighted_mean_over_covered_patches():
    grid = make_grid()                       # 4x4
    r = hot_map(grid, 9)                     # col 1, row 2
    assert score_candidate(r, grid, (0.25, 0.5, 0.5, 0.75)) == pytest.approx(1.0)


def test_mean_does_not_reward_a_larger_box():
    """Sum is monotone in area and would hand every contest to the page box."""
    grid = make_grid()
    r = hot_map(grid, 9)

    tight = score_candidate(r, grid, (0.25, 0.5, 0.5, 0.75))
    whole = score_candidate(r, grid, (0.0, 0.0, 1.0, 1.0))

    assert tight > whole


def test_sum_reduce_is_available_as_a_control():
    grid = make_grid()
    r = hot_map(grid, 9)

    tight = score_candidate(r, grid, (0.25, 0.5, 0.5, 0.75), reduce="sum")
    whole = score_candidate(r, grid, (0.0, 0.0, 1.0, 1.0), reduce="sum")

    assert whole > tight, "sum must show the area bias the bake-off measures"


def test_rank_candidates_orders_by_score_descending():
    grid = make_grid()
    r = hot_map(grid, 9)
    cold = box(0.0, 0.0, 0.25, 0.25, text="cold")
    warm = box(0.25, 0.5, 0.5, 0.75, text="warm")

    ranked = rank_candidates(r, grid, [cold, warm])

    assert [b.text for b, _ in ranked] == ["warm", "cold"]
    assert ranked[0][1] > ranked[1][1]


def test_rank_candidates_is_deterministic_on_ties():
    """Ties break by input order, never by set or dict iteration."""
    grid = make_grid()
    r = np.full(grid.n_image_patches, 0.5)
    a = box(0.0, 0.0, 0.25, 0.25, text="a")
    b = box(0.5, 0.5, 0.75, 0.75, text="b")

    first = [t for t, _ in rank_candidates(r, grid, [a, b])]
    second = [t for t, _ in rank_candidates(r, grid, [a, b])]

    assert first == second == ["a", "b"]


def test_rank_candidates_skips_a_degenerate_box_without_raising():
    """Derived boxes are trusted, but a zero-area one must not kill the query."""
    grid = make_grid()
    r = hot_map(grid, 9)
    good = box(0.25, 0.5, 0.5, 0.75, text="good")
    bad = box(0.5, 0.5, 0.5, 0.5, text="bad")

    ranked = rank_candidates(r, grid, [bad, good])

    assert [t.text for t, _ in ranked] == ["good"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_snap.py -q`
Expected: FAIL, `ImportError: cannot import name 'rank_candidates'`

- [ ] **Step 3: Write the minimal implementation**

Append to `src/visual_verify/grounding/snap.py`:

```python
from typing import Literal

from visual_verify.ingest.boxes import BoxRecord

Reduce = Literal["mean", "sum"]


def score_candidate(
    relevance: np.ndarray, grid: PatchGrid, bbox: BBox, reduce: Reduce = "mean"
) -> float:
    """Relevance of the patches `bbox` covers, weighted by how much it covers.

    "mean" is the default because "sum" is monotone in area: a box covering the
    whole page always sums highest, so sum ranking would return the page. "sum"
    is kept only as the control that demonstrates that bias in the bake-off.
    """
    w = patch_weights(grid, bbox)
    total = float(w.sum())
    if total == 0.0:
        return 0.0
    weighted = float((w * relevance).sum())
    return weighted if reduce == "sum" else weighted / total


def rank_candidates(
    relevance: np.ndarray,
    grid: PatchGrid,
    candidates: list[BoxRecord],
    reduce: Reduce = "mean",
) -> list[tuple[BoxRecord, float]]:
    """Candidates by descending score. Ties break by input order.

    Determinism matters here: the eval harness reruns this and must not see a
    different region because a set iterated differently.
    """
    scored: list[tuple[BoxRecord, float]] = []
    for c in candidates:
        if c.x1 <= c.x0 or c.y1 <= c.y0:
            continue
        scored.append((c, score_candidate(relevance, grid, (c.x0, c.y0, c.x1, c.y1), reduce)))
    # sorted() is stable, so equal scores keep their input order.
    return sorted(scored, key=lambda pair: -pair[1])
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_snap.py -q`
Expected: PASS, 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/grounding/snap.py tests/test_snap.py
git commit -m "feat(grounding): score candidates by weighted mean relevance

Mean rather than sum: sum is monotone in area, so it ranks the whole-page box
first every time. Sum stays available as the control that demonstrates the
bias in the scoring bake-off rather than leaving it asserted."
```

---

### Task 5: Two-stage selection with the ambiguity rule

**Files:**
- Modify: `src/visual_verify/grounding/snap.py`
- Modify: `tests/test_snap.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_snap.py`:

```python
from visual_verify.grounding.snap import Selection, snap_to_box


def page_boxes():
    """Two blocks, two lines each, laid out top to bottom on a 4x4 grid page."""
    return [
        box(0.05, 0.05, 0.45, 0.20, text="alpha one", block_no=0, line_no=0, kind="word"),
        box(0.05, 0.25, 0.45, 0.40, text="alpha two", block_no=0, line_no=1, kind="word"),
        box(0.05, 0.55, 0.45, 0.70, text="beta one", block_no=1, line_no=0, kind="word"),
        box(0.05, 0.75, 0.45, 0.90, text="beta two", block_no=1, line_no=1, kind="word"),
    ]


def test_snap_returns_a_line_when_one_line_clearly_wins():
    grid = make_grid()
    # Patch row 3 (y 0.75-1.0), col 0-1: the region holding "beta two".
    r = np.full(grid.n_image_patches, 0.05)
    r[12] = 1.0
    r[13] = 1.0

    sel = snap_to_box(r, grid, page_boxes())

    assert sel.resolution == "line"
    assert "beta two" in sel.box.text


def test_snap_falls_back_to_the_block_when_lines_are_indistinguishable():
    """Honest about its own resolution instead of guessing a line."""
    grid = make_grid()
    # Uniform heat over the whole lower half: block 1 wins, its lines tie.
    r = np.full(grid.n_image_patches, 0.05)
    r[8:16] = 1.0

    sel = snap_to_box(r, grid, page_boxes())

    assert sel.resolution == "block"
    assert "beta" in sel.box.text


def test_snap_stays_inside_the_winning_block():
    """A stage-2 miss must still land in the right paragraph.

    This is why selection is two-stage rather than a flat ranking over all
    lines: a flat miss can land anywhere on the page.
    """
    grid = make_grid()
    r = np.full(grid.n_image_patches, 0.05)
    r[0] = 1.0                            # top-left, inside block 0

    sel = snap_to_box(r, grid, page_boxes())

    assert "alpha" in sel.box.text
    assert "beta" not in sel.box.text


def test_snap_returns_none_when_there_are_no_candidates():
    grid = make_grid()
    r = np.full(grid.n_image_patches, 0.5)

    assert snap_to_box(r, grid, []) is None


def test_snap_never_invents_a_box():
    """The returned bbox must be one a derived candidate actually has."""
    grid = make_grid()
    r = np.full(grid.n_image_patches, 0.05)
    r[12] = 1.0

    boxes = page_boxes()
    sel = snap_to_box(r, grid, boxes)
    corners = (sel.box.x0, sel.box.y0, sel.box.x1, sel.box.y1)

    from visual_verify.derive import block_boxes, line_boxes

    allowed = {
        (b.x0, b.y0, b.x1, b.y1) for b in line_boxes(boxes) + block_boxes(boxes)
    }
    assert corners in allowed
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_snap.py -q`
Expected: FAIL, `ImportError: cannot import name 'Selection'`

- [ ] **Step 3: Write the minimal implementation**

Append to `src/visual_verify/grounding/snap.py`:

```python
from dataclasses import dataclass

from visual_verify.derive import block_boxes, line_boxes

# Relative gap below which the top two lines are treated as indistinguishable.
# The heatmap resolves roughly 3.6 lines per patch row, so a near-tie between
# two lines is the expected case, not an anomaly.
AMBIGUITY_MARGIN = 0.10

# Relevance is a mean of cosine similarities, so scores live in [-1, 1] and a
# meaningful gap is far above this floor. Without it, a near-zero top score
# becomes its own denominator and a 1e-19 gap reads as a 10 percent margin:
# noise promoted to a confident line selection.
MIN_SCORE_SCALE = 1e-6


@dataclass(frozen=True)
class Selection:
    box: BoxRecord
    score: float
    resolution: Literal["line", "block"]


def snap_to_box(
    relevance: np.ndarray,
    grid: PatchGrid,
    boxes: list[BoxRecord],
    reduce: Reduce = "mean",
    margin: float = AMBIGUITY_MARGIN,
) -> Selection | None:
    """Select a region: rank blocks, then rank lines inside the winner.

    Two stages rather than a flat ranking over every line because the error is
    then bounded. A stage-2 mistake still lands inside the correct paragraph,
    whereas a flat miss can land anywhere on the page.

    When the top two lines are within `margin` relative to each other, the
    block is returned instead. The heatmap resolves about 3.6 lines per patch
    row, so committing to a line it cannot distinguish would be a confident
    guess dressed as evidence.

    Returns None when the page has no candidates at all.
    """
    blocks = block_boxes(boxes) if boxes else []
    ranked_blocks = rank_candidates(relevance, grid, blocks, reduce)
    if not ranked_blocks:
        return None
    best_block, block_score = ranked_blocks[0]

    # Membership by block_no, not geometry. block_boxes builds a block as the
    # bounding envelope of its words, so a wrap-around paragraph's envelope can
    # enclose a figure caption belonging to a different block; centre-in-envelope
    # then feeds stage 2 lines from the wrong paragraph, which is exactly the
    # bounded-error property two stages exist to provide. derive._union carries
    # block_no onto every line, so the exact answer is already available.
    inside = [ln for ln in line_boxes(boxes) if ln.block_no == best_block.block_no]
    ranked_lines = rank_candidates(relevance, grid, inside, reduce)
    if not ranked_lines:
        return Selection(best_block, block_score, "block")

    top_line, top_score = ranked_lines[0]
    if len(ranked_lines) > 1:
        runner_up = ranked_lines[1][1]
        denominator = max(abs(top_score), MIN_SCORE_SCALE)
        if (top_score - runner_up) / denominator < margin:
            return Selection(best_block, block_score, "block")
    return Selection(top_line, top_score, "line")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_snap.py -q`
Expected: PASS, 17 passed

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/grounding/snap.py tests/test_snap.py
git commit -m "feat(grounding): select two-stage, block then line

Bounds the error: a stage-2 miss still lands in the correct paragraph, while
a flat ranking over all 61 lines can miss anywhere on the page. Falls back to
the block when the top two lines are within 10 percent, since the grid
resolves only about 3.6 lines per patch row and committing to one it cannot
distinguish would be a guess presented as evidence."
```

---

### Task 6: The exact text path

**Files:**
- Create: `src/visual_verify/grounding/text_span.py`
- Create: `tests/test_grounding.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_grounding.py`:

```python
"""The text path, and ground() routing."""

import numpy as np
import pytest

from visual_verify.grounding.text_span import text_regions
from visual_verify.ingest.boxes import BoxRecord


def word(x0, y0, x1, y1, text, block_no=0, line_no=0, word_no=0):
    return BoxRecord(
        kind="word", x0=x0, y0=y0, x1=x1, y1=y1, text=text,
        block_no=block_no, line_no=line_no, word_no=word_no,
    )


def two_line_page():
    """'Revenue grew 42 percent' over 'Margins held steady'."""
    first = ["Revenue", "grew", "42", "percent"]
    second = ["Margins", "held", "steady"]
    boxes = []
    for i, t in enumerate(first):
        boxes.append(word(0.1 + i * 0.15, 0.10, 0.22 + i * 0.15, 0.16, t, line_no=0, word_no=i))
    for i, t in enumerate(second):
        boxes.append(word(0.1 + i * 0.15, 0.30, 0.22 + i * 0.15, 0.36, t, line_no=1, word_no=i))
    return boxes


def test_text_regions_finds_an_exact_phrase():
    regions = text_regions("grew 42", two_line_page(), page=3)

    assert len(regions) == 1
    assert regions[0].modality == "text"
    assert regions[0].page == 3
    assert regions[0].text == "grew 42"


def test_text_regions_is_empty_for_an_absent_phrase():
    assert text_regions("revenue fell", two_line_page(), page=0) == []


def test_a_wrapped_phrase_returns_one_rect_per_line_not_a_union():
    """A union over a wrapped match sweeps in every word between the halves.

    On this fixture that is 5.7x the true ink area. The grounding layer must
    pass derive.span_boxes's split through untouched.
    """
    regions = text_regions("percent Margins", two_line_page(), page=0)

    assert len(regions) == 2
    for r in regions:
        assert r.bbox[3] - r.bbox[1] < 0.10, "a rect spanning both lines is a union"


def test_text_regions_score_is_exact():
    regions = text_regions("Revenue", two_line_page(), page=0)

    assert regions[0].score == 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_grounding.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'visual_verify.grounding.text_span'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/visual_verify/grounding/text_span.py`:

```python
"""The exact text path: the reliability floor.

Faithful by construction. The region is the union of the very word boxes whose
text matches the claim, so there is no ranking, no model, and nothing to be
wrong about beyond the match itself.

proposal.tex line 288 builds this before the visual path for exactly that
reason: if the visual path underperforms, the system still delivers a complete
and measured text-span citation system.
"""

from visual_verify.contracts import GroundedRegion
from visual_verify.derive import span_boxes
from visual_verify.ingest.boxes import BoxRecord

# An exact text-layer match is not a similarity, so it does not share a scale
# with a MaxSim score. 1.0 marks it as exact rather than as "very confident".
EXACT = 1.0


def text_regions(claim: str, boxes: list[BoxRecord], page: int) -> list[GroundedRegion]:
    """Regions covering `claim` verbatim, or [] if it is not on the page.

    One region per line the match spans, never a single union. A union over a
    match that wraps across a line break encloses every word in between: on the
    two-line fixture it covers 5.7x the true ink area. span_boxes already
    splits correctly; this function must not re-join.
    """
    if not boxes:
        return []
    return [
        GroundedRegion(
            page=page,
            bbox=(b.x0, b.y0, b.x1, b.y1),
            score=EXACT,
            modality="text",
            text=b.text,
        )
        for b in span_boxes(boxes, claim)
    ]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_grounding.py -q`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/grounding/text_span.py tests/test_grounding.py
git commit -m "feat(grounding): add the exact text-span path

The reliability floor from proposal.tex line 288. Passes span_boxes's
per-line split through untouched: re-joining a wrapped match into one rect
covers 5.7x the true ink area on the two-line fixture, which is fabricated
evidence rather than an imprecise box."
```

---

### Task 7: `ground()` routing

**Files:**
- Create: `src/visual_verify/grounding/core.py`
- Modify: `src/visual_verify/grounding/__init__.py`
- Modify: `tests/test_grounding.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_grounding.py`:

```python
from visual_verify.grounding import GroundingError, ground
from visual_verify.retrieval.geometry import PatchGrid


def make_grid(n_x=4, n_y=4, offset=2, n_suffix=1):
    return PatchGrid(n_x=n_x, n_y=n_y, offset=offset, n_vectors=offset + n_x * n_y + n_suffix)


def planted_vectors(grid, hot_patch, dim=8):
    """Page and query vectors whose MaxSim maximum is `hot_patch`."""
    rng = np.random.default_rng(11)
    page = rng.normal(size=(grid.n_vectors, dim))
    page /= np.linalg.norm(page, axis=1, keepdims=True)
    query = rng.normal(size=(3, dim))
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    page[grid.offset + hot_patch] = query[0]
    return page, query


def test_an_exact_match_short_circuits_without_touching_vectors():
    """Text wins by default, and the visual path is not even reachable here."""
    regions = ground("grew 42", two_line_page(), page=0)

    assert len(regions) == 1
    assert regions[0].modality == "text"


def test_force_visual_bypasses_the_text_match():
    """proposal.tex line 440 requires measuring the visual path on
    text-locatable questions, which is impossible without this."""
    grid = make_grid()
    page_v, query_v = planted_vectors(grid, hot_patch=0)

    regions = ground(
        "grew 42", two_line_page(), page=0,
        page_vectors=page_v, query_vectors=query_v, grid=grid, force="visual",
    )

    assert len(regions) == 1
    assert regions[0].modality == "visual"


def test_no_text_match_falls_back_to_the_visual_path():
    grid = make_grid()
    page_v, query_v = planted_vectors(grid, hot_patch=0)

    regions = ground(
        "not on this page at all", two_line_page(), page=0,
        page_vectors=page_v, query_vectors=query_v, grid=grid,
    )

    assert len(regions) == 1
    assert regions[0].modality == "visual"


def test_a_page_with_no_candidates_returns_nothing():
    """Absence of evidence, not weakness of evidence. S4 never abstains."""
    grid = make_grid()
    page_v, query_v = planted_vectors(grid, hot_patch=0)

    regions = ground(
        "anything", [], page=0,
        page_vectors=page_v, query_vectors=query_v, grid=grid,
    )

    assert regions == []


def test_the_visual_path_without_vectors_raises():
    """Silently returning nothing would look identical to 'no evidence here'."""
    with pytest.raises(GroundingError, match="vectors"):
        ground("not on this page at all", two_line_page(), page=0)


def test_ground_is_deterministic():
    grid = make_grid()
    page_v, query_v = planted_vectors(grid, hot_patch=0)
    args = dict(page=0, page_vectors=page_v, query_vectors=query_v, grid=grid)

    first = ground("absent phrase", two_line_page(), **args)
    second = ground("absent phrase", two_line_page(), **args)

    assert [r.bbox for r in first] == [r.bbox for r in second]


def test_a_visual_region_is_always_an_existing_candidate_box():
    """Snap-to-box, stated as an assertion: never drawn from the heatmap."""
    from visual_verify.derive import block_boxes, line_boxes

    grid = make_grid()
    page_v, query_v = planted_vectors(grid, hot_patch=0)
    boxes = two_line_page()

    regions = ground(
        "absent phrase", boxes, page=0,
        page_vectors=page_v, query_vectors=query_v, grid=grid,
    )

    allowed = {(b.x0, b.y0, b.x1, b.y1) for b in line_boxes(boxes) + block_boxes(boxes)}
    assert regions[0].bbox in allowed
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_grounding.py -q`
Expected: FAIL, `ImportError: cannot import name 'ground'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/visual_verify/grounding/core.py`:

```python
"""ground(): the public seam S5, S6, and S7 all consume.

Routing is deliberately simple and deliberately asymmetric. Text wins when it
can, because it is exact; the visual path is the fallback, and force="visual"
exists so the eval harness can measure it on questions the text path would
otherwise have answered.
"""

from typing import Literal

import numpy as np

from visual_verify.contracts import GroundedRegion
from visual_verify.grounding.heatmap import dense_relevance
from visual_verify.grounding.snap import Reduce, snap_to_box
from visual_verify.grounding.text_span import text_regions
from visual_verify.ingest.boxes import BoxRecord
from visual_verify.retrieval.geometry import PatchGrid


class GroundingError(RuntimeError):
    """Inputs that cannot produce a trustworthy region.

    Defined here rather than in __init__.py on purpose. Putting it in the
    package __init__ and importing it back from this module is a cycle that
    happens to work only because the class is defined above the re-export, so
    reordering two lines in __init__ would break it at import time.
    """


def ground(
    claim: str,
    boxes: list[BoxRecord],
    *,
    page: int,
    page_vectors: np.ndarray | None = None,
    query_vectors: np.ndarray | None = None,
    grid: PatchGrid | None = None,
    force: Literal["text", "visual"] | None = None,
    reduce: Reduce = "mean",
) -> list[GroundedRegion]:
    """Regions of `page` that support `claim`.

    An empty list means NO EVIDENCE EXISTS on this page: the claim is not in
    the text layer and the page has no candidate boxes, as on a scanned page.
    It never means the evidence looked weak. ground() applies no confidence
    threshold; proposal.tex line 381 puts abstention on the verifier's output,
    and a second threshold here would make the ablation unable to separate the
    two contributions.

    Vectors are passed in, never fetched, so this stays inside the core's four
    dependencies and is testable without Qdrant or a GPU.
    """
    if force != "visual":
        found = text_regions(claim, boxes, page)
        if found or force == "text":
            return found

    if page_vectors is None or query_vectors is None or grid is None:
        raise GroundingError(
            "the visual path needs page_vectors, query_vectors, and grid; "
            "returning no region here would be indistinguishable from "
            "'this page holds no evidence'"
        )
    if not boxes:
        return []

    relevance = dense_relevance(query_vectors, page_vectors, grid)
    selection = snap_to_box(relevance, grid, boxes, reduce)
    if selection is None:
        return []
    b = selection.box
    return [
        GroundedRegion(
            page=page,
            bbox=(b.x0, b.y0, b.x1, b.y1),
            score=selection.score,
            modality="visual",
            text=b.text or None,
        )
    ]
```

Then append to `src/visual_verify/grounding/__init__.py`:

```python
from visual_verify.grounding.core import GroundingError, ground

__all__ = ["GroundingError", "ground"]
```

Both symbols now flow one way, from `core` up to `__init__`, so there is no
import cycle to preserve by accident.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_grounding.py -q`
Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/grounding/ tests/test_grounding.py
git commit -m "feat(grounding): add ground() with text-first routing

An empty result means no evidence exists on the page, never that evidence
looked weak: proposal.tex line 381 puts abstention on the verifier, and a
second threshold here would stop the ablation separating the two layers.
Requesting the visual path without vectors raises rather than returning [],
which would be indistinguishable from a genuinely empty page."
```

---

### Task 8: Patch rectangles must land on ink

**Files:**
- Modify: `tests/test_snap.py`

`patch_bbox` is new coordinate arithmetic reaching production for the first time, and this repository has a history of silent coordinate bugs. This is the check that found every one of them.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_snap.py`:

```python
def test_patch_rectangles_land_on_the_page_ink(born_digital_pdf, tmp_path):
    """The heaviest-ink patch must sit on the text, and not next to it.

    Ink presence proves the coordinate transform, nothing more. It cannot
    prove a selection is correct, because every word box on a real page
    contains ink. See tests/test_evidence.py.
    """
    import fitz
    from PIL import Image

    from visual_verify.evidence import has_ink, ink_ratio, shift
    from visual_verify.ingest.render import render_page

    doc = fitz.open(born_digital_pdf)
    rendered = render_page(doc[0], tmp_path / "p.png", dpi=150)
    doc.close()

    grid = PatchGrid(n_x=23, n_y=32, offset=4, n_vectors=4 + 23 * 32 + 7)
    img = Image.open(rendered.path)

    ratios = [ink_ratio(img, grid.patch_bbox(i)) for i in range(grid.n_image_patches)]
    heaviest = grid.patch_bbox(int(np.argmax(ratios)))

    assert has_ink(img, heaviest)
    # Control: the same rect displaced must miss, or the assertion proves
    # nothing on a page with ink scattered across it.
    assert not has_ink(img, shift(heaviest, dy=0.5))


def test_patch_bbox_tiles_the_page_without_gaps_or_overlap():
    grid = make_grid(n_x=4, n_y=3)
    total = sum(
        (b[2] - b[0]) * (b[3] - b[1])
        for b in (grid.patch_bbox(i) for i in range(grid.n_image_patches))
    )
    assert total == pytest.approx(1.0)
```

- [ ] **Step 2: Run the test to verify it fails or passes**

Run: `uv run pytest tests/test_snap.py -q`
Expected: PASS. These pin existing `geometry.py` behaviour rather than driving new code. If either FAILS, stop and report: a coordinate bug in `patch_bbox` invalidates every region this slice produces.

- [ ] **Step 3: Commit**

```bash
git add tests/test_snap.py
git commit -m "test(grounding): pin patch rectangles against rendered ink

patch_bbox reaches production for the first time in this slice, and every
coordinate bug in this repo was found by cropping the render rather than by
reasoning about the arithmetic. Carries the displaced control, without which
'the rect has ink' holds trivially on a dense page."
```

---

### Task 9: Oracle ceiling and random floor

**Files:**
- Create: `tests/test_grounding_oracle.py`

This is the harness section 8 of the spec requires. Pure geometry: no vectors, no GPU. It establishes the two numbers every reported IoU must be quoted against.

- [ ] **Step 1: Write the failing test**

Create `tests/test_grounding_oracle.py`:

```python
"""The ceiling and the floor that every reported IoU must be quoted against.

Granularity caps IoU independently of the selector. When the gold box lies
inside the predicted box, IoU is just the ratio of their areas, so predicting a
line against a 3-word gold span cannot exceed about 0.195 however good the
selector is. Reporting an achieved IoU without this number turns a
near-ceiling result into an apparent failure. See spec section 8.
"""

import random
import statistics
from pathlib import Path

import fitz
import pytest

from visual_verify.derive import block_boxes, line_boxes, span_boxes
from visual_verify.evidence import iou
from visual_verify.ingest.boxes import extract_boxes, word_boxes

REAL_PDF = Path(__file__).parent.parent / "proposal_report" / "proposal.pdf"

pytestmark = pytest.mark.skipif(not REAL_PDF.exists(), reason="proposal.pdf not present")


def _gold_and_containers(seed=0, pages=range(2, 10)):
    """(gold span, containing line, containing block) triples from a real PDF."""
    doc = fitz.open(REAL_PDF)
    rng = random.Random(seed)
    out = []
    for pno in pages:
        boxes = extract_boxes(doc[pno])
        if len(word_boxes(boxes)) < 50:
            continue
        lines, blocks = line_boxes(boxes), block_boxes(boxes)
        for ln in lines:
            words = ln.text.split()
            if len(words) < 5:
                continue
            i = rng.randrange(0, len(words) - 3)
            golds = span_boxes(boxes, " ".join(words[i : i + 3]))
            if not golds:
                continue
            g = golds[0]
            gold = (g.x0, g.y0, g.x1, g.y1)
            cx, cy = (g.x0 + g.x1) / 2, (g.y0 + g.y1) / 2

            def holding(cands):
                hit = [c for c in cands if c.x0 <= cx <= c.x1 and c.y0 <= cy <= c.y1]
                return hit[0] if hit else None

            out.append((gold, holding(lines), holding(blocks), blocks, rng))
    doc.close()
    return out


def test_line_granularity_ceiling_is_about_0_195():
    """Regression guard on the ceiling itself.

    If candidate derivation changes, this moves, and every previously reported
    grounding number silently changes meaning.
    """
    samples = _gold_and_containers()
    scores = [
        iou(gold, (ln.x0, ln.y0, ln.x1, ln.y1)) for gold, ln, _, _, _ in samples if ln
    ]

    assert len(scores) > 50
    assert statistics.mean(scores) == pytest.approx(0.195, abs=0.03)


def test_block_granularity_ceiling_is_about_half_the_line_ceiling():
    samples = _gold_and_containers()
    block_scores = [
        iou(gold, (b.x0, b.y0, b.x1, b.y1)) for gold, _, b, _, _ in samples if b
    ]
    line_scores = [
        iou(gold, (ln.x0, ln.y0, ln.x1, ln.y1)) for gold, ln, _, _, _ in samples if ln
    ]

    assert statistics.mean(block_scores) == pytest.approx(0.097, abs=0.03)
    assert statistics.mean(block_scores) < statistics.mean(line_scores)


def test_the_random_candidate_floor_is_near_zero():
    """The baseline every grounding claim is measured against.

    A selector at or below this floor contributes nothing, whatever its IoU
    looks like in isolation.
    """
    samples = _gold_and_containers()
    scores = []
    for gold, _, _, blocks, rng in samples:
        pick = rng.choice(blocks)
        scores.append(iou(gold, (pick.x0, pick.y0, pick.x1, pick.y1)))

    assert statistics.mean(scores) < 0.02
    assert sum(s >= 0.25 for s in scores) / len(scores) < 0.05
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_grounding_oracle.py -q`
Expected: PASS, 3 passed. If the ceilings differ from 0.195 and 0.097 by more than the tolerance, stop and report: candidate derivation has changed and the spec's numbers need re-measuring.

- [ ] **Step 3: Commit**

```bash
git add tests/test_grounding_oracle.py
git commit -m "test(grounding): pin the IoU ceiling and the random floor

Granularity caps IoU independently of the selector: a perfect selector
reaches 0.195 at line level against a 3-word gold span. Pinning it makes a
later change to candidate derivation fail loudly instead of silently
changing what every previously reported grounding number meant."
```

---

### Task 10: Scoring bake-off on real vectors

**Files:**
- Create: `tests/test_grounding_live.py`

Section 6.1 of the spec requires choosing the scoring rule by measurement rather than by argument. Everything needed is already in Qdrant, so this costs one model load for the query vectors and is otherwise pure numpy.

**GPU discipline.** The GPU is single-tenant with 3.63 GB usable. Never run two model-loading processes at once. Check with `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and kill orphans before blaming the code. Free VRAM in teardown or a later module OOMs during weight loading.

- [ ] **Step 1: Write the test**

Create `tests/test_grounding_live.py`:

```python
"""End-to-end grounding against the real corpus, and the scoring bake-off.

Slow and GPU-bound: skips cleanly when there is no corpus or no GPU.
"""

import gc
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from visual_verify.config import Settings
from visual_verify.derive import block_boxes, line_boxes
from visual_verify.evidence import covers_text
from visual_verify.grounding.heatmap import dense_relevance
from visual_verify.grounding.snap import rank_candidates
from visual_verify.ingest.boxes import BoxRecord
from visual_verify.retrieval.geometry import PatchGrid
from visual_verify.retrieval.index import ORIGINAL, QdrantIndex

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def live():
    """Real page vectors, their boxes, and a real embedder.

    Module-scoped and explicitly torn down. Two live ColQwen2 instances do not
    fit in 4 GB, so a module that leaks one makes the next module OOM during
    weight loading while passing perfectly on its own.
    """
    settings = Settings.from_env()
    if not settings.qdrant_url:
        pytest.skip("VVRAG_QDRANT_URL not set")
    db = ROOT / "data" / "index.db"
    if not db.exists():
        pytest.skip("no ingested corpus at data/index.db")

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    from visual_verify.retrieval.embedder import ColQwen2Embedder

    index = QdrantIndex(settings.qdrant_url, settings.qdrant_api_key)
    points, _ = index.client.scroll(
        index.collection, limit=4, with_payload=True, with_vectors=True
    )
    if not points:
        pytest.skip("vector index is empty")

    con = sqlite3.connect(db)
    pages = []
    for p in points:
        pl = p.payload
        rows = con.execute(
            """
            SELECT b.x0, b.y0, b.x1, b.y1, b.text, b.block_no, b.line_no, b.word_no
            FROM boxes b JOIN pages p ON b.page_id = p.id
            WHERE p.doc_sha = ? AND p.page_no = ? AND b.kind = 'word'
            """,
            (pl["doc_sha"], pl["page_no"]),
        ).fetchall()
        if len(rows) < 40:
            continue
        boxes = [
            BoxRecord(
                kind="word", x0=r[0], y0=r[1], x1=r[2], y1=r[3], text=r[4],
                block_no=r[5], line_no=r[6], word_no=r[7],
            )
            for r in rows
        ]
        grid = PatchGrid(
            n_x=pl["n_patches_x"], n_y=pl["n_patches_y"],
            offset=pl["patch_offset"], n_vectors=len(p.vector[ORIGINAL]),
        )
        pages.append((np.asarray(p.vector[ORIGINAL], dtype=np.float32), grid, boxes))
    con.close()
    if not pages:
        pytest.skip("no indexed page had enough boxes")

    embedder = ColQwen2Embedder()
    yield embedder, pages

    del embedder
    gc.collect()
    torch.cuda.empty_cache()


def _longest_line(boxes):
    return max(line_boxes(boxes), key=lambda ln: len(ln.text))


def test_the_selector_beats_the_random_candidate_floor(live):
    """The headline claim of pillar 2, on real vectors.

    Uses a line's own text as the query, so the correct answer is known. The
    assertion is on the TEXT the selected region covers, never on ink: every
    candidate on the page has ink, so an ink check passes a random selector.
    """
    embedder, pages = live
    hits = 0
    for page_vectors, grid, boxes in pages:
        target = _longest_line(boxes)
        qv = embedder.embed_query(target.text)
        relevance = dense_relevance(qv, page_vectors, grid)
        ranked = rank_candidates(relevance, grid, line_boxes(boxes))
        best = ranked[0][0]
        if covers_text(boxes, (best.x0, best.y0, best.x1, best.y1), target.text):
            hits += 1

    assert hits > 0, (
        f"0/{len(pages)} pages selected the queried line; "
        "the heatmap is contributing nothing over chance"
    )


def test_query_token_counts_and_heatmap_sparsity_are_recorded(live, capsys):
    """Reports the numbers spec section 6.1 rests on, so they can be re-checked."""
    embedder, pages = live
    page_vectors, grid, _ = pages[0]

    with capsys.disabled():
        print("\n  query tokens | distinct patches lit | grid share")
        for q in ("What is the abstention threshold?", "IoU"):
            qv = embedder.embed_query(q)
            sim = qv @ page_vectors.T
            win = sim.argmax(axis=1)
            lo, hi = grid.offset, grid.offset + grid.n_image_patches
            lit = len({int(w) for w in win if lo <= w < hi})
            print(
                f"  {qv.shape[0]:>12} | {lit:>20} | "
                f"{lit / grid.n_image_patches:.2%}   {q!r}"
            )
            assert lit < grid.n_image_patches * 0.1, (
                "attribution is no longer sparse; revisit spec section 6.1, "
                "which chose dense ranking because of this"
            )


def test_dense_mean_beats_dense_sum_and_attribution(live, capsys):
    """The bake-off spec section 6.1 requires. Selects the rule by measurement.

    Records the result so the choice is demonstrated rather than asserted.
    """
    from visual_verify.grounding.heatmap import attribution

    embedder, pages = live
    tally = {"dense_mean": 0, "dense_sum": 0, "attribution_mean": 0}

    for page_vectors, grid, boxes in pages:
        target = _longest_line(boxes)
        qv = embedder.embed_query(target.text)
        lines = line_boxes(boxes)
        dense = dense_relevance(qv, page_vectors, grid)
        attrib = attribution(qv, page_vectors, grid)

        for name, relevance, reduce in (
            ("dense_mean", dense, "mean"),
            ("dense_sum", dense, "sum"),
            ("attribution_mean", attrib, "mean"),
        ):
            best = rank_candidates(relevance, grid, lines, reduce)[0][0]
            if covers_text(boxes, (best.x0, best.y0, best.x1, best.y1), target.text):
                tally[name] += 1

    with capsys.disabled():
        print(f"\n  scoring bake-off over {len(pages)} pages: {tally}")

    assert tally["dense_mean"] >= tally["dense_sum"], (
        "dense_sum matched or beat dense_mean; the area bias argument in spec "
        "section 6.1 does not hold on this corpus and the default must be revisited"
    )
    assert tally["dense_mean"] >= tally["attribution_mean"], (
        "attribution matched or beat dense ranking; spec section 6.1 chose dense "
        "on a sparsity argument that this measurement contradicts"
    )
```

- [ ] **Step 2: Check the GPU is free, then run**

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
uv run pytest tests/test_grounding_live.py -q -s
```

Expected: PASS, 3 passed, with the token counts, sparsity, and bake-off tally printed. If it skips, report which guard skipped it rather than treating a skip as a pass.

- [ ] **Step 3: Record the measured bake-off result in the spec**

Edit `docs/superpowers/specs/2026-08-08-s4-grounding-design.md`, section 6.1, replacing the three-item list of rules to compare with the measured tally, in the form:

```markdown
**Measured** (N pages, each queried with its own longest line, 2026-08-08):
dense mean X/N, dense sum Y/N, attribution mean Z/N. Dense mean is therefore
the default, demonstrated rather than argued.
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_grounding_live.py docs/superpowers/specs/2026-08-08-s4-grounding-design.md
git commit -m "test(grounding): measure the scoring bake-off on real vectors

Section 6.1 required choosing the scoring rule by measurement. Dense sum and
attribution stay in as controls so the area bias and the sparsity are shown
on the corpus rather than argued from first principles, and the test fails if
either control wins, which would mean the spec's reasoning was wrong.

Frees VRAM in teardown: two live ColQwen2 instances do not fit in 4 GB, so a
module that leaks one passes alone and OOMs the next module."
```

---

### Task 11: `vvrag ground` and the fetch adapter

**Files:**
- Modify: `src/visual_verify/cli.py`
- Modify: `tests/test_cli_retrieval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_retrieval.py`:

```python
def test_ground_command_is_registered():
    from visual_verify.cli import build_parser

    args = build_parser().parse_args(["ground", "some claim", "--doc", "abc", "--page", "3"])

    assert args.claim == "some claim"
    assert args.doc == "abc"
    assert args.page == 3
    assert args.force_visual is False


def test_ground_command_accepts_force_visual_and_overlay():
    from visual_verify.cli import build_parser

    args = build_parser().parse_args(
        ["ground", "c", "--doc", "abc", "--page", "1", "--force-visual", "--overlay", "o.png"]
    )

    assert args.force_visual is True
    assert args.overlay == "o.png"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_cli_retrieval.py -q -k ground`
Expected: FAIL, `SystemExit: 2` (argparse rejects the unknown command "ground")

- [ ] **Step 3: Write the minimal implementation**

Add to `src/visual_verify/cli.py`, after `cmd_search`:

```python
def cmd_ground(args: argparse.Namespace) -> int:
    """Ground a claim to a region of one page.

    This is the adapter: it fetches vectors and geometry so that the grounding
    package never has to. Everything it hands over is a plain array or a value
    object, which is what keeps grounding inside the core's four dependencies.
    """
    from visual_verify.grounding import GroundingError, ground
    from visual_verify.retrieval.geometry import PatchGrid
    from visual_verify.retrieval.index import ORIGINAL

    settings = Settings.from_env()
    with _session(settings) as session:
        found = _resolve_document(session, args.doc)
        if found is None or isinstance(found, list):
            print(f"no unique document matching {args.doc!r}")
            return 1
        doc = found
        page = session.scalar(
            select(Page).where(Page.doc_sha == doc.sha256, Page.page_no == args.page)
        )
        if page is None:
            print(f"no page {args.page} in {Path(doc.path).name}")
            return 1
        boxes = [
            _to_record(b)
            for b in session.scalars(select(Box).where(Box.page_id == page.id, Box.kind == "word"))
        ]
        image_path = settings.pages_dir / page.image_path

    page_vectors = query_vectors = grid = None
    # Only pay for the model and the fetch when the visual path can be reached.
    if args.force_visual or not derive.span_boxes(boxes, args.claim):
        index = _make_index(settings)
        payload = index.get_payload(doc.sha256, args.page)
        stored = index.get_vectors(doc.sha256, args.page)[ORIGINAL]
        grid = PatchGrid(
            n_x=payload["n_patches_x"],
            n_y=payload["n_patches_y"],
            offset=payload["patch_offset"],
            n_vectors=stored.shape[0],
        )
        page_vectors = stored
        query_vectors = _make_embedder(settings).embed_query(args.claim)

    try:
        regions = ground(
            args.claim,
            boxes,
            page=args.page,
            page_vectors=page_vectors,
            query_vectors=query_vectors,
            grid=grid,
            force="visual" if args.force_visual else None,
        )
    except GroundingError as exc:
        print(f"cannot ground: {exc}")
        return 1

    if not regions:
        print("no evidence for this claim on this page")
        return 0

    for r in regions:
        x0, y0, x1, y1 = r.bbox
        print(
            f"{r.modality:<6} score {r.score:7.3f}  "
            f"[{x0:.3f} {y0:.3f} {x1:.3f} {y1:.3f}]  {(r.text or '')[:60]}"
        )

    if args.overlay:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for r in regions:
            x0, y0, x1, y1 = r.bbox
            draw.rectangle(
                [x0 * img.width, y0 * img.height, x1 * img.width, y1 * img.height],
                outline=(226, 10, 22) if r.modality == "visual" else (16, 128, 64),
                width=3,
            )
        img.save(args.overlay)
        print(f"wrote {args.overlay}")
    return 0
```

Then register it in `build_parser`, immediately before `return parser`:

```python
    p_ground = sub.add_parser("ground", help="ground a claim to a region of a page")
    p_ground.add_argument("claim", help="the claim to find evidence for")
    p_ground.add_argument("--doc", required=True, help="document sha256, prefix, or path substring")
    p_ground.add_argument("--page", type=int, required=True)
    p_ground.add_argument(
        "--force-visual",
        action="store_true",
        help="use snap-to-box even when the claim is in the text layer (what the eval does)",
    )
    p_ground.add_argument("--overlay", help="write a PNG with the region drawn on the page")
    p_ground.set_defaults(func=cmd_ground)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_cli_retrieval.py -q -k ground`
Expected: PASS, 2 passed

- [ ] **Step 5: Try it by hand and look at the picture**

```bash
uv run vvrag status                       # pick a doc sha and a page with text
uv run vvrag ground "<a phrase you can see on that page>" --doc <sha> --page <n> --overlay /tmp/g.png
```

Expected: a `text` region printed, and `/tmp/g.png` showing a green box on that phrase. Then force the visual path on the same claim:

```bash
uv run vvrag ground "<same phrase>" --doc <sha> --page <n> --force-visual --overlay /tmp/gv.png
```

Expected: a `visual` region, red box, covering the line or block containing the phrase. **Open both PNGs and look at them.** No numeric test replaces seeing whether the rectangle sits on the words.

- [ ] **Step 6: Commit**

```bash
git add src/visual_verify/cli.py tests/test_cli_retrieval.py
git commit -m "feat(cli): add vvrag ground with region overlay

The adapter that keeps grounding core-light: the CLI fetches vectors and
geometry and hands over plain arrays, so the package itself never imports
qdrant_client or torch. Skips the fetch and the model load entirely when the
text path already matched.

--overlay is the human check. Every coordinate bug in this repo was found by
looking at a rendered box, not by reasoning about the arithmetic."
```

---

### Task 12: Boundary check, full suite, and documentation

**Files:**
- Modify: `tests/test_core_is_light.py`
- Modify: `docs/ROADMAP.md`
- Modify: `README.md`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_core_is_light.py`:

```python
def test_grounding_pulls_no_store_or_model_dependency():
    """Grounding must stay usable without Qdrant, torch, or a GPU.

    If this fails, something in grounding/ started fetching its own inputs
    instead of taking them as arguments.
    """
    loaded = _modules_after_importing("visual_verify.grounding")
    leaked = [m for m in FORBIDDEN if m in loaded]
    assert not leaked, f"grounding leaked heavy deps: {leaked}"
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_core_is_light.py -q`
Expected: PASS, 2 passed

- [ ] **Step 3: Run the full suite, once**

```bash
find . -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
uv run pytest -q
```

Expected: all pass, roughly 270 tests, about 9 minutes. A CUDA OOM during weight loading means a model-holding module did not free VRAM in teardown, or an orphaned process is holding the card.

- [ ] **Step 4: Update the roadmap**

In `docs/ROADMAP.md`, change the S4 heading from `## S4: Grounding core (not started)` to `## S4: Grounding core (done)`, tick every S4 checkbox, and update the slice table row for S4 to `Done`. Update the counts in any summary line that names how many boxes are ticked.

- [ ] **Step 5: Document the command in README.md**

Add to the commands section, after the `vvrag search` line:

```bash
uv run vvrag ground "<claim>" --doc <sha> --page <n> --overlay out.png
uv run vvrag ground "<claim>" --doc <sha> --page <n> --force-visual   # what eval measures
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_core_is_light.py docs/ROADMAP.md README.md
git commit -m "test(grounding): enforce the core boundary and close out S4

The boundary test is what keeps ground() honest about taking vectors as
arguments: the moment it fetches its own, grounding needs Qdrant and a GPU to
run at all and stops being testable with hand-built arrays."
```

Do NOT `git add CLAUDE.md`. It is gitignored and must stay that way.

---

## Definition of done

- [ ] `ground()` returns text regions for an exact match and a snapped visual region otherwise
- [ ] Every visual bbox is provably one of `derive.py`'s candidate boxes, asserted in Tasks 5 and 7
- [ ] Special tokens excluded before every argmax, asserted in Tasks 1 and 2
- [ ] Selection asserted with `covers_text`; `has_ink` used only for coordinate checks, and always with the displaced control
- [ ] Ceiling (0.195 line, 0.097 block) and random floor pinned as regression tests
- [ ] The scoring rule chosen by a recorded measurement, with both losing rules kept as controls
- [ ] `vvrag ground --overlay` produces a picture that has been looked at
- [ ] `tests/test_core_is_light.py` proves grounding imports neither `qdrant_client` nor `torch`
- [ ] Full suite green, run once
- [ ] No reader, no verifier, no confidence threshold, no box drawn from pixels
