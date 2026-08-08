"""Scoring and selecting candidate boxes against a relevance map.

Nothing here creates a rectangle. Every box returned came from derive.py.
"""

from typing import Literal

import numpy as np

from visual_verify.contracts import BBox
from visual_verify.ingest.boxes import BoxRecord
from visual_verify.retrieval.geometry import PatchGrid

Reduce = Literal["mean", "sum"]


def _axis_overlap(n: int, lo: float, hi: float) -> np.ndarray:
    """Fraction of each of n equal cells on [0,1] that [lo,hi] covers."""
    edges = np.arange(n + 1, dtype=np.float64) / n
    left, right = edges[:-1], edges[1:]
    covered = np.minimum(right, hi) - np.maximum(left, lo)
    # Mathematically covered <= right - left = 1/n, so the * n below is <= 1.0.
    # Float roundoff can still push a full-cell weight a few parts in 1e13
    # above 1.0 (measured: 1.0000000000004399 at large n). That is noise, not
    # a bug, and harmless in a weighted mean, so it is deliberately left
    # unclipped here.
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
