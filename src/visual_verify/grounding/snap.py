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
