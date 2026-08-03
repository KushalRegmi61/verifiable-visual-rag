"""Row and column mean pooling for the two-stage prefetch representation.

Written in S3 but unused until a later slice. It is here because Qdrant cannot
add a named vector to an existing collection without recreating it, and
recreating means re-upserting every point. Computing these now costs one array
mean against a 21.4 s embed, so provisioning them is effectively free insurance
against a re-index later.

Mean specifically, not max. Published results for this scheme report NDCG@20 of
0.952 for mean pooling against 0.759 for max, so max is not a viable variant and
is deliberately not offered as an option.

Special tokens are carried through unpooled. They score meaningfully in MaxSim
even though they map to no page region, so dropping them would change retrieval
behaviour rather than merely compress it.
"""

import numpy as np

from visual_verify.retrieval.geometry import PatchGrid


def _normalize(v: np.ndarray) -> np.ndarray:
    """Unit-normalize rows. Qdrant's Cosine distance normalizes internally, but
    doing it here keeps stored vectors comparable to locally computed ones, which
    is what the index-agrees-with-numpy test relies on."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-12)


def _split(vectors: np.ndarray, grid: PatchGrid) -> tuple[np.ndarray, np.ndarray]:
    if vectors.shape[0] != grid.n_vectors:
        raise ValueError(f"expected {grid.n_vectors} vectors, got {vectors.shape[0]}")
    start = grid.offset
    end = start + grid.n_image_patches
    patches = vectors[start:end]
    specials = np.concatenate([vectors[:start], vectors[end:]])
    return patches, specials


def mean_pool_rows(vectors: np.ndarray, grid: PatchGrid) -> np.ndarray:
    """One vector per grid ROW, then the special tokens. Shape (n_y + n_special, d).

    patches.reshape(n_y, n_x, d) puts rows on axis 0 and columns on axis 1,
    matching PatchGrid's row-major layout (col = i % n_x, row = i // n_x). A
    per-row mean therefore averages over axis 1 (the columns within each row).
    """
    patches, specials = _split(vectors, grid)
    pooled = patches.reshape(grid.n_y, grid.n_x, -1).mean(axis=1)
    return _normalize(np.concatenate([pooled, specials]))


def mean_pool_cols(vectors: np.ndarray, grid: PatchGrid) -> np.ndarray:
    """One vector per grid COLUMN, then the special tokens. Shape (n_x + n_special, d).

    Same reshape as mean_pool_rows (rows on axis 0, columns on axis 1), so a
    per-column mean averages over axis 0 (the rows within each column).
    """
    patches, specials = _split(vectors, grid)
    pooled = patches.reshape(grid.n_y, grid.n_x, -1).mean(axis=0)
    return _normalize(np.concatenate([pooled, specials]))
