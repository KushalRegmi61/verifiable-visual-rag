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
    sim = np.asarray(query_vectors, dtype=np.float64) @ np.asarray(page_vectors, dtype=np.float64).T
    return sim[:, _image_slice(grid)].max(axis=0)
