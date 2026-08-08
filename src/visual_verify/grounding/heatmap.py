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
    # A real query never has fewer than 14 tokens: ColQwen2 prepends a fixed
    # prompt prefix, so even a 3-character query embeds to 14 tokens. A 1-D
    # or empty query is therefore always a call-site bug, never legitimate
    # input, and left unchecked it fails deep in numpy (IndexError on a 1-D
    # array, "zero-size array to reduction operation" on an empty one)
    # instead of pointing at the actual mistake.
    if query_vectors.ndim != 2 or query_vectors.shape[0] == 0:
        raise ValueError(
            f"query_vectors must be 2-D with at least one token, got shape "
            f"{query_vectors.shape}; a real query never has fewer than 14 tokens"
        )
    # NaN compares greater than every real score, so np.argmax on a relevance
    # array with even one NaN patch silently returns that patch as the
    # top-ranked candidate: no exception, no shape anomaly, nothing about the
    # output looking wrong. Inf is rejected for the same reason it would win
    # or poison every downstream comparison.
    if not np.isfinite(page_vectors).all():
        raise ValueError(
            "page vectors contain NaN or Inf; argmax would silently select the "
            "corrupted patch, since NaN compares greater than every real score"
        )
    if not np.isfinite(query_vectors).all():
        raise ValueError(
            "query vectors contain NaN or Inf; argmax would silently select the "
            "corrupted patch, since NaN compares greater than every real score"
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
