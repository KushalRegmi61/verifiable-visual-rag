"""The embedder seam.

`Embedder` is the interface the pipeline and the CLI depend on. `FakeEmbedder`
implements it with seeded noise, so resumption, provenance refusal, CLI plumbing
and Qdrant round-tripping are all testable in CI with no GPU, no model download,
and no 21 s per page.

That matters more here than in most projects: the real embedder needs a 2.5 GB
torch stack and roughly 4 GB of weights, so without a fake, every pipeline test
would be a slow test and would in practice stop being run.
"""

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from visual_verify.retrieval.geometry import PatchGrid
from visual_verify.retrieval.provenance import EmbedProvenance

DIM = 128


@dataclass(frozen=True)
class PageEmbedding:
    """Multivectors for one page, with the geometry needed to place them."""

    vectors: np.ndarray  # (n_vectors, DIM), unit-normalized
    grid: PatchGrid

    def __post_init__(self) -> None:
        if self.vectors.shape[0] != self.grid.n_vectors:
            raise ValueError(
                f"vector count {self.vectors.shape[0]} does not match grid "
                f"n_vectors {self.grid.n_vectors}"
            )


class Embedder(Protocol):
    """What the pipeline needs. Deliberately narrow."""

    @property
    def provenance(self) -> EmbedProvenance: ...

    def embed_page(self, image_path: str, image_size: tuple[int, int]) -> PageEmbedding: ...

    def embed_query(self, text: str) -> np.ndarray:
        """(n_query_tokens, DIM), unit-normalized."""
        ...


def _rng(seed_text: str) -> np.random.Generator:
    digest = hashlib.sha256(seed_text.encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, "big"))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


class FakeEmbedder:
    """Deterministic stand-in. Same page text always gives the same vectors.

    embed_query(text) seeds from the SAME string as embed_page(text, ...), and
    both draw a numpy Generator's normal() stream in row-major order. That
    means embed_query's (8, DIM) draw is bit-for-bit the first 8 rows of the
    matching page's raw draw, before normalization: normalizing each row
    independently does not change that identity, so the query's own patch
    vectors are exact unit-vector matches (dot product 1) against the first 8
    rows of its own page, and only weakly correlated with any other page. That
    is what lets a query naming a page rank that page first under MaxSim,
    so pipeline and CLI tests can assert real retrieval behaviour rather than
    merely asserting that code ran.
    """

    def __init__(self, provenance: EmbedProvenance | None = None) -> None:
        self._provenance = provenance or EmbedProvenance(
            model_id="fake",
            model_revision="0",
            quantization="none",
            dtype="float32",
            render_dpi=150,
            embed_version=1,
        )

    @property
    def provenance(self) -> EmbedProvenance:
        return self._provenance

    def embed_page(self, image_path: str, image_size: tuple[int, int]) -> PageEmbedding:
        w, h = image_size
        # Aspect-ratio dependent, mirroring the real model, so no test can
        # accidentally rely on a square grid.
        n_x = max(2, round(8 * w / max(w, h)))
        n_y = max(2, round(8 * h / max(w, h)))
        grid = PatchGrid(n_x=n_x, n_y=n_y, offset=4, n_vectors=n_x * n_y + 11)
        v = _rng(image_path).normal(size=(grid.n_vectors, DIM)).astype(np.float32)
        return PageEmbedding(vectors=_unit(v), grid=grid)

    def embed_query(self, text: str) -> np.ndarray:
        v = _rng(text).normal(size=(8, DIM)).astype(np.float32)
        return _unit(v)
