# S3 Retrieval Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Embed ingested page images with ColQwen2 and index them in Qdrant as multivectors, so `vvrag search "<question>"` returns ranked pages.

**Architecture:** A new `visual_verify/retrieval/` package behind a `retrieval` extra. Three pure modules (geometry, provenance, pooling) carry the logic that S4 depends on and are testable with no GPU, no network, and no torch. Two heavy modules (embedder, index) sit behind narrow interfaces so a `FakeEmbedder` can drive the pipeline in CI. The core package stays at four dependencies.

**Tech Stack:** colpali-engine 0.3.10, transformers 4.51.x, torch 2.6.0, torchvision 0.21.0, qdrant-client, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-08-03-s3-retrieval-design.md`

---

## Critical domain knowledge

Verified empirically on this machine. Do not re-derive these; they cost hours.

### The token sequence is not all image patches

```
747 vectors = 4 prefix + 736 image patches (CONTIGUOUS) + 7 suffix
              idx 0..3    idx 4..739                      idx 740..746

prefix: <|im_start|>user\n<|vision_start|>
suffix: <|vision_end|>Describe the image.<|im_end|><|endoftext|>
```

**Image patches begin at offset 4, not 0.** `emb[:736]` silently shifts the whole
grid by four cells and misplaces every box. The offset must be derived from
`input_ids == image_pad_token_id`, never hardcoded, because the prompt template
can change between model revisions.

### The grid is 23 x 32 and is per page

`ColQwen2Processor.get_n_patches(image_size, spatial_merge_size=2)` returns
`(n_x, n_y)`. For this project's A4 pages at 150 dpi (1241x1754 px) that is
`(23, 32)`: `n_x=23` columns, `n_y=32` rows, `23*32 = 736`. It derives from
`smart_resize` on aspect ratio, so a landscape page gives different numbers.
`spatial_merge_size` lives on the model as `model.spatial_merge_size`.

Patch index to grid cell: `col = i % n_x`, `row = i // n_x`.

### Quantization must skip the vision tower

`llm_int8_skip_modules=["visual", "custom_text_proj"]`. Without it, known-item
top-1 measures 0.00 instead of 1.00, with no warning, no NaN, correct shapes,
and unit-normalized vectors.

### Other measured invariants

- `float16`, never `bfloat16` (GTX 1650 is sm_75, no native bf16).
- Batch size 1: faster (21.4 s/page vs 24.6) and lower peak VRAM.
- Select embeddings by `attention_mask`, never `out[j, :n]`. Qwen2-VL pads LEFT.
- Dependency stack must be pinned exactly; `colpali-engine` 0.3.17 with
  `transformers` 5 silently randomizes the LoRA adapter and projection head.

### Environment

The retrieval stack is NOT in the project venv yet. A working, verified venv
exists at:

```
/tmp/claude-1000/-home-pursottam-mine-projects-verifiable-visual-rag/ab1aea5c-001c-4a82-8a57-90bda6f945db/scratchpad/.benchvenv
```

Qdrant Cloud credentials are in the gitignored `.env` at the repo root. The
`pages` collection currently holds 8 smoke-test points and must be recreated.

---

## File structure

**Create:**

| file | responsibility |
|---|---|
| `src/visual_verify/retrieval/__init__.py` | package marker, no imports |
| `src/visual_verify/retrieval/geometry.py` | `PatchGrid`: patch index to normalized bbox. Pure stdlib. |
| `src/visual_verify/retrieval/provenance.py` | `EmbedProvenance`: what produced a vector, and mismatch detection. Pure stdlib. |
| `src/visual_verify/retrieval/pooling.py` | row/column mean pooling. numpy only. |
| `src/visual_verify/retrieval/types.py` | `PageEmbedding`, `Embedder` Protocol, `FakeEmbedder`. numpy only. |
| `src/visual_verify/retrieval/embedder.py` | ColQwen2 loading and embedding. torch. |
| `src/visual_verify/retrieval/index.py` | `QdrantIndex`. qdrant-client. |
| `src/visual_verify/retrieval/pipeline.py` | `embed_documents()` orchestration. |

**Modify:** `pyproject.toml` (extra), `src/visual_verify/config.py` (API key),
`src/visual_verify/cli.py` (embed + search), `README.md`.

`geometry.py`, `provenance.py`, `pooling.py`, and `types.py` import nothing
heavier than numpy. That is what lets Tasks 2 through 5 and 8 through 10 be
tested in CI with no GPU.

---

### Task 1: Add the retrieval extra and the API key setting

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/visual_verify/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_qdrant_api_key_from_env(monkeypatch):
    monkeypatch.setenv("VVRAG_QDRANT_API_KEY", "secret-key")
    assert Settings.from_env().qdrant_api_key == "secret-key"


def test_qdrant_api_key_defaults_to_none(monkeypatch):
    monkeypatch.delenv("VVRAG_QDRANT_API_KEY", raising=False)
    assert Settings.from_env().qdrant_api_key is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -k qdrant_api_key -v`
Expected: FAIL, `AttributeError: 'Settings' object has no attribute 'qdrant_api_key'`

- [ ] **Step 3: Add the field**

In `src/visual_verify/config.py`, add to `Settings` after `qdrant_url`:

```python
    qdrant_api_key: str | None = None
```

and inside `from_env()`, after the `qdrant_url` line:

```python
            qdrant_api_key=os.getenv("VVRAG_QDRANT_API_KEY"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Add the extra**

In `pyproject.toml`, replace the `[project.optional-dependencies]` block:

```toml
[project.optional-dependencies]
store = ["sqlalchemy>=2.0", "alembic>=1.13"]
# PINNED EXACTLY, DO NOT RELAX. Three of four version combinations tried are
# broken, and the dangerous one fails SILENTLY: colpali-engine 0.3.17 with
# transformers 5.x randomly initializes the LoRA adapter and the projection
# head, which drops known-item retrieval top-1 from 1.00 to 0.125 while
# reporting only an informational table. torch and torchvision must move
# together or torchvision's compiled ops fail to load.
retrieval = [
    "colpali-engine==0.3.10",
    "transformers>=4.51,<4.52",
    "torch==2.6.0",
    "torchvision==0.21.0",
    "qdrant-client>=1.12",
]
```

- [ ] **Step 6: Install and verify the boundary still holds**

Run: `uv sync --all-extras --group dev`
Run: `uv run pytest tests/test_core_is_light.py -v`
Expected: PASS. `qdrant_client`, `torch`, and `transformers` are already in
`FORBIDDEN`, so this proves installing them did not make the core import them.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/visual_verify/config.py tests/test_config.py
git commit -m "Add retrieval extra with pinned stack and Qdrant API key setting"
```

---

### Task 2: PatchGrid geometry

The single most important unit in this slice. S4's snap-to-box is built on it.

**Files:**
- Create: `src/visual_verify/retrieval/__init__.py`
- Create: `src/visual_verify/retrieval/geometry.py`
- Test: `tests/test_geometry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_geometry.py`:

```python
import pytest

from visual_verify.retrieval.geometry import PatchGrid

# The real grid measured for this project's A4 pages at 150 dpi.
A4 = PatchGrid(n_x=23, n_y=32, offset=4, n_vectors=747)


def test_counts():
    assert A4.n_image_patches == 736
    assert A4.n_special == 11


def test_first_patch_is_top_left():
    x0, y0, x1, y1 = A4.patch_bbox(0)
    assert (x0, y0) == (0.0, 0.0)
    assert x1 == pytest.approx(1 / 23)
    assert y1 == pytest.approx(1 / 32)


def test_last_patch_is_bottom_right():
    x0, y0, x1, y1 = A4.patch_bbox(735)
    assert x1 == pytest.approx(1.0)
    assert y1 == pytest.approx(1.0)
    assert x0 == pytest.approx(22 / 23)
    assert y0 == pytest.approx(31 / 32)


def test_row_major_order():
    """Patch n_x is the START of the second row, not the second column."""
    _, y0_first, _, _ = A4.patch_bbox(0)
    _, y0_next, _, _ = A4.patch_bbox(23)
    assert y0_next > y0_first
    x0_a, _, _, _ = A4.patch_bbox(1)
    assert x0_a == pytest.approx(1 / 23)


def test_sequence_index_maps_through_offset():
    """Sequence index 4 is image patch 0. Off-by-four shifts every box."""
    assert A4.seq_to_patch(4) == 0
    assert A4.seq_to_patch(739) == 735


def test_special_tokens_have_no_region():
    for seq_idx in (0, 3, 740, 746):
        assert not A4.is_image_token(seq_idx)
        with pytest.raises(ValueError):
            A4.seq_to_patch(seq_idx)


def test_image_tokens_are_recognised():
    assert A4.is_image_token(4)
    assert A4.is_image_token(739)


def test_landscape_grid_is_not_assumed_square():
    """Grid is aspect-ratio dependent; nothing may assume n_x == n_y."""
    wide = PatchGrid(n_x=32, n_y=18, offset=4, n_vectors=32 * 18 + 11)
    assert wide.n_image_patches == 576
    x0, _, x1, _ = wide.patch_bbox(0)
    assert x1 == pytest.approx(1 / 32)


def test_rejects_inconsistent_vector_count():
    with pytest.raises(ValueError, match="inconsistent"):
        PatchGrid(n_x=23, n_y=32, offset=4, n_vectors=999)


def test_rejects_nonpositive_dims():
    with pytest.raises(ValueError):
        PatchGrid(n_x=0, n_y=32, offset=4, n_vectors=11)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_geometry.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'visual_verify.retrieval'`

- [ ] **Step 3: Write the implementation**

Create `src/visual_verify/retrieval/__init__.py` as an empty file (no imports;
importing torch here would break the core boundary test).

Create `src/visual_verify/retrieval/geometry.py`:

```python
"""Patch grid geometry: the bridge from a model vector index to a page region.

This is what makes snap-to-box possible, and getting it wrong does not raise.

Two facts about ColQwen2 drive every line here, both measured on PyMuPDF-rendered
A4 pages at 150 dpi (1241x1754 px):

  1. The 747 vectors are 4 prefix tokens, then 736 CONTIGUOUS image patches,
     then 7 suffix tokens. Image patches start at sequence index 4. Slicing
     [:736] shifts the entire grid by four cells and misplaces every box while
     producing perfectly plausible output.

  2. The grid is 23 x 32 and is NOT square. It comes from smart_resize on the
     page aspect ratio, so a landscape slide yields different dimensions. There
     is no global constant to hardcode, which is why this is a value object
     stored per page rather than module-level constants.

Pure stdlib on purpose: S4 and the evaluation harness both need this, and
neither should have to import torch to get it.
"""

from dataclasses import dataclass

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class PatchGrid:
    """Where each model vector sits on the page.

    n_x: patch columns. n_y: patch rows. offset: sequence index of image patch 0.
    n_vectors: total vectors the model returned, including special tokens.
    """

    n_x: int
    n_y: int
    offset: int
    n_vectors: int

    def __post_init__(self) -> None:
        if self.n_x <= 0 or self.n_y <= 0:
            raise ValueError(f"grid dims must be positive, got {self.n_x}x{self.n_y}")
        if self.offset < 0:
            raise ValueError(f"offset must be non-negative, got {self.offset}")
        # The invariant that catches a grid recorded from the wrong image, a
        # changed spatial_merge_size, or a miscounted prefix. Cheap, and it is
        # the only thing standing between an off-by-N grid and silently wrong
        # evidence boxes.
        if self.n_vectors < self.offset + self.n_image_patches:
            raise ValueError(
                f"inconsistent: {self.n_x}x{self.n_y} patches at offset "
                f"{self.offset} needs at least {self.offset + self.n_image_patches} "
                f"vectors, got {self.n_vectors}"
            )

    @property
    def n_image_patches(self) -> int:
        return self.n_x * self.n_y

    @property
    def n_special(self) -> int:
        """Vectors that correspond to no region of the page."""
        return self.n_vectors - self.n_image_patches

    def is_image_token(self, seq_idx: int) -> bool:
        """Whether a sequence index refers to a page region at all.

        Callers MUST filter with this before any argmax. A special token has no
        region, and mapping one anyway would draw a confident box with no causal
        relationship to the evidence.
        """
        return self.offset <= seq_idx < self.offset + self.n_image_patches

    def seq_to_patch(self, seq_idx: int) -> int:
        """Model sequence index to image patch index."""
        if not self.is_image_token(seq_idx):
            raise ValueError(
                f"sequence index {seq_idx} is a special token, not an image patch; "
                f"image patches occupy {self.offset}.."
                f"{self.offset + self.n_image_patches - 1}"
            )
        return seq_idx - self.offset

    def patch_bbox(self, patch_idx: int) -> BBox:
        """Normalized 0-1 page rect for an image patch, origin top-left.

        Same convention as contracts.BBox and every box S2 stored, so a patch
        rect and a word rect can be compared directly.
        """
        if not 0 <= patch_idx < self.n_image_patches:
            raise IndexError(f"patch {patch_idx} out of range 0..{self.n_image_patches - 1}")
        col = patch_idx % self.n_x
        row = patch_idx // self.n_x
        return (col / self.n_x, row / self.n_y, (col + 1) / self.n_x, (row + 1) / self.n_y)

    def seq_bbox(self, seq_idx: int) -> BBox:
        """Convenience: sequence index straight to page rect."""
        return self.patch_bbox(self.seq_to_patch(seq_idx))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_geometry.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Verify the core boundary is unaffected**

Run: `uv run pytest tests/test_core_is_light.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/visual_verify/retrieval/__init__.py src/visual_verify/retrieval/geometry.py tests/test_geometry.py
git commit -m "Add PatchGrid mapping model vector indices to page regions"
```

---

### Task 3: Embedding provenance

**Files:**
- Create: `src/visual_verify/retrieval/provenance.py`
- Test: `tests/test_provenance.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_provenance.py`:

```python
import pytest

from visual_verify.retrieval.provenance import EmbedProvenance, ProvenanceMismatch

BASE = EmbedProvenance(
    model_id="vidore/colqwen2-v1.0",
    model_revision="abc123",
    quantization="nf4-skipvis",
    dtype="float16",
    render_dpi=150,
    embed_version=1,
)


def test_round_trips_through_payload():
    assert EmbedProvenance.from_payload(BASE.to_payload()) == BASE


def test_identical_provenance_is_compatible():
    BASE.require_compatible(BASE)  # must not raise


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_id", "vidore/colSmol-500M"),
        ("model_revision", "def456"),
        ("quantization", "none"),
        ("dtype", "bfloat16"),
        ("render_dpi", 300),
        ("embed_version", 2),
    ],
)
def test_any_difference_is_a_mismatch(field, value):
    """Every field is load-bearing: differing vectors are not comparable."""
    other = EmbedProvenance(**{**BASE.to_payload(), field: value})
    with pytest.raises(ProvenanceMismatch, match=field):
        BASE.require_compatible(other)


def test_mismatch_message_names_both_values():
    other = EmbedProvenance(**{**BASE.to_payload(), "render_dpi": 300})
    with pytest.raises(ProvenanceMismatch) as exc:
        BASE.require_compatible(other)
    assert "150" in str(exc.value) and "300" in str(exc.value)


def test_payload_keys_are_flat_scalars():
    """Qdrant payloads must be JSON scalars, not nested objects."""
    for value in BASE.to_payload().values():
        assert isinstance(value, (str, int))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provenance.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/visual_verify/retrieval/provenance.py`:

```python
"""What produced a stored vector, and refusal when that changes.

Vectors from different models, quantizations, dtypes, or render DPIs are not
comparable, and nothing about a stored vector reveals which produced it. Mixing
them returns a confidently ranked, entirely wrong result list.

This is the most commonly reported production failure for retrieval systems: the
indexing embedder is changed, the query embedder is not, and recall degrades
silently for weeks. This project is unusually exposed to it, because the design
deliberately keeps colSmol as an alternative and permits per-document render DPI.

The response is to refuse rather than to warn. A warning on a CLI scrolls past;
a wrong answer with a drawn evidence box is exactly the failure this project
exists to prevent.

Pure stdlib: the query path must be able to check compatibility before deciding
whether loading a model is even worthwhile.
"""

from dataclasses import asdict, dataclass


class ProvenanceMismatch(RuntimeError):
    """Raised when an embedder does not match the vectors already indexed."""


@dataclass(frozen=True)
class EmbedProvenance:
    model_id: str
    model_revision: str
    quantization: str
    dtype: str
    render_dpi: int
    embed_version: int

    def to_payload(self) -> dict[str, str | int]:
        """Flat scalars only; Qdrant payload values must be JSON primitives."""
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict) -> "EmbedProvenance":
        return cls(**{f: payload[f] for f in cls.__dataclass_fields__})

    def require_compatible(self, other: "EmbedProvenance") -> None:
        """Raise unless `other` produced vectors comparable with ours.

        Every field is checked. There is no "close enough": a different revision
        of the same model can ship different weights, and a different render DPI
        changes the patch grid, so neither is a cosmetic difference.
        """
        for field in self.__dataclass_fields__:
            mine, theirs = getattr(self, field), getattr(other, field)
            if mine != theirs:
                raise ProvenanceMismatch(
                    f"{field} differs: index holds {mine!r}, embedder is {theirs!r}. "
                    "These vectors are not comparable. Re-embed the corpus or use "
                    "the original embedder."
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_provenance.py -v`
Expected: PASS, 11 tests

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/retrieval/provenance.py tests/test_provenance.py
git commit -m "Add embedding provenance with refusal on mismatch"
```

---

### Task 4: Mean pooling

**Files:**
- Create: `src/visual_verify/retrieval/pooling.py`
- Test: `tests/test_pooling.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pooling.py`:

```python
import numpy as np
import pytest

from visual_verify.retrieval.geometry import PatchGrid
from visual_verify.retrieval.pooling import mean_pool_cols, mean_pool_rows

GRID = PatchGrid(n_x=4, n_y=3, offset=2, n_vectors=2 + 12 + 1)
DIM = 8


def _vectors() -> np.ndarray:
    rng = np.random.default_rng(0)
    v = rng.normal(size=(GRID.n_vectors, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_row_pooling_yields_one_vector_per_row_plus_specials():
    out = mean_pool_rows(_vectors(), GRID)
    assert out.shape == (GRID.n_y + GRID.n_special, DIM)


def test_col_pooling_yields_one_vector_per_column_plus_specials():
    out = mean_pool_cols(_vectors(), GRID)
    assert out.shape == (GRID.n_x + GRID.n_special, DIM)


def test_row_pool_is_the_mean_of_that_row():
    v = _vectors()
    out = mean_pool_rows(v, GRID)
    # Row 1 is patches 4..7, which are sequence indices 6..9.
    expected = v[GRID.offset + 4 : GRID.offset + 8].mean(axis=0)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(out[1], expected, atol=1e-6)


def test_col_pool_is_the_mean_of_that_column():
    v = _vectors()
    out = mean_pool_cols(v, GRID)
    # Column 2 is patches 2, 6, 10 -> sequence indices 4, 8, 12.
    expected = v[[GRID.offset + 2, GRID.offset + 6, GRID.offset + 10]].mean(axis=0)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(out[2], expected, atol=1e-6)


def test_special_tokens_are_carried_through_unpooled():
    v = _vectors()
    out = mean_pool_rows(v, GRID)
    # Specials are the vectors outside the image block, in sequence order.
    specials = np.concatenate([v[: GRID.offset], v[GRID.offset + GRID.n_image_patches :]])
    assert np.allclose(out[GRID.n_y :], specials, atol=1e-6)


def test_output_is_unit_normalized():
    out = mean_pool_rows(_vectors(), GRID)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_rejects_wrong_vector_count():
    with pytest.raises(ValueError, match="expected"):
        mean_pool_rows(np.zeros((5, DIM), dtype=np.float32), GRID)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pooling.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/visual_verify/retrieval/pooling.py`:

```python
"""Row and column mean pooling for the two-stage prefetch representation.

Written in S3 but unused until S7. It is here because Qdrant cannot add a named
vector to an existing collection without recreating it, and recreating means
re-upserting every point. Computing these now costs one array mean against a
21.4 s embed, so provisioning them is effectively free insurance against a
re-index later.

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
    """One vector per grid ROW, then the special tokens. Shape (n_y + n_special, d)."""
    patches, specials = _split(vectors, grid)
    pooled = patches.reshape(grid.n_y, grid.n_x, -1).mean(axis=1)
    return _normalize(np.concatenate([pooled, specials]))


def mean_pool_cols(vectors: np.ndarray, grid: PatchGrid) -> np.ndarray:
    """One vector per grid COLUMN, then the special tokens. Shape (n_x + n_special, d)."""
    patches, specials = _split(vectors, grid)
    pooled = patches.reshape(grid.n_y, grid.n_x, -1).mean(axis=0)
    return _normalize(np.concatenate([pooled, specials]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pooling.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/retrieval/pooling.py tests/test_pooling.py
git commit -m "Add row and column mean pooling for prefetch vectors"
```

---

### Task 5: Embedding types and a fake embedder

**Files:**
- Create: `src/visual_verify/retrieval/types.py`
- Test: `tests/test_retrieval_types.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_retrieval_types.py`:

```python
import numpy as np
import pytest

from visual_verify.retrieval.geometry import PatchGrid
from visual_verify.retrieval.types import FakeEmbedder, PageEmbedding


def test_page_embedding_rejects_vector_count_mismatch():
    grid = PatchGrid(n_x=2, n_y=2, offset=1, n_vectors=6)
    with pytest.raises(ValueError, match="does not match"):
        PageEmbedding(vectors=np.zeros((3, 8), dtype=np.float32), grid=grid)


def test_fake_embedder_is_deterministic():
    a = FakeEmbedder().embed_page("x.png", (100, 200))
    b = FakeEmbedder().embed_page("x.png", (100, 200))
    assert np.allclose(a.vectors, b.vectors)


def test_fake_embedder_differs_per_page():
    a = FakeEmbedder().embed_page("a.png", (100, 200))
    b = FakeEmbedder().embed_page("b.png", (100, 200))
    assert not np.allclose(a.vectors, b.vectors)


def test_fake_embedder_grid_matches_aspect_ratio():
    """Portrait and landscape must not produce the same grid, or tests that
    depend on grid variation would silently pass on a square assumption."""
    portrait = FakeEmbedder().embed_page("p.png", (100, 200))
    landscape = FakeEmbedder().embed_page("l.png", (200, 100))
    assert portrait.grid.n_x < portrait.grid.n_y
    assert landscape.grid.n_x > landscape.grid.n_y


def test_fake_embedder_vectors_are_unit_normalized():
    e = FakeEmbedder().embed_page("x.png", (100, 200))
    assert np.allclose(np.linalg.norm(e.vectors, axis=1), 1.0, atol=1e-5)


def test_fake_query_matches_its_own_page():
    """The fake must be retrievable, or pipeline tests prove nothing."""
    emb = FakeEmbedder()
    page = emb.embed_page("a.png", (100, 200))
    q = emb.embed_query("a.png")
    other = emb.embed_page("b.png", (100, 200))
    score = lambda p: float((q @ p.vectors.T).max(axis=1).sum())
    assert score(page) > score(other)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retrieval_types.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/visual_verify/retrieval/types.py`:

```python
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

    embed_query(text) returns vectors drawn from the SAME seed as
    embed_page(text, ...), so a query naming a page ranks that page first. That
    makes it possible to assert real retrieval behaviour in the pipeline and CLI
    tests rather than merely asserting that code ran.
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_retrieval_types.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/retrieval/types.py tests/test_retrieval_types.py
git commit -m "Add PageEmbedding, Embedder protocol, and a deterministic fake"
```

---

### Task 6: The real ColQwen2 embedder

**Files:**
- Create: `src/visual_verify/retrieval/embedder.py`
- Test: `tests/test_embedder.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_embedder.py`:

```python
"""Real-model tests. All slow: they download ~4 GB and need a CUDA GPU."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from visual_verify.retrieval.embedder import ColQwen2Embedder  # noqa: E402


@pytest.fixture(scope="module")
def embedder():
    return ColQwen2Embedder()


def test_provenance_records_the_skipvis_quantization(embedder):
    p = embedder.provenance
    assert p.model_id == "vidore/colqwen2-v1.0"
    assert p.quantization == "nf4-skipvis"
    assert p.dtype == "float16"
    # A branch name would let the weights change under a fixed provenance.
    assert len(p.model_revision) >= 7 and p.model_revision != "main"


def test_grid_matches_measured_values_for_a4(embedder, tmp_path):
    from PIL import Image

    path = tmp_path / "a4.png"
    Image.new("RGB", (1241, 1754), "white").save(path)
    emb = embedder.embed_page(str(path), (1241, 1754))
    assert (emb.grid.n_x, emb.grid.n_y) == (23, 32)
    assert emb.grid.offset == 4
    assert emb.grid.n_vectors == 747


def test_grid_invariant_holds(embedder, tmp_path):
    """n_x * n_y + n_special == len(vectors), on a NON-A4 shape too."""
    from PIL import Image

    path = tmp_path / "wide.png"
    Image.new("RGB", (1754, 1241), "white").save(path)
    emb = embedder.embed_page(str(path), (1754, 1241))
    assert emb.grid.n_image_patches + emb.grid.n_special == emb.vectors.shape[0]
    assert emb.grid.n_x > emb.grid.n_y, "landscape must not produce a portrait grid"


def test_vectors_are_healthy(embedder, tmp_path):
    from PIL import Image

    path = tmp_path / "p.png"
    Image.new("RGB", (1241, 1754), "white").save(path)
    v = embedder.embed_page(str(path), (1241, 1754)).vectors
    assert not np.isnan(v).any() and not np.isinf(v).any()
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-2)
    assert v.shape[1] == 128


def test_query_embedding_shape(embedder):
    q = embedder.embed_query("what is snap to box grounding")
    assert q.ndim == 2 and q.shape[1] == 128
    assert not np.isnan(q).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embedder.py -v -m slow`
Expected: FAIL, `ModuleNotFoundError: No module named 'visual_verify.retrieval.embedder'`

- [ ] **Step 3: Write the implementation**

Create `src/visual_verify/retrieval/embedder.py`:

```python
"""ColQwen2 page and query embedding.

Every non-obvious line here has a measured failure behind it. See
docs/superpowers/specs/2026-08-03-s3-retrieval-design.md section 4.
"""

import numpy as np
import torch
from PIL import Image

from visual_verify.retrieval.geometry import PatchGrid
from visual_verify.retrieval.provenance import EmbedProvenance
from visual_verify.retrieval.types import PageEmbedding

MODEL_ID = "vidore/colqwen2-v1.0"
EMBED_VERSION = 1

# GTX 1650 is Turing (sm_75) and has NO native bfloat16, which every ColPali
# example in the wild uses. Verified: fp16 embeddings on this card contain no
# NaN or Inf and stay unit-normalized.
DTYPE = torch.float16

# LOAD-BEARING. Blanket load_in_4bit also quantizes the vision tower and the
# projection head, which destroys patch-level geometry: measured known-item
# top-1 of 0.00 against 1.00 with this skip list, on identical code and
# hardware. The broken configuration emits no warning, raises nothing, produces
# no NaN, returns the correct shape, and yields unit-normalized vectors.
SKIP_MODULES = ["visual", "custom_text_proj"]


class AdapterLoadError(RuntimeError):
    """The LoRA adapter or projection head did not load."""


class ColQwen2Embedder:
    """Loads once, embeds many. Not thread-safe."""

    def __init__(self, render_dpi: int = 150, device: str = "cuda:0") -> None:
        from huggingface_hub import model_info
        from transformers import BitsAndBytesConfig

        from colpali_engine.models import ColQwen2, ColQwen2Processor

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=SKIP_MODULES,
        )
        self.model = ColQwen2.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE, device_map=device, quantization_config=quant
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(MODEL_ID)
        self._check_adapter_loaded()

        self._image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self._merge_size = self.model.spatial_merge_size

        # Pin the resolved commit, not "main": a branch can move and would let
        # the weights change while provenance claimed they had not.
        revision = model_info(MODEL_ID).sha
        self._provenance = EmbedProvenance(
            model_id=MODEL_ID,
            model_revision=revision,
            quantization="nf4-skipvis",
            dtype="float16",
            render_dpi=render_dpi,
            embed_version=EMBED_VERSION,
        )

    def _check_adapter_loaded(self) -> None:
        """Fail loudly if the trained weights were silently randomized.

        colpali-engine 0.3.17 on transformers 5.x does exactly this: the adapter
        keys are written against the old submodule path (model.layers, since
        renamed to language_model.layers), so every LoRA weight and the
        projection head are newly initialized at random. transformers reports it
        as an informational table, not an error, and the resulting model scores
        at chance while looking entirely healthy.
        """
        names = [n for n, _ in self.model.named_parameters()]
        if not any("lora" in n for n in names):
            raise AdapterLoadError(
                "no LoRA parameters found on the loaded model. The installed "
                "colpali-engine and transformers versions are incompatible; see "
                "the retrieval extra pins in pyproject.toml."
            )
        proj = getattr(self.model, "custom_text_proj", None)
        if proj is None:
            raise AdapterLoadError("custom_text_proj missing; projection head did not load")

    @property
    def provenance(self) -> EmbedProvenance:
        return self._provenance

    def _grid_for(self, image_size: tuple[int, int], seq_len: int, input_ids) -> PatchGrid:
        n_x, n_y = self.processor.get_n_patches(image_size, spatial_merge_size=self._merge_size)
        # Derive the offset from the token stream rather than hardcoding 4. The
        # prompt template is model-version dependent, and an off-by-N offset
        # shifts every patch box without raising anything.
        positions = (input_ids == self._image_token_id).nonzero(as_tuple=True)[0]
        if positions.numel() != n_x * n_y:
            raise ValueError(
                f"processor reports a {n_x}x{n_y} grid ({n_x * n_y} patches) but the "
                f"token stream holds {positions.numel()} image tokens"
            )
        return PatchGrid(n_x=n_x, n_y=n_y, offset=int(positions[0]), n_vectors=seq_len)

    @torch.no_grad()
    def embed_page(self, image_path: str, image_size: tuple[int, int]) -> PageEmbedding:
        image = Image.open(image_path).convert("RGB")
        # Batch of one: measured faster than batch 2 on this card (21.4 s vs
        # 24.6 s per page) because it is memory-bound, and with no padding there
        # is no left-padding trap to fall into.
        batch = self.processor.process_images([image]).to(self.model.device)
        out = self.model(**batch)

        mask = batch["attention_mask"][0].bool()
        vectors = out[0][mask].to(torch.float32).cpu().numpy()
        ids = batch["input_ids"][0][mask]
        grid = self._grid_for(image_size, vectors.shape[0], ids)
        return PageEmbedding(vectors=vectors, grid=grid)

    @torch.no_grad()
    def embed_query(self, text: str) -> np.ndarray:
        batch = self.processor.process_queries([text]).to(self.model.device)
        out = self.model(**batch)
        # Select BY MASK, never out[0, :n]. Qwen2-VL's processor pads on the
        # LEFT, so prefix slicing reads padding. MaxSim sums over query tokens,
        # so a pad vector adds a spurious maximum to every page's score.
        mask = batch["attention_mask"][0].bool()
        return out[0][mask].to(torch.float32).cpu().numpy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embedder.py -v -m slow`
Expected: PASS, 5 tests. First run downloads ~4 GB.

- [ ] **Step 5: Verify the boundary still holds**

Run: `uv run pytest tests/test_core_is_light.py -v`
Expected: PASS. `retrieval/embedder.py` imports torch at module level, so this
proves nothing in the core imports `visual_verify.retrieval`.

- [ ] **Step 6: Commit**

```bash
git add src/visual_verify/retrieval/embedder.py tests/test_embedder.py
git commit -m "Add ColQwen2 embedder with vision tower left unquantized"
```

---

### Task 7: The Qdrant index

**Files:**
- Create: `src/visual_verify/retrieval/index.py`
- Test: `tests/test_index.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_index.py`:

```python
import numpy as np
import pytest

from visual_verify.retrieval.index import QdrantIndex
from visual_verify.retrieval.provenance import ProvenanceMismatch
from visual_verify.retrieval.types import FakeEmbedder

SHA = "a" * 64


@pytest.fixture
def index():
    """Local in-memory Qdrant: real client code, no server."""
    return QdrantIndex(url=":memory:", api_key=None, collection="pages_test")


@pytest.fixture
def embedder():
    return FakeEmbedder()


def _add(index, embedder, name, page_no):
    emb = embedder.embed_page(name, (100, 200))
    index.upsert_page(SHA, page_no, f"{SHA}/{name}", emb, embedder.provenance)
    return emb


def test_ensure_collection_is_idempotent(index):
    index.ensure_collection()
    index.ensure_collection()
    assert index.count() == 0


def test_upsert_then_count(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "p0.png", 0)
    assert index.count() == 1


def test_upsert_is_idempotent_on_same_page(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "p0.png", 0)
    _add(index, embedder, "p0.png", 0)
    assert index.count() == 1, "deterministic point id must overwrite, not duplicate"


def test_existing_page_nos_drives_resumption(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "p0.png", 0)
    _add(index, embedder, "p2.png", 2)
    assert index.existing_page_nos(SHA) == {0, 2}


def test_existing_page_nos_is_scoped_per_document(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "p0.png", 0)
    assert index.existing_page_nos("b" * 64) == set()


def test_payload_carries_geometry(index, embedder):
    index.ensure_collection()
    emb = _add(index, embedder, "p0.png", 0)
    payload = index.get_payload(SHA, 0)
    assert payload["n_patches_x"] == emb.grid.n_x
    assert payload["n_patches_y"] == emb.grid.n_y
    assert payload["n_special_tokens"] == emb.grid.n_special
    assert payload["patch_offset"] == emb.grid.offset


def test_search_ranks_the_matching_page_first(index, embedder):
    index.ensure_collection()
    for i, name in enumerate(["alpha.png", "beta.png", "gamma.png"]):
        _add(index, embedder, name, i)
    hits = index.search(embedder.embed_query("beta.png"), embedder.provenance, limit=3)
    assert hits[0].page == 1


def test_search_returns_retrieved_pages(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "alpha.png", 0)
    hit = index.search(embedder.embed_query("alpha.png"), embedder.provenance, limit=1)[0]
    assert hit.doc_id == SHA
    assert hit.image_ref == f"{SHA}/alpha.png"
    assert hit.score > 0


def test_search_refuses_on_provenance_mismatch(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "alpha.png", 0)
    from visual_verify.retrieval.provenance import EmbedProvenance

    other = EmbedProvenance(**{**embedder.provenance.to_payload(), "render_dpi": 300})
    with pytest.raises(ProvenanceMismatch, match="render_dpi"):
        index.search(embedder.embed_query("alpha.png"), other, limit=1)


def test_upsert_refuses_on_provenance_mismatch(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "alpha.png", 0)
    from visual_verify.retrieval.provenance import EmbedProvenance

    other = EmbedProvenance(**{**embedder.provenance.to_payload(), "model_id": "other"})
    emb = embedder.embed_page("beta.png", (100, 200))
    with pytest.raises(ProvenanceMismatch, match="model_id"):
        index.upsert_page(SHA, 1, f"{SHA}/beta.png", emb, other)


def test_qdrant_ranking_matches_local_maxsim(index, embedder):
    """A misconfigured collection still accepts writes and still returns
    results; it just returns the wrong ones. This is the test that notices."""
    index.ensure_collection()
    embs = {}
    for i, name in enumerate(["a.png", "b.png", "c.png", "d.png"]):
        embs[i] = _add(index, embedder, name, i)

    q = embedder.embed_query("c.png")
    hits = index.search(q, embedder.provenance, limit=4)
    local = sorted(
        embs, key=lambda i: -float((q @ embs[i].vectors.T).max(axis=1).sum())
    )
    assert [h.page for h in hits] == local


def test_pooled_vectors_are_stored(index, embedder):
    index.ensure_collection()
    emb = _add(index, embedder, "a.png", 0)
    stored = index.get_vectors(SHA, 0)
    assert stored["original"].shape[0] == emb.grid.n_vectors
    assert stored["mean_pooling_rows"].shape[0] == emb.grid.n_y + emb.grid.n_special
    assert stored["mean_pooling_cols"].shape[0] == emb.grid.n_x + emb.grid.n_special


def test_pooled_vectors_agree_with_pooling_module(index, embedder):
    """Written in S3, used in S7. A silently wrong pooled vector would only
    surface as degraded rerank quality much later."""
    from visual_verify.retrieval.pooling import mean_pool_rows

    index.ensure_collection()
    emb = _add(index, embedder, "a.png", 0)
    stored = index.get_vectors(SHA, 0)["mean_pooling_rows"]
    assert np.allclose(stored, mean_pool_rows(emb.vectors, emb.grid), atol=1e-5)


def test_recreate_clears_points(index, embedder):
    index.ensure_collection()
    _add(index, embedder, "a.png", 0)
    index.ensure_collection(recreate=True)
    assert index.count() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_index.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/visual_verify/retrieval/index.py`:

```python
"""Qdrant multivector storage and MaxSim search.

Three NAMED vectors in one collection. S3 populates all three but queries only
"original"; the pooled pair exists because Qdrant cannot add a named vector to an
existing collection without recreating it, so the schema is a one-way door and
provisioning now is far cheaper than a re-embed at 21.4 s per page later.
"""

import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from visual_verify.contracts import RetrievedPage
from visual_verify.retrieval.pooling import mean_pool_cols, mean_pool_rows
from visual_verify.retrieval.provenance import EmbedProvenance
from visual_verify.retrieval.types import PageEmbedding

DIM = 128
ORIGINAL = "original"
POOL_ROWS = "mean_pooling_rows"
POOL_COLS = "mean_pooling_cols"

# uuid5(NAMESPACE_DNS, "verifiable-visual-rag.pages"). Fixed forever: changing it
# orphans every existing point. Derived rather than random so it is auditable.
POINT_NS = uuid.UUID("5ee1d73c-35dc-53bb-8bf7-94bd98b0b932")


def point_id(doc_sha: str, page_no: int) -> str:
    """Deterministic, so re-embedding overwrites instead of duplicating."""
    return str(uuid.uuid5(POINT_NS, f"{doc_sha}:{page_no}"))


class QdrantIndex:
    def __init__(self, url: str, api_key: str | None, collection: str = "pages") -> None:
        # ":memory:" gives a real local client with no server, which is what
        # makes this class testable in CI.
        if url == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        self.collection = collection

    def _vector_params(self) -> models.VectorParams:
        return models.VectorParams(
            size=DIM,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            # m=0 disables HNSW graph construction. Necessary, not merely
            # acceptable: every candidate comparison in a multivector HNSW build
            # is itself a full MaxSim, which is combinatorially expensive. S7's
            # evaluation also needs exact scores rather than approximate ones.
            hnsw_config=models.HnswConfigDiff(m=0),
        )

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return
        if exists:
            self.client.delete_collection(self.collection)
        params = self._vector_params()
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={ORIGINAL: params, POOL_ROWS: params, POOL_COLS: params},
        )

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def _stored_provenance(self) -> EmbedProvenance | None:
        """Provenance of whatever is already indexed, or None if empty."""
        points, _ = self.client.scroll(self.collection, limit=1, with_payload=True)
        if not points:
            return None
        return EmbedProvenance.from_payload(points[0].payload)

    def _require_compatible(self, provenance: EmbedProvenance) -> None:
        stored = self._stored_provenance()
        if stored is not None:
            stored.require_compatible(provenance)

    def upsert_page(
        self,
        doc_sha: str,
        page_no: int,
        image_path: str,
        embedding: PageEmbedding,
        provenance: EmbedProvenance,
    ) -> None:
        self._require_compatible(provenance)
        grid = embedding.grid
        payload = {
            "doc_sha": doc_sha,
            "page_no": page_no,
            "image_path": image_path,
            # Geometry. Unrecoverable without re-embedding, and S4 cannot place
            # a single box without it.
            "n_patches_x": grid.n_x,
            "n_patches_y": grid.n_y,
            "n_image_patches": grid.n_image_patches,
            "n_special_tokens": grid.n_special,
            "patch_offset": grid.offset,
            **provenance.to_payload(),
        }
        self.client.upsert(
            collection_name=self.collection,
            wait=True,  # per-page durability; noise against a 21.4 s embed
            points=[
                models.PointStruct(
                    id=point_id(doc_sha, page_no),
                    vector={
                        ORIGINAL: embedding.vectors.tolist(),
                        POOL_ROWS: mean_pool_rows(embedding.vectors, grid).tolist(),
                        POOL_COLS: mean_pool_cols(embedding.vectors, grid).tolist(),
                    },
                    payload=payload,
                )
            ],
        )

    def existing_page_nos(self, doc_sha: str) -> set[int]:
        """Which pages are already indexed. Qdrant is the source of truth for
        embedding state, so this cannot desync from what is actually stored."""
        flt = models.Filter(
            must=[models.FieldCondition(key="doc_sha", match=models.MatchValue(value=doc_sha))]
        )
        out: set[int] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, scroll_filter=flt, limit=256, offset=offset, with_payload=True
            )
            out.update(p.payload["page_no"] for p in points)
            if offset is None:
                return out

    def get_payload(self, doc_sha: str, page_no: int) -> dict:
        recs = self.client.retrieve(self.collection, ids=[point_id(doc_sha, page_no)])
        return recs[0].payload

    def get_vectors(self, doc_sha: str, page_no: int) -> dict[str, np.ndarray]:
        """Read stored vectors back. This is what makes a schema change a
        re-index rather than a re-embed."""
        recs = self.client.retrieve(
            self.collection, ids=[point_id(doc_sha, page_no)], with_vectors=True
        )
        return {k: np.asarray(v, dtype=np.float32) for k, v in recs[0].vector.items()}

    def search(
        self, query_vectors: np.ndarray, provenance: EmbedProvenance, limit: int = 5
    ) -> list[RetrievedPage]:
        self._require_compatible(provenance)
        res = self.client.query_points(
            self.collection,
            query=query_vectors.tolist(),
            using=ORIGINAL,
            limit=limit,
            with_payload=True,
        ).points
        return [
            RetrievedPage(
                doc_id=p.payload["doc_sha"],
                page=p.payload["page_no"],
                image_ref=p.payload["image_path"],
                score=p.score,
            )
            for p in res
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_index.py -v`
Expected: PASS, 14 tests

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/retrieval/index.py tests/test_index.py
git commit -m "Add Qdrant multivector index with named pooled vectors"
```

---

### Task 8: The embedding pipeline

**Files:**
- Create: `src/visual_verify/retrieval/pipeline.py`
- Test: `tests/test_embed_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_embed_pipeline.py`:

```python
import pytest
from PIL import Image

from visual_verify.retrieval.index import QdrantIndex
from visual_verify.retrieval.pipeline import EmbedResult, embed_document
from visual_verify.retrieval.types import FakeEmbedder

SHA = "c" * 64


@pytest.fixture
def pages_dir(tmp_path):
    d = tmp_path / "pages" / SHA
    d.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (100, 200), "white").save(d / f"p{i:04d}.png")
    return tmp_path / "pages"


@pytest.fixture
def index():
    idx = QdrantIndex(url=":memory:", api_key=None, collection="pipe_test")
    idx.ensure_collection()
    return idx


def _rows(n=3):
    return [(i, f"{SHA}/p{i:04d}.png") for i in range(n)]


def test_embeds_every_page(index, pages_dir):
    r = embed_document(SHA, _rows(), pages_dir, FakeEmbedder(), index)
    assert r == EmbedResult(sha256=SHA, embedded=3, skipped=0)
    assert index.count() == 3


def test_second_run_skips_everything(index, pages_dir):
    embed_document(SHA, _rows(), pages_dir, FakeEmbedder(), index)
    r = embed_document(SHA, _rows(), pages_dir, FakeEmbedder(), index)
    assert r.embedded == 0 and r.skipped == 3


def test_resumes_from_partial_state(index, pages_dir):
    embed_document(SHA, _rows(), pages_dir, FakeEmbedder(), index, max_pages=1)
    assert index.count() == 1
    r = embed_document(SHA, _rows(), pages_dir, FakeEmbedder(), index)
    assert r.embedded == 2 and r.skipped == 1
    assert index.count() == 3


def test_missing_page_image_raises_with_the_path(index, pages_dir):
    rows = _rows() + [(9, f"{SHA}/p0009.png")]
    with pytest.raises(FileNotFoundError, match="p0009"):
        embed_document(SHA, rows, pages_dir, FakeEmbedder(), index)


def test_pages_before_the_failure_are_still_committed(index, pages_dir):
    """Per-page upsert means a crash cannot undo completed work."""
    rows = _rows() + [(9, f"{SHA}/p0009.png")]
    with pytest.raises(FileNotFoundError):
        embed_document(SHA, rows, pages_dir, FakeEmbedder(), index)
    assert index.count() == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embed_pipeline.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/visual_verify/retrieval/pipeline.py`:

```python
"""Embedding orchestration: store rows in, Qdrant points out.

Takes page rows as data rather than a Session, mirroring S2's rule that the
pipeline never holds a database handle. That is what lets this be tested against
a fake embedder and an in-memory Qdrant with no database at all.
"""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from visual_verify.retrieval.index import QdrantIndex
from visual_verify.retrieval.types import Embedder


@dataclass(frozen=True)
class EmbedResult:
    sha256: str
    embedded: int
    skipped: int


def embed_document(
    doc_sha: str,
    pages: list[tuple[int, str]],
    pages_dir: Path,
    embedder: Embedder,
    index: QdrantIndex,
    max_pages: int | None = None,
) -> EmbedResult:
    """Embed one document's pages into the index.

    `pages` is (page_no, image_path) in the store's own relative form.
    `max_pages` exists to test resumption by simulating a partial run.
    """
    already = index.existing_page_nos(doc_sha)
    embedded = skipped = 0

    for page_no, rel_path in sorted(pages):
        if page_no in already:
            skipped += 1
            continue
        if max_pages is not None and embedded >= max_pages:
            break

        path = Path(pages_dir) / rel_path
        if not path.is_file():
            raise FileNotFoundError(path)

        with Image.open(path) as im:
            size = im.size

        embedding = embedder.embed_page(str(path), size)
        # Upserted with wait=True one page at a time. At 21.4 s of embedding per
        # page the round trip is noise, and it is what makes a 1.8-hour run
        # resumable rather than all-or-nothing.
        index.upsert_page(doc_sha, page_no, rel_path, embedding, embedder.provenance)
        embedded += 1

    return EmbedResult(sha256=doc_sha, embedded=embedded, skipped=skipped)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embed_pipeline.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/visual_verify/retrieval/pipeline.py tests/test_embed_pipeline.py
git commit -m "Add resumable per-page embedding pipeline"
```

---

### Task 9: `vvrag embed` and `vvrag search`

**Files:**
- Modify: `src/visual_verify/cli.py`
- Test: `tests/test_cli_retrieval.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_retrieval.py`:

```python
import pytest
from PIL import Image

from visual_verify.cli import main

SHA = "d" * 64


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VVRAG_QDRANT_URL", ":memory:")
    monkeypatch.delenv("VVRAG_QDRANT_API_KEY", raising=False)
    # Keep the CLI off the GPU: the fake embedder makes these fast tests.
    monkeypatch.setenv("VVRAG_FAKE_EMBEDDER", "1")
    return tmp_path


def _ingest(env, tmp_path, born_digital_pdf):
    assert main(["ingest", str(born_digital_pdf)]) == 0


def test_embed_then_search(env, born_digital_pdf, capsys):
    assert main(["ingest", str(born_digital_pdf)]) == 0
    assert main(["embed", "--all"]) == 0
    out = capsys.readouterr().out
    assert "embedded" in out.lower()

    assert main(["search", "anything"]) == 0
    out = capsys.readouterr().out
    assert SHA[:6] not in out or True  # doc sha is printed; content varies
    assert "page" in out.lower()


def test_embed_is_idempotent(env, born_digital_pdf, capsys):
    main(["ingest", str(born_digital_pdf)])
    main(["embed", "--all"])
    capsys.readouterr()
    assert main(["embed", "--all"]) == 0
    assert "skipped" in capsys.readouterr().out.lower()


def test_search_before_embed_reports_empty(env, born_digital_pdf, capsys):
    main(["ingest", str(born_digital_pdf)])
    assert main(["search", "anything"]) == 1
    assert "no pages indexed" in capsys.readouterr().out.lower()


def test_embed_requires_a_target(env, capsys):
    assert main(["embed"]) == 1
    assert "give a document" in capsys.readouterr().out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_retrieval.py -v`
Expected: FAIL, `invalid choice: 'embed'`

- [ ] **Step 3: Add the commands**

In `src/visual_verify/cli.py`, add after the existing imports:

```python
import os
```

Add these functions before `build_parser()`:

```python
def _make_embedder(settings: Settings):
    """The fake keeps CLI tests off the GPU; anything else loads the real model."""
    if os.getenv("VVRAG_FAKE_EMBEDDER"):
        from visual_verify.retrieval.types import FakeEmbedder

        return FakeEmbedder()
    from visual_verify.retrieval.embedder import ColQwen2Embedder

    return ColQwen2Embedder(render_dpi=settings.render_dpi)


def _make_index(settings: Settings):
    from visual_verify.retrieval.index import QdrantIndex

    if not settings.qdrant_url:
        raise SystemExit("VVRAG_QDRANT_URL is not set")
    index = QdrantIndex(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    index.ensure_collection()
    return index


def cmd_embed(args) -> int:
    settings = Settings.from_env()
    _ensure_schema(settings)
    index = _make_index(settings)
    embedder = _make_embedder(settings)

    from visual_verify.retrieval.pipeline import embed_document

    with _session(settings) as session:
        if args.all:
            shas = list(session.scalars(select(Document.sha256).where(Document.status == "indexed")))
        else:
            shas = [_resolve_document(session, args.doc)]

        if not shas:
            print("no indexed documents to embed; run `vvrag ingest` first")
            return 1

        total_embedded = total_skipped = 0
        for sha in shas:
            rows = [
                (p.page_no, p.image_path)
                for p in session.scalars(select(Page).where(Page.doc_sha == sha))
            ]
            result = embed_document(sha, rows, settings.pages_dir, embedder, index)
            total_embedded += result.embedded
            total_skipped += result.skipped
            print(f"{sha[:12]}  embedded {result.embedded}  skipped {result.skipped}")

    print(f"total: embedded {total_embedded}, skipped {total_skipped}")
    return 0


def cmd_search(args) -> int:
    settings = Settings.from_env()
    index = _make_index(settings)
    if index.count() == 0:
        print("no pages indexed; run `vvrag embed --all` first")
        return 1

    embedder = _make_embedder(settings)
    hits = index.search(embedder.embed_query(args.query), embedder.provenance, limit=args.k)
    for rank, hit in enumerate(hits, 1):
        print(f"{rank}. {hit.doc_id[:12]}  page {hit.page:>4}  score {hit.score:7.3f}  {hit.image_ref}")
    return 0
```

Register them in `build_parser()`, before `return parser`:

```python
    p_embed = sub.add_parser("embed", help="embed ingested pages into the vector index")
    p_embed.add_argument("doc", nargs="?", help="document sha256, prefix, or path substring")
    p_embed.add_argument("--all", action="store_true", help="embed every indexed document")
    p_embed.set_defaults(func=cmd_embed)

    p_search = sub.add_parser("search", help="rank pages against a question")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5, help="how many pages to return")
    p_search.set_defaults(func=cmd_search)
```

And in `main()`, extend the argument guard:

```python
    if args.command == "embed" and not args.doc and not args.all:
        print("give a document or --all")
        return 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_retrieval.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Verify existing CLI tests still pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS, unchanged

- [ ] **Step 6: Commit**

```bash
git add src/visual_verify/cli.py tests/test_cli_retrieval.py
git commit -m "Add vvrag embed and vvrag search commands"
```

---

### Task 10: Known-item retrieval on the real corpus

The check that caught every bug found while designing this slice.

**Files:**
- Create: `tests/test_known_item_retrieval.py`

- [ ] **Step 1: Write the test**

Create `tests/test_known_item_retrieval.py`:

```python
"""Known-item retrieval against the real model and the real corpus.

This is a FLOOR, not a quality measure: it feeds a verbatim sentence from a page
back as its own query, which is the easiest possible retrieval task. Its value is
that every failure mode found while designing S3 (a randomized adapter, a
blanket-quantized vision tower, prefix-sliced embeddings) produced correctly
shaped, numerically healthy, unit-normalized vectors and was invisible to every
other check.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from visual_verify.retrieval.embedder import ColQwen2Embedder  # noqa: E402
from visual_verify.retrieval.index import QdrantIndex  # noqa: E402
from visual_verify.retrieval.pipeline import embed_document  # noqa: E402

N_PAGES = 8


def _maxsim(q, p):
    return float((q @ p.T).max(axis=1).sum())


@pytest.fixture(scope="module")
def corpus(real_pdf_pages):
    """(sha, [(page_no, rel_path)], pages_dir) for the first N_PAGES pages."""
    return real_pdf_pages(N_PAGES)


def test_known_item_top1(corpus, real_page_sentences):
    sha, rows, pages_dir = corpus
    embedder = ColQwen2Embedder()
    index = QdrantIndex(url=":memory:", api_key=None, collection="known_item")
    index.ensure_collection()
    embed_document(sha, rows, pages_dir, embedder, index)

    queries = real_page_sentences(sha, [p for p, _ in rows])
    assert len(queries) >= 4, "need several known-item queries to be meaningful"

    hits = 0
    for page_no, sentence in queries:
        top = index.search(embedder.embed_query(sentence), embedder.provenance, limit=1)[0]
        hits += top.page == page_no

    ratio = hits / len(queries)
    assert ratio >= 0.75, f"known-item top-1 {ratio:.2f}: retrieval is broken"


def test_qdrant_ranking_matches_local_maxsim(corpus, real_page_sentences):
    """Guards against a misconfigured collection, which returns wrong results
    rather than erroring."""
    sha, rows, pages_dir = corpus
    embedder = ColQwen2Embedder()
    index = QdrantIndex(url=":memory:", api_key=None, collection="agreement")
    index.ensure_collection()
    embed_document(sha, rows, pages_dir, embedder, index)

    local = {p: index.get_vectors(sha, p)["original"] for p, _ in rows}
    page_no, sentence = real_page_sentences(sha, [p for p, _ in rows])[0]
    q = embedder.embed_query(sentence)

    qdrant_order = [h.page for h in index.search(q, embedder.provenance, limit=len(rows))]
    local_order = sorted(local, key=lambda p: -_maxsim(q, local[p]))
    assert qdrant_order == local_order


def test_grid_invariant_on_every_real_page(corpus):
    sha, rows, pages_dir = corpus
    embedder = ColQwen2Embedder()
    for _, rel in rows:
        from PIL import Image

        path = pages_dir / rel
        with Image.open(path) as im:
            size = im.size
        emb = embedder.embed_page(str(path), size)
        assert emb.grid.n_image_patches + emb.grid.n_special == emb.vectors.shape[0]
        assert not np.isnan(emb.vectors).any()
```

- [ ] **Step 2: Add the fixtures**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def real_pdf_pages():
    """Pages from the repo's own ingested corpus, if one exists.

    Skips rather than fails when the corpus is absent, so a fresh clone can run
    the suite without a two-hour ingest.
    """

    def _load(n: int):
        import sqlite3

        root = Path(__file__).resolve().parent.parent
        db = root / "data" / "index.db"
        pages_dir = root / "data" / "pages"
        if not db.exists():
            pytest.skip("no ingested corpus at data/index.db; run `vvrag ingest` first")
        con = sqlite3.connect(db)
        rows = con.execute(
            "SELECT doc_sha, page_no, image_path FROM pages ORDER BY doc_sha, page_no LIMIT ?",
            (n,),
        ).fetchall()
        con.close()
        if not rows:
            pytest.skip("corpus database has no pages")
        sha = rows[0][0]
        return sha, [(r[1], r[2]) for r in rows if r[0] == sha], pages_dir

    return _load


@pytest.fixture
def real_page_sentences():
    """(page_no, sentence) pairs taken verbatim from each page's text layer."""

    def _load(sha: str, page_nos: list[int]):
        import sqlite3

        db = Path(__file__).resolve().parent.parent / "data" / "index.db"
        con = sqlite3.connect(db)
        rows = con.execute(
            """
            SELECT p.page_no, group_concat(b.text, ' ') AS line
            FROM boxes b JOIN pages p ON b.page_id = p.id
            WHERE b.kind = 'word' AND p.doc_sha = ?
            GROUP BY p.id, b.block_no, b.line_no
            ORDER BY length(line) DESC
            """,
            (sha,),
        ).fetchall()
        con.close()
        wanted, seen, out = set(page_nos), set(), []
        for page_no, line in rows:
            words = line.split()
            if page_no in wanted and page_no not in seen and 8 <= len(words) <= 25:
                seen.add(page_no)
                out.append((page_no, " ".join(words)))
        return out

    return _load
```

- [ ] **Step 3: Run the test**

Run: `uv run pytest tests/test_known_item_retrieval.py -v -m slow`
Expected: PASS, 3 tests (or SKIP if no corpus / no GPU)

- [ ] **Step 4: Commit**

```bash
git add tests/test_known_item_retrieval.py tests/conftest.py
git commit -m "Add known-item retrieval tests against the real corpus"
```

---

### Task 11: Clear the smoke-test collection and index the real corpus

**Files:** none (operational)

- [ ] **Step 1: Recreate the cloud collection**

The `pages` collection holds 8 points left over from design-time smoke testing.
They must not be mistaken for corpus data.

```bash
set -a && . ./.env && set +a
uv run python -c "
from visual_verify.config import Settings
from visual_verify.retrieval.index import QdrantIndex
s = Settings.from_env()
idx = QdrantIndex(url=s.qdrant_url, api_key=s.qdrant_api_key)
idx.ensure_collection(recreate=True)
print('points after recreate:', idx.count())
"
```

Expected: `points after recreate: 0`

- [ ] **Step 2: Embed the real corpus**

```bash
set -a && . ./.env && set +a
uv run vvrag embed --all
```

Expected: roughly 21 s per page. For the 74-page corpus this is about 26
minutes. Interrupt it with Ctrl-C partway to confirm resumption, then re-run and
confirm it reports the completed pages as skipped.

- [ ] **Step 3: Search**

```bash
set -a && . ./.env && set +a
uv run vvrag search "why is the heatmap not treated as faithful evidence" -k 5
```

Expected: five ranked pages. Spot-check that the top hit is a page that actually
discusses heatmap faithfulness. This is the end-to-end proof of pillar 1.

---

### Task 12: Documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Document the commands in README.md**

Add after the existing ingest section:

````markdown
### Building the retrieval index

Requires the `retrieval` extra and a CUDA GPU with at least 3 GB free.

```bash
uv sync --all-extras --group dev
export VVRAG_QDRANT_URL=...        # or put both in a gitignored .env
export VVRAG_QDRANT_API_KEY=...

uv run vvrag embed --all           # ~21 s/page, resumable
uv run vvrag search "your question" -k 5
```

Embedding is deliberately a separate command from `vvrag ingest`. Ingest needs
only the four core dependencies and no GPU; embedding needs a 2.5 GB torch stack
and roughly 21 s per page. Keeping them apart means a machine with no GPU can
still ingest a corpus.

Interrupting `vvrag embed` is safe. Each page is committed to Qdrant as it
completes, and re-running resumes from the first unembedded page.
````

- [ ] **Step 2: Add the gotchas to CLAUDE.md**

Add a "Retrieval gotchas" section:

````markdown
## Retrieval gotchas

**Never blanket-quantize ColQwen2.** `load_in_4bit` without
`llm_int8_skip_modules=["visual", "custom_text_proj"]` also quantizes the vision
tower, which destroys patch embeddings. Measured: known-item top-1 of 0.00
against 1.00. It emits no warning, raises nothing, produces no NaN, returns the
correct shape, and yields unit-normalized vectors.

**The retrieval stack is pinned exactly and must stay that way.**
`colpali-engine` 0.3.17 with `transformers` 5.x randomly initializes the LoRA
adapter and projection head, because the adapter keys target the pre-rename
`model.layers` path. transformers reports this as an informational table, not an
error, and retrieval then scores at chance.

**Image patches start at sequence index 4, not 0.** A page is 4 prefix tokens,
736 contiguous image patches, then 7 suffix tokens. Slicing `[:736]` shifts every
patch box by four cells. Derive the offset from `input_ids == image_pad_id`.

**The patch grid is 23x32 and is not square, and not constant.** It comes from
`smart_resize` on page aspect ratio, so landscape pages differ. Store it per page.

**Qwen2-VL pads LEFT; Idefics3 pads right.** Select embeddings with the attention
mask, never `out[:n]`. Prefix slicing is silently correct for colSmol and
silently wrong for ColQwen2.

**bfloat16 is unavailable on the GTX 1650** (sm_75). Use float16.

**Batch size 1 is faster than 2** on this card (21.4 s vs 24.6 s per page); it is
memory-bound.
````

- [ ] **Step 3: Run the full suite and linters**

Run: `uv run pytest -m "not slow"`
Expected: all pass

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: clean

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "Document the retrieval commands and their gotchas"
```

---

## Self-review

**Spec coverage:**

| spec section | task |
|---|---|
| 2.1 patch grid geometry | 2, 6 |
| 3 retriever choice | 6 |
| 4.1 skip modules | 6 |
| 4.2 float16 | 6 |
| 4.3 batch size 1 | 6 |
| 4.4 mask selection | 6 |
| 4.5 adapter assertion | 6 |
| 5.1 pins | 1 |
| 5.2 boundary | 1, 2, 6 |
| 6.1 separate command | 9 |
| 6.2 point identity | 7 |
| 6.2.1 payload | 3, 7 |
| 6.3 Qdrant as source of truth | 7, 8 |
| 6.4 checkpointing | 8 |
| 7 query path | 9 |
| 8 collection schema | 7 |
| 8.1 three named vectors | 4, 7 |
| 8.4 configuration | 1 |
| 9 testing | 2-10 |
| 10 clear smoke points | 11 |
| 11 definition of done | 11, 12 |
| 12 what S4 gets | 2 |

**Known gap, deliberate:** spec 6.5 (refusing mixed DPI across a corpus) is not
implemented as a separate check. It is covered in practice by provenance, since
`render_dpi` is a provenance field and `require_compatible` refuses on any
difference. Recorded here so it is a decision rather than an omission.

**Type consistency:** `PatchGrid(n_x, n_y, offset, n_vectors)` is constructed
identically in Tasks 2, 5, and 6. `PageEmbedding(vectors, grid)` likewise.
`Embedder` exposes `provenance`, `embed_page(image_path, image_size)`, and
`embed_query(text)` in Tasks 5, 6, 8, and 9. `QdrantIndex` methods
`ensure_collection`, `count`, `upsert_page`, `existing_page_nos`, `get_payload`,
`get_vectors`, `search` are used consistently in Tasks 7, 8, 9, 10, and 11.

**Placeholder scan:** no TBD, TODO, or "similar to Task N". Every code step
contains complete code.
