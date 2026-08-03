"""Real-model tests. All slow: they download ~4 GB and need a CUDA GPU."""

import gc

import numpy as np
import pytest

pytestmark = pytest.mark.slow

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from visual_verify.retrieval.embedder import ColQwen2Embedder  # noqa: E402


@pytest.fixture(scope="module")
def embedder():
    emb = ColQwen2Embedder()
    yield emb
    # Release VRAM before the next slow module loads its own ColQwen2. Two live
    # instances do not fit in the 4 GB card, so without this teardown a full
    # `uv run pytest` fails with CUDA OOM during weight loading even though each
    # module passes on its own. Returning the fixture without freeing is the
    # kind of thing that only breaks when the whole suite runs.
    del emb
    gc.collect()
    torch.cuda.empty_cache()


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
