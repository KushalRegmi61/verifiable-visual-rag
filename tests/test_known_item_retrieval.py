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
from PIL import Image

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


# All three fixtures below are module-scoped. `real_pdf_pages` (in conftest.py)
# is module-scoped too, so `corpus` depending on it does not trip pytest's
# ScopeMismatch check. The embedder loads the model once (~model load time is
# non-trivial on top of the ~21s/page embed cost) and `index` embeds the corpus
# exactly once via embed_document, shared by both tests that need a populated
# collection. Only test_grid_invariant_on_every_real_page re-embeds pages
# directly, because it is specifically exercising embed_page's return shape,
# not the stored index.
@pytest.fixture(scope="module")
def embedder():
    return ColQwen2Embedder()


@pytest.fixture(scope="module")
def corpus(real_pdf_pages):
    """(sha, [(page_no, rel_path)], pages_dir) for the first N_PAGES pages."""
    return real_pdf_pages(N_PAGES)


@pytest.fixture(scope="module")
def index(corpus, embedder):
    sha, rows, pages_dir = corpus
    idx = QdrantIndex(url=":memory:", api_key=None, collection="known_item")
    idx.ensure_collection()
    embed_document(sha, rows, pages_dir, embedder, idx)
    return idx


def test_known_item_top1(corpus, index, embedder, real_page_sentences):
    sha, rows, _ = corpus
    queries = real_page_sentences(sha, [p for p, _ in rows])
    assert len(queries) >= 4, "need several known-item queries to be meaningful"

    hits = 0
    for page_no, sentence in queries:
        top = index.search(embedder.embed_query(sentence), embedder.provenance, limit=1)[0]
        hits += top.page == page_no

    ratio = hits / len(queries)
    assert ratio >= 0.75, f"known-item top-1 {ratio:.2f}: retrieval is broken"


def test_qdrant_ranking_matches_local_maxsim(corpus, index, embedder, real_page_sentences):
    """Guards against a misconfigured collection, which returns wrong results
    rather than erroring."""
    sha, rows, _ = corpus
    local = {p: index.get_vectors(sha, p)["original"] for p, _ in rows}
    page_no, sentence = real_page_sentences(sha, [p for p, _ in rows])[0]
    q = embedder.embed_query(sentence)

    qdrant_order = [h.page for h in index.search(q, embedder.provenance, limit=len(rows))]
    local_order = sorted(local, key=lambda p: -_maxsim(q, local[p]))
    assert qdrant_order == local_order


def test_grid_invariant_on_every_real_page(corpus, embedder):
    sha, rows, pages_dir = corpus
    for _, rel in rows:
        path = pages_dir / rel
        with Image.open(path) as im:
            size = im.size
        emb = embedder.embed_page(str(path), size)
        assert emb.grid.n_image_patches + emb.grid.n_special == emb.vectors.shape[0]
        assert not np.isnan(emb.vectors).any()
