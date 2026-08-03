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
