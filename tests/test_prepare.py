"""prepare_page() is the adapter the CLI and the API share."""

import pytest
from PIL import Image
from sqlalchemy.orm import Session

from visual_verify.cli import main
from visual_verify.config import Settings
from visual_verify.contracts import RetrievedPage
from visual_verify.prepare import PageNotFound, prepare_page, prepare_pages
from visual_verify.store.engine import make_engine


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VVRAG_QDRANT_URL", ":memory:")
    monkeypatch.delenv("VVRAG_QDRANT_API_KEY", raising=False)
    monkeypatch.setenv("VVRAG_FAKE_EMBEDDER", "1")
    return tmp_path


@pytest.fixture
def indexed(env, born_digital_pdf):
    assert main(["ingest", str(born_digital_pdf)]) == 0
    assert main(["embed", "--all"]) == 0
    return Settings.from_env()


def test_it_returns_boxes_vectors_and_a_grid_that_agree(indexed):
    from visual_verify.cli import _make_index

    settings = indexed
    index = _make_index(settings)
    with Session(make_engine(settings.db_url)) as session:
        page = prepare_page(session, index, settings, doc="born_digital", page_no=0)

    assert page.boxes, "the text layer should have produced word boxes"
    assert page.image_path.exists()
    assert page.page_vectors is not None
    # Pins only that the grid was built from the array it was fetched with, so
    # a payload read for one page and vectors read for another cannot pass. It
    # says nothing about orientation: n_vectors is a stored scalar that a swap
    # of n_x and n_y never touches, and PatchGrid validates through n_x * n_y,
    # which is commutative.
    assert page.grid.n_vectors == page.page_vectors.shape[0]

    w, h = Image.open(page.image_path).size
    # A transposed grid keeps n_x * n_y and therefore n_vectors identical, so the
    # count check above passes on a swap. Orientation is the discriminator: the
    # grid comes from smart_resize on the page aspect ratio, so a portrait page
    # must have more rows than columns. This is the S3 bug that placed every box
    # off-page while every shape and dtype looked right.
    assert (page.grid.n_x < page.grid.n_y) == (w < h), (
        f"grid {page.grid.n_x}x{page.grid.n_y} is transposed against a {w}x{h} page"
    )
    assert page.doc_name == "born_digital.pdf"


def test_an_unknown_document_raises_rather_than_returning_none(indexed):
    from visual_verify.cli import _make_index

    settings = indexed
    index = _make_index(settings)
    with Session(make_engine(settings.db_url)) as session:
        with pytest.raises(PageNotFound):
            prepare_page(session, index, settings, doc="no-such-doc", page_no=0)


def test_a_page_beyond_the_document_raises(indexed):
    from visual_verify.cli import _make_index

    settings = indexed
    index = _make_index(settings)
    with Session(make_engine(settings.db_url)) as session:
        with pytest.raises(PageNotFound):
            prepare_page(session, index, settings, doc="born_digital", page_no=99)


def _pdf(path, n_pages, label):
    import fitz

    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=612.0, height=792.0)
        page.insert_text((72.0, 100.0), f"{label} page {i} content here", fontsize=12)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def two_documents(env, tmp_path):
    """Ten pages in doc A, eight in doc B, ingested but not embedded.

    prepare_pages only chooses which pages to prepare, and prepare_page already
    serves an unembedded page (page_vectors=None), so an embed pass here would
    only add runtime.
    """
    from visual_verify.cli import _make_index

    a = _pdf(tmp_path / "alpha.pdf", 10, "Alpha")
    b = _pdf(tmp_path / "beta.pdf", 8, "Beta")
    assert main(["ingest", str(a)]) == 0
    assert main(["ingest", str(b)]) == 0

    settings = Settings.from_env()
    index = _make_index(settings)
    with Session(make_engine(settings.db_url)) as session:
        from visual_verify.prepare import resolve_document

        sha_a = resolve_document(session, "alpha").sha256
        sha_b = resolve_document(session, "beta").sha256
        yield settings, index, session, sha_a, sha_b


def _hit(doc_sha, page, score):
    return RetrievedPage(doc_id=doc_sha, page=page, image_ref=f"{doc_sha}/{page}.png", score=score)


def test_only_pages_of_the_top_hits_document_are_prepared(two_documents):
    """Retrieval is corpus-wide, so the raw top 3 can span documents, and a
    GroundedRegion carries `page` but no document identity: a region from
    another document could not be rendered, and merging two documents into one
    answer hides that it happened. The top hit's document wins and the rest are
    dropped."""
    settings, index, session, sha_a, sha_b = two_documents
    hits = [_hit(sha_a, 3, 9.0), _hit(sha_b, 7, 8.5), _hit(sha_a, 9, 8.0)]

    pages = prepare_pages(session, index, settings, hits)

    assert [(p.doc_sha, p.page_no) for p in pages] == [(sha_a, 3), (sha_a, 9)]


def test_retrieval_order_survives_rather_than_page_order(two_documents):
    """The test above cannot see a sort, because 3 then 9 is already ascending.
    A later task breaks ties between pages by retrieval rank, so a sort or a set
    here would change which page a claim is cited to while the answer still
    looked entirely ordinary."""
    settings, index, session, sha_a, _ = two_documents
    hits = [_hit(sha_a, 9, 9.0), _hit(sha_a, 3, 8.0), _hit(sha_a, 5, 7.0)]

    pages = prepare_pages(session, index, settings, hits)

    assert [p.page_no for p in pages] == [9, 3, 5]


def test_the_limit_caps_how_many_pages_are_prepared(two_documents):
    """Every prepared page costs four queries: a Page select, a Box select, then
    get_payload_or_none and get_vectors against Qdrant. The filter runs first,
    so the cap counts pages that survived it, not raw hits."""
    settings, index, session, sha_a, sha_b = two_documents
    hits = [
        _hit(sha_a, 0, 9.0),
        _hit(sha_b, 1, 8.9),
        _hit(sha_a, 1, 8.0),
        _hit(sha_a, 2, 7.0),
        _hit(sha_a, 4, 6.0),
    ]

    pages = prepare_pages(session, index, settings, hits, limit=3)

    assert [p.page_no for p in pages] == [0, 1, 2]


def test_no_hits_prepares_nothing_rather_than_raising(two_documents):
    """ask_events raises NoPagesIndexed for an empty result before it gets here,
    so this only has to not crash on the way."""
    settings, index, session, _, _ = two_documents

    assert prepare_pages(session, index, settings, []) == []


def test_an_ambiguous_document_needle_raises_instead_of_guessing(indexed, tmp_path):
    """cmd_ask's _resolve_document returns a candidate LIST when a needle
    matches more than one document. Silently taking the first is how
    `inspect proposal` used to pick whichever of proposal.pdf and
    reference_proposal.pdf was inserted first."""
    import fitz

    from visual_verify.cli import _make_index

    second = tmp_path / "born_digital_copy.pdf"
    doc = fitz.open()
    doc.new_page(width=612.0, height=792.0).insert_text((72.0, 100.0), "Other text", fontsize=12)
    doc.save(second)
    doc.close()
    assert main(["ingest", str(second)]) == 0

    settings = indexed
    index = _make_index(settings)
    with Session(make_engine(settings.db_url)) as session:
        with pytest.raises(PageNotFound):
            prepare_page(session, index, settings, doc="born_digital", page_no=0)
