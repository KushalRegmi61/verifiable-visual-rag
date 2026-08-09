"""prepare_page() is the adapter the CLI and the API share."""

import pytest
from sqlalchemy.orm import Session

from visual_verify.cli import main
from visual_verify.config import Settings
from visual_verify.prepare import PageNotFound, prepare_page
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
    # The grid must describe the vectors it was fetched with. A mismatch here
    # is the failure mode that made grounding place boxes off-page in S3.
    assert page.grid.n_vectors == page.page_vectors.shape[0]
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
