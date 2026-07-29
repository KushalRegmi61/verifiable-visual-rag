import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from visual_verify.ingest.boxes import BoxRecord
from visual_verify.ingest.gate import GateError, RejectReason
from visual_verify.ingest.pipeline import ingest_pdf
from visual_verify.ingest.sink import DocumentRecord, PageRecord
from visual_verify.store.engine import make_engine
from visual_verify.store.models import Base, Box, Document, Job, Page
from visual_verify.store.repository import SqlSink, document_status


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_write_page_persists_rows(session):
    sink = SqlSink(session)
    sink.begin_document(DocumentRecord(sha256="abc123", path="/x.pdf", n_pages=1))
    sink.write_page(
        "abc123",
        PageRecord(page_no=0, image_path="abc123/p0000.png", width_px=100, height_px=200, dpi=72),
        [
            BoxRecord(
                kind="word",
                x0=0.1,
                y0=0.2,
                x1=0.3,
                y1=0.25,
                text="hello",
                block_no=0,
                line_no=0,
                word_no=0,
            )
        ],
    )
    session.commit()

    assert session.scalar(select(func.count()).select_from(Document)) == 1
    assert session.scalar(select(func.count()).select_from(Page)) == 1
    assert session.scalar(select(func.count()).select_from(Box)) == 1
    assert session.scalar(select(Box).limit(1)).text == "hello"


def test_done_pages_reports_persisted_pages(session):
    sink = SqlSink(session)
    sink.begin_document(DocumentRecord(sha256="abc123", path="/x.pdf", n_pages=3))
    for n in (0, 2):
        sink.write_page(
            "abc123",
            PageRecord(page_no=n, image_path=f"abc123/p{n}.png", width_px=10, height_px=10, dpi=72),
            [],
        )
    session.commit()

    assert sink.done_pages("abc123") == {0, 2}


def test_done_pages_is_scoped_to_one_document(session):
    """A page is identified by (sha256, page_no), never page_no alone."""
    sink = SqlSink(session)
    for sha in ("aaa", "bbb"):
        sink.begin_document(DocumentRecord(sha256=sha, path=f"/{sha}.pdf", n_pages=1))
        sink.write_page(
            sha,
            PageRecord(page_no=0, image_path=f"{sha}/p0.png", width_px=10, height_px=10, dpi=72),
            [BoxRecord(kind="word", x0=0.1, y0=0.1, x1=0.2, y1=0.2, text=sha)],
        )
    session.commit()

    assert sink.done_pages("aaa") == {0}
    assert sink.done_pages("bbb") == {0}
    assert session.scalar(select(func.count()).select_from(Page)) == 2
    assert session.scalar(select(func.count()).select_from(Box)) == 2


def test_done_pages_empty_for_unknown_document(session):
    assert SqlSink(session).done_pages("nothing") == set()


def test_begin_document_is_idempotent(session):
    sink = SqlSink(session)
    doc = DocumentRecord(sha256="abc123", path="/x.pdf", n_pages=1)
    sink.begin_document(doc)
    sink.begin_document(doc)
    session.commit()

    assert session.scalar(select(func.count()).select_from(Document)) == 1


def test_finish_document_marks_indexed(session):
    sink = SqlSink(session)
    sink.begin_document(DocumentRecord(sha256="abc123", path="/x.pdf", n_pages=1))
    sink.finish_document("abc123")
    session.commit()

    assert session.get(Document, "abc123").status == "indexed"


def test_fail_document_records_reason_and_path(session):
    sink = SqlSink(session)
    sink.fail_document(
        "bad999", "/scans/bad.pdf", RejectReason.NO_TEXT_LAYER, "0% of pages have text"
    )
    session.commit()

    doc = session.get(Document, "bad999")
    assert doc is not None
    assert doc.path == "/scans/bad.pdf"
    assert doc.status == "failed"

    job = session.scalar(select(Job).where(Job.doc_sha == "bad999"))
    assert job.state == "failed"
    assert job.error.startswith("no_text_layer")


def test_full_ingest_through_sql_sink(multipage_pdf, tmp_path, session):
    sink = SqlSink(session)
    result = ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72)
    session.commit()

    assert result.pages_written == 3
    assert session.scalar(select(func.count()).select_from(Page)) == 3
    assert session.scalar(select(func.count()).select_from(Box)) > 0


def test_reingest_is_a_noop(multipage_pdf, tmp_path, session):
    sink = SqlSink(session)
    ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72)
    session.commit()
    boxes_before = session.scalar(select(func.count()).select_from(Box))

    second = ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72)
    session.commit()

    assert second.pages_written == 0
    assert second.pages_skipped == 3
    assert session.scalar(select(func.count()).select_from(Box)) == boxes_before


def test_boxes_round_trip_with_full_fidelity(born_digital_pdf, tmp_path, session):
    """Normalized coordinates must survive the DB unchanged."""
    import fitz

    from visual_verify.ingest.boxes import extract_boxes, word_boxes

    sink = SqlSink(session)
    ingest_pdf(born_digital_pdf, sink, pages_dir=tmp_path, dpi=72)
    session.commit()

    doc = fitz.open(born_digital_pdf)
    expected = word_boxes(extract_boxes(doc[0]))
    doc.close()

    page = session.scalar(select(Page))
    stored = list(
        session.scalars(
            select(Box).where(Box.page_id == page.id, Box.kind == "word").order_by(Box.id)
        )
    )

    assert len(stored) == len(expected)
    for got, want in zip(stored, expected, strict=True):
        assert got.text == want.text
        assert got.x0 == pytest.approx(want.x0)
        assert got.y1 == pytest.approx(want.y1)
        assert got.block_no == want.block_no
        assert got.word_no == want.word_no


def test_document_status_summarizes(multipage_pdf, tmp_path, session):
    sink = SqlSink(session)
    ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72)
    session.commit()

    rows = document_status(session)
    assert len(rows) == 1
    assert rows[0].n_pages == 3
    assert rows[0].pages_done == 3
    assert rows[0].status == "indexed"


def test_document_status_includes_rejected_documents(scanned_pdf, tmp_path, session):
    sink = SqlSink(session)
    with pytest.raises(GateError):
        ingest_pdf(scanned_pdf, sink, pages_dir=tmp_path, dpi=72)
    session.commit()

    rows = document_status(session)
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert rows[0].pages_done == 0


def test_sqlite_enforces_foreign_keys(tmp_path):
    """A page referencing a missing document must fail on SQLite as it would on Postgres."""
    from sqlalchemy.exc import IntegrityError

    engine = make_engine(f"sqlite:///{tmp_path / 'fk.db'}")
    Base.metadata.create_all(engine)

    with Session(engine) as s:
        s.add(
            Page(
                doc_sha="nonexistent",
                page_no=0,
                image_path="x.png",
                width_px=1,
                height_px=1,
                dpi=72,
            )
        )
        with pytest.raises(IntegrityError):
            s.commit()


def test_created_at_round_trips_as_aware(tmp_path):
    from datetime import UTC, datetime

    engine = make_engine(f"sqlite:///{tmp_path / 'tz.db'}")
    Base.metadata.create_all(engine)

    with Session(engine) as s:
        s.add(Document(sha256="a" * 64, path="/x.pdf", n_pages=1, status="pending"))
        s.commit()
        got = s.get(Document, "a" * 64)
        assert got.created_at.tzinfo is not None
        # Must be comparable to an aware datetime without raising.
        assert (datetime.now(UTC) - got.created_at).total_seconds() < 60


def test_fail_does_not_downgrade_an_indexed_document(session):
    sink = SqlSink(session)
    sink.begin_document(DocumentRecord(sha256="abc123", path="/x.pdf", n_pages=1))
    sink.finish_document("abc123")
    session.commit()

    sink.fail_document("abc123", "/x.pdf", RejectReason.NO_TEXT_LAYER, "gate retuned")
    session.commit()

    assert session.get(Document, "abc123").status == "indexed"
    assert session.scalar(select(func.count()).select_from(Job).where(Job.state == "failed")) == 1


def test_crash_mid_document_leaves_completed_pages_durable(multipage_pdf, tmp_path, monkeypatch):
    """The pipeline promises resumption; without a per-page commit it cannot deliver."""
    from visual_verify.ingest import pipeline as pipeline_module

    engine = make_engine(f"sqlite:///{tmp_path / 'crash.db'}")
    Base.metadata.create_all(engine)

    real_render = pipeline_module.render_page
    calls = {"n": 0}

    def flaky(page, out_path, dpi):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("simulated crash on page 3")
        return real_render(page, out_path, dpi)

    monkeypatch.setattr(pipeline_module, "render_page", flaky)

    with Session(engine) as s:
        with pytest.raises(OSError):
            ingest_pdf(multipage_pdf, SqlSink(s), pages_dir=tmp_path, dpi=72)

    # A fresh session must see the two pages that completed before the crash.
    with Session(engine) as s:
        assert s.scalar(select(func.count()).select_from(Page)) == 2

    monkeypatch.setattr(pipeline_module, "render_page", real_render)
    with Session(engine) as s:
        result = ingest_pdf(multipage_pdf, SqlSink(s), pages_dir=tmp_path, dpi=72)
        assert result.pages_written == 1
        assert result.pages_skipped == 2


def test_naive_datetime_is_rejected_and_offsets_are_normalized(tmp_path):
    """The behavior most likely to surprise a caller, currently untested."""
    from datetime import UTC, datetime, timedelta, timezone

    from sqlalchemy.exc import StatementError

    engine = make_engine(f"sqlite:///{tmp_path / 'tz2.db'}")
    Base.metadata.create_all(engine)

    npt = timezone(timedelta(hours=5, minutes=45))
    with Session(engine) as s:
        s.add(
            Document(
                sha256="b" * 64,
                path="/x.pdf",
                n_pages=1,
                created_at=datetime(2026, 1, 1, 12, 0, tzinfo=npt),
            )
        )
        s.commit()
        got = s.get(Document, "b" * 64)
        assert got.created_at == datetime(2026, 1, 1, 6, 15, tzinfo=UTC)

    with Session(engine) as s:
        s.add(
            Document(
                sha256="c" * 64, path="/y.pdf", n_pages=1, created_at=datetime(2026, 1, 1, 12, 0)
            )
        )
        with pytest.raises(StatementError):
            s.commit()


def test_document_recovers_after_a_gate_rejection(session, tmp_path, multipage_pdf):
    """Rejected then accepted must not leave n_pages stuck at 0."""
    sink = SqlSink(session)
    sink.fail_document(
        "will_pass_later", str(multipage_pdf), RejectReason.NO_TEXT_LAYER, "threshold too high"
    )
    session.commit()
    assert session.get(Document, "will_pass_later").n_pages == 0

    sink.begin_document(
        DocumentRecord(sha256="will_pass_later", path=str(multipage_pdf), n_pages=3)
    )
    sink.finish_document("will_pass_later")
    session.commit()

    doc = session.get(Document, "will_pass_later")
    assert doc.n_pages == 3
    assert doc.status == "indexed"
