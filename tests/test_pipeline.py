from visual_verify.ingest.gate import GateError, RejectReason
from visual_verify.ingest.pipeline import IngestResult, ingest_pdf
from visual_verify.ingest.sink import MemorySink


def test_ingests_all_pages(multipage_pdf, tmp_path):
    sink = MemorySink()
    result = ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72)

    assert isinstance(result, IngestResult)
    assert result.pages_written == 3
    assert result.pages_skipped == 0
    assert len(sink.pages) == 3
    assert len(sink.boxes_by_page) == 3


def test_writes_one_image_per_page(multipage_pdf, tmp_path):
    sink = MemorySink()
    ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72)

    for page in sink.pages:
        assert (tmp_path / page.image_path).exists()


def test_image_paths_are_relative_and_hash_scoped(multipage_pdf, tmp_path):
    sink = MemorySink()
    result = ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72)

    for page in sink.pages:
        assert not page.image_path.startswith("/")
        assert result.sha256[:12] in page.image_path


def test_skips_pages_the_sink_already_has(multipage_pdf, tmp_path):
    sink = MemorySink()
    ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72)

    second = ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72)
    assert second.pages_written == 0
    assert second.pages_skipped == 3
    assert len(sink.pages) == 3


def test_resumes_after_partial_ingest(multipage_pdf, tmp_path):
    sink = MemorySink()
    first = ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72, max_pages=2)
    assert first.pages_written == 2

    second = ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72)
    assert second.pages_written == 1
    assert second.pages_skipped == 2
    assert len(sink.pages) == 3


def test_records_document_metadata(born_digital_pdf, tmp_path):
    sink = MemorySink()
    result = ingest_pdf(born_digital_pdf, sink, pages_dir=tmp_path, dpi=72)

    assert sink.document is not None
    assert sink.document.sha256 == result.sha256
    assert sink.document.n_pages == 1


def test_boxes_reach_the_sink(born_digital_pdf, tmp_path):
    sink = MemorySink()
    ingest_pdf(born_digital_pdf, sink, pages_dir=tmp_path, dpi=72)

    boxes = sink.boxes_by_page[0]
    assert [b.text for b in boxes if b.kind == "word"][:2] == ["Revenue", "grew"]


def test_rejects_scanned_and_records_failure(scanned_pdf, tmp_path):
    sink = MemorySink()
    try:
        ingest_pdf(scanned_pdf, sink, pages_dir=tmp_path, dpi=72)
    except GateError as exc:
        assert exc.reason is RejectReason.NO_TEXT_LAYER
    else:
        raise AssertionError("expected GateError")

    assert sink.failure is not None
    assert sink.failure[1] is RejectReason.NO_TEXT_LAYER
