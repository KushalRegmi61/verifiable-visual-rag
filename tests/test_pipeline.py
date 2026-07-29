import pytest

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

    doc = sink.documents[result.sha256]
    assert doc.sha256 == result.sha256
    assert doc.n_pages == 1


def test_boxes_reach_the_sink(born_digital_pdf, tmp_path):
    sink = MemorySink()
    result = ingest_pdf(born_digital_pdf, sink, pages_dir=tmp_path, dpi=72)

    boxes = sink.boxes_by_page[(result.sha256, 0)]
    assert [b.text for b in boxes if b.kind == "word"][:2] == ["Revenue", "grew"]


def test_rejects_scanned_and_records_failure(scanned_pdf, tmp_path):
    sink = MemorySink()
    try:
        ingest_pdf(scanned_pdf, sink, pages_dir=tmp_path, dpi=72)
    except GateError as exc:
        assert exc.reason is RejectReason.NO_TEXT_LAYER
    else:
        raise AssertionError("expected GateError")

    assert sink.failures
    assert sink.failures[0][2] is RejectReason.NO_TEXT_LAYER


def test_finish_is_not_called_on_a_partial_run(multipage_pdf, tmp_path):
    """The trickiest line in the pipeline: finish only when written+skipped == n_pages."""
    sink = MemorySink()
    result = ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72, max_pages=2)
    assert sink.finished == set()

    ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72)
    assert sink.finished == {result.sha256}


def test_resumes_after_a_mid_document_crash(multipage_pdf, tmp_path, monkeypatch):
    """The real production failure path, not the max_pages simulation of it."""
    from visual_verify.ingest import pipeline as pipeline_module

    real_render = pipeline_module.render_page
    calls = {"n": 0}

    def flaky(page, out_path, dpi):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated disk failure")
        return real_render(page, out_path, dpi)

    monkeypatch.setattr(pipeline_module, "render_page", flaky)

    sink = MemorySink()
    with pytest.raises(OSError):
        ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72)
    assert len(sink.pages) == 1
    assert sink.finished == set()

    monkeypatch.setattr(pipeline_module, "render_page", real_render)
    second = ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72)
    assert second.pages_written == 2
    assert second.pages_skipped == 1
    assert len(sink.pages) == 3


def test_two_documents_do_not_collide_in_one_sink(multipage_pdf, born_digital_pdf, tmp_path):
    sink = MemorySink()
    a = ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72)
    b = ingest_pdf(born_digital_pdf, sink, pages_dir=tmp_path, dpi=72)

    assert a.sha256 != b.sha256
    assert len(sink.documents) == 2
    assert len(sink.pages) == 4
    assert (a.sha256, 0) in sink.boxes_by_page
    assert (b.sha256, 0) in sink.boxes_by_page
    assert sink.boxes_by_page[(a.sha256, 0)] != sink.boxes_by_page[(b.sha256, 0)]


def test_identity_is_content_not_filename(multipage_pdf, tmp_path):
    import shutil

    renamed = tmp_path / "different_name.pdf"
    shutil.copy(multipage_pdf, renamed)

    sink = MemorySink()
    ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72)
    second = ingest_pdf(renamed, sink, pages_dir=tmp_path, dpi=72)

    assert second.pages_written == 0
    assert second.pages_skipped == 3


def test_missing_file_raises_rather_than_recording_a_failure(tmp_path):
    sink = MemorySink()
    with pytest.raises(FileNotFoundError):
        ingest_pdf(tmp_path / "nope.pdf", sink, pages_dir=tmp_path, dpi=72)
    assert sink.failures == []


def test_encrypted_document_records_its_path(encrypted_pdf, tmp_path):
    """Only NO_TEXT_LAYER was covered; the failure row must carry the path."""
    sink = MemorySink()
    with pytest.raises(GateError):
        ingest_pdf(encrypted_pdf, sink, pages_dir=tmp_path, dpi=72)

    sha, path, reason, _detail = sink.failures[0]
    assert reason is RejectReason.ENCRYPTED
    assert path == str(encrypted_pdf)
    assert len(sha) == 64


def test_page_record_dimensions_match_the_written_png(born_digital_pdf, tmp_path):
    from PIL import Image

    sink = MemorySink()
    ingest_pdf(born_digital_pdf, sink, pages_dir=tmp_path, dpi=72)

    page = sink.pages[0]
    assert Image.open(tmp_path / page.image_path).size == (page.width_px, page.height_px)


def test_refuses_to_mix_dpi_within_a_document(multipage_pdf, tmp_path):
    sink = MemorySink()
    ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=72, max_pages=1)

    with pytest.raises(ValueError, match="already has pages rendered at 72 dpi"):
        ingest_pdf(multipage_pdf, sink, pages_dir=tmp_path, dpi=150)
