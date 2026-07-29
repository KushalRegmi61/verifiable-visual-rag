"""Ingest orchestration.

Idempotency and resumability are both keyed on the content hash: re-ingesting an
unchanged PDF is free, and a 300-page batch that dies at page 200 resumes at 201.
"""

from dataclasses import dataclass
from pathlib import Path

from visual_verify.ingest.boxes import extract_boxes
from visual_verify.ingest.gate import GateError, fingerprint, open_and_check
from visual_verify.ingest.render import render_page
from visual_verify.ingest.sink import DocumentRecord, PageRecord, Sink


@dataclass(frozen=True)
class IngestResult:
    sha256: str
    path: str
    n_pages: int
    pages_written: int
    pages_skipped: int


def ingest_pdf(
    pdf_path: Path,
    sink: Sink,
    pages_dir: Path,
    dpi: int = 150,
    min_text_page_ratio: float = 0.6,
    max_pages: int | None = None,
) -> IngestResult:
    """Ingest one PDF: gate, render, extract boxes, persist.

    max_pages exists to test resumption by simulating a partial run. Production
    callers leave it None.
    """
    pdf_path = Path(pdf_path)
    sha = fingerprint(pdf_path) if pdf_path.exists() else ""

    try:
        doc = open_and_check(pdf_path, min_text_page_ratio=min_text_page_ratio)
    except GateError as exc:
        sink.fail_document(sha, exc.reason, exc.detail)
        raise

    try:
        sink.begin_document(
            DocumentRecord(sha256=sha, path=str(pdf_path), n_pages=doc.page_count)
        )
        already = sink.done_pages(sha)
        written = skipped = 0

        for page_no, page in enumerate(doc):
            if page_no in already:
                skipped += 1
                continue
            if max_pages is not None and written >= max_pages:
                break

            rel = f"{sha[:12]}/p{page_no:04d}.png"
            rendered = render_page(page, pages_dir / rel, dpi=dpi)
            boxes = extract_boxes(page)

            sink.write_page(
                sha,
                PageRecord(
                    page_no=page_no,
                    image_path=rel,
                    width_px=rendered.width_px,
                    height_px=rendered.height_px,
                    dpi=dpi,
                ),
                boxes,
            )
            written += 1

        if written + skipped == doc.page_count:
            sink.finish_document(sha)

        return IngestResult(
            sha256=sha,
            path=str(pdf_path),
            n_pages=doc.page_count,
            pages_written=written,
            pages_skipped=skipped,
        )
    finally:
        doc.close()
