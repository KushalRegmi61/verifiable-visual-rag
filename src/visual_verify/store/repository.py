"""SQLAlchemy implementation of the Sink protocol, plus status queries.

Box inserts go through bulk_insert_mappings rather than per-row ORM adds. At
roughly 500 words per page this is the difference between a usable ingest and a
crawling one, and it is what keeps the Postgres path viable.
"""

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from visual_verify.ingest.boxes import BoxRecord
from visual_verify.ingest.gate import RejectReason
from visual_verify.ingest.sink import DocumentRecord, PageRecord
from visual_verify.store.models import Box, Document, Job, Page


class SqlSink:
    """Persists ingest output. Idempotent: re-running writes nothing new."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def begin_document(self, doc: DocumentRecord) -> None:
        existing = self.session.get(Document, doc.sha256)
        if existing is None:
            self.session.add(
                Document(sha256=doc.sha256, path=doc.path, n_pages=doc.n_pages,
                         status="pending")
            )
            self.session.flush()

    def done_pages(self, sha256: str) -> set[int]:
        rows = self.session.scalars(select(Page.page_no).where(Page.doc_sha == sha256))
        return set(rows)

    def write_page(self, sha256: str, page: PageRecord, boxes: list[BoxRecord]) -> None:
        row = Page(
            doc_sha=sha256,
            page_no=page.page_no,
            image_path=page.image_path,
            width_px=page.width_px,
            height_px=page.height_px,
            dpi=page.dpi,
        )
        self.session.add(row)
        self.session.flush()  # assigns row.id

        if boxes:
            self.session.bulk_insert_mappings(
                Box,
                [
                    {
                        "page_id": row.id,
                        "kind": b.kind,
                        "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1,
                        "text": b.text,
                        "block_no": b.block_no,
                        "line_no": b.line_no,
                        "word_no": b.word_no,
                    }
                    for b in boxes
                ],
            )

        self.session.add(
            Job(doc_sha=sha256, stage="page", state="done", page_no=page.page_no)
        )

    def checkpoint(self) -> None:
        """Commit. The pipeline calls this per page so a crash cannot undo them."""
        self.session.commit()

    def finish_document(self, sha256: str) -> None:
        doc = self.session.get(Document, sha256)
        if doc is not None:
            doc.status = "indexed"

    def fail_document(
        self, sha256: str, path: str, reason: RejectReason, detail: str
    ) -> None:
        """Record a rejection.

        The gate runs before begin_document, so this may be the first time the
        store sees this document. Upsert the Document row so the rejected path
        is queryable, and append a Job row so repeated attempts are a log rather
        than silently overwriting each other.

        An already-indexed document is NOT downgraded to failed. Identity is the
        content hash, so a re-gate failing means the gate changed its mind, not
        that the stored pages became invalid. Flipping the status while leaving
        the pages in place would leave the store in an uninterpretable state
        (status=failed with pages_done=n_pages). The Job row still records the
        attempt.
        """
        doc = self.session.get(Document, sha256)
        if doc is None:
            self.session.add(
                Document(sha256=sha256, path=path, n_pages=0, status="failed")
            )
        elif doc.status != "indexed":
            doc.status = "failed"
        self.session.add(
            Job(
                doc_sha=sha256,
                stage="gate",
                state="failed",
                error=f"{reason.value}: {detail}" if detail else reason.value,
            )
        )


@dataclass(frozen=True)
class DocumentStatus:
    sha256: str
    path: str
    n_pages: int
    pages_done: int
    status: str


def document_status(session: Session) -> list[DocumentStatus]:
    """One row per document, for `vvrag status`. Includes rejected documents."""
    counts = dict(
        session.execute(
            select(Page.doc_sha, func.count(Page.id)).group_by(Page.doc_sha)
        ).all()
    )
    return [
        DocumentStatus(
            sha256=d.sha256,
            path=d.path,
            n_pages=d.n_pages,
            pages_done=counts.get(d.sha256, 0),
            status=d.status,
        )
        # sha256 is the tiebreaker: same-second inserts would otherwise order
        # nondeterministically, which makes CLI output diffs noisy for no reason.
        for d in session.scalars(
            select(Document).order_by(Document.created_at, Document.sha256)
        )
    ]
