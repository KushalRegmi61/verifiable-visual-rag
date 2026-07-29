"""The persistence seam.

The pipeline writes through this protocol and never imports a database driver.
That is what keeps SQLAlchemy behind the `store` extra while the ingest code
stays in the dependency-light core, and it makes the pipeline trivially testable
against MemorySink with no database at all.
"""

from dataclasses import dataclass, field
from typing import Protocol

from visual_verify.ingest.boxes import BoxRecord
from visual_verify.ingest.gate import RejectReason


@dataclass(frozen=True)
class DocumentRecord:
    sha256: str
    path: str
    n_pages: int


@dataclass(frozen=True)
class PageRecord:
    page_no: int
    image_path: str
    width_px: int
    height_px: int
    dpi: int


class Sink(Protocol):
    """Where ingest output goes. Implementations must be idempotent."""

    def begin_document(self, doc: DocumentRecord) -> None: ...

    def done_pages(self, sha256: str) -> set[int]:
        """Page numbers already persisted, so ingest can resume."""
        ...

    def write_page(self, sha256: str, page: PageRecord, boxes: list[BoxRecord]) -> None: ...

    def finish_document(self, sha256: str) -> None: ...

    def fail_document(self, sha256: str, reason: RejectReason, detail: str) -> None: ...


@dataclass
class MemorySink:
    """In-memory Sink for tests."""

    document: DocumentRecord | None = None
    pages: list[PageRecord] = field(default_factory=list)
    boxes_by_page: dict[int, list[BoxRecord]] = field(default_factory=dict)
    done: set[tuple[str, int]] = field(default_factory=set)
    finished: bool = False
    failure: tuple[str, RejectReason, str] | None = None

    def begin_document(self, doc: DocumentRecord) -> None:
        self.document = doc

    def done_pages(self, sha256: str) -> set[int]:
        return {p for s, p in self.done if s == sha256}

    def write_page(self, sha256: str, page: PageRecord, boxes: list[BoxRecord]) -> None:
        self.pages.append(page)
        self.boxes_by_page[page.page_no] = boxes
        self.done.add((sha256, page.page_no))

    def finish_document(self, sha256: str) -> None:
        self.finished = True

    def fail_document(self, sha256: str, reason: RejectReason, detail: str) -> None:
        self.failure = (sha256, reason, detail)
