"""The born-digital gate.

Scope boundary: this project reads embedded text layers and runs no OCR. A
scanned PDF has no candidate boxes, so grounding is impossible on it. That must
fail loudly at ingest rather than silently producing empty pages.
"""

import hashlib
from enum import Enum
from pathlib import Path

import fitz

CHUNK = 1 << 20


class RejectReason(str, Enum):
    CORRUPT = "corrupt"
    ENCRYPTED = "encrypted"
    NO_TEXT_LAYER = "no_text_layer"
    EMPTY = "empty"


class GateError(Exception):
    """Raised when a document may not be ingested."""

    def __init__(self, reason: RejectReason, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}" if detail else reason.value)


def fingerprint(path: Path) -> str:
    """SHA-256 of file bytes. Identity is content, not filename."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def text_page_ratio(doc: fitz.Document) -> float:
    """Fraction of pages carrying at least one extractable word."""
    if doc.page_count == 0:
        return 0.0
    with_text = sum(1 for page in doc if page.get_text("words"))
    return with_text / doc.page_count


def open_and_check(path: Path, min_text_page_ratio: float = 0.6) -> fitz.Document:
    """Open a PDF and reject it unless it is ingestable. Caller closes the result."""
    try:
        doc = fitz.open(path)
    except RuntimeError as exc:
        raise GateError(RejectReason.CORRUPT, str(exc)) from exc

    if doc.needs_pass:
        doc.close()
        raise GateError(RejectReason.ENCRYPTED, "password required")

    if doc.page_count == 0:
        doc.close()
        raise GateError(RejectReason.EMPTY, "no pages")

    ratio = text_page_ratio(doc)
    if ratio < min_text_page_ratio:
        doc.close()
        raise GateError(
            RejectReason.NO_TEXT_LAYER,
            f"only {ratio:.0%} of pages have a text layer, need {min_text_page_ratio:.0%}",
        )

    return doc
