"""Candidate box extraction: the foundation everything downstream stands on.

Boxes are stored at WORD granularity with their block/line/word indices, plus
table cells. Coarser granularities (line, block, span) are derived by grouping
at query time, never stored. This is what lets S4 retune which candidate set
snap-to-box ranks without re-ingesting, and lets S7 union the words covering an
arbitrary answer substring to build a gold box.

COORDINATE HANDLING, THE THING TO GET RIGHT:
page.rect and get_pixmap() both use DISPLAYED space, and that is the space every
stored box is normalized against. The two sources of boxes do NOT agree about
which space they hand back, and neither raises if you get it wrong:

  - get_text("words") returns UNROTATED text space. On a /Rotate 90 page its
    coordinates are byte-identical to the unrotated page's. These must be
    multiplied by page.rotation_matrix (the identity when unrotated).
  - find_tables() returns DISPLAYED space already. Multiplying these by
    rotation_matrix rotates them a second time and puts every cell off-page.

Both were verified empirically on PyMuPDF 1.28; see the per-call-site comments.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import fitz

logger = logging.getLogger(__name__)

BoxKind = Literal["word", "table_cell"]


@dataclass(frozen=True)
class BoxRecord:
    """One candidate box, normalized to 0-1 against the displayed page rect."""

    kind: BoxKind
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block_no: int = -1
    line_no: int = -1
    word_no: int = -1

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """The box as a plain tuple, matching contracts.BBox."""
        return (self.x0, self.y0, self.x1, self.y1)


def word_boxes(boxes: list[BoxRecord]) -> list[BoxRecord]:
    """Only the word boxes.

    Table cells carry block_no/line_no/word_no of -1. Grouping a mixed list by
    those fields would collapse every cell on the page into one fabricated
    "line". Any consumer that groups by hierarchy must filter through here first.
    """
    return [b for b in boxes if b.kind == "word"]


def _normalize(rect: fitz.Rect, page: fitz.Page) -> tuple[float, float, float, float] | None:
    """Normalize an already-DISPLAYED-space rect to 0-1 against page.rect.

    Callers are responsible for getting their rect into displayed space first;
    the two sources disagree about this and each call site says which it is.

    Returns None for degenerate boxes, which are dropped rather than stored.
    """
    r = fitz.Rect(rect)
    r.normalize()  # rotation can invert x0/x1 or y0/y1

    w, h = page.rect.width, page.rect.height
    if w <= 0 or h <= 0:
        raise ValueError(f"page {page.number} has non-positive rect {page.rect}")

    x0 = min(max(r.x0 / w, 0.0), 1.0)
    y0 = min(max(r.y0 / h, 0.0), 1.0)
    x1 = min(max(r.x1 / w, 0.0), 1.0)
    y1 = min(max(r.y1 / h, 0.0), 1.0)

    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _word_boxes(page: fitz.Page) -> list[BoxRecord]:
    out: list[BoxRecord] = []
    for x0, y0, x1, y1, word, block_no, line_no, word_no in page.get_text("words"):
        if not word.strip():
            continue
        # Words come back in UNROTATED text space, so map them into displayed
        # space. On an unrotated page rotation_matrix is the identity.
        norm = _normalize(fitz.Rect(x0, y0, x1, y1) * page.rotation_matrix, page)
        if norm is None:
            continue
        nx0, ny0, nx1, ny1 = norm
        out.append(
            BoxRecord(
                kind="word",
                x0=nx0,
                y0=ny0,
                x1=nx1,
                y1=ny1,
                text=word,
                block_no=block_no,
                line_no=line_no,
                word_no=word_no,
            )
        )
    return out


def _table_cell_boxes(page: fitz.Page) -> list[BoxRecord]:
    """Table cells give snap-to-box a candidate granularity words cannot express.

    find_tables is best-effort: a page with no ruled table yields nothing, and a
    malformed table must not abort the whole ingest.
    """
    out: list[BoxRecord] = []
    try:
        finder = page.find_tables()
        for table in finder.tables:
            for cell in table.cells:
                if cell is None:
                    continue
                # VERIFIED ON PyMuPDF 1.28: unlike get_text("words"),
                # find_tables already returns DISPLAYED-space coordinates. On a
                # /Rotate 90 page a cell comes back as (602, 72, 632, 192) while
                # the same page's words are still unrotated. Applying
                # rotation_matrix here would rotate them a second time.
                norm = _normalize(fitz.Rect(cell), page)
                if norm is None:
                    continue
                x0, y0, x1, y1 = norm
                out.append(BoxRecord(kind="table_cell", x0=x0, y0=y0, x1=x1, y1=y1, text=""))
    except Exception:
        # Broad on purpose: PyMuPDF's table API is undocumented enough that
        # narrowing would be guesswork. But log loudly — silently yielding zero
        # cells on every page would surface much later as degraded retrieval and
        # be misattributed to the model.
        logger.warning("find_tables failed on page %s", page.number, exc_info=True)
        return []
    return out


def extract_boxes(page: fitz.Page) -> list[BoxRecord]:
    """All candidate boxes for one page: words first, then table cells."""
    return _word_boxes(page) + _table_cell_boxes(page)
