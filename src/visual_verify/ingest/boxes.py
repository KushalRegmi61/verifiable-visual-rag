"""Candidate box extraction: the foundation everything downstream stands on.

Boxes are stored at WORD granularity with their block/line/word indices, plus
table cells. Coarser granularities (line, block, span) are derived by grouping
at query time, never stored. This is what lets S4 retune which candidate set
snap-to-box ranks without re-ingesting, and lets S7 union the words covering an
arbitrary answer substring to build a gold box.

COORDINATE HANDLING, THE THING TO GET RIGHT:
get_text("words") returns coordinates in UNROTATED page space, while page.rect
and get_pixmap() both use DISPLAYED space. On a /Rotate 90 page these differ.
Multiplying by page.rotation_matrix maps text space into displayed space; on an
unrotated page it is the identity, so it is always safe to apply.
"""

from dataclasses import dataclass
from typing import Literal

import fitz

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


def _normalize(rect: fitz.Rect, page: fitz.Page) -> tuple[float, float, float, float] | None:
    """Map a text-space rect into displayed space and normalize to 0-1.

    Returns None for degenerate boxes, which are dropped rather than stored.
    """
    r = rect * page.rotation_matrix
    r.normalize()  # rotation can invert x0/x1 or y0/y1

    w, h = page.rect.width, page.rect.height
    if w <= 0 or h <= 0:
        return None

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
        norm = _normalize(fitz.Rect(x0, y0, x1, y1), page)
        if norm is None:
            continue
        out.append(
            BoxRecord(
                kind="word",
                x0=norm[0],
                y0=norm[1],
                x1=norm[2],
                y1=norm[3],
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
    try:
        finder = page.find_tables()
    except Exception:
        return []

    out: list[BoxRecord] = []
    for table in finder.tables:
        for cell in table.cells:
            if cell is None:
                continue
            norm = _normalize(fitz.Rect(cell), page)
            if norm is None:
                continue
            out.append(
                BoxRecord(
                    kind="table_cell",
                    x0=norm[0],
                    y0=norm[1],
                    x1=norm[2],
                    y1=norm[3],
                    text="",
                )
            )
    return out


def extract_boxes(page: fitz.Page) -> list[BoxRecord]:
    """All candidate boxes for one page: words first, then table cells."""
    return _word_boxes(page) + _table_cell_boxes(page)
