"""Page rendering at a fixed DPI.

DPI is fixed per corpus so that ingest-time geometry and display-time geometry
agree. Boxes are stored normalized (see boxes.py), so DPI never leaks into any
consumer, but the patch-grid math in S4 still needs a consistent render.
"""

from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class RenderedPage:
    path: Path
    width_px: int
    height_px: int
    dpi: int
    rect_width: float
    rect_height: float


def render_page(page: fitz.Page, out_path: Path, dpi: int) -> RenderedPage:
    """Render one page to PNG.

    page.rect is the DISPLAYED rect, already accounting for /Rotate, and
    get_pixmap renders that same space. Recording rect dimensions here lets
    boxes.py normalize against exactly what was rendered.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix = page.get_pixmap(dpi=dpi)
    pix.save(out_path)
    return RenderedPage(
        path=out_path,
        width_px=pix.width,
        height_px=pix.height,
        dpi=dpi,
        rect_width=page.rect.width,
        rect_height=page.rect.height,
    )
