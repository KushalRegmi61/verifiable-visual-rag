"""Smoke test against a genuine PDF from this repository.

The synthetic fixtures give exact expected coordinates; this proves the pipeline
survives a real document with real fonts, figures, and a bibliography.
"""

from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from visual_verify.ingest.pipeline import ingest_pdf
from visual_verify.store.engine import make_engine
from visual_verify.store.models import Base, Box, Page
from visual_verify.store.repository import SqlSink

REAL_PDF = Path(__file__).parent.parent / "proposal_report" / "proposal.pdf"


@pytest.mark.skipif(not REAL_PDF.exists(), reason="proposal.pdf not present")
def test_ingests_the_project_proposal(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'smoke.db'}")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        result = ingest_pdf(REAL_PDF, SqlSink(session), pages_dir=tmp_path, dpi=150)
        session.commit()

        assert result.pages_written > 5
        assert session.scalar(select(func.count()).select_from(Page)) == result.pages_written

        n_boxes = session.scalar(select(func.count()).select_from(Box))
        assert n_boxes > 500, "a real proposal should yield plenty of word boxes"

        # Every stored box must satisfy the normalization invariant.
        bad = session.scalar(
            select(func.count())
            .select_from(Box)
            .where(
                (Box.x0 < 0)
                | (Box.x1 > 1)
                | (Box.y0 < 0)
                | (Box.y1 > 1)
                | (Box.x1 <= Box.x0)
                | (Box.y1 <= Box.y0)
            )
        )
        assert bad == 0


@pytest.mark.skipif(not REAL_PDF.exists(), reason="proposal.pdf not present")
def test_real_pdf_boxes_land_on_ink(tmp_path):
    """Ground truth on a real document, not a synthetic fixture.

    Arithmetic about coordinate systems is easy to get confidently wrong; ink is
    the only thing that actually proves a box points at its evidence.
    """
    from PIL import Image

    engine = make_engine(f"sqlite:///{tmp_path / 'ink.db'}")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        ingest_pdf(REAL_PDF, SqlSink(session), pages_dir=tmp_path, dpi=150)
        session.commit()

        page = session.scalar(select(Page).where(Page.page_no == 3))
        boxes = list(
            session.scalars(select(Box).where(Box.page_id == page.id, Box.kind == "word").limit(60))
        )
        image_path = tmp_path / page.image_path
        width_px, height_px = page.width_px, page.height_px

    img = Image.open(image_path).convert("L")
    on_ink = 0
    for b in boxes:
        crop = img.crop(
            (
                int(b.x0 * width_px),
                int(b.y0 * height_px),
                int(b.x1 * width_px),
                int(b.y1 * height_px),
            )
        )
        # getextrema()[0] is the darkest pixel; same test as min(getdata()) but
        # without Pillow's getdata deprecation warning.
        if crop.getbbox() is not None and crop.getextrema()[0] < 128:
            on_ink += 1

    assert len(boxes) > 20, "expected a text-bearing page"
    assert on_ink == len(boxes), f"only {on_ink}/{len(boxes)} boxes contained ink"
