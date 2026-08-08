"""Smoke test against a genuine PDF from this repository.

The synthetic fixtures give exact expected coordinates; this proves the pipeline
survives a real document with real fonts, figures, and a bibliography.
"""

from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from visual_verify.evidence import covers_text, has_ink, shift
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

    img = Image.open(image_path)
    bboxes = [(b.x0, b.y0, b.x1, b.y1) for b in boxes]
    on_ink = sum(has_ink(img, bb) for bb in bboxes)

    assert len(boxes) > 20, "expected a text-bearing page"

    # Control: the same boxes displaced must NOT land on ink. Without this,
    # "every box contains ink" could hold trivially on a dense page. Measured on
    # proposal.pdf page 3: true 60/60, shifted 25% in x 16/60, in y 13/60.
    displaced = sum(has_ink(img, shift(bb, dx=0.25)) for bb in bboxes)

    assert on_ink == len(boxes), f"only {on_ink}/{len(boxes)} boxes contained ink"
    assert displaced < len(boxes) // 2, (
        f"{displaced}/{len(boxes)} displaced boxes also hit ink; "
        "the alignment assertion is not discriminating on this page"
    )


@pytest.mark.skipif(not REAL_PDF.exists(), reason="proposal.pdf not present")
def test_a_region_reads_back_its_own_line_and_not_a_neighbour(tmp_path):
    """The check S4 needs, on a real page: identity of the region, not ink.

    Ink presence cannot fail here. Every one of this page's 435 word boxes
    contains ink, so a selector returning a random candidate passes the test
    above every time while being right once in 435. Reading the covered text
    back is what separates the right region from a plausible one.
    """
    engine = make_engine(f"sqlite:///{tmp_path / 'text.db'}")
    Base.metadata.create_all(engine)

    # The ORM Box already carries x0/y0/x1/y1/text, which is the whole surface
    # text_in_bbox duck-types against, so it is passed straight through. The
    # assertions stay inside the session because those attributes expire on exit.
    with Session(engine) as session:
        ingest_pdf(REAL_PDF, SqlSink(session), pages_dir=tmp_path, dpi=150)
        session.commit()

        page = session.scalar(select(Page).where(Page.page_no == 3))
        boxes = list(session.scalars(select(Box).where(Box.page_id == page.id, Box.kind == "word")))

        # Group into real lines by shared block and line index, which ingest
        # stores precisely so downstream granularity can be retuned.
        rows: dict[tuple[int, int], list] = {}
        for b in boxes:
            rows.setdefault((b.block_no, b.line_no), []).append(b)
        lines = sorted(rows.values(), key=len, reverse=True)
        line, other = lines[0], lines[1]

        def region_of(row):
            return (
                min(b.x0 for b in row),
                min(b.y0 for b in row),
                max(b.x1 for b in row),
                max(b.y1 for b in row),
            )

        def text_of(row):
            return " ".join(b.text for b in sorted(row, key=lambda b: b.x0))

        assert covers_text(boxes, region_of(line), text_of(line))

        # A different line's text must NOT be readable from this region. This is
        # the assertion that has no ink-based equivalent.
        assert not covers_text(boxes, region_of(line), text_of(other))
