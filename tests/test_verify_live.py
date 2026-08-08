"""One end-to-end ask against the real corpus and the configured models.

Skips unless everything it needs is present, following test_grounding_live's
convention: VVRAG_QDRANT_URL for the index, an ingested PDF, a CUDA GPU for
the local verifier, and an API key for the hosted reader.

This is also where the compute-path measurement promised by spec 3.1 and 4
lands: run `vvrag ask` by hand and record peak VRAM, load time, and whether
the default pairing actually produces a grounded, verified answer.
"""

import pytest

pytestmark = pytest.mark.slow


def _settings_or_skip():
    import os

    if not os.environ.get("VVRAG_QDRANT_URL"):
        pytest.skip("VVRAG_QDRANT_URL not set")
    from visual_verify.config import Settings

    return Settings.from_env()


def _page_or_skip(settings, doc_needle: str, page_no: int):
    from pathlib import Path

    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from visual_verify.store.engine import make_engine
    from visual_verify.store.models import Box, Document, Page

    with Session(make_engine(settings.db_url)) as session:
        doc = session.scalars(select(Document)).first()
        if doc is None:
            pytest.skip("no documents ingested; run `vvrag ingest` first")
        page = session.scalar(
            select(Page).where(Page.doc_sha == doc.sha256, Page.page_no == page_no)
        )
        if page is None:
            pytest.skip(f"no page {page_no} in {Path(doc.path).name}")
        boxes = [
            b
            for b in session.scalars(
                select(Box).where(Box.page_id == page.id, Box.kind == "word")
            )
        ]
        image_path = settings.pages_dir / page.image_path
    return boxes, image_path


def test_ask_answers_a_text_question_end_to_end():
    """The text path needs no vectors: a claim taken verbatim from the page
    text layer must ground, be judged, and survive the gate with the models
    actually configured. The verifier's label is whatever the real model
    says; the pipeline shape is what is asserted here.
    """
    import os

    from PIL import Image

    from visual_verify.cli import _build_reader, _build_verifier
    from visual_verify.derive import line_boxes
    from visual_verify.verify import verify

    if not os.environ.get("VVRAG_READER_URL"):
        pytest.skip("VVRAG_READER_URL not set")

    settings = _settings_or_skip()
    boxes, image_path = _page_or_skip(settings, None, 0)
    if len(boxes) < 10:
        pytest.skip("page has too few word boxes to be meaningful")

    text_layer = "\n".join(b.text for b in line_boxes(boxes))
    first_line = text_layer.splitlines()[0]
    if not first_line or len(first_line.split()) < 3:
        pytest.skip("first line is too short to be a claim")

    question = f"what does this page say? ({first_line})"
    reader = _build_reader()
    verifier = _build_verifier()
    ans = verify(
        question,
        reader,
        verifier,
        page=0,
        image=Image.open(image_path).convert("RGB"),
        text_layer=text_layer,
        boxes=boxes,
    )
    assert ans.question == question
    # Whatever the models decide, the Answer must carry claims or an explicit
    # abstention - never a missing field and never a silent drop.
    assert len(ans.claims) > 0 or ans.abstained_overall
