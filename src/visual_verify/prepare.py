"""Assemble everything one page's worth of grounding needs.

Lifted out of cli.cmd_ask so the API layer does not reimplement it. This is
the adapter: it talks to SQLAlchemy and Qdrant so that grounding and the agent
never have to, and everything it hands back is a plain array or a value object.
That is what keeps those packages inside the core's four dependencies.

Requires the `store` and `retrieval` extras, like the CLI. Nothing in the core
imports this module.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from visual_verify.config import Settings
from visual_verify.ingest.boxes import BoxRecord
from visual_verify.retrieval.geometry import PatchGrid
from visual_verify.store.models import Box, Document, Page


class PageNotFound(LookupError):
    """No single page matched. Also raised for an ambiguous document needle:
    picking the first of several matches is a wrong answer wearing the costume
    of a right one."""


@dataclass(frozen=True)
class PreparedPage:
    """One page, ready to ground against."""

    doc_sha: str
    doc_name: str
    page_no: int
    image_path: Path
    boxes: list[BoxRecord]
    # None when the page has not been embedded. ground() then has no visual
    # fallback and raises GroundingError for any claim it cannot find in the
    # text layer, which answer_stream turns into insufficient_evidence.
    page_vectors: np.ndarray | None
    grid: PatchGrid | None


def to_record(b: Box) -> BoxRecord:
    """Re-hydrate a stored Box row into the dataclass derive works over.

    Kept out of the store: derive is a pure core function and must not learn
    about ORM rows. It lives here rather than in the CLI because the CLI is no
    longer the only reader of boxes; the API service goes through prepare_page
    for the same rows.
    """
    return BoxRecord(
        kind=b.kind,
        x0=b.x0,
        y0=b.y0,
        x1=b.x1,
        y1=b.y1,
        text=b.text,
        block_no=b.block_no,
        line_no=b.line_no,
        word_no=b.word_no,
    )


def resolve_document(session: Session, needle: str) -> Document:
    """Exact sha256 first, then a unique path or sha prefix match."""
    exact = session.get(Document, needle)
    if exact is not None:
        return exact

    matches = list(
        session.scalars(
            select(Document)
            .where(Document.path.contains(needle) | Document.sha256.startswith(needle))
            .order_by(Document.path)
        )
    )
    if not matches:
        raise PageNotFound(f"no document matching {needle!r}")
    if len(matches) > 1:
        names = ", ".join(Path(m.path).name for m in matches)
        raise PageNotFound(f"{needle!r} matches more than one document: {names}")
    return matches[0]


def prepare_page(
    session: Session,
    index,
    settings: Settings,
    *,
    doc: str,
    page_no: int,
) -> PreparedPage:
    """Everything needed to ground a claim against one page.

    `index` is a QdrantIndex, untyped here so this module does not import
    qdrant_client at module scope for the sake of an annotation.

    Vectors are fetched unconditionally. The caller cannot know whether the
    text path will suffice, because the claims come from a reader that has not
    run yet, so a page that grounds entirely through the text layer simply
    never uses them. Two Qdrant calls, get_payload_or_none then get_vectors,
    against two model calls per claim.
    """
    document = resolve_document(session, doc)
    page = session.scalar(
        select(Page).where(Page.doc_sha == document.sha256, Page.page_no == page_no)
    )
    if page is None:
        raise PageNotFound(f"no page {page_no} in {Path(document.path).name}")

    boxes = [
        to_record(b)
        for b in session.scalars(select(Box).where(Box.page_id == page.id, Box.kind == "word"))
    ]

    # Deferred on purpose: visual_verify.retrieval.index drags in qdrant_client,
    # whereas retrieval.geometry (imported at module scope above) is pure
    # stdlib by design. Moving this up would put qdrant_client behind every
    # import of this module.
    from visual_verify.retrieval.index import ORIGINAL

    vectors: np.ndarray | None = None
    grid: PatchGrid | None = None
    # One point lookup, not `page_no in existing_page_nos(sha)`, which scrolls
    # every point of the document with payloads attached to answer a question
    # about a single page. This function runs on every request the service
    # serves.
    payload = index.get_payload_or_none(document.sha256, page_no)
    if payload is not None:
        vectors = index.get_vectors(document.sha256, page_no)[ORIGINAL]
        grid = PatchGrid(
            n_x=payload["n_patches_x"],
            n_y=payload["n_patches_y"],
            offset=payload["patch_offset"],
            n_vectors=vectors.shape[0],
        )

    return PreparedPage(
        doc_sha=document.sha256,
        doc_name=Path(document.path).name,
        page_no=page_no,
        image_path=settings.pages_dir / page.image_path,
        boxes=boxes,
        page_vectors=vectors,
        grid=grid,
    )


def prepare_pages(
    session: Session,
    index,
    settings: Settings,
    hits,
    limit: int = 3,
) -> list[PreparedPage]:
    """The top `limit` retrieved pages that belong to the TOP HIT's document.

    `hits` is whatever QdrantIndex.search returned: anything with `doc_id` and
    `page`. Retrieval is corpus-wide and takes no document filter, so the raw
    top 3 routinely spans documents, and pages from anything but the top hit's
    document are dropped rather than read.

    Two reasons, both load-bearing. A GroundedRegion carries `page` and no
    document identity, and the frontend builds its page image URL from the
    answer's doc_sha plus that page, so a region on another document's page
    would be drawn on the wrong image with nothing saying so. And merging two
    documents into one answer without disclosing it contradicts the whole point
    of a system that shows where each sentence came from. Cross-document is a
    later slice and needs GroundedRegion.doc_sha first.

    Retrieval ORDER is preserved, never page order. Grounding breaks ties
    between pages by rank, so sorting or de-duplicating through a set would
    silently change which page a claim gets cited to, and the result would
    still look like a perfectly ordinary answer.
    """
    if not hits:
        return []

    top_doc = hits[0].doc_id
    same_document = [h for h in hits if h.doc_id == top_doc][:limit]
    return [
        prepare_page(session, index, settings, doc=top_doc, page_no=h.page) for h in same_document
    ]
