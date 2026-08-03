"""Embedding orchestration: store rows in, Qdrant points out.

Takes page rows as data rather than a Session, mirroring S2's rule that the
pipeline never holds a database handle. That is what lets this be tested against
a fake embedder and an in-memory Qdrant with no database at all.
"""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from visual_verify.retrieval.index import QdrantIndex
from visual_verify.retrieval.types import Embedder


@dataclass(frozen=True)
class EmbedResult:
    sha256: str
    embedded: int
    skipped: int


def embed_document(
    doc_sha: str,
    pages: list[tuple[int, str]],
    pages_dir: Path,
    embedder: Embedder,
    index: QdrantIndex,
    max_pages: int | None = None,
) -> EmbedResult:
    """Embed one document's pages into the index.

    `pages` is (page_no, image_path) in the store's own relative form.
    `max_pages` exists to test resumption by simulating a partial run.
    Does not call `index.ensure_collection()`; that is the caller's job, same
    as `ingest_pdf` never creates tables.
    """
    already = index.existing_page_nos(doc_sha)
    embedded = skipped = 0

    for page_no, rel_path in sorted(pages):
        if page_no in already:
            skipped += 1
            continue
        if max_pages is not None and embedded >= max_pages:
            break

        path = Path(pages_dir) / rel_path
        if not path.is_file():
            raise FileNotFoundError(path)

        with Image.open(path) as im:
            size = im.size

        embedding = embedder.embed_page(str(path), size)
        # Upserted with wait=True one page at a time. At 21.4 s of embedding per
        # page the round trip is noise, and it is what makes a 1.8-hour run
        # resumable rather than all-or-nothing.
        index.upsert_page(doc_sha, page_no, rel_path, embedding, embedder.provenance)
        embedded += 1

    return EmbedResult(sha256=doc_sha, embedded=embedded, skipped=skipped)
