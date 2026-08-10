"""The `vvrag` command line.

Requires the `store` extra. This is the layer that wires the dependency-light
pipeline to SQLAlchemy persistence; the core itself never does.

argparse rather than click or typer, so the CLI adds no dependency at all.
"""

import argparse
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from sqlalchemy import select
from sqlalchemy.orm import Session

from visual_verify import derive
from visual_verify.config import Settings
from visual_verify.contracts import Answer, Claim
from visual_verify.ingest.gate import GateError
from visual_verify.ingest.pipeline import ingest_pdf
from visual_verify.prepare import to_record
from visual_verify.store.engine import make_engine
from visual_verify.store.models import Box, Document, Page
from visual_verify.store.repository import SqlSink, document_status

BOX_COLORS = {
    "word": (0, 131, 215),
    "table_cell": (244, 88, 19),
    "line": (120, 94, 240),
    "block": (100, 100, 100),
    # Deliberately the loudest colour in the set: a --find overlay is the
    # project's central claim rendered as a picture, and it should read as
    # different in kind from a routine box dump.
    "span": (26, 176, 80),
}


def _ensure_schema(settings: Settings) -> None:
    """Bring the database up to head.

    Deliberately NOT Base.metadata.create_all(): that would create the tables
    without stamping alembic_version, and the next `alembic upgrade head` would
    fail with "table already exists". Migrations are the single source of truth.
    """
    import logging

    from alembic import command
    from alembic.config import Config

    # _ensure_schema runs on every invocation, including read-only commands like
    # `vvrag status`. Alembic's INFO logging would then print two lines of
    # migration-runtime noise before any command's real output. The attribute
    # stops migrations/env.py from reapplying alembic.ini's INFO level on top.
    logging.getLogger("alembic").setLevel(logging.WARNING)

    # Resolved from the package directory, not by counting .parent calls up into
    # a source tree. alembic.ini and migrations/ ship inside the package, so this
    # is the same path in a checkout and in an installed wheel; the old
    # parent.parent.parent landed on site-packages/ and every subcommand died
    # with "Path doesn't exist: .../migrations".
    ini = Path(__file__).resolve().parent / "alembic.ini"
    cfg = Config(str(ini))
    cfg.attributes["configure_logger"] = False
    cfg.set_main_option("script_location", str(ini.parent / "migrations"))
    # env.py also derives the URL from Settings.from_env(), but relying on that
    # made settings a silently unused argument here. ConfigParser reads "%" as
    # interpolation syntax, and a managed-Postgres password routinely contains
    # one, so it must be doubled.
    cfg.set_main_option("sqlalchemy.url", settings.db_url.replace("%", "%%"))
    command.upgrade(cfg, "head")


def _session(settings: Settings) -> Session:
    """A session against a schema known to be at head.

    make_engine, not create_engine: it enables PRAGMA foreign_keys=ON for
    SQLite, without which the CLI would silently lose the FK enforcement that
    every test and Postgres both have.
    """
    _ensure_schema(settings)
    return Session(make_engine(settings.db_url))


def _ingest_one(path: Path, sink: SqlSink, session: Session, settings: Settings, dpi: int) -> bool:
    try:
        result = ingest_pdf(
            path,
            sink,
            pages_dir=settings.pages_dir,
            dpi=dpi,
            min_text_page_ratio=settings.min_text_page_ratio,
        )
    except FileNotFoundError:
        print(f"  {path.name}: no such file ({path})")
        return False
    except GateError as exc:
        print(f"  {path.name}: rejected ({exc})")
        return False
    except Exception as exc:  # noqa: BLE001 - one bad file must not abort a batch
        # A PermissionError, a full disk, or a PyMuPDF failure on file 3 of 50
        # must not cost the remaining 47. The pipeline checkpoints per page, so
        # this rollback discards only the incomplete tail of this one document.
        session.rollback()
        print(f"  {path.name}: failed ({type(exc).__name__}: {exc})")
        return False
    print(
        f"  {path.name}: {result.pages_written} pages written, "
        f"{result.pages_skipped} skipped ({result.n_pages} pages total)"
    )
    return True


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    dpi = args.dpi or settings.render_dpi

    if args.directory:
        targets = sorted(Path(args.directory).glob("*.pdf"))
        if not targets:
            print(f"no PDFs found in {args.directory}")
            return 1
    else:
        targets = [Path(args.pdf)]

    ok = 0
    with _session(settings) as session:
        sink = SqlSink(session)
        for path in targets:
            if _ingest_one(path, sink, session, settings, dpi):
                ok += 1
            # Kept even though the pipeline checkpoints per page: the final
            # finish_document only checkpoints when every page was accounted
            # for, so a --max-pages-style partial run would otherwise leave the
            # last status update sitting uncommitted in the session.
            session.commit()

    return 0 if ok == len(targets) else 1


def cmd_status(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    with _session(settings) as session:
        rows = document_status(session)

    if not rows:
        print("no documents ingested")
        return 0

    # Widen the name column to the longest name in this result set rather than
    # a fixed :<40. Real corpus filenames (ENCT354MINORPROJECT_...pdf is 43
    # chars) overflowed the fixed width and shifted every later column right.
    names = [Path(r.path).name for r in rows]
    width = max(len("document"), *(len(n) for n in names))

    print(f"{'document':<{width}} {'pages':>10}  status")
    for name, r in zip(names, rows, strict=True):
        print(f"{name:<{width}} {f'{r.pages_done}/{r.n_pages}':>10}  {r.status}")
    return 0


def _resolve_document(session: Session, needle: str) -> Document | None | list[Document]:
    """Find the one document a user meant.

    Resolution order: exact sha256, then every document whose path contains the
    needle or whose sha256 starts with it. Returns the document, None when
    nothing matched, or the candidate list when more than one matched. The old
    code took `.limit(1)`, so `inspect proposal` silently picked whichever of
    proposal.pdf / reference_proposal.pdf was inserted first.

    A raising twin, prepare.resolve_document, does the same query but turns
    both "nothing matched" and "several matched" into PageNotFound. cmd_ask
    is on that one, through prepare_page; cmd_inspect, cmd_embed and cmd_ground
    are still on this one. They have not been merged because the two contracts
    differ in what the caller must print: these three enumerate the ambiguous
    candidates with their sha prefixes so the user can pick one, which the
    exception's single message cannot carry. Collapsing them is a real
    behaviour change and belongs in its own commit.
    """
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
        return None
    if len(matches) == 1:
        return matches[0]
    return matches


def cmd_inspect(args: argparse.Namespace) -> int:
    """Draw stored boxes over the rendered page.

    The human check that extraction is correct. No numeric test replaces looking
    at whether the rectangles actually sit on the words.
    """
    settings = Settings.from_env()
    with _session(settings) as session:
        found = _resolve_document(session, args.doc)
        if found is None:
            print(f"no document matching {args.doc!r}")
            return 1
        if isinstance(found, list):
            print(f"ambiguous: {args.doc!r} matches {len(found)} documents")
            width = max(len(Path(d.path).name) for d in found)
            for d in found:
                print(f"  {Path(d.path).name:<{width}}  ({d.sha256[:12]})")
            print("use a longer substring or a sha256 prefix")
            return 1
        doc = found

        page = session.scalar(
            select(Page).where(Page.doc_sha == doc.sha256, Page.page_no == args.page)
        )
        if page is None:
            print(f"no page {args.page} in {Path(doc.path).name}")
            return 1

        stored = [to_record(b) for b in session.scalars(select(Box).where(Box.page_id == page.id))]
        doc_name = Path(doc.path).name
        page_no = page.page_no
        image_path = settings.pages_dir / page.image_path

    print(f"{doc_name} page {page_no}: {len(stored)} boxes")

    if args.find:
        # The live demonstration of the project's claim: name a phrase, get the
        # rectangles that cover it. span_boxes splits at line breaks rather than
        # returning one union, so a wrapped phrase does not sweep in the words
        # between its two halves.
        boxes = derive.span_boxes(stored, args.find) if stored else []
        if not boxes:
            print(f"phrase {args.find!r} not found on this page")
            return 0
        print(f"{len(boxes)} rect(s) match {args.find!r}")
    elif args.kind == "word":
        boxes = stored
    else:
        derived = derive.line_boxes if args.kind == "line" else derive.block_boxes
        boxes = derived(stored) if stored else []
        print(f"{len(boxes)} {args.kind} boxes derived from {len(stored)} stored boxes")

    if not args.overlay:
        return 0

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for b in boxes:
        draw.rectangle(
            [b.x0 * img.width, b.y0 * img.height, b.x1 * img.width, b.y1 * img.height],
            outline=BOX_COLORS.get(b.kind, (226, 10, 22)),
            width=2,
        )
    Path(args.overlay).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.overlay)
    print(f"wrote overlay to {args.overlay}")
    return 0


# A real Qdrant `:memory:` client has no server behind it: each QdrantClient(
# ":memory:") call constructs an independent, empty backend, so a fresh
# QdrantIndex for every CLI invocation would make `vvrag embed` and a later
# `vvrag search` never see each other's data. Cache one QdrantIndex per
# (qdrant_url, db_url) pair instead of per (qdrant_url, collection): the url
# string ":memory:" alone is identical across every test, but db_url is
# derived from each test's own tmp_path, so it doubles as the isolation key
# tests already have for free, with no extra reset hook needed. Real Postgres
# and Qdrant Cloud never hit this branch at all. Not thread-safe, but the CLI
# is invoked once per process, never concurrently.
_MEMORY_INDEX_CACHE: dict[tuple[str, str], "object"] = {}


def _make_embedder(settings: Settings):
    """The fake keeps CLI tests off the GPU; anything else loads the real model."""
    if os.getenv("VVRAG_FAKE_EMBEDDER"):
        from visual_verify.retrieval.types import FakeEmbedder

        return FakeEmbedder()
    from visual_verify.retrieval.embedder import ColQwen2Embedder

    return ColQwen2Embedder(render_dpi=settings.render_dpi)


def _make_index(settings: Settings):
    from visual_verify.retrieval.index import QdrantIndex

    if not settings.qdrant_url:
        raise SystemExit("VVRAG_QDRANT_URL is not set")

    if settings.qdrant_url == ":memory:":
        key = (settings.qdrant_url, settings.db_url)
        index = _MEMORY_INDEX_CACHE.get(key)
        if index is None:
            index = QdrantIndex(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            _MEMORY_INDEX_CACHE[key] = index
    else:
        index = QdrantIndex(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    index.ensure_collection()
    return index


def cmd_embed(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    index = _make_index(settings)
    embedder = _make_embedder(settings)

    from visual_verify.retrieval.pipeline import embed_document

    with _session(settings) as session:
        if args.all:
            shas = list(
                session.scalars(select(Document.sha256).where(Document.status == "indexed"))
            )
        else:
            found = _resolve_document(session, args.doc)
            if found is None:
                print(f"no document matching {args.doc!r}")
                return 1
            if isinstance(found, list):
                print(f"ambiguous: {args.doc!r} matches {len(found)} documents")
                width = max(len(Path(d.path).name) for d in found)
                for d in found:
                    print(f"  {Path(d.path).name:<{width}}  ({d.sha256[:12]})")
                print("use a longer substring or a sha256 prefix")
                return 1
            shas = [found.sha256]

        if not shas:
            print("no indexed documents to embed; run `vvrag ingest` first")
            return 1

        total_embedded = total_skipped = 0
        for sha in shas:
            rows = [
                (p.page_no, p.image_path)
                for p in session.scalars(select(Page).where(Page.doc_sha == sha))
            ]
            result = embed_document(sha, rows, settings.pages_dir, embedder, index)
            total_embedded += result.embedded
            total_skipped += result.skipped
            print(f"{sha[:12]}  embedded {result.embedded}  skipped {result.skipped}")

    print(f"total: embedded {total_embedded}, skipped {total_skipped}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    index = _make_index(settings)
    if index.count() == 0:
        print("no pages indexed; run `vvrag embed --all` first")
        return 1

    embedder = _make_embedder(settings)
    hits = index.search(embedder.embed_query(args.query), embedder.provenance, limit=args.k)
    for rank, hit in enumerate(hits, 1):
        print(
            f"{rank}. {hit.doc_id[:12]}  page {hit.page:>4}  "
            f"score {hit.score:7.3f}  {hit.image_ref}"
        )
    return 0


def cmd_ground(args: argparse.Namespace) -> int:
    """Ground a claim to a region of one page.

    This is the adapter: it fetches vectors and geometry so that the grounding
    package never has to. Everything it hands over is a plain array or a value
    object, which is what keeps grounding inside the core's four dependencies.
    """
    from visual_verify.grounding import ground
    from visual_verify.retrieval.geometry import PatchGrid
    from visual_verify.retrieval.index import ORIGINAL

    settings = Settings.from_env()
    with _session(settings) as session:
        found = _resolve_document(session, args.doc)
        if found is None or isinstance(found, list):
            print(f"no unique document matching {args.doc!r}")
            return 1
        doc = found
        page = session.scalar(
            select(Page).where(Page.doc_sha == doc.sha256, Page.page_no == args.page)
        )
        if page is None:
            print(f"no page {args.page} in {Path(doc.path).name}")
            return 1
        boxes = [
            to_record(b)
            for b in session.scalars(select(Box).where(Box.page_id == page.id, Box.kind == "word"))
        ]
        image_path = settings.pages_dir / page.image_path

    page_vectors = query_vectors = grid = None
    # Only pay for the model and the fetch when the visual path can be reached.
    if args.force_visual or not derive.span_boxes(boxes, args.claim):
        index = _make_index(settings)
        if index.count() == 0:
            print("no pages indexed; run `vvrag embed` first")
            return 1
        payload = index.get_payload(doc.sha256, args.page)
        stored = index.get_vectors(doc.sha256, args.page)[ORIGINAL]
        grid = PatchGrid(
            n_x=payload["n_patches_x"],
            n_y=payload["n_patches_y"],
            offset=payload["patch_offset"],
            n_vectors=stored.shape[0],
        )
        page_vectors = stored
        query_vectors = _make_embedder(settings).embed_query(args.claim)

    # No try/except GroundingError here: vectors are None only when
    # span_boxes(boxes, args.claim) was truthy and force_visual is not set, in
    # which case ground() finds the same text match first (force != "visual")
    # and returns before ever reaching the code path that raises. There is no
    # input that reaches this call with vectors unset and force="visual".
    regions = ground(
        args.claim,
        boxes,
        page=args.page,
        page_vectors=page_vectors,
        query_vectors=query_vectors,
        grid=grid,
        force="visual" if args.force_visual else None,
    )

    if not regions:
        print("no evidence for this claim on this page")
        return 0

    for r in regions:
        x0, y0, x1, y1 = r.bbox
        marker = f" [{r.resolution}]" if r.resolution else ""
        print(
            f"{r.modality:<6}{marker} score {r.score:7.3f}  "
            f"[{x0:.3f} {y0:.3f} {x1:.3f} {y1:.3f}]  {(r.text or '')[:60]}"
        )

    if args.overlay:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for r in regions:
            x0, y0, x1, y1 = r.bbox
            draw.rectangle(
                [x0 * img.width, y0 * img.height, x1 * img.width, y1 * img.height],
                outline=(226, 10, 22) if r.modality == "visual" else (16, 128, 64),
                width=3,
            )
        img.save(args.overlay)
        print(f"wrote {args.overlay}")
    return 0


def _print_claim(c: Claim, indent: str) -> None:
    # label is None until the verifier runs; a claim that never reached it
    # (now correctly included below, see the withheld comment above) would
    # otherwise crash `:<22` formatting on NoneType instead of printing.
    label = c.label if c.label is not None else "unverified"
    flag = " [compound]" if c.compound else ""
    print(f"{indent}{label:<22} {c.confidence:.2f}  {c.text}{flag}")
    for r in c.regions:
        x0, y0, x1, y1 = r.bbox
        box = f"[{x0:.3f} {y0:.3f} {x1:.3f} {y1:.3f}]"
        print(f"{indent}                      {r.modality:<6} {box}")


def _print_ask_result(result: Answer, threshold: float) -> None:
    """Print an Answer as two labelled sections: shown, then withheld.

    This is a diagnostic view for inspecting the verifier's behaviour from a
    terminal, not the product answer surface: S6's UI must read
    `Answer.shown` and never touch `result.claims` directly, because iterating
    `claims` puts a claim the verifier refused in front of a user. Here the
    opposite is deliberate: a CLI whose entire purpose is showing how
    abstention works would be useless if it hid abstained claims, because
    there would be no way to distinguish "the reader never proposed this" from
    "the reader proposed it and the verifier said no". The withheld claims are
    still printed, just under a heading that says outright they are not part
    of the answer, because "withheld" names a choice the system made, not data
    that went missing.

    `threshold` is printed so a saved transcript states the bar claims were
    judged against. Without it, a run at --threshold 0 looks structurally
    identical to a fully verified run: unsupported claims sit under the same
    "Answer" heading with no marker of how permissive the gate was.
    """
    shown = result.shown
    # withheld must be the complement of shown, both read from Claim.withheld.
    # Filtering on `c.abstained` instead is a second, narrower predicate: a
    # claim that never reached the verifier (label=None, abstained=False) is
    # not `abstained` but is `withheld`, so it fell out of BOTH lists and
    # vanished from the transcript instead of appearing under "Withheld", which
    # is exactly the outcome this docstring says the CLI exists to avoid.
    withheld = [c for c in result.claims if c.withheld]

    print(f"threshold: {threshold}")
    print(f"Answer ({len(shown)} claim(s) shown):")
    for c in shown:
        _print_claim(c, indent="  ")

    if withheld:
        print(f"\nWithheld ({len(withheld)} claim(s), not part of the answer):")
        for c in withheld:
            _print_claim(c, indent="  ")

    if result.abstained_overall:
        print("\nabstained: the question was not answered with verified evidence")


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer a question from one page, with every claim verified before it shows.

    Page assembly (document resolution, word boxes, stored vectors, patch
    grid) is delegated to prepare.prepare_page, which the API service also
    calls: a PatchGrid that disagrees with the vectors it describes places
    boxes off-page while every shape and dtype still looks right, so it is
    built in exactly one place.

    The embedder is constructed here, up front, rather than inside answer().
    cmd_ground knows its single claim from a CLI argument and can embed it
    once; here the reader produces an unknown number of claims at runtime, so
    answer() takes a bound `embed_query` callable and embeds each claim itself.
    Building the embedder once per command rather than once per claim is what
    keeps the model weights loading a single time.
    """
    import math

    from visual_verify.agent import AgentError, answer
    from visual_verify.agent.cache import CachedChat
    from visual_verify.agent.models import MissingApiKey, UnknownProvider, make_chat
    from visual_verify.prepare import PageNotFound, prepare_page

    if not math.isfinite(args.threshold):
        print(f"--threshold must be a finite number, got {args.threshold}")
        return 1

    settings = Settings.from_env()
    index = _make_index(settings)
    if index.count() == 0:
        print("no pages indexed; run `vvrag embed` first")
        return 1

    with _session(settings) as session:
        try:
            prepared = prepare_page(session, index, settings, doc=args.doc, page_no=args.page)
        except PageNotFound as exc:
            print(str(exc))
            return 1

    # Printed BEFORE the reader runs, so the user can Ctrl-C before paying for
    # any model call. prepare_page returns page_vectors=None for a page that
    # was ingested but never embedded, which the API layer wants (serve a
    # text-only page rather than a 500) but which degrades silently here:
    # ground() has no visual fallback, a reader paraphrases by default, and so
    # most claims come back insufficient_evidence after a reader call and one
    # verifier call each. The old unhandled IndexError out of get_payload was
    # ugly but loud; a quiet wrong-looking success is the worse trade in a
    # system whose whole claim is that it says when it does not know.
    if prepared.page_vectors is None:
        print(
            f"warning: page {prepared.page_no} of {prepared.doc_name} is not embedded.\n"
            "  Grounding will use the text layer only, so any claim the reader\n"
            "  paraphrases rather than quotes verbatim will come back as\n"
            "  insufficient_evidence. Run `vvrag embed` on this document to fix it."
        )

    # No embedder at all in that branch: _make_embedder loads a 2.6 GB model and
    # answer_stream would call embed_query once per claim to build vectors that
    # ground() is then structurally guaranteed to discard. embed_query is
    # Callable | None in answer_stream and is only called when not None.
    embedder = _make_embedder(settings) if prepared.page_vectors is not None else None

    try:
        reader = CachedChat(make_chat("reader", settings), settings.agent_cache_dir)
        verifier = CachedChat(make_chat("verifier", settings), settings.agent_cache_dir)
    except (MissingApiKey, UnknownProvider) as exc:
        print(f"cannot build the models: {exc}")
        return 1

    try:
        result = answer(
            args.question,
            prepared.image_path,
            prepared.boxes,
            page=prepared.page_no,
            reader_chat=reader,
            verifier_chat=verifier,
            threshold=args.threshold,
            page_vectors=prepared.page_vectors,
            embed_query=embedder.embed_query if embedder is not None else None,
            grid=prepared.grid,
        )
    except AgentError as exc:
        print(f"cannot answer: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - a corrupt cache entry or a
        # provider/network error must print a sentence, not a raw traceback;
        # AgentError above already covers misconfiguration, this covers
        # everything else that can surface from the client and cache layers.
        print(f"cannot answer: {type(exc).__name__}: {exc}")
        return 1

    _print_ask_result(result, args.threshold)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vvrag", description="Verifiable Visual RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest a PDF or a directory of PDFs")
    p_ingest.add_argument("pdf", nargs="?", help="path to a PDF")
    p_ingest.add_argument("--dir", dest="directory", help="ingest every PDF in this directory")
    p_ingest.add_argument("--dpi", type=int, default=None, help="render DPI")
    p_ingest.set_defaults(func=cmd_ingest)

    p_status = sub.add_parser("status", help="show ingest status per document")
    p_status.set_defaults(func=cmd_status)

    p_inspect = sub.add_parser("inspect", help="inspect a page and optionally draw its boxes")
    p_inspect.add_argument(
        "doc", help="document sha256 (or a prefix of it) or a substring of its path"
    )
    p_inspect.add_argument("--page", type=int, default=0)
    p_inspect.add_argument(
        "--kind",
        choices=["word", "line", "block"],
        default="word",
        help="granularity to overlay; line and block are derived from word boxes",
    )
    p_inspect.add_argument(
        "--find",
        help="draw only the rects covering this phrase (overrides --kind)",
    )
    p_inspect.add_argument("--overlay", help="write a PNG with boxes drawn on the page")
    p_inspect.set_defaults(func=cmd_inspect)

    p_embed = sub.add_parser("embed", help="embed ingested pages into the vector index")
    p_embed.add_argument("doc", nargs="?", help="document sha256, prefix, or path substring")
    p_embed.add_argument("--all", action="store_true", help="embed every indexed document")
    p_embed.set_defaults(func=cmd_embed)

    p_search = sub.add_parser("search", help="rank pages against a question")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5, help="how many pages to return")
    p_search.set_defaults(func=cmd_search)

    p_ground = sub.add_parser("ground", help="ground a claim to a region of a page")
    p_ground.add_argument("claim", help="the claim to find evidence for")
    p_ground.add_argument("--doc", required=True, help="document sha256, prefix, or path substring")
    p_ground.add_argument("--page", type=int, required=True)
    p_ground.add_argument(
        "--force-visual",
        action="store_true",
        help="use snap-to-box even when the claim is in the text layer (what the eval does)",
    )
    p_ground.add_argument("--overlay", help="write a PNG with the region drawn on the page")
    p_ground.set_defaults(func=cmd_ground)

    p_ask = sub.add_parser("ask", help="answer a question from a page, with verification")
    p_ask.add_argument("question")
    p_ask.add_argument("--doc", required=True, help="document sha256, prefix, or path substring")
    p_ask.add_argument("--page", type=int, required=True)
    p_ask.add_argument(
        "--threshold",
        type=float,
        # Settings.abstain_threshold, not a literal: VVRAG_ABSTAIN_THRESHOLD
        # must actually change what an unflagged `vvrag ask` uses. Read once
        # per parser build, which is once per CLI invocation.
        default=Settings.from_env().abstain_threshold,
        help="abstain below this score; the rubric's supported floor by default",
    )
    p_ask.set_defaults(func=cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest" and not args.pdf and not args.directory:
        print("give a PDF path or --dir")
        return 1
    if args.command == "embed" and not args.doc and not args.all:
        print("give a document or --all")
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
