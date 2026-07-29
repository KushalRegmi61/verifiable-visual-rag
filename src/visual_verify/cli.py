"""The `vvrag` command line.

Requires the `store` extra. This is the layer that wires the dependency-light
pipeline to SQLAlchemy persistence; the core itself never does.

argparse rather than click or typer, so the CLI adds no dependency at all.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from sqlalchemy import select
from sqlalchemy.orm import Session

from visual_verify.config import Settings
from visual_verify.ingest.gate import GateError
from visual_verify.ingest.pipeline import ingest_pdf
from visual_verify.store.engine import make_engine
from visual_verify.store.models import Box, Document, Page
from visual_verify.store.repository import SqlSink, document_status

BOX_COLORS = {"word": (0, 131, 215), "table_cell": (244, 88, 19)}


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

    ini = Path(__file__).resolve().parent.parent.parent / "alembic.ini"
    cfg = Config(str(ini))
    cfg.attributes["configure_logger"] = False
    cfg.set_main_option("script_location", str(ini.parent / "migrations"))
    command.upgrade(cfg, "head")


def _session(settings: Settings) -> Session:
    """A session against a schema known to be at head.

    make_engine, not create_engine: it enables PRAGMA foreign_keys=ON for
    SQLite, without which the CLI would silently lose the FK enforcement that
    every test and Postgres both have.
    """
    _ensure_schema(settings)
    return Session(make_engine(settings.db_url))


def _ingest_one(path: Path, sink: SqlSink, settings: Settings, dpi: int) -> bool:
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
            if _ingest_one(path, sink, settings, dpi):
                ok += 1
            session.commit()

    return 0 if ok == len(targets) else 1


def cmd_status(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    with _session(settings) as session:
        rows = document_status(session)

    if not rows:
        print("no documents ingested")
        return 0

    print(f"{'document':<40} {'pages':>10}  status")
    for r in rows:
        print(f"{Path(r.path).name:<40} {f'{r.pages_done}/{r.n_pages}':>10}  {r.status}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Draw stored boxes over the rendered page.

    The human check that extraction is correct. No numeric test replaces looking
    at whether the rectangles actually sit on the words.
    """
    settings = Settings.from_env()
    with _session(settings) as session:
        doc = session.scalar(
            select(Document).where(Document.path.contains(args.doc)).limit(1)
        ) or session.get(Document, args.doc)
        if doc is None:
            print(f"no document matching {args.doc!r}")
            return 1

        page = session.scalar(
            select(Page).where(Page.doc_sha == doc.sha256, Page.page_no == args.page)
        )
        if page is None:
            print(f"no page {args.page} in {Path(doc.path).name}")
            return 1

        boxes = list(session.scalars(select(Box).where(Box.page_id == page.id)))
        doc_name = Path(doc.path).name
        page_no = page.page_no
        image_path = settings.pages_dir / page.image_path

    print(f"{doc_name} page {page_no}: {len(boxes)} boxes")
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
    p_inspect.add_argument("doc", help="document sha256 or a substring of its path")
    p_inspect.add_argument("--page", type=int, default=0)
    p_inspect.add_argument("--overlay", help="write a PNG with boxes drawn on the page")
    p_inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest" and not args.pdf and not args.directory:
        print("give a PDF path or --dir")
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
