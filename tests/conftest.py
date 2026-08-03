"""Synthetic PDF fixtures.

Built with PyMuPDF at test time so tests assert exact known coordinates
rather than tolerances, and so no binary files land in git.
"""

from pathlib import Path

import fitz
import pytest

# LOAD-BEARING. Later test files assert exact coordinates derived from these
# values (e.g. TEXT_ORIGIN[0] / PAGE_W). Changing the text, the origin, or the
# page size will change those expectations. Grep for consumers before editing.
# US Letter at 72 dpi. Word positions below are chosen against these dimensions.
PAGE_W, PAGE_H = 612.0, 792.0

# insert_text places the text BASELINE at this point.
TEXT_ORIGIN = (72.0, 100.0)
FIRST_LINE = "Revenue grew 42 percent"
SECOND_LINE = "Margins held steady"


def _two_line_doc() -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_text(TEXT_ORIGIN, FIRST_LINE, fontsize=12)
    page.insert_text((TEXT_ORIGIN[0], TEXT_ORIGIN[1] + 40), SECOND_LINE, fontsize=12)
    return doc


@pytest.fixture
def born_digital_pdf(tmp_path: Path) -> Path:
    """A normal two-line, single-page, unrotated PDF."""
    path = tmp_path / "born_digital.pdf"
    doc = _two_line_doc()
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def multipage_pdf(tmp_path: Path) -> Path:
    """Three pages, each with distinct text."""
    path = tmp_path / "multipage.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_text(TEXT_ORIGIN, f"Page {i} content here", fontsize=12)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def rotated_pdf(tmp_path: Path) -> Path:
    """Same content as born_digital, but with /Rotate 90.

    This is the fixture that catches the coordinate-space bug: text coords stay
    unrotated while page.rect and the pixmap both rotate.
    """
    path = tmp_path / "rotated.pdf"
    doc = _two_line_doc()
    doc[0].set_rotation(90)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    """An image-only PDF: no text layer at all."""
    path = tmp_path / "scanned.pdf"
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200))
    pix.clear_with(128)
    page.insert_image(fitz.Rect(50, 50, 250, 250), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def encrypted_pdf(tmp_path: Path) -> Path:
    """A password-protected PDF."""
    path = tmp_path / "encrypted.pdf"
    doc = _two_line_doc()
    doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()
    return path


@pytest.fixture
def corrupt_pdf(tmp_path: Path) -> Path:
    """Bytes that are not a PDF at all."""
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.7\nthis is not a real pdf\n")
    return path


@pytest.fixture(scope="module")
def real_pdf_pages():
    """Pages from the repo's own ingested corpus, if one exists.

    Skips rather than fails when the corpus is absent, so a fresh clone can run
    the suite without a two-hour ingest. Module-scoped: the only consumer is the
    known-item retrieval suite, which embeds these pages with a real GPU model at
    ~21s/page, so re-resolving the corpus per test (not per module) would not
    itself be expensive, but it must match the scope of anything built on top of
    it (see test_known_item_retrieval.py's module-scoped `corpus` fixture).
    """

    def _load(n: int):
        import sqlite3

        root = Path(__file__).resolve().parent.parent
        db = root / "data" / "index.db"
        pages_dir = root / "data" / "pages"
        if not db.exists():
            pytest.skip("no ingested corpus at data/index.db; run `vvrag ingest` first")
        con = sqlite3.connect(db)
        rows = con.execute(
            "SELECT doc_sha, page_no, image_path FROM pages ORDER BY doc_sha, page_no LIMIT ?",
            (n,),
        ).fetchall()
        con.close()
        if not rows:
            pytest.skip("corpus database has no pages")
        sha = rows[0][0]
        return sha, [(r[1], r[2]) for r in rows if r[0] == sha], pages_dir

    return _load


@pytest.fixture(scope="module")
def real_page_sentences():
    """(page_no, sentence) pairs taken verbatim from each page's text layer."""

    def _load(sha: str, page_nos: list[int]):
        import sqlite3

        db = Path(__file__).resolve().parent.parent / "data" / "index.db"
        con = sqlite3.connect(db)
        rows = con.execute(
            """
            SELECT p.page_no, group_concat(b.text, ' ') AS line
            FROM boxes b JOIN pages p ON b.page_id = p.id
            WHERE b.kind = 'word' AND p.doc_sha = ?
            GROUP BY p.id, b.block_no, b.line_no
            ORDER BY length(line) DESC
            """,
            (sha,),
        ).fetchall()
        con.close()
        wanted, seen, out = set(page_nos), set(), []
        for page_no, line in rows:
            words = line.split()
            if page_no in wanted and page_no not in seen and 8 <= len(words) <= 25:
                seen.add(page_no)
                out.append((page_no, " ".join(words)))
        return out

    return _load
