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
