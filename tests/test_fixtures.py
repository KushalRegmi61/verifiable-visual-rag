import fitz

from conftest import PAGE_H, TEXT_ORIGIN


def test_born_digital_has_text(born_digital_pdf):
    doc = fitz.open(born_digital_pdf)
    assert len(doc[0].get_text("words")) > 0
    doc.close()


def test_scanned_has_no_text(scanned_pdf):
    doc = fitz.open(scanned_pdf)
    assert doc[0].get_text("words") == []
    doc.close()


def test_encrypted_needs_password(encrypted_pdf):
    doc = fitz.open(encrypted_pdf)
    assert doc.needs_pass
    doc.close()


def test_rotated_reports_rotation_90(rotated_pdf):
    doc = fitz.open(rotated_pdf)
    page = doc[0]
    assert page.rotation == 90
    # The trap this fixture exists to catch: rect is rotated, text coords are not.
    assert page.rect.width == PAGE_H
    assert page.get_text("words")[0][0] == TEXT_ORIGIN[0]
    doc.close()


def test_multipage_has_three_pages(multipage_pdf):
    doc = fitz.open(multipage_pdf)
    assert doc.page_count == 3
    doc.close()
