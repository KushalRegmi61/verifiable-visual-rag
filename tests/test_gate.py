import pytest

from visual_verify.ingest.gate import (
    GateError,
    RejectReason,
    fingerprint,
    open_and_check,
)


def test_fingerprint_is_stable(born_digital_pdf):
    assert fingerprint(born_digital_pdf) == fingerprint(born_digital_pdf)
    assert len(fingerprint(born_digital_pdf)) == 64


def test_fingerprint_differs_by_content(born_digital_pdf, multipage_pdf):
    assert fingerprint(born_digital_pdf) != fingerprint(multipage_pdf)


def test_accepts_born_digital(born_digital_pdf):
    doc = open_and_check(born_digital_pdf)
    assert doc.page_count == 1
    doc.close()


def test_rejects_scanned(scanned_pdf):
    with pytest.raises(GateError) as exc:
        open_and_check(scanned_pdf)
    assert exc.value.reason is RejectReason.NO_TEXT_LAYER


def test_rejects_encrypted(encrypted_pdf):
    with pytest.raises(GateError) as exc:
        open_and_check(encrypted_pdf)
    assert exc.value.reason is RejectReason.ENCRYPTED


def test_rejects_corrupt(corrupt_pdf):
    with pytest.raises(GateError) as exc:
        open_and_check(corrupt_pdf)
    assert exc.value.reason is RejectReason.CORRUPT


def test_rejects_missing_file(tmp_path):
    with pytest.raises(GateError) as exc:
        open_and_check(tmp_path / "nope.pdf")
    assert exc.value.reason is RejectReason.CORRUPT


def test_ratio_threshold_is_configurable(multipage_pdf, tmp_path):
    import fitz

    # 3 text pages plus 3 blank pages = exactly 50% text coverage.
    doc = fitz.open(multipage_pdf)
    for _ in range(3):
        doc.new_page(width=612, height=792)
    path = tmp_path / "half.pdf"
    doc.save(path)
    doc.close()

    with pytest.raises(GateError):
        open_and_check(path, min_text_page_ratio=0.6)

    ok = open_and_check(path, min_text_page_ratio=0.5)
    assert ok.page_count == 6
    ok.close()
