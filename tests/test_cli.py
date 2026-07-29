from PIL import Image

from visual_verify.cli import main


def test_ingest_single_file(multipage_pdf, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))

    assert main(["ingest", str(multipage_pdf), "--dpi", "72"]) == 0
    assert "3 pages" in capsys.readouterr().out


def test_ingest_directory(multipage_pdf, born_digital_pdf, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))

    assert main(["ingest", "--dir", str(multipage_pdf.parent), "--dpi", "72"]) == 0
    out = capsys.readouterr().out
    assert "multipage.pdf" in out and "born_digital.pdf" in out


def test_ingest_reports_rejection_without_crashing(scanned_pdf, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))

    assert main(["ingest", str(scanned_pdf), "--dpi", "72"]) == 1
    assert "no_text_layer" in capsys.readouterr().out


def test_directory_ingest_continues_past_a_bad_file(
    multipage_pdf, scanned_pdf, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))

    main(["ingest", "--dir", str(multipage_pdf.parent), "--dpi", "72"])
    out = capsys.readouterr().out
    assert "no_text_layer" in out
    assert "3 pages" in out


def test_status_lists_documents(multipage_pdf, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    main(["ingest", str(multipage_pdf), "--dpi", "72"])
    capsys.readouterr()

    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "indexed" in out and "3/3" in out


def test_inspect_writes_overlay(multipage_pdf, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    main(["ingest", str(multipage_pdf), "--dpi", "72"])
    capsys.readouterr()

    overlay = tmp_path / "overlay.png"
    assert main(["inspect", "multipage", "--page", "0", "--overlay", str(overlay)]) == 0
    assert overlay.exists()
    assert Image.open(overlay).size == (612, 792)


def test_inspect_reports_unknown_document(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))

    assert main(["inspect", "nosuchdoc", "--page", "0"]) == 1
    assert "no document" in capsys.readouterr().out.lower()


def test_inspect_reports_ambiguous_match(
    multipage_pdf, born_digital_pdf, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    main(["ingest", "--dir", str(multipage_pdf.parent), "--dpi", "72"])
    capsys.readouterr()

    # Both fixtures live in tmp_path and both paths contain "pdf".
    assert main(["inspect", ".pdf", "--page", "0"]) == 1
    out = capsys.readouterr().out.lower()
    assert "ambiguous" in out
    assert "multipage.pdf" in out and "born_digital.pdf" in out


def test_inspect_accepts_an_exact_sha256(multipage_pdf, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    main(["ingest", str(multipage_pdf), "--dpi", "72"])
    capsys.readouterr()

    from visual_verify.ingest.gate import fingerprint

    sha = fingerprint(multipage_pdf)
    assert main(["inspect", sha, "--page", "0"]) == 0


def test_inspect_accepts_a_sha256_prefix(multipage_pdf, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    main(["ingest", str(multipage_pdf), "--dpi", "72"])
    capsys.readouterr()

    from visual_verify.ingest.gate import fingerprint

    assert main(["inspect", fingerprint(multipage_pdf)[:12], "--page", "0"]) == 0


def test_status_columns_align_with_long_filenames(tmp_path, monkeypatch, capsys):
    import fitz

    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))

    long_name = tmp_path / ("A" * 60 + ".pdf")
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "hello world", fontsize=12)
    doc.save(long_name)
    doc.close()

    main(["ingest", str(long_name), "--dpi", "72"])
    capsys.readouterr()

    main(["status"])
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    header, row = lines[0], lines[1]
    # The status column must start at the same offset in header and row.
    assert header.index("status") == row.index("indexed")


def test_missing_file_reports_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))

    assert main(["ingest", str(tmp_path / "nope.pdf"), "--dpi", "72"]) == 1
    assert "no such file" in capsys.readouterr().out


def test_engine_enforces_foreign_keys(tmp_path, monkeypatch):
    """The CLI must route through make_engine or the FK guarantee evaporates."""
    from sqlalchemy import text

    from visual_verify.cli import _session
    from visual_verify.config import Settings

    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'fk.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))

    with _session(Settings.from_env()) as s:
        assert s.execute(text("PRAGMA foreign_keys")).scalar() == 1


def test_directory_ingest_survives_an_unexpected_error(
    multipage_pdf, born_digital_pdf, tmp_path, monkeypatch, capsys
):
    """A non-gate failure on one file must not abandon the rest of the batch."""
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))

    from visual_verify import cli as cli_module

    real = cli_module.ingest_pdf

    def explode_on_multipage(path, *a, **kw):
        if "multipage" in str(path):
            raise PermissionError("simulated disk failure")
        return real(path, *a, **kw)

    monkeypatch.setattr(cli_module, "ingest_pdf", explode_on_multipage)

    rc = main(["ingest", "--dir", str(multipage_pdf.parent), "--dpi", "72"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "PermissionError" in out
    assert "born_digital.pdf" in out and "1 pages written" in out


def test_inspect_find_draws_a_span(born_digital_pdf, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    main(["ingest", str(born_digital_pdf), "--dpi", "72"])
    capsys.readouterr()

    overlay = tmp_path / "found.png"
    rc = main(
        ["inspect", "born_digital", "--page", "0", "--find", "grew 42", "--overlay", str(overlay)]
    )
    assert rc == 0
    assert overlay.exists()
    out = capsys.readouterr().out
    assert "1" in out and "match" in out.lower()


def test_inspect_find_reports_a_miss(born_digital_pdf, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    main(["ingest", str(born_digital_pdf), "--dpi", "72"])
    capsys.readouterr()

    assert main(["inspect", "born_digital", "--page", "0", "--find", "no such phrase"]) == 0
    assert "not found" in capsys.readouterr().out.lower()


def test_inspect_derives_line_boxes(born_digital_pdf, tmp_path, monkeypatch, capsys):
    """--kind line must collapse the page's words into fewer, wider boxes."""
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    main(["ingest", str(born_digital_pdf), "--dpi", "72"])
    capsys.readouterr()

    overlay = tmp_path / "lines.png"
    assert (
        main(
            ["inspect", "born_digital", "--page", "0", "--kind", "line", "--overlay", str(overlay)]
        )
        == 0
    )
    out = capsys.readouterr().out
    # The fixture is two lines of seven words total.
    assert "2 line boxes derived from 7 stored boxes" in out
    assert overlay.exists()
