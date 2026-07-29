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


def test_engine_enforces_foreign_keys(tmp_path, monkeypatch):
    """The CLI must route through make_engine or the FK guarantee evaporates."""
    from sqlalchemy import text

    from visual_verify.cli import _session
    from visual_verify.config import Settings

    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'fk.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))

    with _session(Settings.from_env()) as s:
        assert s.execute(text("PRAGMA foreign_keys")).scalar() == 1
