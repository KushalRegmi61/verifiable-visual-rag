import pytest

from visual_verify.cli import main


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VVRAG_QDRANT_URL", ":memory:")
    monkeypatch.delenv("VVRAG_QDRANT_API_KEY", raising=False)
    # Keep the CLI off the GPU: the fake embedder makes these fast tests.
    monkeypatch.setenv("VVRAG_FAKE_EMBEDDER", "1")
    return tmp_path


def test_embed_then_search(env, born_digital_pdf, capsys):
    assert main(["ingest", str(born_digital_pdf)]) == 0
    capsys.readouterr()

    assert main(["embed", "--all"]) == 0
    assert "embedded" in capsys.readouterr().out.lower()

    assert main(["search", "anything"]) == 0
    assert "page" in capsys.readouterr().out.lower()


def test_embed_is_idempotent(env, born_digital_pdf, capsys):
    main(["ingest", str(born_digital_pdf)])
    main(["embed", "--all"])
    capsys.readouterr()
    assert main(["embed", "--all"]) == 0
    assert "skipped" in capsys.readouterr().out.lower()


def test_search_before_embed_reports_empty(env, born_digital_pdf, capsys):
    main(["ingest", str(born_digital_pdf)])
    capsys.readouterr()
    assert main(["search", "anything"]) == 1
    assert "no pages indexed" in capsys.readouterr().out.lower()


def test_embed_requires_a_target(env, capsys):
    assert main(["embed"]) == 1
    assert "give a document" in capsys.readouterr().out.lower()
