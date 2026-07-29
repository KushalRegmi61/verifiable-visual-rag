from pathlib import Path

from visual_verify.config import Settings


def test_defaults_are_local(monkeypatch):
    monkeypatch.delenv("VVRAG_DB_URL", raising=False)
    monkeypatch.delenv("VVRAG_DATA_DIR", raising=False)
    s = Settings.from_env()
    assert s.db_url == "sqlite:///data/index.db"
    assert s.data_dir == Path("data")
    assert s.render_dpi == 150


def test_env_overrides_db_url(monkeypatch):
    monkeypatch.setenv("VVRAG_DB_URL", "postgresql+psycopg://user@host/db")
    assert Settings.from_env().db_url == "postgresql+psycopg://user@host/db"


def test_env_overrides_dpi(monkeypatch):
    monkeypatch.setenv("VVRAG_RENDER_DPI", "72")
    assert Settings.from_env().render_dpi == 72


def test_pages_dir_is_under_data_dir(monkeypatch):
    monkeypatch.setenv("VVRAG_DATA_DIR", "/tmp/vv")
    s = Settings.from_env()
    assert s.pages_dir == Path("/tmp/vv/pages")
