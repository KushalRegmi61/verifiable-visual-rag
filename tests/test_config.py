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


def test_env_overrides_qdrant_url(monkeypatch):
    monkeypatch.setenv("VVRAG_QDRANT_URL", "https://xyz.qdrant.cloud")
    assert Settings.from_env().qdrant_url == "https://xyz.qdrant.cloud"


def test_qdrant_url_defaults_to_none(monkeypatch):
    monkeypatch.delenv("VVRAG_QDRANT_URL", raising=False)
    assert Settings.from_env().qdrant_url is None


def test_env_overrides_min_text_page_ratio(monkeypatch):
    monkeypatch.setenv("VVRAG_MIN_TEXT_PAGE_RATIO", "0.8")
    assert Settings.from_env().min_text_page_ratio == 0.8


def test_qdrant_api_key_from_env(monkeypatch):
    monkeypatch.setenv("VVRAG_QDRANT_API_KEY", "secret-key")
    assert Settings.from_env().qdrant_api_key == "secret-key"


def test_qdrant_api_key_defaults_to_none(monkeypatch):
    monkeypatch.delenv("VVRAG_QDRANT_API_KEY", raising=False)
    assert Settings.from_env().qdrant_api_key is None


def test_abstain_threshold_defaults_to_the_rubrics_supported_floor(monkeypatch):
    from visual_verify.agent.rubric import SUPPORTED_FLOOR

    monkeypatch.delenv("VVRAG_ABSTAIN_THRESHOLD", raising=False)
    assert Settings.from_env().abstain_threshold == SUPPORTED_FLOOR


def test_env_overrides_abstain_threshold(monkeypatch):
    """VVRAG_ABSTAIN_THRESHOLD was read into Settings but nothing consumed it;
    a user setting it got no effect and no warning. cmd_ask's --threshold
    default is built from this setting, so this pins the value that flows
    into an unflagged `vvrag ask`.
    """
    monkeypatch.setenv("VVRAG_ABSTAIN_THRESHOLD", "4.0")
    assert Settings.from_env().abstain_threshold == 4.0

    from visual_verify.cli import build_parser

    args = build_parser().parse_args(["ask", "q", "--doc", "abc", "--page", "1"])
    assert args.threshold == 4.0
