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


def test_verifier_api_keys_collects_the_pool_with_the_single_key_first(monkeypatch):
    """VVRAG_VERIFIER_API_KEY must stay index 0 so an existing single-key
    deployment tries the same key first, and a key repeated in KEY_1..KEY_6
    must not be tried twice while a distinct one is starved."""
    monkeypatch.setenv("VVRAG_VERIFIER_API_KEY", "primary")
    monkeypatch.setenv("KEY_1", "primary")
    monkeypatch.setenv("KEY_2", "second")
    monkeypatch.delenv("KEY_3", raising=False)
    monkeypatch.delenv("KEY_4", raising=False)
    monkeypatch.delenv("KEY_5", raising=False)
    monkeypatch.delenv("KEY_6", raising=False)

    assert Settings.from_env().verifier_api_keys == ("primary", "second")


def test_verifier_api_keys_is_empty_with_nothing_set(monkeypatch):
    monkeypatch.delenv("VVRAG_VERIFIER_API_KEY", raising=False)
    for i in range(1, 7):
        monkeypatch.delenv(f"KEY_{i}", raising=False)

    assert Settings.from_env().verifier_api_keys == ()


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


def test_a_non_finite_threshold_from_the_environment_is_refused(monkeypatch):
    """NaN is the dangerous one, and it is not an exotic input: a templating
    system rendering an unset value, or a plain typo, produces it.

    `score < threshold` is False for every comparison against NaN, so nothing
    ever abstains: unsupported claims come back with abstained=False,
    Claim.withheld is False, the API ships their regions, and the UI draws
    evidence boxes for them. The abstention gate is off and every surface still
    reports success. A bare float() accepts "nan" without complaint, which is
    how the environment became the one entry point that did not check.
    """
    import pytest

    monkeypatch.setenv("VVRAG_ABSTAIN_THRESHOLD", "nan")
    with pytest.raises(ValueError, match="VVRAG_ABSTAIN_THRESHOLD"):
        Settings.from_env()


def test_the_infinities_are_refused_too(monkeypatch):
    """inf withholds every claim rather than none, which is the safe direction
    and still a misconfiguration that should not start silently."""
    import pytest

    monkeypatch.setenv("VVRAG_ABSTAIN_THRESHOLD", "inf")
    with pytest.raises(ValueError, match="VVRAG_ABSTAIN_THRESHOLD"):
        Settings.from_env()


def test_a_non_numeric_threshold_names_the_variable(monkeypatch):
    """float() raises "could not convert string to float: 'high'", which names
    neither the variable nor the file it came from."""
    import pytest

    monkeypatch.setenv("VVRAG_ABSTAIN_THRESHOLD", "high")
    with pytest.raises(ValueError, match="VVRAG_ABSTAIN_THRESHOLD"):
        Settings.from_env()


def test_cors_origins_default_to_the_dev_frontend(monkeypatch):
    monkeypatch.delenv("VVRAG_CORS_ORIGINS", raising=False)
    assert Settings.from_env().cors_origins == ("http://localhost:3000",)


def test_cors_origins_are_configurable(monkeypatch):
    """The frontend's API base is already an environment variable, so a
    hardcoded origin on this side made the pair unconfigurable: a UI on 3001,
    on 127.0.0.1, or on a real host had every request blocked by preflight
    while the server logged a normal 200."""
    monkeypatch.setenv("VVRAG_CORS_ORIGINS", "https://app.example.com, http://127.0.0.1:3001")
    assert Settings.from_env().cors_origins == (
        "https://app.example.com",
        "http://127.0.0.1:3001",
    )


def test_an_empty_cors_value_falls_back_rather_than_blocking_everything(monkeypatch):
    """An empty allow-list blocks every browser request and looks identical to
    a working service from the server side, so it is not something to reach by
    setting a variable to the empty string."""
    monkeypatch.setenv("VVRAG_CORS_ORIGINS", "   ")
    assert Settings.from_env().cors_origins == ("http://localhost:3000",)
