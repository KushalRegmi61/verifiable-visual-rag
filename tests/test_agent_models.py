"""Client construction from environment, without importing LangChain."""

import pytest

from visual_verify.config import Settings


def test_settings_read_both_roles_from_env(monkeypatch):
    monkeypatch.setenv("VVRAG_READER_PROVIDER", "openai")
    monkeypatch.setenv("VVRAG_READER_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("VVRAG_VERIFIER_PROVIDER", "google")
    monkeypatch.setenv("VVRAG_VERIFIER_MODEL", "gemini-2.0-flash")

    s = Settings.from_env()

    assert s.reader_model == "gpt-4o-mini"
    assert s.verifier_model == "gemini-2.0-flash"


def test_the_default_threshold_is_the_supported_floor():
    """Bands are [0,1], [2,3], [4,5], [6,7], so 6.0 admits only 'supported'.

    Spacing of 1 would let partially_supported at confidence 1.0 tie the
    supported floor exactly and be shown."""
    from visual_verify.agent.rubric import abstention_score

    s = Settings()
    assert abstention_score("supported", 0.0) >= s.abstain_threshold
    assert abstention_score("partially_supported", 1.0) < s.abstain_threshold


def test_an_unknown_provider_names_the_env_var(monkeypatch):
    from visual_verify.agent.models import UnknownProvider, make_chat

    monkeypatch.setenv("VVRAG_READER_PROVIDER", "anthropic-typo")
    with pytest.raises(UnknownProvider, match="VVRAG_READER_PROVIDER"):
        make_chat("reader", Settings.from_env())


def test_a_missing_api_key_names_the_variable(monkeypatch):
    """A key error must say WHICH variable is unset. This is the most common
    first-run failure and a bare KeyError from inside a client is useless."""
    from visual_verify.agent.models import MissingApiKey, make_chat

    monkeypatch.setenv("VVRAG_READER_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingApiKey, match="OPENAI_API_KEY"):
        make_chat("reader", Settings.from_env())
