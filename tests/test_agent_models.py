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


def test_an_openai_compatible_endpoint_is_identified_by_its_host_not_its_provider():
    """The id is the response cache key. Two gateways serving the same model
    NAME are different weights behind an identical string, so a shared key
    would attribute one vendor's answer to another."""
    from visual_verify.agent.models import model_id

    a = model_id("openai_compatible", "llama-4-scout", "https://openrouter.ai/api/v1")
    b = model_id("openai_compatible", "llama-4-scout", "https://api.groq.com/openai/v1")

    assert a == "openrouter.ai:llama-4-scout"
    assert a != b


def test_the_wired_providers_keep_their_existing_ids():
    """Changing these would invalidate every cached response on disk."""
    from visual_verify.agent.models import model_id

    assert model_id("openai", "gpt-4o", None) == "openai:gpt-4o"
    assert model_id("google", "gemini-2.0-flash", None) == "google:gemini-2.0-flash"


def test_openai_compatible_without_a_base_url_is_refused_by_name(monkeypatch):
    from visual_verify.agent.models import UnknownProvider, make_chat
    from visual_verify.config import Settings

    settings = Settings(verifier_provider="openai_compatible", verifier_model="m")

    with pytest.raises(UnknownProvider, match="VVRAG_VERIFIER_BASE_URL"):
        make_chat("verifier", settings)


def test_openai_compatible_uses_a_per_role_key_so_two_gateways_can_differ(monkeypatch):
    """The reader and the verifier are expected behind DIFFERENT gateways, so a
    single shared key variable would make the independent-vendor setup the
    design rests on impossible to express."""
    from visual_verify.agent.models import MissingApiKey, make_chat
    from visual_verify.config import Settings

    monkeypatch.delenv("VVRAG_VERIFIER_API_KEY", raising=False)
    settings = Settings(
        verifier_provider="openai_compatible",
        verifier_model="m",
        verifier_base_url="https://openrouter.ai/api/v1",
    )

    with pytest.raises(MissingApiKey, match="VVRAG_VERIFIER_API_KEY"):
        make_chat("verifier", settings)


def test_an_unknown_provider_names_all_three_options():
    from visual_verify.agent.models import UnknownProvider, make_chat
    from visual_verify.config import Settings

    with pytest.raises(UnknownProvider, match="openai_compatible"):
        make_chat("reader", Settings(reader_provider="anthropic"))
