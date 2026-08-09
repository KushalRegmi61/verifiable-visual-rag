"""A misconfigured service must refuse to start.

Coming up and failing on the first question looks healthy to /health and to
anyone watching it boot, and the moment that gets discovered is during a demo.
"""

import pytest

from visual_verify.api.resources import StartupRefused, check_configuration, model_id_for
from visual_verify.config import Settings


def test_identical_reader_and_verifier_is_refused():
    """The self-preference argument is the reason S5 is shaped as it is. A
    misconfiguration pointing both at one model would be invisible in output."""
    settings = Settings(
        qdrant_url=":memory:",
        reader_provider="openai",
        reader_model="gpt-4o",
        verifier_provider="openai",
        verifier_model="gpt-4o",
    )

    with pytest.raises(StartupRefused, match="same model"):
        check_configuration(settings)


def test_different_models_pass():
    settings = Settings(
        qdrant_url=":memory:",
        reader_provider="openai",
        reader_model="gpt-4o",
        verifier_provider="google",
        verifier_model="gemini-2.0-flash",
    )

    check_configuration(settings)


def test_a_missing_qdrant_url_is_refused():
    settings = Settings(
        qdrant_url=None,
        reader_provider="openai",
        reader_model="gpt-4o",
        verifier_provider="google",
        verifier_model="gemini-2.0-flash",
    )

    with pytest.raises(StartupRefused, match="VVRAG_QDRANT_URL"):
        check_configuration(settings)


def test_the_error_names_the_environment_variable_to_fix():
    """A refusal that does not say what to set is a refusal nobody can act on."""
    settings = Settings(
        qdrant_url=":memory:",
        reader_provider="openai",
        reader_model="gpt-4o",
        verifier_provider="openai",
        verifier_model="gpt-4o",
    )

    with pytest.raises(StartupRefused) as exc:
        check_configuration(settings)

    assert "VVRAG_VERIFIER_MODEL" in str(exc.value)


def test_the_startup_check_compares_the_id_the_agent_compares(monkeypatch):
    """The startup check must reject exactly what answer() rejects.

    answer() compares `reader_chat.model_id == verifier_chat.model_id` on the
    real clients. If model_id_for built that string in any other shape, a
    config this module waved through would still blow up on the first
    question, which is strictly worse than having no startup check at all: the
    service would have reported itself healthy AND paid the twenty second model
    load before failing. Built through make_chat rather than asserted against a
    literal, so a change to LangChainChat's id format fails here.
    """
    pytest.importorskip("langchain_openai")
    from visual_verify.agent.models import make_chat

    # Never reaches a network: ChatOpenAI only requires the variable to be set.
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    settings = Settings(qdrant_url=":memory:", reader_provider="openai", reader_model="gpt-4o-mini")

    assert make_chat("reader", settings).model_id == model_id_for("reader", settings)


def test_the_id_pin_holds_for_an_openai_compatible_gateway_too(monkeypatch):
    """The compatible provider derives its id from the endpoint host, not from
    the provider string, so it is the case where the startup check and the
    client are most likely to drift."""
    pytest.importorskip("langchain_openai")
    from visual_verify.agent.models import make_chat

    monkeypatch.setenv("VVRAG_VERIFIER_API_KEY", "not-a-real-key")
    settings = Settings(
        qdrant_url=":memory:",
        verifier_provider="openai_compatible",
        verifier_model="llama-4-scout",
        verifier_base_url="https://openrouter.ai/api/v1",
    )

    assert make_chat("verifier", settings).model_id == model_id_for("verifier", settings)
    assert model_id_for("verifier", settings) == "openrouter.ai:llama-4-scout"


def test_two_gateways_serving_the_same_model_name_are_not_the_same_model():
    """Reader on one gateway and verifier on another, both running a model of
    the same name, is a self-preference risk the id cannot see: it is the same
    weights twice. The startup check compares ids, so this pair is ALLOWED, and
    that limit is recorded here rather than left to be discovered."""
    settings = Settings(
        qdrant_url=":memory:",
        reader_provider="openai_compatible",
        reader_model="llama-4-scout",
        reader_base_url="https://openrouter.ai/api/v1",
        verifier_provider="openai_compatible",
        verifier_model="llama-4-scout",
        verifier_base_url="https://api.groq.com/openai/v1",
    )

    assert model_id_for("reader", settings) != model_id_for("verifier", settings)
    check_configuration(settings)


def test_an_unknown_role_is_a_programming_error():
    """model_id_for mirrors make_chat's role dispatch, so it must reject the
    same inputs rather than silently formatting an empty pair."""
    with pytest.raises(ValueError, match="reader"):
        model_id_for("judge", Settings())
