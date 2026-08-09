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


def test_two_gateways_serving_the_same_model_name_are_refused():
    """Reader on one gateway and verifier on another, both running a model of
    the same name, is the same weights twice: a self-preference risk the ids
    cannot see, because each id carries its own endpoint host.

    The ids genuinely must differ (they are cache keys, and one gateway's answer
    must not be served for another's), so the check cannot be folded into the id
    comparison. It is a separate family comparison, and this pins that the
    stricter check is the one that fires.
    """
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
    with pytest.raises(StartupRefused, match="llama-4-scout"):
        check_configuration(settings)


def test_a_gateway_vendor_prefix_does_not_disguise_the_same_model():
    """The failure this check exists for. A deployer with no Gemini credit
    points the verifier at OpenRouter, which addresses gpt-4o as
    `openai/gpt-4o`. The ids are `openai:gpt-4o` and
    `openrouter.ai:openai/gpt-4o`, they differ, and both the startup check and
    answer_stream's own guard used to pass. gpt-4o then graded its own output
    for every claim while /health displayed two different-looking names.
    """
    settings = Settings(
        qdrant_url=":memory:",
        reader_provider="openai",
        reader_model="gpt-4o",
        verifier_provider="openai_compatible",
        verifier_model="openai/gpt-4o",
        verifier_base_url="https://openrouter.ai/api/v1",
    )

    assert model_id_for("reader", settings) != model_id_for("verifier", settings)
    with pytest.raises(StartupRefused, match="gpt-4o"):
        check_configuration(settings)


def test_the_family_check_does_not_refuse_genuinely_different_models():
    """The guard above is only useful if it still admits a valid pair. A
    normalizer that collapsed too much would refuse every configuration and
    would be discovered as "the service will not start"."""
    settings = Settings(
        qdrant_url=":memory:",
        reader_provider="openai",
        reader_model="gpt-4o",
        verifier_provider="openai_compatible",
        verifier_model="meta-llama/llama-4-scout",
        verifier_base_url="https://openrouter.ai/api/v1",
    )

    check_configuration(settings)


def test_a_threshold_no_score_can_clear_is_refused():
    """A negative threshold makes `score < threshold` False for every claim, so
    nothing is ever withheld and the abstention gate is off while /health still
    reports the service healthy."""
    settings = Settings(qdrant_url=":memory:", abstain_threshold=-1.0)

    with pytest.raises(StartupRefused, match="VVRAG_ABSTAIN_THRESHOLD"):
        check_configuration(settings)


def test_an_unknown_role_is_a_programming_error():
    """model_id_for mirrors make_chat's role dispatch, so it must reject the
    same inputs rather than silently formatting an empty pair."""
    with pytest.raises(ValueError, match="reader"):
        model_id_for("judge", Settings())
