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


def test_model_family_strips_a_gateway_vendor_prefix():
    """OpenRouter addresses gpt-4o as `openai/gpt-4o`. The id keeps the whole
    string because it is a cache key; the identity test must not."""
    from visual_verify.agent.models import model_family

    assert model_family("openai/gpt-4o") == "gpt-4o"
    assert model_family("gpt-4o") == "gpt-4o"
    assert model_family("meta-llama/llama-4-scout") == "llama-4-scout"


def test_model_family_does_not_collapse_different_models():
    """A normalizer that over-collapsed would refuse every valid pair and
    surface as "the service will not start"."""
    from visual_verify.agent.models import model_family

    assert model_family("openai/gpt-4o") != model_family("openai/gpt-4o-mini")
    assert model_family("gemini-2.0-flash") != model_family("gpt-4o")


def test_a_base_url_with_a_non_compatible_provider_is_refused(monkeypatch):
    """Set with provider=openai it was silently discarded: ChatOpenAI was built
    without it, every call went to api.openai.com billed to a key the operator
    believed unused, and model_id dropped it too so /health showed
    `openai:gpt-4o` and looked correct."""
    import pytest

    from visual_verify.agent.models import UnknownProvider, make_chat

    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    settings = Settings(
        reader_provider="openai",
        reader_model="gpt-4o",
        reader_base_url="https://my-gateway/v1",
    )

    with pytest.raises(UnknownProvider) as exc:
        make_chat("reader", settings)

    # Both variables, because the fix is a choice between them.
    assert "VVRAG_READER_BASE_URL" in str(exc.value)
    assert "VVRAG_READER_PROVIDER" in str(exc.value)


def test_verifier_rotates_through_the_key_pool_on_a_rate_limit(monkeypatch):
    """A 429 on key N must retry the SAME request on key N+1, not raise. This
    is the whole reason for the KEY_1..KEY_6 pool: one Groq account's per-key
    limit must not abort a verifier call another key in the same pool would
    have served."""
    import httpx
    from openai import RateLimitError

    from visual_verify.agent.models import LangChainChat

    calls: list[str] = []

    class FakeChain:
        def __init__(self, key: str) -> None:
            self._key = key

        def with_structured_output(self, schema):
            return self

        def with_retry(self, **kwargs):
            return self

        def invoke(self, messages):
            calls.append(self._key)
            if self._key != "good-key":
                response = httpx.Response(429, request=httpx.Request("POST", "https://x"))
                raise RateLimitError("rate limited", response=response, body=None)
            return "ok"

    monkeypatch.setattr(LangChainChat, "_build_llm", lambda self, api_key: FakeChain(api_key))

    chat = LangChainChat(
        "openai_compatible",
        "m",
        base_url="https://api.groq.com/openai/v1",
        api_keys=["bad-1", "bad-2", "good-key"],
    )

    assert chat.structured("prompt", [], schema=None) == "ok"
    assert calls == ["bad-1", "bad-2", "good-key"]


def test_verifier_reraises_once_the_whole_key_pool_is_rate_limited(monkeypatch):
    """Every key exhausted must fail loudly, not return silently with no
    answer: a caller expecting an exception on total failure must get one."""
    import httpx
    from openai import RateLimitError

    from visual_verify.agent.models import LangChainChat

    class AlwaysLimited:
        def with_structured_output(self, schema):
            return self

        def with_retry(self, **kwargs):
            return self

        def invoke(self, messages):
            response = httpx.Response(429, request=httpx.Request("POST", "https://x"))
            raise RateLimitError("rate limited", response=response, body=None)

    monkeypatch.setattr(LangChainChat, "_build_llm", lambda self, api_key: AlwaysLimited())

    chat = LangChainChat(
        "openai_compatible",
        "m",
        base_url="https://api.groq.com/openai/v1",
        api_keys=["bad-1", "bad-2"],
    )

    with pytest.raises(RateLimitError):
        chat.structured("prompt", [], schema=None)


def test_make_chat_wires_the_verifier_key_pool_from_settings(monkeypatch):
    """Settings.verifier_api_keys, not just VVRAG_VERIFIER_API_KEY, must reach
    the client the verifier actually calls, or the pool is collected for
    nothing."""
    from visual_verify.agent.models import LangChainChat, make_chat
    from visual_verify.config import Settings

    captured: dict = {}
    original_init = LangChainChat.__init__

    def spy_init(self, provider, model, base_url=None, api_key=None, api_keys=None):
        captured["api_keys"] = api_keys
        original_init(self, provider, model, base_url=base_url, api_key=api_key, api_keys=api_keys)

    monkeypatch.setattr(LangChainChat, "__init__", spy_init)

    settings = Settings(
        verifier_provider="openai_compatible",
        verifier_model="qwen/qwen3.6-27b",
        verifier_base_url="https://api.groq.com/openai/v1",
        verifier_api_keys=("k1", "k2", "k3"),
    )

    make_chat("verifier", settings)

    assert captured["api_keys"] == ["k1", "k2", "k3"]


def test_the_other_role_is_unaffected_by_a_base_url(monkeypatch):
    """The guard reads the per-role variable, so a reader base_url must not
    refuse the verifier."""
    pytest.importorskip("langchain_google_genai")
    from visual_verify.agent.models import make_chat

    monkeypatch.setenv("GOOGLE_API_KEY", "not-a-real-key")
    settings = Settings(
        reader_base_url="https://my-gateway/v1",
        verifier_provider="google",
        verifier_model="gemini-2.0-flash",
    )

    assert make_chat("verifier", settings).model_id == "google:gemini-2.0-flash"
