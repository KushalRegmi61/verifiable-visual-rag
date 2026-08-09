"""The ONLY file in this package that imports LangChain.

Everything else takes a StructuredChat. That boundary is enforced by a
subprocess test in tests/test_core_is_light.py, and it is what lets the reader,
the verifier, and the whole pipeline be tested with no network and no key.

Imports are function-local so that importing visual_verify.agent does not drag
LangChain in. The boundary test checks exactly that.
"""

import base64
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from visual_verify.config import Settings

S = TypeVar("S", bound=BaseModel)

# Environment variable each provider's client library reads for its key.
# `openai_compatible` is deliberately absent: its key is per role, because the
# reader and the verifier are expected to sit behind DIFFERENT gateways.
_KEY_VAR = {"openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}

# Any OpenAI-shaped endpoint: OpenRouter, Groq, Together, DeepSeek, a local
# vLLM or Ollama. Served by langchain_openai with a base_url, so it adds no
# dependency. This is what makes the vendor an environment variable instead of
# a code change, which matters because pillar 3 rests on the reader and the
# verifier coming from different model families and a single-vendor outage
# would otherwise force them onto one.
COMPATIBLE = "openai_compatible"

PROVIDERS = (*sorted(_KEY_VAR), COMPATIBLE)


def _role_var(role: str, suffix: str) -> str:
    return f"VVRAG_{role.upper()}_{suffix}"


def endpoint_label(base_url: str) -> str:
    """The host of `base_url`, used in the model id.

    The id goes into the response cache key, so two gateways serving the same
    model NAME must not collide: they are different weights behind an identical
    string, and a cache hit across them would attribute one vendor's answer to
    another. The host disambiguates them and stays readable in a cache path,
    which a full URL with a key in it would not.
    """
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    return parsed.netloc or parsed.path.strip("/") or base_url


def model_family(model: str) -> str:
    """The bare model name, with any vendor routing prefix stripped.

    Gateways address a model as `<vendor>/<name>`: OpenRouter serves gpt-4o as
    `openai/gpt-4o`. The model id built below keeps the endpoint host so that a
    response cache cannot confuse two gateways, and that is correct for a CACHE
    KEY and wrong for an IDENTITY TEST. `openai:gpt-4o` and
    `openrouter.ai:openai/gpt-4o` are different strings naming one model, so an
    independence check comparing ids alone lets gpt-4o grade its own output
    while /health displays two different-looking names. Comparing families
    catches that pair.

    Deliberately conservative. It cannot see that `gpt-4o` and
    `gpt-4o-2024-08-06` are the same weights, or that two gateways both serving
    `llama-4-scout` are one model. Those remain possible and undetectable here;
    see test_two_gateways_serving_the_same_model_name_are_not_the_same_model.
    """
    return model.rsplit("/", 1)[-1].strip().lower()


def model_id(provider: str, model: str, base_url: str | None) -> str:
    """The identity answer() compares and the cache keys on.

    Kept as a free function so `Settings` can be turned into an id without
    building a client, which needs LangChain installed and a key present.
    """
    if provider == COMPATIBLE and base_url:
        return f"{endpoint_label(base_url)}:{model}"
    return f"{provider}:{model}"


class UnknownProvider(RuntimeError):
    """Provider string is not one this project supports."""


class MissingApiKey(RuntimeError):
    """The provider's key variable is unset."""


class LangChainChat:
    """StructuredChat backed by a LangChain chat model."""

    def __init__(
        self, provider: str, model: str, base_url: str | None = None, api_key: str | None = None
    ) -> None:
        self._model_id = model_id(provider, model, base_url)
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(model=model, temperature=0)
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        elif provider == COMPATIBLE:
            from langchain_openai import ChatOpenAI

            # Same client class as "openai", pointed elsewhere. The gateway has
            # to speak the OpenAI chat-completions shape, because structured()
            # sends the page as an OpenAI-style image_url content block and
            # relies on with_structured_output, which needs tool calling or a
            # native JSON-schema mode. A gateway lacking either degrades to
            # prompt-and-parse, which is the correctly-shaped wrong output this
            # whole layer exists to prevent. Pick a vision model that supports
            # tools, and confirm it on a real call rather than assuming.
            self._llm = ChatOpenAI(model=model, temperature=0, base_url=base_url, api_key=api_key)
        else:  # pragma: no cover - guarded by make_chat
            raise UnknownProvider(provider)

    @property
    def model_id(self) -> str:
        return self._model_id

    def structured(self, prompt: str, image_path: Path | None, schema: type[S]) -> S:
        content: list[dict] = [{"type": "text", "text": prompt}]
        if image_path is not None:
            encoded = base64.b64encode(Path(image_path).read_bytes()).decode()
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            )
        # with_structured_output is why LangChain earns its weight here: one
        # call gives schema-validated output on both providers, so a malformed
        # response raises instead of parsing into something plausible.
        # with_retry covers a transient schema-invalid response, which spec
        # section 10 requires. It retries and then raises: it never coerces a
        # bad response into a valid-looking object, because a silently
        # mis-parsed claim list is the failure this whole layer exists to stop.
        chain = self._llm.with_structured_output(schema).with_retry(stop_after_attempt=3)
        return chain.invoke([{"role": "user", "content": content}])


def make_chat(role: str, settings: Settings) -> LangChainChat:
    """Build the reader's or the verifier's client. role is 'reader' or 'verifier'."""
    if role == "reader":
        provider, model = settings.reader_provider, settings.reader_model
    elif role == "verifier":
        provider, model = settings.verifier_provider, settings.verifier_model
    else:
        raise ValueError(f"role must be 'reader' or 'verifier', got {role!r}")

    if provider not in PROVIDERS:
        raise UnknownProvider(
            f"{_role_var(role, 'PROVIDER')} is {provider!r}; expected one of {list(PROVIDERS)}"
        )

    base_url = settings.reader_base_url if role == "reader" else settings.verifier_base_url

    if provider != COMPATIBLE and base_url:
        # Refused rather than ignored. `openai` plus a base_url reads as "use my
        # gateway", and the branch below would build a plain ChatOpenAI without
        # it: every call would go to api.openai.com billed to a key the operator
        # believed unused, or fail 401 with a message naming OPENAI_API_KEY and
        # never mentioning the endpoint that was discarded. model_id drops the
        # base_url for these providers too, so /health would show `openai:gpt-4o`
        # and look correct.
        raise UnknownProvider(
            f"{_role_var(role, 'BASE_URL')} is set but {_role_var(role, 'PROVIDER')} is "
            f"{provider!r}, which always calls the vendor's own endpoint and would "
            f"silently ignore it. Set {_role_var(role, 'PROVIDER')}={COMPATIBLE!r} to use "
            f"that endpoint, or unset {_role_var(role, 'BASE_URL')}."
        )

    if provider == COMPATIBLE:
        if not base_url:
            raise UnknownProvider(
                f"{_role_var(role, 'PROVIDER')} is {COMPATIBLE!r} but "
                f"{_role_var(role, 'BASE_URL')} is not set, so there is no endpoint to call"
            )
        key_var = _role_var(role, "API_KEY")
        api_key = os.getenv(key_var)
        if not api_key:
            raise MissingApiKey(
                f"{key_var} is not set, which the {role} needs to reach {endpoint_label(base_url)}"
            )
        return LangChainChat(provider, model, base_url=base_url, api_key=api_key)

    key_var = _KEY_VAR[provider]
    if not os.getenv(key_var):
        raise MissingApiKey(f"{key_var} is not set, which the {role} needs to reach {provider}")
    return LangChainChat(provider, model)
