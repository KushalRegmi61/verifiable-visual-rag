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


def _is_rate_limit(exc: Exception) -> bool:
    """True for a 429 from the openai SDK that langchain_openai raises through.

    Import is function-local like every other client import in this file: this
    check must work whether or not `openai` happens to be installed under
    whatever provider is active, and it must not become the thing that drags
    the dependency into a process that never configured a compatible endpoint.
    """
    try:
        from openai import RateLimitError
    except ImportError:  # pragma: no cover - openai ships with langchain_openai
        RateLimitError = ()  # type: ignore[assignment]
    if isinstance(exc, RateLimitError):
        return True
    return getattr(exc, "status_code", None) == 429


class LangChainChat:
    """StructuredChat backed by a LangChain chat model."""

    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_keys: list[str] | None = None,
    ) -> None:
        self._model_id = model_id(provider, model, base_url)
        self._provider = provider
        self._model = model
        self._base_url = base_url
        # api_keys is the Groq-rotation pool (verifier only, see make_chat); a
        # bare api_key is the single-key path every other caller still uses.
        # Normalized to one list so structured() has one rotation code path
        # regardless of which one was given, even though a length-1 pool never
        # actually rotates.
        self._api_keys = list(api_keys) if api_keys else [api_key]
        self._key_index = 0
        self._llm = self._build_llm(self._api_keys[0])

    def _build_llm(self, api_key: str | None):
        provider, model, base_url = self._provider, self._model, self._base_url
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=model, temperature=0)
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(model=model, temperature=0)
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
            return ChatOpenAI(model=model, temperature=0, base_url=base_url, api_key=api_key)
        else:  # pragma: no cover - guarded by make_chat
            raise UnknownProvider(provider)

    def _rotate_key(self) -> None:
        self._key_index = (self._key_index + 1) % len(self._api_keys)
        self._llm = self._build_llm(self._api_keys[self._key_index])

    @property
    def model_id(self) -> str:
        return self._model_id

    def structured(self, prompt: str, image_paths: list[Path], schema: type[S]) -> S:
        content: list[dict] = [{"type": "text", "text": prompt}]
        # One block per page, in the order given. The prompt numbers the pages
        # in that same order, so a reordering here would leave the model
        # describing page 8 while the text calls it page 7, and nothing about
        # the response would look wrong.
        for image_path in image_paths:
            encoded = base64.b64encode(Path(image_path).read_bytes()).decode()
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            )
        messages = [{"role": "user", "content": content}]

        # One attempt per key in the pool. A 429 rotates to the next Groq key
        # and retries the SAME request; anything else (schema-invalid,
        # timeout) is left to with_retry below and never rotates a key, because
        # those are not the failure a second account fixes. The last key's
        # exception propagates rather than being swallowed, so a pool that is
        # entirely rate-limited still fails loudly instead of returning nothing.
        for attempt in range(len(self._api_keys)):
            # with_structured_output is why LangChain earns its weight here:
            # one call gives schema-validated output on both providers, so a
            # malformed response raises instead of parsing into something
            # plausible. with_retry covers a transient schema-invalid
            # response, which spec section 10 requires. It retries and then
            # raises: it never coerces a bad response into a valid-looking
            # object, because a silently mis-parsed claim list is the failure
            # this whole layer exists to stop.
            chain = self._llm.with_structured_output(schema).with_retry(stop_after_attempt=3)
            try:
                return chain.invoke(messages)
            except Exception as exc:
                is_last = attempt == len(self._api_keys) - 1
                if is_last or not _is_rate_limit(exc):
                    raise
                self._rotate_key()
        raise AssertionError("unreachable: loop above always returns or raises")  # pragma: no cover


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
        # The verifier gets the KEY_1..KEY_6 rotation pool when one was
        # collected (Settings._verifier_api_keys already folds
        # VVRAG_VERIFIER_API_KEY into it as the first entry); every other
        # role/provider combination keeps the single-key path unchanged.
        if role == "verifier" and settings.verifier_api_keys:
            return LangChainChat(
                provider, model, base_url=base_url, api_keys=list(settings.verifier_api_keys)
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
