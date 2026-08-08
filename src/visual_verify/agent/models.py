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
_KEY_VAR = {"openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}


class UnknownProvider(RuntimeError):
    """Provider string is not one this project supports."""


class MissingApiKey(RuntimeError):
    """The provider's key variable is unset."""


class LangChainChat:
    """StructuredChat backed by a LangChain chat model."""

    def __init__(self, provider: str, model: str) -> None:
        self._model_id = f"{provider}:{model}"
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(model=model, temperature=0)
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._llm = ChatGoogleGenerativeAI(model=model, temperature=0)
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

    if provider not in _KEY_VAR:
        raise UnknownProvider(
            f"VVRAG_{role.upper()}_PROVIDER is {provider!r}; expected one of {sorted(_KEY_VAR)}"
        )
    key_var = _KEY_VAR[provider]
    if not os.getenv(key_var):
        raise MissingApiKey(f"{key_var} is not set, which the {role} needs to reach {provider}")
    return LangChainChat(provider, model)
