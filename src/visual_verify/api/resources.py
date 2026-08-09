"""What the service holds for its whole lifetime, and what it refuses to start without.

Loading ColQwen2 costs about 20 seconds and 2.6 GB. Every `vvrag search`
invocation pays that; a request-scoped embedder would make the UI unusable and
would not fit alongside itself. So the models load once, here, and the first
question is fast at the cost of a slow boot.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from visual_verify.config import Settings

if TYPE_CHECKING:
    # Real types for the editor and the type checker, no imports at runtime.
    # Every one of these lives behind an extra (sqlalchemy, qdrant-client,
    # torch), and importing any of them at module scope would make
    # check_configuration untestable without the whole stack installed and
    # would put a 2.5 GB torch import on the path of a plain `import
    # visual_verify.api.resources`. Annotations are strings, so dataclass never
    # evaluates them.
    from sqlalchemy.engine import Engine

    from visual_verify.agent.types import StructuredChat
    from visual_verify.retrieval.index import QdrantIndex
    from visual_verify.retrieval.types import Embedder


class StartupRefused(RuntimeError):
    """The service will not come up in this configuration."""


def model_id_for(role: str, settings: Settings) -> str:
    """The id answer() compares, derived from settings without building a client.

    Must stay identical to LangChainChat's `f"{provider}:{model}"`. It is
    duplicated rather than imported because constructing a LangChainChat needs
    LangChain installed and an API key present, which is exactly what a
    configuration check must work without. test_api_resources.py pins the two
    together by building a real client and comparing, so a change to the id
    format fails there rather than silently splitting the startup check from
    the check it is supposed to anticipate.
    """
    if role == "reader":
        return f"{settings.reader_provider}:{settings.reader_model}"
    if role == "verifier":
        return f"{settings.verifier_provider}:{settings.verifier_model}"
    raise ValueError(f"role must be 'reader' or 'verifier', got {role!r}")


def check_configuration(settings: Settings) -> None:
    """Fail now, loudly, rather than on the first question.

    answer() carries the same reader-verifier check, but it only fires once a
    question has been asked. By then the service has reported itself healthy,
    the browser is open, and somebody is watching.
    """
    # First, because everything downstream needs it and because the tests below
    # it would otherwise be reached with a half-usable service.
    if not settings.qdrant_url:
        raise StartupRefused("VVRAG_QDRANT_URL is not set; the service cannot retrieve anything")

    reader = model_id_for("reader", settings)
    verifier = model_id_for("verifier", settings)
    if reader == verifier:
        raise StartupRefused(
            f"reader and verifier are the same model ({reader}); a model grading "
            "its own output is biased toward it, which is the reason this project "
            "uses two providers. Set VVRAG_VERIFIER_PROVIDER and "
            "VVRAG_VERIFIER_MODEL to something else."
        )


@dataclass
class Resources:
    """Held on app.state for the process lifetime."""

    settings: Settings
    engine: "Engine"
    index: "QdrantIndex"
    embedder: "Embedder"
    reader_chat: "StructuredChat"
    verifier_chat: "StructuredChat"


def build(settings: Settings) -> Resources:
    """Construct everything once. Raises StartupRefused on misconfiguration.

    Imports are function-local so that importing this module (which the tests
    do, to reach check_configuration) does not drag in torch or LangChain.
    """
    check_configuration(settings)

    from visual_verify.agent.cache import CachedChat
    from visual_verify.agent.models import MissingApiKey, UnknownProvider, make_chat
    from visual_verify.cli import _ensure_schema, _make_embedder, _make_index
    from visual_verify.store.engine import make_engine

    _ensure_schema(settings)
    engine = make_engine(settings.db_url)
    index = _make_index(settings)
    embedder = _make_embedder(settings)

    try:
        reader_chat = CachedChat(make_chat("reader", settings), settings.agent_cache_dir)
        verifier_chat = CachedChat(make_chat("verifier", settings), settings.agent_cache_dir)
    except (MissingApiKey, UnknownProvider) as exc:
        # Both name the environment variable to set. Re-raised as
        # StartupRefused so the lifespan has exactly one exception type to
        # catch and report, rather than letting an unhandled RuntimeError out
        # of a boot path.
        raise StartupRefused(str(exc)) from exc

    return Resources(
        settings=settings,
        engine=engine,
        index=index,
        embedder=embedder,
        reader_chat=reader_chat,
        verifier_chat=verifier_chat,
    )
