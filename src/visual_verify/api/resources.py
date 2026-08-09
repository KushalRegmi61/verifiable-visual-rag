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

    Delegates to agent.models.model_id, which LangChainChat also uses, so the
    two cannot drift. It used to re-spell `f"{provider}:{model}"` here, and that
    was survivable only while the format was one line: the openai_compatible
    provider derives its id from the endpoint host instead, and a second
    hand-written copy would have let a config pass startup that answer() then
    rejected on the first question.

    Importing agent.models is safe without LangChain installed, because that
    module's LangChain imports are all function-local.
    """
    from visual_verify.agent.models import model_id

    if role == "reader":
        return model_id(settings.reader_provider, settings.reader_model, settings.reader_base_url)
    if role == "verifier":
        return model_id(
            settings.verifier_provider, settings.verifier_model, settings.verifier_base_url
        )
    raise ValueError(f"role must be 'reader' or 'verifier', got {role!r}")


def check_configuration(settings: Settings) -> None:
    """Fail now, loudly, rather than on the first question.

    answer() carries the same reader-verifier check, but it only fires once a
    question has been asked. By then the service has reported itself healthy,
    the browser is open, and somebody is watching.
    """
    from visual_verify.agent.models import model_family
    from visual_verify.agent.rubric import SCORE_CEILING

    # First, because everything downstream needs it and because the tests below
    # it would otherwise be reached with a half-usable service.
    if not settings.qdrant_url:
        raise StartupRefused("VVRAG_QDRANT_URL is not set; the service cannot retrieve anything")

    # A threshold that no score can reach withholds every claim; one that every
    # score clears withholds none, which turns the abstention gate off while the
    # service still reports itself healthy. Settings.from_env already refuses a
    # non-finite value, so by here the comparison is meaningful.
    if not 0.0 <= settings.abstain_threshold <= SCORE_CEILING:
        raise StartupRefused(
            f"VVRAG_ABSTAIN_THRESHOLD is {settings.abstain_threshold}, outside the "
            f"0 to {SCORE_CEILING} range abstention_score can produce. Below 0 nothing "
            "is ever withheld and the abstention gate is off; above the ceiling every "
            "claim is withheld."
        )

    reader = model_id_for("reader", settings)
    verifier = model_id_for("verifier", settings)
    if reader == verifier:
        raise StartupRefused(
            f"reader and verifier are the same model ({reader}); a model grading "
            "its own output is biased toward it, which is the reason this project "
            "uses two providers. Set VVRAG_VERIFIER_PROVIDER and "
            "VVRAG_VERIFIER_MODEL to something else."
        )

    # Ids alone are not enough. They carry the endpoint host for the
    # openai_compatible provider, so `openai:gpt-4o` against a reader and
    # `openrouter.ai:openai/gpt-4o` against a verifier are two spellings of one
    # model that compare as different, and the check above waves them through
    # while /health displays two names that look independent.
    #
    # This is deliberately STRICTER than answer_stream's own guard, which has
    # only two clients and can compare nothing but their ids. A configuration
    # refused here would have run; the earlier failure is the point, and a
    # startup check that accepted more than the runtime would be the harmful
    # direction. See test_the_startup_check_compares_the_id_the_agent_compares.
    if model_family(settings.reader_model) == model_family(settings.verifier_model):
        raise StartupRefused(
            f"reader and verifier both resolve to the model "
            f"{model_family(settings.reader_model)!r} ({reader} and {verifier}); routing "
            "one of them through a gateway changes the endpoint, not the weights, so "
            "the model would still be grading its own output. Set VVRAG_VERIFIER_MODEL "
            "to a different model."
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
