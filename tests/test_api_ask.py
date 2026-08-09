"""The service's own generator: search, prepare, then answer_stream."""

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from visual_verify.agent.events import AnswerComplete, ClaimVerified
from visual_verify.agent.schemas import ClaimList, Verdict
from visual_verify.agent.types import FakeChat
from visual_verify.api.ask import AskRequest, NoPagesIndexed, Retrieved, ask_events
from visual_verify.cli import _make_index, main
from visual_verify.config import Settings
from visual_verify.store.engine import make_engine


@pytest.fixture
def indexed(tmp_path, monkeypatch, born_digital_pdf):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VVRAG_QDRANT_URL", ":memory:")
    monkeypatch.delenv("VVRAG_QDRANT_API_KEY", raising=False)
    monkeypatch.setenv("VVRAG_FAKE_EMBEDDER", "1")
    assert main(["ingest", str(born_digital_pdf)]) == 0
    assert main(["embed", "--all"]) == 0
    return Settings.from_env()


def chats():
    reader = FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent"])])
    verifier = FakeChat("v", [Verdict(label="supported", confidence=0.9, reason="matches")])
    return reader, verifier


class NoSearchIndex:
    """Delegates everything except search, which is a hard failure.

    A test named "skips retrieval" that only inspects the emitted event proves
    nothing: an empty candidate list is equally consistent with a search that
    ran and had its results thrown away. This makes the skip observable.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def search(self, *args, **kwargs):
        raise AssertionError("index.search was called although doc and page were pinned")


def run(settings, request, wrap=lambda index: index):
    from visual_verify.retrieval.types import FakeEmbedder

    reader, verifier = chats()
    index = wrap(_make_index(settings))
    with Session(make_engine(settings.db_url)) as session:
        return list(
            ask_events(
                request,
                session=session,
                index=index,
                embedder=FakeEmbedder(),
                reader_chat=reader,
                verifier_chat=verifier,
                settings=settings,
            )
        )


def test_retrieval_runs_first_and_names_the_page_it_chose(indexed):
    events = run(indexed, AskRequest(question="What happened?"))

    assert isinstance(events[0], Retrieved)
    assert events[0].page.page_no == 0
    assert events[0].page.doc_name == "born_digital.pdf"


def test_the_answer_stream_follows_retrieval(indexed):
    events = run(indexed, AskRequest(question="What happened?"))

    assert any(isinstance(e, ClaimVerified) for e in events)
    assert isinstance(events[-1], AnswerComplete)


def test_an_explicit_page_skips_retrieval_and_reports_no_candidates(indexed):
    """Clicking a candidate re-asks with doc and page pinned. The event shape
    stays identical so the frontend has one code path."""
    events = run(indexed, AskRequest(question="q", doc="born_digital", page=0), NoSearchIndex)

    assert isinstance(events[0], Retrieved)
    assert events[0].candidates == []
    assert events[0].score is None
    assert isinstance(events[-1], AnswerComplete)


def test_an_unindexed_corpus_raises_rather_than_answering_from_nothing(
    tmp_path, monkeypatch, born_digital_pdf
):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VVRAG_QDRANT_URL", ":memory:")
    monkeypatch.setenv("VVRAG_FAKE_EMBEDDER", "1")
    assert main(["ingest", str(born_digital_pdf)]) == 0
    settings = Settings.from_env()

    with pytest.raises(NoPagesIndexed):
        run(settings, AskRequest(question="q"))


def test_a_page_without_a_doc_is_rejected():
    """Without this the request falls through to the retrieval branch, which
    ignores `page` entirely and reads whichever page ranked first. The caller
    asked for one page and would be shown another, with nothing in the response
    saying so."""
    with pytest.raises(ValidationError):
        AskRequest(question="q", page=0)


def test_a_page_with_a_doc_is_accepted():
    """Pins the field ORDER, not just the rule. Pydantic v2 populates
    `info.data` in declaration order, so the validator can only see `doc`
    because `doc` is declared above `page`. Moving it below would make the
    check above silently vacuous."""
    request = AskRequest(question="q", doc="d", page=0)

    assert request.doc == "d"
    assert request.page == 0


def test_a_doc_without_a_page_is_rejected_rather_than_silently_ignored():
    """It reads as "ask within this document". The retrieval branch would
    instead search the whole corpus and answer from whatever ranked first
    anywhere, while reporting success."""
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="doc requires page"):
        AskRequest(question="q", doc="born_digital")


def test_an_empty_corpus_raises_at_call_time_not_on_first_advance(
    tmp_path, monkeypatch, born_digital_pdf
):
    """The HTTP layer maps this to a 409. As a generator it would not run until
    StreamingResponse had already committed a 200, turning an operator mistake
    into an error frame inside a successful response."""
    from visual_verify.retrieval.types import FakeEmbedder

    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'k.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "datak"))
    monkeypatch.setenv("VVRAG_QDRANT_URL", ":memory:")
    monkeypatch.setenv("VVRAG_FAKE_EMBEDDER", "1")
    assert main(["ingest", str(born_digital_pdf)]) == 0
    settings = Settings.from_env()

    reader, verifier = chats()
    index = _make_index(settings)
    with Session(make_engine(settings.db_url)) as session:
        # No list(), no next(). The call itself must raise.
        with pytest.raises(NoPagesIndexed):
            ask_events(
                AskRequest(question="q"),
                session=session,
                index=index,
                embedder=FakeEmbedder(),
                reader_chat=reader,
                verifier_chat=verifier,
                settings=settings,
            )


def test_a_threshold_below_the_floor_is_refused():
    """Unbounded, any unauthenticated caller could POST threshold=-1 and make
    `score < threshold` False for every claim: nothing abstains, every claim
    passes Claim.withheld, and unsupported claims are displayed with their
    regions drawn on the page. That puts the one safety property in pillar 3
    under remote control. S7 sweeps the threshold in-process against answer(),
    so nothing is lost by refusing it here."""
    with pytest.raises(ValidationError):
        AskRequest(question="q", threshold=-1.0)


def test_a_threshold_above_the_ceiling_is_refused():
    from visual_verify.agent.rubric import SCORE_CEILING

    with pytest.raises(ValidationError):
        AskRequest(question="q", threshold=SCORE_CEILING + 0.1)


def test_the_whole_producible_range_is_still_accepted():
    """The bound is only correct if it admits every score abstention_score can
    actually return. A tighter one would make part of the rubric unreachable."""
    from visual_verify.agent.rubric import SCORE_CEILING

    assert AskRequest(question="q", threshold=0.0).threshold == 0.0
    assert AskRequest(question="q", threshold=SCORE_CEILING).threshold == SCORE_CEILING


def test_nan_is_still_refused_despite_the_bounds():
    """ge/le alone do NOT catch NaN: it compares False against both, so a
    bounded field would silently admit the one value that disables the gate.
    The finiteness validator is not redundant with the bounds."""
    with pytest.raises(ValidationError):
        AskRequest(question="q", threshold=float("nan"))
