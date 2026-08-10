"""The service's own generator: search, prepare, then answer_stream."""

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from helpers import claim_list
from visual_verify.agent.events import AnswerComplete, ClaimVerified
from visual_verify.agent.schemas import Verdict
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
    reader = FakeChat("r", [claim_list("Revenue grew 42 percent")])
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


class UnembeddedIndex:
    """A page that was ingested but never embedded.

    prepare_page returns page_vectors=None for it, which is right for a service
    (serve text-only rather than a 500) but degrades into a plausible wrong
    answer: ground() has no visual fallback, so most claims come back
    insufficient_evidence and the UI blames the verifier.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get_payload_or_none(self, doc_sha, page_no):
        return None


def test_an_unembedded_page_warns_before_any_model_call(indexed):
    """index.count() cannot catch this once any other document is indexed, and
    the CLI already warns. The service is the surface a user actually sees."""
    from visual_verify.api.ask import UNEMBEDDED_WARNING

    events = run(indexed, AskRequest(question="q"), wrap=UnembeddedIndex)

    retrieved = events[0]
    assert isinstance(retrieved, Retrieved)
    assert retrieved.page.page_vectors is None
    assert retrieved.warning == UNEMBEDDED_WARNING
    # First event, so a caller can stop before paying for a reader call and one
    # verifier call per claim.
    assert events.index(retrieved) == 0


def test_an_embedded_page_carries_no_warning(indexed):
    """Otherwise the banner would be permanent and stop meaning anything."""
    events = run(indexed, AskRequest(question="q"))

    assert events[0].warning is None


def test_an_unembedded_page_does_not_spend_a_gpu_call_per_claim(indexed):
    """answer_stream calls embed_query once per claim to build vectors that
    ground() is structurally guaranteed to discard when page_vectors is None.
    Passing the embedder there costs a multi-second GPU call per claim to
    compute nothing."""
    from visual_verify.retrieval.types import FakeEmbedder

    calls = []

    class CountingEmbedder(FakeEmbedder):
        def embed_query(self, text):
            calls.append(text)
            return super().embed_query(text)

    reader, verifier = chats()
    index = UnembeddedIndex(_make_index(indexed))
    with Session(make_engine(indexed.db_url)) as session:
        list(
            ask_events(
                AskRequest(question="q"),
                session=session,
                index=index,
                embedder=CountingEmbedder(),
                reader_chat=reader,
                verifier_chat=verifier,
                settings=indexed,
            )
        )

    # Exactly one: the retrieval query. None per claim.
    assert len(calls) == 1


def test_candidates_from_another_document_carry_its_name(indexed):
    """Retrieval is corpus-wide and QdrantIndex.search takes no document
    filter, so a candidate is frequently in a different document than the one
    being displayed. Found by opening the UI: a chip labelled "page 24" was
    page 24 of reference_proposal.pdf while the header read proposal.pdf, and
    clicking it swapped the document with no indication."""
    events = run(indexed, AskRequest(question="q"))

    retrieved = events[0]
    for cand in retrieved.candidates:
        assert cand.doc_id in retrieved.doc_names
        assert retrieved.doc_names[cand.doc_id].endswith(".pdf")


def test_a_pinned_request_needs_no_names(indexed):
    """The pinned branch reports no candidates, so it must not pay for the
    lookup either."""
    doc = run(indexed, AskRequest(question="q"))[0].page.doc_sha

    events = run(indexed, AskRequest(question="q", doc=doc, page=0), wrap=NoSearchIndex)

    assert events[0].candidates == []
    assert events[0].doc_names == {}
