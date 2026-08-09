"""One question, start to finish: search, prepare, read, ground, verify.

No HTTP here on purpose. The whole sequence is plain objects and an iterator,
so it is testable without a client, a server, or a socket.

This is the first place the online pipeline exists end to end. The CLI splits
it: `vvrag search` ranks pages and stops, `vvrag ask` requires a page number.
proposal.tex lines 340 to 342 specify retrieve, read, ground, verify, and this
is where that becomes one call.

WHEN THE ERRORS ARRIVE. `ask_events` is NOT a generator. It validates, does the
retrieval, and constructs the answer stream, then returns a generator over the
result. So every failure below is raised at CALL time, before the caller has
written a status line:

- `NoPagesIndexed`  - empty corpus, or retrieval returned nothing.
- `PageNotFound`    - `doc` matched no document, or matched several, or the
                      page number does not exist. Only when the caller pinned a
                      page; the retrieval branch prepares a page the index just
                      returned.
- `AgentError`      - reader and verifier are the same model. `answer_stream`
                      is likewise an eager wrapper around its own generator, so
                      this surfaces here rather than mid-answer.

Written this way on purpose. As a generator, none of the above would run until
the first `next()`, which for an HTTP caller is after `StreamingResponse` has
committed a 200 and its headers. An empty corpus would then arrive as an SSE
error frame inside a successful response instead of a 409, and a misconfigured
verifier as an error frame instead of a refusal. Everything that can be known
before streaming is now known before streaming.

Nothing after the returned generator starts is knowable in advance: a provider
failure on claim two genuinely is mid-stream, and belongs in an error frame.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from visual_verify.agent.core import answer_stream
from visual_verify.agent.events import AnswerEvent
from visual_verify.agent.rubric import SCORE_CEILING
from visual_verify.agent.types import StructuredChat
from visual_verify.config import Settings
from visual_verify.contracts import RetrievedPage
from visual_verify.prepare import PreparedPage, prepare_page
from visual_verify.store.models import Document

DEFAULT_K = 5


class NoPagesIndexed(RuntimeError):
    """The corpus has documents but no embeddings."""


class AskRequest(BaseModel):
    """The POST body.

    `threshold` defaults to None and is resolved against Settings at use time,
    so the service honours VVRAG_ABSTAIN_THRESHOLD exactly as the CLI does and
    no literal is repeated here. S7 sweeps this to produce the confident-wrong
    against coverage curve, which is why it is a request field at all.
    """

    question: str = Field(min_length=1)
    doc: str | None = None
    page: int | None = Field(default=None, ge=0)
    k: int = Field(default=DEFAULT_K, ge=1, le=20)
    # Bounded to the range abstention_score can actually produce. Unbounded, any
    # unauthenticated caller could POST threshold=-1 and make `score < threshold`
    # False for every claim: nothing abstains, every claim passes
    # `Claim.withheld`, and unsupported claims are displayed with their regions
    # drawn on the page. That would put the one safety property in pillar 3 under
    # remote control. S7 sweeps the threshold in-process against answer(), so
    # nothing is lost by refusing values on the wire that no score can straddle.
    threshold: float | None = Field(default=None, ge=0.0, le=SCORE_CEILING)

    @field_validator("threshold")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        # Runs alongside the ge/le bounds above rather than instead of them. NaN
        # compares False against both, so a bounded field alone still admits it.
        if v is not None and not math.isfinite(v):
            raise ValueError("threshold must be a finite number")
        return v

    @field_validator("page")
    @classmethod
    def _page_needs_a_doc(cls, v: int | None, info) -> int | None:
        # Only works because `doc` is DECLARED above `page`: pydantic v2 runs
        # field validators in declaration order and info.data holds only the
        # fields already validated. Reordering the two fields would turn this
        # into a no-op that still looks like a guard, and a page without a doc
        # would then fall through to the retrieval branch and read some other
        # page while reporting success.
        if v is not None and not info.data.get("doc"):
            raise ValueError("page requires doc; supply both or neither")
        return v

    @model_validator(mode="after")
    def _doc_needs_a_page(self) -> "AskRequest":
        """`doc` alone is refused rather than quietly ignored.

        It reads as "ask within this document", and the retrieval branch would
        instead search the WHOLE corpus and answer from whatever ranked first
        anywhere, reporting success. QdrantIndex.search takes no doc filter, so
        scoping it is a real feature and not something to fake here. Refusing is
        the honest version until that exists.
        """
        if self.doc is not None and self.page is None:
            raise ValueError(
                "doc requires page; searching within a single document is not "
                "supported yet, so supply both or neither"
            )
        return self


UNEMBEDDED_WARNING = (
    "This page is not embedded, so grounding can only use the text layer. Any "
    "claim the reader paraphrases rather than quotes verbatim will come back as "
    "insufficient_evidence. Run `vvrag embed` on this document to fix it."
)


@dataclass(frozen=True)
class Retrieved:
    """Which page will be read, and what else was considered.

    `score` is None and `candidates` empty when the caller pinned the page, so
    the frontend gets one event shape either way.

    `warning` carries anything the user should see BEFORE the answer, because it
    explains an answer that will otherwise look like a verdict. It rides on this
    event rather than getting its own so the frontend has one place to read it,
    and it is sent before any model call so a user can stop rather than pay for
    one reader call and a verifier call per claim.
    """

    page: PreparedPage
    score: float | None
    candidates: list[RetrievedPage] = field(default_factory=list)
    warning: str | None = None
    # sha -> document name, for the candidates only. Retrieval is corpus-wide
    # and takes no document filter, so a candidate frequently belongs to a
    # DIFFERENT document than the one being displayed. Without the name, a chip
    # reading "page 24" is indistinguishable from page 24 of the document on
    # screen, and clicking it silently swaps the document under the user.
    doc_names: dict[str, str] = field(default_factory=dict)


AskEvent = Retrieved | AnswerEvent


def _choose_page(
    request: AskRequest,
    session: Session,
    index,
    embedder,
    settings: Settings,
) -> Retrieved:
    """Resolve the request to one prepared page, however it was addressed.

    Split out of ask_events so the page is bound by a return value rather than
    by both arms of an if/else agreeing to assign the same local.
    """
    if request.doc is not None and request.page is not None:
        prepared = prepare_page(session, index, settings, doc=request.doc, page_no=request.page)
        return Retrieved(page=prepared, score=None, candidates=[])

    hits = index.search(
        embedder.embed_query(request.question), embedder.provenance, limit=request.k
    )
    if not hits:
        raise NoPagesIndexed("retrieval returned no pages")
    top = hits[0]
    prepared = prepare_page(session, index, settings, doc=top.doc_id, page_no=top.page)
    candidates = list(hits[1:])
    return Retrieved(
        page=prepared,
        score=top.score,
        candidates=candidates,
        doc_names=_names_for({c.doc_id for c in candidates}, session),
    )


def _names_for(shas: set[str], session: Session) -> dict[str, str]:
    """Display names for the candidate documents, in one query.

    Cheap next to the GPU embed that produced the hits, and it is the only way
    the frontend can tell a candidate from another document apart from one in
    the document on screen.
    """
    if not shas:
        return {}
    rows = session.execute(
        select(Document.sha256, Document.path).where(Document.sha256.in_(shas))
    ).all()
    return {sha: Path(path).name for sha, path in rows}


def ask_events(
    request: AskRequest,
    *,
    session: Session,
    index,
    embedder,
    reader_chat: StructuredChat,
    verifier_chat: StructuredChat,
    settings: Settings,
) -> Iterator[AskEvent]:
    """A Retrieved event, then everything answer_stream yields.

    Not a generator. See the module docstring for why, and for which exceptions
    reach the caller before it commits a status line.
    """
    if index.count() == 0:
        raise NoPagesIndexed("no pages indexed; run `vvrag embed --all` first")

    retrieved = _choose_page(request, session, index, embedder, settings)
    prepared = retrieved.page

    # prepare_page returns page_vectors=None for a page that was ingested but
    # never embedded. Serving it text-only is right for a service, but it
    # degrades into a plausible wrong answer: ground() has no visual fallback, a
    # reader paraphrases by default, so most claims come back
    # insufficient_evidence and the UI reports "the verifier rejected every
    # claim", which reads as a judgement about the evidence when the truth is
    # that nobody ran `vvrag embed`. index.count() cannot catch this once any
    # other document is indexed. The CLI warns and skips the embedder; this is
    # the same handling on the surface a user actually sees.
    if prepared.page_vectors is None:
        retrieved = replace(retrieved, warning=UNEMBEDDED_WARNING)

    threshold = request.threshold if request.threshold is not None else settings.abstain_threshold
    stream = answer_stream(
        request.question,
        prepared.image_path,
        prepared.boxes,
        page=prepared.page_no,
        reader_chat=reader_chat,
        verifier_chat=verifier_chat,
        threshold=threshold,
        page_vectors=prepared.page_vectors,
        # None in that branch, not embedder.embed_query. answer_stream calls it
        # once per claim to build query vectors that ground() is structurally
        # guaranteed to discard when page_vectors is None, so passing it spends
        # a multi-second GPU call per claim to compute nothing.
        embed_query=embedder.embed_query if prepared.page_vectors is not None else None,
        grid=prepared.grid,
    )
    return _emit(retrieved, stream)


def _emit(retrieved: Retrieved, stream: Iterator[AnswerEvent]) -> Iterator[AskEvent]:
    """The generator half of ask_events(). Everything is already resolved."""
    yield retrieved
    yield from stream
