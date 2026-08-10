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
                      page number does not exist. Usually a pinned request, but
                      the retrieval branch raises it too when Qdrant and SQLite
                      disagree: they are separate stores, so a point in the
                      index with no row behind it, left by a partially rolled
                      back ingest or a document deleted from the DB, is found by
                      prepare_page. The retrieval branch now prepares up to
                      DEFAULT_PAGES pages rather than one, so there is more of
                      that surface than there was. Either way it surfaces here,
                      eagerly, before a status line is written.
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
from visual_verify.prepare import PreparedPage, prepare_page, prepare_pages
from visual_verify.store.models import Document

DEFAULT_K = 5
# How many pages of the top hit's document are prepared and read. Smaller than
# DEFAULT_K on purpose: retrieval ranks k pages so there is something to choose
# between, but each prepared page costs about four queries, and the pages past
# the third rarely support a claim. The exact count varies: five for the first
# page of a document, because resolve_document's session.get is a real query;
# four for the second and third, because the identity map serves the same
# Document without a round trip; three for a page with no payload, because
# get_vectors sits inside the `if`.
DEFAULT_PAGES = 3


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


# Scoped to THE PAGE ON SCREEN, and worded that way deliberately. It used to
# say "grounding can only use the text layer", which stopped being true once an
# answer grounded against three pages: with this page unembedded and page 2
# embedded, grounding is actively using page 2's heatmap while the banner
# announces there is no visual path at all. The embedder gate in ask_events
# asks whether ANY prepared page has vectors, so the old copy contradicted the
# code three lines below it.
UNEMBEDDED_WARNING = (
    "The page shown is not embedded, so a claim grounded to it can only use its "
    "text layer. Any such claim the reader paraphrases rather than quotes "
    "verbatim will come back as insufficient_evidence. Run `vvrag embed` on this "
    "document to fix it."
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
    # Every page that was prepared, in retrieval order, all from `page`'s
    # document. `page` stays the top one because the wire event and the UI's
    # initial view are built from it; `pages` is what grounding searches, so a
    # claim can be cited to whichever page actually supports it.
    pages: list[PreparedPage] = field(default_factory=list)
    candidates: list[RetrievedPage] = field(default_factory=list)
    warning: str | None = None
    # sha -> document name, for the candidates only. Retrieval is corpus-wide
    # and takes no document filter, so a candidate frequently belongs to a
    # DIFFERENT document than the one being displayed. Without the name, a chip
    # reading "page 24" is indistinguishable from page 24 of the document on
    # screen, and clicking it silently swaps the document under the user.
    doc_names: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A Retrieved always has at least the page it names. Defaulting here
        # rather than at each construction site keeps `pages[0] is page` true
        # for every one of them, including the pinned branch and any caller that
        # only cares about the top page: an empty list would otherwise read to
        # anything iterating `pages` as "nothing was prepared", and grounding
        # would find no evidence on a page that is sitting right there.
        #
        # A supplied list is CHECKED, not trusted. Filling in only the empty
        # case would leave the invariant resting on both construction sites
        # happening to satisfy it, which is the convention this replaced:
        # Retrieved(page=a, pages=[b, a]) would construct happily, and the UI
        # would open on `a` while the answer's leading citation pointed at `b`.
        if not self.pages:
            object.__setattr__(self, "pages", [self.page])
        elif self.pages[0] is not self.page:
            raise ValueError("pages[0] must be the page this Retrieved names")


AskEvent = Retrieved | AnswerEvent


def _choose_pages(
    request: AskRequest,
    session: Session,
    index,
    embedder,
    settings: Settings,
) -> Retrieved:
    """Resolve the request to the pages that will be read, however it was
    addressed.

    Split out of ask_events so the pages are bound by a return value rather than
    by both arms of an if/else agreeing to assign the same local.

    A pinned request gets exactly the page it pinned. The caller named one page,
    usually by clicking a candidate chip, so widening it to the neighbours would
    answer a question the user did not ask.
    """
    if request.doc is not None and request.page is not None:
        prepared = prepare_page(session, index, settings, doc=request.doc, page_no=request.page)
        return Retrieved(page=prepared, score=None, pages=[prepared], candidates=[])

    hits = index.search(
        embedder.embed_query(request.question), embedder.provenance, limit=request.k
    )
    if not hits:
        raise NoPagesIndexed("retrieval returned no pages")
    top = hits[0]
    # Only the top hit's document. hits[1:] still reports every candidate,
    # including those from other documents, because the chips are a way to
    # re-ask elsewhere; they are just not read into this answer.
    pages = prepare_pages(session, index, settings, hits, limit=DEFAULT_PAGES)
    candidates = list(hits[1:])
    return Retrieved(
        page=pages[0],
        score=top.score,
        pages=pages,
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

    retrieved = _choose_pages(request, session, index, embedder, settings)

    # prepare_page returns page_vectors=None for a page that was ingested but
    # never embedded. Serving it text-only is right for a service, but it
    # degrades into a plausible wrong answer: ground() has no visual fallback, a
    # reader paraphrases by default, so most claims come back
    # insufficient_evidence and the UI reports "the verifier rejected every
    # claim", which reads as a judgement about the evidence when the truth is
    # that nobody ran `vvrag embed`. index.count() cannot catch this once any
    # other document is indexed. The CLI warns and skips the embedder; this is
    # the same handling on the surface a user actually sees.
    #
    # The TOP page only. It is the page on screen and the one the advice is
    # about, and test_the_warning_follows_the_top_page_not_any_prepared_page
    # pins that: firing because some other prepared page lacks vectors would
    # tell a user to embed a document whose visible page is embedded.
    #
    # The old defence for it, "a document is embedded or it is not", was wrong
    # and is deliberately not repeated: `vvrag embed` runs page by page and can
    # be interrupted, which is exactly why answer_stream catches GroundingError
    # per page. What changed instead is the WORDING, because the copy, not the
    # predicate, was what contradicted the embedder gate below.
    #
    # KNOWN GAP, recorded rather than fixed here. A run whose top page is
    # embedded and whose second and third are not stays silent, and a claim
    # grounded to one of those pages still comes back insufficient_evidence
    # with nothing explaining why. Closing it means the banner naming the
    # pages, which is a copy and UI decision, not a one-line predicate swap.
    if retrieved.page.page_vectors is None:
        retrieved = replace(retrieved, warning=UNEMBEDDED_WARNING)

    threshold = request.threshold if request.threshold is not None else settings.abstain_threshold
    stream = answer_stream(
        request.question,
        # Every prepared page, in retrieval order. The reader sees all of them
        # and grounding searches all of them, so a claim is cited to whichever
        # page actually supports it rather than to whichever page ranked first.
        retrieved.pages,
        reader_chat=reader_chat,
        verifier_chat=verifier_chat,
        threshold=threshold,
        # None in that branch, not embedder.embed_query. answer_stream calls it
        # once per claim to build query vectors that ground() is structurally
        # guaranteed to discard when no page has vectors, so passing it spends
        # a multi-second GPU call per claim to compute nothing.
        embed_query=(
            embedder.embed_query
            if any(p.page_vectors is not None for p in retrieved.pages)
            else None
        ),
    )
    return _emit(retrieved, stream)


def _emit(retrieved: Retrieved, stream: Iterator[AnswerEvent]) -> Iterator[AskEvent]:
    """The generator half of ask_events(). Everything is already resolved."""
    yield retrieved
    yield from stream
