"""answer_stream(): reader, then grounding, then a different model's judgement.

answer() is the thin drain over it that returns a complete Answer.

Order is fixed by proposal.tex lines 340 to 342: retrieve, read, ground,
verify. Grounding runs per claim BETWEEN the reader and the verifier, which is
what gives the text path an exact string to search for.

The READER's output is never streamed. Showing a claim before the verifier has
judged it would display exactly what the system exists to withhold, and
retracting it visibly is worse than a pause. See spec section 9. Verified
claims, by contrast, are yielded one at a time by `answer_stream`, which is
safe precisely because each one has already been judged.
"""

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np

from visual_verify.agent.events import (
    AnswerComplete,
    AnswerEvent,
    ClaimsProduced,
    ClaimVerified,
    ReadingStarted,
)
from visual_verify.agent.reader import is_compound, read, shares_a_term
from visual_verify.agent.rubric import SUPPORTED_FLOOR, abstention_score
from visual_verify.agent.types import StructuredChat
from visual_verify.agent.verifier import verify
from visual_verify.contracts import Answer, Claim, GroundedRegion
from visual_verify.grounding import GroundingError, ground

if TYPE_CHECKING:
    # Type-checking only, and it must stay that way. visual_verify.prepare
    # imports SQLAlchemy, and tests/test_core_is_light.py asserts that
    # importing visual_verify.agent drags in no store dependency, so a
    # module-scope import here would fail the suite rather than this file.
    # Nothing at runtime needs the class: this module only reads attributes
    # off whatever it is handed.
    from visual_verify.prepare import PreparedPage

# The 'supported' floor. Derived from the rubric, not repeated as a literal,
# so this and Settings.abstain_threshold's default cannot drift apart.
DEFAULT_THRESHOLD = SUPPORTED_FLOOR


class AgentError(RuntimeError):
    """A configuration that would invalidate the verification."""


def _best_region(
    claim: str,
    pages: Sequence["PreparedPage"],
    query_vectors: np.ndarray | None,
) -> tuple["PreparedPage", list[GroundedRegion]]:
    """The strongest region for `claim` across every prepared page.

    Text first, everywhere, because an exact span match is not a score on the
    same scale as MaxSim and beats it categorically. A text region's `score`
    comes from span matching and a visual one's from a MaxSim mean over patch
    embeddings, so `max()` over the two together compares nothing: an
    arbitrarily confident visual region on page 1 would outrank the page where
    the claim is written out word for word, and the answer would cite a box
    that merely looks busy. Ties break by retrieval order, which is why `pages`
    must stay ranked and must not be sorted here.

    Only when NO page yields a text region do visual scores compete, and those
    ARE the same quantity across pages, so comparing them is sound.

    Returns (page, regions). Regions is empty when nothing was found, which is
    the same meaning ground() gives it. The page returned alongside is the one
    whose image must go to the verifier: verify() takes ONE image, and judging
    a page-3 region against the page-1 image asks it to check a box that is not
    in front of it, which is a fabricated citation reached by another route.
    """
    for page in pages:
        # force="text" cannot raise GroundingError. It returns before the
        # vector check, which is the only thing that raises, so a page with no
        # embeddings still gets its text layer searched here.
        found = ground(claim, page.boxes, page=page.page_no, force="text")
        if found:
            return page, found

    best_page: PreparedPage | None = None
    best_regions: list[GroundedRegion] = []
    best_score = 0.0
    for page in pages:
        try:
            # Unforced rather than force="visual". The text pass above already
            # came back empty for every page, so this repeats a search that
            # cannot succeed and then falls through to the heatmap; keeping it
            # unforced means this loop asks for "the best region ground() can
            # find", which is the same question the single-page caller asked
            # before pages became a list.
            regions = ground(
                claim,
                page.boxes,
                page=page.page_no,
                page_vectors=page.page_vectors,
                query_vectors=query_vectors,
                grid=page.grid,
            )
        except GroundingError:
            # Per page, never around the loop. This is raised when the visual
            # path has no vectors, which is exactly what a page that was
            # ingested but never embedded looks like. Letting it escape would
            # make one unembedded page discard the evidence sitting on the
            # other two, and the answer would report insufficient_evidence with
            # nothing anywhere saying why.
            continue
        if regions and (best_page is None or max(r.score for r in regions) > best_score):
            best_page, best_regions = page, regions
            best_score = max(r.score for r in regions)

    if best_page is not None:
        return best_page, best_regions
    # No evidence anywhere. The top page is returned so the verifier still has
    # an image to look at and can say insufficient_evidence about the page the
    # user is looking at, rather than about no page at all.
    return pages[0], []


def answer_stream(
    question: str,
    pages: Sequence["PreparedPage"],
    *,
    reader_chat: StructuredChat,
    verifier_chat: StructuredChat,
    threshold: float = DEFAULT_THRESHOLD,
    embed_query: Callable[[str], np.ndarray] | None = None,
) -> Iterator[AnswerEvent]:
    """The loop, yielding as each claim is judged. See answer() for the arguments.

    This exists so a consumer can show a verified claim while later claims are
    still being verified. It does NOT weaken the guarantee: a ClaimVerified
    event is yielded only after verify() has returned for that claim, so
    nothing leaves this function unjudged. Streaming the READER's output is a
    different thing and is still ruled out (S5 spec section 9).

    answer() is a drain over this. Keeping one loop is the point: the
    GroundingError recovery and the `score < threshold` comparison must not
    exist twice, because the copy a product hits is the one no test covers.

    NOT a generator itself. It validates, then returns one. That distinction is
    load-bearing: a generator body does not run until first advance, so with the
    guard inside it the AgentError would surface wherever the caller happened to
    start iterating. The API reaches this through ask_events(), which yields a
    `retrieved` event first, so the raise would land after a 200 and its headers
    were already committed and arrive at the browser as an SSE error frame
    rather than a refusal to start. Validate eagerly and the caller can still
    turn it into a status code.
    """
    if reader_chat.model_id == verifier_chat.model_id:
        raise AgentError(
            f"reader and verifier are the same model ({reader_chat.model_id}); "
            "a model grading its own output is biased toward it, which is the "
            "reason this slice uses two providers"
        )
    # Eager, for the same reason the model guard above is: with no page there
    # is nothing to read and nothing to ground against, and discovering that on
    # first advance would surface it as an SSE error frame after a 200 rather
    # than as a refusal to start.
    if not pages:
        raise AgentError("no pages were prepared; there is nothing to read")

    return _answer_events(
        question,
        pages,
        reader_chat=reader_chat,
        verifier_chat=verifier_chat,
        threshold=threshold,
        embed_query=embed_query,
    )


def _answer_events(
    question: str,
    pages: Sequence["PreparedPage"],
    *,
    reader_chat: StructuredChat,
    verifier_chat: StructuredChat,
    threshold: float,
    embed_query: Callable[[str], np.ndarray] | None,
) -> Iterator[AnswerEvent]:
    """The generator half of answer_stream(). Preconditions already checked."""
    yield ReadingStarted()
    # Every page, in retrieval order. The reader answers from what it can see,
    # so a page withheld here is evidence the answer can never reach.
    drafted = read(reader_chat, [p.image_path for p in pages], question)
    yield ClaimsProduced(n=len(drafted))

    claims: list[Claim] = []
    for index, draft in enumerate(drafted):
        text = draft.text
        # Once per claim, not once per page: the query vector depends only on
        # the claim, and embedding it again for each page would pay a
        # multi-second GPU call for an identical array.
        query_vectors = embed_query(text) if embed_query is not None else None
        # GroundingError is caught inside _best_region, per page. A reader
        # model paraphrases by default, so a claim that is not findable
        # verbatim and has no visual path to fall back on (no vectors on any
        # page) is the EXPECTED case, not an edge one. Losing every
        # already-verified claim, after paying for the API calls that produced
        # them, is worse than one claim arriving with no evidence: the verifier
        # still runs, empty regions in hand, and the rubric's
        # insufficient_evidence label makes the gap visible instead of raising
        # a traceback that discards the whole answer.
        #
        # This is NOT swallowing a configuration error. If grounding is
        # unusable for every claim (no vectors ever supplied and nothing in the
        # text layer matches), the visible result is an answer where every
        # claim is insufficiently evidenced, not a crash.
        source, regions = _best_region(text, pages, query_vectors)

        # Drop a region whose text names nothing the claim names. This is
        # applied HERE and not inside ground() on purpose: ground()'s contract
        # says an empty list means no evidence exists on the page and never
        # that the evidence looked weak, and a filter in there would also stop
        # S7's ablation separating the grounder's contribution from the
        # verifier's. This is the point where the pipeline decides what to
        # CITE, so it is the right place to refuse to cite a non-sequitur.
        #
        # Measured, and the reason this exists: the visual path returned the
        # page number, an 11 by 22 px box, as the evidence for three different
        # claims across two unrelated questions on proposal.pdf page 14, and
        # the verifier scored two of them supported at 0.90 and 0.95. It judges
        # whether the CLAIM is true, and the claims were true, so a fabricated
        # citation passes the abstention gate untouched. Nothing else in the
        # pipeline looks at whether the region is about the claim at all.
        #
        # Dropping rather than flagging, unlike is_compound and the other
        # advisory checks. Those keep a claim whose EVIDENCE is real and whose
        # phrasing is imperfect. Here the evidence itself is the thing that is
        # wrong, and a citation nobody should trust is worse than no citation:
        # with regions empty the verifier returns insufficient_evidence and the
        # gate withholds the claim, which is the honest outcome.
        cited = [r for r in regions if shares_a_term(text, r.text)]
        regions = cited

        # The winning page's image, never pages[0]'s. verify() takes ONE image
        # and is asked whether the listed boxes support the claim, so handing
        # it the top page while the region sits on page 3 asks it to check a
        # box that is not in the picture. It would answer anyway, from the
        # claim's own plausibility, and a citation nobody can see would come
        # back supported.
        verdict = verify(verifier_chat, source.image_path, text, regions)
        score = abstention_score(verdict.label, verdict.confidence)
        claim = Claim(
            text=text,
            regions=regions,
            confidence=verdict.confidence,
            label=verdict.label,
            reason=verdict.reason,
            # A claim with no region cannot be shown, whatever the verifier
            # said. The prompt asks it to answer insufficient_evidence when it
            # is handed no regions, and it complies INCONSISTENTLY: measured on
            # proposal.pdf page 14, one region-less claim came back
            # insufficient_evidence at 1.00 and another came back supported at
            # 0.90 in the same answer. An instruction a model follows most of
            # the time is not a guarantee, and this project's whole claim is
            # region-level evidence, so a sentence displayed with no region is
            # an answer with no evidence behind it.
            #
            # The verifier's raw verdict is preserved on `label` rather than
            # overwritten, because S7 computes confident-wrong against coverage
            # and needs what the judge actually said. Only the display gate
            # moves.
            abstained=score < threshold or not regions,
            compound=is_compound(text),
            starts_paragraph=draft.starts_paragraph,
        )
        claims.append(claim)
        yield ClaimVerified(index=index, claim=claim)

    yield AnswerComplete(
        # abstained_overall is derived on Answer from the same predicate
        # `shown` filters on, so it is not passed here. It used to be computed
        # as all(c.abstained ...), which disagrees with Claim.withheld for a
        # claim that never reached the verifier.
        answer=Answer(question=question, claims=claims)
    )


def answer(
    question: str,
    pages: Sequence["PreparedPage"],
    *,
    reader_chat: StructuredChat,
    verifier_chat: StructuredChat,
    threshold: float = DEFAULT_THRESHOLD,
    embed_query: Callable[[str], np.ndarray] | None = None,
) -> Answer:
    """Answer `question` from these pages, with every claim judged before it shows.

    `pages` is ranked, top hit first, and must stay that way: grounding breaks
    ties between pages by this order, so sorting it by page number would
    silently change which page a claim is cited to and the answer would still
    look perfectly ordinary. Passing `[one_page]` is the pinned-page case and
    is not a special path.

    The image path, boxes, vectors, and patch grid all ride on the PreparedPage
    they describe. They used to be five parallel arguments, which meant a
    caller could hand over page 4's boxes with page 7's grid and get boxes
    placed off-page with every shape and dtype still looking right.

    `threshold` is a parameter, not a constant, because S7 sweeps it to produce
    the confident-wrong against coverage curve. A hardcoded value would make the
    project's headline figure unproducible.

    `embed_query` rather than a precomputed vector: there is one reader-produced
    claim per grounding call, not one for the whole answer, so the query vector
    has to be recomputed per claim. The pages stay fixed across claims; only the
    query changes. The caller loads the embedder once and passes its bound
    `embed_query` method, so the model is loaded once per command, not once per
    claim.

    A thin drain over answer_stream(). The loop lives there so a streaming
    consumer and this one cannot diverge.
    """
    for event in answer_stream(
        question,
        pages,
        reader_chat=reader_chat,
        verifier_chat=verifier_chat,
        threshold=threshold,
        embed_query=embed_query,
    ):
        if isinstance(event, AnswerComplete):
            return event.answer
    # An internal invariant, not a configuration problem, so deliberately NOT
    # AgentError: the CLI catches that one and presents it as something the user
    # should go and fix, which this is not. Unreachable today. It stays because
    # answer() is annotated -> Answer, nothing here runs a type checker, and an
    # early return added to _answer_events later would otherwise make this
    # function return None to be discovered somewhere far away.
    raise RuntimeError("answer_stream ended without an AnswerComplete event")
