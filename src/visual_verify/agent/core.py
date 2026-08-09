"""answer(): reader, then grounding, then a different model's judgement.

Order is fixed by proposal.tex lines 340 to 342: retrieve, read, ground,
verify. Grounding runs per claim BETWEEN the reader and the verifier, which is
what gives the text path an exact string to search for.

The READER's output is never streamed. Showing a claim before the verifier has
judged it would display exactly what the system exists to withhold, and
retracting it visibly is worse than a pause. See spec section 9. Verified
claims, by contrast, are yielded one at a time by `answer_stream`, which is
safe precisely because each one has already been judged.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from visual_verify.agent.events import (
    AnswerComplete,
    AnswerEvent,
    ClaimsProduced,
    ClaimVerified,
    ReadingStarted,
)
from visual_verify.agent.reader import is_compound, read
from visual_verify.agent.rubric import SUPPORTED_FLOOR, abstention_score
from visual_verify.agent.types import StructuredChat
from visual_verify.agent.verifier import verify
from visual_verify.contracts import Answer, Claim
from visual_verify.grounding import GroundingError, ground
from visual_verify.ingest.boxes import BoxRecord
from visual_verify.retrieval.geometry import PatchGrid

# The 'supported' floor. Derived from the rubric, not repeated as a literal,
# so this and Settings.abstain_threshold's default cannot drift apart.
DEFAULT_THRESHOLD = SUPPORTED_FLOOR


class AgentError(RuntimeError):
    """A configuration that would invalidate the verification."""


def answer_stream(
    question: str,
    image_path: Path,
    boxes: list[BoxRecord],
    *,
    page: int,
    reader_chat: StructuredChat,
    verifier_chat: StructuredChat,
    threshold: float = DEFAULT_THRESHOLD,
    page_vectors: np.ndarray | None = None,
    embed_query: Callable[[str], np.ndarray] | None = None,
    grid: PatchGrid | None = None,
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
    """
    if reader_chat.model_id == verifier_chat.model_id:
        raise AgentError(
            f"reader and verifier are the same model ({reader_chat.model_id}); "
            "a model grading its own output is biased toward it, which is the "
            "reason this slice uses two providers"
        )

    yield ReadingStarted()
    texts = read(reader_chat, image_path, question)
    yield ClaimsProduced(n=len(texts))

    claims: list[Claim] = []
    for index, text in enumerate(texts):
        query_vectors = embed_query(text) if embed_query is not None else None
        try:
            regions = ground(
                text,
                boxes,
                page=page,
                page_vectors=page_vectors,
                query_vectors=query_vectors,
                grid=grid,
            )
        except GroundingError:
            # A reader model paraphrases by default, so a claim that is not
            # findable verbatim and has no visual path to fall back on (no
            # vectors supplied) is the EXPECTED case, not an edge one. Losing
            # every already-verified claim, after paying for the API calls
            # that produced them, is worse than one claim arriving with no
            # evidence: the verifier still runs, empty regions in hand, and
            # the rubric's insufficient_evidence label makes the gap visible
            # instead of raising a traceback that discards the whole answer.
            #
            # This is NOT swallowing a configuration error. If grounding is
            # unusable for every claim (no vectors ever supplied and nothing
            # in the text layer matches), the visible result is an answer
            # where every claim is insufficiently evidenced, not a crash.
            regions = []
        verdict = verify(verifier_chat, image_path, text, regions)
        score = abstention_score(verdict.label, verdict.confidence)
        claim = Claim(
            text=text,
            regions=regions,
            confidence=verdict.confidence,
            label=verdict.label,
            reason=verdict.reason,
            abstained=score < threshold,
            compound=is_compound(text),
        )
        claims.append(claim)
        yield ClaimVerified(index=index, claim=claim)

    yield AnswerComplete(
        answer=Answer(
            question=question,
            claims=claims,
            abstained_overall=not claims or all(c.abstained for c in claims),
        )
    )


def answer(
    question: str,
    image_path: Path,
    boxes: list[BoxRecord],
    *,
    page: int,
    reader_chat: StructuredChat,
    verifier_chat: StructuredChat,
    threshold: float = DEFAULT_THRESHOLD,
    page_vectors: np.ndarray | None = None,
    embed_query: Callable[[str], np.ndarray] | None = None,
    grid: PatchGrid | None = None,
) -> Answer:
    """Answer `question` from one page, with every claim judged before it shows.

    `threshold` is a parameter, not a constant, because S7 sweeps it to produce
    the confident-wrong against coverage curve. A hardcoded value would make the
    project's headline figure unproducible.

    `embed_query` rather than a precomputed vector: there is one reader-produced
    claim per grounding call, not one for the whole answer, so the query vector
    has to be recomputed per claim. `page_vectors` and `grid` describe the page
    and stay fixed across claims; only the query changes. The caller loads the
    embedder once and passes its bound `embed_query` method, so the model is
    loaded once per command, not once per claim.

    A thin drain over answer_stream(). The loop lives there so a streaming
    consumer and this one cannot diverge; this signature and return type are
    unchanged, which is what keeps `vvrag ask` and S7's harness working.
    """
    for event in answer_stream(
        question,
        image_path,
        boxes,
        page=page,
        reader_chat=reader_chat,
        verifier_chat=verifier_chat,
        threshold=threshold,
        page_vectors=page_vectors,
        embed_query=embed_query,
        grid=grid,
    ):
        if isinstance(event, AnswerComplete):
            return event.answer
    raise AgentError("answer_stream ended without an AnswerComplete event")
