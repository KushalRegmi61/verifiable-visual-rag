"""answer(): reader, then grounding, then a different model's judgement.

Order is fixed by proposal.tex lines 340 to 342: retrieve, read, ground,
verify. Grounding runs per claim BETWEEN the reader and the verifier, which is
what gives the text path an exact string to search for.

Nothing is streamed. Showing a claim before the verifier has judged it would
display exactly what the system exists to withhold, and retracting it visibly
is worse than a pause. See spec section 9.
"""

from pathlib import Path

import numpy as np

from visual_verify.agent.reader import read
from visual_verify.agent.rubric import abstention_score
from visual_verify.agent.types import StructuredChat
from visual_verify.agent.verifier import verify
from visual_verify.contracts import Answer, Claim
from visual_verify.grounding import ground
from visual_verify.ingest.boxes import BoxRecord
from visual_verify.retrieval.geometry import PatchGrid

# The 'supported' floor. Ranks are spaced by 2 against a confidence width of
# 1, so the bands do not touch and only a supported claim reaches 6.0.
DEFAULT_THRESHOLD = 6.0


class AgentError(RuntimeError):
    """A configuration that would invalidate the verification."""


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
    query_vectors: np.ndarray | None = None,
    grid: PatchGrid | None = None,
) -> Answer:
    """Answer `question` from one page, with every claim judged before it shows.

    `threshold` is a parameter, not a constant, because S7 sweeps it to produce
    the confident-wrong against coverage curve. A hardcoded value would make the
    project's headline figure unproducible.
    """
    if reader_chat.model_id == verifier_chat.model_id:
        raise AgentError(
            f"reader and verifier are the same model ({reader_chat.model_id}); "
            "a model grading its own output is biased toward it, which is the "
            "reason this slice uses two providers"
        )

    claims: list[Claim] = []
    for text in read(reader_chat, image_path, question):
        regions = ground(
            text,
            boxes,
            page=page,
            page_vectors=page_vectors,
            query_vectors=query_vectors,
            grid=grid,
        )
        verdict = verify(verifier_chat, image_path, text, regions)
        score = abstention_score(verdict.label, verdict.confidence)
        claims.append(
            Claim(
                text=text,
                regions=regions,
                confidence=verdict.confidence,
                label=verdict.label,
                abstained=score < threshold,
            )
        )

    return Answer(
        question=question,
        claims=claims,
        abstained_overall=not claims or all(c.abstained for c in claims),
    )
