"""Events to JSON-serializable wire payloads.

This module owns one guarantee: a claim the verifier rejected leaves this
process with no geometry attached. Not styled differently, not flagged for the
frontend to handle. Absent.

It is enforced here and not in answer() on purpose. S7 computes confident-wrong
against coverage, which needs the regions of rejected claims, so stripping them
in the core would break the evaluation rather than protect anybody. The right
place for the rule is where the data crosses out of the process.
"""

from visual_verify.agent.events import (
    AnswerComplete,
    AnswerEvent,
    ClaimsProduced,
    ClaimVerified,
    ReadingStarted,
)
from visual_verify.contracts import Claim, GroundedRegion


def _region(r: GroundedRegion) -> dict:
    return {
        "bbox": list(r.bbox),
        "score": r.score,
        "modality": r.modality,
        # Kept deliberately. "block" means the heatmap could not separate the
        # lines inside the winning block, and the overlay draws that dashed.
        # Without it, a coarse fallback and a confident line hit look the same.
        "resolution": r.resolution,
        "text": r.text,
    }


def _claim(index: int, c: Claim) -> dict:
    withheld = c.abstained or c.label is None
    return {
        "index": index,
        "text": c.text,
        "label": c.label,
        "confidence": c.confidence,
        "reason": c.reason,
        "compound": c.compound,
        "withheld": withheld,
        "regions": [] if withheld else [_region(r) for r in c.regions],
    }


def to_frame(event: AnswerEvent) -> tuple[str, dict]:
    """(event name, payload) for one event."""
    if isinstance(event, ReadingStarted):
        return "reading", {}
    if isinstance(event, ClaimsProduced):
        return "claims", {"n": event.n}
    if isinstance(event, ClaimVerified):
        return "claim", _claim(event.index, event.claim)
    if isinstance(event, AnswerComplete):
        answer = event.answer
        shown = len(answer.shown)
        return "done", {
            "shown": shown,
            "withheld": len(answer.claims) - shown,
            "abstained_overall": answer.abstained_overall,
        }
    raise TypeError(f"no wire mapping for {type(event).__name__}")
