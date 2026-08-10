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
    ClaimsProduced,
    ClaimVerified,
    ReadingStarted,
)
from visual_verify.contracts import LEAD_INDEX, Claim, GroundedRegion


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
    # Claim.withheld, not a local re-derivation. Answer.shown filters on the
    # same property, so the display gate and this geometry strip are one rule
    # rather than two statements of it that can drift.
    withheld = c.withheld
    return {
        "index": index,
        "text": c.text,
        "label": c.label,
        "confidence": c.confidence,
        "reason": c.reason,
        "compound": c.compound,
        "withheld": withheld,
        "regions": [] if withheld else [_region(r) for r in c.regions],
        # Deliberately OUTSIDE the strip above, and sent for a withheld claim
        # too. The strip exists for geometry; a paragraph break is not geometry.
        # The frontend lays out every claim it receives, so a withheld claim
        # that reported the wrong break would move the paragraph onto the wrong
        # sentence. This line is where someone would be tempted to tidy it into
        # the `withheld` branch, and a test named for that failure guards it.
        "starts_paragraph": c.starts_paragraph,
        # Tells the browser the answer is ALREADY abstaining, before `done`
        # arrives. _answer_events yields one ClaimVerified per claim and only
        # then AnswerComplete, so a UI that waits for `done` to learn the lead
        # was withheld spends one verifier call per remaining claim, tens of
        # seconds on a serial GPU, presenting supporting detail under an
        # "Answer" heading and then retracting it. What gets retracted is the
        # assertion that these sentences answer the question, which is the one
        # thing the lead rule exists to prevent.
        #
        # Both halves are single-sourced: LEAD_INDEX is the same constant
        # Answer.lead_withheld indexes with, and `withheld` is the local already
        # bound from Claim.withheld above. This flag strictly implies
        # abstained_overall, so a client acting on it early can never announce a
        # refusal the eventual `done` frame contradicts.
        "abstains_answer": index == LEAD_INDEX and withheld,
    }


def to_frame(event) -> tuple[str, dict]:
    """(event name, payload) for one event."""
    # Function-local import: ask.py imports prepare.py, which imports
    # SQLAlchemy. At module scope that would drag the store into every import
    # of this module, and tests/test_api_wire.py would stop being a pure unit
    # test of the region strip.
    from visual_verify.api.ask import Retrieved

    if isinstance(event, Retrieved):
        return "retrieved", {
            "doc_sha": event.page.doc_sha,
            "doc_name": event.page.doc_name,
            "page": event.page.page_no,
            "score": event.score,
            "candidates": [
                {
                    "doc_sha": c.doc_id,
                    "page": c.page,
                    "score": c.score,
                    # Retrieval is corpus-wide, so a candidate is often in
                    # another document. A chip reading only "page 24" would
                    # look like page 24 of the document on screen.
                    "doc_name": event.doc_names.get(c.doc_id, c.doc_id[:12]),
                }
                for c in event.candidates
            ],
            # Sent before any model call, so a user can stop rather than pay for
            # a reader call plus a verifier call per claim to reach an answer
            # this sentence already explains.
            "warning": event.warning,
        }
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
