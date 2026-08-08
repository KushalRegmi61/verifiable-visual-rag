"""The exact text path: the reliability floor.

Faithful by construction. The region is the union of the very word boxes whose
text matches the claim, so there is no ranking, no model, and nothing to be
wrong about beyond the match itself.

proposal.tex line 288 builds this before the visual path for exactly that
reason: if the visual path underperforms, the system still delivers a complete
and measured text-span citation system.
"""

from visual_verify.contracts import GroundedRegion
from visual_verify.derive import span_boxes
from visual_verify.ingest.boxes import BoxRecord

# An exact text-layer match is not a similarity, so it does not share a scale
# with a MaxSim score. 1.0 marks it as exact rather than as "very confident".
EXACT = 1.0


def text_regions(claim: str, boxes: list[BoxRecord], page: int) -> list[GroundedRegion]:
    """Regions covering `claim` verbatim, or [] if it is not on the page.

    One region per line the match spans, never a single union. A union over a
    match that wraps across a line break encloses every word in between: on the
    two-line fixture it covers 5.7x the true ink area. span_boxes already
    splits correctly; this function must not re-join.
    """
    if not boxes:
        return []
    return [
        GroundedRegion(
            page=page,
            bbox=(b.x0, b.y0, b.x1, b.y1),
            score=EXACT,
            modality="text",
            text=b.text,
        )
        for b in span_boxes(boxes, claim)
    ]
