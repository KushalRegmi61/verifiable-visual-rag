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
    """Regions covering `claim` verbatim, or [] if it is not matched on the page.

    One region per line the match spans, never a single union. A union over a
    match that wraps across a line break encloses every word in between: on the
    two-line fixture it covers 5.7x the true ink area. span_boxes already
    splits correctly; this function must not re-join.

    An empty result is not proof the claim is absent from the page: a ligature
    or a hyphenated line break in the text layer can defeat exact matching even
    when the text is there, so `[]` means "not matched", not "absent". Separately,
    when the phrase occurs more than once on the page, only the first occurrence
    in reading order is returned and that ambiguity is not signalled in the
    result. See derive.span_boxes for both caveats in full.
    """
    return [
        GroundedRegion(
            page=page,
            bbox=(b.x0, b.y0, b.x1, b.y1),
            score=EXACT,
            modality="text",
            text=b.text,
        )
        for b in span_boxes(boxes, claim)
        # Mirrors rank_candidates's degenerate-box drop in snap.py. Ingest
        # normally prevents a zero-area span from reaching here, but this
        # seam also takes hand-built BoxRecords (S5, S7), and without this a
        # zero-area box would reach GroundedRegion and pydantic would raise
        # ValidationError, an exception type ground()'s contract never
        # mentions and cmd_ground does not catch.
        if b.x1 > b.x0 and b.y1 > b.y0
    ]
