"""ground(): the public seam S5, S6, and S7 all consume.

Routing is deliberately simple and deliberately asymmetric. Text wins when it
can, because it is exact; the visual path is the fallback, and force="visual"
exists so the eval harness can measure it on questions the text path would
otherwise have answered.
"""

from typing import Literal

import numpy as np

from visual_verify.contracts import GroundedRegion
from visual_verify.grounding.heatmap import dense_relevance
from visual_verify.grounding.snap import Reduce, snap_to_box
from visual_verify.grounding.text_span import text_regions
from visual_verify.ingest.boxes import BoxRecord
from visual_verify.retrieval.geometry import PatchGrid


class GroundingError(RuntimeError):
    """Inputs that cannot produce a trustworthy region.

    Defined here rather than in __init__.py on purpose. Putting it in the
    package __init__ and importing it back from this module is a cycle that
    happens to work only because the class is defined above the re-export, so
    reordering two lines in __init__ would break it at import time.
    """


def ground(
    claim: str,
    boxes: list[BoxRecord],
    *,
    page: int,
    page_vectors: np.ndarray | None = None,
    query_vectors: np.ndarray | None = None,
    grid: PatchGrid | None = None,
    force: Literal["text", "visual"] | None = None,
    reduce: Reduce = "mean",
) -> list[GroundedRegion]:
    """Regions of `page` that support `claim`.

    An empty list means NO EVIDENCE EXISTS on this page: the claim is not in
    the text layer and the page has no candidate boxes, as on a scanned page.
    It never means the evidence looked weak. ground() applies no confidence
    threshold; proposal.tex line 381 puts abstention on the verifier's output,
    and a second threshold here would make the ablation unable to separate the
    two contributions.

    Vectors are passed in, never fetched, so this stays inside the core's four
    dependencies and is testable without Qdrant or a GPU.
    """
    if force != "visual":
        found = text_regions(claim, boxes, page)
        if found or force == "text":
            return found

    if page_vectors is None or query_vectors is None or grid is None:
        raise GroundingError(
            "the visual path needs page_vectors, query_vectors, and grid; "
            "returning no region here would be indistinguishable from "
            "'this page holds no evidence'"
        )
    if not boxes:
        return []

    relevance = dense_relevance(query_vectors, page_vectors, grid)
    selection = snap_to_box(relevance, grid, boxes, reduce)
    if selection is None:
        return []
    b = selection.box
    return [
        GroundedRegion(
            page=page,
            bbox=(b.x0, b.y0, b.x1, b.y1),
            score=selection.score,
            modality="visual",
            text=b.text or None,
        )
    ]
