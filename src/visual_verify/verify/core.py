"""verify(): the read-ground-judge-gate pipeline over one page.

Everything except the two model calls is pure and tested with fakes.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
from PIL import Image

from visual_verify.contracts import Answer, Claim, GroundedRegion
from visual_verify.grounding import GroundingError, ground
from visual_verify.ingest.boxes import BoxRecord
from visual_verify.retrieval.geometry import PatchGrid
from visual_verify.verify.backends import Reader, Verifier
from visual_verify.verify.claims import ReaderOutput
from visual_verify.verify.errors import VerifierError
from visual_verify.verify.evidence import Evidence, best_region, build_evidence
from visual_verify.verify.rubric import is_answered, sufficiency


def verify(
    question: str,
    reader: Reader,
    verifier: Verifier,
    *,
    page: int,
    image: Image.Image | None,
    text_layer: str | None,
    boxes: list[BoxRecord],
    embed: Callable[[str], np.ndarray] | None = None,
    page_vectors: np.ndarray | None = None,
    grid: PatchGrid | None = None,
    force: Literal["text", "visual"] | None = None,
    threshold: float = 0.5,
) -> Answer:
    """Answer `question` from one page, per claim, with abstention."""
    output = _read(reader, question, image, text_layer)
    if not output.claims:
        return Answer(question=question, claims=[], abstained_overall=True)

    claims: list[Claim] = []
    for text in output.claims:
        regions = _ground_claim(
            text, boxes, page=page, embed=embed, page_vectors=page_vectors,
            grid=grid, force=force,
        )
        region = best_region(regions)
        evidence = build_evidence(region, image) if region is not None else Evidence()
        label = _judge(verifier, text, evidence)
        score = sufficiency(label)
        claims.append(
            Claim(
                text=text,
                regions=regions,
                confidence=score,
                abstained=not is_answered(label, threshold),
            )
        )
    return Answer(
        question=question,
        claims=claims,
        abstained_overall=all(c.abstained for c in claims),
    )


def _read(
    reader: Reader, question: str, image: Image.Image | None, text_layer: str | None
) -> ReaderOutput:
    try:
        return reader.read(question, image, text_layer)
    except VerifierError:
        raise
    except Exception as exc:
        raise VerifierError(f"reader failed: {exc}") from exc


def _judge(verifier: Verifier, claim: str, evidence: Evidence) -> str:
    try:
        return verifier.judge(claim, evidence).label
    except VerifierError:
        raise
    except Exception as exc:
        raise VerifierError(f"verifier failed: {exc}") from exc


def _ground_claim(
    claim: str,
    boxes: list[BoxRecord],
    *,
    page: int,
    embed: Callable[[str], np.ndarray] | None,
    page_vectors: np.ndarray | None,
    grid: PatchGrid | None,
    force: Literal["text", "visual"] | None,
) -> list[GroundedRegion]:
    if force != "visual":
        try:
            regions = ground(claim, boxes, page=page, force=force)
        except GroundingError:
            regions = []
        if regions or force == "text":
            return regions
        if not boxes:
            return []

    if embed is None or page_vectors is None or grid is None:
        if not boxes:
            return []
        raise VerifierError(
            "the visual path needs embed, page_vectors, and grid; "
            "without them this claim cannot be grounded"
        )
    return ground(
        claim,
        boxes,
        page=page,
        page_vectors=page_vectors,
        query_vectors=embed(claim),
        grid=grid,
        force="visual",
    )
