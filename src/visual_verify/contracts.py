"""Frozen public contracts.

These are the seam every later slice (retrieval, grounding, reader, eval)
builds against. Changing them is a breaking change; adding optional fields is not.

Coordinate convention: every bbox is (x0, y0, x1, y1) NORMALIZED to 0-1 against
the displayed page rect, origin top-left. One coordinate system everywhere.
Pixel conversion happens only at the point of drawing.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

BBox = tuple[float, float, float, float]


class GroundedRegion(BaseModel):
    """A region of a page put forward as evidence for a claim."""

    page: int = Field(ge=0)
    bbox: BBox
    score: float
    modality: Literal["visual", "text"]
    crop_ref: str | None = None
    text: str | None = None

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v: BBox) -> BBox:
        x0, y0, x1, y1 = v
        if not all(0.0 <= c <= 1.0 for c in v):
            raise ValueError(f"bbox must be normalized to 0-1, got {v}")
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"bbox must have positive area with x0<x1 and y0<y1, got {v}")
        return v


class Claim(BaseModel):
    """One atomic assertion, with the regions offered as its evidence."""

    text: str
    regions: list[GroundedRegion] = Field(default_factory=list)
    confidence: float
    abstained: bool = False


class Answer(BaseModel):
    """The full response to a question."""

    question: str
    claims: list[Claim] = Field(default_factory=list)
    abstained_overall: bool = False


class RetrievedPage(BaseModel):
    """A page returned by retrieval, before any grounding happens."""

    doc_id: str
    page: int = Field(ge=0)
    image_ref: str
    text_layer: str | None = None
    score: float
