"""Frozen public contracts.

These are the seam every later slice (retrieval, grounding, reader, eval)
builds against. Changing them is a breaking change; adding optional fields is not.

Coordinate convention: every bbox is (x0, y0, x1, y1) NORMALIZED to 0-1 against
the displayed page rect, origin top-left. One coordinate system everywhere.
Pixel conversion happens only at the point of drawing.
"""

from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator

BBox = tuple[float, float, float, float]


class GroundedRegion(BaseModel):
    """A region of a page put forward as evidence for a claim."""

    page: int = Field(ge=0)
    bbox: BBox
    # Retrieval/grounding score. Deliberately unbounded: MaxSim sums a per-token
    # maximum across query tokens, so it is not a probability and can exceed 1.
    score: float
    modality: Literal["visual", "text"]
    crop_ref: str | None = None
    text: str | None = None
    # "block" means the heatmap could not separate the lines inside the winning
    # block and the region deliberately stayed coarse. Without this a coarse
    # fallback is indistinguishable from a confident line hit, so the UI cannot
    # flag it and the eval cannot report stage-1 and stage-2 failures apart.
    # None for text-path regions, which never go through snap_to_box.
    resolution: Literal["line", "block"] | None = None

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
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool = False
    # The verifier's rubric label, None until the verifier has run. Optional so
    # existing consumers are unaffected; see this file's docstring. Kept as the
    # label rather than reduced to `confidence` alone because the label is what
    # decides show-or-abstain, and the UI and the eval both need to say WHICH
    # verdict a claim received, not merely how strong it was.
    label: (
        Literal["supported", "partially_supported", "insufficient_evidence", "unsupported"] | None
    ) = None
    # Flagged by the reader's conjunction check: this claim appears to assert
    # more than one thing, so its single region can only evidence part of it.
    # Advisory, never a reason to drop the claim, because discarding it would
    # lose an answer. The eval counts these as decomposition failures.
    compound: bool = False
    # The verifier's one-sentence justification. None until the verifier has
    # run, like `label`. The product UI shows this for a claim it withholds:
    # a count alone tells a user nothing, and the region cannot be shown, so
    # the reason is the only thing left that explains the refusal. S7 puts it
    # in the eval output for the same purpose.
    reason: str | None = None

    @property
    def withheld(self) -> bool:
        """Whether this claim must not be displayed, and its regions not sent.

        The single statement of the rule. `Answer.shown` filters on it and the
        API's wire serialiser strips regions on it, so the display gate and the
        geometry strip cannot drift apart into two subtly different predicates
        guarding the same guarantee.

        True when the verifier rejected the claim, and ALSO when no verdict
        exists: a `Claim` that never reached the verifier defaults to
        `abstained=False`, which would otherwise read as passing. Absence of a
        verdict is not a passing verdict.
        """
        return self.label is None or self.abstained


class Answer(BaseModel):
    """The full response to a question."""

    question: str
    # EVERY claim, including the ones that failed verification. The evaluation
    # harness needs both to compute confident-wrong against coverage, so an
    # abstained claim is marked rather than removed.
    claims: list[Claim] = Field(default_factory=list)

    @computed_field
    @property
    def abstained_overall(self) -> bool:
        """Whether the system is declining to answer, rather than answering.

        Derived, not stored. It was a settable field that core.py filled in with
        `all(c.abstained for c in claims)`, which is a DIFFERENT predicate from
        the one `shown` filters on: `Claim.withheld` is also true for a claim
        that never reached the verifier, because absence of a verdict is not a
        passing verdict. So a set of unjudged claims produced nothing to show
        and no abstention either, which is a state that is neither an answer nor
        a refusal and has no rendering.

        Nothing on the answer_stream path could reach it, since every claim
        there gets a verdict, so `abstained` and `withheld` agreed by accident.
        That is not a property worth resting a safety guarantee on, and this is
        the same rule `Claim.withheld` follows: state it once, in the place both
        readers already look.

        Computed rather than validated on input so that no caller can record an
        abstention the claims contradict. It stays in model_dump() output, so
        the wire format is unchanged.
        """
        return not self.shown

    @property
    def shown(self) -> list["Claim"]:
        """Only the claims that passed verification. Use this to display.

        `claims` holds rejected claims too, so iterating it directly puts a
        claim the verifier refused in front of a user, with its regions, and
        withholding that is the entire point of the system. The guarantee would
        otherwise rest on every consumer remembering a boolean. This exists so
        the safe path is also the shortest one.

        The predicate lives on `Claim.withheld` rather than inline here,
        because the API strips a rejected claim's regions using the same rule
        and two hand-written copies of it would eventually disagree.
        """
        return [c for c in self.claims if not c.withheld]


class RetrievedPage(BaseModel):
    """A page returned by retrieval, before any grounding happens."""

    doc_id: str
    page: int = Field(ge=0)
    image_ref: str
    text_layer: str | None = None
    # Retrieval score; deliberately unbounded (not a probability). See
    # GroundedRegion.score.
    score: float
