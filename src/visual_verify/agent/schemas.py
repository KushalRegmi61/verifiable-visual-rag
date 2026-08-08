"""What the models are required to return.

Schema-validated on the way in, so a malformed response raises instead of
parsing into something plausible. That matters more here than usual: a silently
mis-parsed claim list is exactly the correctly-shaped wrong output this
repository keeps getting caught by.
"""

from pydantic import BaseModel, Field, field_validator

from visual_verify.agent.rubric import Label


class ClaimList(BaseModel):
    """The reader's output: atomic claims, one assertion each."""

    claims: list[str] = Field(default_factory=list)

    @field_validator("claims")
    @classmethod
    def _no_blank_claims(cls, v: list[str]) -> list[str]:
        blank = [i for i, c in enumerate(v) if not c.strip()]
        if blank:
            raise ValueError(f"claims at {blank} are blank; the reader returned junk")
        return v


class Verdict(BaseModel):
    """The verifier's output for one claim."""

    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    # Not decoration. A verdict with no stated reason cannot be debugged after
    # the fact, and this string goes into the eval output.
    reason: str = Field(min_length=1)
