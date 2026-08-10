"""What the models are required to return.

Schema-validated on the way in, so a malformed response raises instead of
parsing into something plausible. That matters more here than usual: a silently
mis-parsed claim list is exactly the correctly-shaped wrong output this
repository keeps getting caught by.
"""

from pydantic import BaseModel, Field, field_validator

from visual_verify.agent.rubric import Label


class DraftedClaim(BaseModel):
    """One sentence of the drafted answer.

    `starts_paragraph` is metadata, not text. It adds nothing unverified to the
    screen and leaves every sentence mapped to exactly one region, which is
    what keeps hover-to-region and click-to-evidence working. It exists because
    an answer long enough to cover two topics reads badly as one block.
    """

    text: str
    starts_paragraph: bool = False

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        """Stated on the type, not on the list that usually holds it.

        A blank claim grounds to nothing and verifies as insufficient, so it
        costs an API call to learn the model returned junk. The rule used to
        live on ClaimList, which was enough while ClaimList was the only way to
        build a claim. read() now hands DraftedClaim out directly, so helpers
        and eval fixtures can construct one, and a rule on the container would
        no longer cover them.
        """
        if not v.strip():
            raise ValueError("claim text is blank; the reader returned junk")
        return v


class ClaimList(BaseModel):
    """The reader's output: the drafted answer, one sentence per claim."""

    claims: list[DraftedClaim] = Field(default_factory=list)

    @field_validator("claims", mode="before")
    @classmethod
    def _accept_bare_strings(cls, v: object) -> object:
        """A plain string becomes a claim that starts no paragraph.

        It exists for size: around forty construction sites across the test
        suite pass strings, and rewriting all of them to dictionaries would be
        a large diff that tests nothing.

        It is not a reassurance about live providers. When this branch fires
        against a real model, the reader ignored the schema and returned bare
        strings, every claim silently takes starts_paragraph False, and the
        answer renders as one paragraph forever with no error, no warning, and
        no failing test. That is harmless only while nothing asks the model for
        the field. Once the prompt requests it, this branch firing means the
        paragraph feature is inert, and the coercion is what hides it.
        """
        if isinstance(v, list):
            return [{"text": c} if isinstance(c, str) else c for c in v]
        return v


class Verdict(BaseModel):
    """The verifier's output for one claim."""

    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    # Not decoration. A verdict with no stated reason cannot be debugged after
    # the fact, and this string goes into the eval output.
    reason: str = Field(min_length=1)
