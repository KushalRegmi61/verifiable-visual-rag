"""What answer_stream() yields.

Deliberately plain frozen dataclasses with no HTTP, no JSON, and no knowledge
of a transport. The API layer converts them; the eval layer ignores them.

Retrieval is NOT represented here. answer_stream() is handed the ranked pages
and never chooses them, so an event about which pages won belongs to the caller
that ran the search. Which of those pages a given CLAIM was cited to is a
different question, and it rides on the region inside ClaimVerified.
"""

from dataclasses import dataclass

from visual_verify.contracts import Answer, Claim


@dataclass(frozen=True)
class ReadingStarted:
    """The reader has been called. Nothing is known yet."""


@dataclass(frozen=True)
class ClaimsProduced:
    """The reader returned. `n` is how many verdicts are still coming."""

    n: int


@dataclass(frozen=True)
class ClaimVerified:
    """One claim, grounded and judged. `claim.label` is never None here."""

    index: int
    claim: Claim


@dataclass(frozen=True)
class AnswerComplete:
    """The last event, always. Carries the same Answer answer() returns."""

    answer: Answer


AnswerEvent = ReadingStarted | ClaimsProduced | ClaimVerified | AnswerComplete
