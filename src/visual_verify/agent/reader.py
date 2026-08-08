"""The reader: page image plus question, out come atomic claims.

Claims are emitted directly as structured output rather than as prose that a
second call splits. One API call instead of two, and the model that wrote the
answer is the one deciding where it separates.

There is no separate prose answer. The displayed answer is the claims joined,
so nothing can drift between what is shown and what is verified.
"""

import re
from pathlib import Path

from visual_verify.agent.schemas import ClaimList
from visual_verify.agent.types import StructuredChat

PROMPT = """You are reading one page of a document to answer a question.

Answer ONLY from what is visible on this page. If the page does not answer the
question, return an empty list of claims.

Break your answer into atomic claims. Each claim must assert exactly ONE thing,
because each claim will be matched to a single region of the page as its
evidence. A claim asserting two things cannot be evidenced by one region.

Question: {question}"""

# A clause-joining conjunction: " and " followed by something with its own verb.
# Deliberately conservative. "Revenue and margin both rose" is ONE assertion
# about two subjects, and flagging it would report a decomposition failure that
# did not happen.
_COMPOUND = re.compile(
    r"\b(?:and|but|while|whereas)\b\s+\w+\s+"
    r"(?:is|are|was|were|has|have|had|grew|fell|rose|held|remained|increased|decreased)\b",
    re.IGNORECASE,
)


def is_compound(claim: str) -> bool:
    """Whether a claim appears to assert more than one thing.

    The schema cannot enforce atomicity, and the roadmap requires that a
    sentence asserting two things is not grounded to one region. Flagged, not
    rejected: dropping the claim would lose an answer, and the useful response
    is to surface it in the eval as a decomposition failure.
    """
    return bool(_COMPOUND.search(claim))


def read(chat: StructuredChat, image_path: Path, question: str) -> list[str]:
    """Atomic claims answering `question` from the page at `image_path`."""
    out = chat.structured(PROMPT.format(question=question), image_path, ClaimList)
    return list(out.claims)
