"""The reader: page image plus question, out come atomic claims.

Claims are emitted directly as structured output rather than as prose that a
second call splits. One API call instead of two, and the model that wrote the
answer is the one deciding where it separates.

There is no separate prose answer. The displayed answer is the claims joined,
so nothing can drift between what is shown and what is verified.
"""

import re
from pathlib import Path

from visual_verify.agent.schemas import ClaimList, DraftedClaim
from visual_verify.agent.types import StructuredChat

PROMPT = """You are reading one page of a document to answer a question.

Answer ONLY from what is visible on this page. If the page does not answer the
question, return an empty list of claims.

Break your answer into atomic claims. Each claim must assert exactly ONE thing,
because each claim will be matched to a single region of the page as its
evidence. A claim asserting two things cannot be evidenced by one region.

Question: {question}"""

# A joiner that MIGHT be stitching two clauses together: a conjunction or a
# semicolon. Matching the joiner alone says nothing; "Revenue and margin rose"
# has one too and is a single assertion with a compound subject.
_JOINER = re.compile(r"\b(?:and|but|while|whereas)\b|;", re.IGNORECASE)

# A verb, widened past the original 13-word list to cover ordinary report
# register: totals, positions, and trend language ("stood at", "posted",
# "indicates", "shows", "improved", "declined", "climbed", "dropped",
# "rising", "falling") in addition to the original set.
_VERB = re.compile(
    r"\b(?:is|are|was|were|has|have|had|grew|fell|rose|held|remained|increased|decreased|"
    r"totalled|totaled|stood|posted|indicates|indicated|shows|showed|improved|declined|"
    r"climbed|dropped|rising|falling|gained|lost|reached|expanded|contracted)\b",
    re.IGNORECASE,
)


def is_compound(claim: str) -> bool:
    """Whether a claim appears to assert more than one thing.

    The schema cannot enforce atomicity, and the roadmap requires that a
    sentence asserting two things is not grounded to one region. Flagged, not
    rejected: dropping the claim would lose an answer, and the useful response
    is to surface it in the eval as a decomposition failure.

    The test is structural, not "is there a verb near a conjunction": a
    compound SUBJECT ("Revenue and margin rose") has no verb before the
    joiner, only after, and is one assertion. Joined CLAUSES ("Revenue grew
    and margins held steady") have a verb on both sides of the joiner,
    however many words sit between the joiner and either verb, and are two
    assertions. So this flags only when a verb from `_VERB` appears both
    before and after the first joiner (a conjunction or a semicolon).

    This is a heuristic, not a parser, and it is honest about what it misses:
    it only ever looks at the FIRST joiner, so a claim with a compound
    subject followed by a genuinely joined clause later in the sentence can
    still slip through. It only recognizes the verbs listed in `_VERB`, so a
    joined clause using a verb outside that list is not caught. It has no
    notion of comma splices, "however", "therefore", or any joiner other than
    the conjunctions and the semicolon above. Expect it to catch the common
    joined-clause shapes and to stay quiet on compound subjects; do not read
    "not flagged" as "verified atomic."
    """
    match = _JOINER.search(claim)
    if not match:
        return False
    before, after = claim[: match.start()], claim[match.end() :]
    return bool(_VERB.search(before) and _VERB.search(after))


# Sentence-initial only. A pronoun in the MIDDLE of a sentence resolves within
# that sentence and is fine; it is the opening that a reader resolves against
# whatever came before, and whatever came before may have been withheld.
_ANAPHORA = re.compile(
    r"^\s*(?:it|its|they|them|their|this|these|those|such|"
    r"additionally|also|furthermore|moreover|however)\b",
    re.IGNORECASE,
)


def opens_with_anaphora(claim: str) -> bool:
    """Whether this claim depends on the sentence before it to make sense.

    Verification runs after drafting, so any claim may be removed before the
    answer is shown. A claim opening with "It" or "Each of them" is meaningless
    once its predecessor is withheld, and it does not announce that it is
    broken: it stays grammatical and quietly says something other than what was
    verified.

    Flagged, never rejected. The same rule `is_compound` follows: dropping the
    claim would lose verified content with a real region behind it, and the
    useful response is to surface the rate in the eval and fix the prompt.

    `\\b` matters. A prefix match would flag "Itemised costs appear in Table 2",
    where "It" is the first two letters of a word and nothing is dangling.
    """
    return bool(_ANAPHORA.match(claim))


def read(chat: StructuredChat, image_path: Path, question: str) -> list[DraftedClaim]:
    """The drafted answer for `question`, one sentence per claim."""
    out = chat.structured(PROMPT.format(question=question), image_path, ClaimList)
    return list(out.claims)
