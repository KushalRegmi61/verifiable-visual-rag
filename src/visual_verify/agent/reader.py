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
#
# The demonstrative group (this/these/those/that) carries a negative lookahead
# that exempts it before a page-deictic noun. "This page", "this table", "this
# figure" point at the image the reader is looking at, not at the previous
# sentence, and they survive their predecessor being withheld completely
# intact. Only a demonstrative that is NOT pointing at the page dangles.
_PAGE_DEICTIC = r"page|document|slide|figure|table|chart|section|paragraph"

_ANAPHORA = re.compile(
    r"^\s*(?:it|its|they|them|their|such|"
    r"additionally|also|furthermore|moreover|however|"
    r"therefore|hence|consequently|instead|"
    rf"(?:this|these|those|that)(?!\s+(?:{_PAGE_DEICTIC})\b)"
    r")\b",
    re.IGNORECASE,
)


def opens_with_anaphora(claim: str) -> bool:
    """Whether this claim depends on the sentence before it to make sense.

    Verification runs after drafting, so any claim may be removed before the
    answer is shown. A claim opening with "It" or "However" is meaningless
    once its predecessor is withheld, and it does not announce that it is
    broken: it stays grammatical and quietly says something other than what was
    verified.

    Flagged, never rejected. The same rule `is_compound` follows: dropping the
    claim would lose verified content with a real region behind it, and the
    useful response is to surface the rate in the eval and fix the prompt.

    `\\b` matters. A prefix match would flag "Itemised costs appear in Table 2",
    where "It" is the first two letters of a word and nothing is dangling.

    A demonstrative before a page noun ("This page", "this table", "this
    figure") is exempt by design: it resolves against the image the reader is
    looking at rather than against the predecessor, so it survives the
    predecessor being withheld intact. Of the exempted nouns, `page`,
    `document`, and `slide` name the one artifact in view, so the deictic
    reading is certain. `figure`, `table`, and `chart` name visually bounded
    objects a reader can point at, so deixis still dominates. `section` and
    `paragraph` are not visually bounded on a page image; "This section
    defines three variants" has usually taken "section" from something named
    in text, which is the predecessor, so those two are the members of the
    list where the deictic reading is least certain and most likely wrong.

    This is a heuristic, not a parser, and it is honest about what it misses,
    in both directions. Bare quantifier openers ("Both are scored", "Each of
    them") are not flagged, on purpose rather than by oversight: the shape
    splits. "Both are scored on three metrics" dangles because "both" has no
    noun of its own and resolves against the previous sentence, but "Both
    variants are scored" does not dangle at all, because "variants" supplies
    its own referent. A flat word list cannot tell those two apart, so
    quantifiers are left out rather than flagged wrong half the time.
    Correlatives ("The former", "The latter", "Doing so") are missed for the
    same reason: catching them needs more than a word list. `there` is
    deliberately EXCLUDED, not missed: it is an expletive subject, not a
    referring pronoun, so "There are three variants" is fully self-contained
    and flagging it would be a pure false positive.

    Two known FALSE POSITIVES, kept rather than chased. `that` is distal and
    points backward by default, unlike proximal `this`, so "That figure
    appears in Table 2" is more likely anaphoric than "This figure appears in
    Table 2" and the identical exemption given to `that` is the least
    defensible part of the lookahead. Sentence-initial `that` as a
    complementiser or free-relative head is also wrongly flagged: "That the
    ablation improves recall is clear." and "That which is verified is shown."
    are both self-contained and both return True. No cheap regex distinguishes
    a complementiser from a demonstrative pronoun, and the drafting prompt
    makes either shape unlikely from a VLM, so this is recorded rather than
    fixed.

    As with `is_compound`, do not read "not flagged" as "self-contained," and
    do not read "flagged" as "confirmed dangling."
    """
    return bool(_ANAPHORA.match(claim))


# Closed class only. A long stopword list would start removing the words that
# actually carry the chain; these are the ones that appear in essentially every
# sentence and therefore signal nothing about whether two claims are connected.
# Kept consistent with `_ANAPHORA`: a word that function already treats as a
# contentless referring word (it, its, they, them, their, such) belongs here
# too, since otherwise the same file would call it contentless in one place
# and treat it as chaining evidence in the next.
_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from
    has have he him his however if in is it its not of on one only or other
    she such than that the their them then there these they this those to
    was we were which while who whom will with would you your
    all any both each""".split()
)

# A leading letter, then at least one more letter or digit: excludes bare
# numbers ("3", "5.1" tokenises to "5" and "1") and the lone "s" a possessive
# leaves behind ("document's" -> "document", "s"), none of which name
# anything shared between two claims. Acronyms and mixed tokens survive
# ("em", "f1", "iou", "slidevqa").
_WORD = re.compile(r"[a-z][a-z0-9]+")


def _content_words(text: str) -> set[str]:
    """Lowercased words with stopwords dropped and a trailing s stripped.

    The stopword filter runs BEFORE the trailing s is stripped, on purpose.
    "does" is not itself in `_STOPWORDS`, only "do"; if stripping ran first
    "does" would already be "doe" by the time the filter saw it, and "doe" is
    not a word the filter will ever recognize. Filtering the raw token first
    means the list only has to name the word people actually type, not every
    stem an inflection could turn it into.

    Strips at most ONE trailing s, not `str.rstrip("s")`'s "all of them":
    rstrip removes the whole trailing run, so a word ending in a doubled s
    ("caress") loses both and collapses to "care", the same stem an unrelated
    word ("cares") also reduces to. That is a false match between two words
    that share no meaning, not a rounding error, so a single conditional
    slice is used instead. It is still a heuristic, not a stemmer: "class" and
    "process" become "clas" and "proces", which do not equal any inflected
    form of themselves either way, and a genuine "-es" plural ("processes")
    is not reunited with its singular. Good enough for the common case this
    exists to catch (a bare final s marking a plural), and honest that it is
    not morphology.
    """
    words = [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]
    return {(w[:-1] if w.endswith("s") and len(w) > 1 else w) for w in words}


def shares_content_word(previous: str, claim: str) -> bool:
    """Whether `claim` picks up anything from `previous`.

    A proxy for the chaining rule the reader prompt asks for: connect each
    sentence to the one before it by repeating a noun phrase from its end. That
    is what gives the answer human flow while surviving the removal of any
    sentence, since repeating the noun works where a pronoun would dangle.

    Deliberately crude, and honest about it. A reader can satisfy this while
    writing badly, so it is a floor rather than a measure of quality. What it
    does catch is the failure this project actually saw: a reader reverting to
    a list of disconnected facts, where adjacent claims share no content word
    at all.

    The stopword list is what makes it mean anything. Every pair of English
    sentences shares "the" or "is", so without it the answer is always True.

    Known false-chain sources, in the direction that actually costs
    something: two claims can share a content word by pure accident of
    English function-word usage and get counted as chained when they are not.
    `_STOPWORDS` is a closed, hand-maintained list, not a linguistic
    resource, so any function word outside it (a preposition, a modal, a
    determiner nobody added yet) still counts as content and can produce a
    match neither claim intended. A measured chained-pair rate from this
    function is therefore an upper bound on real chaining, not a
    measurement of it: it can overstate how connected an answer is, never
    understate it.
    """
    return bool(_content_words(previous) & _content_words(claim))


def read(chat: StructuredChat, image_path: Path, question: str) -> list[DraftedClaim]:
    """The drafted answer for `question`, one sentence per claim."""
    out = chat.structured(PROMPT.format(question=question), image_path, ClaimList)
    return list(out.claims)
