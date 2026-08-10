"""The reader: page image plus question, out come atomic claims.

Claims are emitted directly as structured output rather than as prose that a
second call splits. One API call instead of two, and the model that wrote the
answer is the one deciding where it separates.

There is no separate prose answer. The displayed answer is the claims joined,
so nothing can drift between what is shown and what is verified.
"""

import re
import warnings
from pathlib import Path

from visual_verify.agent.schemas import ClaimList, DraftedClaim
from visual_verify.agent.types import StructuredChat

PROMPT = """You are answering a question from one page of a document, for
someone who cannot see the page.

Answer only from what is visible on this page. If the page does not answer the
question, return an empty list of claims.

Compose the answer as connected prose, then return it as one sentence per
claim, in the order they should be read. Every sentence is checked separately
against the page, and any sentence may be removed before the answer is shown,
so the sentences must obey four rules.

1. The FIRST sentence answers the question directly. Not background, not what
   the page is about. Someone who read only that sentence should have the
   direct answer, not every detail of it. The detail goes in the sentences
   after it, one thing per sentence.

2. Each sentence asserts exactly ONE thing. Each sentence is matched to a
   single region of the page as its evidence, and a sentence asserting two
   things cannot be evidenced by one region.

3. Each sentence stands on its own. Never refer back to the previous sentence
   or to anything above. No sentence may begin with it, its, they, them,
   their, such, this, that, these, or those, and no sentence may begin with a
   connective such as Additionally, Also, However, Therefore, Instead,
   Furthermore, Moreover, Hence, or Consequently. No sentence may use a
   pronoun whose meaning is only in an earlier sentence. Repeat the noun
   instead. The one exception is pointing at what you are looking at, which
   depends on no other sentence: this, these, those and that are allowed only
   when the very next word is page, document, slide, figure, table, chart,
   section or paragraph, as in "This page", "This figure", "This table".
   "These three metrics" is not allowed, because the metrics were named in an
   earlier sentence.

4. Connect each sentence to the one before it by repeating a noun phrase from
   the end of that sentence, never by a pronoun. "The evaluation compares three
   variants. Each of the three variants is scored on ..." reads as one answer.
   "The evaluation compares three variants. Each of them is scored on ..."
   becomes nonsense the moment the first sentence is removed.

Write the way you would answer a colleague who asked you out loud. Do not
describe the page. State what it says. A complete answer is usually three to
six sentences, and a sentence is usually shorter than twenty words.

Set starts_paragraph on a sentence that opens a new topic, and leave it false
otherwise. Most answers are a single paragraph.

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

    One whole CLASS is invisible to it, and no growth of `_VERB` will change
    that: a single clause that piles up several assertions without coordinating
    any clauses. Measured, from the first output of the drafting prompt: "The
    evaluation methodology evaluates both the baseline document RAG system and
    the proposed system using answer accuracy, grounding overlap, and abstention
    quality so that the contribution of the added layer can be measured directly
    through an ablation." Thirty-seven words asserting three things, and
    `is_compound` returns False, correctly by its own definition. The "and"s
    join a compound OBJECT and a list, "using ..." is a participial adjunct, and
    "so that ..." is a purpose clause. None of the three is a coordinated
    clause, so there is no second independent verb for the check to find. A
    length cap is the only cheap signal that sees this shape.
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
    Because the filter sees the RAW token, `_STOPWORDS` has to name the
    inflected forms people actually type: "is", "as", "has", "was", "its",
    "his", "does", not their stems. If stripping ran first, every one of
    those entries would be dead on arrival: the filter would be handed "i",
    "a", "ha", "wa", "it", "hi", "doe" instead, and match none of them.
    "does" to "doe" is the illustration: filter-first catches "does" because
    it is in the list; strip-first would have already turned it into "doe"
    by the time any filter ran, and "doe" is not a word the list will ever
    recognize.

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


# Same vocabulary as _WORD above, minus the leading-letter requirement. A bare
# number cannot chain two sentences together, which is why _WORD excludes it,
# but a number absolutely can be the thing a claim cites: "42%" is the evidence
# for "revenue grew 42 percent". Two questions, two patterns, one stopword list.
_TERM = re.compile(r"[a-z0-9]+")


def shares_a_term(claim: str, region_text: str | None) -> bool:
    """Whether a region's text is plausibly about `claim` at all.

    NOT a relevance score and not a threshold. It answers one categorical
    question: does the text under this box name anything the claim names? A
    region that shares nothing is not weak evidence, it is a non-sequitur, and
    citing it asserts a connection that does not exist.

    Built for a measured failure. On proposal.pdf page 14 the visual path
    selected [0.526 0.940 0.535 0.953], an 11 by 22 px box holding the page
    number, as the evidence for three different claims across two unrelated
    questions. The verifier scored two of them supported at 0.90 and 0.95,
    because it judges whether the CLAIM is true and the claims were. The claim
    was true, the citation was fabricated, and the box landed on real ink so
    nothing looked wrong. That is the failure this project exists to prevent,
    arriving through the one door the abstention gate does not watch.

    Deliberately NOT `shares_content_word`. That function's pattern requires a
    leading letter so a bare number cannot chain two sentences, which is right
    for chaining and wrong here: it would score every purely numeric region as
    fabricated and drop the citation a numeric claim most needs.

    Returns True when there is no text to judge. A visual region can snap to a
    box with no text layer, and an unjudgeable citation must not be called
    fabricated: absence of text is absence of evidence ABOUT the citation, not
    evidence against it. Those regions keep whatever trust the grounder gave
    them, which is the honest default and also the one that cannot regress a
    scanned page into answering nothing.

    The stopword list is what makes it mean anything, exactly as it is for
    `shares_content_word`. A text-layer line almost always contains "the", so
    without the filter every region cites every claim.
    """
    if region_text is None or not region_text.strip():
        return True
    claim_terms = {w for w in _TERM.findall(claim.lower()) if w not in _STOPWORDS}
    region_terms = {w for w in _TERM.findall(region_text.lower()) if w not in _STOPWORDS}
    return bool(claim_terms & region_terms)


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

    Errs in both directions, and a measured chained-pair rate is neither an
    upper nor a lower bound on real chaining.

    Overcounting is the more likely and more costly direction: two claims can
    share a content word by pure accident of English function-word usage and
    get counted as chained when they are not. `_STOPWORDS` is a closed,
    hand-maintained list, not a linguistic resource, so any function word
    outside it (a preposition, a modal, a determiner nobody added yet) still
    counts as content and can produce a match neither claim intended.

    But it also undercounts, on genuine chains that use an inflection
    `_content_words` does not reunite. A "-es" or "-ies" plural does not
    match its singular: "analyses" strips to "analyse" against "analysis",
    and "policies" strips to "policie" against "policy". And `_WORD` requiring
    a leading letter, which exists to keep bare numbers out of false chains,
    also drops a genuinely repeated bare number: "2026" in one claim and
    "2026" in the next never becomes a token at all, so a real repetition is
    invisible to this function. Whoever sets a floor on this rate should have
    both misses in view, not just the overcount.
    """
    return bool(_content_words(previous) & _content_words(claim))


def read(chat: StructuredChat, image_path: Path, question: str) -> list[DraftedClaim]:
    """The drafted answer for `question`, one sentence per claim.

    Warns when the provider returned bare strings instead of objects. The
    prompt asks for `starts_paragraph`, so bare strings mean the schema was
    ignored: `ClaimList` coerces them and every claim silently takes False, the
    answer renders as one paragraph forever, and nothing raises or fails. A
    warning rather than an exception, because a single-paragraph answer is
    still an answer and refusing to return it would be the worse failure.

    "Warns" means once per call here, and once per source location under
    Python's default filter, so a second question from the same caller in the
    same process is silent. It does NOT survive `CachedChat`. The cache stores
    `model_dump_json`, which writes claims as objects, and the replay path
    revalidates that, so the coercion never fires again and the flag is a
    PrivateAttr the cache does not carry. The first live run against a
    schema-ignoring provider warns and every replay is quiet, including the
    offline defense demo. Persisting it would mean changing the cache entry
    format, which is also the reproducibility record, so it is recorded here
    instead: if the answer is one paragraph and no warning appeared, check
    whether the response came from the cache before concluding the provider
    behaved.
    """
    out = chat.structured(PROMPT.format(question=question), [image_path], ClaimList)
    if out.from_bare_strings:
        warnings.warn(
            f"{chat.model_id} returned claims as bare strings instead of objects, so it "
            "ignored the output schema: starts_paragraph is False on every claim and the "
            "answer will render as a single paragraph regardless of its content",
            stacklevel=2,
        )
    return list(out.claims)
