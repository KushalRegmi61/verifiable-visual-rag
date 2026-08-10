"""Claim extraction, and the compound-claim check the schema cannot do."""

from pathlib import Path

from visual_verify.agent.reader import is_compound, opens_with_anaphora, read, shares_content_word
from visual_verify.agent.schemas import ClaimList
from visual_verify.agent.types import FakeChat


def test_read_returns_the_models_claims():
    chat = FakeChat("m", [ClaimList(claims=["Revenue grew.", "Margins held."])])
    claims = read(chat, Path("page.png"), "What happened?")
    assert [c.text for c in claims] == ["Revenue grew.", "Margins held."]


def test_read_sends_the_question_and_the_page_image():
    chat = FakeChat("m", [ClaimList(claims=["a"])])
    read(chat, Path("page.png"), "What is the threshold?")

    call = chat.calls[0]
    assert "What is the threshold?" in call.prompt
    assert call.image_path == Path("page.png")


def test_read_returns_an_empty_list_when_the_page_answers_nothing():
    chat = FakeChat("m", [ClaimList(claims=[])])
    assert read(chat, Path("page.png"), "unrelated question") == []


def test_a_conjunction_joined_claim_is_flagged_as_compound():
    """The schema cannot enforce atomicity. A claim asserting two things
    cannot be grounded to one region, so it must be visible in the eval
    rather than silently accepted."""
    assert is_compound("Revenue grew 42 percent and margins held steady.")


def test_a_single_assertion_is_not_flagged():
    assert not is_compound("Revenue grew 42 percent in the third quarter.")


def test_a_noun_phrase_conjunction_is_not_flagged():
    """'X and Y rose' is one assertion about two subjects, not two claims.
    Flagging it would report a decomposition failure that did not happen."""
    assert not is_compound("Revenue and margin both rose.")


def test_a_compound_subject_without_filler_word_is_not_flagged():
    """Same shape as the 'both rose' case with the filler word removed.

    A verb-shortly-after-conjunction check flags this, wrongly: there is no
    verb before the conjunction, only one shared verb after it, so this is
    still one assertion with a compound subject."""
    assert not is_compound("Revenue and margin rose.")


def test_a_compound_subject_with_a_widened_verb_is_not_flagged():
    assert not is_compound("Revenue and expenses both totalled 5 million.")


def test_semicolon_joined_clauses_is_flagged():
    """A semicolon joins two independent clauses the same way 'and' does."""
    assert is_compound("Revenue grew; margins held steady.")


def test_a_multi_word_gap_between_the_joiner_and_the_second_verb_is_flagged():
    """The old adjacency check missed this; a verb on each side of the
    joiner, however far apart, is the actual signal for joined clauses."""
    assert is_compound("Revenue grew and net margins also held steady.")


def test_the_widened_verb_vocabulary_catches_report_register():
    assert is_compound("Revenue improved and expenses declined.")


def test_a_sentence_opening_with_a_pronoun_is_flagged():
    assert opens_with_anaphora("It also isolates the added layer.") is True
    assert opens_with_anaphora("They are scored on three metrics.") is True
    assert opens_with_anaphora("This approach avoids annotation.") is True
    assert opens_with_anaphora("Additionally, the ablation removes the layer.") is True


def test_a_sentence_merely_containing_a_pronoun_is_not_flagged():
    """THE discriminating test. A substring match anywhere in the sentence
    flags almost every English sentence, which would make the check useless
    while looking like it works. Only the OPENING can dangle, because only the
    opening is what a reader resolves against the previous sentence.
    """
    assert opens_with_anaphora("The variant that supports it is Grounded RAG.") is False
    assert opens_with_anaphora("The metrics and their definitions appear below.") is False


def test_a_self_contained_sentence_is_not_flagged():
    assert opens_with_anaphora("The evaluation compares three system variants.") is False
    assert opens_with_anaphora("Each of the three variants is scored separately.") is False


def test_a_word_starting_with_a_pronoun_is_not_flagged():
    """Word boundary, not prefix. "Itemised" and "Thistle" both begin with a
    pronoun's letters and neither is anaphora."""
    assert opens_with_anaphora("Itemised costs appear in Table 2.") is False
    assert opens_with_anaphora("Thistle grows wild here.") is False
    assert opens_with_anaphora("Those results are listed in Table 2.") is True


def test_leading_whitespace_does_not_defeat_the_anchor():
    """The anchor is `^\\s*`, not `^`; a claim with incidental leading
    whitespace must still be checked from its first real word."""
    assert opens_with_anaphora("  This approach avoids annotation.") is True


def test_a_page_deictic_demonstrative_is_not_flagged():
    """'This page', 'this table', 'this figure' point at the image the reader
    is looking at, not at the sentence before. They are deictic, not
    anaphoric, and survive their predecessor being withheld intact. A live
    test in test_answer.py asks "What is this page about?" and expects a
    correct "This page ..." answer to pass with zero tolerance, so a false
    positive here is not academic."""
    assert opens_with_anaphora("This page defines three variants.") is False
    assert opens_with_anaphora("This table lists the results.") is False
    assert opens_with_anaphora("This figure compares two baselines.") is False
    assert opens_with_anaphora("This document specifies the ablation design.") is False
    assert opens_with_anaphora("This approach avoids annotation.") is True


def test_the_lookahead_is_a_word_not_a_prefix():
    """_PAGE_DEICTIC is a bare alternation, so the lookahead needs its own
    trailing \\b or it matches a prefix: 'chart' inside 'charter', 'document'
    inside 'documentation', 'section' inside 'sectional'. All three genuinely
    dangle and must still be flagged."""
    assert opens_with_anaphora("This charter defines the scope.") is True
    assert opens_with_anaphora("This documentation covers the ablation.") is True
    assert opens_with_anaphora("This sectional view is shown.") is True


def test_that_is_covered_like_the_rest_of_the_demonstrative_paradigm():
    """'That' is the fourth member of this/these/those and follows the same
    deictic exemption; there was no principled reason it was left out."""
    assert opens_with_anaphora("That approach avoids annotation.") is True
    assert opens_with_anaphora("That figure appears in Table 2.") is False


def test_resultative_connectives_are_flagged_like_however():
    """however/moreover/furthermore already dangle; therefore/hence/
    consequently/instead resolve against the previous sentence the same way
    and were missing for no principled reason."""
    assert opens_with_anaphora("Therefore, the claim is withheld.") is True
    assert opens_with_anaphora("Hence the ablation removes the layer.") is True
    assert opens_with_anaphora("Consequently the score drops.") is True
    assert opens_with_anaphora("Instead, the reader abstains.") is True


def test_expletive_there_is_not_flagged():
    """'There' is an expletive subject, not a referring pronoun. "There are
    three variants" is fully self-contained; flagging it would be a pure
    false positive, so it is excluded by design."""
    assert opens_with_anaphora("There are three variants.") is False


def test_bare_quantifier_openers_are_a_known_miss():
    """Documented, not silently wrong: 'Both'/'Each of them' dangle without a
    noun of their own, but a flat word list cannot tell that shape apart from
    'Both variants are scored', which is self-contained. Left unflagged by
    design; see the docstring."""
    assert opens_with_anaphora("Both are scored on three metrics.") is False
    assert opens_with_anaphora("Each of them is scored.") is False


def test_a_chained_pair_shares_a_content_word():
    assert shares_content_word(
        "The evaluation compares three system variants on SlideVQA.",
        "Each of the three variants is scored on answer accuracy.",
    ) is True


def test_two_unrelated_claims_share_nothing():
    """The second claim is deliberately given its own leading "The", so the two
    sentences share a stopword ("the") and nothing else. A version without a
    shared stopword passes even with `_STOPWORDS` emptied out, by coincidence
    rather than by exercising the guarantee this test exists to pin; this
    phrasing fails under that mutation, as it must."""
    assert shares_content_word(
        "The evaluation compares three system variants on SlideVQA.",
        "The ground truth is derived automatically.",
    ) is False


def test_a_stopword_overlap_does_not_count_as_chaining():
    """THE test of this function. Every pair of English sentences shares "the"
    or "is". Without a stopword list the check returns True for everything,
    which is worse than not having it: it would report perfect chaining on a
    reader that had reverted to listing disconnected facts.
    """
    assert shares_content_word("The page is here.", "The chart is there.") is False


def test_a_plural_matches_its_singular():
    """Claims chain through a noun phrase that often changes number across the
    join: "three variants" then "each variant"."""
    assert shares_content_word(
        "The evaluation compares three variants.",
        "Each variant is scored separately.",
    ) is True


def test_punctuation_does_not_block_a_match():
    assert shares_content_word("Scores come from SlideVQA.", "SlideVQA has 2000 slides.") is True


def test_a_doubled_trailing_s_is_not_over_stripped():
    """str.rstrip("s") removes every trailing s, not one. "caress" ends in a
    doubled s ("...ess"), and stripping both collapses it to "care", which
    collides with the unrelated word "cares" (itself stripped to "care") even
    though a caress and caring about something share no meaning. Stripping at
    most a single trailing character keeps "caress" as "cares", distinct from
    "care"/"cares", and the false match disappears."""
    assert shares_content_word(
        "She cares about the outcome.",
        "He gave a gentle caress.",
    ) is False


def test_bare_numbers_and_possessive_s_do_not_count_as_chaining():
    """A page number, figure number, section number, or possessive is
    ordinary in an answer about a document page. `_WORD` requires a leading
    letter so a bare number token ("3", or "1" from "5.1") is never a content
    word, and a possessive's stray "s" ("document's" -> "document", "s")
    never becomes one either."""
    assert shares_content_word(
        "Page 3 lists the evaluation metrics.",
        "Figure 3 plots the ablation.",
    ) is False
    assert shares_content_word(
        "The rubric defines section 5.1.",
        "Recall improves by 1 point.",
    ) is False
    assert shares_content_word(
        "The document's title is bold.",
        "The reader's output is prose.",
    ) is False


def test_each_does_not_count_as_chaining():
    """"each" appears in essentially every sentence about a set of things and
    carries no topic of its own; two claims about unrelated sets should not
    chain just because both use it."""
    assert shares_content_word(
        "Each variant is evaluated separately.",
        "Each metric is reported once.",
    ) is False
