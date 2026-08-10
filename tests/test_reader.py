"""Claim extraction, and the compound-claim check the schema cannot do."""

from pathlib import Path

from visual_verify.agent.reader import is_compound, read
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
    from visual_verify.agent.reader import opens_with_anaphora

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
    from visual_verify.agent.reader import opens_with_anaphora

    assert opens_with_anaphora("The variant that supports it is Grounded RAG.") is False
    assert opens_with_anaphora("The metrics and their definitions appear below.") is False


def test_a_self_contained_sentence_is_not_flagged():
    from visual_verify.agent.reader import opens_with_anaphora

    assert opens_with_anaphora("The evaluation compares three system variants.") is False
    assert opens_with_anaphora("Each of the three variants is scored separately.") is False


def test_a_word_starting_with_a_pronoun_is_not_flagged():
    """Word boundary, not prefix. "Itemised" and "Thistle" both begin with a
    pronoun's letters and neither is anaphora."""
    from visual_verify.agent.reader import opens_with_anaphora

    assert opens_with_anaphora("Itemised costs appear in Table 2.") is False
    assert opens_with_anaphora("Those results are listed in Table 2.") is True
