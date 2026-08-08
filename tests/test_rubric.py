"""The four-label rubric and the score an abstention threshold acts on."""

import pytest

from visual_verify.agent.rubric import LABELS, Label, abstention_score


def test_all_four_proposal_labels_exist():
    """Fixed by proposal.tex line 377. Renaming one breaks the report."""
    assert set(LABELS) == {
        "supported",
        "partially_supported",
        "unsupported",
        "insufficient_evidence",
    }


def test_labels_rank_supported_highest_and_unsupported_lowest():
    ranked = sorted(LABELS, key=lambda label: abstention_score(label, 0.0))
    assert ranked[0] == "unsupported"
    assert ranked[-1] == "supported"


def test_confidence_orders_within_a_label_but_never_across_one():
    """The label decides; confidence only breaks ties inside it.

    A confident 'partially supported' must never outrank a hesitant
    'supported', or a self-reported number would be overriding the rubric.
    """
    confident_partial = abstention_score("partially_supported", 1.0)
    hesitant_supported = abstention_score("supported", 0.0)

    assert confident_partial < hesitant_supported


def test_confidence_does_order_within_a_label():
    assert abstention_score("supported", 0.9) > abstention_score("supported", 0.1)


def test_score_rejects_an_out_of_range_confidence():
    with pytest.raises(ValueError, match="0 and 1"):
        abstention_score("supported", 1.5)


def test_score_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="unknown label"):
        abstention_score("looks_fine", 0.5)


def test_label_is_a_plain_string_type():
    """Label must stay JSON-serializable for the cache and the eval output."""
    assert Label.__args__ == (
        "supported",
        "partially_supported",
        "insufficient_evidence",
        "unsupported",
    )
