"""The four-label rubric and the threshold that acts on it.

The model emits a label; this module owns the mapping label -> number and
the gate number >= threshold. The gate is a knob: S7 tunes one threshold
and measures confident-wrong against coverage.
"""

import pytest
from pydantic import ValidationError

from visual_verify.verify.errors import VerifierError
from visual_verify.verify.rubric import LABELS, Judgement, is_answered, sufficiency


def test_sufficiency_maps_every_label():
    assert {sufficiency(label) for label in LABELS} == {1.0, 0.5, 0.0}


def test_supported_and_partial_pass_default_threshold():
    assert is_answered("supported", 0.5)
    assert is_answered("partial", 0.5)
    assert not is_answered("unsupported", 0.5)
    assert not is_answered("insufficient", 0.5)


def test_threshold_moves_the_gate():
    assert is_answered("partial", 0.4)
    assert not is_answered("partial", 0.6)


def test_unknown_label_raises():
    with pytest.raises(VerifierError):
        sufficiency("maybe")


def test_judgement_rejects_unknown_labels_at_construction():
    with pytest.raises(ValidationError):
        Judgement(label="maybe")
