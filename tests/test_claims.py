"""The reader's JSON output contract: parse it or refuse to proceed.

A free-form answer is not an answer. If the reader's output cannot be
parsed into claims, raising is the only honest behaviour: silently
dropping claims would turn an uncheckable answer into a confident one.
"""

import pytest

from visual_verify.verify.claims import ReaderOutput, parse_reader_output
from visual_verify.verify.errors import VerifierError


def test_parse_valid_output():
    out = parse_reader_output('{"answer": "It is 42.", "claims": ["The answer is 42."]}')
    assert out == ReaderOutput(answer="It is 42.", claims=["The answer is 42."])


def test_parse_strips_whitespace():
    out = parse_reader_output('{"answer": "  It is 42.  ", "claims": ["  42.  "]}')
    assert out.answer == "It is 42."
    assert out.claims == ["42."]


def test_parse_empty_claims_is_valid_but_vacuous():
    out = parse_reader_output('{"answer": "No idea.", "claims": []}')
    assert out.claims == []


def test_parse_rejects_non_json():
    with pytest.raises(VerifierError):
        parse_reader_output("It is 42.")


def test_parse_rejects_json_with_wrong_shape():
    with pytest.raises(VerifierError):
        parse_reader_output("[1, 2, 3]")
    with pytest.raises(VerifierError):
        parse_reader_output('{"answer": 42, "claims": []}')
    with pytest.raises(VerifierError):
        parse_reader_output('{"answer": "", "claims": []}')
    with pytest.raises(VerifierError):
        parse_reader_output('{"answer": "x", "claims": "one claim"}')
    with pytest.raises(VerifierError):
        parse_reader_output('{"answer": "x", "claims": [""]}')


def test_parse_rejects_missing_fields():
    with pytest.raises(VerifierError):
        parse_reader_output('{"claims": []}')
