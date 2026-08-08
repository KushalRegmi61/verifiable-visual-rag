"""The reader's structured output: one answer, N atomic claims."""

import json

from pydantic import BaseModel

from visual_verify.verify.errors import VerifierError


class ReaderOutput(BaseModel):
    answer: str
    claims: list[str]


def parse_reader_output(raw: str) -> ReaderOutput:
    """Validate the reader's JSON contract, or raise VerifierError."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VerifierError("reader output is not valid JSON") from exc
    if not isinstance(data, dict):
        raise VerifierError("reader output must be a JSON object")
    answer = data.get("answer")
    claims = data.get("claims")
    if not isinstance(answer, str) or not answer.strip():
        raise VerifierError("reader output has no answer text")
    if not isinstance(claims, list) or not all(isinstance(c, str) and c.strip() for c in claims):
        raise VerifierError("reader output claims must be a list of non-empty strings")
    return ReaderOutput(answer=answer.strip(), claims=[c.strip() for c in claims])
