"""The four-label rubric, the sufficiency mapping, and the abstention gate.

The verifier model emits one of LABELS. The number a threshold acts on is
computed HERE, from a table, and pinned by tests: the model never emits
numbers and the gate is auditable.
"""

from typing import Literal

from pydantic import BaseModel

from visual_verify.verify.errors import VerifierError

LABELS = ("supported", "partial", "unsupported", "insufficient")

SUFFICIENCY = {
    "supported": 1.0,
    "partial": 0.5,
    "unsupported": 0.0,
    "insufficient": 0.0,
}

RubricLabel = Literal["supported", "partial", "unsupported", "insufficient"]


class Judgement(BaseModel):
    label: RubricLabel


def sufficiency(label: str) -> float:
    if label not in SUFFICIENCY:
        raise VerifierError(f"unknown rubric label {label!r}")
    return SUFFICIENCY[label]


def is_answered(label: str, threshold: float) -> bool:
    return sufficiency(label) >= threshold
