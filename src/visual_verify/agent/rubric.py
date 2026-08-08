"""The four-label rubric and the score an abstention threshold acts on.

Labels are fixed by proposal.tex line 377 and are part of the deliverable.
Do not rename, reorder the public tuple, or add a fifth.

WHY THE SCORE IS 2 * label_rank + confidence, and not either alone:

The label decides whether a claim is shown. Confidence only orders claims
WITHIN a label, so a confident "partially supported" can never outrank a
hesitant "supported": a self-reported number must not override the rubric.

Confidence is there because S7's headline metric is confident-wrong against
coverage, and a curve swept over four labels has four operating points. The
fractional part gives the curve resolution without giving it authority.

Self-reported confidence is NOT calibrated. The report must say so. Conformal
calibration is named future work in proposal.tex line 381.
"""

from typing import Literal

Label = Literal[
    "supported",
    "partially_supported",
    "insufficient_evidence",
    "unsupported",
]

# Ranks are spaced by _BAND=2 while confidence spans a width of 1, so each
# label occupies a band of width 1 with a gap of 1 before the next: [0,1],
# [2,3], [4,5], [6,7]. Spacing of 1 would NOT work: confidence is inclusive
# of 1.0, so partially_supported at 1.0 would equal supported at 0.0 exactly,
# and a partially supported claim would tie the supported floor and be SHOWN.
# That is the precise failure this project exists to prevent, which is why
# the separation must be a gap and not a touching boundary.
_RANK: dict[str, int] = {
    "supported": 3,
    "partially_supported": 2,
    "insufficient_evidence": 1,
    "unsupported": 0,
}

# Multiplier that turns the ranks above into non-touching bands. See the
# comment on _RANK before changing it.
_BAND = 2

LABELS: tuple[str, ...] = tuple(_RANK)


def abstention_score(label: str, confidence: float) -> float:
    """Rank the claim for the abstention threshold. Higher means more supported."""
    if label not in _RANK:
        raise ValueError(f"unknown label {label!r}; expected one of {sorted(_RANK)}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    return _BAND * _RANK[label] + confidence


# The "supported" floor at zero confidence: the default abstention threshold,
# admitting only fully supported claims. core.py's DEFAULT_THRESHOLD and
# config.py's Settings.abstain_threshold default both used to repeat this as
# an independent literal 6.0; deriving it here makes it one source, so a
# future change to _RANK or _BAND cannot desynchronize the three copies.
SUPPORTED_FLOOR = abstention_score("supported", 0.0)
