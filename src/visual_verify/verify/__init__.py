"""Reader and verifier: pillar 3 of the project.

verify() answers a question from one page, splits the answer into atomic
claims, grounds each claim with S4's ground(), and has a model different
from the reader judge each claim. Weak judgements abstain: a wrong answer
with a confident box drawn on it is worse than no answer.

The core never constructs a model. Reader and verifier arrive as objects,
which is what keeps everything but backends.py testable with fakes.
"""

from visual_verify.verify.core import verify
from visual_verify.verify.errors import VerifierError

__all__ = ["verify", "VerifierError"]
