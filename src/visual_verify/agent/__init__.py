"""Reader, verifier, and abstention: pillars 2 and 3.

The reader and the verifier are deliberately DIFFERENT hosted models. A model
grading its own output is biased toward it (proposal.tex line 377), so the
separation is the point, not an implementation detail.

Everything here takes a StructuredChat as an argument and never constructs one.
LangChain is imported in exactly one file, models.py, which is what keeps the
whole pipeline testable with no network and no API key.

Task 10 adds the public re-exports here. Nothing else belongs in this file.
"""

from visual_verify.agent.core import AgentError, answer

__all__ = ["AgentError", "answer"]
