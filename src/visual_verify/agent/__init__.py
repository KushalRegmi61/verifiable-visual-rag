"""Reader, verifier, and abstention: pillars 2 and 3.

The reader and the verifier are deliberately DIFFERENT hosted models. A model
grading its own output is biased toward it (proposal.tex line 377), so the
separation is the point, not an implementation detail.

Everything here takes a StructuredChat as an argument and never constructs one.
LangChain is imported in exactly one file, models.py, which is what keeps the
whole pipeline testable with no network and no API key.

The different-models check in core.answer() compares `StructuredChat.model_id`,
the configured provider:model string, not the underlying weights. Two
differently-named deployments of the same weights would pass the check; it
catches misconfiguration, not genuine weight identity, so a freshly swapped
reader should not be over-trusted on the strength of this check alone.
"""

from visual_verify.agent.core import AgentError, answer, answer_stream

__all__ = ["AgentError", "answer", "answer_stream"]
