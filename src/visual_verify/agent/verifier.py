"""The verifier: a DIFFERENT model judges whether the evidence supports a claim.

Different by construction, not by convention. proposal.tex line 377 requires a
separate judge because a model grading its own output is biased toward it, and
the two roles are configured to different providers.

verify() takes data and a chat, never a client handle it built itself. The same
discipline that kept ground() free of Qdrant and a GPU, and it buys the same
thing: the whole rubric path is testable with no network and no key.
"""

from pathlib import Path

from visual_verify.agent.schemas import Verdict
from visual_verify.agent.types import StructuredChat
from visual_verify.contracts import GroundedRegion

PROMPT = """You are checking whether a claim is supported by specific evidence
from a document page. You did not write the claim. Judge it strictly.

Claim: {claim}

Evidence regions selected from the page:
{evidence}

Choose exactly one label:
- supported: the evidence clearly establishes the claim
- partially_supported: the evidence establishes part of the claim
- unsupported: the evidence contradicts the claim, or is about something else
- insufficient_evidence: there is not enough evidence to judge

Give a confidence between 0 and 1, and one sentence of reasoning."""

NO_EVIDENCE = "(no regions were found for this claim)"


def _render(regions: list[GroundedRegion]) -> str:
    if not regions:
        return NO_EVIDENCE
    lines = []
    for r in regions:
        x0, y0, x1, y1 = r.bbox
        where = f"page {r.page}, {r.modality}, box [{x0:.3f} {y0:.3f} {x1:.3f} {y1:.3f}]"
        lines.append(f"- {where}: {r.text or '(no text layer here)'}")
    return "\n".join(lines)


def verify(
    chat: StructuredChat, image_path: Path, claim: str, regions: list[GroundedRegion]
) -> Verdict:
    """Judge one claim against its regions.

    A claim with NO regions is still sent. insufficient_evidence is a label the
    rubric already has, and routing an ungrounded claim around the verifier
    would discard the signal the project exists to measure.
    """
    prompt = PROMPT.format(claim=claim, evidence=_render(regions))
    return chat.structured(prompt, image_path, Verdict)
