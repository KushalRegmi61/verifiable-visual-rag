"""Can the verifier say no?

The rubric has four labels and an abstention threshold built on top of it,
and all of that is decoration if the judge approves whatever it is handed.
Measured on 2026-08-10, roughly 30 of 32 claims in live runs came back
supported at 90 to 100 percent. That is equally consistent with an easy corpus,
where the reader only claims what it can see, and with a lenient judge. These
probes separate the two by handing the verifier claims that are false in a
specific way.

Every probe asserts `label != "supported"` rather than a specific label.
Which of the three non-supported labels fits a given probe is a judgement call,
and pinning one would make the test brittle for no gain.
"""

import os
from pathlib import Path

import pytest

from visual_verify.config import Settings
from visual_verify.contracts import GroundedRegion

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="needs OPENAI_API_KEY for the verifier",
)

FIXTURE = Path(__file__).parent.parent / "data" / "pages"


def _a_page() -> Path:
    if not FIXTURE.exists():
        pytest.skip("no rendered pages; run `vvrag ingest` first")
    pages = sorted(FIXTURE.rglob("*.png"))
    if not pages:
        pytest.skip("no rendered pages found")
    return pages[0]


def _region(text: str) -> GroundedRegion:
    return GroundedRegion(
        page=0,
        bbox=(0.13, 0.30, 0.87, 0.32),
        score=1.0,
        modality="text",
        text=text,
        resolution="line",
    )


def _judge(claim: str, region_text: str) -> str:
    from visual_verify.agent.models import make_chat
    from visual_verify.agent.verifier import verify

    chat = make_chat("verifier", Settings.from_env())
    try:
        verdict = verify(chat, _a_page(), claim, [_region(region_text)])
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is a quota state
        text = str(exc)
        if "429" in text or "RESOURCE_EXHAUSTED" in text or "insufficient_quota" in text:
            pytest.skip(f"provider reachable but unprovisioned: {text[:160]}")
        raise
    return verdict.label


EVIDENCE = "Evaluation on SlideVQA with three metrics"


def test_the_control_probe_is_supported():
    """The floor. If this fails the probes below prove nothing, because a
    verifier that rejects everything passes all four of them."""
    assert _judge("The evaluation uses SlideVQA.", EVIDENCE) == "supported"


def test_a_changed_number_is_not_supported():
    assert _judge("The evaluation reports seven metrics.", EVIDENCE) != "supported"


def test_a_swapped_entity_is_not_supported():
    assert _judge("The evaluation runs on DocVQA.", EVIDENCE) != "supported"


def test_absent_content_is_not_supported():
    assert _judge("The system uses conformal calibration.", EVIDENCE) != "supported"


def test_true_of_the_page_but_absent_from_the_regions_is_not_supported():
    """THE probe. The other three are false anywhere. This one may well be true
    of the page image the verifier is also shown, and it is not established by
    the region it was handed. A verifier reading the image instead of the
    evidence passes the first three and fails this one, which is the exact
    failure mode that would make grounding decorative.
    """
    assert _judge("The evaluation reports exact match and F1.", EVIDENCE) != "supported"
