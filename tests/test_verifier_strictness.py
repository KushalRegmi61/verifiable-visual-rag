"""Can the verifier say no?

The rubric has four labels and an abstention threshold built on top of it,
and all of that is decoration if the judge approves whatever it is handed.
Measured on 2026-08-10, roughly 30 of 32 claims in live runs came back
supported at 90 to 100 percent. That is equally consistent with an easy corpus,
where the reader only claims what it can see, and with a lenient judge. These
probes separate the two by handing the verifier claims that are false in a
specific way.

The page is BUILT here, not selected from data/pages. An earlier version took
`sorted(rglob("*.png"))[0]`, which resolved to the proposal's title page: a
crest, author names, and a date. Nothing about evaluation or metrics appeared
on it, so the flagship probe below was false of the region AND false of the
page, which made it a duplicate of the absent-content probe and exercised the
image-versus-region failure mode not at all. Selection was also by SHA-256
directory order under a gitignored directory, so the page changed per machine
and changed again on the next ingest. A probe whose premise is "this fact is
true of the page" cannot rest on that. Following tests/conftest.py, the page is
generated with PyMuPDF at test time so no binary lands in git.

Every probe asserts `label != "supported"` rather than a specific label.
Which of the three non-supported labels fits a given probe is a judgement call,
and pinning one would make the test brittle for no gain.
"""

import os
from pathlib import Path

import fitz
import pytest

from conftest import PAGE_H, PAGE_W, _skip_if_no_quota
from visual_verify.agent.schemas import Verdict
from visual_verify.config import Settings
from visual_verify.contracts import GroundedRegion

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="needs OPENAI_API_KEY for the verifier",
)

# LOAD-BEARING, both of them, and in opposite directions.
#
# PARA_A is handed to the verifier as its evidence region, and every probe
# claim is written against it: three metrics, SlideVQA, no mention of
# calibration.
#
# PARA_B is deliberately NOT handed over. It sits elsewhere on the same page,
# visible in the image the verifier is also shown, and it is what makes
# test_true_of_the_page_but_absent_from_the_regions_is_not_supported
# discriminating: that claim is TRUE of the page and unestablished by the
# region. Deleting PARA_B, or naming its metrics inside PARA_A, silently turns
# that probe back into a duplicate of the absent-content one.
PARA_A = "Evaluation on SlideVQA with three metrics"
PARA_B = "The metrics are exact match and F1."

EVIDENCE = PARA_A  # a real line of the rendered page, not a paraphrase of one

RENDER_DPI = 150


@pytest.fixture(scope="session")
def probe_page(tmp_path_factory) -> Path:
    """One page, two well-separated paragraphs, rasterised to PNG.

    Session-scoped because it is identical for all five probes and rendering it
    per test would be pure waste. The two paragraphs are 120 points apart so no
    plausible region could be read as covering both.
    """
    out = tmp_path_factory.mktemp("probe_page")
    pdf = out / "probe.pdf"
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_text((72.0, 200.0), PARA_A, fontsize=14)
    page.insert_text((72.0, 320.0), PARA_B, fontsize=14)
    doc.save(pdf)
    doc.close()

    doc = fitz.open(pdf)
    png = out / "probe.png"
    doc[0].get_pixmap(dpi=RENDER_DPI).save(png)
    doc.close()
    return png


@pytest.fixture(scope="session")
def evidence_bbox() -> tuple[float, float, float, float]:
    """The normalised box of PARA_A on the rendered page.

    Derived from the same insert_text origin the fixture uses rather than
    hardcoded, so the two cannot drift. Not load-bearing for the verdict:
    verify() renders the coordinates into the prompt as text and never crops
    the image, so the numbers are read by the model as a location description
    and nothing else. They are computed honestly anyway, because a box that
    pointed somewhere else would be a lie inside a test about not fabricating
    evidence.
    """
    ascent, descent = 14.0, 4.0
    width = fitz.get_text_length(PARA_A, fontsize=14)
    return (
        72.0 / PAGE_W,
        (200.0 - ascent) / PAGE_H,
        (72.0 + width) / PAGE_W,
        (200.0 + descent) / PAGE_H,
    )


def _judge(page: Path, bbox, claim: str) -> Verdict:
    """Return the whole Verdict, never just the label.

    These calls are nondeterministic and paid, so a failure gets one look. The
    label alone prints as `assert 'supported' != 'supported'`, which cannot
    distinguish a verifier that misread the evidence from one that read the
    image instead from one that was being defensibly strict. The reason
    sentence is the only thing that can.
    """
    from visual_verify.agent.models import make_chat
    from visual_verify.agent.verifier import verify

    region = GroundedRegion(
        page=0,
        bbox=bbox,
        score=1.0,
        modality="text",
        text=EVIDENCE,
        resolution="line",
    )
    chat = make_chat("verifier", Settings.from_env())
    try:
        return verify(chat, page, claim, [region])
    except Exception as exc:  # noqa: BLE001 - narrowed inside the helper
        _skip_if_no_quota(exc)


def test_the_control_probe_is_supported(probe_page, evidence_bbox):
    """The floor. If this fails the probes below prove nothing, because a
    verifier that rejects everything passes all four of them.

    It carries a second signal for free: it also shows the verifier is reading
    the region at all, since a judge ignoring the evidence entirely would have
    to reach this from the image, and the image says the same thing only
    because PARA_A is on it.
    """
    v = _judge(probe_page, evidence_bbox, "The evaluation uses SlideVQA.")
    assert v.label == "supported", f"{v.label} @ {v.confidence}: {v.reason}"


def test_a_changed_number_is_not_supported(probe_page, evidence_bbox):
    v = _judge(probe_page, evidence_bbox, "The evaluation reports seven metrics.")
    assert v.label != "supported", f"{v.label} @ {v.confidence}: {v.reason}"


def test_a_swapped_entity_is_not_supported(probe_page, evidence_bbox):
    v = _judge(probe_page, evidence_bbox, "The evaluation runs on DocVQA.")
    assert v.label != "supported", f"{v.label} @ {v.confidence}: {v.reason}"


def test_absent_content_is_not_supported(probe_page, evidence_bbox):
    """False of the region and absent from the page image too, which is the
    axis the probe below deliberately does not test."""
    v = _judge(probe_page, evidence_bbox, "The system uses conformal calibration.")
    assert v.label != "supported", f"{v.label} @ {v.confidence}: {v.reason}"


def test_true_of_the_page_but_absent_from_the_regions_is_not_supported(probe_page, evidence_bbox):
    """THE probe. The other three are false of the page as well as the region.
    This one is TRUE of the page image the verifier is shown, printed on it as
    PARA_B, and is not established by the region it was handed. A verifier
    reading the image instead of the evidence passes the first three and fails
    this one, which is the exact failure mode that would make grounding
    decorative.
    """
    v = _judge(probe_page, evidence_bbox, "The evaluation reports exact match and F1.")
    assert v.label != "supported", f"{v.label} @ {v.confidence}: {v.reason}"
