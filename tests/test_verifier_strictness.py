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

from conftest import PAGE_H, PAGE_W
from helpers import skip_if_no_quota
from visual_verify.agent.schemas import Verdict
from visual_verify.config import Settings
from visual_verify.contracts import GroundedRegion

# Per-test, not a module-level pytestmark. test_the_fixture_premise_holds
# spends no call and guards the rest of the file, so it must still run on a
# fresh clone with no key. Under a module mark it skipped with everything else,
# which is precisely when a broken fixture would go unnoticed.
needs_key = pytest.mark.skipif(
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
# region. Deleting PARA_B, or shrinking FONT_SIZE or RENDER_DPI far enough that
# PARA_B stops being legible to a vision model, turns that probe back into a
# duplicate of the absent-content one, and it keeps passing while it happens.
# Those are the edits that fail silently. Naming exact match and F1 inside
# PARA_A is a different thing and safe: the region would then establish the
# claim, the verifier would answer supported, and the probe would fail loudly.
# test_the_fixture_premise_holds guards the silent direction.
PARA_A = "Evaluation on SlideVQA with three metrics"
PARA_B = "The metrics are exact match and F1."

# One spelling each, read by both the fixture that draws the page and the
# fixture that reports the evidence box. They were duplicated literals in two
# functions, under a docstring claiming they could not drift.
PARA_A_ORIGIN = (72.0, 200.0)
PARA_B_ORIGIN = (72.0, 320.0)
FONT_SIZE = 14.0

EVIDENCE = PARA_A  # a real line of the rendered page, not a paraphrase of one

RENDER_DPI = 150


@pytest.fixture(scope="session")
def probe_pdf(tmp_path_factory) -> Path:
    """One page, two well-separated paragraphs.

    Kept as its own fixture so the premise test can read the text layer back
    out. The paragraphs are 120 points apart, so no plausible region around one
    could be read as covering the other.
    """
    pdf = tmp_path_factory.mktemp("probe_page") / "probe.pdf"
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_text(PARA_A_ORIGIN, PARA_A, fontsize=FONT_SIZE)
    page.insert_text(PARA_B_ORIGIN, PARA_B, fontsize=FONT_SIZE)
    doc.save(pdf)
    doc.close()
    return pdf


@pytest.fixture(scope="session")
def probe_page(probe_pdf: Path) -> Path:
    """The rendered PNG, which is what the verifier actually sees.

    Session-scoped because it is identical for all five probes and rendering it
    per test would be pure waste.
    """
    doc = fitz.open(probe_pdf)
    png = probe_pdf.with_suffix(".png")
    doc[0].get_pixmap(dpi=RENDER_DPI).save(png)
    doc.close()
    return png


@pytest.fixture(scope="session")
def evidence_bbox() -> tuple[float, float, float, float]:
    """The normalised box of PARA_A on the rendered page.

    Reads PARA_A_ORIGIN and FONT_SIZE, the same constants probe_pdf draws with,
    so moving the paragraph moves this box too. Not load-bearing for the
    verdict: verify() renders the coordinates into the prompt as text and never
    crops the image, so the model reads them as a location description and
    nothing else. They are computed honestly anyway, because a box pointing
    somewhere else would be a lie inside a test about not fabricating evidence,
    and test_the_fixture_premise_holds checks them against the laid-out text
    rather than against this arithmetic.
    """
    x0, baseline = PARA_A_ORIGIN
    ascent, descent = FONT_SIZE, FONT_SIZE * 0.29
    width = fitz.get_text_length(PARA_A, fontsize=FONT_SIZE)
    return (
        x0 / PAGE_W,
        (baseline - ascent) / PAGE_H,
        (x0 + width) / PAGE_W,
        (baseline + descent) / PAGE_H,
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
        skip_if_no_quota(exc)


def test_the_fixture_premise_holds(probe_pdf, evidence_bbox):
    """Guards the flagship probe without spending a call.

    If PARA_B stops being on the page, or starts falling inside the region
    handed to the verifier, the probe below silently degrades into the
    absent-content one and still passes. That is how the first version of this
    file was wrong, and this assertion would have caught it that day.

    Everything here is measured against the laid-out text, never against the
    arithmetic in evidence_bbox, which is the S2 rule: verify coordinates
    against rendered ink, not against the numbers that produced them.
    """
    page = fitz.open(probe_pdf)[0]

    found_b = page.search_for(PARA_B)
    assert found_b, f"PARA_B is not on the page at all: {page.get_text()!r}"
    b = found_b[0]

    found_a = page.search_for(PARA_A)
    assert found_a, f"PARA_A is not on the page at all: {page.get_text()!r}"
    a = found_a[0]

    # The evidence box must actually sit on PARA_A. This is what makes the
    # "cannot drift" claim in evidence_bbox true rather than asserted: move
    # PARA_A_ORIGIN and only one of the two would follow, and this fails.
    ex0, ey0, ex1, ey1 = evidence_bbox
    assert abs(ex0 * PAGE_W - a.x0) < 2.0, f"evidence box x0 {ex0 * PAGE_W} vs PARA_A {a.x0}"
    assert abs(ey0 * PAGE_H - a.y0) < 6.0, f"evidence box y0 {ey0 * PAGE_H} vs PARA_A {a.y0}"
    assert abs(ex1 * PAGE_W - a.x1) < 6.0, f"evidence box x1 {ex1 * PAGE_W} vs PARA_A {a.x1}"

    # And it must not reach PARA_B, or the verifier would have been handed the
    # very text the flagship claim needs.
    assert not fitz.Rect(ex0 * PAGE_W, ey0 * PAGE_H, ex1 * PAGE_W, ey1 * PAGE_H).intersects(b), (
        f"the evidence region overlaps PARA_B: {evidence_bbox} vs {b}"
    )

    # Legible at the DPI the PNG is rendered at. A font or DPI small enough to
    # make PARA_B unreadable to a vision model is the other silent failure, and
    # it leaves the text layer above perfectly intact.
    px = b.height * RENDER_DPI / 72.0
    assert px >= 12.0, f"PARA_B renders {px:.1f}px tall, too small to read"


@needs_key
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


@needs_key
def test_a_changed_number_is_not_supported(probe_page, evidence_bbox):
    v = _judge(probe_page, evidence_bbox, "The evaluation reports seven metrics.")
    assert v.label != "supported", f"{v.label} @ {v.confidence}: {v.reason}"


@needs_key
def test_a_swapped_entity_is_not_supported(probe_page, evidence_bbox):
    v = _judge(probe_page, evidence_bbox, "The evaluation runs on DocVQA.")
    assert v.label != "supported", f"{v.label} @ {v.confidence}: {v.reason}"


@needs_key
def test_absent_content_is_not_supported(probe_page, evidence_bbox):
    """False of the region and absent from the page image too, which is the
    axis the probe below deliberately does not test."""
    v = _judge(probe_page, evidence_bbox, "The system uses conformal calibration.")
    assert v.label != "supported", f"{v.label} @ {v.confidence}: {v.reason}"


@needs_key
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
