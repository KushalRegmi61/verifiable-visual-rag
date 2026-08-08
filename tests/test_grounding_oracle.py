"""The ceiling and the floor that every reported IoU must be quoted against.

Granularity caps IoU independently of the selector. When the gold box lies
inside the predicted box, IoU is just the ratio of their areas, so predicting a
line against a 3-word gold span cannot exceed about 0.195 however good the
selector is. Reporting an achieved IoU without this number turns a
near-ceiling result into an apparent failure. See spec section 8.
"""

import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import fitz
import pytest

from visual_verify.derive import block_boxes, line_boxes, span_boxes
from visual_verify.evidence import iou
from visual_verify.ingest.boxes import BoxRecord, extract_boxes, word_boxes

REAL_PDF = Path(__file__).parent.parent / "proposal_report" / "proposal.pdf"

pytestmark = pytest.mark.skipif(not REAL_PDF.exists(), reason="proposal.pdf not present")


def _holding(cands: list[BoxRecord], cx: float, cy: float) -> BoxRecord | None:
    """The first candidate whose rect contains the point (cx, cy), if any."""
    for c in cands:
        if c.x0 <= cx <= c.x1 and c.y0 <= cy <= c.y1:
            return c
    return None


@dataclass(frozen=True)
class Sample:
    gold: tuple[float, float, float, float]
    line: BoxRecord | None
    block: BoxRecord | None
    blocks: list[BoxRecord]


def _gold_and_containers(seed=0, pages=None) -> list[Sample]:
    """(gold span, containing line, containing block) samples from a real PDF."""
    if pages is None:
        pages = range(2, 10)
    doc = fitz.open(REAL_PDF)
    rng = random.Random(seed)
    out: list[Sample] = []
    for pno in pages:
        boxes = extract_boxes(doc[pno])
        if len(word_boxes(boxes)) < 50:
            continue
        lines, blocks = line_boxes(boxes), block_boxes(boxes)
        for ln in lines:
            words = ln.text.split()
            if len(words) < 5:
                continue
            i = rng.randrange(0, len(words) - 3)
            golds = span_boxes(boxes, " ".join(words[i : i + 3]))
            if not golds:
                continue
            g = golds[0]
            gold = (g.x0, g.y0, g.x1, g.y1)
            cx, cy = (g.x0 + g.x1) / 2, (g.y0 + g.y1) / 2
            out.append(
                Sample(
                    gold=gold,
                    line=_holding(lines, cx, cy),
                    block=_holding(blocks, cx, cy),
                    blocks=blocks,
                )
            )
    doc.close()
    return out


def test_line_granularity_ceiling_is_about_0_195():
    """Regression guard on the ceiling itself.

    If candidate derivation changes, this moves, and every previously reported
    grounding number silently changes meaning.
    """
    samples = _gold_and_containers()
    scores = [iou(s.gold, (s.line.x0, s.line.y0, s.line.x1, s.line.y1)) for s in samples if s.line]

    assert len(scores) > 50
    assert statistics.mean(scores) == pytest.approx(0.195, abs=0.03)


def test_block_granularity_ceiling_is_about_half_the_line_ceiling():
    samples = _gold_and_containers()
    block_scores = [
        iou(s.gold, (s.block.x0, s.block.y0, s.block.x1, s.block.y1)) for s in samples if s.block
    ]
    line_scores = [
        iou(s.gold, (s.line.x0, s.line.y0, s.line.x1, s.line.y1)) for s in samples if s.line
    ]

    assert statistics.mean(block_scores) == pytest.approx(0.097, abs=0.03)
    assert statistics.mean(block_scores) < statistics.mean(line_scores)


def test_the_random_candidate_floor_is_near_zero():
    """The baseline every grounding claim is measured against.

    A selector at or below this floor contributes nothing, whatever its IoU
    looks like in isolation.
    """
    samples = _gold_and_containers()
    rng = random.Random(1)
    scores = []
    for s in samples:
        pick = rng.choice(s.blocks)
        scores.append(iou(s.gold, (pick.x0, pick.y0, pick.x1, pick.y1)))

    assert statistics.mean(scores) < 0.02
    assert sum(sc >= 0.25 for sc in scores) / len(scores) < 0.05
