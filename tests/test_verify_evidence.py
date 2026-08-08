"""What a verify() claim is judged against: the region S4 actually returned.

Text regions are judged from their text; visual regions get the crop cut
from the page render. A claim is judged against its best region only.

(This is the S5 counterpart of tests/test_evidence.py, which tests the S4
evidence checkers. Two files, two slices, no name collision.)
"""

from PIL import Image

from visual_verify.contracts import GroundedRegion
from visual_verify.verify.evidence import best_region, build_evidence, crop_region


def make_region(page=0, bbox=(0.1, 0.2, 0.4, 0.5), score=1.0, modality="text", text="x"):
    return GroundedRegion(page=page, bbox=bbox, score=score, modality=modality, text=text)


def test_text_region_is_judged_from_text_only():
    r = make_region()
    ev = build_evidence(r, None)
    assert ev.text == "x"
    assert ev.image is None


def test_visual_region_crops_the_page_image():
    r = make_region(bbox=(0.25, 0.0, 0.75, 0.5), modality="visual")
    img = Image.new("RGB", (400, 200))
    ev = build_evidence(r, img)
    assert ev.image is not None
    assert ev.image.size == (200, 100)


def test_visual_region_without_image_keeps_text():
    r = make_region(modality="visual", text="chart label")
    ev = build_evidence(r, None)
    assert ev.text == "chart label"
    assert ev.image is None


def test_best_region_is_highest_score_ties_to_first():
    regions = [
        make_region(bbox=(0.0, 0.0, 0.1, 0.1), score=0.4, text="low"),
        make_region(bbox=(0.1, 0.0, 0.2, 0.1), score=0.9, text="high"),
        make_region(bbox=(0.2, 0.0, 0.3, 0.1), score=0.9, text="high tie"),
    ]
    assert best_region(regions).text == "high"


def test_best_region_of_empty_is_none():
    assert best_region([]) is None


def test_crop_region_clamps_to_integer_pixels():
    r = make_region(bbox=(0.1, 0.1, 0.9, 0.9), modality="visual")
    img = Image.new("RGB", (300, 300))
    assert crop_region(img, r.bbox).size == (240, 240)
