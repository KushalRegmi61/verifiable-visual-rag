import pytest
from pydantic import ValidationError

from visual_verify.contracts import Answer, Claim, GroundedRegion, RetrievedPage


def test_grounded_region_accepts_normalized_bbox():
    r = GroundedRegion(page=3, bbox=(0.1, 0.2, 0.4, 0.25), score=0.9, modality="text",
                       text="Revenue grew 42 percent")
    assert r.bbox == (0.1, 0.2, 0.4, 0.25)
    assert r.modality == "text"


def test_grounded_region_rejects_out_of_range_bbox():
    with pytest.raises(ValidationError):
        GroundedRegion(page=1, bbox=(0.1, 0.2, 1.4, 0.25), score=0.9, modality="text")


def test_grounded_region_rejects_inverted_bbox():
    with pytest.raises(ValidationError):
        GroundedRegion(page=1, bbox=(0.5, 0.2, 0.3, 0.25), score=0.9, modality="text")


def test_grounded_region_rejects_unknown_modality():
    with pytest.raises(ValidationError):
        GroundedRegion(page=1, bbox=(0.1, 0.2, 0.3, 0.25), score=0.9, modality="pixels")


def test_claim_and_answer_nest():
    r = GroundedRegion(page=1, bbox=(0.1, 0.2, 0.3, 0.25), score=0.9, modality="visual",
                       crop_ref="crops/p1_0.png")
    c = Claim(text="Revenue grew 42 percent", regions=[r], confidence=0.8, abstained=False)
    a = Answer(question="How much did revenue grow?", claims=[c], abstained_overall=False)
    assert a.claims[0].regions[0].crop_ref == "crops/p1_0.png"


def test_retrieved_page_defaults_text_layer_to_none():
    p = RetrievedPage(doc_id="abc", page=2, image_ref="data/pages/abc/p0002.png", score=0.5)
    assert p.text_layer is None
