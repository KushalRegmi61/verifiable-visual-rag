"""verify() orchestration: order, force forwarding, the gate, abstention."""

import numpy as np
import pytest
from PIL import Image

from visual_verify.contracts import Answer
from visual_verify.ingest.boxes import BoxRecord
from visual_verify.retrieval.geometry import PatchGrid
from visual_verify.verify.claims import ReaderOutput
from visual_verify.verify.core import verify
from visual_verify.verify.errors import VerifierError
from visual_verify.verify.evidence import Evidence
from visual_verify.verify.rubric import Judgement


class FakeReader:
    def __init__(self, output: ReaderOutput):
        self.output = output
        self.seen: list[tuple] = []

    def read(self, question, image, text_layer):
        self.seen.append((question, image, text_layer))
        return self.output


class FakeVerifier:
    def __init__(self, label: str = "supported"):
        self.label = label
        self.seen: list[tuple[str, Evidence]] = []

    def judge(self, claim: str, evidence: Evidence) -> Judgement:
        self.seen.append((claim, evidence))
        return Judgement(label=self.label)


def make_page_image():
    img = Image.new("RGB", (100, 100))
    for y in range(10, 30):
        for x in range(20, 60):
            img.putpixel((x, y), (255, 0, 0))
    return img


def make_box():
    return BoxRecord(kind="word", x0=0.2, y0=0.3, x1=0.6, y1=0.6, text="c", word_no=0)


def make_grid():
    return PatchGrid(n_x=4, n_y=3, offset=2, n_vectors=2 + 12 + 1)


def test_verify_reads_grounds_judges_and_returns_answer_shape():
    reader = FakeReader(ReaderOutput(answer="42.", claims=["The answer is 42."]))
    verifier = FakeVerifier("supported")
    ans = verify(
        "What is the answer?",
        reader,
        verifier,
        page=3,
        image=make_page_image(),
        text_layer="The answer is 42.",
        boxes=[],
    )
    assert isinstance(ans, Answer)
    assert len(ans.claims) == 1
    c = ans.claims[0]
    assert c.text == "The answer is 42."
    assert c.confidence == 1.0
    assert not c.abstained
    assert not ans.abstained_overall
    assert len(reader.seen) == 1
    assert reader.seen[0][0] == "What is the answer?"


def test_reader_receives_question_image_and_text_layer():
    reader = FakeReader(ReaderOutput(answer="a", claims=["c"]))
    img = make_page_image()
    verify("Q", reader, FakeVerifier(), page=0, image=img, text_layer="layer", boxes=[])
    q, image, layer = reader.seen[0]
    assert q == "Q"
    assert image is img
    assert layer == "layer"


def test_weak_labels_abstain_the_claim():
    ans = verify(
        "Q",
        FakeReader(ReaderOutput(answer="a", claims=["c1", "c2"])),
        FakeVerifier("unsupported"),
        page=0, image=make_page_image(), text_layer="", boxes=[], threshold=0.5,
    )
    assert [c.abstained for c in ans.claims] == [True, True]
    assert ans.abstained_overall
    assert [c.confidence for c in ans.claims] == [0.0, 0.0]


def test_partial_below_threshold_abstains_partial_above_passes():
    reader = FakeReader(ReaderOutput(answer="a", claims=["c"]))
    ans = verify("Q", reader, FakeVerifier("partial"), page=0, image=make_page_image(),
                 text_layer="", boxes=[], threshold=0.6)
    assert ans.claims[0].abstained
    ans2 = verify("Q", reader, FakeVerifier("partial"), page=0, image=make_page_image(),
                  text_layer="", boxes=[], threshold=0.5)
    assert not ans2.claims[0].abstained


def test_zero_claims_abstains_the_whole_answer():
    ans = verify("Q", FakeReader(ReaderOutput(answer="No idea.", claims=[])), FakeVerifier(),
                 page=0, image=make_page_image(), text_layer="", boxes=[])
    assert ans.abstained_overall
    assert ans.claims == []


def test_reader_failure_raises_with_role_named():
    class BrokenReader:
        def read(self, question, image, text_layer):
            raise OSError("no network")

    with pytest.raises(VerifierError, match="reader"):
        verify("Q", BrokenReader(), FakeVerifier(), page=0, image=make_page_image(),
               text_layer="", boxes=[])


def test_verifier_failure_raises_with_role_named():
    class BrokenVerifier:
        def judge(self, claim, evidence):
            raise OSError("oom")

    with pytest.raises(VerifierError, match="verifier"):
        verify("Q", FakeReader(ReaderOutput(answer="a", claims=["c"])), BrokenVerifier(),
               page=0, image=make_page_image(), text_layer="", boxes=[])


def test_text_grounding_feeds_the_verifier_the_matched_span():
    reader = FakeReader(ReaderOutput(answer="a", claims=["c"]))
    verifier = FakeVerifier("supported")
    ans = verify("Q", reader, verifier, page=0, image=make_page_image(),
                 text_layer="", boxes=[make_box()])
    assert ans.claims[0].regions
    claim, evidence = verifier.seen[0]
    assert claim == "c"
    assert evidence.text == "c"


def test_visual_path_requires_embed_and_vectors():
    reader = FakeReader(ReaderOutput(answer="a", claims=["c"]))
    with pytest.raises(VerifierError, match="embed"):
        verify("Q", reader, FakeVerifier(), page=0, image=make_page_image(),
               text_layer="", boxes=[make_box()], force="visual")


def test_force_visual_embeds_the_claim_and_uses_it():
    reader = FakeReader(ReaderOutput(answer="a", claims=["c"]))
    embed_calls = []

    def embed(claim):
        embed_calls.append(claim)
        return np.random.default_rng(0).normal(size=(3, 8))

    grid = make_grid()
    page_vecs = np.random.default_rng(1).normal(size=(grid.n_vectors, 8))
    ans = verify(
        "Q", reader, FakeVerifier(), page=0, image=make_page_image(),
        text_layer="", boxes=[make_box()], embed=embed, page_vectors=page_vecs,
        grid=grid, force="visual",
    )
    assert embed_calls == ["c"]
    assert ans.claims[0].regions
    assert ans.claims[0].regions[0].modality == "visual"


def test_no_evidence_claim_still_gets_judged_and_can_abstain():
    """A claim with no match and no candidates reaches the verifier with
    no evidence; the label decides. Nothing is dropped silently."""
    verifier = FakeVerifier("insufficient")
    ans = verify("Q", FakeReader(ReaderOutput(answer="a", claims=["c"])), verifier,
                 page=0, image=make_page_image(), text_layer="", boxes=[])
    assert ans.claims[0].regions == []
    assert verifier.seen[0][1].text is None
    assert verifier.seen[0][1].image is None
    assert ans.claims[0].abstained
    assert ans.abstained_overall
