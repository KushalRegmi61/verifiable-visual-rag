"""answer_stream() is the loop; answer() is a drain over it.

The one property that matters is that they cannot diverge. A second copy of
the read-ground-verify loop would put the GroundingError recovery and the
`score < threshold` comparison in two places, and the copy the product hits
is the one no S5 test covers.
"""

from pathlib import Path

import pytest

from helpers import claim_list
from visual_verify.agent import AgentError, answer, answer_stream
from visual_verify.agent.events import AnswerComplete, ClaimsProduced, ClaimVerified, ReadingStarted
from visual_verify.agent.reader import shares_a_term
from visual_verify.agent.schemas import ClaimList, Verdict
from visual_verify.agent.types import FakeChat
from visual_verify.ingest.boxes import BoxRecord
from visual_verify.prepare import PreparedPage


def prepared(boxes, *, page=0, image="p.png", vectors=None, grid=None):
    """One PreparedPage, the real class the service and the CLI both build.

    The real dataclass rather than a stand-in, because the whole point of the
    argument becoming a page object is that the image, the boxes, the vectors
    and the grid travel together; a test double with the same five attributes
    would keep passing if they were ever pulled apart again.
    """
    return PreparedPage(
        doc_sha="0" * 64,
        doc_name="doc.pdf",
        page_no=page,
        image_path=Path(image),
        boxes=boxes,
        page_vectors=vectors,
        grid=grid,
    )


def page_boxes():
    """Two lines, both findable in the text layer.

    Defined here rather than imported from tests/test_agent.py: `tests/` has no
    __init__.py, so `from tests.test_agent import ...` depends on rootdir being
    on sys.path and breaks under a plain `pytest tests/test_answer_stream.py`.

    Every claim scripted below must be findable verbatim, or ground() falls
    through to the visual path, which needs page vectors and a grid and would
    raise GroundingError instead of exercising the loop.
    """
    first = ["Revenue", "grew", "42", "percent"]
    second = ["Margins", "held", "steady"]
    boxes = [
        BoxRecord(
            kind="word",
            x0=0.1 + i * 0.15,
            y0=0.10,
            x1=0.22 + i * 0.15,
            y1=0.16,
            text=t,
            block_no=0,
            line_no=0,
            word_no=i,
        )
        for i, t in enumerate(first)
    ]
    boxes += [
        BoxRecord(
            kind="word",
            x0=0.1 + i * 0.15,
            y0=0.30,
            x1=0.22 + i * 0.15,
            y1=0.36,
            text=t,
            block_no=0,
            line_no=1,
            word_no=i,
        )
        for i, t in enumerate(second)
    ]
    return boxes


def script():
    reader = FakeChat("r", [claim_list("Revenue grew 42 percent", "Margins held steady")])
    verifier = FakeChat(
        "v",
        [
            Verdict(label="supported", confidence=0.9, reason="matches"),
            Verdict(label="unsupported", confidence=0.8, reason="contradicted"),
        ],
    )
    return reader, verifier


def run_stream():
    reader, verifier = script()
    return list(
        answer_stream(
            "What happened?",
            [prepared(page_boxes())],
            reader_chat=reader,
            verifier_chat=verifier,
        )
    )


def test_the_event_order_is_reading_then_count_then_claims_then_complete():
    events = run_stream()

    assert isinstance(events[0], ReadingStarted)
    assert isinstance(events[1], ClaimsProduced)
    assert events[1].n == 2
    assert isinstance(events[2], ClaimVerified)
    assert isinstance(events[3], ClaimVerified)
    assert isinstance(events[4], AnswerComplete)
    assert len(events) == 5


def test_claim_events_are_indexed_in_order():
    events = [e for e in run_stream() if isinstance(e, ClaimVerified)]

    assert [e.index for e in events] == [0, 1]


def test_every_streamed_claim_already_has_a_verdict():
    """The reason S5 refused to stream the reader: a claim must never reach a
    consumer before the verifier has judged it. If this can fail, the product
    can display something it exists to withhold."""
    for event in run_stream():
        if isinstance(event, ClaimVerified):
            assert event.claim.label is not None


def test_answer_returns_exactly_what_the_stream_finished_with():
    """Pins the drain. If answer() ever grows its own loop, this fails."""
    streamed = [e for e in run_stream() if isinstance(e, AnswerComplete)][0].answer

    reader, verifier = script()
    direct = answer(
        "What happened?",
        [prepared(page_boxes())],
        reader_chat=reader,
        verifier_chat=verifier,
    )

    assert direct == streamed


def test_the_same_model_guard_raises_before_anything_is_iterated():
    """Eagerly, at call time, with no list() to drive the generator.

    The earlier version of this test wrapped the call in list(), which drives
    the generator to exhaustion and therefore passed whether the guard fired at
    call time, on first advance, or on the last claim. It could not distinguish
    the property it was named for.

    Eager matters concretely. The API reaches this through ask_events(), which
    yields a `retrieved` event first, so a guard that waited for first advance
    would raise after the 200 and its headers were committed and reach the
    browser as an SSE error frame instead of a refusal to start.
    """
    same = FakeChat("same", [claim_list("x")])
    other = FakeChat("same", [Verdict(label="supported", confidence=0.5, reason="r")])

    with pytest.raises(AgentError):
        answer_stream(
            "q",
            [prepared(page_boxes())],
            reader_chat=same,
            verifier_chat=other,
        )


def test_a_reader_that_returns_nothing_reports_a_count_of_zero():
    reader = FakeChat("r", [ClaimList(claims=[])])
    verifier = FakeChat("v", [])

    events = list(
        answer_stream(
            "q",
            [prepared(page_boxes())],
            reader_chat=reader,
            verifier_chat=verifier,
        )
    )

    assert isinstance(events[0], ReadingStarted)
    assert isinstance(events[1], ClaimsProduced)
    assert events[1].n == 0
    assert isinstance(events[2], AnswerComplete)
    assert events[2].answer.abstained_overall is True
    assert len(events) == 3


def test_the_paragraph_break_survives_the_trip_from_reader_to_claim():
    """The only place starts_paragraph crosses a layer boundary.

    Everything else about the field is schema-level, so deleting the kwarg in
    core.py, or hardcoding it to False, leaves the rest of the suite green while
    the displayed answer renders as one paragraph forever. That is this repo's
    recurring failure shape: correctly-shaped output, wrong value, nothing
    notices.

    Two claims, not one, and they disagree. A single-claim fixture passes
    against a hardcoded True as easily as against the real copy.
    """
    reader = FakeChat(
        "r",
        [
            ClaimList(
                claims=[
                    {"text": "Revenue grew 42 percent"},
                    {"text": "Margins held steady", "starts_paragraph": True},
                ]
            )
        ],
    )
    verifier = FakeChat(
        "v",
        [
            Verdict(label="supported", confidence=0.9, reason="matches"),
            Verdict(label="supported", confidence=0.9, reason="matches"),
        ],
    )

    out = answer(
        "What happened?",
        [prepared(page_boxes())],
        reader_chat=reader,
        verifier_chat=verifier,
    )

    assert out.claims[0].starts_paragraph is False
    assert out.claims[1].starts_paragraph is True


def test_a_non_sequitur_region_reaches_the_verifier_as_no_evidence():
    """THE test of the citation filter, and it pins WHERE the filter lives.

    Measured live on 2026-08-10: the visual path returned [0.526 0.940 0.535
    0.953] on proposal.pdf page 14, an 11 by 22 px box holding the page number,
    as the cited evidence for three different claims across two unrelated
    questions. The verifier scored two of them supported at 0.90 and 0.95,
    because it judges whether the CLAIM is true and the claims were true. A
    fabricated citation passed the abstention gate untouched.

    Asserted on the verifier's PROMPT rather than on the returned Claim,
    because that is the thing the filter has to change. A test that only
    checked `claim.regions` would pass with the filter moved into ground(),
    where it must not be: ground()'s contract says an empty list means no
    evidence exists on the page, and S7's ablation needs the grounder measured
    without it.

    The visual path is forced by scripting a claim that is NOT in the text
    layer, which is the only way a region can arrive that does not already
    contain the claim's own words.
    """
    import numpy as np

    from visual_verify.agent import answer
    from visual_verify.agent.verifier import NO_EVIDENCE
    from visual_verify.ingest.boxes import BoxRecord
    from visual_verify.retrieval.geometry import PatchGrid

    page_number = [
        BoxRecord(
            kind="word",
            x0=0.526,
            y0=0.940,
            x1=0.535,
            y1=0.953,
            text="7",
            block_no=0,
            line_no=0,
            word_no=0,
        )
    ]
    grid = PatchGrid(n_x=4, n_y=4, offset=0, n_vectors=16)
    page_vectors = np.ones((16, 8), dtype=np.float32)
    reader = FakeChat("r", [claim_list("Revenue grew 42 percent")])
    verifier = FakeChat(
        "v", [Verdict(label="supported", confidence=0.95, reason="looks fine to me")]
    )

    out = answer(
        "q",
        [prepared(page_number, image="page.png", vectors=page_vectors, grid=grid)],
        reader_chat=reader,
        verifier_chat=verifier,
        threshold=0.0,
        embed_query=lambda _: np.ones((2, 8), dtype=np.float32),
    )

    prompt = verifier.calls[0].prompt
    assert NO_EVIDENCE in prompt, (
        "the page-number region was cited to the verifier instead of being dropped"
    )
    assert "0.526" not in prompt, "the dropped region's geometry still reached the verifier"
    assert out.claims[0].regions == []


def test_a_region_that_names_nothing_in_the_claim_is_not_cited():
    """The page-number sink, pinned at the layer that decides what to cite.

    A GroundedRegion whose text shares no term with the claim reaches the
    verifier as no evidence at all, so the rubric returns insufficient_evidence
    and the gate withholds the claim. Before this, the region was cited, the
    verifier judged only whether the CLAIM was true, and a true claim with a
    fabricated citation passed at 0.95.

    Driven through answer() rather than by calling the filter directly, because
    the filter living in the wrong place is the failure mode: ground() must not
    do it, and a unit test of the predicate cannot tell where it was applied.
    """
    from visual_verify.agent import answer

    reader = FakeChat("r", [claim_list("Revenue grew 42 percent")])
    verifier = FakeChat(
        "v", [Verdict(label="supported", confidence=0.95, reason="looks fine to me")]
    )

    out = answer(
        "q",
        [prepared(page_boxes(), image="page.png")],
        reader_chat=reader,
        verifier_chat=verifier,
        threshold=0.0,
    )

    # The text path finds this claim verbatim, so it is cited normally and the
    # filter is not what produced this. That is the control half.
    assert out.claims[0].regions, "the verbatim claim should still carry its region"
    assert all(shares_a_term(out.claims[0].text, r.text) for r in out.claims[0].regions), (
        "every cited region must name something the claim names"
    )


def test_the_verifier_sees_no_evidence_when_every_region_was_a_non_sequitur():
    """The half the fixture above cannot reach, because the text path always
    finds a real line. Asserts the CONTRACT: what the verifier is handed is the
    filtered list, so a page-number-only grounding arrives as empty rather than
    as a citation the verifier will rubber-stamp."""
    from visual_verify.contracts import GroundedRegion

    page_number = GroundedRegion(
        page=0,
        bbox=(0.526, 0.940, 0.535, 0.953),
        score=1.0,
        modality="visual",
        text="7",
        resolution="line",
    )
    claim = "The three evaluation metrics are answer accuracy and grounding overlap."

    assert [r for r in [page_number] if shares_a_term(claim, r.text)] == []


def test_a_claim_with_no_region_is_withheld_whatever_the_verifier_said():
    """Structural, because the prompt instruction is not obeyed reliably.

    The verifier prompt asks for insufficient_evidence when no regions are
    listed. Measured on proposal.pdf page 14, it complied for one region-less
    claim (insufficient_evidence at 1.00) and not for another in the SAME
    answer (supported at 0.90). A sentence displayed with no region is an
    answer with no evidence behind it, which is the one thing this project
    exists to refuse, so it cannot rest on the model cooperating.

    The raw verdict stays on `label` for the S7 eval; only the display gate
    moves.
    """
    import numpy as np

    from visual_verify.agent import answer
    from visual_verify.ingest.boxes import BoxRecord
    from visual_verify.retrieval.geometry import PatchGrid

    page_number = [
        BoxRecord(
            kind="word",
            x0=0.526,
            y0=0.940,
            x1=0.535,
            y1=0.953,
            text="8",
            block_no=0,
            line_no=0,
            word_no=0,
        )
    ]
    reader = FakeChat("r", [claim_list("Revenue grew 42 percent")])
    verifier = FakeChat(
        "v", [Verdict(label="supported", confidence=1.0, reason="I like it anyway")]
    )

    out = answer(
        "q",
        [
            prepared(
                page_number,
                image="page.png",
                vectors=np.ones((16, 8), dtype=np.float32),
                grid=PatchGrid(n_x=4, n_y=4, offset=0, n_vectors=16),
            )
        ],
        reader_chat=reader,
        verifier_chat=verifier,
        threshold=0.0,
        embed_query=lambda _: np.ones((2, 8), dtype=np.float32),
    )

    claim = out.claims[0]
    assert claim.regions == []
    assert claim.label == "supported", "the verifier's raw verdict must survive for S7"
    assert claim.withheld is True, "a claim with no evidence must not be displayable"
    assert out.shown == []


def _visual_page(word_text, *, page, image, magnitude=1.0):
    """A page whose ONLY route to a region is the heatmap.

    One word, so nothing on it matches a whole claim verbatim and the text path
    comes back empty, and vectors scaled by `magnitude` so two such pages have
    genuinely different MaxSim scores rather than a tie broken by iteration
    order. The word shares a term with the claims used below on purpose: a
    region that names nothing the claim names is dropped by the citation
    filter, and a test whose wrong answer is filtered away silently stops
    testing the thing it is named for.
    """
    import numpy as np

    from visual_verify.retrieval.geometry import PatchGrid

    box = BoxRecord(
        kind="word",
        x0=0.4,
        y0=0.9,
        x1=0.6,
        y1=0.95,
        text=word_text,
        block_no=0,
        line_no=0,
        word_no=0,
    )
    return prepared(
        [box],
        page=page,
        image=image,
        vectors=np.full((16, 8), magnitude, dtype=np.float32),
        grid=PatchGrid(n_x=4, n_y=4, offset=0, n_vectors=16),
    )


def _ones_query(_claim):
    import numpy as np

    return np.ones((2, 8), dtype=np.float32)


def test_a_text_hit_on_a_later_page_beats_a_visual_hit_on_the_first():
    """Score scales are not comparable. A text-path region's score comes from
    span matching and a visual one's from MaxSim, so max() across both is
    meaningless. An exact match wins outright wherever it is.

    Deliberately rigged so the two are not merely different but INVERTED. The
    text region scores EXACT = 1.0; the visual region here scores 8.0, because
    relevance is a raw dot product and these vectors are not unit-norm. A
    max() over the pooled list therefore cites page 0's page-number-sized box
    for a claim written out word for word on page 1, and does it while looking
    like a confident answer.
    """
    visual = _visual_page("Revenue", page=0, image="page0.png")
    textual = prepared(page_boxes(), page=1, image="page1.png")
    reader = FakeChat("r", [claim_list("Revenue grew 42 percent")])
    verifier = FakeChat("v", [Verdict(label="supported", confidence=0.9, reason="matches")])

    out = answer(
        "q",
        [visual, textual],
        reader_chat=reader,
        verifier_chat=verifier,
        threshold=0.0,
        embed_query=_ones_query,
    )

    regions = out.claims[0].regions
    assert regions, "the claim is on page 1 verbatim and must be cited"
    assert [r.page for r in regions] == [1]
    assert [r.modality for r in regions] == ["text"]


def test_the_first_pages_visual_region_really_would_have_won_on_score():
    """The control the test above needs to mean anything.

    Without it, "page 1 was cited" is satisfied by a page 0 that produced no
    region at all, and the ordering rule would never have been exercised. This
    pins that page 0 does produce a region, and that its score is above the
    text path's EXACT 1.0, so pooling the two lists and taking max() would pick
    it.
    """
    reader = FakeChat("r", [claim_list("Revenue grew 42 percent")])
    verifier = FakeChat("v", [Verdict(label="supported", confidence=0.9, reason="matches")])

    out = answer(
        "q",
        [_visual_page("Revenue", page=0, image="page0.png")],
        reader_chat=reader,
        verifier_chat=verifier,
        threshold=0.0,
        embed_query=_ones_query,
    )

    regions = out.claims[0].regions
    assert [r.modality for r in regions] == ["visual"]
    assert regions[0].score > 1.0, "the visual score must beat EXACT for the trap to exist"


def test_the_claim_is_verified_against_the_page_its_region_came_from():
    """verify() takes ONE image. Handing it the top page while the region is on
    page 3 asks it to check a box it cannot see, which is a fabricated citation
    by a different route.

    The verifier would not complain. It is shown a rectangle and a claim, and
    with the rectangle pointing at nothing on the image in front of it, it
    falls back on whether the claim sounds true. That is precisely how the
    page-number citation came back supported at 0.95.
    """
    visual = _visual_page("Revenue", page=0, image="page0.png")
    textual = prepared(page_boxes(), page=1, image="page1.png")
    reader = FakeChat("r", [claim_list("Revenue grew 42 percent")])
    verifier = FakeChat("v", [Verdict(label="supported", confidence=0.9, reason="matches")])

    answer(
        "q",
        [visual, textual],
        reader_chat=reader,
        verifier_chat=verifier,
        threshold=0.0,
        embed_query=_ones_query,
    )

    assert verifier.calls[0].image_paths == [Path("page1.png")]


def test_the_reader_sees_every_prepared_page():
    """The whole slice is inert if it does not. A reader shown only the top
    page writes a fluent answer that simply misses the evidence two pages
    later, and every claim it drafts still grounds and still verifies, so
    nothing anywhere reports a problem."""
    pages = [
        _visual_page("Revenue", page=0, image="page0.png"),
        prepared(page_boxes(), page=1, image="page1.png"),
    ]
    reader = FakeChat("r", [claim_list("Revenue grew 42 percent")])
    verifier = FakeChat("v", [Verdict(label="supported", confidence=0.9, reason="matches")])

    answer(
        "q",
        pages,
        reader_chat=reader,
        verifier_chat=verifier,
        threshold=0.0,
        embed_query=_ones_query,
    )

    assert reader.calls[0].image_paths == [Path("page0.png"), Path("page1.png")]


def test_visual_scores_are_compared_across_pages_when_no_page_has_a_text_hit():
    """Once no page matches the claim in its text layer, the scores in play are
    all MaxSim over patch embeddings, which IS one quantity, so the highest
    wins wherever it sits.

    Run twice with the magnitudes swapped between the pages. One run alone
    cannot tell "picks the strongest" apart from "picks the first" or "picks
    the last", and both of those are one-line mistakes.
    """
    for stronger in (0, 1):
        pages = [
            _visual_page(
                "Revenue",
                page=0,
                image="page0.png",
                magnitude=1.0 if stronger == 0 else 0.25,
            ),
            _visual_page(
                "Revenue",
                page=1,
                image="page1.png",
                magnitude=1.0 if stronger == 1 else 0.25,
            ),
        ]
        reader = FakeChat("r", [claim_list("Margins outpaced Revenue everywhere")])
        verifier = FakeChat("v", [Verdict(label="supported", confidence=0.9, reason="matches")])

        out = answer(
            "q",
            pages,
            reader_chat=reader,
            verifier_chat=verifier,
            threshold=0.0,
            embed_query=_ones_query,
        )

        regions = out.claims[0].regions
        assert [r.page for r in regions] == [stronger]
        assert [r.modality for r in regions] == ["visual"]


def test_one_unembedded_page_does_not_lose_the_evidence_on_the_others():
    """GroundingError is caught PER PAGE. It is raised when the visual path has
    no vectors, which is exactly what a page that was ingested but never
    embedded looks like, and a document is routinely embedded page by page. If
    it escaped the loop, the region sitting on the embedded page would be
    thrown away and the claim would come back insufficient_evidence with
    nothing saying why."""
    unembedded = prepared(
        [
            BoxRecord(
                kind="word",
                x0=0.4,
                y0=0.9,
                x1=0.6,
                y1=0.95,
                text="Revenue",
                block_no=0,
                line_no=0,
                word_no=0,
            )
        ],
        page=0,
        image="page0.png",
    )
    embedded = _visual_page("Revenue", page=1, image="page1.png")
    reader = FakeChat("r", [claim_list("Margins outpaced Revenue everywhere")])
    verifier = FakeChat("v", [Verdict(label="supported", confidence=0.9, reason="matches")])

    out = answer(
        "q",
        [unembedded, embedded],
        reader_chat=reader,
        verifier_chat=verifier,
        threshold=0.0,
        embed_query=_ones_query,
    )

    regions = out.claims[0].regions
    assert [r.page for r in regions] == [1]
    assert out.claims[0].withheld is False


def test_no_pages_is_refused_before_anything_is_read():
    """Eagerly, like the same-model guard, and for the same reason: it reaches
    the browser through ask_events, which has already yielded a retrieved event
    and committed a 200."""
    reader = FakeChat("r", [claim_list("x")])
    verifier = FakeChat("v", [])

    with pytest.raises(AgentError):
        answer_stream("q", [], reader_chat=reader, verifier_chat=verifier)


def test_two_text_hits_break_the_tie_by_retrieval_order_not_page_number():
    """Both pages match the claim verbatim, so both score EXACT = 1.0. The one
    retrieval ranked first wins, and `pages` must never be sorted.

    Page numbers DESCEND against list order on purpose, and that is the whole
    point of the test. Every other multi-page fixture in this file is
    [page=0, page=1], where pages[0], min(page_no) and sorted(pages)[0] all name
    the same page, so a tidy-up to `for page in sorted(pages, key=page_no)`
    passes the entire suite while silently citing the wrong page. This project
    has now shipped that shape twice: the S3 patch grid, where a transposition
    survived because the fixture was square, and the S6 toStyle test, which
    passed under the transposition it was written to catch. Ascending
    expectations do not test ordering.
    """
    pages = [
        prepared(page_boxes(), page=7, image="p7.png"),
        prepared(page_boxes(), page=2, image="p2.png"),
    ]
    reader = FakeChat("r", [claim_list("Revenue grew 42 percent")])
    verifier = FakeChat("v", [Verdict(label="supported", confidence=0.9, reason="matches")])

    out = answer(
        "q",
        pages,
        reader_chat=reader,
        verifier_chat=verifier,
        threshold=0.0,
    )

    regions = out.claims[0].regions
    assert [r.page for r in regions] == [7], "the tie must break by rank, not by page number"
    # The verifier follows the winner. A sort would send it p2.png, and it
    # would judge a page-7 box against page 2's image without complaining.
    assert verifier.calls[0].image_paths == [Path("p7.png")]


def test_a_single_character_region_is_not_cited_to_the_verifier():
    """Pins WHERE the degeneracy check runs, not just that it exists.

    The claim names "8" and so does the region, so `shares_a_term` passes it:
    this is the numeric coincidence the term check structurally cannot catch. A
    unit test of the predicate alone would not notice if the filter were never
    wired into the loop.
    """
    import numpy as np

    from visual_verify.agent import answer
    from visual_verify.agent.verifier import NO_EVIDENCE
    from visual_verify.ingest.boxes import BoxRecord
    from visual_verify.retrieval.geometry import PatchGrid

    page_number = [
        BoxRecord(
            kind="word",
            x0=0.526,
            y0=0.940,
            x1=0.535,
            y1=0.953,
            text="8",
            block_no=0,
            line_no=0,
            word_no=0,
        )
    ]
    reader = FakeChat("r", [claim_list("Figure 8 shows the overall pipeline")])
    verifier = FakeChat(
        "v", [Verdict(label="supported", confidence=1.0, reason="looks fine to me")]
    )

    out = answer(
        "q",
        [
            prepared(
                page_number,
                page=0,
                image="page0.png",
                vectors=np.ones((16, 8), dtype=np.float32),
                grid=PatchGrid(n_x=4, n_y=4, offset=0, n_vectors=16),
            )
        ],
        reader_chat=reader,
        verifier_chat=verifier,
        threshold=0.0,
        embed_query=lambda _: np.ones((2, 8), dtype=np.float32),
    )

    assert NO_EVIDENCE in verifier.calls[0].prompt
    assert out.claims[0].regions == []
    assert out.claims[0].withheld is True
