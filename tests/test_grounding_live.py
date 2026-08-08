"""Scoring bake-off against the real indexed corpus.

Design spec section 6.1 requires the scoring rule to be chosen by measurement,
not by argument. Vectors are already stored in Qdrant, so this costs one model
load for query embeddings and is otherwise pure numpy: no page re-embedding.

Three rounds of measurement each found a different flaw in how this comparison
was posed, and the final shape below is what survived all three:

  1. A three-page sample, one query per page, tied 1/3-1/3-1/3 and separated
     nothing.
  2. Widened to the whole corpus but queried with each page's three LONGEST
     lines. That handed the bake-off to dense sum, because sum prefers large
     candidates and the gold answer was, by construction, always one of the
     longest lines on the page. That measured the sampling, not the rule, so
     sampled lines are now drawn at RANDOM (fixed seed), never sorted by
     length.
  3. Scored by `covers_text` containment, which only penalizes MISSING the
     gold text and cannot penalize over-covering, so a rule with a genuine
     area bias can WIN by swallowing the gold line inside a needlessly large
     region. Scoring is now IoU against the gold line's own box, the metric
     `proposal.tex` line 434 and spec section 8 actually specify, because it
     penalizes both misses and over-covering.

Corrected for all three, the measurement contradicts the spec: attribution
mean leads dense mean on two-stage IoU, which the sparsity argument did not
predict. The scoring rule is NOT flipped on this result. All 193 trials come
from one document, the project's own proposal PDF, a homogeneous single-column
A4 report; S7 evaluates on SlideVQA, landscape slides with sparse text and
heavy figures. This corpus is strong enough to say the spec's argument for
dense over attribution was unsupported. It is not strong enough to say which
rule is right on the corpus that actually matters, so the choice is recorded
here and deferred to S7's benchmark evaluation rather than decided on n=1.

The whole corpus is scrolled (up to SCROLL_LIMIT points), not a handful, and
each qualifying page is queried with several of its own lines so the tally
measures the rules rather than which pages happened to be easy.

Two tallies are reported. "flat line ranking" ranks every line on the page in
one pass, which is not what the system does. "two-stage snap" runs the real
`snap_to_box`: blocks first, then lines within the winning block. Flat and
two-stage disagree measurably (see test_two_stage_trades_iou_for_bounded_error
below), so results are reported for both and the two-stage numbers, being the
production path, are what any assertion is made against.
"""

import gc
import random
import sqlite3
import statistics
from pathlib import Path

import numpy as np
import pytest

from visual_verify.config import Settings

pytestmark = pytest.mark.slow

INDEX_DB = Path("data/index.db")
MIN_WORD_BOXES = 40
SCROLL_LIMIT = 128
MIN_LINE_WORDS = 8
LINES_PER_PAGE = 3
RANDOM_FLOOR_MULTIPLE = 5
SAMPLE_SEED = 0


def _settings_or_skip() -> Settings:
    settings = Settings.from_env()
    if not settings.qdrant_url:
        pytest.skip("VVRAG_QDRANT_URL not set")
    return settings


def _word_boxes_for(conn: sqlite3.Connection, doc_sha: str, page_no: int) -> list:
    from visual_verify.ingest.boxes import BoxRecord

    rows = conn.execute(
        """
        SELECT b.x0, b.y0, b.x1, b.y1, b.text, b.block_no, b.line_no, b.word_no
        FROM boxes b
        JOIN pages p ON b.page_id = p.id
        WHERE p.doc_sha = ? AND p.page_no = ? AND b.kind = 'word'
        """,
        (doc_sha, page_no),
    ).fetchall()
    return [
        BoxRecord(
            kind="word",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            text=text,
            block_no=block_no,
            line_no=line_no,
            word_no=word_no,
        )
        for x0, y0, x1, y1, text, block_no, line_no, word_no in rows
    ]


@pytest.fixture(scope="module")
def live():
    """(embedder, pages, meta).

    pages = [(page_vectors, grid, boxes), ...] for every scrolled page with at
    least MIN_WORD_BOXES word boxes (pages below that are mostly figures, where
    line ranking is not meaningful; the threshold is not lowered to manufacture
    a bigger sample). meta = {"scrolled": N, "qualifying": M} so the sample size
    is visible in test output rather than implied.

    Module-scoped so the model loads once for the whole file. Guards, in
    order: env var present, local index.db present, CUDA present, at least one
    qualifying page.
    """
    settings = _settings_or_skip()
    if not INDEX_DB.exists():
        pytest.skip(f"{INDEX_DB} not present; run `vvrag ingest` first")

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")

    from visual_verify.retrieval.embedder import ColQwen2Embedder
    from visual_verify.retrieval.geometry import PatchGrid
    from visual_verify.retrieval.index import ORIGINAL, QdrantIndex

    index = QdrantIndex(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    points, _ = index.client.scroll(
        index.collection, limit=SCROLL_LIMIT, with_payload=True, with_vectors=True
    )
    if not points:
        pytest.skip("Qdrant collection has no points")

    conn = sqlite3.connect(str(INDEX_DB))
    pages = []
    try:
        for point in points:
            payload = point.payload
            doc_sha, page_no = payload["doc_sha"], payload["page_no"]
            boxes = _word_boxes_for(conn, doc_sha, page_no)
            if len(boxes) < MIN_WORD_BOXES:
                continue
            page_vectors = np.asarray(point.vector[ORIGINAL], dtype=np.float32)
            grid = PatchGrid(
                n_x=payload["n_patches_x"],
                n_y=payload["n_patches_y"],
                offset=payload["patch_offset"],
                n_vectors=page_vectors.shape[0],
            )
            pages.append((page_vectors, grid, boxes))
    finally:
        conn.close()

    meta = {"scrolled": len(points), "qualifying": len(pages)}
    if not pages:
        pytest.skip(f"no scrolled page (of {meta['scrolled']}) had >= {MIN_WORD_BOXES} word boxes")

    embedder = ColQwen2Embedder()
    yield embedder, pages, meta
    # Two live ColQwen2 instances do not fit in 4 GB of VRAM. Without freeing
    # here, this module passes alone and the next model-loading module OOMs
    # during weight loading, exactly the failure CLAUDE.md documents for
    # test_known_item_retrieval.py and test_embedder.py.
    del embedder
    gc.collect()
    torch.cuda.empty_cache()


def _sample_lines(boxes, rng: random.Random):
    """(all lines on the page, up to LINES_PER_PAGE lines with >= 8 words,
    chosen at RANDOM). Not sorted by length anywhere: a length-biased sample
    would hand an area-biased rule a win it did not earn. See the module
    docstring for the earlier version that got this wrong."""
    from visual_verify.derive import line_boxes

    all_lines = line_boxes(boxes)
    candidates = [ln for ln in all_lines if len(ln.text.split()) >= MIN_LINE_WORDS]
    k = min(LINES_PER_PAGE, len(candidates))
    sample = rng.sample(candidates, k) if k else []
    return all_lines, sample


def _expected_random_rate(trials) -> float:
    """Mean of 1/len(candidates) over trials: the average random-pick hit
    probability, used as the floor both known-item tests measure against."""
    return statistics.mean(1 / len(all_lines) for _, _, _, all_lines, _ in trials)


def _trials(pages):
    """(page_vectors, grid, boxes, all_lines, line) per (page, sampled line).

    One Random instance shared across the whole corpus scan, seeded fixed, so
    a run is reproducible: which lines get sampled depends only on page order
    and this seed, never on wall-clock or hash randomization.
    """
    rng = random.Random(SAMPLE_SEED)
    out = []
    for page_vectors, grid, boxes in pages:
        all_lines, sample = _sample_lines(boxes, rng)
        for line in sample:
            out.append((page_vectors, grid, boxes, all_lines, line))
    return out


def test_the_selector_beats_the_random_candidate_floor(live, capsys):
    """Known-item retrieval: each query is copied verbatim from one of its own
    page's lines (>= 8 words, chosen at random), so the correct answer is known
    in advance. The assertion checks covered TEXT, not ink: every candidate box
    on a real page contains ink (measured 435/435 on a representative page), so
    an ink check passes a random selector as often as a correct one and proves
    nothing.

    The assertion is against the measured RANDOM floor, not an absolute rate.
    Flat line ranking is expected to be mediocre in absolute terms because the
    patch grid resolves only about 3.6 lines per patch row, so most of a page's
    lines are indistinguishable to it; a page with ~60 lines has a random floor
    of about 1.7%, and even a modest-looking hit rate can be 20x that. The
    ratio to the floor is the real evidence, not the raw rate.
    """
    from visual_verify.evidence import covers_text
    from visual_verify.grounding.heatmap import dense_relevance
    from visual_verify.grounding.snap import rank_candidates

    embedder, pages, meta = live
    trials = _trials(pages)
    assert trials, "no page produced a line with >= 8 words to query with"

    hits = 0
    floors = []
    for page_vectors, grid, boxes, all_lines, line in trials:
        query_vectors = embedder.embed_query(line.text)
        relevance = dense_relevance(query_vectors, page_vectors, grid)
        ranked = rank_candidates(relevance, grid, all_lines, reduce="mean")
        top_box, _ = ranked[0]
        hits += covers_text(boxes, top_box.bbox, line.text)
        floors.append(1 / len(all_lines))

    measured_rate = hits / len(trials)
    expected_floor = statistics.mean(floors)
    ratio = measured_rate / expected_floor if expected_floor > 0 else float("inf")

    with capsys.disabled():
        print(
            f"\nknown-item floor: {meta['qualifying']}/{meta['scrolled']} pages qualified "
            f"(>= {MIN_WORD_BOXES} word boxes), {len(trials)} (page, line) trials. "
            f"hits={hits}/{len(trials)} measured_rate={measured_rate:.2%} "
            f"expected_random_rate={expected_floor:.2%} ratio={ratio:.1f}x"
        )

    assert measured_rate >= RANDOM_FLOOR_MULTIPLE * expected_floor, (
        f"measured hit rate {measured_rate:.2%} is not at least "
        f"{RANDOM_FLOOR_MULTIPLE}x the expected random rate {expected_floor:.2%} "
        f"(ratio {ratio:.1f}x); the selector is not clearly beating chance on this corpus"
    )


def test_query_token_counts_and_heatmap_sparsity_are_recorded(live, capsys):
    """Records the sparsity numbers section 6.1 cites, on the live corpus
    rather than a fixture, so the design's numbers are traceable to this run.
    """
    from visual_verify.grounding.heatmap import attribution

    embedder, pages, meta = live
    trials = _trials(pages)
    fractions = []
    token_counts = []

    with capsys.disabled():
        print(
            f"\nsparsity over {meta['qualifying']}/{meta['scrolled']} qualifying pages, "
            f"{len(trials)} (page, line) trials:"
        )
        for page_vectors, grid, _boxes, _all_lines, line in trials:
            query_vectors = embedder.embed_query(line.text)
            n_tokens = query_vectors.shape[0]
            attr = attribution(query_vectors, page_vectors, grid)
            n_lit = int((attr > 0).sum())
            fraction = n_lit / grid.n_image_patches
            fractions.append(fraction)
            token_counts.append(n_tokens)

        print(
            f"query tokens: mean={statistics.mean(token_counts):.1f} "
            f"min={min(token_counts)} max={max(token_counts)}"
        )
        print(
            f"lit fraction of grid: mean={statistics.mean(fractions):.2%} max={max(fractions):.2%}"
        )

    max_fraction = max(fractions)
    assert max_fraction < 0.10, (
        f"attribution lit {max_fraction:.2%} of the grid on the worst trial; if this is "
        "no longer under 10%, attribution is no longer sparse and spec section 6.1's "
        "reasoning for excluding it from ranking needs revisiting"
    )


def _word_count(text: str) -> int:
    return len(text.split())


def _compute_bakeoff(live):
    """Shared trial loop for the two bake-off tests below. Returns
    (n, gold_words, flat, two_stage, expected_random_rate) so both tests print
    and assert against the same measurement rather than two independent runs
    that could disagree by re-sampling.
    """
    from visual_verify.evidence import covers_text, iou
    from visual_verify.grounding.heatmap import attribution, dense_relevance
    from visual_verify.grounding.snap import rank_candidates, snap_to_box

    embedder, pages, meta = live
    trials = _trials(pages)
    rules = ("dense_mean", "dense_sum", "attribution_mean")
    iou_hit_threshold = 0.25

    def _new_stage_stats():
        return {
            "tally": dict.fromkeys(rules, 0),
            "ious": {r: [] for r in rules},
            "iou_hits": dict.fromkeys(rules, 0),
            "words": {r: [] for r in rules},
        }

    flat = _new_stage_stats()
    two_stage = _new_stage_stats()
    gold_words = []

    for page_vectors, grid, boxes, all_lines, line in trials:
        query_vectors = embedder.embed_query(line.text)
        gold_words.append(_word_count(line.text))
        gold_box = line.bbox

        dense = dense_relevance(query_vectors, page_vectors, grid)
        attr = attribution(query_vectors, page_vectors, grid)

        for name, relevance, reduce in (
            ("dense_mean", dense, "mean"),
            ("dense_sum", dense, "sum"),
            ("attribution_mean", attr, "mean"),
        ):
            ranked = rank_candidates(relevance, grid, all_lines, reduce=reduce)
            flat_box, _ = ranked[0]
            flat["words"][name].append(_word_count(flat_box.text))
            flat_iou = iou(gold_box, flat_box.bbox)
            flat["ious"][name].append(flat_iou)
            flat["iou_hits"][name] += flat_iou >= iou_hit_threshold
            if covers_text(boxes, flat_box.bbox, line.text):
                flat["tally"][name] += 1

            selection = snap_to_box(relevance, grid, boxes, reduce=reduce)
            if selection is not None:
                two_stage["words"][name].append(_word_count(selection.box.text))
                ts_iou = iou(gold_box, selection.box.bbox)
                two_stage["ious"][name].append(ts_iou)
                two_stage["iou_hits"][name] += ts_iou >= iou_hit_threshold
                if covers_text(boxes, selection.box.bbox, line.text):
                    two_stage["tally"][name] += 1

    n = len(trials)
    expected_random_rate = _expected_random_rate(trials)
    return n, meta, rules, iou_hit_threshold, gold_words, flat, two_stage, expected_random_rate


def _print_bakeoff(capsys, n, meta, rules, iou_hit_threshold, gold_words, flat, two_stage):
    with capsys.disabled():
        print(
            f"\nbake-off over {meta['qualifying']}/{meta['scrolled']} qualifying pages, "
            f"{n} (page, line) trials"
        )
        print(f"gold line word count: mean={statistics.mean(gold_words):.1f}")
        for label, stage in (("flat line ranking", flat), ("two-stage snap", two_stage)):
            print(f"{label}:")
            for r in rules:
                mean_iou = statistics.mean(stage["ious"][r])
                hit_rate = stage["iou_hits"][r] / n
                containment_rate = stage["tally"][r] / n
                mean_words = statistics.mean(stage["words"][r])
                print(
                    f"  {r}: mean IoU={mean_iou:.3f} hit@{iou_hit_threshold}={hit_rate:.2%} "
                    f"({stage['iou_hits'][r]}/{n}) containment={containment_rate:.2%} "
                    f"({stage['tally'][r]}/{n}) selected word count mean={mean_words:.1f}"
                )


def test_scoring_bakeoff_is_recorded_not_asserted(live, capsys):
    """The bake-off, tallied per (page, line) trial rather than per page so the
    result measures the rules rather than which pages happened to be easy.

    THIS TEST DOES NOT PICK A WINNER. Three rounds of measurement each found a
    different flaw in how the comparison was posed: a three-page sample that
    separated nothing (1/3 each), queries drawn from each page's longest
    lines (which handed the result to whichever rule prefers large
    candidates), and `covers_text` containment scoring (which pays a rule for
    over-selecting, since it only checks whether the gold text is CONTAINED
    in the selected region and cannot penalize a region that swallows the
    gold line inside something much larger). Corrected for all three and
    scored by IoU against the gold line's own box, which is the metric
    `proposal.tex` line 434 and spec section 8 actually specify, the result
    is: attribution mean leads at 0.593 mean IoU against dense mean's 0.483 on
    the two-stage path, while lighting only about 2% of the grid, which the
    sparsity argument in spec section 6.1 said should make it the weakest.

    That is measured, not asserted here, because all 193 trials come from ONE
    document: the project's own proposal PDF, a homogeneous single-column A4
    report. S7 evaluates on SlideVQA, landscape slides with sparse text and
    heavy figures. This corpus can say the spec's argument for dense over
    attribution was unsupported; it cannot say which rule is right on the
    corpus that actually matters. Picking a winner here would be fitting the
    design to the only corpus available rather than deferring to the
    benchmark evaluation this needs. See spec section 6.1 for the recorded
    numbers and the deferral.

    What IS asserted is the claim snap-to-box actually makes: that the
    heatmap ranking does real work over a random pick, for every rule, not
    just the favored one.
    """
    n, meta, rules, iou_hit_threshold, gold_words, flat, two_stage, expected_random_rate = (
        _compute_bakeoff(live)
    )
    _print_bakeoff(capsys, n, meta, rules, iou_hit_threshold, gold_words, flat, two_stage)

    # Every rule must clear the random floor by a wide margin. That is the
    # claim snap-to-box actually makes: the heatmap ranking does real work.
    # Which rule ranks BEST is deliberately not asserted; see the docstring
    # above and spec section 6.1. All 193 trials come from one document, so
    # this corpus cannot settle that.
    for name in rules:
        hit_at_25 = two_stage["iou_hits"][name] / n
        assert hit_at_25 >= RANDOM_FLOOR_MULTIPLE * expected_random_rate, (
            f"{name} hit@{iou_hit_threshold} {hit_at_25:.1%} is under "
            f"{RANDOM_FLOOR_MULTIPLE}x the random floor {expected_random_rate:.1%}; "
            "the heatmap is contributing little over chance"
        )


def test_two_stage_trades_iou_for_bounded_error(live, capsys):
    """Two-stage selection scores LOWER than flat line ranking, by design.

    Measured on this corpus: flat dense_sum reaches 0.720 mean IoU against
    two-stage's 0.500. A wrong block in stage 1 cannot be recovered in stage
    2, and the ambiguity fallback returns block-sized boxes against a
    line-sized gold. What two-stage buys instead is a bounded error, since a
    stage-2 miss stays inside the right paragraph, and an honest resolution
    flag. This test exists so that trade is a recorded, measured decision
    rather than an accident nobody noticed.
    """
    n, meta, rules, iou_hit_threshold, gold_words, flat, two_stage, _ = _compute_bakeoff(live)
    _print_bakeoff(capsys, n, meta, rules, iou_hit_threshold, gold_words, flat, two_stage)

    flat_mean = {r: statistics.mean(flat["ious"][r]) for r in rules}
    two_stage_mean = {r: statistics.mean(two_stage["ious"][r]) for r in rules}
    with capsys.disabled():
        print("flat vs two-stage mean IoU:")
        for r in rules:
            print(f"  {r}: flat={flat_mean[r]:.3f} two_stage={two_stage_mean[r]:.3f}")

    flat_wins = sum(flat_mean[r] > two_stage_mean[r] for r in rules)
    assert flat_wins >= 2, (
        f"flat line ranking beat two-stage snap on mean IoU for only {flat_wins}/3 rules "
        f"(flat={flat_mean}, two_stage={two_stage_mean}); if this reverses, the "
        "bounded-error trade-off spec section 7 describes has changed and needs "
        "revisiting"
    )
