"""Scoring bake-off against the real indexed corpus.

Design spec section 6.1 requires the scoring rule to be chosen by measurement,
not by argument. Vectors are already stored in Qdrant, so this costs one model
load for query embeddings and is otherwise pure numpy: no page re-embedding.

Two controls ride along with the default so the spec's reasoning is
demonstrated on data rather than asserted:

  dense sum          area-biased. Should lose to dense mean because sum is
                      monotone in area and always prefers the largest candidate.
  attribution mean    sparse. Measured elsewhere in this slice to light only
                      4 to 14 of 736 patches, so most candidates score zero.

If either control wins, the assertions below FAIL on purpose: that would mean
the spec's reasoning about area bias or sparsity does not hold on real data and
the default needs revisiting, not a looser assertion.
"""

import gc
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from visual_verify.config import Settings

pytestmark = pytest.mark.slow

INDEX_DB = Path("data/index.db")
MIN_WORD_BOXES = 40
MAX_PAGES = 4


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
    """(embedder, pages), pages = [(page_vectors, grid, boxes), ...].

    Module-scoped so the model loads once for the whole file. Guards, in
    order: env var present, local index.db present, CUDA present, at least
    one usable Qdrant point with enough word boxes to be a meaningful sample.
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
        index.collection, limit=MAX_PAGES, with_payload=True, with_vectors=True
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

    if not pages:
        pytest.skip(f"no scrolled page had >= {MIN_WORD_BOXES} word boxes")

    embedder = ColQwen2Embedder()
    yield embedder, pages
    # Two live ColQwen2 instances do not fit in 4 GB of VRAM. Without freeing
    # here, this module passes alone and the next model-loading module OOMs
    # during weight loading, exactly the failure CLAUDE.md documents for
    # test_known_item_retrieval.py and test_embedder.py.
    del embedder
    gc.collect()
    torch.cuda.empty_cache()


def _longest_line(boxes):
    from visual_verify.derive import line_boxes

    lines = line_boxes(boxes)
    return max(lines, key=lambda ln: len(ln.text))


def test_the_selector_beats_the_random_candidate_floor(live):
    """Known-item retrieval: the query text is copied verbatim from the page's
    own longest line, so the correct answer is known in advance. This is a
    floor, not a quality measure. The assertion checks covered TEXT, not ink:
    every candidate box on a real page contains ink (measured 435/435 on a
    representative page), so an ink check passes a random selector as often as
    a correct one and proves nothing.
    """
    from visual_verify.derive import line_boxes
    from visual_verify.evidence import covers_text
    from visual_verify.grounding.heatmap import dense_relevance
    from visual_verify.grounding.snap import rank_candidates

    embedder, pages = live
    hits = 0
    for page_vectors, grid, boxes in pages:
        line = _longest_line(boxes)
        query_vectors = embedder.embed_query(line.text)
        relevance = dense_relevance(query_vectors, page_vectors, grid)
        ranked = rank_candidates(relevance, grid, line_boxes(boxes), reduce="mean")
        top_box, _ = ranked[0]
        if covers_text(boxes, top_box.bbox, line.text):
            hits += 1

    ratio = hits / len(pages)
    assert hits >= 1, f"selector beat the floor on {hits}/{len(pages)} pages ({ratio:.2f})"


def test_query_token_counts_and_heatmap_sparsity_are_recorded(live, capsys):
    """Records the sparsity numbers section 6.1 cites, on the live corpus
    rather than a fixture, so the design's numbers are traceable to this run.
    """
    from visual_verify.grounding.heatmap import attribution

    embedder, pages = live
    max_fraction = 0.0
    with capsys.disabled():
        print()
        for i, (page_vectors, grid, boxes) in enumerate(pages):
            line = _longest_line(boxes)
            query_vectors = embedder.embed_query(line.text)
            n_tokens = query_vectors.shape[0]
            attr = attribution(query_vectors, page_vectors, grid)
            n_lit = int((attr > 0).sum())
            fraction = n_lit / grid.n_image_patches
            max_fraction = max(max_fraction, fraction)
            print(
                f"page {i}: query tokens={n_tokens} distinct patches lit={n_lit} "
                f"grid={grid.n_x}x{grid.n_y} ({fraction:.2%} of {grid.n_image_patches})"
            )

    assert max_fraction < 0.10, (
        f"attribution lit {max_fraction:.2%} of the grid on the worst page; if this is "
        "no longer under 10%, attribution is no longer sparse and spec section 6.1's "
        "reasoning for excluding it from ranking needs revisiting"
    )


def test_dense_mean_beats_dense_sum_and_attribution(live, capsys):
    """The bake-off itself. Dense mean is the spec's chosen default; dense sum
    and attribution mean are controls kept in specifically so a control winning
    fails this test, rather than being explained away after the fact.
    """
    from visual_verify.derive import line_boxes
    from visual_verify.evidence import covers_text
    from visual_verify.grounding.heatmap import attribution, dense_relevance
    from visual_verify.grounding.snap import rank_candidates

    embedder, pages = live
    tally = {"dense_mean": 0, "dense_sum": 0, "attribution_mean": 0}

    for page_vectors, grid, boxes in pages:
        line = _longest_line(boxes)
        query_vectors = embedder.embed_query(line.text)
        candidates = line_boxes(boxes)

        dense = dense_relevance(query_vectors, page_vectors, grid)
        attr = attribution(query_vectors, page_vectors, grid)

        for name, relevance, reduce in (
            ("dense_mean", dense, "mean"),
            ("dense_sum", dense, "sum"),
            ("attribution_mean", attr, "mean"),
        ):
            ranked = rank_candidates(relevance, grid, candidates, reduce=reduce)
            top_box, _ = ranked[0]
            if covers_text(boxes, top_box.bbox, line.text):
                tally[name] += 1

    n = len(pages)
    with capsys.disabled():
        print(
            f"\nbake-off over {n} pages: dense_mean={tally['dense_mean']}/{n} "
            f"dense_sum={tally['dense_sum']}/{n} attribution_mean={tally['attribution_mean']}/{n}"
        )

    assert tally["dense_mean"] >= tally["dense_sum"], (
        f"dense sum ({tally['dense_sum']}/{n}) beat dense mean "
        f"({tally['dense_mean']}/{n}); the spec's area-bias reasoning for preferring "
        "mean over sum does not hold on this corpus and needs revisiting"
    )
    assert tally["dense_mean"] >= tally["attribution_mean"], (
        f"attribution mean ({tally['attribution_mean']}/{n}) beat dense mean "
        f"({tally['dense_mean']}/{n}); the spec's sparsity reasoning for excluding "
        "attribution from ranking does not hold on this corpus and needs revisiting"
    )
