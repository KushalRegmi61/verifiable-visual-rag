"""Derived box granularities.

Only word boxes are stored. Lines, blocks, and spans are unions computed here,
so the candidate set S4 ranks can be retuned without re-ingesting anything.

Pure functions over BoxRecord lists: no database, no PDF, no I/O. This is the
shape every core function takes, and it is why the store can stay behind an extra.
"""

from itertools import groupby

from visual_verify.ingest.boxes import BoxKind, BoxRecord, word_boxes


def _union(boxes: list[BoxRecord], text: str, kind: BoxKind) -> BoxRecord:
    return BoxRecord(
        kind=kind,
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
        text=text,
        block_no=boxes[0].block_no,
        line_no=boxes[0].line_no,
        word_no=-1,
    )


def _sorted_words(boxes: list[BoxRecord]) -> list[BoxRecord]:
    """Word boxes in reading order.

    The sort is load-bearing: groupby only groups CONSECUTIVE equal keys, so
    unsorted input would fragment one line into several boxes.
    """
    words = word_boxes(boxes)
    if boxes and not words:
        raise ValueError(
            f"no word boxes in input of {len(boxes)} boxes "
            f"(kinds: {sorted({b.kind for b in boxes})}); "
            "derived boxes cannot be re-derived"
        )
    return sorted(words, key=lambda b: (b.block_no, b.line_no, b.word_no))


def _grouped_by_line(words: list[BoxRecord]) -> list[list[BoxRecord]]:
    """Segment an already-ordered run of words at (block_no, line_no) changes.

    Each group is materialized with list() exactly once: groupby hands back a
    shared iterator that is invalidated as soon as the next group is requested,
    so materializing is the only safe way to read a group more than once.
    """
    return [list(group) for _, group in groupby(words, key=lambda b: (b.block_no, b.line_no))]


def line_boxes(boxes: list[BoxRecord]) -> list[BoxRecord]:
    """One box per (block, line), text joined in reading order."""
    return [
        _union(members, " ".join(m.text for m in members), kind="line")
        for members in _grouped_by_line(_sorted_words(boxes))
    ]


def block_boxes(boxes: list[BoxRecord]) -> list[BoxRecord]:
    """One box per block, text joined in reading order."""
    out: list[BoxRecord] = []
    for _, group in groupby(_sorted_words(boxes), key=lambda b: b.block_no):
        members = list(group)
        out.append(_union(members, " ".join(m.text for m in members), kind="block"))
    return out


def span_boxes(boxes: list[BoxRecord], needle: str) -> list[BoxRecord]:
    """Rects covering `needle`, split at line boundaries. Empty list if absent.

    Returns one rect per (block, line) the match spans, NOT a single union.
    A single union over a match that wraps across a line break sweeps in every
    intervening word: on the two-line fixture, matching "percent Margins"
    unions to 5.7x the true ink area and encloses all seven words on the page.
    This function generates the evaluation harness's gold boxes, so an
    over-covering rect is fabricated ground truth, not merely an imprecise one.

    Matching is case-insensitive over whitespace-split tokens in reading order,
    so a phrase that wraps across a line break is still found. Only the returned
    geometry is split.

    First match wins. If the same phrase occurs more than once on a page, the
    first occurrence in reading order is returned and the ambiguity is not
    signalled. The eval harness may want to exclude such examples; surfacing
    all occurrences is deferred until there is a consumer that needs it.

    Known limitation, deferred to S4: ligatures and hyphenated line breaks in
    the text layer defeat exact matching on real pages. Text is stored raw on
    purpose, because normalizing at ingest would break its correspondence to
    the stored boxes.
    """
    wanted = needle.lower().split()
    if not wanted:
        return []

    words = _sorted_words(boxes)
    lowered = [w.text.lower() for w in words]

    for i in range(len(words) - len(wanted) + 1):
        if lowered[i : i + len(wanted)] == wanted:
            matched = words[i : i + len(wanted)]
            return [
                _union(members, " ".join(m.text for m in members), kind="span")
                for members in _grouped_by_line(matched)
            ]
    return []
