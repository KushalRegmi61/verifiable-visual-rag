"""Derived box granularities.

Only word boxes are stored. Lines, blocks, and spans are unions computed here,
so the candidate set S4 ranks can be retuned without re-ingesting anything.

Pure functions over BoxRecord lists: no database, no PDF, no I/O. This is the
shape every core function takes, and it is why the store can stay behind an extra.
"""

from itertools import groupby

from visual_verify.ingest.boxes import BoxRecord, word_boxes


def _union(boxes: list[BoxRecord], text: str, kind: str = "word") -> BoxRecord:
    return BoxRecord(
        kind=kind,  # type: ignore[arg-type]
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
    return sorted(word_boxes(boxes), key=lambda b: (b.block_no, b.line_no, b.word_no))


def line_boxes(boxes: list[BoxRecord]) -> list[BoxRecord]:
    """One box per (block, line), text joined in reading order."""
    out: list[BoxRecord] = []
    for _, group in groupby(_sorted_words(boxes), key=lambda b: (b.block_no, b.line_no)):
        members = list(group)
        out.append(_union(members, " ".join(m.text for m in members)))
    return out


def block_boxes(boxes: list[BoxRecord]) -> list[BoxRecord]:
    """One box per block, text joined in reading order."""
    out: list[BoxRecord] = []
    for _, group in groupby(_sorted_words(boxes), key=lambda b: b.block_no):
        members = list(group)
        out.append(_union(members, " ".join(m.text for m in members)))
    return out


def span_box(boxes: list[BoxRecord], needle: str) -> BoxRecord | None:
    """Union of the words covering `needle`, or None if it is not present.

    Matching is case-insensitive on whitespace-split tokens. This is what S7's
    auto-derived gold box is built from: locate the answer string, union its words.

    Known limitation, deferred to S4: ligatures and hyphenated line breaks in the
    text layer will defeat exact matching on real pages. Text is stored raw here
    on purpose, because normalizing at ingest would break its correspondence to
    the stored boxes.
    """
    wanted = needle.lower().split()
    if not wanted:
        return None

    words = _sorted_words(boxes)
    lowered = [w.text.lower() for w in words]

    for i in range(len(words) - len(wanted) + 1):
        if lowered[i : i + len(wanted)] == wanted:
            members = words[i : i + len(wanted)]
            return _union(members, " ".join(m.text for m in members))
    return None
