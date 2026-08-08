"""Checking that a box actually points at its evidence.

Every coordinate bug in this repository was found by converting a normalized box
back to pixels and looking at what it crops. None were found by reasoning about
the code, because a wrong box is a perfectly well-formed four-tuple: normalized,
positive-area, correctly typed. Nothing about it looks wrong.

THE TWO QUESTIONS ARE NOT THE SAME. Keep them apart:

  1. "Is the coordinate transform correct?"  ->  has_ink, plus a displaced control.
     A transform bug (unapplied rotation matrix, normalizing against the pixmap
     instead of page.rect, an off-by-one in the patch offset) moves a box onto
     whitespace, so ink presence catches it.

  2. "Is this the RIGHT box?"  ->  covers_text.
     Ink presence CANNOT answer this and must never be used to. Measured on
     proposal.pdf page 3 at 150 dpi: all 435 word boxes contain ink, so a
     selector returning a uniformly random word box scores 200/200 on has_ink
     while being correct 1/200. Grounding chooses among candidates that all
     already have ink; only identity of the covered text discriminates.

Pixel conversion lives here and nowhere else. Three test modules previously
inlined the same int(x0 * width_px) arithmetic and only one of them carried the
displaced control that makes the assertion discriminating.
"""

from PIL import Image

from visual_verify.contracts import BBox

# A pixel darker than this counts as ink. 128 is the midpoint of an 8-bit
# greyscale channel: antialiased glyph edges land well above it, glyph cores
# well below, so the choice is not sensitive on rendered text.
INK_THRESHOLD = 128

# Default share of a candidate box that must fall inside the region before its
# text is treated as covered. Half, so a word clipped by the region boundary
# does not silently contribute its whole string to the evidence.
MIN_OVERLAP = 0.5


def to_pixels(bbox: BBox, width_px: int, height_px: int) -> tuple[int, int, int, int]:
    """Normalized box to an integer pixel rect for cropping.

    The one place this conversion is written. Truncation rather than rounding
    keeps the crop inside the box, so ink found here is ink the box genuinely
    covers rather than a neighbouring glyph pulled in by rounding outward.
    """
    x0, y0, x1, y1 = bbox
    return (
        int(x0 * width_px),
        int(y0 * height_px),
        int(x1 * width_px),
        int(y1 * height_px),
    )


def ink_ratio(image: Image.Image, bbox: BBox, threshold: int = INK_THRESHOLD) -> float:
    """Fraction of the box's pixels that are ink, 0.0 for an empty crop.

    A ratio rather than a bool because grounding needs to compare regions: a box
    around one word and a box around a whole paragraph both "have ink", and only
    the density separates a tight region from a lazily large one.
    """
    crop = image.convert("L").crop(to_pixels(bbox, image.width, image.height))
    n = crop.width * crop.height
    if n == 0:
        return 0.0
    hist = crop.histogram()
    return sum(hist[:threshold]) / n


def has_ink(image: Image.Image, bbox: BBox, threshold: int = INK_THRESHOLD) -> bool:
    """Whether the box crops any dark pixel at all.

    Answers question 1 only. See the module docstring before reaching for this
    to validate a grounding result.
    """
    crop = image.convert("L").crop(to_pixels(bbox, image.width, image.height))
    if crop.width == 0 or crop.height == 0:
        return False
    # getextrema()[0] is the darkest pixel present, and unlike min(getdata())
    # it does not trip Pillow's getdata deprecation warning.
    return crop.getextrema()[0] < threshold


def shift(bbox: BBox, dx: float = 0.0, dy: float = 0.0) -> BBox:
    """Displace a box, clamped to the page, for use as a negative control.

    "Every box contains ink" is trivially true on a dense page, so an alignment
    assertion means nothing without showing that a WRONG box mostly does not.
    Measured on proposal.pdf page 3: true boxes 60/60, the same boxes shifted
    25% in x 16/60. Without this control the assertion is decoration.
    """
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    x0 = min(max(x0 + dx, 0.0), 1.0 - w)
    y0 = min(max(y0 + dy, 0.0), 1.0 - h)
    return (x0, y0, x0 + w, y0 + h)


def overlap_fraction(inner: BBox, outer: BBox) -> float:
    """How much of `inner`'s area falls inside `outer`."""
    ax0, ay0, ax1, ay1 = inner
    bx0, by0, bx1, by1 = outer
    w = min(ax1, bx1) - max(ax0, bx0)
    h = min(ay1, by1) - max(ay0, by0)
    if w <= 0 or h <= 0:
        return 0.0
    area = (ax1 - ax0) * (ay1 - ay0)
    return (w * h) / area if area > 0 else 0.0


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union. The localisation metric S7 reports."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    w = min(ax1, bx1) - max(ax0, bx0)
    h = min(ay1, by1) - max(ay0, by0)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _norm(s: str) -> str:
    return " ".join(s.split()).casefold()


def text_in_bbox(boxes, bbox: BBox, min_overlap: float = MIN_OVERLAP) -> str:
    """Text of the candidate boxes that `bbox` covers.

    `boxes` is any iterable of objects carrying x0/y0/x1/y1/text, which both the
    ingest BoxRecord and the stored ORM Box satisfy. Deliberately duck-typed so
    this works against a live extraction and against SQLite without either
    module importing the other.

    Reading text back out of the SAME stored boxes that grounding selects from
    is what makes this a check on selection rather than on extraction. It cannot
    catch a bad box in the text layer; test_smoke_real_pdf covers that.
    """
    covered = [b for b in boxes if overlap_fraction((b.x0, b.y0, b.x1, b.y1), bbox) >= min_overlap]
    covered.sort(key=lambda b: (round(b.y0, 3), b.x0))
    return " ".join(b.text for b in covered if b.text)


def covers_text(boxes, bbox: BBox, expected: str, min_overlap: float = MIN_OVERLAP) -> bool:
    """Whether the region's text contains `expected`, ignoring case and spacing.

    This is the grounding assertion. Unlike has_ink it can fail, because it is
    answering which box was chosen rather than whether some box was drawn.
    """
    return _norm(expected) in _norm(text_in_bbox(boxes, bbox, min_overlap))
