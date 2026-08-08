"""Evidence assembly: the region S4 returned, shaped for the verifier."""

from dataclasses import dataclass

from PIL import Image

from visual_verify.contracts import GroundedRegion


@dataclass(frozen=True)
class Evidence:
    text: str | None = None
    image: Image.Image | None = None


def crop_region(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    """Cut the normalized 0-1 bbox out of the page render."""
    w, h = image.size
    x0, y0, x1, y1 = bbox
    return image.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


def build_evidence(region: GroundedRegion, image: Image.Image | None) -> Evidence:
    """Text regions need no pixels; visual regions get the crop."""
    if region.modality == "text":
        return Evidence(text=region.text)
    return Evidence(
        text=region.text,
        image=crop_region(image, region.bbox) if image is not None else None,
    )


def best_region(regions: list[GroundedRegion]) -> GroundedRegion | None:
    """The region the verifier judges against: highest score, ties to first.

    Judging against every region would let a weak claim pass on a stray
    match. All regions still travel on the returned Claim; only the
    judgement is per-claim.
    """
    if not regions:
        return None
    return max(regions, key=lambda r: r.score)
