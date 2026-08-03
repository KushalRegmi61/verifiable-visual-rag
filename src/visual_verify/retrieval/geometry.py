"""Patch grid geometry: the bridge from a model vector index to a page region.

This is what makes snap-to-box possible, and getting it wrong does not raise.

Two facts about ColQwen2 drive every line here, both measured on PyMuPDF-rendered
A4 pages at 150 dpi (1241x1754 px):

  1. The 747 vectors are 4 prefix tokens, then 736 CONTIGUOUS image patches,
     then 7 suffix tokens. Image patches start at sequence index 4. Slicing
     [:736] shifts the entire grid by four cells and misplaces every box while
     producing perfectly plausible output.

  2. The grid is 23 x 32 and is NOT square. It comes from smart_resize on the
     page aspect ratio, so a landscape slide yields different dimensions. There
     is no global constant to hardcode for the GRID, which is why this is a
     value object stored per page rather than module-level constants.

The 7 trailing suffix tokens, unlike the grid, are a fixed artifact of
ColQwen2's chat template: they follow the image patches regardless of the
page's aspect ratio or patch count. That is what SUFFIX_TOKENS below encodes,
and it is also what makes a miscounted or corrupted n_vectors detectable: for
any given (n_x, n_y, offset), there is exactly one correct n_vectors, not a
range of acceptable ones.

Pure stdlib on purpose: S4 and the evaluation harness both need this, and
neither should have to import torch to get it.
"""

from dataclasses import dataclass

BBox = tuple[float, float, float, float]

# Fixed count of trailing special tokens ColQwen2 appends after the image
# patches (end-of-image / chat-template tokens). This does not vary with page
# aspect ratio the way the grid dims do, so unlike n_x/n_y it is safe to fix
# here rather than pass in per page.
SUFFIX_TOKENS = 7


@dataclass(frozen=True)
class PatchGrid:
    """Where each model vector sits on the page.

    n_x: patch columns. n_y: patch rows. offset: sequence index of image patch 0.
    n_vectors: total vectors the model returned, including special tokens.
    """

    n_x: int
    n_y: int
    offset: int
    n_vectors: int

    def __post_init__(self) -> None:
        if self.n_x <= 0 or self.n_y <= 0:
            raise ValueError(f"grid dims must be positive, got {self.n_x}x{self.n_y}")
        if self.offset < 0:
            raise ValueError(f"offset must be non-negative, got {self.offset}")
        # Given n_x, n_y, and offset, exactly one n_vectors is correct: prefix
        # tokens (offset) + image patches (n_image_patches) + the fixed 7
        # trailing suffix tokens. This is deliberately an equality, not a
        # lower bound: a lower bound only catches a grid recorded from too
        # few vectors, but a caller who passes a wildly larger n_vectors (a
        # miscounted prefix, a grid read from the wrong page, a stale
        # constant) is just as wrong and must not be waved through. Cheap,
        # and it is the only thing standing between an off-by-N grid and
        # silently wrong evidence boxes.
        expected = self.offset + self.n_image_patches + SUFFIX_TOKENS
        if self.n_vectors != expected:
            raise ValueError(
                f"inconsistent: {self.n_x}x{self.n_y} patches at offset "
                f"{self.offset} with {SUFFIX_TOKENS} suffix tokens implies "
                f"{expected} vectors, got {self.n_vectors}"
            )

    @property
    def n_image_patches(self) -> int:
        return self.n_x * self.n_y

    @property
    def n_special(self) -> int:
        """Vectors that correspond to no region of the page."""
        return self.n_vectors - self.n_image_patches

    def is_image_token(self, seq_idx: int) -> bool:
        """Whether a sequence index refers to a page region at all.

        Callers MUST filter with this before any argmax. A special token has no
        region, and mapping one anyway would draw a confident box with no causal
        relationship to the evidence.
        """
        return self.offset <= seq_idx < self.offset + self.n_image_patches

    def seq_to_patch(self, seq_idx: int) -> int:
        """Model sequence index to image patch index."""
        if not self.is_image_token(seq_idx):
            raise ValueError(
                f"sequence index {seq_idx} is a special token, not an image patch; "
                f"image patches occupy {self.offset}.."
                f"{self.offset + self.n_image_patches - 1}"
            )
        return seq_idx - self.offset

    def patch_bbox(self, patch_idx: int) -> BBox:
        """Normalized 0-1 page rect for an image patch, origin top-left.

        Same convention as contracts.BBox and every box S2 stored, so a patch
        rect and a word rect can be compared directly.
        """
        if not 0 <= patch_idx < self.n_image_patches:
            raise IndexError(f"patch {patch_idx} out of range 0..{self.n_image_patches - 1}")
        col = patch_idx % self.n_x
        row = patch_idx // self.n_x
        return (col / self.n_x, row / self.n_y, (col + 1) / self.n_x, (row + 1) / self.n_y)

    def seq_bbox(self, seq_idx: int) -> BBox:
        """Convenience: sequence index straight to page rect."""
        return self.patch_bbox(self.seq_to_patch(seq_idx))
