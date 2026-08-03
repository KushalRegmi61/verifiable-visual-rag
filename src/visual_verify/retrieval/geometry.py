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

The 7 trailing suffix tokens, like the 4-token prefix, come from the chat
prompt template, which is model-version dependent in exactly the way `offset`
is: an upstream colpali-engine or transformers bump can shift either count by
a token without changing anything else about the grid. That rules out pinning
an exact expected n_vectors. Instead the sanity check below bounds the total
count of non-image vectors (prefix + suffix together) to a generous range,
which still catches a grid recorded from the wrong image or a miscounted
vector total without rejecting correct output from a shifted template.

Pure stdlib on purpose: S4 and the evaluation harness both need this, and
neither should have to import torch to get it.
"""

from dataclasses import dataclass

BBox = tuple[float, float, float, float]

# Generous upper bound rather than an exact count. The special tokens come
# from the chat prompt template, which is model-version dependent in exactly
# the way `offset` is, so pinning an exact number would reject correct output
# from a model whose template shifted by a token. The bound still catches a
# grid recorded from the wrong image or a miscounted vector total, which is
# what this invariant is actually for.
MAX_SPECIAL_TOKENS = 32


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
        # n_special counts every vector that maps to no page region: prefix
        # plus suffix together, same quantity as the n_special property below
        # (kept in lockstep with it deliberately, not derived separately). A
        # lower bound alone only catches a grid recorded from too few
        # vectors; a caller who passes a wildly larger n_vectors (a grid read
        # from the wrong page, a stale constant, a miscounted total) is just
        # as wrong and must not be waved through. The upper bound is
        # generous rather than exact so a template that shifts the true
        # prefix/suffix split by a token or two still validates.
        n_special = self.n_vectors - self.n_image_patches
        if not 0 <= n_special <= MAX_SPECIAL_TOKENS:
            raise ValueError(
                f"inconsistent: {self.n_x}x{self.n_y} patches at offset {self.offset} "
                f"implies {n_special} special tokens, outside the sane range "
                f"0..{MAX_SPECIAL_TOKENS} (n_vectors={self.n_vectors})"
            )

    @property
    def n_image_patches(self) -> int:
        return self.n_x * self.n_y

    @property
    def n_special(self) -> int:
        """Vectors that correspond to no region of the page: prefix + suffix.

        This is n_vectors - n_image_patches, not n_vectors - offset -
        n_image_patches. offset is only the prefix count; subtracting it too
        would silently drop the suffix and undercount. The validation in
        __post_init__ checks this exact quantity, so keep the two in sync.
        """
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
