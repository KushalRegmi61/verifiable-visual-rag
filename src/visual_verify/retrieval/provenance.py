"""What produced a stored vector, and refusal when that changes.

Vectors from different models, quantizations, dtypes, or render DPIs are not
comparable, and nothing about a stored vector reveals which produced it. Mixing
them returns a confidently ranked, entirely wrong result list.

This is the most commonly reported production failure for retrieval systems: the
indexing embedder is changed, the query embedder is not, and recall degrades
silently for weeks. This project is unusually exposed to it, because the design
deliberately keeps colSmol as an alternative and permits per-document render DPI.

The response is to refuse rather than to warn. A warning on a CLI scrolls past;
a wrong answer with a drawn evidence box is exactly the failure this project
exists to prevent.

Pure stdlib on purpose: the query path must be able to check compatibility
before deciding whether loading a multi-gigabyte model is even worthwhile.
"""

from dataclasses import asdict, dataclass, fields


class ProvenanceMismatch(RuntimeError):
    """Raised when an embedder does not match the vectors already indexed."""


@dataclass(frozen=True)
class EmbedProvenance:
    """What produced a stored vector.

    Every field is load-bearing: vectors from different models,
    quantizations, dtypes, or render DPIs are not comparable.
    """

    model_id: str
    model_revision: str
    quantization: str
    dtype: str
    render_dpi: int
    embed_version: int

    def __post_init__(self) -> None:
        # An empty or blank string field would compare equal to another
        # empty field, so two indexes that share nothing but their
        # emptiness would pass require_compatible. That is the guard
        # silently failing to guard, so blanks are rejected here rather
        # than left to be caught downstream.
        for field_name in ("model_id", "model_revision", "quantization", "dtype"):
            value = getattr(self, field_name)
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty, got {value!r}")
        if self.render_dpi <= 0:
            raise ValueError(f"render_dpi must be positive, got {self.render_dpi}")
        if self.embed_version < 0:
            raise ValueError(f"embed_version must be non-negative, got {self.embed_version}")

    def to_payload(self) -> dict[str, str | int]:
        """Flat scalars only; Qdrant payload values must be JSON primitives."""
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict) -> "EmbedProvenance":
        """Build from a Qdrant point payload.

        The real payload also carries doc_sha, page_no, image_path, and
        friends alongside these fields, so this pulls out only the known
        provenance keys rather than splatting the whole dict as kwargs; the
        latter would raise TypeError on the very first unrelated key Qdrant
        happens to store.
        """
        return cls(**{f.name: payload[f.name] for f in fields(cls)})

    def require_compatible(self, other: "EmbedProvenance") -> None:
        """Raise unless `other` produced vectors comparable with ours.

        Every field is checked. There is no "close enough": a different revision
        of the same model can ship different weights, and a different render DPI
        changes the patch grid, so neither is a cosmetic difference.
        """
        for f in fields(self):
            mine, theirs = getattr(self, f.name), getattr(other, f.name)
            if mine != theirs:
                raise ProvenanceMismatch(
                    f"{f.name} differs: index holds {mine!r}, embedder is {theirs!r}. "
                    "These vectors are not comparable. Re-embed the corpus or use "
                    "the original embedder."
                )
