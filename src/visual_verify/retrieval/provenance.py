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

from dataclasses import asdict, dataclass


class ProvenanceMismatch(RuntimeError):
    """Raised when an embedder does not match the vectors already indexed."""


@dataclass(frozen=True)
class EmbedProvenance:
    model_id: str
    model_revision: str
    quantization: str
    dtype: str
    render_dpi: int
    embed_version: int

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
        return cls(**{f: payload[f] for f in cls.__dataclass_fields__})

    def require_compatible(self, other: "EmbedProvenance") -> None:
        """Raise unless `other` produced vectors comparable with ours.

        Every field is checked. There is no "close enough": a different revision
        of the same model can ship different weights, and a different render DPI
        changes the patch grid, so neither is a cosmetic difference.
        """
        for field in self.__dataclass_fields__:
            mine, theirs = getattr(self, field), getattr(other, field)
            if mine != theirs:
                raise ProvenanceMismatch(
                    f"{field} differs: index holds {mine!r}, embedder is {theirs!r}. "
                    "These vectors are not comparable. Re-embed the corpus or use "
                    "the original embedder."
                )
