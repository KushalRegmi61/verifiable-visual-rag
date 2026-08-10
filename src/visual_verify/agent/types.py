"""The narrow seam every agent module talks through.

Deliberately smaller than LangChain's surface. Modules depend on THIS, not on
LangChain, which is what lets the whole pipeline run in tests with no network,
no API key, and no heavy import. Same pattern as retrieval.types.Embedder and
FakeEmbedder, for the same reason.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

S = TypeVar("S", bound=BaseModel)


class StructuredChat(Protocol):
    """A chat model that returns schema-validated output."""

    @property
    def model_id(self) -> str:
        """Provider-qualified, e.g. 'openai:gpt-4o'. The cache keys on this."""
        ...

    def structured(self, prompt: str, image_paths: list[Path], schema: type[S]) -> S:
        """One turn over zero or more page images. Raises if the response does
        not satisfy `schema`.

        A list, not `Path | None` and not an overload: the reader now sees the
        top pages of a document in one call. There is deliberately no
        single-image spelling left, because a compatibility shim would leave a
        quiet path that still type-checks while carrying one page, and the
        symptom of a dropped page is a plausible answer that simply misses the
        evidence rather than anything that raises.
        """
        ...


@dataclass(frozen=True)
class RecordedCall:
    prompt: str
    image_paths: list[Path]


@dataclass
class FakeChat:
    """Scripted stand-in. No network, no key, no LangChain import.

    Responses are returned in order and the script must not be over-consumed:
    repeating the last response would let a test that calls the model more
    times than it expects still pass, which hides a duplicated API call.
    """

    _model_id: str
    responses: list[BaseModel]
    calls: list[RecordedCall] = field(default_factory=list)
    _next: int = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    def structured(self, prompt: str, image_paths: list[Path], schema: type[S]) -> S:
        # Copied, not aliased: a caller that builds the list once and mutates it
        # between calls would otherwise rewrite the record of what it already
        # sent, and the test asserting which pages reached the model would read
        # back the latest call's pages for every call.
        self.calls.append(RecordedCall(prompt=prompt, image_paths=list(image_paths)))
        assert self._next < len(self.responses), (
            f"script exhausted after {self._next} call(s); the code under test "
            "called the model more times than the test scripted"
        )
        out = self.responses[self._next]
        self._next += 1
        assert isinstance(out, schema), (
            f"scripted response {self._next - 1} is {type(out).__name__}, "
            f"but the caller asked for {schema.__name__}"
        )
        return out
