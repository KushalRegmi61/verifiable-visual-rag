"""Content-addressed cache over any StructuredChat.

Three jobs, only one of which is speed:

1. The defense demo runs offline. Pre-run the questions and the room needs no
   network.
2. It is the reproducibility record. Hosted models drift, so a cached raw
   response is the only evidence that a number reported in March was real.
3. Re-running the eval after a code change is free when the prompts did not
   change.

The key includes the MODEL ID on purpose. Without it, switching provider
silently returns the other model's answer, which would make an A/B comparison
compare a model against itself and report a difference of zero.

The key is built from the rendered prompt text, the image bytes, and the schema
name: never from LangChain objects. LangChain's representations are free to
change across versions, and keying on them would silently invalidate every
entry while still appearing to hit.
"""

import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from visual_verify.agent.types import StructuredChat

S = TypeVar("S", bound=BaseModel)


def _digest(model_id: str, prompt: str, image_path: Path | None, schema_name: str) -> str:
    h = hashlib.sha256()
    for part in (model_id, schema_name, prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    if image_path is not None:
        h.update(hashlib.sha256(Path(image_path).read_bytes()).hexdigest().encode())
    return h.hexdigest()


class CachedChat:
    """Wraps a StructuredChat. Same protocol, so it is a drop-in."""

    def __init__(self, inner: StructuredChat, cache_dir: Path) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    def structured(self, prompt: str, image_path: Path | None, schema: type[S]) -> S:
        key = _digest(self.inner.model_id, prompt, image_path, schema.__name__)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return schema.model_validate(json.loads(path.read_text()))
        out = self.inner.structured(prompt, image_path, schema)
        # Written after a successful call, so a failed request leaves no entry
        # that a later run would treat as a real answer.
        path.write_text(out.model_dump_json())
        return out
