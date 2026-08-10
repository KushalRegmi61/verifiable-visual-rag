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

The key is built from the rendered prompt text, the bytes of every image in the
order they were sent, and the schema name: never from LangChain objects.
LangChain's representations are free to change across versions, and keying on
them would silently invalidate every entry while still appearing to hit.
"""

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from visual_verify.agent.types import StructuredChat

S = TypeVar("S", bound=BaseModel)


def _digest(model_id: str, prompt: str, image_paths: list[Path], schema_name: str) -> str:
    """Unambiguous key over the parts that must distinguish one call from another.

    Length-prefixed rather than separator-joined. A bare separator is not
    injective: with parts joined by a null byte, ("X\\0Y", "Z") and ("X", "Y\\0Z")
    hash identically, so two different calls would share a cache file and one
    would silently return the other's answer. Unreachable today, because
    model_id is a fixed provider string, but this module is the record that
    makes a reported number reproducible, so ambiguity here is not worth
    carrying.

    That ambiguity stopped being hypothetical once a call could carry several
    pages. EVERY image's digest goes in, length-prefixed and in order: without
    it a call over pages [7, 8, 9] and a call over pages [7, 8] key the same,
    and the second returns the first's answer from disk with nothing to show
    for it. That is the worst failure this module can produce, because the
    cache exists to make a reported eval number reproducible, so a collision
    corrupts the record rather than one run.

    The BYTES are hashed, never the path string. Page images live under a
    sha256-named directory that changes whenever the document is recompiled,
    so a path is not a stable identity for content: keying on it would miss on
    every re-ingest of an unchanged page, and would hit across two different
    documents that happened to render to the same relative name.
    """
    h = hashlib.sha256()
    for part in (model_id, schema_name, prompt):
        raw = part.encode("utf-8")
        h.update(len(raw).to_bytes(8, "big"))
        h.update(raw)
    h.update(len(image_paths).to_bytes(8, "big"))
    for image_path in image_paths:
        raw = hashlib.sha256(Path(image_path).read_bytes()).hexdigest().encode()
        h.update(len(raw).to_bytes(8, "big"))
        h.update(raw)
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

    def structured(self, prompt: str, image_paths: list[Path], schema: type[S]) -> S:
        key = _digest(self.inner.model_id, prompt, image_paths, schema.__name__)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                return schema.model_validate(json.loads(path.read_text()))
            except Exception as exc:
                raise ValueError(f"corrupt cache entry at {path}") from exc
        out = self.inner.structured(prompt, image_paths, schema)
        # Written after a successful call, so a failed request leaves no entry
        # that a later run would treat as a real answer.
        #
        # Written to a temp file in the same directory and moved into place
        # with os.replace, which is atomic on POSIX. The eval harness may run
        # several workers over overlapping questions against one cache
        # directory; a plain write_text truncates before writing, so a worker
        # reading a key another worker is mid-write on would see a half
        # written file.
        tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(out.model_dump_json())
        os.replace(tmp_path, path)
        return out
