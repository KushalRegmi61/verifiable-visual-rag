"""The model seam.

Two roles, each with a protocol, a hosted HTTP implementation using only
stdlib, and a local HuggingFace implementation that imports torch and
transformers lazily. The CLI wires the default pairing; the independence
rule (spec 3.2) is a wiring assertion, tested in test_backends.

The pairing below is the measured default (spec 3.1): the local verifier is
Qwen2-VL-2B, which loads on this card at 4.2 GB fp16 or 1.5 GB 4-bit and judges
in ~16-20 s. The reader defaults to a hosted Gemini model and the verifier to a
local Qwen model precisely because those families differ, which is what makes
the pair independent.
"""

import json
from typing import Protocol

from PIL import Image

from visual_verify.verify.claims import ReaderOutput, parse_reader_output
from visual_verify.verify.evidence import Evidence
from visual_verify.verify.rubric import Judgement

DEFAULT_READER_MODEL = "gemini-2.5-flash"
DEFAULT_VERIFIER_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


class Reader(Protocol):
    def read(
        self, question: str, image: Image.Image | None, text_layer: str | None
    ) -> ReaderOutput: ...


class Verifier(Protocol):
    def judge(self, claim: str, evidence: Evidence) -> Judgement: ...


def _post(url: str, body: str, key: str | None = None) -> str:
    """One stdlib POST. A module function so tests can stub it."""
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        url, data=body.encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


_READER_PROMPT = (
    "Answer the question from the page provided. Return ONLY JSON:\n"
    '{"answer": "the full answer as a sentence", "claims": ["one atomic claim per assertion"]}\n'
    "Each claim must be a single assertion that can be located on the page by itself."
)


def _reader_prompt(question: str, text_layer: str | None) -> str:
    layer = text_layer if text_layer else "(no text layer; use the image)"
    return f"{_READER_PROMPT}\n\nQuestion: {question}\n\nPage text:\n{layer}"


_VERIFIER_PROMPT = (
    "Judge whether the EVIDENCE supports the CLAIM. Return ONLY JSON:\n"
    '{"label": "supported" | "partial" | "unsupported" | "insufficient"}\n'
    "supported: the evidence establishes the claim.\n"
    "partial: the evidence establishes part of the claim.\n"
    "unsupported: the evidence contradicts the claim or fails to establish it.\n"
    "insufficient: the evidence does not address the claim at all."
)


def _verifier_prompt(claim: str, evidence_text: str | None) -> str:
    ev = evidence_text if evidence_text else "(no text; only an image crop)"
    return f"{_VERIFIER_PROMPT}\n\nCLAIM: {claim}\n\nEVIDENCE: {ev}"


class HostedAPIReader:
    """Reader through any OpenAI-style chat-completions endpoint."""

    def __init__(self, url: str, key: str | None = None, model: str = DEFAULT_READER_MODEL):
        self.url = url
        self.key = key
        self.model = model

    def read(
        self, question: str, image: Image.Image | None, text_layer: str | None
    ) -> ReaderOutput:
        messages = [{"role": "user", "content": _reader_prompt(question, text_layer)}]
        body = json.dumps({"model": self.model, "messages": messages})
        response = json.loads(_post(self.url, body, self.key))
        raw = response["choices"][0]["message"]["content"]
        return parse_reader_output(raw)


class HostedAPIVerifier:
    """Verifier through any OpenAI-style chat-completions endpoint."""

    def __init__(self, url: str, key: str | None = None, model: str = DEFAULT_VERIFIER_MODEL):
        self.url = url
        self.key = key
        self.model = model

    def judge(self, claim: str, evidence: Evidence) -> Judgement:
        messages = [{"role": "user", "content": _verifier_prompt(claim, evidence.text)}]
        body = json.dumps({"model": self.model, "messages": messages})
        response = json.loads(_post(self.url, body, self.key))
        raw = response["choices"][0]["message"]["content"]
        return Judgement(**json.loads(raw))


class LocalVLMReader:
    """HuggingFace generative VLM as reader.

    Lazy imports: the core must be importable without torch, and so must
    this module. The heavy load and the fit measurement live in _local,
    which is the ONLY module allowed to import transformers/torch.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "cuda"):
        self.model_id = model_id
        self.device = device

    def read(
        self, question: str, image: Image.Image | None, text_layer: str | None
    ) -> ReaderOutput:
        from visual_verify.verify._local import generate_json

        raw = generate_json(self.model_id, self.device, _reader_prompt(question, text_layer), image)
        return parse_reader_output(raw)


class LocalVLMVerifier:
    """HuggingFace generative VLM as verifier. Same lazy-load discipline."""

    def __init__(self, model_id: str = DEFAULT_VERIFIER_MODEL, device: str = "cuda"):
        self.model_id = model_id
        self.device = device

    def judge(self, claim: str, evidence: Evidence) -> Judgement:
        from visual_verify.verify._local import generate_json

        raw = generate_json(
            self.model_id, self.device, _verifier_prompt(claim, evidence.text), evidence.image
        )
        return Judgement(**json.loads(raw))
