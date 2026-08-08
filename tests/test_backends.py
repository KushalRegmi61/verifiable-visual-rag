"""The model seam: protocols, hosted request shape, and the pairing rule.

The independence rule (spec 3.2) is enforced here, in wiring, because it
cannot be enforced in the core: reader and verifier arrive as objects and
the core has no way to know their model families.
"""

import json

from visual_verify.verify.backends import (
    DEFAULT_READER_MODEL,
    DEFAULT_VERIFIER_MODEL,
    HostedAPIReader,
    HostedAPIVerifier,
    _post,
)
from visual_verify.verify.claims import ReaderOutput
from visual_verify.verify.evidence import Evidence
from visual_verify.verify.rubric import Judgement


def test_default_pairing_satisfies_the_independence_rule():
    assert DEFAULT_READER_MODEL != DEFAULT_VERIFIER_MODEL
    assert "Qwen" not in DEFAULT_READER_MODEL
    assert "Qwen" in DEFAULT_VERIFIER_MODEL


def test_hosted_reader_builds_a_json_request(monkeypatch):
    payload = {}

    def fake_post(url, body, key=None):
        payload["url"] = url
        payload["key"] = key
        payload["body"] = json.loads(body)
        content = '{"answer": "42.", "claims": ["42."]}'
        return json.dumps({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("visual_verify.verify.backends._post", fake_post)
    reader = HostedAPIReader(url="https://example.test/read", key="k")
    out = reader.read("Q?", None, None)
    assert isinstance(out, ReaderOutput)
    assert out.claims == ["42."]
    assert payload["url"] == "https://example.test/read"
    assert payload["key"] == "k"
    assert payload["body"]["model"] == DEFAULT_READER_MODEL
    assert "Question: Q?" in payload["body"]["messages"][0]["content"]


def test_hosted_verifier_parses_the_label(monkeypatch):
    def fake_post(url, body, key=None):
        return json.dumps({"choices": [{"message": {"content": '{"label": "supported"}'}}]})

    monkeypatch.setattr("visual_verify.verify.backends._post", fake_post)
    verifier = HostedAPIVerifier(url="https://example.test/judge", key="k")
    j = verifier.judge("c", Evidence(text="evidence"))
    assert j == Judgement(label="supported")


def test_hosted_verifier_rejects_a_garbage_label(monkeypatch):
    def fake_post(url, body, key=None):
        return json.dumps({"choices": [{"message": {"content": '{"label": "maybe"}'}}]})

    monkeypatch.setattr("visual_verify.verify.backends._post", fake_post)
    verifier = HostedAPIVerifier(url="https://example.test/judge", key="k")
    try:
        verifier.judge("c", Evidence(text="evidence"))
    except Exception as exc:
        assert "label" in str(exc)
    else:
        raise AssertionError("garbage label must not produce a Judgement")


def test_post_sends_an_authorization_header(monkeypatch):
    request = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(req, timeout):
        request["headers"] = req.headers
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    body = _post("https://example.test/x", "{}", key="secret")
    assert request["headers"]["Authorization"] == "Bearer secret"
    assert body == '{"ok": true}'
