"""The HTTP surface, end to end, with fakes.

No GPU, no API key, no network. This must NOT become a fourth module that
loads ColQwen2: three already fragment the 3.63 GB card badly enough to need
expandable_segments, and a fourth would need process separation.
"""

import contextlib
import json

import pytest
from fastapi.testclient import TestClient

from visual_verify.agent.schemas import ClaimList, Verdict
from visual_verify.agent.types import FakeChat
from visual_verify.api.app import build_app
from visual_verify.api.resources import Resources
from visual_verify.cli import _make_index, main
from visual_verify.config import Settings
from visual_verify.store.engine import make_engine


@pytest.fixture
def client(tmp_path, monkeypatch, born_digital_pdf):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VVRAG_QDRANT_URL", ":memory:")
    monkeypatch.delenv("VVRAG_QDRANT_API_KEY", raising=False)
    monkeypatch.setenv("VVRAG_FAKE_EMBEDDER", "1")
    assert main(["ingest", str(born_digital_pdf)]) == 0
    assert main(["embed", "--all"]) == 0

    from visual_verify.retrieval.types import FakeEmbedder

    settings = Settings.from_env()
    resources = Resources(
        settings=settings,
        engine=make_engine(settings.db_url),
        index=_make_index(settings),
        embedder=FakeEmbedder(),
        reader_chat=FakeChat("r", [ClaimList(claims=["Revenue grew 42 percent"])]),
        verifier_chat=FakeChat("v", [Verdict(label="supported", confidence=0.9, reason="matches")]),
    )
    app = build_app(resources)
    with TestClient(app) as c:
        yield c


def parse_sse(text):
    """[(event name, payload dict)] from a raw SSE body."""
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        name = lines[0].removeprefix("event: ")
        payload = json.loads(lines[1].removeprefix("data: "))
        out.append((name, payload))
    return out


def test_health_reports_the_two_model_ids(client):
    body = client.get("/health").json()

    assert body["reader_model"] == "r"
    assert body["verifier_model"] == "v"
    assert body["indexed_pages"] == 1


def test_documents_lists_what_was_ingested(client):
    body = client.get("/documents").json()

    assert len(body) == 1
    assert body[0]["name"] == "born_digital.pdf"
    assert body[0]["n_pages"] == 1


def test_the_page_image_is_served_as_png(client):
    sha = client.get("/documents").json()[0]["sha"]

    res = client.get(f"/documents/{sha}/pages/0/image")

    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert res.content.startswith(b"\x89PNG")


def test_an_unknown_page_image_is_404(client):
    sha = client.get("/documents").json()[0]["sha"]

    assert client.get(f"/documents/{sha}/pages/99/image").status_code == 404


def test_ask_streams_retrieved_then_claims_then_done(client):
    res = client.post("/ask", json={"question": "What happened?"})

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    names = [name for name, _ in parse_sse(res.text)]
    assert names == ["retrieved", "reading", "claims", "claim", "done"]


def test_every_streamed_claim_carries_a_verdict(client):
    """A claim reaching the wire unverified is the failure the streaming
    decision exists to avoid."""
    res = client.post("/ask", json={"question": "What happened?"})

    for name, payload in parse_sse(res.text):
        if name == "claim":
            assert payload["label"] is not None


def test_a_non_finite_threshold_is_rejected(client):
    # Sent as a raw body rather than json={...}: httpx serializes with
    # allow_nan=False and raises ValueError client-side, so `json={"threshold":
    # float("nan")}` never reaches the server and the route is never exercised.
    # Bare NaN is what a hand-rolled client (or Python's own json.dumps, which
    # does allow it) puts on the wire, and stdlib json.loads accepts it, so the
    # AskRequest validator is the only thing standing between a non-finite
    # threshold and a comparison that is False against every confidence.
    res = client.post(
        "/ask",
        content='{"question": "q", "threshold": NaN}',
        headers={"content-type": "application/json"},
    )

    assert res.status_code == 422


def test_a_page_without_a_doc_is_rejected(client):
    res = client.post("/ask", json={"question": "q", "page": 0})

    assert res.status_code == 422


def test_an_unknown_document_is_404(client):
    """PageNotFound reaches the handler at call time, before the response is
    committed, so it can still be a status code."""
    res = client.post("/ask", json={"question": "q", "doc": "no-such-doc", "page": 0})

    assert res.status_code == 404


def test_asking_with_nothing_indexed_is_409(tmp_path, monkeypatch, born_digital_pdf):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'j.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data2"))
    monkeypatch.setenv("VVRAG_QDRANT_URL", ":memory:")
    monkeypatch.setenv("VVRAG_FAKE_EMBEDDER", "1")
    assert main(["ingest", str(born_digital_pdf)]) == 0

    from visual_verify.retrieval.types import FakeEmbedder

    settings = Settings.from_env()
    resources = Resources(
        settings=settings,
        engine=make_engine(settings.db_url),
        index=_make_index(settings),
        embedder=FakeEmbedder(),
        reader_chat=FakeChat("r", []),
        verifier_chat=FakeChat("v", []),
    )
    with TestClient(build_app(resources)) as c:
        assert c.post("/ask", json={"question": "q"}).status_code == 409


def test_the_streaming_body_is_closed_even_when_send_is_cancelled():
    """Starlette's stream_response drives body_iterator with a bare `async for`.

    On client disconnect it is cancelled while suspended in `await send(...)`,
    so the body generator stays parked at its `yield` and is never aclose()d:
    its `finally` runs only when the generator is garbage collected through the
    loop's asyncgen hook. This body's finally releases the single-permit GPU
    semaphore, so a reference cycle holding it past the request means every
    later POST /ask blocks forever on acquire and the service needs a restart.

    Driven at the ASGI layer rather than through TestClient, because a test
    client that drains politely never produces the disconnect.
    """
    import asyncio

    from visual_verify.api.app import ClosingStreamingResponse

    released = []

    async def body():
        try:
            yield "first\n"
            yield "second\n"
        finally:
            released.append("released")

    async def run():
        response = ClosingStreamingResponse(body(), media_type="text/event-stream")

        sent = 0

        async def send(message):
            nonlocal sent
            if message["type"] == "http.response.body":
                sent += 1
                if sent == 1:
                    # The disconnect: cancelled while the body is suspended at
                    # its yield, exactly where Starlette would be.
                    raise asyncio.CancelledError

        with contextlib.suppress(asyncio.CancelledError):
            await response.stream_response(send)
        # Read INSIDE the loop. asyncio.run() finalizes every pending async
        # generator during shutdown, so a check after it returns finds the
        # finally has run whether or not anything closed the body, and passes
        # identically with the fix removed. Measured: the first version of this
        # test passed against a stream_response with the aclosing stripped out.
        return list(released)

    ran_before_loop_shutdown = asyncio.run(asyncio.wait_for(run(), timeout=5))

    assert ran_before_loop_shutdown == ["released"], (
        "the body generator's finally did not run inside the request; the GPU "
        "semaphore release is deferred to garbage collection"
    )


def test_a_normal_stream_still_closes_its_body_exactly_once():
    """The wrapper must not double-close or swallow the normal path."""
    import asyncio

    from visual_verify.api.app import ClosingStreamingResponse

    closed = []

    async def body():
        try:
            yield "only\n"
        finally:
            closed.append(1)

    async def run():
        response = ClosingStreamingResponse(body(), media_type="text/event-stream")

        async def send(message):
            return None

        await response.stream_response(send)

    asyncio.run(asyncio.wait_for(run(), timeout=5))
    assert closed == [1]
