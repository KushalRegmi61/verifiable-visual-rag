"""The four routes.

build_app(resources) takes its resources rather than constructing them, which
is what lets the tests run the whole surface against FakeChat and FakeEmbedder
with no GPU and no key. create_app() is the production entry point that builds
them from the environment.
"""

import asyncio
import math
from collections.abc import AsyncIterator
from contextlib import aclosing
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from visual_verify.agent.core import AgentError
from visual_verify.api.ask import AskRequest, NoPagesIndexed, ask_events
from visual_verify.api.resources import Resources
from visual_verify.api.sse import frame
from visual_verify.api.stream import iter_in_thread
from visual_verify.api.wire import to_frame
from visual_verify.prepare import PageNotFound
from visual_verify.store.models import Document, Page


def _json_safe(value):
    """The same structure with every non-finite float replaced by its name.

    Only ever applied to error payloads. NaN and the infinities have no JSON
    spelling, and json.dumps in strict mode raises rather than emitting one.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class ClosingStreamingResponse(StreamingResponse):
    """A StreamingResponse that always closes its body iterator.

    Starlette's stream_response drives `body_iterator` with a bare `async for`.
    On client disconnect it is cancelled while suspended in `await send(...)`,
    so the generator stays parked at its `yield` and is never `aclose()`d: its
    `finally` runs only when the generator is garbage collected through the
    loop's asyncgen hook. For a body whose `finally` releases a single-permit
    GPU semaphore, a reference cycle holding it to the next gc pass means every
    later POST /ask blocks forever on acquire and the service needs a restart.

    This is the same failure `iter_in_thread` documents one level down, fixed
    there with `aclosing` on the consumer side. The response is the one place
    that can guarantee it for the body, because the response owns the iteration.
    """

    async def stream_response(self, send) -> None:
        async with aclosing(self.body_iterator):
            await super().stream_response(send)


def _cors_origins(settings) -> list[str]:
    """Origins allowed to call this service.

    Configurable because the frontend's own API base already is, via
    NEXT_PUBLIC_API. A hardcoded localhost:3000 means a UI on 127.0.0.1, on port
    3001 because 3000 was taken, or on any real host, has every fetch and every
    page image blocked by preflight while the server logs a normal 200 and the
    browser shows only "TypeError: Failed to fetch".

    Deliberately not defaulting to "*": this service holds two billable API keys
    behind it, and /ask spends money per call.
    """
    return list(settings.cors_origins)


def build_app(resources: Resources) -> FastAPI:
    app = FastAPI(title="Verifiable Visual RAG")
    app.state.resources = resources

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(resources.settings),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # The GPU is single-tenant on a 3.63 GB card. Two concurrent asks would put
    # two queries through one embedder and, under load, OOM it. A second
    # request waits instead. Released in a finally, or one disconnected client
    # deadlocks every later question.
    ask_lock = asyncio.Semaphore(1)
    # Exposed because "is the lock still held" is otherwise unobservable from
    # outside the closure, and it is the only visible symptom of a leaked
    # release path.
    app.state.ask_lock = ask_lock

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """422 with the offending value made printable.

        FastAPI's default handler echoes the rejected input back inside the
        error detail, and starlette's JSONResponse renders with strict
        json.dumps. A body carrying bare `NaN` (which stdlib json.loads accepts
        and stdlib json.dumps emits) therefore validated correctly, was
        correctly rejected by AskRequest, and then raised ValueError while
        SERIALIZING the rejection. Measured before this handler existed: the
        request died inside request_validation_exception_handler instead of
        returning 422, so the one input the threshold validator exists to stop
        was also the one input that could not be reported.
        """
        return JSONResponse(
            status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))}
        )

    @app.get("/health")
    def health() -> dict:
        return {
            "reader_model": resources.reader_chat.model_id,
            "verifier_model": resources.verifier_chat.model_id,
            "embedder_resident": resources.embedder is not None,
            "indexed_pages": resources.index.count(),
        }

    @app.get("/documents")
    def documents() -> list[dict]:
        with Session(resources.engine) as session:
            docs = session.scalars(select(Document).order_by(Document.path)).all()
            return [
                {
                    "sha": d.sha256,
                    "name": Path(d.path).name,
                    "n_pages": d.n_pages,
                    "status": d.status,
                }
                for d in docs
            ]

    @app.get("/documents/{sha}/pages/{page_no}/image")
    def page_image(sha: str, page_no: int) -> FileResponse:
        """The filename is resolved through the database, never from the URL,
        so no user-supplied string reaches a filesystem path."""
        with Session(resources.engine) as session:
            page = session.scalar(select(Page).where(Page.doc_sha == sha, Page.page_no == page_no))
        if page is None:
            raise HTTPException(404, f"no page {page_no} in {sha}")
        path = resources.settings.pages_dir / page.image_path
        if not path.exists():
            raise HTTPException(404, "the page image is missing from disk")
        return FileResponse(path, media_type="image/png")

    @app.post("/ask")
    async def ask(request: AskRequest) -> StreamingResponse:
        # The semaphore is taken HERE, before ask_events, not around the
        # streaming body alone. ask_events does the retrieval embed_query on
        # its way to returning an iterator, so guarding only the body would
        # leave that first GPU call unguarded and two simultaneous questions
        # would put two queries through one embedder while it is resident.
        # The consequence is that a queued request holds the lock across the
        # whole of retrieval, page preparation and the answer loop, so asks are
        # strictly serial rather than merely non-overlapping on the GPU. On a
        # 3.63 GB single-tenant card that is the intended trade.
        await ask_lock.acquire()
        session = Session(resources.engine)
        try:
            # Off the event loop. ask_events is a plain blocking function and
            # does all of retrieval on its way to returning an iterator:
            # index.count() over the network, a multi-second embed_query on the
            # GPU, then prepare_page's DB query and two Qdrant round trips. Run
            # inline, the loop is blocked from the moment the user hits Ask
            # until the first `retrieved` frame, so the browser's parallel GET
            # for the page image hangs, /health hangs, and a second tab looks
            # frozen rather than queued. Asks are serial by design because of
            # the semaphore; the HTTP server stalling with them is not part of
            # that trade. Exceptions propagate exactly as before.
            iterator = await asyncio.to_thread(
                ask_events,
                request,
                session=session,
                index=resources.index,
                embedder=resources.embedder,
                reader_chat=resources.reader_chat,
                verifier_chat=resources.verifier_chat,
                settings=resources.settings,
            )
        except BaseException as exc:
            # Nothing has been written yet, so every one of these can still be
            # a status code rather than an error frame inside a 200.
            session.close()
            ask_lock.release()
            if isinstance(exc, NoPagesIndexed):
                # 409: the corpus is a real resource in a state that forbids
                # the request, and the fix (`vvrag embed --all`) is the
                # operator's, not a retry.
                raise HTTPException(409, str(exc)) from exc
            if isinstance(exc, PageNotFound):
                raise HTTPException(404, str(exc)) from exc
            if isinstance(exc, AgentError):
                # 500, not 503. AgentError here means reader and verifier are
                # the same model, which is a deployment mistake that no amount
                # of waiting fixes; 503 would invite a retry loop against a
                # service that will refuse identically every time.
                raise HTTPException(500, str(exc)) from exc
            raise

        async def body() -> AsyncIterator[str]:
            try:
                # aclosing is load-bearing. iter_in_thread joins its producer
                # thread only when it is CLOSED, and a consumer that breaks out
                # of an `async for`, or is cancelled inside one, never closes
                # it. Measured in tests/test_api_stream.py: with aclosing a
                # cancelled consumer returns in 0.401 s with the producer
                # joined, without it in 0.000 s with the producer still
                # running. Since the finally below releases the GPU semaphore,
                # dropping aclosing would hand the card to the next request
                # while ColQwen2 work from this one is still in flight.
                #
                # No test here pins it, and the attempt is worth recording. A
                # client disconnect driven straight at the ASGI app (real
                # scope, real http.disconnect) cancels the body while it is
                # suspended inside iter_in_thread's `await queue.get()`, and a
                # CancelledError delivered THERE runs that generator's own
                # finally, so the producer is joined either way: measured 1.01 s
                # to release with aclosing and 1.01 s without, with the worker
                # finished at release in both. The case aclosing governs is the
                # one where this generator is suspended at its `yield` instead,
                # which needs a client that has stopped draining the socket and
                # which could not be produced from a test harness. The
                # underlying difference is measured in tests/test_api_stream.py.
                async with aclosing(iter_in_thread(lambda: iterator)) as events:
                    async for event in events:
                        name, payload = to_frame(event)
                        yield frame(name, payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reported to the client
                # A provider dying on claim two is genuinely mid-stream. The
                # claims already delivered stay on the screen, and ending the
                # stream silently would be indistinguishable from a short
                # successful answer.
                yield frame("error", {"message": str(exc)})
            finally:
                session.close()
                ask_lock.release()

        return ClosingStreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def create_app() -> FastAPI:
    """Production entry point: uvicorn visual_verify.api.app:create_app --factory"""
    from visual_verify.api.resources import StartupRefused, build
    from visual_verify.config import Settings

    try:
        resources = build(Settings.from_env())
    except StartupRefused:
        # Already a readable sentence naming the variable to set. Anything
        # wrapping it here would only bury that.
        raise
    except SystemExit as exc:
        # _make_index raises SystemExit when VVRAG_QDRANT_URL is unset. Under
        # uvicorn --factory that would exit the process with a bare code and no
        # explanation, which reads as a crash rather than a refusal.
        raise StartupRefused(f"the retrieval index could not be built: {exc}") from exc
    except Exception as exc:
        # SchemaMismatch on a collection built with the wrong vector config,
        # and connection errors from QdrantIndex, both reach here. build() has
        # no test coverage past its configuration check, so the honest thing is
        # to name the failing stage rather than let a raw traceback out of a
        # lifespan and leave the operator guessing which of the four
        # constructions blew up.
        raise StartupRefused(
            f"startup failed while building resources: {type(exc).__name__}: {exc}"
        ) from exc
    return build_app(resources)
