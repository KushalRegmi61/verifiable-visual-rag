"""Bridge a blocking generator into an async consumer."""

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Iterator

_DONE = object()


async def iter_in_thread[T](make_iter: Callable[[], Iterator[T]]) -> AsyncIterator[T]:
    """Run `make_iter()` on a worker thread, yielding its items as they arrive.

    The queue is unbounded on purpose. A bounded one would give backpressure,
    but if the consumer stops early (a client disconnects) the producer would
    block forever on a full queue and the awaited task would never finish. An
    answer produces a handful of events, so there is nothing to bound.

    The producer is awaited before returning even when the consumer stops
    early. A hosted model call cannot be cancelled cheaply, so the thread runs
    to completion regardless; awaiting it here is what lets the caller hold a
    GPU semaphore until the work is genuinely over rather than until the
    browser lost interest. That await cannot deadlock: `put_nowait` on an
    unbounded queue never blocks, so the producer finishes whether or not
    anyone is still reading.

    The join only happens when this generator is CLOSED, which a caller that
    breaks out of an `async for` does not do. Measured with a producer that
    sleeps 0.5 s past its first item and a consumer cancelled after 0.1 s:
    wrapped in `contextlib.aclosing`, cancellation took 0.401 s and the
    producer was joined; without it, cancellation returned in 0.000 s and the
    producer was still running, finalized later by the loop's asyncgen hook.
    So a caller releasing a GPU semaphore after this must use `aclosing`;
    breaking out on its own defers the join past the end of the request.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    # Set on the worker thread itself, so the join below has a signal that does
    # not require the event loop to be in a runnable state. The asyncio future
    # is not enough: completing it goes through call_soon_threadsafe, which the
    # loop only processes when something is awaiting normally.
    finished = threading.Event()

    def pump() -> None:
        try:
            for item in make_iter():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer side
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)
            finished.set()

    task = loop.run_in_executor(None, pump)
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        cancelled = False
        if not task.done():
            try:
                # Shielded, so cancelling the consumer does not cancel the work
                # whose completion the caller's GPU semaphore is waiting on.
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled and not finished.is_set():
            # ONE retry is not enough and a retry LOOP is worse. Plain
            # asyncio.Task cancellation delivers CancelledError once, which the
            # await above absorbs; an anyio cancel scope, which is how Starlette
            # cancels a StreamingResponse body on client disconnect, re-raises
            # it at EVERY await point for as long as the scope is cancelled. So
            # `while not task.done(): await shield(task)` returns immediately
            # every iteration and spins a core flat out for the remaining tens
            # of seconds of an in-flight reader or verifier call. The 0.401 s
            # measurement in tests/test_api_stream.py uses task.cancel() and
            # cannot see it.
            #
            # Blocking the loop thread here is deliberate and is the lesser
            # cost: the request is over either way, the stall lasts exactly as
            # long as the spin would have, and it does not also burn a core on a
            # machine that is already GPU-starved. Bounded because it cannot
            # deadlock: put_nowait on an unbounded queue never blocks, so pump
            # always reaches its finally.
            finished.wait()
        if cancelled:
            raise asyncio.CancelledError
