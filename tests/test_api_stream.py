"""answer_stream() blocks on a GPU and on two hosted providers.

Iterating it inside an async endpoint would freeze the event loop for the whole
answer, so /health would hang and the browser would receive nothing until the
end, which defeats the reason for streaming at all.
"""

import asyncio
import threading
import time

import pytest

from visual_verify.api.stream import iter_in_thread


async def collect(make_iter):
    return [item async for item in iter_in_thread(make_iter)]


def test_items_arrive_in_order():
    assert asyncio.run(collect(lambda: iter([1, 2, 3]))) == [1, 2, 3]


def test_an_exception_from_the_generator_propagates_to_the_consumer():
    """A provider failure mid-answer must reach the endpoint so it can emit an
    error frame. Swallowing it would end the stream indistinguishably from a
    successful short answer."""

    def boom():
        yield 1
        raise RuntimeError("provider exploded")

    with pytest.raises(RuntimeError, match="provider exploded"):
        asyncio.run(collect(boom))


def test_items_before_the_exception_are_still_delivered():
    """Claims already verified and paid for must not be discarded because a
    later one failed."""

    def boom():
        yield 1
        yield 2
        raise RuntimeError("later")

    seen = []

    async def run():
        async for item in iter_in_thread(boom):
            seen.append(item)

    with pytest.raises(RuntimeError):
        asyncio.run(run())

    assert seen == [1, 2]


def test_the_event_loop_keeps_running_while_the_generator_blocks():
    """The whole point. If the generator ran inline, the ticker below could
    not advance while it slept."""
    import time

    ticks = 0

    def slow():
        time.sleep(0.3)
        yield "done"

    async def run():
        nonlocal ticks

        async def tick():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        ticker = asyncio.create_task(tick())
        out = [item async for item in iter_in_thread(slow)]
        ticker.cancel()
        return out

    assert asyncio.run(run()) == ["done"]
    assert ticks > 5, f"the loop was blocked; only {ticks} ticks in 0.3 s"


def test_abandoning_the_stream_early_does_not_hang():
    """A browser tab closing mid-answer aborts the `async for` in the endpoint.
    The generator's cleanup then runs at aclose() time, inside a GeneratorExit,
    and it still has to join the worker thread. If that join deadlocked, every
    disconnect would leak a hung request and the GPU semaphore with it."""

    def three():
        yield 1
        yield 2
        yield 3

    async def run():
        seen = []
        async for item in iter_in_thread(three):
            seen.append(item)
            break
        return seen

    assert asyncio.run(asyncio.wait_for(run(), timeout=5)) == [1]


def test_a_cancelled_consumer_joins_the_producer_only_when_it_closes_the_stream():
    """A disconnect cancels the endpoint coroutine mid-answer. Whoever holds
    the GPU semaphore must not release it while ColQwen2 is still resident, and
    a cancelled `async for` does NOT close the generator, so the join is
    deferred to the loop's asyncgen finalizer. `aclosing` is what pulls it back
    inside the request. This pins the difference so the endpoint cannot quietly
    drop the wrapper."""
    import time
    from contextlib import aclosing

    async def case(use_aclosing):
        joined = []

        def slow():
            yield 1
            time.sleep(0.4)
            joined.append("done")
            yield 2

        async def consumer():
            if use_aclosing:
                async with aclosing(iter_in_thread(slow)) as gen:
                    async for _ in gen:
                        await asyncio.sleep(3600)
            else:
                async for _ in iter_in_thread(slow):
                    await asyncio.sleep(3600)

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return "done" in joined

    async def run():
        return await case(True), await case(False)

    with_aclosing, without = asyncio.run(asyncio.wait_for(run(), timeout=20))
    assert with_aclosing, "aclosing must join the worker before the request ends"
    assert not without, (
        "a bare cancelled `async for` was expected to leave the producer running; "
        "if this now joins, the docstring's aclosing warning is stale"
    )


def test_finalization_does_not_spin_when_cancellation_is_re_delivered():
    """The old finalizer was `while not task.done(): await shield(task)`.

    Plain asyncio.Task cancellation delivers CancelledError once, so that loop
    awaited once and looked correct, which is what the 0.401 s measurement
    above exercises. An anyio cancel scope, which is how Starlette cancels a
    StreamingResponse body on client disconnect, re-raises CancelledError at
    EVERY await point for as long as the scope stays cancelled. The loop then
    returned immediately on every iteration and spun a core flat out for the
    remaining tens of seconds of an in-flight reader or verifier call.

    Simulated here by making the await itself always re-raise, which is the
    property of a cancelled scope that matters, and asserting the finalizer
    stops asking rather than counting CPU time.
    """
    import contextlib

    import visual_verify.api.stream as stream_mod

    shields = []

    async def _cancelled_await():
        # Yields to the loop first, then raises. Both halves matter: a cancelled
        # anyio scope does let the loop run, it just re-raises at the next
        # checkpoint. Raising synchronously instead would starve the executor
        # future forever and make the unfixed code hang here rather than fail,
        # and a hanging test reports worse than a failing one.
        await asyncio.sleep(0)
        raise asyncio.CancelledError

    def counting_shield(arg, **kwargs):
        shields.append(1)
        return _cancelled_await()

    async def run():
        started = threading.Event()

        def producer():
            yield 1
            started.set()
            time.sleep(0.3)
            yield 2

        gen = stream_mod.iter_in_thread(producer)
        async with contextlib.aclosing(gen) as events:
            async for _ in events:
                break
        return None

    monkey = pytest.MonkeyPatch()
    monkey.setattr(stream_mod.asyncio, "shield", counting_shield)
    try:
        with contextlib.suppress(asyncio.CancelledError):
            asyncio.run(asyncio.wait_for(run(), timeout=10))
    finally:
        monkey.undo()

    # One attempt, then the blocking join. The old loop asked until the task
    # completed, which on a 0.3 s producer is thousands of iterations.
    assert len(shields) <= 1, f"finalizer re-awaited {len(shields)} times; it is spinning"


def test_the_producer_is_still_joined_when_cancellation_is_re_delivered():
    """The no-spin fix must not buy its quiet by abandoning the thread. The
    caller releases a GPU semaphore after this returns, so a producer still
    running at that point hands the card to the next request while ColQwen2
    work from this one is in flight."""
    import contextlib

    import visual_verify.api.stream as stream_mod

    finished_at_return = {}

    async def _cancelled_await():
        await asyncio.sleep(0)
        raise asyncio.CancelledError

    def counting_shield(arg, **kwargs):
        return _cancelled_await()

    done = threading.Event()

    async def run():
        def producer():
            yield 1
            time.sleep(0.3)
            done.set()
            yield 2

        gen = stream_mod.iter_in_thread(producer)
        async with contextlib.aclosing(gen) as events:
            async for _ in events:
                break

    monkey = pytest.MonkeyPatch()
    monkey.setattr(stream_mod.asyncio, "shield", counting_shield)
    try:
        with contextlib.suppress(asyncio.CancelledError):
            asyncio.run(asyncio.wait_for(run(), timeout=10))
        finished_at_return["done"] = done.is_set()
    finally:
        monkey.undo()

    assert finished_at_return["done"], "returned before the producer thread finished"
