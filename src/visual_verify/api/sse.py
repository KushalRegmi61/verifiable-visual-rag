"""Server-Sent Events framing.

Hand-rolled rather than pulling in sse-starlette. The format is three lines,
this is directly unit-testable, and the repository gates everything else behind
extras for a reason.
"""

import json


def frame(event: str, data: dict) -> str:
    """One SSE frame: an event name, a JSON payload, and a blank separator.

    json.dumps is load-bearing beyond serialization. A raw newline inside a
    `data:` field terminates that field, so a verifier reason spanning two
    lines would truncate the event and the browser would parse the tail as a
    separate malformed frame. Escaping it is what stops that.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
