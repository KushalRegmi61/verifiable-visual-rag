"""Plain helpers shared by test modules.

Not conftest: conftest is for fixtures, and importing a name from it directly is
documented as unsupported. It works today only because tests/ has no __init__.py
and nothing sets --import-mode, so pytest's prepend mode puts tests/ on sys.path.
That breaks under --import-mode=importlib, which is where pytest is heading.
"""

import pytest


def skip_if_no_quota(exc: Exception) -> None:
    """Turn an unreachable provider into a skip, but never a wrong answer.

    A 429 with `limit: 0` means the key's project has no quota for the model at
    all, which is a billing state and not a defect in this code. Failing on it
    would leave the suite red on any machine whose account is not provisioned,
    including a fresh clone.

    Deliberately narrow. Only transport and quota problems skip. A response
    that arrives and is malformed, or a verdict that is simply wrong, must
    still FAIL: those are the two things the live files exist to catch, and
    swallowing them would make a broken verifier look like an unconfigured one.

    Shared rather than copied, because every live caller must report the same
    thing about the same 429. It was inlined into the strictness probes once and
    the copy silently lost the rate-limit branch, which would have let one file
    skip where the other failed.
    """
    text = str(exc)
    unreachable = (
        "RESOURCE_EXHAUSTED" in text
        or "429" in text
        or "insufficient_quota" in text
        or "rate limit" in text.lower()
    )
    if unreachable:
        pytest.skip(f"provider reachable but unprovisioned: {text[:160]}")
    raise exc
