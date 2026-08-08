"""The shippable core must not transitively import the store.

This runs in a subprocess on purpose. The test environment has SQLAlchemy
installed (we need it to test the store at all), so an in-process check could
only prove the package is uninstalled, not that the core avoids importing it.
"""

import subprocess
import sys

# torch and transformers are listed before S3 introduces them, so the guard is
# armed the moment the retriever is tempted to import a model at package import.
FORBIDDEN = ["sqlalchemy", "alembic", "qdrant_client", "fastapi", "torch", "transformers"]


def _modules_after_importing(module: str) -> set[str]:
    code = f"import {module}, sys; print('\\n'.join(sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def test_core_import_pulls_no_store_dependency():
    loaded = _modules_after_importing("visual_verify")
    leaked = [m for m in FORBIDDEN if m in loaded]
    assert not leaked, f"core leaked heavy deps: {leaked}"


def test_grounding_pulls_no_store_or_model_dependency():
    """Grounding must stay usable without Qdrant, torch, or a GPU.

    If this fails, something in grounding/ started fetching its own inputs
    instead of taking them as arguments.
    """
    loaded = _modules_after_importing("visual_verify.grounding")
    leaked = [m for m in FORBIDDEN if m in loaded]
    assert not leaked, f"grounding leaked heavy deps: {leaked}"


def test_verify_pulls_no_store_or_model_dependency():
    """verify() must import without torch or transformers, ever.

    backends.py lives inside the package and its local classes import the
    model libraries lazily, at call time. If this fails, a lazy import
    drifted to module level and the whole package became unimportable
    without a GPU machine.
    """
    loaded = _modules_after_importing("visual_verify.verify")
    leaked = [m for m in FORBIDDEN if m in loaded]
    assert not leaked, f"verify leaked heavy deps: {leaked}"
