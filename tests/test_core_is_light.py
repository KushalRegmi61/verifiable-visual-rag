"""The shippable core must not transitively import the store.

This runs in a subprocess on purpose. The test environment has SQLAlchemy
installed (we need it to test the store at all), so an in-process check could
only prove the package is uninstalled, not that the core avoids importing it.
"""

import subprocess
import sys

FORBIDDEN = ["sqlalchemy", "alembic", "qdrant_client", "fastapi"]


def _modules_after_importing(module: str) -> set[str]:
    code = f"import {module}, sys; print('\\n'.join(sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def test_core_import_pulls_no_store_dependency():
    loaded = _modules_after_importing("visual_verify")
    leaked = [m for m in FORBIDDEN if m in loaded]
    assert not leaked, f"core leaked heavy deps: {leaked}"
