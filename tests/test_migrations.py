"""The migration must produce the same schema the models declare.

Autogenerate drift is silent and only bites at deploy time, so compare the two.
"""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from visual_verify.store.models import Base

EXPECTED_TABLES = {"documents", "pages", "boxes", "jobs"}
REPO_ROOT = Path(__file__).parent.parent
# The packaged ini, the one the installed `vvrag` loads. The migration assets
# live inside the package so they ship in the wheel; the repo-root alembic.ini
# is only a shim pointing back here, and is covered by its own test below.
ALEMBIC_INI = REPO_ROOT / "src" / "visual_verify" / "alembic.ini"
ROOT_ALEMBIC_INI = REPO_ROOT / "alembic.ini"


def _upgrade_to(db_path: Path, monkeypatch, ini: Path = ALEMBIC_INI) -> None:
    """Run migrations against a throwaway database.

    Uses the Alembic Python API rather than a subprocess so monkeypatch's env
    var actually reaches env.py, which reads VVRAG_DB_URL via Settings.
    """
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{db_path}")
    command.upgrade(Config(str(ini)), "head")


def test_repo_root_ini_runs_the_same_migrations(tmp_path, monkeypatch):
    """`uv run alembic upgrade head` from the repo root must still work.

    The README documents it, and the shim's script_location is a hand-written
    relative path that nothing else would catch if it went stale.
    """
    db = tmp_path / "from_root_ini.db"
    _upgrade_to(db, monkeypatch, ini=ROOT_ALEMBIC_INI)

    tables = set(inspect(create_engine(f"sqlite:///{db}")).get_table_names())
    assert EXPECTED_TABLES <= tables


def test_migration_creates_expected_tables(tmp_path, monkeypatch):
    db = tmp_path / "migrated.db"
    _upgrade_to(db, monkeypatch)

    tables = set(inspect(create_engine(f"sqlite:///{db}")).get_table_names())
    assert EXPECTED_TABLES <= tables


def test_models_and_migration_agree_on_columns(tmp_path, monkeypatch):
    db = tmp_path / "migrated.db"
    _upgrade_to(db, monkeypatch)

    inspector = inspect(create_engine(f"sqlite:///{db}"))
    for table in EXPECTED_TABLES:
        migrated = {c["name"] for c in inspector.get_columns(table)}
        declared = {c.name for c in Base.metadata.tables[table].columns}
        assert declared == migrated, f"{table} drifted: {declared ^ migrated}"


def test_migration_creates_the_page_uniqueness_index(tmp_path, monkeypatch):
    """The (doc_sha, page_no) uniqueness backstop must survive into the migration."""
    db = tmp_path / "migrated.db"
    _upgrade_to(db, monkeypatch)

    indexes = inspect(create_engine(f"sqlite:///{db}")).get_indexes("pages")
    unique_cols = [set(i["column_names"]) for i in indexes if i["unique"]]
    assert {"doc_sha", "page_no"} in unique_cols


def test_no_model_migration_drift(tmp_path, monkeypatch):
    """The strongest possible drift assertion, in one line.

    Runs against a throwaway database upgraded to head rather than the developer's
    own data/index.db, so the result depends on the migrations alone and not on
    whatever state the local file happens to be in.
    """
    db = tmp_path / "drift.db"
    _upgrade_to(db, monkeypatch)

    command.check(Config(str(ALEMBIC_INI)))  # raises if models and migrations disagree
