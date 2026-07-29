"""Engine construction.

SQLite disables foreign key enforcement by default, so a Page referencing a
missing Document would succeed in development and fail on Postgres. Turning the
pragma on keeps the two backends behaving the same way.
"""

from sqlalchemy import Engine, create_engine, event


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def make_engine(db_url: str, **kwargs) -> Engine:
    """Create an engine with backend differences smoothed over."""
    engine = create_engine(db_url, **kwargs)
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine
