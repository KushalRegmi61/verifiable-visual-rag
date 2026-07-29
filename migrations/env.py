from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import make_url

from visual_verify.config import Settings
from visual_verify.store.models import Base, UtcDateTime

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# The connection string comes from the environment, never from this repo. Keeps
# migrations under the same "no module hardcodes a connection" rule as the app.
_url = make_url(Settings.from_env().db_url)
# set_main_option feeds ConfigParser, which reads "%" as interpolation syntax.
# Managed-Postgres passwords routinely contain percent-encoded characters, so a
# raw "%" must be doubled or the URL silently mangles.
config.set_main_option(
    "sqlalchemy.url", _url.render_as_string(hide_password=False).replace("%", "%%")
)

# On a fresh clone `data/` does not exist yet, and SQLite will not create a
# missing parent directory: it fails with a bare "unable to open database file".
# The CLI creates the directory, but migrations run before the CLI ever does.
if _url.drivername.startswith("sqlite") and _url.database not in (None, "", ":memory:"):
    Path(_url.database).parent.mkdir(parents=True, exist_ok=True)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def render_item(type_, obj, autogen_context):
    """Keep generated migrations free of imports from the application package.

    Autogenerate would otherwise emit a bare
    `visual_verify.store.models.UtcDateTime()` reference with no import, which
    fails at NameError. UtcDateTime is a Python-side coercion whose DDL is
    exactly DateTime(timezone=True), so rendering the underlying type is both
    correct and keeps old migrations runnable after the model is refactored.
    """
    if type_ == "type" and isinstance(obj, UtcDateTime):
        return "sa.DateTime(timezone=True)"
    return False

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_item=render_item,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_item=render_item,
            # SQLite cannot ALTER a constraint into existence: adding the jobs
            # foreign key raises NotImplementedError without this. Batch mode
            # rebuilds the table via copy-and-move instead. It is a no-op on
            # Postgres, which alters in place.
            render_as_batch=connection.dialect.name == "sqlite",
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
