"""Alembic environment for crucible-state.

The connection string comes from ``CRUCIBLE_STATE_DSN`` and nowhere else.
``alembic.ini`` carries no ``sqlalchemy.url``.

- **Offline** (``--sql``) renders DDL for the postgresql dialect with no
  connection. With no DSN set, the dialect is named directly. This is what CI
  runs (``tests/test_state_schema.py``).
- **Online** refuses to start without the DSN. A migration that silently fell
  back to a default database is exactly the accident the missing URL is there
  to prevent.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from crucible.state import DSN_ENV
from crucible.state.models import metadata

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)


def run_migrations_offline() -> None:
    dsn = os.environ.get(DSN_ENV)
    if dsn:
        context.configure(
            url=dsn, target_metadata=metadata, literal_binds=True, include_schemas=True
        )
    else:
        context.configure(
            dialect_name="postgresql",
            target_metadata=metadata,
            literal_binds=True,
            include_schemas=True,
        )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    dsn = os.environ.get(DSN_ENV)
    if not dsn:
        raise SystemExit(
            f"{DSN_ENV} is not set; refusing to run crucible-state migrations online. "
            "Render offline with --sql, or set the DSN for the target branch."
        )
    engine = create_engine(dsn, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=metadata, include_schemas=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
