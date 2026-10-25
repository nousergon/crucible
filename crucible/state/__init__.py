"""crucible-state: the Postgres schema for crucible's ledgers and history.

Plan S-22 (alpha-engine-config-I12510, epic I12488, milestone M4). This package
declares the schema and the Alembic history that creates it. It is code only.
Nothing in crucible imports it yet, and nothing here opens a connection unless
``CRUCIBLE_STATE_DSN`` is set and a migration is run online. Dual-writing to it
is S-23 (I12511), and readers switch only through a promotion (S-26).

- :mod:`crucible.state.models` is the one declaration of the head schema, as
  SQLAlchemy Core ``Table`` objects.
- ``migrations/`` is the Alembic history. Revision ``0001`` creates schema v1,
  and ``tests/test_state_schema.py`` asserts that it renders to exactly the
  tables in ``models``.

Rules (plan §5): plain Postgres only, with no extension and no Neon-only
feature. Pointers and claims stay on S3, where conditional writes already make
them safe. While dual-write is in observe mode, S3 stays authoritative.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["ALEMBIC_INI", "DSN_ENV", "MIGRATIONS_DIR"]

#: The only place a connection string comes from. alembic.ini carries none.
DSN_ENV = "CRUCIBLE_STATE_DSN"

_HERE = Path(__file__).resolve().parent
ALEMBIC_INI = _HERE / "alembic.ini"
MIGRATIONS_DIR = _HERE / "migrations"
