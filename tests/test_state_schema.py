"""crucible-state schema v1 and its Alembic history, checked offline.

alpha-engine-config-I12510 (plan S-22). No test here opens a connection.
``alembic upgrade head --sql`` renders the postgresql DDL with no database.
The live upgrade/downgrade round trip on the staging branch is S-22's
acceptance step, and it needs S-10's DSN.
"""

from __future__ import annotations

import configparser
import importlib.util
import io
import re

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from crucible import ledger
from crucible.models import TrialRow
from crucible.state import ALEMBIC_INI, DSN_ENV, MIGRATIONS_DIR
from crucible.state import models as m


def _config() -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.attributes["configure_logger"] = False
    return cfg


def _render(monkeypatch, *, downgrade: bool = False) -> str:
    monkeypatch.delenv(DSN_ENV, raising=False)
    cfg = _config()
    cfg.output_buffer = io.StringIO()
    if downgrade:
        command.downgrade(cfg, "0001:base", sql=True)
    else:
        command.upgrade(cfg, "head", sql=True)
    return cfg.output_buffer.getvalue()


def _norm(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


def _migration_module():
    path = MIGRATIONS_DIR / "versions" / "0001_schema_v1.py"
    spec = importlib.util.spec_from_file_location("crucible_state_0001", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# History shape
# --------------------------------------------------------------------------


def test_history_has_one_head_and_it_is_0001() -> None:
    script = ScriptDirectory.from_config(_config())
    assert script.get_heads() == ["0001"]
    assert script.get_revision("0001").down_revision is None


def test_alembic_ini_carries_no_database_url() -> None:
    parser = configparser.ConfigParser()
    parser.read(ALEMBIC_INI)
    assert not parser.has_option("alembic", "sqlalchemy.url")


def test_online_migration_refuses_without_a_dsn(monkeypatch) -> None:
    monkeypatch.delenv(DSN_ENV, raising=False)
    with pytest.raises(SystemExit, match=DSN_ENV):
        command.upgrade(_config(), "head")


# --------------------------------------------------------------------------
# 0001 renders to exactly the models
# --------------------------------------------------------------------------


def test_upgrade_creates_exactly_the_model_tables(monkeypatch) -> None:
    sql = _render(monkeypatch)
    created = set(re.findall(rf"CREATE TABLE {m.SCHEMA}\.(\w+) \(", sql))
    assert created == {t.name for t in m.metadata.sorted_tables}
    assert set(_migration_module().TABLES) == created
    assert m.APPEND_ONLY_TABLES and set(m.APPEND_ONLY_TABLES) == created


def _elements(create_table_sql: str) -> set[str]:
    """The column and constraint clauses of one CREATE TABLE, order-free."""
    body = create_table_sql[create_table_sql.index("(") + 1 : create_table_sql.rindex(")")]
    return {line.strip().rstrip(",") for line in body.splitlines() if line.strip()}


def test_upgrade_ddl_matches_the_models_table_by_table(monkeypatch) -> None:
    """Every column, type, default and constraint: 0001 is the models at v1.

    Compared clause by clause, because Alembic orders a table's constraints
    differently from ``CreateTable``; the clauses themselves must be identical.
    """
    sql = _render(monkeypatch)
    dialect = postgresql.dialect()
    for table in m.metadata.sorted_tables:
        match = re.search(rf"^CREATE TABLE {m.SCHEMA}\.{table.name} \(\n.*?^\)", sql, re.M | re.S)
        assert match, f"0001 does not create {table.name}"
        declared = str(CreateTable(table).compile(dialect=dialect))
        assert _elements(match.group(0)) == _elements(declared), table.name
        for index in table.indexes:
            assert _norm(str(CreateIndex(index).compile(dialect=dialect))) in _norm(sql)


def test_schema_is_plain_postgres(monkeypatch) -> None:
    sql = _render(monkeypatch).upper()
    assert "CREATE EXTENSION" not in sql
    assert "NEON" not in sql


# --------------------------------------------------------------------------
# The two 10-21 S7 gaps
# --------------------------------------------------------------------------


def test_trial_key_is_the_ledgers_own_identity_as_of() -> None:
    """Gap 1: the issue says cycle_date, the ledger keys on as_of."""
    pk = [c.name for c in m.trial.primary_key.columns]
    assert pk == ["slot", "arm_id", "as_of"]
    row = {"slot": "M", "arm_id": "arm", "as_of": "2026-10-22"}
    assert ledger._identity(row) == tuple(row[c] for c in pk)
    assert "cycle_date" not in m.trial.c


def test_trial_columns_cover_every_core_trial_row_field() -> None:
    columns = set(m.trial.c.keys())
    renamed = {"written_at_utc": "written_at"}  # the row's own timestamp, never now()
    for name in TrialRow.model_fields:
        column = renamed.get(name, name)
        assert column in columns, f"TrialRow.{name} has no trial column"
    assert m.trial.c.written_at.server_default is None
    assert isinstance(m.trial.c.as_of.type, sa.Date)


def test_fill_is_keyed_on_the_broker_exec_id_and_refuses_an_empty_one() -> None:
    """Gap 2: a fill's identity is IB's execId, never its own fields."""
    pk = [c.name for c in m.fill.primary_key.columns]
    assert pk == ["account", "broker_exec_id"]
    assert m.fill.c.broker_exec_id.nullable is False
    checks = {
        c.name: str(c.sqltext) for c in m.fill.constraints if isinstance(c, sa.CheckConstraint)
    }
    assert checks["fill_broker_exec_id_nonempty"] == "broker_exec_id <> ''"


# --------------------------------------------------------------------------
# Money: exact decimals, double entry balanced at commit, append-only
# --------------------------------------------------------------------------


def test_money_and_quantity_columns_are_exact_decimals() -> None:
    money = re.compile(r"(_usd|price|amount|quantity)$")
    hits = 0
    for table in m.metadata.sorted_tables:
        for column in table.c:
            if money.search(column.name):
                hits += 1
                assert isinstance(column.type, sa.Numeric), f"{table.name}.{column.name}"
                assert not isinstance(column.type, sa.Float), f"{table.name}.{column.name}"
                assert (column.type.precision, column.type.scale) == (20, 6)
    assert hits >= 10


def test_every_table_is_append_only(monkeypatch) -> None:
    sql = _norm(_render(monkeypatch))
    assert f"CREATE FUNCTION {m.SCHEMA}.refuse_mutation() RETURNS trigger" in sql
    for name in m.APPEND_ONLY_TABLES:
        assert (
            f"CREATE TRIGGER {name}_append_only BEFORE UPDATE OR DELETE ON {m.SCHEMA}.{name} "
            f"FOR EACH ROW EXECUTE FUNCTION {m.SCHEMA}.refuse_mutation()"
        ) in sql
        assert (
            f"CREATE TRIGGER {name}_no_truncate BEFORE TRUNCATE ON {m.SCHEMA}.{name} "
            f"FOR EACH STATEMENT EXECUTE FUNCTION {m.SCHEMA}.refuse_mutation()"
        ) in sql


def test_journal_balance_is_checked_at_commit_on_both_tables(monkeypatch) -> None:
    sql = _norm(_render(monkeypatch))
    for name in ("journal_entry", "posting"):
        assert (
            f"CREATE CONSTRAINT TRIGGER {name}_balanced AFTER INSERT ON {m.SCHEMA}.{name} "
            "DEFERRABLE INITIALLY DEFERRED "
            f"FOR EACH ROW EXECUTE FUNCTION {m.SCHEMA}.assert_entry_balanced()"
        ) in sql
    # The function refuses fewer than two postings and any non-zero per-currency sum.
    assert "IF n_postings < 2 THEN" in sql
    assert "GROUP BY currency HAVING sum(amount) <> 0" in sql


def test_money_ledger_shapes_match_the_producer() -> None:
    """crucible-executor executor/money_ledger.py, money_ledger.v1."""
    assert re.fullmatch(m.ENTRY_ID_PATTERN, "je_" + "0" * 32)
    assert not re.fullmatch(m.ENTRY_ID_PATTERN, "je_" + "0" * 64)
    for account in (
        "assets:cash:USD",
        "assets:accrued_interest:USD",
        "assets:securities:BRK.B",
        "assets:broker_mark_adjustment",
        "income:unrealized_gain:valuation_correction",
    ):
        assert re.fullmatch(m.ACCOUNT_PATTERN, account), account
    assert not re.fullmatch(m.ACCOUNT_PATTERN, "cash")
    posting_fks = {fk.parent.name: fk.column.table.name for fk in m.posting.foreign_keys}
    assert posting_fks == {"entry_id": "journal_entry", "account": "ledger_account"}
    assert [k for k, _, _ in _migration_module().SEED_KINDS] == [
        "ib_mark_to_settled_close",
        "settled_restrike",
    ]


# --------------------------------------------------------------------------
# Downgrade never drops a ledger that holds rows
# --------------------------------------------------------------------------


def test_downgrade_refuses_before_dropping_anything(monkeypatch) -> None:
    sql = _render(monkeypatch, downgrade=True)
    refuse = sql.index("downgrade of 0001 refused")
    first_drop = sql.index("DROP TABLE")
    assert refuse < first_drop
    for name in m.APPEND_ONLY_TABLES:
        assert f"EXISTS (SELECT 1 FROM {m.SCHEMA}.{name})" in sql
