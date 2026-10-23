"""crucible-state schema v1: trial ledger, trader ledger, money ledger, history.

Revision ID: 0001
Revises:
Create Date: 2026-10-23

alpha-engine-config-I12510 (plan S-22). The tables are exactly those of
:mod:`crucible.state.models` at v1, and ``tests/test_state_schema.py`` asserts
that. This file is FROZEN once applied anywhere: a schema change is a new
revision, never an edit here.

Beyond the tables, this revision installs:

- ``refuse_mutation()``, a BEFORE UPDATE/DELETE row trigger and a BEFORE
  TRUNCATE statement trigger on every table. The schema is append-only, and a
  correction is a new row (for money, a new journal entry).
- ``assert_entry_balanced()``, a DEFERRABLE INITIALLY DEFERRED constraint
  trigger on ``journal_entry`` and ``posting``. At commit, every entry touched
  by the transaction must have at least two postings that sum to zero per
  currency, or the commit fails. An unbalanced entry cannot be committed
  (Brian, 2026-10-22).
- The two journal-entry kinds the producer posts today (crucible-executor
  ``money_ledger.KINDS``). The chart of accounts is NOT seeded: accounts are
  created by the writer, and which accounts exist is aec#12558's review.

``downgrade()`` refuses to drop a table that holds a row. Dropping a ledger is
an irreversible delete (red line 4), so it is never a side effect of a
migration rollback.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import Text
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "crucible_state"

#: Creation order (dependencies first); downgrade drops in reverse.
TABLES: tuple[str, ...] = (
    "arena_cycle",
    "champion_history",
    "execution_shortfall",
    "fill",
    "gate_reading",
    "journal_entry_kind",
    "ledger_account",
    "morning_history",
    "nav_mark",
    "position_snapshot",
    "reconciliation",
    "trade_order",
    "trial",
    "journal_entry",
    "reconciliation_discrepancy",
    "trade_execution",
    "posting",
)

#: crucible-executor ``money_ledger.KINDS`` (source ``daily_auto_correct``).
SEED_KINDS: tuple[tuple[str, str, str], ...] = (
    (
        "ib_mark_to_settled_close",
        "daily_auto_correct",
        "The live run moving the broker's marks (IB, including after-hours prints) "
        "onto Polygon's settled closes.",
    ),
    (
        "settled_restrike",
        "daily_auto_correct",
        "The T+1 audit re-striking a published row on corrected settled data, or "
        "onto the settled basis.",
    ),
)

REFUSE_MUTATION = f"""
CREATE FUNCTION {SCHEMA}.refuse_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '{SCHEMA}.% is append-only: % refused', TG_TABLE_NAME, TG_OP
        USING ERRCODE = 'restrict_violation',
              HINT = 'A correction is a new row; for money, a new journal entry.';
END
$$
"""

ASSERT_ENTRY_BALANCED = f"""
CREATE FUNCTION {SCHEMA}.assert_entry_balanced() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    n_postings integer;
    unbalanced text;
BEGIN
    SELECT count(*) INTO n_postings
      FROM {SCHEMA}.posting WHERE entry_id = NEW.entry_id;
    IF n_postings < 2 THEN
        RAISE EXCEPTION 'journal entry % has % posting(s); an entry needs at least two',
            NEW.entry_id, n_postings
            USING ERRCODE = 'check_violation';
    END IF;
    SELECT string_agg(currency || ' ' || total::text, ', ' ORDER BY currency)
      INTO unbalanced
      FROM (SELECT currency, sum(amount) AS total
              FROM {SCHEMA}.posting
             WHERE entry_id = NEW.entry_id
             GROUP BY currency
            HAVING sum(amount) <> 0) AS per_currency;
    IF unbalanced IS NOT NULL THEN
        RAISE EXCEPTION 'journal entry % does not balance: %', NEW.entry_id, unbalanced
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NULL;
END
$$
"""


def _append_only(table: str) -> list[str]:
    return [
        f"CREATE TRIGGER {table}_append_only BEFORE UPDATE OR DELETE ON {SCHEMA}.{table} "
        f"FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.refuse_mutation()",
        f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {SCHEMA}.{table} "
        f"FOR EACH STATEMENT EXECUTE FUNCTION {SCHEMA}.refuse_mutation()",
    ]


def _balanced(table: str) -> str:
    return (
        f"CREATE CONSTRAINT TRIGGER {table}_balanced AFTER INSERT ON {SCHEMA}.{table} "
        "DEFERRABLE INITIALLY DEFERRED "
        f"FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.assert_entry_balanced()"
    )


def _refuse_if_any_rows() -> str:
    probes = " OR ".join(f"EXISTS (SELECT 1 FROM {SCHEMA}.{t})" for t in TABLES)
    return (
        "DO $$ BEGIN "
        f"IF {probes} THEN "
        f"RAISE EXCEPTION '{SCHEMA} holds rows; downgrade of 0001 refused "
        "(dropping a ledger is an irreversible delete)' USING ERRCODE = 'restrict_violation'; "
        "END IF; END $$"
    )


def upgrade() -> None:
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    op.create_table(
        "arena_cycle",
        sa.Column("slot", sa.Text(), nullable=False),
        sa.Column("trading_day", sa.Date(), nullable=False),
        sa.Column("s3_key", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=Text()), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("slot", "trading_day", name="arena_cycle_pkey"),
        schema="crucible_state",
    )
    op.create_table(
        "champion_history",
        sa.Column("slot", sa.Text(), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("arm_id", sa.Text(), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("code_sha", sa.Text(), nullable=False),
        sa.Column("promotion_source", sa.Text(), nullable=False),
        sa.Column("manifest_key", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=Text()), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("slot", "decided_at", name="champion_history_pkey"),
        schema="crucible_state",
    )
    op.create_table(
        "execution_shortfall",
        sa.Column("trading_day", sa.Date(), nullable=False),
        sa.Column("calendar_date", sa.Date(), nullable=False),
        sa.Column("champion", sa.Text(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("outcome_reason", sa.Text(), nullable=True),
        sa.Column("n_orders", sa.Integer(), nullable=True),
        sa.Column("n_filled", sa.Integer(), nullable=True),
        sa.Column("notional_usd", sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column("shortfall_usd", sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column("shortfall_bps_notional_weighted", sa.Double(), nullable=True),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=Text()), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("trading_day", "champion", name="execution_shortfall_pkey"),
        schema="crucible_state",
    )
    op.create_table(
        "fill",
        sa.Column("trading_day", sa.Date(), nullable=False),
        sa.Column("account", sa.Text(), nullable=False),
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("shares", sa.Integer(), nullable=False),
        sa.Column("cash_usd", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column("broker_exec_id", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=Text()), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("broker_exec_id <> ''", name="fill_broker_exec_id_nonempty"),
        sa.PrimaryKeyConstraint("account", "broker_exec_id", name="fill_pkey"),
        schema="crucible_state",
    )
    op.create_table(
        "gate_reading",
        sa.Column("gate", sa.Text(), nullable=False),
        sa.Column("trading_day", sa.Date(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=Text()), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("gate", "trading_day", name="gate_reading_pkey"),
        schema="crucible_state",
    )
    op.create_table(
        "journal_entry_kind",
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("kind", name="journal_entry_kind_pkey"),
        schema="crucible_state",
    )
    op.create_table(
        "ledger_account",
        sa.Column("account", sa.Text(), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "account ~ '^(assets|liabilities|equity|income|expenses)(:[A-Za-z0-9_./ -]+)+$'",
            name="ledger_account_path",
        ),
        sa.PrimaryKeyConstraint("account", name="ledger_account_pkey"),
        schema="crucible_state",
    )
    op.create_table(
        "morning_history",
        sa.Column("trading_day", sa.Date(), nullable=False),
        sa.Column("calendar_date", sa.Date(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=Text()), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("trading_day", "calendar_date", name="morning_history_pkey"),
        schema="crucible_state",
    )
    op.create_table(
        "nav_mark",
        sa.Column("trading_day", sa.Date(), nullable=False),
        sa.Column("calendar_date", sa.Date(), nullable=False),
        sa.Column("slot", sa.Text(), nullable=False),
        sa.Column("book", sa.Text(), nullable=False),
        sa.Column("fill_basis", sa.Text(), nullable=False),
        sa.Column("nav_usd", sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=Text()), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("trading_day", "slot", "book", name="nav_mark_pkey"),
        schema="crucible_state",
    )
    op.create_table(
        "position_snapshot",
        sa.Column("trading_day", sa.Date(), nullable=False),
        sa.Column("account", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("shares", sa.Integer(), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("source IN ('expected', 'broker')", name="position_snapshot_source"),
        sa.PrimaryKeyConstraint(
            "trading_day", "account", "source", "ticker", name="position_snapshot_pkey"
        ),
        schema="crucible_state",
    )
    op.create_table(
        "reconciliation",
        sa.Column("trading_day", sa.Date(), nullable=False),
        sa.Column("account", sa.Text(), nullable=False),
        sa.Column("genesis", sa.Boolean(), nullable=False),
        sa.Column("n_positions_expected", sa.Integer(), nullable=False),
        sa.Column("n_positions_broker", sa.Integer(), nullable=False),
        sa.Column("n_comparable", sa.Integer(), nullable=False),
        sa.Column("n_matched", sa.Integer(), nullable=False),
        sa.Column("expected_cash_usd", sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column("broker_cash_usd", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column("cash_residual_usd", sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column("clean", sa.Boolean(), nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=Text()), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("trading_day", "account", name="reconciliation_pkey"),
        schema="crucible_state",
    )
    op.create_table(
        "trade_order",
        sa.Column("order_key", sa.Text(), nullable=False),
        sa.Column("trading_day", sa.Date(), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("side", sa.Text(), nullable=False),
        sa.Column("quantity", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column("decision_price", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("order_ref", sa.Text(), nullable=False),
        sa.Column("account", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=Text()), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("side IN ('buy', 'sell')", name="trade_order_side"),
        sa.PrimaryKeyConstraint("order_key", name="trade_order_pkey"),
        sa.UniqueConstraint("order_ref", name="trade_order_order_ref_key"),
        sa.UniqueConstraint("trading_day", "symbol", name="trade_order_day_symbol_key"),
        schema="crucible_state",
    )
    op.create_table(
        "trial",
        sa.Column("slot", sa.Text(), nullable=False),
        sa.Column("arm_id", sa.Text(), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("control", sa.Boolean(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("benchmark", sa.Text(), nullable=False),
        sa.Column("n_dates_scored", sa.Integer(), nullable=False),
        sa.Column("first_date", sa.Date(), nullable=True),
        sa.Column("last_date", sa.Date(), nullable=True),
        sa.Column("mean_score_ratio", sa.Double(), nullable=True),
        sa.Column("lineage", postgresql.JSONB(astext_type=Text()), nullable=True),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("arena_cycle_key", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=Text()), nullable=False),
        sa.Column("written_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("n_dates_scored >= 0", name="trial_n_dates_scored_nonneg"),
        sa.PrimaryKeyConstraint("slot", "arm_id", "as_of", name="trial_pkey"),
        schema="crucible_state",
    )
    op.create_table(
        "journal_entry",
        sa.Column("entry_id", sa.Text(), nullable=False),
        sa.Column("effective_date", sa.Date(), nullable=False),
        sa.Column("posted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("reverses_entry_id", sa.Text(), nullable=True),
        sa.Column(
            "evidence",
            postgresql.JSONB(astext_type=Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=Text()), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("btrim(reason) <> ''", name="journal_entry_reason_nonempty"),
        sa.CheckConstraint("entry_id ~ '^je_[0-9a-f]{32}$'", name="journal_entry_id_shape"),
        sa.CheckConstraint(
            "reverses_entry_id IS NULL OR reverses_entry_id <> entry_id",
            name="journal_entry_not_self_reversing",
        ),
        sa.ForeignKeyConstraint(
            ["kind"], ["crucible_state.journal_entry_kind.kind"], name="journal_entry_kind_fkey"
        ),
        sa.ForeignKeyConstraint(
            ["reverses_entry_id"],
            ["crucible_state.journal_entry.entry_id"],
            name="journal_entry_reverses_fkey",
        ),
        sa.PrimaryKeyConstraint("entry_id", name="journal_entry_pkey"),
        schema="crucible_state",
    )
    op.create_table(
        "reconciliation_discrepancy",
        sa.Column("trading_day", sa.Date(), nullable=False),
        sa.Column("account", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("klass", sa.Text(), nullable=False),
        sa.Column("instrument", sa.Text(), nullable=False),
        sa.Column("expected", sa.Double(), nullable=True),
        sa.Column("actual", sa.Double(), nullable=True),
        sa.Column("delta", sa.Double(), nullable=True),
        sa.Column("side", sa.Text(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["trading_day", "account"],
            ["crucible_state.reconciliation.trading_day", "crucible_state.reconciliation.account"],
            name="reconciliation_discrepancy_reconciliation_fkey",
        ),
        sa.PrimaryKeyConstraint(
            "trading_day", "account", "ordinal", name="reconciliation_discrepancy_pkey"
        ),
        schema="crucible_state",
    )
    op.create_table(
        "trade_execution",
        sa.Column("order_key", sa.Text(), nullable=False),
        sa.Column("disposition", sa.Text(), nullable=False),
        sa.Column("broker_status", sa.Text(), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("filled_quantity", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column("fill_price", sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=Text()), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "disposition IN ('placed', 'adopted')", name="trade_execution_disposition"
        ),
        sa.CheckConstraint(
            "(filled_quantity > 0) = (fill_price IS NOT NULL)",
            name="trade_execution_price_iff_fill",
        ),
        sa.ForeignKeyConstraint(
            ["order_key"],
            ["crucible_state.trade_order.order_key"],
            name="trade_execution_order_fkey",
        ),
        sa.PrimaryKeyConstraint("order_key", "submitted_at", name="trade_execution_pkey"),
        schema="crucible_state",
    )
    op.create_table(
        "posting",
        sa.Column("entry_id", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("account", sa.Text(), nullable=False),
        sa.Column("amount", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column("currency", sa.Text(), nullable=False),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("currency ~ '^[A-Z]{3}$'", name="posting_currency_iso"),
        sa.CheckConstraint("amount <> 0", name="posting_amount_nonzero"),
        sa.ForeignKeyConstraint(
            ["account"], ["crucible_state.ledger_account.account"], name="posting_account_fkey"
        ),
        sa.ForeignKeyConstraint(
            ["entry_id"], ["crucible_state.journal_entry.entry_id"], name="posting_entry_fkey"
        ),
        sa.PrimaryKeyConstraint("entry_id", "ordinal", name="posting_pkey"),
        schema="crucible_state",
    )
    op.create_index(
        "trial_slot_control", "trial", ["slot", "control"], unique=False, schema="crucible_state"
    )
    op.create_index(
        "journal_entry_effective_date",
        "journal_entry",
        ["effective_date"],
        unique=False,
        schema="crucible_state",
    )
    op.create_index(
        "posting_account", "posting", ["account"], unique=False, schema="crucible_state"
    )
    op.bulk_insert(
        sa.table(
            "journal_entry_kind",
            sa.column("kind", sa.Text),
            sa.column("source", sa.Text),
            sa.column("description", sa.Text),
            schema=SCHEMA,
        ),
        [{"kind": k, "source": s, "description": d} for k, s, d in SEED_KINDS],
    )
    op.execute(REFUSE_MUTATION)
    op.execute(ASSERT_ENTRY_BALANCED)
    for table in TABLES:
        for statement in _append_only(table):
            op.execute(statement)
    op.execute(_balanced("journal_entry"))
    op.execute(_balanced("posting"))


def downgrade() -> None:
    op.execute(_refuse_if_any_rows())
    for table in reversed(TABLES):
        op.drop_table(table, schema=SCHEMA)
    op.execute(f"DROP FUNCTION {SCHEMA}.assert_entry_balanced()")
    op.execute(f"DROP FUNCTION {SCHEMA}.refuse_mutation()")
    op.execute(f"DROP SCHEMA {SCHEMA}")
