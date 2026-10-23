"""Schema v1 of crucible-state, declared once as SQLAlchemy Core tables.

alpha-engine-config-I12510 (plan S-22). The design and the shape check against
the current JSON documents are in that issue's 2026-10-22 comments. This module
is the declaration. Alembic revision ``0001`` must render to exactly these
tables (``tests/test_state_schema.py``).

Conventions, applied to every table:

- **One schema, ``crucible_state``.**
- **``payload jsonb NOT NULL`` holds the source document unchanged.** The
  extracted columns exist for identity and query only. The S-24 parity check
  diffs JSON, not a lossy projection, and a new JSON field needs no migration.
- **Money and quantities are ``numeric(20, 6)``, never float.** Fractional
  ``filled_quantity`` exists. Ratios and scores stay ``double precision``.
- **Timestamps are ``timestamptz``.** ``trading_day`` is the §4.12 key.
  ``calendar_date`` is kept where the document carries one.
- **Append-only.** Revision ``0001`` installs a trigger on every table that
  refuses UPDATE, DELETE and TRUNCATE. A correction is a new row, or for
  money a new journal entry.

Two gaps from the shape check (10-21 S7, settled here):

1. **The trial key is ``as_of``, not ``cycle_date``.** The issue text says
   ``(slot, arm, cycle_date)``. The ledger's identity is
   ``crucible.ledger._identity``, which is ``(slot, arm_id, as_of)``. ``trial``
   keys on that, under the code's own name, so a dual-written row and its
   JSONL row have one identity.
2. **A broker fill has no natural key without IB's ``execId``.**
   ``crucible_trader.reconciliation.Fill`` carries ticker, shares and cash
   only. Two identical partial fills would collide on any key built from those
   fields. ``fill`` is keyed on ``(account, broker_exec_id)``, and
   ``broker_exec_id`` is NOT NULL and non-empty. The SDK's unattached default
   ``CommissionReport()`` carries ``execId == ""``
   (``crucible_trader/reconcile_entrypoint.py``), so an empty id is refused
   rather than stored. S-23 must carry ``execId`` before ``fill`` can be
   dual-written.

The money ledger (``ledger_account``, ``journal_entry_kind``,
``journal_entry``, ``posting``) follows Brian's 2026-10-22 ruling that money is
a double-entry ledger that always balances. It matches the producer already
posting in that shape (crucible-executor ``executor/money_ledger.py``,
``money_ledger.v1``), so its S3 copies backfill without transformation.
Balance is enforced in the database: a deferred constraint trigger refuses, at
commit, any entry whose postings do not sum to zero per currency or that has
fewer than two postings. Accounting scope beyond that (accounts seeding, FX,
lots, corporate actions) is aec#12558, and it reviews these four tables before
anything is applied.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

__all__ = [
    "ACCOUNT_PATTERN",
    "APPEND_ONLY_TABLES",
    "ENTRY_ID_PATTERN",
    "SCHEMA",
    "metadata",
]

SCHEMA = "crucible_state"

#: crucible-executor ``money_ledger._ACCOUNT_RE``, as a Postgres regex.
ACCOUNT_PATTERN = r"^(assets|liabilities|equity|income|expenses)(:[A-Za-z0-9_./ -]+)+$"
#: ``je_`` plus the first 32 hex digits of the entry's content sha256.
ENTRY_ID_PATTERN = r"^je_[0-9a-f]{32}$"

metadata = sa.MetaData(schema=SCHEMA)

MONEY = sa.Numeric(20, 6)
QUANTITY = sa.Numeric(20, 6)
TZ = sa.DateTime(timezone=True)


def _payload() -> sa.Column:
    return sa.Column("payload", JSONB, nullable=False)


def _schema_version() -> sa.Column:
    return sa.Column("schema_version", sa.Text, nullable=False)


def _written_at() -> sa.Column:
    return sa.Column("written_at", TZ, nullable=False, server_default=sa.text("now()"))


# --------------------------------------------------------------------------
# Trial ledger: crucible/ledger.py, trial.v1
# --------------------------------------------------------------------------

trial = sa.Table(
    "trial",
    metadata,
    sa.Column("slot", sa.Text, nullable=False),
    sa.Column("arm_id", sa.Text, nullable=False),
    sa.Column("as_of", sa.Date, nullable=False),
    sa.Column("control", sa.Boolean, nullable=False),
    sa.Column("active", sa.Boolean, nullable=False),
    sa.Column("benchmark", sa.Text, nullable=False),
    sa.Column("n_dates_scored", sa.Integer, nullable=False),
    sa.Column("first_date", sa.Date),
    sa.Column("last_date", sa.Date),
    sa.Column("mean_score_ratio", sa.Double),
    # NULL is not '{}': a pre-lineage row carries no key at all (ledger.py).
    sa.Column("lineage", JSONB),
    sa.Column("run_id", sa.Text, nullable=False),
    sa.Column("arena_cycle_key", sa.Text, nullable=False),
    _schema_version(),
    _payload(),
    # From the row's own written_at_utc, never now(): a backfill keeps history.
    sa.Column("written_at", TZ, nullable=False),
    sa.PrimaryKeyConstraint("slot", "arm_id", "as_of", name="trial_pkey"),
    sa.CheckConstraint("n_dates_scored >= 0", name="trial_n_dates_scored_nonneg"),
    sa.Index("trial_slot_control", "slot", "control"),
)


# --------------------------------------------------------------------------
# Trader ledger: crucible-trader
# --------------------------------------------------------------------------

trade_order = sa.Table(
    "trade_order",
    metadata,
    sa.Column("order_key", sa.Text, nullable=False),
    sa.Column("trading_day", sa.Date, nullable=False),
    sa.Column("symbol", sa.Text, nullable=False),
    sa.Column("side", sa.Text, nullable=False),
    sa.Column("quantity", QUANTITY, nullable=False),
    sa.Column("decision_price", MONEY, nullable=False),
    sa.Column("decided_at", TZ, nullable=False),
    sa.Column("order_ref", sa.Text, nullable=False),
    sa.Column("account", sa.Text, nullable=False),
    _payload(),
    _written_at(),
    sa.PrimaryKeyConstraint("order_key", name="trade_order_pkey"),
    sa.UniqueConstraint("order_ref", name="trade_order_order_ref_key"),
    sa.UniqueConstraint("trading_day", "symbol", name="trade_order_day_symbol_key"),
    sa.CheckConstraint("side IN ('buy', 'sell')", name="trade_order_side"),
)

trade_execution = sa.Table(
    "trade_execution",
    metadata,
    sa.Column("order_key", sa.Text, nullable=False),
    sa.Column("disposition", sa.Text, nullable=False),
    sa.Column("broker_status", sa.Text, nullable=False),
    sa.Column("submitted_at", TZ, nullable=False),
    sa.Column("filled_quantity", QUANTITY, nullable=False),
    sa.Column("fill_price", MONEY),
    _payload(),
    _written_at(),
    sa.PrimaryKeyConstraint("order_key", "submitted_at", name="trade_execution_pkey"),
    sa.ForeignKeyConstraint(
        ["order_key"], [f"{SCHEMA}.trade_order.order_key"], name="trade_execution_order_fkey"
    ),
    sa.CheckConstraint("disposition IN ('placed', 'adopted')", name="trade_execution_disposition"),
    # order_router.Fill: a price is present iff something filled.
    sa.CheckConstraint(
        "(filled_quantity > 0) = (fill_price IS NOT NULL)", name="trade_execution_price_iff_fill"
    ),
)

fill = sa.Table(
    "fill",
    metadata,
    sa.Column("trading_day", sa.Date, nullable=False),
    sa.Column("account", sa.Text, nullable=False),
    sa.Column("ticker", sa.Text, nullable=False),
    sa.Column("shares", sa.Integer, nullable=False),
    sa.Column("cash_usd", MONEY, nullable=False),
    # Gap 2 (module docstring): IB execId, never derived from the fill's fields.
    sa.Column("broker_exec_id", sa.Text, nullable=False),
    _payload(),
    _written_at(),
    sa.PrimaryKeyConstraint("account", "broker_exec_id", name="fill_pkey"),
    sa.CheckConstraint("broker_exec_id <> ''", name="fill_broker_exec_id_nonempty"),
)

position_snapshot = sa.Table(
    "position_snapshot",
    metadata,
    sa.Column("trading_day", sa.Date, nullable=False),
    sa.Column("account", sa.Text, nullable=False),
    sa.Column("source", sa.Text, nullable=False),
    sa.Column("ticker", sa.Text, nullable=False),
    sa.Column("shares", sa.Integer, nullable=False),
    _written_at(),
    sa.PrimaryKeyConstraint(
        "trading_day", "account", "source", "ticker", name="position_snapshot_pkey"
    ),
    sa.CheckConstraint("source IN ('expected', 'broker')", name="position_snapshot_source"),
)

nav_mark = sa.Table(
    "nav_mark",
    metadata,
    sa.Column("trading_day", sa.Date, nullable=False),
    sa.Column("calendar_date", sa.Date, nullable=False),
    sa.Column("slot", sa.Text, nullable=False),
    sa.Column("book", sa.Text, nullable=False),
    sa.Column("fill_basis", sa.Text, nullable=False),
    # Nullable until shadow_books.v1 pins its books[] item shape.
    sa.Column("nav_usd", MONEY),
    _payload(),
    _written_at(),
    sa.PrimaryKeyConstraint("trading_day", "slot", "book", name="nav_mark_pkey"),
)

reconciliation = sa.Table(
    "reconciliation",
    metadata,
    sa.Column("trading_day", sa.Date, nullable=False),
    sa.Column("account", sa.Text, nullable=False),
    sa.Column("genesis", sa.Boolean, nullable=False),
    sa.Column("n_positions_expected", sa.Integer, nullable=False),
    sa.Column("n_positions_broker", sa.Integer, nullable=False),
    sa.Column("n_comparable", sa.Integer, nullable=False),
    sa.Column("n_matched", sa.Integer, nullable=False),
    sa.Column("expected_cash_usd", MONEY),
    sa.Column("broker_cash_usd", MONEY, nullable=False),
    sa.Column("cash_residual_usd", MONEY),
    sa.Column("clean", sa.Boolean, nullable=False),
    _schema_version(),
    _payload(),
    _written_at(),
    sa.PrimaryKeyConstraint("trading_day", "account", name="reconciliation_pkey"),
)

reconciliation_discrepancy = sa.Table(
    "reconciliation_discrepancy",
    metadata,
    sa.Column("trading_day", sa.Date, nullable=False),
    sa.Column("account", sa.Text, nullable=False),
    sa.Column("ordinal", sa.Integer, nullable=False),
    sa.Column("klass", sa.Text, nullable=False),
    sa.Column("instrument", sa.Text, nullable=False),
    sa.Column("expected", sa.Double),
    sa.Column("actual", sa.Double),
    sa.Column("delta", sa.Double),
    sa.Column("side", sa.Text, nullable=False),
    sa.Column("detail", sa.Text, nullable=False),
    _written_at(),
    sa.PrimaryKeyConstraint(
        "trading_day", "account", "ordinal", name="reconciliation_discrepancy_pkey"
    ),
    sa.ForeignKeyConstraint(
        ["trading_day", "account"],
        [f"{SCHEMA}.reconciliation.trading_day", f"{SCHEMA}.reconciliation.account"],
        name="reconciliation_discrepancy_reconciliation_fkey",
    ),
)

execution_shortfall = sa.Table(
    "execution_shortfall",
    metadata,
    sa.Column("trading_day", sa.Date, nullable=False),
    sa.Column("calendar_date", sa.Date, nullable=False),
    sa.Column("champion", sa.Text, nullable=False),
    sa.Column("outcome", sa.Text, nullable=False),
    sa.Column("outcome_reason", sa.Text),
    sa.Column("n_orders", sa.Integer),
    sa.Column("n_filled", sa.Integer),
    sa.Column("notional_usd", MONEY),
    sa.Column("shortfall_usd", MONEY),
    sa.Column("shortfall_bps_notional_weighted", sa.Double),
    _schema_version(),
    _payload(),
    _written_at(),
    sa.PrimaryKeyConstraint("trading_day", "champion", name="execution_shortfall_pkey"),
)


# --------------------------------------------------------------------------
# Money ledger: double entry, balanced at commit (Brian 2026-10-22; aec#12558)
# --------------------------------------------------------------------------

ledger_account = sa.Table(
    "ledger_account",
    metadata,
    sa.Column("account", sa.Text, nullable=False),
    _written_at(),
    sa.PrimaryKeyConstraint("account", name="ledger_account_pkey"),
    sa.CheckConstraint(f"account ~ '{ACCOUNT_PATTERN}'", name="ledger_account_path"),
)

journal_entry_kind = sa.Table(
    "journal_entry_kind",
    metadata,
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("source", sa.Text, nullable=False),
    sa.Column("description", sa.Text, nullable=False),
    _written_at(),
    sa.PrimaryKeyConstraint("kind", name="journal_entry_kind_pkey"),
)

journal_entry = sa.Table(
    "journal_entry",
    metadata,
    sa.Column("entry_id", sa.Text, nullable=False),
    sa.Column("effective_date", sa.Date, nullable=False),
    sa.Column("posted_at", TZ, nullable=False),
    sa.Column("source", sa.Text, nullable=False),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("reason", sa.Text, nullable=False),
    sa.Column("reverses_entry_id", sa.Text),
    sa.Column("evidence", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    _schema_version(),
    _payload(),
    _written_at(),
    sa.PrimaryKeyConstraint("entry_id", name="journal_entry_pkey"),
    sa.ForeignKeyConstraint(
        ["kind"], [f"{SCHEMA}.journal_entry_kind.kind"], name="journal_entry_kind_fkey"
    ),
    sa.ForeignKeyConstraint(
        ["reverses_entry_id"],
        [f"{SCHEMA}.journal_entry.entry_id"],
        name="journal_entry_reverses_fkey",
    ),
    sa.CheckConstraint(f"entry_id ~ '{ENTRY_ID_PATTERN}'", name="journal_entry_id_shape"),
    sa.CheckConstraint("btrim(reason) <> ''", name="journal_entry_reason_nonempty"),
    sa.CheckConstraint(
        "reverses_entry_id IS NULL OR reverses_entry_id <> entry_id",
        name="journal_entry_not_self_reversing",
    ),
    sa.Index("journal_entry_effective_date", "effective_date"),
)

posting = sa.Table(
    "posting",
    metadata,
    sa.Column("entry_id", sa.Text, nullable=False),
    sa.Column("ordinal", sa.Integer, nullable=False),
    sa.Column("account", sa.Text, nullable=False),
    sa.Column("amount", MONEY, nullable=False),
    sa.Column("currency", sa.Text, nullable=False),
    _written_at(),
    sa.PrimaryKeyConstraint("entry_id", "ordinal", name="posting_pkey"),
    sa.ForeignKeyConstraint(
        ["entry_id"], [f"{SCHEMA}.journal_entry.entry_id"], name="posting_entry_fkey"
    ),
    sa.ForeignKeyConstraint(
        ["account"], [f"{SCHEMA}.ledger_account.account"], name="posting_account_fkey"
    ),
    # A zero line is not a posting (money_ledger.Posting refuses it too).
    sa.CheckConstraint("amount <> 0", name="posting_amount_nonzero"),
    sa.CheckConstraint("currency ~ '^[A-Z]{3}$'", name="posting_currency_iso"),
    sa.Index("posting_account", "account"),
)


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------

arena_cycle = sa.Table(
    "arena_cycle",
    metadata,
    sa.Column("slot", sa.Text, nullable=False),
    sa.Column("trading_day", sa.Date, nullable=False),
    sa.Column("s3_key", sa.Text, nullable=False),
    _schema_version(),
    _payload(),
    _written_at(),
    sa.PrimaryKeyConstraint("slot", "trading_day", name="arena_cycle_pkey"),
)

champion_history = sa.Table(
    "champion_history",
    metadata,
    sa.Column("slot", sa.Text, nullable=False),
    sa.Column("decided_at", TZ, nullable=False),
    sa.Column("arm_id", sa.Text, nullable=False),
    sa.Column("as_of", sa.Date, nullable=False),
    sa.Column("run_id", sa.Text, nullable=False),
    sa.Column("code_sha", sa.Text, nullable=False),
    sa.Column("promotion_source", sa.Text, nullable=False),
    sa.Column("manifest_key", sa.Text, nullable=False),
    _schema_version(),
    _payload(),
    _written_at(),
    # The POINTER itself stays on S3 (plan §5); this is its decision history.
    sa.PrimaryKeyConstraint("slot", "decided_at", name="champion_history_pkey"),
)

# gate_reading and morning_history have no pinned JSON schema yet, so they are
# payload-only in v1. Key columns are extracted once a schema exists.
gate_reading = sa.Table(
    "gate_reading",
    metadata,
    sa.Column("gate", sa.Text, nullable=False),
    sa.Column("trading_day", sa.Date, nullable=False),
    _payload(),
    _written_at(),
    sa.PrimaryKeyConstraint("gate", "trading_day", name="gate_reading_pkey"),
)

morning_history = sa.Table(
    "morning_history",
    metadata,
    sa.Column("trading_day", sa.Date, nullable=False),
    sa.Column("calendar_date", sa.Date, nullable=False),
    _payload(),
    _written_at(),
    sa.PrimaryKeyConstraint("trading_day", "calendar_date", name="morning_history_pkey"),
)

#: Every table: 0001 puts the refuse-mutation trigger on each one.
APPEND_ONLY_TABLES: tuple[str, ...] = tuple(t.name for t in metadata.sorted_tables)
