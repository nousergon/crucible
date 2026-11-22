"""Add the metadata-only LLM trace index.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-19

Prompt and response bodies remain in object storage.  ``content_s3_key`` is
nullable so ruling-C hash-only traces can be indexed without an object.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import Text
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "crucible_state"
TABLE = "llm_trace_index"


def upgrade() -> None:
    op.create_table(
        TABLE,
        sa.Column("trace_id", sa.Text(), nullable=False),
        sa.Column("span_id", sa.Text(), nullable=False),
        sa.Column("call_site", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("route", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column(
            "citations",
            postgresql.JSONB(astext_type=Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("input_content_hash", sa.Text(), nullable=False),
        sa.Column("output_content_hash", sa.Text(), nullable=False),
        sa.Column("content_s3_key", sa.Text(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "written_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("cost_usd >= 0", name="llm_trace_index_cost_nonneg"),
        sa.CheckConstraint("input_tokens >= 0", name="llm_trace_index_input_tokens_nonneg"),
        sa.CheckConstraint("output_tokens >= 0", name="llm_trace_index_output_tokens_nonneg"),
        sa.PrimaryKeyConstraint("trace_id", "span_id", name="llm_trace_index_pkey"),
        schema=SCHEMA,
    )
    for column in ("call_site", "model", "route", "input_content_hash", "output_content_hash"):
        suffix = {"input_content_hash": "input_hash", "output_content_hash": "output_hash"}.get(
            column, column
        )
        op.create_index(f"llm_trace_index_{suffix}", TABLE, [column], unique=False, schema=SCHEMA)
    op.execute(
        f"CREATE TRIGGER {TABLE}_append_only BEFORE UPDATE OR DELETE ON {SCHEMA}.{TABLE} "
        f"FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.refuse_mutation()"
    )
    op.execute(
        f"CREATE TRIGGER {TABLE}_no_truncate BEFORE TRUNCATE ON {SCHEMA}.{TABLE} "
        f"FOR EACH STATEMENT EXECUTE FUNCTION {SCHEMA}.refuse_mutation()"
    )


def downgrade() -> None:
    op.execute(
        "DO $$ BEGIN "
        f"IF EXISTS (SELECT 1 FROM {SCHEMA}.{TABLE}) THEN "
        f"RAISE EXCEPTION '{SCHEMA}.{TABLE} holds rows; downgrade of 0002 refused "
        "(dropping indexed traces is an irreversible delete)' "
        "USING ERRCODE = 'restrict_violation'; "
        "END IF; END $$"
    )
    op.drop_table(TABLE, schema=SCHEMA)
