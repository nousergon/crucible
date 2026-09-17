"""`crucible acceptance.publish` — the acceptance reading's registered job.

Normative source: `alpha-engine-config-I10968`, the second of the two store
writers that sat outside `crucible.cli.JOBS`.

`.github/workflows/ci.yml`'s acceptance job computed the plan §12 rule-3
progress figure with `tests/acceptance/check_reading.py --write-json` and then
shipped the resulting file with a raw `aws s3 cp` — not through
`crucible.store.Store` at all, so the ONE producer of the figure the plan
names as progress wrote no run manifest: no lineage, no spend, no code sha,
and nothing that said it had run. `crucible report.morning`, `crucible board`
and `crucible.gate`'s phase-0 clause all read that artifact as evidence.

This module is the job that replaces the copy. It deliberately does not
re-implement the grader: `check_reading.py` still computes and still writes
`acceptance-reading.json`, and this job PUBLISHES the file it wrote, through
`crucible.runner.run_job`, at `crucible.keys.acceptance_reading_key(day)`.
The split is the same one `crucible.integration_summary` makes with
`pytest tests/integration` — the harness does not grade, it records.

**Why the file rather than a recomputation.** `ci.yml` grades the reading and
publishes it whether or not it moved (`continue-on-error` on the check step,
deliberately — a red count is rule 3's whole point). Recomputing here would
make the published document a SECOND measurement of a tree the first one
already graded, and the two could disagree while both were honest. One
measurement, published.

**Validated against the CONSUMER's parser.** `crucible.keys.
parse_acceptance_reading` is what `crucible.morning` and `crucible.board`
read the document with, so a file that parser would reject is refused HERE,
at the producer, naming the file — rather than published and rendered as
"no acceptance reading" on two surfaces that cannot say why.

**No schedule, no deadline.** The row declares both `null`: this job is
triggered by `ci.yml`'s own `push`/`pull_request` triggers, never a clock. A
deadline would page for a Saturday on which nobody merged anything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from crucible.keys import acceptance_reading_key, parse_acceptance_reading
from crucible.runner import RunContext

__all__ = [
    "ACCEPTANCE_PUBLISH_JOB",
    "ACCEPTANCE_READING_SCHEMA_VERSION",
    "acceptance_publish_body",
    "acceptance_publish_handler",
]

#: The job name registered in `crucible.cli.JOBS`, `crucible/components.yaml`
#: and `crucible.models.JOB_VALUES` — a constant imported by `cli.py` rather
#: than a literal restated at each of those sites, the same shape
#: `crucible.integration_summary.INTEGRATION_TEST_JOB` uses.
ACCEPTANCE_PUBLISH_JOB = "acceptance.publish"

#: The `schema_version` this job's `outputs[]` row declares for the published
#: document. The artifact itself carries no version field (its shape is
#: `crucible.keys.acceptance_reading_key`'s documented producer contract, read
#: by `crucible.keys.parse_acceptance_reading`), so the manifest is where the
#: version is stated.
ACCEPTANCE_READING_SCHEMA_VERSION = "acceptance_reading.v1"


class AcceptanceReadingUnpublishable(RuntimeError):
    """The local reading cannot be published as written.

    Fatal, never degraded (`AGENTS.md` rule 5): "the reading was published"
    and "a file that is not a reading was published" are different answers,
    and a surface reading the second cannot tell it from the first.
    """


def acceptance_publish_body(ctx: RunContext, *, reading_path: Path) -> None:
    """Publish ``reading_path`` at this trading day's acceptance reading key.

    Separated from the handler so the whole body is exercisable against a
    :class:`~crucible.runner.RunContext` without a CLI or a store URI — the
    same split `crucible.integration_summary.integration_test_body` uses.
    """
    if not reading_path.is_file():
        raise AcceptanceReadingUnpublishable(
            f"{reading_path} does not exist. It is written by "
            "`tests/acceptance/check_reading.py --write-json`, before any of that "
            "script's own fail branches can return early, so an absent file means the "
            "grader never ran — not that the reading is unchanged."
        )
    payload = reading_path.read_bytes()
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AcceptanceReadingUnpublishable(
            f"{reading_path} is not JSON ({exc}); publishing it would put a document on "
            "the one progress figure's key that every consumer reads as absent"
        ) from exc
    reading = parse_acceptance_reading(document)
    if reading is None:
        raise AcceptanceReadingUnpublishable(
            f"{reading_path} is not an acceptance reading `crucible.keys."
            "parse_acceptance_reading` accepts — the reader `crucible.morning` and "
            "`crucible.board` both use. Publishing it would render as 'no acceptance "
            "reading' on both surfaces, with nothing anywhere saying why."
        )
    ctx.record_output(
        acceptance_reading_key(ctx.trading_day.isoformat()),
        payload,
        schema_version=ACCEPTANCE_READING_SCHEMA_VERSION,
    )
    ctx.record_rows(rows_in=1, rows_out=1)
    total = reading.met + reading.unmet
    ctx.record_metric(
        {
            "name": "acceptance_clauses_met",
            "module": "crucible.acceptance_publish",
            "metric_type": "coverage",
            "value": float(reading.met),
            "unit": "clauses",
            "n_floor": 1,
            # Never `FAIL` on a red count. The §2 suite is red BY DESIGN until
            # the phases land (plan §6), so grading the count here would page
            # every merge for months and train the operator to ignore the one
            # figure §12 rule 3 names as progress. What this job can fail on
            # is publishing, and that is the manifest's own `status`.
            "status": "OK",
            "status_reason": (
                f"{reading.met} of {total} plan §2 clauses met at commit "
                f"{reading.commit} ({reading.unmeasurable} of the unmet could not be "
                "read at all)"
            ),
            "source_path": "tests/acceptance/ratchet.json",
            "last_updated_utc": ctx.started.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )


def acceptance_publish_handler(args: argparse.Namespace) -> int:
    """`crucible acceptance.publish --reading PATH [--store URI]`."""
    from crucible.cli import _resolve_store
    from crucible.runner import run_job

    store = _resolve_store(args)

    def job(ctx: RunContext) -> None:
        acceptance_publish_body(ctx, reading_path=Path(args.reading))

    ctx = run_job(
        ACCEPTANCE_PUBLISH_JOB,
        job,
        store=store,
        trading_day=args.trading_day,
        dry_run=bool(getattr(args, "dry_run", False)),
        run_mode=getattr(args, "run_mode", None),
    )
    print(json.dumps({"run_id": ctx.run_id, "job": ctx.job}, indent=2))
    return 0
