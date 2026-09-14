"""One real case per exercisable CLI job (24 of 25 — see README.md; `weekly`,
`experiment.grade` and `promote` added by `alpha-engine-config-I10633`;
`report.morning` added by `alpha-engine-config-I10458`, against a dedicated
tracker repo and a non-notifying Telegram destination, never Brian's real
operator chat or the private production tracker).
`test.integration` is the 25th job README.md's own count includes — it is
exercised by `.github/workflows/integration-nightly.yml` invoking this
whole suite, not by a case within it.

Every case invokes `crucible.cli.main`, the real process entry point, not a
handler function directly — the wire a spot instance actually dispatches
through. Every case reads back the real `run.json` this produced, through
`crucible.manifest.read_manifest`, the same reader every other consumer of a
manifest uses — never `store.get_bytes` + `json.loads` (AGENTS.md rule 1: a
manifest prefix is a namespace, not a manifest list; this module reads one
key at a time by its own constructed key, so `is_manifest_key` filtering does
not apply, but the schema-validating reader still does).

`crucible gate` and `crucible gate.close` exit non-zero on a measurement that
succeeded but read UNMET/nothing-due (`track_f.gate_handler`'s own
docstring: "the job succeeds when the MEASUREMENT succeeds; the PROCESS
exits non-zero when the gate is not met") — this module asserts the
MANIFEST's status, never the process exit code, for exactly that reason.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any

import pytest

from crucible.calendar import assert_trading_day
from crucible.cli import main as cli_main
from crucible.documents import read_manifests_under
from crucible.keys import manifest_prefix
from crucible.manifest import read_manifest
from crucible.store import Store
from tests.integration.conftest import INTEGRATION_TRADING_DAY, SETTLED_TRADING_DAY

pytestmark = pytest.mark.usefixtures("_dedicated_topic_env")


def _manifest(
    store: Store,
    job: str,
    *,
    discriminator: str | None = None,
    trading_day: str = INTEGRATION_TRADING_DAY,
) -> dict[str, Any]:
    return read_manifest(store, job, trading_day, discriminator=discriminator)


def _assert_ok(
    store: Store,
    job: str,
    *,
    discriminator: str | None = None,
    trading_day: str = INTEGRATION_TRADING_DAY,
) -> dict[str, Any]:
    # `trading_day` (`alpha-engine-config-I10633`): every case before
    # `test_experiment_grade` keys its manifest at `INTEGRATION_TRADING_DAY`
    # (the default, unchanged) — `experiment.grade`/`promote` and the second
    # `data.daily` run above key theirs at `SETTLED_TRADING_DAY` instead, and
    # a helper that stayed hardcoded to the one module constant would read
    # back the WRONG manifest (or none at all) for every one of them.
    manifest = _manifest(store, job, discriminator=discriminator, trading_day=trading_day)
    assert manifest["status"] == "ok", (
        f"{job}: expected status ok against the dedicated store, got "
        f"{manifest['status']!r}: {manifest.get('reason')}"
    )
    return manifest


def _assert_latest_ok(
    store: Store,
    job: str,
    *,
    trading_day: str = INTEGRATION_TRADING_DAY,
) -> dict[str, Any]:
    """`_assert_ok` for a job whose discriminator is its own CALENDAR date.

    `alerts.sweep` (`track_c.sweep_handler`) and `report.morning`
    (`morning_handler`) both pass
    `discriminator=lambda ctx: ctx.calendar_date.isoformat()` to `run_job`, so
    a job firing more than once between two trading-day rollovers cannot
    overwrite its own record. Both cases read the UNDISCRIMINATED key and
    failed with `KeyError: 'runs/<job>/2026-09-08/run.json' is not present`
    (`alpha-engine-config-I10705`) — the same defect `test_release_lock`
    carried, in the same shape, and both hidden until this session behind an
    earlier failure in the same case.

    Read through the prefix rather than by reconstructing the discriminator:
    `calendar_date` is the run's own UTC wall-clock date, and a test that
    recomputed it would be asserting against the clock across a midnight
    boundary. `read_manifests_under` is the sanctioned reader (AGENTS.md rule
    1) and preserves sorted listing order, so the LAST document under the
    day's prefix is this run's.
    """
    prefix = manifest_prefix(job, trading_day)
    read = read_manifests_under(store, prefix)
    read.raise_if_unlistable()
    assert read.documents, f"{job} wrote no manifest under {prefix!r}"
    _key, manifest = read.documents[-1]
    assert manifest["status"] == "ok", (
        f"{job}: expected status ok against the dedicated store, got "
        f"{manifest['status']!r}: {manifest.get('reason')}"
    )
    return manifest


# ── experiment.new — the one arena-slot job this tier can exercise for real
#    (see README.md for why experiment.run/grade/promote cannot yet) ───────


def test_experiment_new(integration_store_uri: str, integration_store: Store, strategy_dir) -> None:
    cli_main(
        [
            "experiment.new",
            "--slot",
            "u",
            # `--arm` is required for `experiment.new` specifically
            # (`crucible/cli.py`: `required=spec.name == "experiment.new"`) —
            # the argv this test carried before named no arm and failed with
            # `SystemExit: 2` ("the following arguments are required: --arm"),
            # measured against run 34797033396 (`alpha-engine-config-I10701`).
            # `momentum_sleeve` is one of `strategy_dir`'s three fixture
            # recipes (see that fixture's own docstring).
            "--arm",
            "momentum_sleeve",
            "--store",
            integration_store_uri,
            "--strategy-dir",
            str(strategy_dir),
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    manifest = _assert_ok(integration_store, "experiment.new")
    assert manifest["outputs"], "experiment.new wrote no outputs — the arm register never landed"


# ── data.daily / data.weekly / data.heal — the dedicated-library override
#    (`alpha-engine-config-I10457`) unblocks all three: each reads real
#    ArcticDB rows through `ArcticPriceSource(library=...)`, never the
#    production `universe` library ──────────────────────────────────────────


def test_data_daily(
    integration_store_uri: str,
    integration_store: Store,
    integration_arctic_library: str,
    integration_arctic_symbols: list[str],
) -> None:
    cli_main(
        [
            "data.daily",
            "--store",
            integration_store_uri,
            "--arctic-library",
            integration_arctic_library,
            "--symbols",
            ",".join(integration_arctic_symbols),
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    manifest = _assert_ok(integration_store, "data.daily")
    assert manifest["outputs"], "data.daily wrote no outputs — the feature layer never landed"


def test_data_weekly(
    integration_store_uri: str,
    integration_store: Store,
    integration_arctic_library: str,
    integration_arctic_symbols: list[str],
    compiled_week_panels: list[str],
) -> None:
    """`compiled_week_panels` (`alpha-engine-config-I10705`) is the whole
    difference between this case and a `DataGapError`: `run_weekly` compiles
    only its own session and requires the week's other sessions to already
    carry a panel, exactly as production's per-weekday `data.daily` schedule
    supplies them. The fixture runs that real job for each of them."""
    assert compiled_week_panels, (
        "the week backfill compiled nothing — `week_sessions` named no session besides "
        f"{INTEGRATION_TRADING_DAY}, so this case would pass without exercising the "
        "week-gap invariant at all"
    )
    cli_main(
        [
            "data.weekly",
            "--store",
            integration_store_uri,
            "--arctic-library",
            integration_arctic_library,
            "--symbols",
            ",".join(integration_arctic_symbols),
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    _assert_ok(integration_store, "data.weekly")


def test_data_heal(
    integration_store_uri: str,
    integration_store: Store,
    integration_arctic_library: str,
    integration_arctic_symbols: list[str],
) -> None:
    """A 1-session range — well inside `LAPTOP_SESSION_ALLOWANCE` (3), so this
    runs on a GitHub-hosted (non-EC2) runner with no `--i-am-in-region`."""
    cli_main(
        [
            "data.heal",
            "--gap",
            "integration-tier-probe",
            "--from",
            INTEGRATION_TRADING_DAY,
            "--to",
            INTEGRATION_TRADING_DAY,
            "--store",
            integration_store_uri,
            "--arctic-library",
            integration_arctic_library,
            "--symbols",
            ",".join(integration_arctic_symbols),
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    _assert_ok(integration_store, "data.heal")


# ── experiment.run — never touches ArcticDB directly
#    (`crucible.slots.cycle.run_produce` reads the feature layer `data.daily`
#    above just wrote into THIS store); its non-degenerate exercise was
#    blocked only by `data.daily` having nothing real to read
#    (`alpha-engine-config-I10457`), never by a gap of its own. This shadow
#    is what `test_experiment_grade` below settles, against a second, LATER
#    trading day. ─────────────────────────────────────────────────────────


def test_experiment_run(integration_store_uri: str, integration_store: Store) -> None:
    cli_main(
        [
            "experiment.run",
            "--slot",
            "u",
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    manifest = _assert_ok(integration_store, "experiment.run", discriminator="u")
    assert manifest["outputs"], "experiment.run wrote no outputs — no arm's shadow landed"


# ── weekly — the arc's own `data.weekly` stage against the dedicated
#    ArcticDB library (`alpha-engine-config-I10633`) ────────────────────────


def test_weekly(
    integration_store_uri: str,
    integration_store: Store,
    integration_arctic_library: str,
    integration_arctic_symbols: list[str],
    compiled_week_panels: list[str],
) -> None:
    """`crucible.weekly.Stage.argv`/`run_arc` now thread `--arctic-library`
    onto every `ARCTIC_LIBRARY_JOBS` stage. Before this, an arc run here
    would have had its `data.weekly` stage silently read the PRODUCTION
    `universe` library within the shared integration bucket rather than the
    dedicated one — the opposite of this tier's isolation guarantee (see
    README.md, "Not exercised, and why").

    Only `data.weekly` is invoked through the real `crucible.cli.main` wire
    here; every other arc stage is stubbed. The R slot's own arm
    registration (`alpha-engine-config-I10628`, a sibling track's own
    scope) and the arc's remaining stages are exercised standalone, for
    real, by this module's other cases — chaining them all through one real
    `weekly` invocation would entangle this issue's own deliverable
    (`--arctic-library` threading) with that unrelated, unowned surface.
    `CRUCIBLE_UNIVERSE_URI` (`conftest.py::_declared_universe_env`) is what
    lets the real `data.weekly` stage resolve a universe at all: `Stage.argv`
    carries no `--symbols` for any stage, by design
    (`crucible/data/universe.py`'s own docstring).
    """
    from crucible.weekly import run_arc

    trading_day = dt.date.fromisoformat(INTEGRATION_TRADING_DAY)

    def main(argv: list[str]) -> int:
        if argv[0] == "data.weekly":
            return cli_main(argv)
        return 0

    ran = run_arc(
        trading_day,
        store=integration_store_uri,
        run_mode="live",
        main=main,
        arctic_library=integration_arctic_library,
    )
    assert any(s.job == "data.weekly" for s in ran), "the arc did not run a data.weekly stage"
    _assert_ok(integration_store, "data.weekly")


# ── experiment.grade / promote — a second, LATER seeded trading day
#    (`alpha-engine-config-I10633`) settles the shadow `test_experiment_run`
#    above produced at `INTEGRATION_TRADING_DAY`, exactly
#    `DEFAULT_HORIZON_TRADING_DAYS` sessions earlier
#    (`conftest.py::SETTLED_TRADING_DAY`) ─────────────────────────────────


def test_experiment_grade(
    integration_store_uri: str,
    integration_store: Store,
    integration_arctic_library: str,
    integration_arctic_symbols: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The panel `experiment.grade` reads is keyed by ITS OWN `--date`
    # (`crucible.slots.cycle._read_panel`), never the shadow's produce date —
    # so grading against `SETTLED_TRADING_DAY` needs a panel compiled AT
    # `SETTLED_TRADING_DAY`, covering both it and `INTEGRATION_TRADING_DAY`
    # inside `data.daily`'s trailing lookback window.
    cli_main(
        [
            "data.daily",
            "--store",
            integration_store_uri,
            "--arctic-library",
            integration_arctic_library,
            "--symbols",
            ",".join(integration_arctic_symbols),
            "--run-mode",
            "live",
            "--date",
            SETTLED_TRADING_DAY,
        ]
    )
    _assert_ok(integration_store, "data.daily", trading_day=SETTLED_TRADING_DAY)
    # DRAIN the capture before the invocation whose stdout is parsed below
    # (`alpha-engine-config-I10705`). `capsys` accumulates across both
    # `cli_main` calls in this case, and every handler prints a JSON document,
    # so `json.loads(capsys.readouterr().out)` was parsing `data.daily`'s
    # document followed by `experiment.grade`'s and raising
    # `JSONDecodeError: Extra data: line 11 column 1`. Dead code until now:
    # this case had never once reached the parse.
    capsys.readouterr()

    cli_main(
        [
            "experiment.grade",
            "--slot",
            "u",
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            SETTLED_TRADING_DAY,
        ]
    )
    manifest = _assert_ok(
        integration_store, "experiment.grade", discriminator="u", trading_day=SETTLED_TRADING_DAY
    )
    assert manifest["outputs"], "experiment.grade wrote no outputs — no arena cycle landed"

    # `handle_experiment_grade` prints `settled_dates` — the direct evidence
    # this cycle scored a REAL settled cut, not merely that it exited 0.
    result = json.loads(capsys.readouterr().out)
    assert result["settled_dates"], (
        "experiment.grade scored zero settled cuts — the seeded second trading day "
        f"({SETTLED_TRADING_DAY}) did not actually settle the shadow "
        f"{INTEGRATION_TRADING_DAY} produced; this is exactly the degenerate case a "
        "single fixed trading day could never avoid."
    )
    assert result["scored_arms"], "experiment.grade scored no arms at all"


def test_promote(integration_store_uri: str, integration_store: Store) -> None:
    """Runs after `test_experiment_grade` above, against the same graded
    cycle — `promote` is evidence-gated (policy §3) and may legitimately
    hold rather than move the pointer on a single week's cycle; only the
    manifest's own status is asserted, same as `test_gate`."""
    cli_main(
        [
            "promote",
            "--slot",
            "u",
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            SETTLED_TRADING_DAY,
        ]
    )
    # `discriminator=args.slot` (`crucible/cli.py::_promote`) — one job name,
    # four slots, one trading day. This case read the undiscriminated key and
    # failed with `KeyError: 'runs/promote/2026-10-07/run.json' is not
    # present` (`alpha-engine-config-I10705`), the fourth instance in this
    # module of the same missing-discriminator defect.
    _assert_ok(integration_store, "promote", discriminator="u", trading_day=SETTLED_TRADING_DAY)


# ── explain — walks the lineage of the register experiment.new just wrote ──


def test_explain(integration_store_uri: str, integration_store: Store) -> None:
    from crucible.keys import arm_register_key

    target = arm_register_key("u")
    rc = cli_main(
        [
            "explain",
            target,
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    assert rc == 0
    _assert_ok(integration_store, "explain")


# ── release.pin / release.lock / smoke — a minimal, real published release


@pytest.fixture(scope="session")
def published_release_sha(integration_store_uri: str, integration_store: Store) -> str:
    """Publish a minimal, real release under the dedicated store.

    `_verify_release_artifacts` (smoke's gate) checks hash consistency
    between `release.json` and the wheel bytes — it never inflates or
    installs the wheel — so a tiny synthetic payload is a fully real
    exercise of the publish/verify wire, not a shortcut around it.
    """
    from crucible.deploy import main as deploy_main
    from crucible.release import wheel_filename_for

    sha = hashlib.sha1(f"crucible-integration-tier-{INTEGRATION_TRADING_DAY}".encode()).hexdigest()
    wheel_bytes = f"integration-tier synthetic wheel for {sha}".encode()
    wheel_filename = wheel_filename_for(sha)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        wheel_path = Path(td) / wheel_filename
        wheel_path.write_bytes(wheel_bytes)
        release_json = Path(td) / "release.json"
        release_json.write_text(
            json.dumps(
                {
                    "schema_version": "release.v3",
                    "sha": sha,
                    "lockfile_sha256": hashlib.sha256(b"integration-tier-lockfile").hexdigest(),
                    "wheel_sha256": hashlib.sha256(wheel_bytes).hexdigest(),
                    "wheel_filename": wheel_filename,
                    "python_requires": ">=3.12,<3.13",
                    "extra": {},
                }
            ),
            encoding="utf-8",
        )
        provenance_json = Path(td) / "provenance.json"
        provenance_json.write_text(
            json.dumps(
                {
                    "schema_version": "release_provenance.v1",
                    "sha": sha,
                    "run_id": "integration-tier",
                    "run_attempt": "1",
                    "built_at": f"{INTEGRATION_TRADING_DAY}T00:00:00Z",
                    "workflow_run_url": "https://github.com/nousergon/crucible/actions",
                    "test_summary": "integration tier fixture — not a real CI test run",
                }
            ),
            encoding="utf-8",
        )
        rc = deploy_main(
            [
                "publish",
                "--sha",
                sha,
                "--store",
                integration_store_uri,
                "--wheel",
                str(wheel_path),
                "--release-json",
                str(release_json),
                "--provenance-json",
                str(provenance_json),
            ]
        )
    assert rc == 0, f"publishing the integration-tier fixture release failed: rc={rc}"
    return sha


def test_release_pin(
    integration_store_uri: str, integration_store: Store, published_release_sha: str
) -> None:
    rc = cli_main(
        [
            "release.pin",
            published_release_sha,
            "--target",
            "current",
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    assert rc == 0
    _assert_ok(integration_store, "release.pin")


def test_release_lock(
    integration_store_uri: str, integration_store: Store, published_release_sha: str
) -> None:
    from crucible.release_retention import RELEASE_LOCK_JOB

    rc = cli_main(
        [
            RELEASE_LOCK_JOB,
            published_release_sha,
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    assert rc == 0
    # `discriminator=published_release_sha`, not the bare key
    # (`alpha-engine-config-I10705`): `release_lock_handler` passes
    # `discriminator=sha` to `run_job` on purpose, so two repairs on one
    # trading day cannot overwrite one another's manifest. This case read
    # back `runs/release.lock/{day}/run.json` — a key the job never writes —
    # and failed with `KeyError: 'runs/release.lock/2026-09-08/run.json' is
    # not present`, which the issue grouped with the `data.weekly` cascade
    # and is in fact an independent defect in this assertion.
    _assert_ok(integration_store, RELEASE_LOCK_JOB, discriminator=published_release_sha)


def test_smoke(
    integration_store_uri: str, integration_store: Store, published_release_sha: str
) -> None:
    cli_main(
        [
            "smoke",
            "--release",
            published_release_sha,
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    _assert_ok(integration_store, "smoke")


# ── the daily/weekly observing surfaces — S3-only, safe against an empty
#    dedicated store by construction (they report gaps, they do not require
#    prior evidence to run) ─────────────────────────────────────────────────


def test_drift(integration_store_uri: str, integration_store: Store) -> None:
    cli_main(
        [
            "drift",
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    _assert_ok(integration_store, "drift")


def test_console(integration_store_uri: str, integration_store: Store) -> None:
    cli_main(
        [
            "console",
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    _assert_ok(integration_store, "console")


def test_board(integration_store_uri: str, integration_store: Store) -> None:
    cli_main(
        [
            "board",
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    _assert_ok(integration_store, "board")


def test_report(integration_store_uri: str, integration_store: Store) -> None:
    cli_main(
        [
            "report",
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    _assert_ok(integration_store, "report")


def test_alerts_sweep(integration_store_uri: str, integration_store: Store) -> None:
    """Real SNS publish path — to the DEDICATED pages/muted topics only
    (`_dedicated_topic_env`, session-autouse). Never the production ones.
    """
    cli_main(
        [
            "alerts.sweep",
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    _assert_latest_ok(integration_store, "alerts.sweep")


def test_heartbeat(integration_store_uri: str, integration_store: Store) -> None:
    """Publishes a real heartbeat to the dedicated pages topic — that IS the
    job's own purpose (weekly proof the alerting path itself is alive), so
    proving it against a real, isolated topic is more faithful than a mock.
    """
    cli_main(
        [
            "heartbeat",
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    _assert_ok(integration_store, "heartbeat")


def test_report_morning(integration_store_uri: str, integration_store: Store) -> None:
    """The 25th and last job (`alpha-engine-config-I10458`; see README.md,
    "The dedicated destinations"). `conftest.py::_morning_destination_env`
    (session-autouse) points this run at TWO dedicated destinations, never
    Brian's real operator chat or the private production tracker:

    * the GitHub half posts a REAL comment to the rolling `[v2 board] daily
      update` issue on the PUBLIC `nousergon/crucible` repo itself — there is
      no muted-tracker equivalent to a zero-subscriber SNS topic, so a
      dedicated PUBLIC repo is the isolation boundary here, the same role a
      dedicated library plays for ArcticDB;
    * the Telegram half is a REAL `krepis.alerts.publish()` call routed to
      `destination="console_only"` with a real `console_artifact` — fully
      exercised, `ok=True`, and incapable of reaching a phone (`krepis.
      alerts.resolve_destination`'s own contract), which is what makes this
      a proof of the delivery PATH rather than a `--dry-run` stub that would
      skip the send and prove nothing about it.

    A missing `CRUCIBLE_TRACKER_APP_SSM_PREFIX` grant on the integration
    role fails this ONE case with a named `TrackerError` — loud, not absent
    (`crucible/AGENTS.md` rule 5) — until `alpha-engine-config-I10462`'s own
    IAM scope is extended to it.
    """
    cli_main(
        [
            "report.morning",
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    manifest = _assert_latest_ok(integration_store, "report.morning")
    assert manifest["outputs"], (
        "report.morning wrote no outputs — the message/update/history rows never landed"
    )


def test_gate(integration_store_uri: str, integration_store: Store) -> None:
    """The gate ALWAYS reads NOT MET against a fresh dedicated store — that
    is a correct, expected reading, not a test failure. Only the manifest's
    own status is asserted.
    """
    from crucible.track_f import gate_names

    # `--publish` (`alpha-engine-config-I10576`): without it `crucible gate`
    # writes nothing at all, manifest included, so `_assert_ok` would have no
    # manifest to grade. The integration store is dedicated and disposable,
    # so publishing the dated reading and the ladder into it is safe.
    cli_main(
        [
            "gate",
            "--gate",
            gate_names()[0],
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
            "--publish",
        ]
    )
    _assert_ok(integration_store, "gate")


def test_gate_close(integration_store_uri: str, integration_store: Store) -> None:
    """Files a closing record for any phase reading MET. Against a fresh
    dedicated store nothing reads MET, so this exercises the real read path
    and writes zero closing records — safe by construction, not by a flag.
    """
    from crucible.track_f import GATE_CLOSE_JOB

    cli_main(
        [
            GATE_CLOSE_JOB,
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    _assert_ok(integration_store, GATE_CLOSE_JOB)


# ── migrate.history — read-only against a v1 source; the dedicated store
#    itself, which holds no v1 artifacts, so `--allow-missing` names that
#    honestly instead of failing on an absence this tier cannot supply ─────


def test_migrate_history(integration_store_uri: str, integration_store: Store) -> None:
    cli_main(
        [
            "migrate.history",
            "--v1-store",
            integration_store_uri,
            "--allow-missing",
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    _assert_ok(integration_store, "migrate.history")


# ── fault.record / fault.probe — plan §10.7's own exercise machinery ───────


def test_fault_record(
    integration_store_uri: str, integration_store: Store, published_release_sha: str
) -> None:
    """Record an `unreachable` fault against a fault the evidence registry
    actually covers, selected FROM that registry.

    **Was `SCRIPTED_FAULTS[0]`** (`alpha-engine-config-I10705`, group C),
    which resolves to `spot_terminated_mid_job` and failed with
    `FaultRecordRefusedError: no machine-checkable evidence is declared ...
    Registered: ['stale_release_pointer']`. Two things were wrong with that
    argv and only one of them is the index:

    * **`unreachable` is the wrong OUTCOME for that fault.** Read read-only
      2026-09-14 with `AWS_PROFILE=ne-admin`, the production record
      `faults/2026-08-11/spot_terminated_mid_job.json` carries
      `outcome: absorbed` with `attempt: {n: 2, reason: spot_interruption}`
      against `runs/data.daily/2026-08-11/run.json`. A spot reclamation
      demonstrably REACHES the system and is absorbed by the declared
      transient retry — so registering an `unreachable` probe for it would
      be registering a claim the production evidence contradicts, which is
      the attestation-a-human-typed shape `crucible.faults` exists to refuse.
      `crucible.faults.UNREACHABLE_PROBES` is right to omit it.
    * **A positional index picks whichever fault happens to be first.**
      `SCRIPTED_FAULTS` is ordered by plan §10.7's narration, not by what is
      recordable, so `[0]` silently re-selects a different fault the day that
      tuple is reordered. Selecting from `UNREACHABLE_PROBES` instead makes
      the case structurally incapable of naming a fault with no registered
      evidence — the class fix, not this instance's.

    `published_release_sha` is requested for ordering, not for its value:
    `stale_release_pointer`'s two probes are executed for real against this
    store, and `_probe_published_release_objects_are_retained` refuses a
    record when not one published release object is found ("vacuously true
    and evidence of nothing"). The fixture is what puts one there.
    """
    from crucible.faults import UNREACHABLE_PROBES
    from crucible.gate import SCRIPTED_FAULTS

    recordable = [fault for fault in SCRIPTED_FAULTS if fault in UNREACHABLE_PROBES]
    assert recordable, (
        "no scripted fault has registered machine-checkable evidence, so no `unreachable` "
        "record can be filed at all — a finding about `crucible.faults`, not a reason to "
        "skip this case"
    )

    cli_main(
        [
            "fault.record",
            "--fault",
            recordable[0],
            "--outcome",
            "unreachable",
            "--store",
            integration_store_uri,
            "--run-mode",
            "live",
            "--date",
            INTEGRATION_TRADING_DAY,
        ]
    )
    _assert_ok(integration_store, "fault.record")


def test_fault_probe(integration_store_uri: str, integration_store: Store) -> None:
    """Deliberately induces a real router transport failure on the real
    dispatched path (plan §10.7 fault 3). `run_job` always re-raises a job's
    own exception after writing the manifest (`crucible.runner.run_job`,
    "Re-raises on failure") and `fault_probe_handler`'s own docstring says
    the same ("Exits non-zero, always, by design") — so `cli_main` raising
    `FaultProbeFailure` here, on EVERY exec context, IS the job succeeding at
    its purpose; the manifest underneath it still records the induced
    failure, and this case reads THAT record, never the process's own exit
    path.

    **Was a bare `cli_main([...])` call with no `pytest.raises`**
    (`alpha-engine-config-I10701`, third measurement, run 34797908861): given
    the paragraph above, `cli_main` raises for `fault.probe` on every
    environment this tier has ever run in, so the `manifest["status"]`
    assertion that used to follow it was dead code from the day this case
    was written — never reached, on any exec context. Caught here for the
    first time.

    `classify_probe_failure`'s outcome legitimately differs by exec context.
    `chaos_probe`'s two registered members declare `reachable_from: [ec2]`
    only (`alpha-engine-config/private-docs/LLM_MODEL_REGISTRY.yaml`) — the
    group targets the dashboard box's OWN loopback always-503 listener
    (`nous-ergon-ops-PR1178`), a property of that one spot box, not of the
    fault-injection capability in the abstract. From `ci`
    (`KREPIS_EXEC_CONTEXT=ci`, no such listener, no local LiteLLM proxy
    either) the router cannot even attempt the call, which
    `classify_probe_failure` correctly reads as `routing_refusal` rather than
    the `upstream_transport_failure` an `ec2`/`laptop` run produces — both
    are real outcomes, both write a `status: failed` manifest, and either is
    what this case exists to prove happened. Widening `chaos_probe`'s
    `reachable_from` to include `ci` was considered and rejected (this
    issue's own second option): it would either still point CI at a box it
    cannot reach (no change) or require standing up a second, CI-local
    always-503 listener purely to fake reachability — proving a mock 503
    gets classified correctly, not that this tier's real dispatched path
    works, which is `test.integration`'s actual contract. So `routing_refusal`
    is accepted here by name, not silently swallowed: this assertion is the
    recorded reason a CI run cannot produce `upstream_transport_failure`.
    """
    from crucible.fault_probe import (
        PROBE_OUTCOME_ROUTING_REFUSAL,
        PROBE_OUTCOME_UPSTREAM_TRANSPORT_FAILURE,
        FaultProbeFailure,
        probe_outcome_from_reason,
    )
    from crucible.llm import FAULT_INJECTION_CAPABILITY_CLASSES

    with pytest.raises(FaultProbeFailure):
        cli_main(
            [
                "fault.probe",
                "--fault-capability-class",
                sorted(FAULT_INJECTION_CAPABILITY_CLASSES)[0],
                "--store",
                integration_store_uri,
                "--run-mode",
                "live",
                "--date",
                INTEGRATION_TRADING_DAY,
            ]
        )
    # Not `_assert_ok`: a chaos probe's whole purpose is inducing a router
    # failure on the real dispatched path, so `status` legitimately reads
    # `failed` here with the induced cause as `reason` — the manifest simply
    # has to EXIST, the same durable telemetry every other job produces.
    manifest = _manifest(integration_store, "fault.probe")
    assert manifest["status"] == "failed", manifest
    outcome = probe_outcome_from_reason(manifest.get("reason", ""))
    assert outcome in (PROBE_OUTCOME_ROUTING_REFUSAL, PROBE_OUTCOME_UPSTREAM_TRANSPORT_FAILURE), (
        f"fault.probe's manifest recorded outcome {outcome!r}, not one of the two this tier "
        "can legitimately produce — see this test's own docstring for why routing_refusal "
        "(ci, chaos_probe structurally unreachable) and upstream_transport_failure (ec2/"
        f"laptop, the real induced fault) are both acceptable here: {manifest.get('reason')!r}"
    )


def test_the_trading_day_used_by_this_module_is_real() -> None:
    """Self-test for the fixed literal every case above keys its manifests
    under — mirrors AGENTS.md's own "give every guard a self-test that
    shows it firing"."""
    assert_trading_day(INTEGRATION_TRADING_DAY, context="tests.integration.INTEGRATION_TRADING_DAY")
