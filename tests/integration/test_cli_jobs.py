"""One real case per exercisable CLI job (16 of 24 — see README.md).

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

import hashlib
import json
from typing import Any

import pytest

from crucible.calendar import assert_trading_day
from crucible.cli import main as cli_main
from crucible.manifest import read_manifest
from crucible.store import Store
from tests.integration.conftest import INTEGRATION_TRADING_DAY

pytestmark = pytest.mark.usefixtures("_dedicated_topic_env")


def _manifest(store: Store, job: str, *, discriminator: str | None = None) -> dict[str, Any]:
    return read_manifest(store, job, INTEGRATION_TRADING_DAY, discriminator=discriminator)


def _assert_ok(store: Store, job: str, *, discriminator: str | None = None) -> dict[str, Any]:
    manifest = _manifest(store, job, discriminator=discriminator)
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
) -> None:
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
#    (`alpha-engine-config-I10457`), never by a gap of its own. See the note
#    below `test_experiment_run` for why `experiment.grade`/`promote` are
#    NOT added here. ─────────────────────────────────────────────────────


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


# `experiment.grade`/`promote` are NOT added here. Investigation for
# `alpha-engine-config-I10457` found their blocker is NOT the ArcticDB gap
# this PR fixes: `crucible.slots.cycle.run_grade` scores only SETTLED shadows
# (`forward_returns` needs price data `horizon_trading_days` — default 21 —
# AFTER the shadow's own date), and this tier's `INTEGRATION_TRADING_DAY` is
# a fixed literal with no later trading day ever produced in the dedicated
# store, by design (AGENTS.md, "Test discipline": fixed date literals, never
# `today` arithmetic). A grade run here would legitimately score zero
# settled cuts, forever, on every nightly run — proving nothing beyond what
# `experiment.run`, `data.daily` and `data.weekly` above already prove.
# Exercising `experiment.grade`/`promote` for real needs a multi-week
# historical panel (several distinct trading days, each settled relative to
# a later one), which is real design work distinct from this issue's
# ArcticDB-library scope. Filed as a follow-up rather than landed
# speculatively and unverifiable from this session (no real S3/ArcticDB
# access — `alpha-engine-config-I9771`).


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
    _assert_ok(integration_store, RELEASE_LOCK_JOB)


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
    _assert_ok(integration_store, "alerts.sweep")


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


def test_fault_record(integration_store_uri: str, integration_store: Store) -> None:
    from crucible.gate import SCRIPTED_FAULTS

    cli_main(
        [
            "fault.record",
            "--fault",
            SCRIPTED_FAULTS[0],
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
    dispatched path (plan §10.7 fault 3) — the manifest is expected to
    record the induced failure; that IS the job succeeding at its purpose.
    """
    from crucible.llm import FAULT_INJECTION_CAPABILITY_CLASSES

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
    assert manifest["status"] in ("ok", "failed"), manifest


def test_the_trading_day_used_by_this_module_is_real() -> None:
    """Self-test for the fixed literal every case above keys its manifests
    under — mirrors AGENTS.md's own "give every guard a self-test that
    shows it firing"."""
    assert_trading_day(INTEGRATION_TRADING_DAY, context="tests.integration.INTEGRATION_TRADING_DAY")
