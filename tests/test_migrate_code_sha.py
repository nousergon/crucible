"""`crucible migrate.code_sha` — repairing the all-zero code_sha placeholder.

Normative source: `alpha-engine-config-I10626`. 89 of 164 production run
manifests carry `code_sha == "0" * 40`, written before the schema-level
refusal (`alpha-engine-config-I10454`, crucible-PR219) existed, so
`crucible.manifest.validate` — and therefore `crucible explain` — cannot read
them. This module is a one-off REPAIR, not a job: it patches the field in
place, derived from the durable record of what `releases/current` pointed at
when the broken manifest ran, and refuses rather than guesses wherever that
record is missing, conflicting, or the manifest is on the money path.

Written before `crucible/migrate.py` carried `run_migrate_code_sha` (fleet
TDD rule) and seen failing.
"""

from __future__ import annotations

import json

from crucible.keys import champion_key, manifest_key
from crucible.manifest import RUN_MANIFEST_SCHEMA_VERSION, validate
from crucible.migrate import CodeShaMigrationReport, run_migrate_code_sha
from crucible.release import release_json_key
from crucible.store import LocalStore, PointerConflictError

PLACEHOLDER = "0" * 40
REAL_SHA_A = "a" * 40
REAL_SHA_B = "b" * 40


def _manifest(
    *,
    job: str = "experiment.run",
    trading_day: str = "2026-08-28",
    started: str = "2026-08-28T14:00:00Z",
    finished: str = "2026-08-28T14:04:00Z",
    code_sha: str = REAL_SHA_A,
    status: str = "ok",
    outputs: list[dict] | None = None,
    run_id: str = "01JG0000000000000000000000",
) -> dict:
    """The floor a job must write (mirrors `test_manifest_schema.py::_valid_manifest`)."""
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "job": job,
        "run_mode": "live",
        "trading_day": trading_day,
        "calendar_date": trading_day,
        "status": status,
        "reason": "" if status == "ok" else "induced for the test",
        "started": started,
        "finished": finished,
        "code_sha": code_sha,
        "release_sha": code_sha,
        "seed": 20260828,
        "inputs": [],
        "outputs": outputs or [],
        "rows_in": 0,
        "rows_out": 0,
        "rows_rejected": [],
        "cost_usd": 0.0,
        "llm_calls": [],
        "resource": {
            "instance_type": "c7i.xlarge",
            "spot": True,
            "escalated_to_on_demand": False,
            "interruptions": 0,
            "mem_peak_mb": 0.0,
            "disk_free_mb": 0.0,
        },
        "metrics": [],
        "attempts": [{"n": 1, "reason": "initial"}],
    }


def _deploy(
    *, trading_day: str = "2026-08-27", code_sha: str = REAL_SHA_A, status: str = "ok"
) -> dict:
    """A `deploy` manifest that FINISHED well before any target manifest's
    default `started` ("2026-08-28T14:00:00Z") — the ordinary case, a deploy
    the evening before the trading day it governs."""
    return _manifest(
        job="deploy",
        trading_day=trading_day,
        started="2026-08-27T09:00:00Z",
        finished="2026-08-27T09:01:00Z",
        code_sha=code_sha,
        status=status,
    )


def _put(store: LocalStore, key: str, document: dict) -> None:
    store.put_bytes(key, json.dumps(document, indent=2, sort_keys=True).encode("utf-8"))


def _seed_release(store: LocalStore, sha: str) -> None:
    store.put_bytes(release_json_key(sha), b'{"sha": "' + sha.encode() + b'"}')


class TestDerivationAndRewrite:
    def test_rewrites_from_the_nearest_prior_successful_deploy(self, store: LocalStore) -> None:
        _put(store, manifest_key("deploy", "2026-08-27"), _deploy())
        _seed_release(store, REAL_SHA_A)
        target_key = manifest_key("experiment.run", "2026-08-28", discriminator="u")
        _put(store, target_key, _manifest(job="experiment.run", code_sha=PLACEHOLDER))

        report = run_migrate_code_sha(store)

        assert isinstance(report, CodeShaMigrationReport)
        assert report.counts == {"rewritten": 1, "refused": 0}
        rewritten = json.loads(store.get_bytes(target_key))
        assert rewritten["code_sha"] == REAL_SHA_A
        validate(rewritten)  # the whole point: it now reads
        assert rewritten["inputs"][-1]["key"] == manifest_key("deploy", "2026-08-27")
        assert rewritten["metrics"][-1]["name"] == "code_sha_migrated"
        assert rewritten["metrics"][-1]["migrated_from_code_sha"] == PLACEHOLDER

    def test_a_second_run_is_idempotent(self, store: LocalStore) -> None:
        _put(
            store,
            manifest_key("deploy", "2026-08-27"),
            _deploy(),
        )
        _seed_release(store, REAL_SHA_A)
        target_key = manifest_key("experiment.run", "2026-08-28", discriminator="u")
        _put(store, target_key, _manifest(job="experiment.run", code_sha=PLACEHOLDER))

        first = run_migrate_code_sha(store)
        second = run_migrate_code_sha(store)

        assert first.counts == {"rewritten": 1, "refused": 0}
        assert second.counts == {"rewritten": 0, "refused": 0}, (
            "nothing left carries the placeholder, so a rerun must find nothing to do"
        )

    def test_writes_a_durable_summary_document(self, store: LocalStore) -> None:
        _put(
            store,
            manifest_key("deploy", "2026-08-27"),
            _deploy(),
        )
        _seed_release(store, REAL_SHA_A)
        target_key = manifest_key("experiment.run", "2026-08-28", discriminator="u")
        _put(store, target_key, _manifest(job="experiment.run", code_sha=PLACEHOLDER))

        report = run_migrate_code_sha(store)

        # `migration_key`'s trading_day is resolved from wall-clock "now" at
        # migration time, not the target's — found by run_id, not composed.
        candidates = [k for k in store.list_keys("migrations/") if report.migration_run_id in k]
        assert len(candidates) == 1
        summary = json.loads(store.get_bytes(candidates[0]))
        assert summary["schema_version"] == "migrate_code_sha.v1"
        assert summary["counts"] == {"rewritten": 1, "refused": 0}

    def test_dry_run_writes_nothing(self, store: LocalStore) -> None:
        _put(
            store,
            manifest_key("deploy", "2026-08-27"),
            _deploy(),
        )
        _seed_release(store, REAL_SHA_A)
        target_key = manifest_key("experiment.run", "2026-08-28", discriminator="u")
        _put(store, target_key, _manifest(job="experiment.run", code_sha=PLACEHOLDER))

        report = run_migrate_code_sha(store, dry_run=True)

        assert report.dry_run is True
        assert report.counts == {"rewritten": 1, "refused": 0}
        assert report.rewritten[0]["dry_run"] is True
        still = json.loads(store.get_bytes(target_key))
        assert still["code_sha"] == PLACEHOLDER
        assert list(store.list_keys("migrations/")) == []


class TestRefusals:
    def test_refuses_a_manifest_on_the_money_path(self, store: LocalStore) -> None:
        _put(
            store,
            manifest_key("deploy", "2026-08-27"),
            _deploy(),
        )
        _seed_release(store, REAL_SHA_A)
        target_key = manifest_key("promote", "2026-08-28", discriminator="u")
        _put(
            store,
            target_key,
            _manifest(
                job="promote",
                code_sha=PLACEHOLDER,
                outputs=[
                    {
                        "key": champion_key("u"),
                        "sha256": "c" * 64,
                        "schema_version": "champion_pointer.v1",
                    }
                ],
            ),
        )

        report = run_migrate_code_sha(store)

        assert report.counts == {"rewritten": 0, "refused": 1}
        assert "money path" in report.refused[0]["reason"]
        assert json.loads(store.get_bytes(target_key))["code_sha"] == PLACEHOLDER

    def test_refuses_a_deploy_manifest_itself(self, store: LocalStore) -> None:
        target_key = manifest_key("deploy", "2026-08-28")
        _put(
            store,
            target_key,
            _manifest(job="deploy", trading_day="2026-08-28", code_sha=PLACEHOLDER, status="ok"),
        )

        report = run_migrate_code_sha(store)

        assert report.counts == {"rewritten": 0, "refused": 1}
        assert "job=deploy" in report.refused[0]["reason"]

    def test_refuses_when_no_prior_deploy_record_exists(self, store: LocalStore) -> None:
        target_key = manifest_key("experiment.run", "2026-08-28", discriminator="u")
        _put(store, target_key, _manifest(job="experiment.run", code_sha=PLACEHOLDER))

        report = run_migrate_code_sha(store)

        assert report.counts == {"rewritten": 0, "refused": 1}
        assert "no successful deploy manifest" in report.refused[0]["reason"]

    def test_refuses_on_conflicting_same_day_peers(self, store: LocalStore) -> None:
        _put(
            store,
            manifest_key("deploy", "2026-08-27"),
            _deploy(),
        )
        _seed_release(store, REAL_SHA_A)
        _seed_release(store, REAL_SHA_B)
        # A second deploy landed the SAME trading_day as the target, so the
        # deploy-history candidate (REAL_SHA_A, from the day before) and a
        # peer that already saw the day's later release disagree.
        _put(
            store,
            manifest_key("experiment.run", "2026-08-28", discriminator="r"),
            _manifest(job="experiment.run", trading_day="2026-08-28", code_sha=REAL_SHA_B),
        )
        target_key = manifest_key("experiment.run", "2026-08-28", discriminator="u")
        _put(store, target_key, _manifest(job="experiment.run", code_sha=PLACEHOLDER))

        report = run_migrate_code_sha(store)

        assert report.counts == {"rewritten": 0, "refused": 1}
        assert "ambiguous" in report.refused[0]["reason"]
        assert json.loads(store.get_bytes(target_key))["code_sha"] == PLACEHOLDER

    def test_refuses_when_the_derived_sha_has_no_release_record(self, store: LocalStore) -> None:
        _put(
            store,
            manifest_key("deploy", "2026-08-27"),
            _deploy(),
        )
        # Deliberately no `_seed_release` call: REAL_SHA_A has no
        # `releases/{sha}/release.json` in this store.
        target_key = manifest_key("experiment.run", "2026-08-28", discriminator="u")
        _put(store, target_key, _manifest(job="experiment.run", code_sha=PLACEHOLDER))

        report = run_migrate_code_sha(store)

        assert report.counts == {"rewritten": 0, "refused": 1}
        assert "release.json" in report.refused[0]["reason"]

    def test_a_failed_deploy_is_not_evidence(self, store: LocalStore) -> None:
        _put(
            store,
            manifest_key("deploy", "2026-08-27"),
            _manifest(
                job="deploy",
                trading_day="2026-08-27",
                code_sha=REAL_SHA_A,
                status="failed",
            ),
        )
        _seed_release(store, REAL_SHA_A)
        target_key = manifest_key("experiment.run", "2026-08-28", discriminator="u")
        _put(store, target_key, _manifest(job="experiment.run", code_sha=PLACEHOLDER))

        report = run_migrate_code_sha(store)

        assert report.counts == {"rewritten": 0, "refused": 1}
        assert "no successful deploy manifest" in report.refused[0]["reason"]


class _RaceStore(LocalStore):
    """A `LocalStore` that flips one key's `code_sha` to a real value on the
    SECOND `get_bytes` read — simulating a second writer landing between this
    migration's listing pass and its own re-check immediately before the
    compare-and-swap, so the refuse-rather-than-clobber branch has a real
    test rather than only existing in prose."""

    def __init__(self, root, *, race_key: str) -> None:
        super().__init__(root)
        self._race_key = race_key
        self._reads = 0

    def get_bytes(self, key: str) -> bytes:
        payload = super().get_bytes(key)
        if key == self._race_key:
            self._reads += 1
            if self._reads == 2:
                document = json.loads(payload)
                document["code_sha"] = REAL_SHA_B
                super().put_bytes(key, json.dumps(document).encode("utf-8"))
                return super().get_bytes(key)
        return payload


class TestConcurrency:
    def test_a_manifest_fixed_between_listing_and_write_is_reported_not_clobbered(
        self, tmp_path
    ) -> None:
        target_key = manifest_key("experiment.run", "2026-08-28", discriminator="u")
        store = _RaceStore(tmp_path / "store", race_key=target_key)
        _put(
            store,
            manifest_key("deploy", "2026-08-27"),
            _deploy(),
        )
        _seed_release(store, REAL_SHA_A)
        _put(store, target_key, _manifest(job="experiment.run", code_sha=PLACEHOLDER))

        report = run_migrate_code_sha(store)

        assert report.counts == {"rewritten": 0, "refused": 1}
        assert "already fixed" in report.refused[0]["reason"]
        assert json.loads(store.get_bytes(target_key))["code_sha"] == REAL_SHA_B, (
            "the other writer's value must survive untouched"
        )


def test_pointer_conflict_is_reported_not_raised(store: LocalStore, monkeypatch) -> None:
    _put(
        store,
        manifest_key("deploy", "2026-08-27"),
        _deploy(),
    )
    _seed_release(store, REAL_SHA_A)
    target_key = manifest_key("experiment.run", "2026-08-28", discriminator="u")
    _put(store, target_key, _manifest(job="experiment.run", code_sha=PLACEHOLDER))

    def _refuse(*args, **kwargs):
        raise PointerConflictError("simulated race on compare_and_swap")

    monkeypatch.setattr(LocalStore, "compare_and_swap", _refuse)

    report = run_migrate_code_sha(store)

    assert report.counts == {"rewritten": 0, "refused": 1}
    assert "compare-and-swap conflict" in report.refused[0]["reason"]


def test_refuses_when_the_started_timestamp_is_unreadable(store: LocalStore) -> None:
    _put(
        store,
        manifest_key("deploy", "2026-08-27"),
        _deploy(),
    )
    _seed_release(store, REAL_SHA_A)
    target_key = manifest_key("experiment.run", "2026-08-28", discriminator="u")
    broken = _manifest(job="experiment.run", code_sha=PLACEHOLDER)
    broken["started"] = "not-a-timestamp"
    _put(store, target_key, broken)

    report = run_migrate_code_sha(store)

    assert report.counts == {"rewritten": 0, "refused": 1}
    assert "no readable `started`" in report.refused[0]["reason"]


def test_every_target_manifest_appears_in_exactly_one_bucket(store: LocalStore) -> None:
    """The report is exhaustive: nothing carrying the placeholder is silently
    dropped from both `rewritten` and `refused`."""
    _put(
        store,
        manifest_key("deploy", "2026-08-27"),
        _deploy(),
    )
    _seed_release(store, REAL_SHA_A)
    ok_key = manifest_key("experiment.run", "2026-08-28", discriminator="u")
    _put(store, ok_key, _manifest(job="experiment.run", code_sha=PLACEHOLDER))
    stuck_key = manifest_key("experiment.run", "2026-08-20", discriminator="u")
    _put(
        store,
        stuck_key,
        _manifest(
            job="experiment.run",
            trading_day="2026-08-20",
            started="2026-08-19T14:00:00Z",
            finished="2026-08-19T14:04:00Z",
            code_sha=PLACEHOLDER,
        ),
    )

    report = run_migrate_code_sha(store)

    touched = {row["key"] for row in report.rewritten} | {row["key"] for row in report.refused}
    assert touched == {ok_key, stuck_key}
