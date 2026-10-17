"""Phase 0 of the M-slot rebuild: can the offline evaluation infra be trusted?

`alpha-engine-config-I12395`. Before a pre-registered experiment programme
spends its trial budget on the model zoo, the harness that grades it has to
pass four checks of its own. Each test here is one of them, and each drives
the REAL path (`grade_arm` / `cpcv_oos_ic` / the `crucible` CLI), never a
re-implementation of it:

1. **Overlapping-label null.** A seeded random walk carries no edge, its
   features are persistent functions of its past and its 21-session labels
   overlap. A grader that leaks the label overlap into training — the
   one-sided purge `crucible-PR414` fixed — reads a positive OOS IC on it.
   CPCV and walk-forward must both sit within two standard errors of zero.
2. **Planted control through the FIT path.** The slot's own planted/null
   control arms (`crucible.slots.cycle`) are scored on the realized return
   directly and never fitted, so they prove the scoring half and nothing
   about `_fit_rows`. Here a planted feature of known rank correlation and a
   pure-noise feature are each FITTED as ordinary recipes, and the planted
   arm's OOS IC must land inside a two-sided band derived from the planted
   correlation — not merely carry the right sign.
3. **Isolation.** A produce -> grade -> promote run pointed at a scratch
   `--store`, or run with `--dry-run`, writes nothing into the live store:
   no `champions/`, `predictions/`, `arena/` or `ledger/` key, and no S3
   call at all. (`--run-mode replay` against the live store DOES write live
   keys today; nothing here relies on or changes that.)
4. **Byte-identical replay.** Two produce -> grade runs in separate
   processes, on the same seeded inputs and different hash seeds, write
   byte-identical artifacts. The only bytes allowed to differ are the
   declared per-run provenance (manifests, and the ledger row's `run_id` /
   `written_at_utc`).

No network, no AWS: every store is a `LocalStore` under `tmp_path`, and the
S3 variant of (3) replaces `boto3` with a recorder that fails on first use.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from crucible.cli import main as cli_main
from crucible.keys import strategy_arm_key
from crucible.slots.model import (
    CPCVSpec,
    EstimatorSpec,
    FeaturePanel,
    ModelRecipe,
    TrainingWindowSpec,
    cpcv_oos_ic,
    grade_arm,
)
from tests.support.panels import trading_days
from tests.test_slot_model_cycle_job import (
    BASE_COLUMN,
    SESSIONS,
    _build_store,
    _write_recipe,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The production label horizon. Overlap is the point: a 21-session forward
#: return observed on consecutive days shares 20 of its 21 daily returns.
HORIZON = 21


# --------------------------------------------------------------------------
# Synthetic panels — a random walk with no edge, and one with a planted edge.
# --------------------------------------------------------------------------


def _random_walk(
    *, seed: int, n_days: int, n_names: int, lookback: int = 63
) -> tuple[np.ndarray, int]:
    """Cumulative log prices of an iid Gaussian random walk, and its offset.

    ``lookback`` sessions of history precede the panel's first day and
    ``HORIZON`` follow its last, so every panel row has a full feature
    window and a fully realized label. Per-name independent, zero drift: no
    feature computed from the past can predict the future, by construction.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.01, size=(lookback + n_days + HORIZON, n_names))
    return np.cumsum(steps, axis=0), lookback


#: Slow per-name characteristics: an AR(1) with this coefficient keeps
#: ~98% of its value across a 21-session label window.
CHARACTERISTIC_PERSISTENCE = 0.999
MOMENTUM_COLUMNS = ("mom_21d_ratio", "mom_63d_ratio")
CHARACTERISTIC_COLUMNS = tuple(f"char{k}_zscore" for k in range(5))


def _null_panel(*, seed: int, n_days: int = 260, n_names: int = 40) -> FeaturePanel:
    """No edge, PERSISTENT features, overlapping 21-session labels.

    Persistence is what gives a leak somewhere to come from. With iid
    features a fit on a training day adjacent to the test block learns
    nothing transferable even when its label overlaps the test label; with
    persistent features the adjacent day's feature vector is nearly the test
    day's, so an overlapping label hands a fit with the capacity to memorise
    it the test block's realized cross-section — the bias the two-sided
    purge removes. Two kinds: trailing returns of the walk itself, and slow
    characteristics drawn independently of it.
    """
    prices, offset = _random_walk(seed=seed, n_days=n_days, n_names=n_names)
    rows = np.arange(offset, offset + n_days)
    rng = np.random.default_rng(seed + 7)
    features = {
        "mom_21d_ratio": prices[rows] - prices[rows - 21],
        "mom_63d_ratio": prices[rows] - prices[rows - 63],
    }
    innovation = math.sqrt(1.0 - CHARACTERISTIC_PERSISTENCE**2)
    for column in CHARACTERISTIC_COLUMNS:
        block = np.empty((n_days, n_names))
        block[0] = rng.normal(0.0, 1.0, n_names)
        for t in range(1, n_days):
            block[t] = CHARACTERISTIC_PERSISTENCE * block[t - 1] + innovation * rng.normal(
                0.0, 1.0, n_names
            )
        features[column] = block
    return FeaturePanel(
        dates=tuple(trading_days(n_days)),
        names=tuple(f"T{i:03d}" for i in range(n_names)),
        features=features,
        forward_returns=prices[rows + HORIZON] - prices[rows],
        feature_version="v1",
    )


def _zscore_rows(block: np.ndarray) -> np.ndarray:
    return (block - block.mean(axis=1, keepdims=True)) / block.std(axis=1, keepdims=True)


#: The planted cross-sectional Pearson correlation between the planted
#: feature and the 21-session label.
PLANTED_RHO = 0.3


def _planted_panel(*, seed: int, n_days: int = 260, n_names: int = 100) -> FeaturePanel:
    """The overlapping-label random walk plus two columns of known content.

    ``planted_signal_zscore`` is ``rho * z(label) + sqrt(1 - rho^2) * noise``
    per date — a bivariate-normal cross-section whose population Spearman
    correlation with the label is ``(6/pi) * asin(rho/2)`` exactly (Pearson
    1907). ``null_noise_zscore`` is independent noise. Both are look-ahead
    columns on purpose: this is the §10.1 control pair, built so the fit
    path has to carry them.
    """
    prices, offset = _random_walk(seed=seed, n_days=n_days, n_names=n_names)
    rows = np.arange(offset, offset + n_days)
    label = prices[rows + HORIZON] - prices[rows]
    rng = np.random.default_rng(seed + 1_000_003)
    noise = rng.normal(0.0, 1.0, size=label.shape)
    planted = PLANTED_RHO * _zscore_rows(label) + math.sqrt(1.0 - PLANTED_RHO**2) * noise
    return FeaturePanel(
        dates=tuple(trading_days(n_days)),
        names=tuple(f"T{i:03d}" for i in range(n_names)),
        features={
            "planted_signal_zscore": planted,
            "null_noise_zscore": rng.normal(0.0, 1.0, size=label.shape),
        },
        forward_returns=label,
        feature_version="v1",
    )


def _recipe(
    name: str,
    features: tuple[str, ...],
    *,
    estimator: EstimatorSpec | None = None,
    min_days: int = 60,
) -> ModelRecipe:
    """A production-shaped recipe: 21-session label, 6-choose-2 CPCV, ridge."""
    return ModelRecipe(
        name=name,
        features=features,
        estimator=estimator or EstimatorSpec(kind="ridge", params={"alpha": 1.0}),
        label_horizon_trading_days=HORIZON,
        refit_cadence_trading_days=5,
        training_window=TrainingWindowSpec(kind="expanding", min_trading_days=min_days),
        cpcv=CPCVSpec(n_groups=6, k_test=2, embargo_trading_days=2),
        # Before the fixture's first session, so the OOS clock opens at the
        # end of the warm-up and not at registration.
        registered_at="2020-01-02",
    )


def _walk_forward_ic(grade) -> float:
    assert grade.status == "ok", grade.reason
    return float(np.mean(list(grade.series.scores.values())))


# --------------------------------------------------------------------------
# 1. The overlapping-label null.
# --------------------------------------------------------------------------

#: Independent replicates. Each is a fresh random walk, so the spread of the
#: per-seed means IS the sampling distribution of the statistic — no model
#: of the overlap-induced serial correlation is needed to get its SE right.
#: Declared before any reading was taken; not tuned to pass.
NULL_SEEDS = tuple(range(20261017, 20261017 + 12))

#: A tree ensemble with the capacity to memorise a persistent cross-section,
#: graded by CPCV alone (a walk-forward of it refits ~280 ensembles a seed).
#: This is the arm with POWER against the purge: measured on these seeds with
#: the after-side purge reverted to the embargo alone (the pre-`crucible-PR414`
#: construction), its mean CPCV IC reads +0.0274 at SE 0.0086 (3.2 SE) and
#: this test fails; with the two-sided purge it reads -0.0062 at SE 0.0072.
#: The ridge arm's bias under the old purge was ~0.001 — a linear fit over
#: five-sixths of the panel is barely moved by 21 leaked days.
NULL_TREES = EstimatorSpec(
    kind="lightgbm",
    params={"num_boost_round": 50, "seed": 1, "learning_rate": 0.1, "num_leaves": 15},
)


@pytest.fixture(scope="module")
def null_readings() -> dict[str, np.ndarray]:
    linear = _recipe("null_momentum", MOMENTUM_COLUMNS)
    trees = _recipe("null_characteristics", CHARACTERISTIC_COLUMNS, estimator=NULL_TREES)
    readings: dict[str, list[float]] = {"walk_forward": [], "cpcv_linear": [], "cpcv_trees": []}
    for seed in NULL_SEEDS:
        panel = _null_panel(seed=seed)
        grade = grade_arm(linear, panel, as_of=panel.dates[-1])
        assert grade.cpcv.status == "ok", grade.cpcv.reason
        readings["walk_forward"].append(_walk_forward_ic(grade))
        readings["cpcv_linear"].append(grade.cpcv.mean_ic)
        tree_cpcv = cpcv_oos_ic(
            panel, recipe=trees, cpcv=trees.cpcv, label_horizon_trading_days=HORIZON
        )
        assert tree_cpcv.status == "ok", tree_cpcv.reason
        readings["cpcv_trees"].append(tree_cpcv.mean_ic)
    return {name: np.array(values) for name, values in readings.items()}


class TestTheOverlappingLabelNullReadsZero:
    @pytest.mark.parametrize("statistic", ["walk_forward", "cpcv_linear", "cpcv_trees"])
    def test_the_mean_oos_ic_is_within_two_standard_errors_of_zero(
        self, null_readings, statistic
    ) -> None:
        values = null_readings[statistic]
        mean = float(values.mean())
        se = float(values.std(ddof=1) / math.sqrt(values.size))
        assert se > 0.0, f"{statistic}: every seed read the same IC — the replicates are not"
        assert abs(mean) <= 2.0 * se, (
            f"{statistic} OOS IC on a no-edge random walk with overlapping {HORIZON}-session "
            f"labels reads {mean:+.4f} across {values.size} seeds, outside ±2·SE = "
            f"±{2.0 * se:.4f} (per-seed: {np.round(values, 4).tolist()}). A grader that "
            "trains on labels overlapping its test labels reads a positive IC here."
        )

    def test_the_cpcv_folds_purge_the_horizon_on_both_sides(self) -> None:
        """The construction the null reading rests on, read off the folds.

        No training date may sit within ``HORIZON`` sessions of a test date on
        either side — a purge sized to the embargo alone (2 sessions) left 14
        of 15 folds training on overlapping labels before `crucible-PR414`.
        """
        recipe = _recipe("null_momentum", MOMENTUM_COLUMNS)
        panel = _null_panel(seed=NULL_SEEDS[0])
        result = cpcv_oos_ic(
            panel, recipe=recipe, cpcv=recipe.cpcv, label_horizon_trading_days=HORIZON
        )
        index = {day: i for i, day in enumerate(panel.dates)}
        assert len(result.folds) == math.comb(6, 2)
        for fold in result.folds:
            test = np.array([index[d] for d in fold.test_dates])
            train = np.array([index[d] for d in fold.train_dates])
            gap = np.abs(train[:, None] - test[None, :]).min()
            assert gap > HORIZON, (
                f"fold {fold.test_groups}: a training date sits {gap} session(s) from a "
                f"test date, inside the {HORIZON}-session label overlap"
            )


# --------------------------------------------------------------------------
# 2. The planted control, through the fit path.
# --------------------------------------------------------------------------

#: Spearman's rho of a bivariate normal with Pearson correlation
#: ``PLANTED_RHO`` — the planted arm's population per-date rank IC. A
#: one-feature ridge prediction is monotone in its feature, so this is what
#: the fitted arm's IC estimates whatever coefficient the fit lands on.
EXPECTED_PLANTED_IC = (6.0 / math.pi) * math.asin(PLANTED_RHO / 2.0)


def _planted_band_se(n_names: int, n_dates: int, value: float) -> float:
    """A CONSERVATIVE standard error for the mean of ``n_dates`` per-date ICs.

    Per date: Fieller, Hartley & Pearson (1957), ``var(atanh(r_s)) ~ 1.06 /
    (n - 3)``, mapped back through the delta method. Across dates: one
    effective observation per ``HORIZON`` sessions, as if every date inside
    a label window were perfectly correlated with its neighbours. The
    planted feature's noise is fresh each date, so the true SE is smaller;
    the band is wide on purpose and still excludes half the planted edge.
    """
    per_date = (1.0 - value**2) * math.sqrt(1.06 / (n_names - 3))
    effective = max(1, n_dates // HORIZON)
    return per_date / math.sqrt(effective)


@pytest.fixture(scope="module")
def graded() -> dict[str, object]:
    panel = _planted_panel(seed=20261018)
    as_of = panel.dates[-1]
    return {
        "panel": panel,
        "planted": grade_arm(
            _recipe("control_planted_fit", ("planted_signal_zscore",)), panel, as_of=as_of
        ),
        "null": grade_arm(_recipe("control_null_fit", ("null_noise_zscore",)), panel, as_of=as_of),
    }


class TestThePlantedControlThroughTheFitPath:
    def test_the_planted_fit_learns_the_planted_direction(self, graded) -> None:
        coefficients = graded["planted"].fit.coefficients
        assert coefficients.shape == (1,)
        assert coefficients[0] > 0.0, (
            "a positive planted correlation must fit a positive coefficient; "
            f"got {coefficients[0]!r}"
        )

    @pytest.mark.parametrize("statistic", ["cpcv", "walk_forward"])
    def test_the_planted_ic_lands_in_the_two_sided_expected_band(self, graded, statistic) -> None:
        planted = graded["planted"]
        panel = graded["panel"]
        value = planted.cpcv.mean_ic if statistic == "cpcv" else _walk_forward_ic(planted)
        se = _planted_band_se(len(panel.names), planted.oos_n, EXPECTED_PLANTED_IC)
        low, high = EXPECTED_PLANTED_IC - 3.0 * se, EXPECTED_PLANTED_IC + 3.0 * se
        assert low <= value <= high, (
            f"planted {statistic} OOS IC {value:+.4f} is outside the expected band "
            f"[{low:+.4f}, {high:+.4f}] around (6/pi)·asin({PLANTED_RHO}/2) = "
            f"{EXPECTED_PLANTED_IC:+.4f}. Below it the fit path is losing a known edge; "
            "above it the grader is reading more than was planted, which is a leak."
        )

    @pytest.mark.parametrize("statistic", ["cpcv", "walk_forward"])
    def test_the_null_arm_reads_zero_and_the_planted_arm_dominates_it(
        self, graded, statistic
    ) -> None:
        planted, null, panel = graded["planted"], graded["null"], graded["panel"]
        if statistic == "cpcv":
            assert null.cpcv.status == "ok", null.cpcv.reason
            planted_ic, null_ic = planted.cpcv.mean_ic, null.cpcv.mean_ic
        else:
            planted_ic, null_ic = _walk_forward_ic(planted), _walk_forward_ic(null)
        se = _planted_band_se(len(panel.names), null.oos_n, 0.0)
        assert abs(null_ic) <= 3.0 * se, (
            f"the null control fitted on pure noise reads {statistic} IC {null_ic:+.4f}, "
            f"outside ±3·SE = ±{3.0 * se:.4f}"
        )
        assert planted_ic - null_ic >= 6.0 * se, (
            f"planted {planted_ic:+.4f} vs null {null_ic:+.4f} ({statistic}): the margin "
            f"{planted_ic - null_ic:+.4f} is under 6·SE = {6.0 * se:.4f}, so the fit path "
            "cannot separate a known edge from noise with room to spare"
        )

    def test_the_two_arms_are_graded_on_the_same_out_of_sample_dates(self, graded) -> None:
        planted, null = graded["planted"], graded["null"]
        assert set(planted.series.scores) == set(null.series.scores)
        assert planted.oos_n == null.oos_n > 100


# --------------------------------------------------------------------------
# 3 & 4. The CLI path: isolation and byte-identical replay.
# --------------------------------------------------------------------------

#: Every live prefix a produce -> grade -> promote run can write to.
LIVE_PREFIXES = ("champions/", "predictions/", "arena/", "ledger/")

#: Sessions that settle a 21-session shadow by the grade day.
HISTORY_DAYS = SESSIONS[14:26]
HISTORY_GRADE_DAY = SESSIONS[40]
#: The run under test.
RUN_DAY = SESSIONS[41]


def _cli(*argv: str) -> None:
    """One `crucible` invocation through the console-script entry point."""
    with contextlib.redirect_stdout(io.StringIO()):
        code = cli_main(list(argv))
    assert code == 0, f"crucible {' '.join(argv)} exited {code}"


def _seed_store(root: Path) -> Path:
    """A real feature layer, price panel and strategy tree under ``root/store``.

    The recipe is synced INTO the store (`strategy/current/arms/m/`), the
    shape a spot box with no checkout reads, so no `CRUCIBLE_STRATEGY_DIR`
    is involved and `promote` resolves the seated arm from the same tree.
    """
    store = _build_store(root)
    checkout = root / "checkout" / "arms" / "m"
    _write_recipe(checkout, "base", features=[BASE_COLUMN])
    store.put_bytes(strategy_arm_key("m", "base"), (checkout / "base.yaml").read_bytes())
    return Path(store.root)


def _with_history(store_root: Path) -> str:
    """Produce, grade and seat a champion for real, the way a store gets here.

    Returns the seated arm id. Seating it is what makes `experiment.run`
    publish the trader's `predictions/` feed, so the run under test has a
    write owed to every one of :data:`LIVE_PREFIXES` (`promote` re-seats
    `champions/` or holds it).
    """
    store = str(store_root)
    for day in HISTORY_DAYS:
        _cli(
            "experiment.run", "--slot", "m", "--date", day, "--store", store, "--run-mode", "replay"
        )
    _cli(
        "experiment.grade", "--slot", "m", "--date", HISTORY_GRADE_DAY,
        "--store", store, "--run-mode", "replay",
    )  # fmt: skip
    register = (store_root / "arms" / "m" / "register.jsonl").read_text(encoding="utf-8")
    [arm_id] = [
        row["arm_id"]
        for row in map(json.loads, register.splitlines())
        if not row["record"]["control"]
    ]
    _cli(
        "promote", "--slot", "m", "--date", HISTORY_GRADE_DAY, "--store", store,
        "--run-mode", "replay", "--revert-to", arm_id, "--reason", "phase-0 fixture seat",
    )  # fmt: skip
    assert (store_root / "champions" / "m" / "current.json").is_file()
    return arm_id


def _run_under_test(*extra: str, promote_day: str = RUN_DAY) -> None:
    """produce -> grade for :data:`RUN_DAY`, then promote, plus ``extra`` flags.

    ``promote_day`` exists for the dry run: `promote` acts on the cycle a
    REAL `experiment.grade` run filed (it refuses without that manifest), and
    a dry-run grade files nothing, so the rehearsed promote reads the history
    day's genuine cycle instead.
    """
    for job, day in (
        ("experiment.run", RUN_DAY),
        ("experiment.grade", RUN_DAY),
        ("promote", promote_day),
    ):
        _cli(job, "--slot", "m", "--date", day, "--run-mode", "replay", *extra)


def _snapshot(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _changed(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


class _Boto3Recorder:
    """A `boto3` stand-in that records every use and refuses it.

    Refusing as well as recording: a store that reached for S3 must fail the
    run in front of the test rather than be handed a fake that lets it
    "succeed" against nothing.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(f"{args!r} {kwargs!r}")
        raise AssertionError(f"boto3 reached from an isolated run: {args!r} {kwargs!r}")


class TestAScratchOrDryRunNeverWritesTheLiveStore:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch) -> dict[str, Path]:
        monkeypatch.delenv("CRUCIBLE_STRATEGY_DIR", raising=False)
        monkeypatch.delenv("CRUCIBLE_STORE", raising=False)
        live = _seed_store(tmp_path / "live")
        scratch = _seed_store(tmp_path / "scratch")
        # Both stores carry a seated champion and a settled history, so a
        # write misrouted to the live store would SUCCEED there rather than
        # fail on a missing input and hide.
        _with_history(live)
        _with_history(scratch)
        return {"live": live, "scratch": scratch}

    def test_a_scratch_store_run_writes_nothing_to_the_live_local_store(
        self, env, monkeypatch
    ) -> None:
        monkeypatch.setenv("CRUCIBLE_STORE", str(env["live"]))
        live_before = _snapshot(env["live"])
        scratch_before = _snapshot(env["scratch"])

        _run_under_test("--store", str(env["scratch"]))

        assert _changed(live_before, _snapshot(env["live"])) == []
        # The positive control: the run DID write every live prefix — into
        # the scratch store it was pointed at — so the empty diff above is
        # isolation and not a run that wrote nothing anywhere.
        written = _changed(scratch_before, _snapshot(env["scratch"]))
        for prefix in ("predictions/", "arena/", "ledger/"):
            assert any(k.startswith(prefix) for k in written), (prefix, written)

    def test_a_scratch_store_run_makes_no_s3_call_when_live_is_s3(self, env, monkeypatch) -> None:
        import boto3

        recorder = _Boto3Recorder()
        monkeypatch.setattr(boto3, "client", recorder)
        monkeypatch.setattr(boto3, "resource", recorder)
        monkeypatch.setattr(boto3, "Session", recorder)
        monkeypatch.setattr(boto3.session, "Session", recorder)
        monkeypatch.setenv("CRUCIBLE_STORE", "s3://phase0-live-store-sentinel/crucible")

        _run_under_test("--store", str(env["scratch"]))

        assert recorder.calls == []

    def test_a_dry_run_against_the_live_store_writes_nothing(self, env, monkeypatch) -> None:
        from crucible.store import CaptureLedger

        captured: list[str] = []
        record = CaptureLedger.record

        def _record(self, write):
            captured.append(write.key)
            return record(self, write)

        monkeypatch.setattr(CaptureLedger, "record", _record)
        monkeypatch.setenv("CRUCIBLE_STORE", str(env["live"]))
        live_before = _snapshot(env["live"])

        _run_under_test("--dry-run", promote_day=HISTORY_GRADE_DAY)

        changed = _changed(live_before, _snapshot(env["live"]))
        assert changed == [], f"--dry-run wrote {changed} into the live store"
        assert not [k for k in changed if k.startswith(LIVE_PREFIXES)]
        # The rehearsal ran its body to the writes and RECORDED them — the
        # same prefixes the scratch run really wrote.
        for prefix in ("predictions/", "arena/", "ledger/"):
            assert any(k.startswith(prefix) for k in captured), (prefix, captured)


#: The per-run provenance a replay is ALLOWED to change, and nothing else.
#: Manifests carry the run's ULID and wall-clock timings by contract; a
#: ledger row carries the same two facts about the run that appended it.
_PROVENANCE_FIELDS = ("run_id", "written_at_utc")


def replay_digest(root: str) -> str:
    """Seed a store under ``root``, run produce -> grade, digest the outputs.

    Run in a SUBPROCESS by :class:`TestAReplayIsByteIdentical`; module-level
    so ``python -c`` can import it. Every file the run wrote enters the
    digest byte-for-byte except run manifests; a ledger row enters with
    :data:`_PROVENANCE_FIELDS` removed and every other field intact.
    """
    store_root = _seed_store(Path(root))
    seeded = set(_snapshot(store_root))
    store = str(store_root)
    for day in HISTORY_DAYS:
        _cli(
            "experiment.run", "--slot", "m", "--date", day, "--store", store, "--run-mode", "replay"
        )
    _cli(
        "experiment.grade", "--slot", "m", "--date", HISTORY_GRADE_DAY,
        "--store", store, "--run-mode", "replay",
    )  # fmt: skip
    digest = hashlib.sha256()
    for key in sorted(set(_snapshot(store_root)) - seeded):
        if key.startswith("runs/") and key.endswith("/run.json"):
            continue
        payload = (store_root / key).read_bytes()
        if key.startswith("ledger/"):
            rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
            for row in rows:
                for name in _PROVENANCE_FIELDS:
                    row.pop(name, None)
            payload = "\n".join(json.dumps(r, sort_keys=True) for r in rows).encode("utf-8")
        digest.update(key.encode("utf-8") + b"\0" + hashlib.sha256(payload).digest())
    return digest.hexdigest()


class TestAReplayIsByteIdentical:
    def _replay(self, root: Path, hash_seed: str) -> str:
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        env.pop("CRUCIBLE_STORE", None)
        env.pop("CRUCIBLE_STRATEGY_DIR", None)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from tests.test_slot_model_phase0_infra import replay_digest; "
                "print(replay_digest(sys.argv[1]))",
                str(root),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        return completed.stdout.strip().splitlines()[-1]

    def test_two_processes_write_the_same_bytes(self, tmp_path) -> None:
        first = self._replay(tmp_path / "first", "1")
        second = self._replay(tmp_path / "second", "2")
        assert len(first) == 64
        assert first == second, (
            "two produce -> grade runs on identical seeded inputs wrote different bytes; "
            "a replay that does not reproduce its artifacts cannot reproduce a verdict"
        )

    def test_the_ledger_differs_only_in_declared_provenance(self, tmp_path, monkeypatch) -> None:
        """The exclusion in :func:`replay_digest` is not a hiding place.

        Two in-process runs: every ledger field outside
        :data:`_PROVENANCE_FIELDS` must match row for row, and the rows must
        actually carry those fields (so the strip is not stripping nothing).
        """
        monkeypatch.delenv("CRUCIBLE_STRATEGY_DIR", raising=False)
        ledgers = []
        for name in ("a", "b"):
            replay_digest(str(tmp_path / name))
            text = (tmp_path / name / "store" / "ledger" / "trials.jsonl").read_text("utf-8")
            ledgers.append([json.loads(line) for line in text.splitlines()])
        first, second = ledgers
        assert len(first) == len(second) > 0
        for a, b in zip(first, second, strict=True):
            assert set(_PROVENANCE_FIELDS) <= set(a)
            assert {k: v for k, v in a.items() if k not in _PROVENANCE_FIELDS} == {
                k: v for k, v in b.items() if k not in _PROVENANCE_FIELDS
            }
