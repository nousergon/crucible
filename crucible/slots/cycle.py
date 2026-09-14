"""The produce and grade drivers. One engine, used by every slot.

Normative source: plan §4.4 ("four slots, one engine") and §10 component 1.

`crucible/slots/universe.py` and `crucible/slots/research.py` are bindings
over these two functions: they supply the slot key and the key of the feed
the slot's champion writes, and nothing else. A second copy of this loop per
slot is exactly the shape the plan replaces — four drifting implementations
of one policy.

**Produce** (`experiment.run`) reads the feature layer for one trading day,
runs every registered arm's recipe, and writes each arm's selection. It
knows no outcome, and it writes no verdict.

**Grade** (`experiment.grade`) reads the price panel at the cycle date,
finds every shadow whose horizon has since settled, scores each into a
verdict, checks the controls, and hands the assembled series to
`nousergon_lib.arena.engine.run_cycle`. It writes the `arena_cycle`
artifact and one trial-ledger row per graded arm-cycle.

It does NOT move the pointer. `promote` is a separate job with its own
manifest, so "the evidence said X" and "the pointer moved to X" are two
records and a disagreement between them is visible.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from nousergon_lib.arena.engine import ServingPrecondition
from nousergon_lib.arena.window import ArmSeries

from crucible.calendar import assert_trading_day
from crucible.config import Settings
from crucible.data.point_in_time import EARLIEST_SNAPSHOT_BACKFILL_MODE, POINT_IN_TIME_MODE
from crucible.documents import load_store_document
from crucible.features import DEFAULT_FEATURE_VERSION, read_features
from crucible.keys import (
    arena_cycle_key,
    arm_series_key,
    champion_key,
    data_panel_key,
    experiments_prefix,
    features_key,
    shadow_key,
    verdict_key,
)
from crucible.ledger import append_trials, trial_rows
from crucible.slots import arm_name as name_component
from crucible.slots import get_slot, promotable_arms
from crucible.slots.arms import (
    control_specs,
    load_arm_specs,
    read_register,
    register_arms,
    write_register,
)
from crucible.slots.grading import (
    DEFAULT_HORIZON_TRADING_DAYS,
    ForwardReturnWindow,
    GraderControlError,
    PopulationIntegrityError,
    ScoredCrossSection,
    SelectionMissError,
    ShadowSelection,
    assert_label_control,
    control_selection,
    cross_section_key,
    forward_returns,
    grade_slot,
    produce_cross_section,
    produce_shadow,
    raise_training_integrity,
    reference_forward_returns,
    score_selection,
    settle_cross_section,
    training_ok,
    write_cross_section,
    write_cross_section_settled,
    write_shadow,
    write_verdict,
)
from crucible.slots.inputs import InputRefusal, SlotUnservableError
from crucible.slots.rankers import MissingFeatureError, get_ranker

if TYPE_CHECKING:
    import pandas as pd

    from crucible.runner import RunContext
    from crucible.store import Store

__all__ = [
    "ARM_SERIES_SCHEMA_VERSION",
    "BASELINE_CONTROL_KIND",
    "INCUMBENT_SOURCE_FIELD",
    "SECTOR_SOURCE_MODE_FIELD",
    "MIN_ACTIVE_ARMS_FINDING_METRIC",
    "MissingArtifactError",
    "baseline_control_arm",
    "min_active_arms_finding",
    "partition_by_catalog",
    "run_grade",
    "run_produce",
]

#: The version stamped on the per-arm score series `run_grade` writes and
#: `crucible.promote.load_slot_inputs` reads (`alpha-engine-config-I10705`).
#: `v1` because the document's field set — `arm_id`, `scores`, `misses` — is
#: exactly what that loader has always parsed and what every `promote` unit
#: test has always hand-written; this names and persists an existing shape
#: rather than introducing a new one. A named constant, not a literal at the
#: write site, so the producer's stamp and the manifest `outputs` row it is
#: recorded under cannot drift apart.
ARM_SERIES_SCHEMA_VERSION = "arm_series.v1"

#: `alpha-engine-config-I9759` / `-I10687`: §10.1's control kind that stands
#: in for an ABSENT incumbent. The null control is pure noise by
#: construction, so "beats the null control on the configured evidence over
#: `promote_min_weeks` paired weeks" is the same statement the incumbent
#: comparison makes, against the only baseline a slot with no champion has.
#: The PLANTED control is never this: it reads next-period returns, so an arm
#: that merely beat it would still be a look-ahead-relative measurement.
BASELINE_CONTROL_KIND = "null"

#: `alpha-engine-config-I10687`: the crucible-owned key recording WHERE the
#: incumbent this cycle decided against came from — the champion pointer, a
#: substituted §10.1 baseline, or nothing at all. Written onto the
#: `arena_cycle` payload and onto the grade job's own manifest as a metric,
#: because "this slot graded against noise" and "this slot graded against a
#: seated champion" are two different readings of the same `decided` status
#: and no reader should have to recover the difference from an arm id.
INCUMBENT_SOURCE_FIELD = "incumbent_source"

#: `alpha-engine-config-I10733`, Brian's ruling (a). The crucible-owned key on
#: `arena_cycle` and on each `scores/` series document naming, per arm, which
#: SCORED dates rest on which sector source mode — read back off the feature
#: row each day's shadow was produced from. Values: the mode the feature layer's
#: `sector_earliest_snapshot_backfill_raw` encodes (`point_in_time`,
#: `earliest_snapshot_backfill`),
#: `sector_unmeasured` when that session's sector group measured nothing, and
#: `unrecorded` when the shadow names no feature version or its layer predates
#: the column. The same modes ride on `ArmSeries.lineage` under this name, so
#: the library's ladders carry them too.
SECTOR_SOURCE_MODE_FIELD = "sector_source_mode"
SECTOR_MODE_UNMEASURED = "sector_unmeasured"
SECTOR_MODE_UNRECORDED = "unrecorded"

#: `alpha-engine-config-I10636`: the metric name the below-floor finding is
#: recorded under on the grade job's own manifest, so an operator reading
#: `runs/experiment.grade/{slot}/{day}/run.json` sees it beside the other
#: three control/pointer metrics rather than having to open the `arena_cycle`
#: artifact to learn the slot is unservable.
MIN_ACTIVE_ARMS_FINDING_METRIC = "min_active_arms_finding"

#: The metric one refused arm files on the producing job's manifest — the
#: same name the M slot uses (`crucible.slots.model.ARM_REFUSED_METRIC`),
#: restated here rather than imported because `crucible.slots.model` pulls
#: the fitting stack in and this module is on the U/R path that must not.
#: `tests/test_cycle_refuses_by_name.py` pins the two names equal.
ARM_REFUSED_METRIC = "arm_refused_at_registration"


class MissingArtifactError(RuntimeError):
    """A required upstream artifact is absent. Its exact key is in the message.

    Never a fabricated substitute and never a skip: the operator's next
    action is to produce that key, and a reason that does not name it sends
    them looking.
    """


def min_active_arms_finding(slot_spec: Any, promotable: Sequence[str]) -> dict[str, Any]:
    """§10.1 / `alpha-engine-config-I10636`: the floor, read against REAL arms.

    `ArenaCycle.active_arms` (the library's own field) deliberately still
    counts controls — that is documented at `nousergon_lib.arena.engine
    .evaluate_retirements` as correct, since the register's own
    ``active_arms()`` must report every live arm. Comparing that count to
    ``min_active_arms`` is exactly the defect this issue measured: the two
    control arms every slot carries make a one-real-arm slot read as three
    against a floor of three.

    ``promotable`` is the caller's own `crucible.slots.promotable_arms(...)`
    result — controls already excluded — so this function forks no shape of
    the library's; it only compares a count the library never compares.

    Rendered unconditionally, with an explicit status, so a healthy slot and
    an unmeasured one are never the same reading (principles §7): a reader of
    the `arena_cycle` artifact or the grade job's manifest sees ``OK`` or
    ``BELOW_FLOOR``, never silence.
    """
    floor = int(slot_spec.min_active_arms)
    count = len(promotable)
    below = count < floor
    return {
        "status": "BELOW_FLOOR" if below else "OK",
        "min_active_arms": floor,
        "promotable_arm_count": count,
        "promotable_arms": list(promotable),
        "reason": (
            f"{count} promotable arm(s) (controls excluded, §10.1) against a floor of "
            f"{floor}; a slot below its floor produces zero comparisons"
            if below
            else f"{count} promotable arm(s) (controls excluded, §10.1) meets the floor of {floor}"
        ),
    }


def _read_features(store: Store, version: str, trading_day: dt.date) -> pd.DataFrame:
    key = features_key(version, trading_day.isoformat())
    if not store.exists(key):
        raise MissingArtifactError(
            f"the feature layer is absent at {key}. U and R read this artifact and "
            "nothing else (§10.4); compile it with:\n"
            f"    crucible data.daily --date {trading_day}"
        )
    return read_features(store.get_bytes(key))


def _sector_source_mode(store: Store, feature_version: str | None, day: str) -> str:
    """The sector source mode the feature row behind one scored date carries.

    Read from the feature layer, not from the shadow: the row is the one
    place the mode is written, so the grade cannot disagree with it.
    """
    if not feature_version:
        return SECTOR_MODE_UNRECORDED
    key = features_key(feature_version, day)
    if not store.exists(key):
        # Not swallowed: an absent layer is recorded as `unrecorded` on the
        # arm's series and the arena_cycle artifact (SECTOR_SOURCE_MODE_FIELD).
        return SECTOR_MODE_UNRECORDED
    frame = read_features(store.get_bytes(key))
    column = "sector_earliest_snapshot_backfill_raw"
    if column not in frame.columns:
        return SECTOR_MODE_UNRECORDED
    values = {float(v) for v in frame[column].dropna().unique()}
    if not values:
        return SECTOR_MODE_UNMEASURED
    if len(values) != 1 or not values <= {0.0, 1.0}:
        raise MissingArtifactError(
            f"{key} carries {column} values {sorted(values)}; the column is one 0/1 value "
            "per session by construction, so this layer is corrupt"
        )
    return EARLIEST_SNAPSHOT_BACKFILL_MODE if values.pop() == 1.0 else POINT_IN_TIME_MODE


def _sector_mode_dates(
    sector_modes_by_arm: dict[str, dict[str, list[str]]], arm_id: str
) -> dict[str, list[str]]:
    """mode -> sorted scored dates for one arm; `{}` for an arm with no shadow
    scored here (a control, or a caller-supplied S series)."""
    return {
        mode: sorted(days) for mode, days in sorted(sector_modes_by_arm.get(arm_id, {}).items())
    }


def _read_panel(store: Store, trading_day: dt.date) -> pd.DataFrame:
    import io

    import pandas as pd

    key = data_panel_key(trading_day.isoformat())
    if not store.exists(key):
        raise MissingArtifactError(
            f"the price panel is absent at {key}; forward returns are computed from it. "
            f"Compile it with:\n    crucible data.daily --date {trading_day}"
        )
    return pd.read_parquet(io.BytesIO(store.get_bytes(key)))


def _incumbent(store: Store, slot: str) -> str | None:
    key = champion_key(slot)
    if not store.exists(key):
        return None
    return load_store_document(store, key).get("arm_id")


def baseline_control_arm(slot_spec: Any, series_by_arm: dict[str, Any]) -> str | None:
    """The slot's scored :data:`BASELINE_CONTROL_KIND` control arm, if any.

    Matched on the control's KIND rather than on the register's ``control``
    flag, because the flag says *whether* an arm is a control and this needs
    *which kind* — the register records the first and not the second, and
    the planted control must never stand in as a baseline.

    Addressed by the same ``{slot}:{name}:{spec_hash}`` identity every other
    arm carries (:func:`crucible.slots.arms.control_specs` derives it), so
    the id returned is one `series_by_arm` and the register both know.

    ``None`` when the slot registers no null control or it was not scored
    this cycle. The caller does NOT fall back: a slot with nothing scored yet
    is not an error, it is an empty slot, and the engine already answers that
    with `unservable` and a stated reason — a verdict-backed non-promotion
    that writes no pointer. The refusal belongs at the one place a champion
    would otherwise be SEATED on no evidence, which is
    `crucible.promote._write_pointer_if_moved`'s `bootstrap` branch.
    """
    for spec in control_specs(slot_spec):
        if spec.control_kind == BASELINE_CONTROL_KIND and spec.arm_id in series_by_arm:
            return spec.arm_id
    return None


def _shadow_dates(store: Store, arm_id: str) -> list[str]:
    """Every trading day this arm has a shadow for, ascending.

    Listed from the store rather than derived from a date range: an arm
    registered mid-window has no shadow before its registration, and
    inventing the dates would turn its absence into a run of misses.
    """
    prefix = experiments_prefix(arm_id)
    days = []
    for key in store.list_keys(prefix):
        if key.endswith("/shadow.json"):
            days.append(key[len(prefix) :].split("/", 1)[0])
    return sorted(days)


def partition_by_catalog(
    specs: Sequence[Any], *, catalog_columns: Sequence[str]
) -> tuple[list[Any], list[InputRefusal]]:
    """Split ``specs`` into the arms this feature layer can rank and those it
    declares no producer for — PER ARM, as values, never an exception.

    The U/R half of `alpha-engine-config-I9955` (the M slot's
    :func:`crucible.slots.inputs.partition_producible`). Two absences that
    must never render alike, and the catalogue is what tells them apart:

    * a ranker column the feature CATALOG **never declared** — `predicted_alpha_ratio`
      (the M slot materialises it, phase 3) or an LLM-derived rating (phase 5).
      The arm's own recipe says it "refuses BY NAME until then". That is a
      refusal at registration: the arm does not register this cycle, its
      siblings run, and the refusal reaches the manifest as
      :data:`ARM_REFUSED_METRIC` naming the arm and the column. Measured
      2026-09-05 on the first phase-1 replay arc (weekly@2026-08-07): the
      predictor-ranked R arm raised `TrainingIntegrityError` and took the
      whole R slot — and the arc — down, though the catalogue had never
      promised the column;
    * a column the catalogue DOES declare but today's frame lacks — a
      compromised input. That stays :class:`MissingFeatureError` →
      `TrainingIntegrityError` inside the produce loop, slot-wide, exactly as
      plan §4.4 and the 2026-08-29 ruling require.

    An arm naming an unknown ranker still raises from :func:`get_ranker`: a
    recipe nothing can run is malformed, not refused.
    """
    produced = set(catalog_columns)
    producible: list[Any] = []
    refused: list[InputRefusal] = []
    for spec in specs:
        ranker = get_ranker(spec.ranker)
        undeclared = tuple(c for c in ranker.reads if c not in produced)
        if not undeclared:
            producible.append(spec)
            continue
        refused.append(
            InputRefusal(
                arm=spec.name,
                unresolvable=undeclared,
                reason=(
                    f"arm {spec.name!r} ranks with {ranker.name!r}, which reads "
                    f"{list(undeclared)}; the feature catalogue declares no producer for "
                    "them, so this is not a compromised input but a column another slot "
                    "materialises later (predictions: the M slot, phase 3; LLM ratings: "
                    "phase 5). Refused BY NAME at registration; the slot's other arms run."
                ),
            )
        )
    return producible, refused


def _refusal_metric(slot: str, refusal: InputRefusal) -> dict[str, Any]:
    return {
        "name": ARM_REFUSED_METRIC,
        "module": f"crucible.slots.{slot}",
        "metric_type": "count",
        "value": float(len(refusal.unresolvable)),
        "unit": "inputs",
        "n_floor": 1,
        "status": "unservable",
        "status_reason": refusal.reason,
        "source_path": f"strategy/current/arms/{slot}/{refusal.arm}.yaml",
        "last_updated_utc": _utc_now(),
    }


def run_produce(
    ctx: RunContext,
    *,
    slot: str,
    settings: Settings,
    feed_key_for: Any,
    arm_name: str | None = None,
    feature_version: str = DEFAULT_FEATURE_VERSION,
) -> dict[str, Any]:
    """Run every registered arm's recipe for one trading day and write the shadows."""
    slot_spec = get_slot(slot)
    trading_day = ctx.trading_day
    assert_trading_day(trading_day, context=f"experiment.run --slot {slot} --date {trading_day}")

    features = _read_features(ctx.store, feature_version, trading_day)
    ctx.record_input(
        features_key(feature_version, trading_day.isoformat()),
        ctx.store.get_bytes(features_key(feature_version, trading_day.isoformat())),
        schema_version="features.v1",
    )

    specs = load_arm_specs(slot, store=ctx.store, strategy_dir=settings.strategy_dir)
    if arm_name is not None:
        # A bare name or a registered `{slot}:{name}:{spec_hash}` id, resolved
        # through the one parser that knows both shapes. `experiment.new`
        # PRINTS ids, so the id is what an operator has in the terminal when
        # they narrow the next command to the arm they just registered; a
        # matcher that compared it to `spec.name` refused every one of them
        # with "no arm named" — a selector that only accepts the form nobody
        # is holding. `name_component` RAISES on a string that is neither
        # shape, so a malformed selector is still a refusal, not a silent
        # no-match.
        selector = name_component(arm_name)
        specs = [s for s in specs if s.name == selector]
        if not specs:
            raise MissingArtifactError(
                f"no arm named {selector!r} in slot {slot!r}. Producing nothing and "
                "exiting 0 would be indistinguishable from an arm that ran and selected "
                "nothing."
            )

    from crucible.features import CATALOG  # noqa: PLC0415 - one call site, keeps import light

    specs, refused = partition_by_catalog(specs, catalog_columns=[f.name for f in CATALOG])
    for refusal in refused:
        ctx.record_metric(_refusal_metric(slot, refusal))
    if refused and not specs:
        # Every arm refused: the slot can serve nothing, and that PAGES
        # through the ordinary failed-manifest path (plan §7 `unservable`).
        raise SlotUnservableError(tuple(refused))

    register = read_register(ctx.store, slot)
    register, _ = register_arms(register, specs + control_specs(slot_spec))
    write_register(ctx.store, slot, register)

    produced: list[ShadowSelection] = []
    for spec in specs:
        try:
            shadow = produce_shadow(spec, features, trading_day, feature_version=feature_version)
            # shadow.v2 (alpha-engine-config-I9778): the WHOLE ranked
            # cross-section, not only the top-N `shadow.v1` selection above.
            # Produced from the same `features` frame and the same recipe as
            # the selection it sits beside, so a rank IC computed from it
            # later is a rank IC over what this arm actually saw on this
            # day — never a second, drifted ranking.
            cross_section: ScoredCrossSection | None = produce_cross_section(
                spec, features, trading_day
            )
        except MissingFeatureError as exc:
            # plan §4.4: a defective input fails the whole slot's run. It is
            # never a miss — "this arm had nothing to say" and "this arm's
            # inputs were broken" must not render identically.
            raise_training_integrity(spec.arm_id, exc)
        else:
            write_shadow(ctx.store, shadow)
            ctx.record_output(
                shadow_key(shadow.arm_id, shadow.trading_day),
                json.dumps(shadow.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
                schema_version="shadow.v1",
            )
            write_cross_section(ctx.store, cross_section)
            ctx.record_output(
                cross_section_key(cross_section.arm_id, cross_section.trading_day),
                json.dumps(cross_section.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
                schema_version="cross_section.v2",
            )
            produced.append(shadow)

    champion = _incumbent(ctx.store, slot)
    feed_written = None
    if champion is not None:
        served = next((s for s in produced if s.arm_id == champion), None)
        if served is None:
            raise MissingArtifactError(
                f"the champion pointer for slot {slot!r} names {champion!r}, which produced "
                "no shadow this cycle. The serving path resolves the pointer — it never "
                "imports a ranking function directly — so a pointer to an arm that did "
                "not produce means production has no feed today."
            )
        feed_written = feed_key_for(trading_day.isoformat())
        ctx.record_output(
            feed_written,
            json.dumps(
                {
                    "schema_version": "feed.v1",
                    "slot": slot,
                    "trading_day": trading_day.isoformat(),
                    "champion": champion,
                    "members": list(served.selection),
                },
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
            schema_version="feed.v1",
        )

    ctx.record_rows(rows_in=int(len(features)), rows_out=len(produced))
    ctx.record_metric(
        {
            "name": "arms_produced",
            "module": f"crucible.slots.{slot}",
            "metric_type": "count",
            "value": float(len(produced)),
            "unit": "arms",
            "n_floor": 1,
            "status": "OK",
            "status_reason": (
                f"slot {slot}: {len(produced)} registered arm(s) produced a shadow for "
                f"{trading_day}; population {len(features)} names"
            ),
            # Not `experiments_prefix(arm_id)`: this metric is about every arm
            # PRODUCED this cycle, not one arm — the `*` stands in for "any of
            # them", and `crucible.keys` has no per-arm-set wildcard shape to
            # call (alpha-engine-config-I9852). Display-only source_path, not
            # a key any store call reads.
            "source_path": f"experiments/*/{trading_day.isoformat()}/shadow.json",
            "last_updated_utc": _utc_now(),
        }
    )
    return {
        "slot": slot,
        "trading_day": trading_day.isoformat(),
        "arms": [s.arm_id for s in produced],
        "refused": [{"arm": r.arm, "unresolvable": list(r.unresolvable)} for r in refused],
        "champion": champion,
        "feed_key": feed_written,
        "feature_version": feature_version,
    }


def run_grade(
    ctx: RunContext,
    *,
    slot: str,
    settings: Settings,
    horizon_trading_days: int = DEFAULT_HORIZON_TRADING_DAYS,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    specs: Sequence[Any] | None = None,
    preconditions: dict[str, list[ServingPrecondition]] | None = None,
    series: dict[str, ArmSeries] | None = None,
    settled_dates: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score every settled shadow, verify the controls, run the cycle, write it.

    Four keyword seams, all additive and all defaulting to the U/R behaviour
    this function has always had. Each exists because one slot states a fact
    this engine cannot derive — never because that slot grades through a
    second copy of the engine (plan §4.4, "four slots, one engine").

    ``specs`` is the slot's loaded recipe set. It defaults to
    :func:`crucible.slots.arms.load_arm_specs`, which serves U and R; M and S
    supply their own, because their recipes are read by their own loaders and
    `load_arm_specs` refuses both slots BY NAME
    (`crucible.slots.arms.FOREIGN_RECIPE_LOADERS`). A parameter rather than a
    loader table here: this module must not import `crucible.slots.model` or
    `crucible.slots.strategy`, which would pull the fitting stack and the
    portfolio solver onto the U/R path. Only two things are read off a spec —
    what to register, and ``params['top_n']`` for the count-matched controls —
    so any recipe type carrying those two facts grades through this one engine
    rather than through a second copy of it (`alpha-engine-config-I9957`).

    ``preconditions`` are per-arm SERVING preconditions the caller has already
    EVALUATED (policy §5.3: supplied to the engine as evaluated results; the
    engine does not compute them and must not be given a default). The
    control-arm exclusion below is merged into whatever the caller supplied,
    never replaced by it: §10.1 is the harness's rule, not a slot's, and a
    caller that passed a precondition for a control must not be able to
    displace it.

    ``series`` and ``settled_dates`` are the S slot's seam, and they travel
    together (`alpha-engine-config-I10512`). U, R and M are SELECTION-shaped:
    each writes a shadow on the decision date and its score is that
    selection's realized excess return, which the loop below computes. An S
    arm's score is not a selection's return at all — it is its realized
    book's return against SPY net of the cost the engine charged, produced by
    `crucible.slots.strategy.grade_arm` over a book `crucible.portfolio`
    constructed. Re-deriving that here would be a second portfolio engine, so
    the S job hands the finished :class:`~nousergon_lib.arena.window.ArmSeries`
    over and names the settled decision dates the CONTROLS must be scored on
    — which is the one thing the shadow loop would otherwise have supplied.
    Everything after remains shared: the register, both halves of the control
    battery, the pointer decision, the `arena_cycle` artifact and the trial
    ledger. Supplying one without the other is refused rather than defaulted:
    a caller-supplied series with no dates would score the controls on nothing
    and publish a cycle whose grader was never checked.
    """
    if (series is None) != (settled_dates is None):
        raise ValueError(
            "`series` and `settled_dates` are supplied together or not at all; got "
            f"series={'set' if series is not None else 'None'}, "
            f"settled_dates={'set' if settled_dates is not None else 'None'}. A "
            "caller-supplied series with no settled dates would score the slot's "
            "controls on no date at all, and §10.1's whole point is that a cycle "
            "whose grader was not checked has void verdicts."
        )
    slot_spec = get_slot(slot)
    as_of = ctx.trading_day
    assert_trading_day(as_of, context=f"experiment.grade {slot} --date {as_of}")

    panel = _read_panel(ctx.store, as_of)
    ctx.record_input(
        data_panel_key(as_of.isoformat()),
        ctx.store.get_bytes(data_panel_key(as_of.isoformat())),
        schema_version="panel.v1",
    )

    loaded_specs = (
        list(specs)
        if specs is not None
        else load_arm_specs(slot, store=ctx.store, strategy_dir=settings.strategy_dir)
    )
    controls = control_specs(slot_spec)
    register = read_register(ctx.store, slot)
    register, _ = register_arms(register, loaded_specs + controls)
    write_register(ctx.store, slot, register)

    # Kind -> REGISTERED arm id. A control is addressed by the same
    # `{slot}:{name}:{spec_hash}` identity every other arm carries; using its
    # bare name would look up a series no register row backs.
    control_by_kind = {c.control_kind: c.arm_id for c in controls}
    control_ids = set(control_by_kind.values())
    scored_arms = register.scored_arms(as_of.isoformat(), slot_spec.retired_trailing_cycles)

    # A supplied series for an arm the register does not score is a caller
    # error, not a silent extra row on the cycle: `run_cycle` would carry an
    # arm no register row backs, and its verdicts would speak about an arm the
    # retirement and cap math has never heard of.
    unregistered = sorted(set(series or {}) - set(scored_arms))
    if unregistered:
        raise MissingArtifactError(
            f"slot {slot!r} was supplied a graded series for {unregistered}, which the "
            f"register does not score as of {as_of}. An arm scored without a register "
            "row is policy §3's defect exactly."
        )

    # One returns cache per settled decision date, shared by every arm: the
    # forward return of a ticker from date d does not depend on who picked it,
    # and recomputing it per arm is how two arms end up scored against two
    # slightly different benchmarks.
    returns_cache: dict[str, ForwardReturnWindow] = {}
    unsettled_days: dict[str, str] = {}
    verdicts: dict[str, dict[str, float]] = {}
    # `alpha-engine-config-I9963`, deliverable 3. The DISTINCT feature-layer
    # versions each arm's SCORED dates were produced under, read back off each
    # day's own shadow rather than taken from this run's `feature_version`
    # argument. The two are not the same fact: the argument is the version
    # resolved for the GRADE date, while a day scored here may have been
    # produced weeks ago under whatever the catalogue hashed to then, and
    # attaching the grade-date version to it would be a fabrication.
    #
    # A shadow written before `feature_version` was recorded contributes
    # NOTHING rather than a placeholder — the dimension is simply absent for
    # an arm whose whole series predates the field, which is the honest
    # reading and the one `ArmSeries` refuses to let be an empty claim.
    lineage_by_arm: dict[str, set[str]] = {}
    # `alpha-engine-config-I10733`: arm -> mode -> scored dates.
    sector_modes_by_arm: dict[str, dict[str, list[str]]] = {}
    sector_mode_cache: dict[tuple[str | None, str], str] = {}
    unsettled: dict[str, list[str]] = {}
    misses: dict[str, list[str]] = {}
    label_control: dict[str, dict[str, Any]] = {}

    def _returns_for(day: str) -> ForwardReturnWindow | None:
        """The settled forward-return window from ``day``, or None if unsettled.

        Cached BOTH ways. An unsettled date is remembered as unsettled, so a
        second arm holding a shadow for the same day does not re-derive the
        same refusal — and, more to the point, cannot land on a different
        answer than the first arm did. Two arms scored against two different
        benchmarks for one date is the defect this cache exists to preclude.

        **The label control runs here, once per settled date, before any arm
        is scored against the date** (§10.1). The planted/null pair cannot
        see this class: both controls are generated from and scored against
        this very mapping, so a label defect moves them identically and their
        margin survives it. `assert_label_control` recomputes the labels
        through a second construction and raises
        :class:`~crucible.slots.grading.GraderControlError` when the two
        disagree or when the span measured is not the span declared — which
        is the reproduced defect: a forced 5-session horizon published a
        clean cycle while every verdict claimed 21.
        """
        if day in returns_cache:
            return returns_cache[day]
        if day in unsettled_days:
            return None
        try:
            window = forward_returns(
                panel,
                start=dt.date.fromisoformat(day),
                horizon_trading_days=horizon_trading_days,
            )
        except ValueError as exc:
            # The horizon has not settled. NOT a miss and NOT a zero: the date
            # is not yet a measurement, and it enters the series on the first
            # cycle after it settles.
            unsettled_days[day] = str(exc)
            return None
        label_control[day] = assert_label_control(
            window,
            reference_forward_returns(
                panel,
                start=dt.date.fromisoformat(day),
                horizon_trading_days=horizon_trading_days,
            ),
            slot=slot,
            declared_horizon_trading_days=horizon_trading_days,
        )
        returns_cache[day] = window
        return window

    # The S seam: the caller has already produced every real arm's series, so
    # there is no shadow to score — only the settled dates the controls need.
    # `_returns_for` is still what resolves them, so the label control (§10.1)
    # runs over exactly the dates this cycle publishes, on every slot alike.
    for day in settled_dates or ():
        _returns_for(day)

    for arm_id in scored_arms:
        if arm_id in control_ids:
            continue
        # Every real arm the register says to score is SUPPLIED a series, even
        # when it has nothing in it — `run_cycle` requires one per arm and the
        # pairing is per comparison, so an arm with nothing to say cannot null
        # another arm's figure (I9745, closed by construction).
        verdicts.setdefault(arm_id, {})
        if series is not None:
            continue
        for day in _shadow_dates(ctx.store, arm_id):
            window = _returns_for(day)
            if window is None:
                unsettled.setdefault(arm_id, []).append(day)
                continue
            shadow = load_store_document(ctx.store, shadow_key(arm_id, day))
            try:
                score, detail = score_selection(
                    tuple(shadow["selection"]), tuple(shadow["population"]), window.returns
                )
            except SelectionMissError:
                # plan §4.4 / policy §3: "a cycle in which an arm legitimately
                # selects nothing is a MISS", and a miss is data. Every name
                # this arm picked delisted or was halted, while the population
                # it drew from is intact and every other arm scores normally.
                #
                # This is the ONLY swallow in this loop and it is not a
                # degrade: the failure mode absorbed is "one arm's picks are
                # all unscoreable on one date", the date is recorded on the
                # `misses` surface of this run's manifest and on the
                # `arm_misses` metric below, and it stays OUT of the arm's
                # series so a miss can never render as a zero. What is NOT
                # absorbed is one line down.
                misses.setdefault(arm_id, []).append(day)
                continue
            except PopulationIntegrityError as exc:
                # The other half of the distinction, and it still fails the
                # slot. The benchmark could not be formed, every arm is scored
                # against it, so the cycle's shared inputs are compromised.
                raise_training_integrity(arm_id, exc)
            else:
                verdicts[arm_id][day] = score
                produced_under = shadow.get("feature_version")
                if produced_under:
                    lineage_by_arm.setdefault(arm_id, set()).add(str(produced_under))
                cache_key = (str(produced_under) if produced_under else None, day)
                if cache_key not in sector_mode_cache:
                    sector_mode_cache[cache_key] = _sector_source_mode(ctx.store, *cache_key)
                sector_modes_by_arm.setdefault(arm_id, {}).setdefault(
                    sector_mode_cache[cache_key], []
                ).append(day)
                # Claimed as an OUTPUT, not only written: `crucible explain
                # <verdict key>` resolves a key through the manifest that
                # claims it, and until 2026-09-05 no manifest claimed a
                # verdict — the first `explain` against a real verdict on the
                # box read "neither a run_id nor a key any run claims" while
                # the verdict sat in the store (alpha-engine-config-I9757,
                # `explain_walks_a_verdict`).
                verdict_payload = write_verdict(
                    ctx.store,
                    arm_id=arm_id,
                    trading_day=day,
                    slot=slot,
                    score=score,
                    window=window,
                    benchmark=slot_spec.benchmark,
                    detail=detail,
                    control=False,
                )
                ctx.record_output(
                    verdict_key(arm_id, day), verdict_payload, schema_version="verdict.v1"
                )
                # shadow.v2 settlement (alpha-engine-config-I9778): join the
                # produce-time cross-section against the SAME `window.returns`
                # the verdict above was scored against, so the rank IC
                # `crucible.report` later reduces this into can only ever
                # agree with what the verdict for this date says settled.
                # Guarded rather than required: a shadow produced before this
                # artifact existed has no cross_section.json, and that is a
                # migration date, not a defect — `crucible report`'s rank IC
                # row reads n_samples off what settlement actually wrote.
                cs_key = cross_section_key(arm_id, day)
                if ctx.store.exists(cs_key):
                    cross_section_doc = load_store_document(ctx.store, cs_key)
                    settled = settle_cross_section(
                        cross_section_doc,
                        returns=window.returns,
                        horizon_trading_days=window.horizon_trading_days,
                        settled_on=window.end,
                    )
                    write_cross_section_settled(
                        ctx.store, arm_id=arm_id, trading_day=day, document=settled
                    )

    settled_days = sorted(d for d, w in returns_cache.items() if w.returns)
    if not settled_days:
        raise MissingArtifactError(
            f"slot {slot!r} has no shadow whose {horizon_trading_days}-session horizon has "
            f"settled by {as_of}. There is nothing to grade — which is a state, not a "
            "verdict, so this run FAILS rather than publishing an empty cycle. Produce "
            "shadows at least that many sessions back first:\n"
            f"    crucible experiment.run --slot {slot} --date <earlier trading day>"
        )

    # Controls are produced HERE, at grade time, on the same settled dates the
    # real arms were scored on. A planted edge is a look-ahead by
    # construction: generating it in the produce path would put a look-ahead
    # artifact on the real-time write path (§10.1).
    for control in controls:
        verdicts.setdefault(control.arm_id, {})
        for day in settled_days:
            window = returns_cache[day]
            returns = window.returns
            top_n = _control_top_n(loaded_specs)
            selection = control_selection(
                control.control_kind,
                returns,
                top_n=top_n,
                seed=int(day.replace("-", "")),
            )
            # No miss handler here, deliberately. A control draws its picks
            # FROM the settled returns, so every pick is settled by
            # construction; a `SelectionMissError` on a control would be a
            # defect in the harness itself and must reach the operator as a
            # failed run rather than as a control that quietly missed.
            score, detail = score_selection(selection, tuple(sorted(returns)), returns)
            verdicts[control.arm_id][day] = score
            detail["control_kind"] = control.control_kind
            control_payload = write_verdict(
                ctx.store,
                arm_id=control.arm_id,
                trading_day=day,
                slot=slot,
                score=score,
                window=window,
                benchmark=slot_spec.benchmark,
                detail=detail,
                control=True,
            )
            ctx.record_output(
                verdict_key(control.arm_id, day), control_payload, schema_version="verdict.v1"
            )

    # A caller-supplied series WINS for the arm it names and is never merged
    # into: `series` is the whole of that arm's scored history on its own
    # axis, and folding a selection-shaped score into it would put two
    # different measurements in one series under one name.
    supplied = dict(series or {})
    series_by_arm: dict[str, ArmSeries] = {
        arm_id: supplied[arm_id]
        if arm_id in supplied
        else ArmSeries(
            arm_id=arm_id,
            scores=scores,
            # A control has no shadow — it is generated here, at grade time,
            # from the settled returns — so it contributes no dimension and
            # its lineage is `{}`. That is "this arm declares none", which is
            # exactly right and is distinct from a missing key.
            lineage={
                **(
                    {"feature_version": tuple(sorted(lineage_by_arm[arm_id]))}
                    if lineage_by_arm.get(arm_id)
                    else {}
                ),
                **(
                    {SECTOR_SOURCE_MODE_FIELD: tuple(sorted(sector_modes_by_arm[arm_id]))}
                    if sector_modes_by_arm.get(arm_id)
                    else {}
                ),
            },
        )
        for arm_id, scores in sorted({**{a: {} for a in supplied}, **verdicts}.items())
    }
    # An arm with NO settled score is still SUPPLIED, carrying an empty
    # series. Deliberate on both counts: `run_cycle` requires a series for
    # every arm the register says to score, and the pairing happens per
    # comparison, so an arm with nothing to say cannot null another arm's
    # figure. That is I9745 closed by construction rather than by a rule.

    # ── Persist the series, because the consumer reads it from the store ──
    #
    # `alpha-engine-config-I10705`. `crucible.keys.arm_series_key` declares
    # this document "as produced by `experiment.grade`" and
    # `crucible.promote.load_slot_inputs` reads it for EVERY arm in the
    # register — and until this line nothing under `crucible/` wrote it. The
    # gap was invisible from the unit suite because every `promote` test
    # hand-writes the series before calling the loader; it surfaced the first
    # time the two jobs ran in sequence against a real store (the integration
    # tier, measured 2026-09-14): a grade that exited `ok` and wrote eleven
    # verdicts left `scores/` empty, and `crucible promote` raised
    # `KeyError: ... is registered in slot 'u' but has no series`.
    #
    # Written HERE, from the same `series_by_arm` the engine is about to be
    # handed, rather than reconstructed by `promote` from the verdict
    # documents: one producer, one artifact, and the score `promote` acts on
    # is byte-identical to the score the cycle decided on. Every arm the
    # register names gets a document, including an arm whose series is empty
    # — an empty `scores` map is "this arm has nothing settled to say", which
    # the loader must be able to read as such and cannot read from an absent
    # key.
    for arm_id, arm_series in series_by_arm.items():
        ctx.record_output(
            arm_series_key(slot, arm_id),
            json.dumps(
                {
                    "schema_version": ARM_SERIES_SCHEMA_VERSION,
                    "arm_id": arm_series.arm_id,
                    "scores": {day: float(score) for day, score in arm_series.scores.items()},
                    "misses": sorted(arm_series.misses or ()),
                    SECTOR_SOURCE_MODE_FIELD: _sector_mode_dates(sector_modes_by_arm, arm_id),
                },
                indent=2,
                sort_keys=True,
            ).encode(),
            schema_version=ARM_SERIES_SCHEMA_VERSION,
        )

    # §10.1: a control never serves. Expressed as a SERVING PRECONDITION —
    # the engine's own mechanism for "this arm may not take the pointer" —
    # rather than as an assumption that the engine will not pick it. The
    # planted control ranks on the realized forward return, so it would win
    # the pointer on evidence and be a look-ahead in production; the
    # precondition puts the refusal, and its reason, into the cycle artifact
    # where an operator can read it.
    #
    # Merged ONTO whatever the caller supplied rather than replacing it: a
    # slot may add its own evaluated preconditions (M supplies the §5.3
    # behavioural veto; S supplies the contamination attestation), and the
    # control exclusion must survive that — a caller able to displace it is a
    # look-ahead arm one dict key away from the pointer.
    #
    # `alpha-engine-config-I9759` / `-I10687` — THE FIRST CHAMPION.
    #
    # With no incumbent the library cold-starts (§9.1): it ranks the eligible
    # arms and takes the top one with `status="bootstrap"`, on no evidence at
    # all. That pointer is written with `promotion_source="bootstrap"`, which
    # the §6 phase-3 clause rejects by name — "a bootstrap or an operator
    # revert is not a promotion the system won". So a slot with no champion
    # could never win one: its FIRST pointer was, by construction, the one
    # kind of pointer that does not count, and every subsequent promotion
    # would be measured against an arm nothing had ever compared.
    #
    # The fix is not a new promotion source. §10.1 already registers a NULL
    # control in every slot, every cycle, precisely so that "is this arm
    # better than noise?" is a measured question — so with no incumbent the
    # null control STANDS IN as the baseline and the engine's ordinary
    # `decided`/`held` path runs unchanged: the same paired window, the same
    # `promote_min_weeks`, the same `promote_evidence`. A first champion is
    # then won on evidence or not won at all, and a cycle that wins nothing
    # files a `held` decision with a stated reason — the verdict-backed
    # non-promotion the same clause accepts.
    #
    # It lives HERE, in the one writer that computes the cycle, rather than
    # in `crucible.promote`: promote acts on the cycle this function already
    # decided (`alpha-engine-config-I10679`), so a substitution applied there
    # would be a second cycle computation and the two jobs could disagree.
    champion = _incumbent(ctx.store, slot)
    baseline = baseline_control_arm(slot_spec, series_by_arm) if champion is None else None
    incumbent = champion if baseline is None else baseline
    if champion is not None:
        incumbent_source = {
            "arm_id": champion,
            "source": "champion_pointer",
            "baseline_control_kind": None,
            "reason": (
                f"slot {slot!r} graded against its seated champion {champion}, read "
                f"from {champion_key(slot)}"
            ),
        }
    elif baseline is not None:
        incumbent_source = {
            "arm_id": baseline,
            "source": "baseline_control",
            "baseline_control_kind": BASELINE_CONTROL_KIND,
            "reason": (
                f"slot {slot!r} has no champion pointer; §10.1's "
                f"{BASELINE_CONTROL_KIND} control {baseline} stood in as the baseline "
                "incumbent so a first champion is won on evidence rather than "
                "cold-started on none. The baseline is "
                "exempt from the control veto because an incumbent that fails a "
                "serving precondition forces the pointer off itself; it is still "
                "barred from the promotable pool and can never be seated."
            ),
        }
    else:
        incumbent_source = {
            "arm_id": None,
            "source": "none",
            "baseline_control_kind": None,
            "reason": (
                f"slot {slot!r} has neither a champion pointer nor a scored "
                f"{BASELINE_CONTROL_KIND} control this cycle; the engine decides on "
                "its own §9.1 terms and states its reason"
            ),
        }

    evaluated: dict[str, list[ServingPrecondition]] = {
        arm_id: list(rules) for arm_id, rules in (preconditions or {}).items()
    }
    for control in controls:
        # The substituted baseline is the ONE arm this veto is not applied
        # to: it has to be ELIGIBLE to be the incumbent, because the engine
        # forces the pointer off an incumbent that fails a precondition —
        # which would turn a cold slot's first cycle into an unbarred
        # promotion of whatever ranked first. Exempting it does not make it
        # promotable: a challenger is never compared to itself, so `moved` is
        # false whenever the baseline "wins", it stays out of `promotable`
        # below, and `_write_pointer_if_moved` refuses a control outright.
        # Only the baseline is ever exempt; the planted control never is.
        if control.arm_id == baseline:
            continue
        evaluated.setdefault(control.arm_id, []).append(
            ServingPrecondition(
                name="not_a_control_arm",
                passed=False,
                reason=(
                    f"{control.control_kind} control: scored every cycle to verify the "
                    "grader, excluded from the pointer (§10.1)"
                ),
            )
        )

    cycle, control_detail = grade_slot(
        slot_spec,
        as_of=as_of,
        register=register,
        control_ids=control_by_kind,
        series_by_arm=series_by_arm,
        incumbent=incumbent,
        baseline=baseline,
        preconditions=evaluated,
        training=training_ok(list(register.active_arms())),
    )

    # §10.1: the cycle's control record is BOTH halves of the grader. The
    # planted/null margin checks the scoring half; the label control checks
    # the half that constructs the number being scored, and until it existed
    # a `verdicts void` finding could only ever come from one of the two.
    control_detail["label_control"] = {
        "dates_checked": sorted(label_control),
        "n_dates_checked": len(label_control),
        "declared_horizon_trading_days": horizon_trading_days,
        "measured_horizon_trading_days": sorted(
            {d["horizon_trading_days"] for d in label_control.values()}
        ),
        "max_relative_disagreement": max(
            (d["max_relative_disagreement"] for d in label_control.values()), default=0.0
        ),
        "per_date": {day: dict(detail) for day, detail in sorted(label_control.items())},
    }

    # The horizon EVERY verdict this run wrote actually claims, read back off
    # the measurements rather than off the argument. One value or the label
    # control would already have raised; asserted here so a future edit that
    # bypasses that control cannot quietly reintroduce a mixed-horizon run.
    measured_horizons = {w.horizon_trading_days for w in returns_cache.values()}
    if measured_horizons != {horizon_trading_days}:
        raise GraderControlError(
            f"slot {slot!r} measured horizons {sorted(measured_horizons)} while declaring "
            f"{horizon_trading_days}. A cycle whose verdicts span more than one horizon "
            "compares arms on different axes (policy §4: same benchmark and horizon "
            "across every arm in a slot), so its verdicts are void."
        )
    measured_horizon = measured_horizons.pop()

    # §10.1 as an EXCLUSION that BINDS here, not as a field this run reports.
    #
    # `promotable_arms` is the slot registry's filter and stays the shared
    # implementation of the rule. It is now REGISTER-BACKED
    # (`alpha-engine-config-I9943`): `register_arms` forwards `ArmSpec.control`
    # onto the registered `ArmRecord.control`, so `is_control_arm` reads the
    # recorded flag for any id this `register` carries rather than matching
    # the name component against `slot_spec.control_arms`. Passing `register`
    # here is the end state this call site's prior comment named as correct
    # once the registry filter could resolve registered ids itself — the
    # `not in control_ids` intersection below is now redundant with the
    # filter and is KEPT anyway as the belt-and-suspenders check the raise
    # immediately after it depends on: a control that reached `promotable`
    # despite the register-backed filter is exactly the defect
    # `GraderControlError` exists to catch, and removing the redundancy would
    # remove the second witness that catches it.
    #
    # Computed here, BEFORE the `arena_cycle` write, so the below-floor
    # finding (`alpha-engine-config-I10636`) can ride on the same artifact
    # rather than being derivable only from a later job's own return value —
    # the `active_arms` field on that artifact is the library's own and
    # deliberately still counts controls (`nousergon_lib.arena.engine
    # .evaluate_retirements`), so a reader of the artifact who does not also
    # read this finding would see 3 active arms against a floor of 3 and
    # read a one-real-arm slot as healthy.
    promotable = [
        a
        for a in promotable_arms(slot_spec, list(cycle.active_arms), register)
        if a not in control_ids
    ]
    leaked = sorted(set(promotable) & control_ids)
    if leaked:
        raise GraderControlError(
            f"control arm(s) {leaked} reached the promotion pool for slot {slot!r}. The "
            "planted control ranks on the realized forward return, so an arm of this "
            "kind in the promotable pool is a look-ahead one promotion away from "
            "production (§10.1)."
        )
    floor_finding = min_active_arms_finding(slot_spec, promotable)

    cycle_key = arena_cycle_key(slot, as_of.isoformat())
    cycle_payload = cycle.to_dict()
    # A crucible-owned key alongside the library's own fields, never a field
    # the library's `ArenaCycle` shape defines — `active_arms` above stays
    # exactly what the library wrote. Forking that shape here would pass
    # every value assertion today and silently drift the first time the
    # library gained a field of its own with this name
    # (`alpha-engine-config-I10636`; library-side tracking issue filed
    # separately since the shape is `nousergon_lib.arena`'s to own).
    cycle_payload[MIN_ACTIVE_ARMS_FINDING_METRIC] = floor_finding
    # `alpha-engine-config-I10687`: same shape, same reason — a crucible-owned
    # key beside the library's own fields. `decision.incumbent` names WHICH
    # arm; this names WHERE it came from, which is the difference between a
    # slot measured against a seated champion and one measured against noise.
    cycle_payload[INCUMBENT_SOURCE_FIELD] = incumbent_source
    # `alpha-engine-config-I10733`: which part of each arm's score rests on a
    # backfilled sector map. Same crucible-owned-key shape as the two above.
    cycle_payload[SECTOR_SOURCE_MODE_FIELD] = {
        arm_id: _sector_mode_dates(sector_modes_by_arm, arm_id) for arm_id in sorted(series_by_arm)
    }
    ctx.record_output(
        cycle_key,
        json.dumps(cycle_payload, indent=2, sort_keys=True).encode("utf-8"),
        schema_version="arena_cycle.v1",
    )

    rows = trial_rows(
        cycle,
        slot=slot,
        as_of=as_of.isoformat(),
        run_id=ctx.run_id,
        control_ids=control_ids,
        series_by_arm=series_by_arm,
        arena_cycle_key=cycle_key,
    )
    append_trials(ctx.store, rows)

    ctx.record_rows(
        rows_in=len(series_by_arm),
        rows_out=sum(len(s.scores) for s in series_by_arm.values()),
    )
    ctx.record_metric(
        {
            "name": "control_margin_ratio",
            "module": f"crucible.slots.{slot}",
            "metric_type": "control",
            "value": float(control_detail["margin_ratio"]),
            "unit": "ratio",
            "n_floor": 1,
            "status": "OK",
            "status_reason": (
                f"planted control led the null control by "
                f"{control_detail['margin_ratio']:.6f} over "
                f"{control_detail['n_paired_dates']} shared date(s); the planted arm's "
                f"signal is constructed with an IC of {control_detail['planted_ic']}"
            ),
            "source_path": cycle_key,
            "last_updated_utc": _utc_now(),
            # The horizon MEASURED, not the one requested. §4.12 wants this
            # number to be a count of trading days; I9757 wants it to be a
            # count of the trading days that were actually walked.
            "horizon_trading_days": measured_horizon,
            "baseline": 0.0,
        }
    )
    ctx.record_metric(
        {
            "name": "label_control_max_disagreement_ratio",
            "module": f"crucible.slots.{slot}",
            "metric_type": "control",
            "value": float(control_detail["label_control"]["max_relative_disagreement"]),
            "unit": "ratio",
            "n_floor": 1,
            "status": "OK",
            "status_reason": (
                f"two independent label constructions agreed on "
                f"{len(label_control)} settled date(s) over a measured horizon of "
                f"{measured_horizon} session(s); the cycle declared "
                f"{horizon_trading_days}. A disagreement, or a measured horizon other "
                "than the declared one, voids the cycle (§10.1)"
            ),
            "source_path": cycle_key,
            "last_updated_utc": _utc_now(),
            "horizon_trading_days": measured_horizon,
            "baseline": 0.0,
        }
    )
    ctx.record_metric(
        {
            # A miss is DATA, so it gets a surface. Policy §3: silent absence
            # and a genuine zero must never render identically — a miss that
            # only ever appeared as a gap in a series would be exactly that.
            "name": "arm_miss_dates",
            "module": f"crucible.slots.{slot}",
            "metric_type": "count",
            "value": float(sum(len(days) for days in misses.values())),
            "unit": "arm_dates",
            "n_floor": 0,
            "status": "OK",
            "status_reason": (
                f"{sum(len(d) for d in misses.values())} arm-date(s) across "
                f"{len(misses)} arm(s) had every selected name unscoreable while the "
                "population was intact: a miss, not a failure and not a zero. Arms: "
                f"{sorted(misses) or 'none'}"
            ),
            "source_path": cycle_key,
            "last_updated_utc": _utc_now(),
        }
    )
    ctx.record_metric(
        {
            "name": "slot_pointer_status",
            "module": f"crucible.slots.{slot}",
            "metric_type": "decision",
            "value": 1.0 if cycle.decision.moved else 0.0,
            "unit": "indicator",
            "n_floor": 1,
            "status": "OK" if cycle.decision.status in ("decided", "held", "bootstrap") else "FAIL",
            "status_reason": f"{cycle.decision.status}: {cycle.decision.reason}",
            "source_path": cycle_key,
            "last_updated_utc": _utc_now(),
        }
    )
    ctx.record_metric(
        {
            # `alpha-engine-config-I10687`: rendered unconditionally, so the
            # grade job's own manifest says whether this cycle was decided
            # against a seated champion or against §10.1's null control. A
            # `decided` status means two different things in those two cases
            # and the manifest must not render them identically (policy §3).
            "name": INCUMBENT_SOURCE_FIELD,
            "module": f"crucible.slots.{slot}",
            "metric_type": "decision",
            "value": 1.0 if incumbent_source["source"] == "baseline_control" else 0.0,
            "unit": "indicator",
            "n_floor": 0,
            "status": "OK",
            "status_reason": str(incumbent_source["reason"]),
            "source_path": cycle_key,
            "last_updated_utc": _utc_now(),
        }
    )
    ctx.record_metric(
        {
            # `alpha-engine-config-I10636`: rendered unconditionally, with an
            # explicit OK/FAIL status, so a slot stranded below its floor is
            # a reading nobody has to infer from `active_arms` still counting
            # the two controls (§10.1).
            "name": MIN_ACTIVE_ARMS_FINDING_METRIC,
            "module": f"crucible.slots.{slot}",
            "metric_type": "count",
            "value": float(floor_finding["promotable_arm_count"]),
            "unit": "arms",
            "n_floor": 0,
            "status": "FAIL" if floor_finding["status"] == "BELOW_FLOOR" else "OK",
            "status_reason": floor_finding["reason"],
            "source_path": cycle_key,
            "last_updated_utc": _utc_now(),
        }
    )

    return {
        "slot": slot,
        "as_of": as_of.isoformat(),
        "arena_cycle_key": cycle_key,
        "scored_arms": list(cycle.scored_arms),
        "active_arms": list(cycle.active_arms),
        "promotable_arms": promotable,
        MIN_ACTIVE_ARMS_FINDING_METRIC: floor_finding,
        INCUMBENT_SOURCE_FIELD: incumbent_source,
        "settled_dates": settled_days,
        "horizon_trading_days": measured_horizon,
        "unsettled": {a: sorted(d) for a, d in unsettled.items()},
        # A miss is recorded, never inferred from a gap. `unsettled` and
        # `misses` are separate keys because they are separate events: the
        # first is "not yet a measurement", the second is "measured, and this
        # arm had nothing scoreable to say" (plan §4.4).
        "misses": {a: sorted(d) for a, d in misses.items()},
        "pointer": cycle.decision.to_dict(),
        "controls": control_detail,
        "trial_rows_appended": len(rows),
        "verdict_keys": [
            verdict_key(arm, day)
            for arm, scores in sorted(verdicts.items())
            for day in sorted(scores)
        ],
    }


def _control_top_n(specs: list[Any], default: int = 10) -> int:
    """Controls select as many names as the arms they are checking.

    A control drawing a different count would be compared on a different
    axis: the count-matched benchmark is exact for any size, but a 5-name
    control and a 50-name arm have different dispersion, and the margin
    between them would partly measure that.
    """
    counts = {int(s.params.get("top_n", default)) for s in specs}
    if len(counts) == 1:
        return counts.pop()
    return max(counts) if counts else default


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
