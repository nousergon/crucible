"""The weekly arc: one declared sequence, derived from the registry.

Normative source: plan §4.1 ("control flow lives in Python and is tested with
pytest, not in a state machine"), §4.6, §9.2, and the §6.1 phase-1 gate.

`components.yaml` declares six jobs as `schedule: weekly, Saturday`, each with
a deadline and each watched for absence. Something has to run them. Before
this module the EventBridge schedule dispatched exactly one of them
(`data.weekly`) and nothing chained the rest, so five declared components
would have been absent every Saturday — pages for work no scheduler was ever
going to start. A declared cadence is not evidence of a cadence.

**The sequence is DERIVED, never listed.** A row joins the arc by declaring
`dispatch: arc`, and its position is its own deadline. A hand-written stage
list is the bug class §11.1 is about: it is written from the jobs someone
remembered, and it goes stale in the direction of running fewer of them.

**Every stage runs through `crucible.cli.main`** — the exact argv an operator
would type. Not the handler, not a copy of the dispatch table: a parallel path
can be healthy while the real one is broken, and the arc exists precisely to
be the thing the phase gate measures.

**A failed stage fails the arc, immediately.** No `continue`, no partial
success, no "the rest still ran": the downstream stages read what the failed
one was supposed to write, and running them anyway produces a report card
built on an absent input — a well-formed artifact containing nothing.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from crucible.components import Component, load_registry
from crucible.slots import SLOTS, dispatchable_slots

__all__ = [
    "ARC_JOB",
    "ARC_SLOT_JOBS",
    "ARCTIC_LIBRARY_JOBS",
    "SELECT_NEWEST_VERDICT_JOBS",
    "Stage",
    "arc_stages",
    "assert_arc_champions_producible",
    "run_arc",
]

#: The job name the arc itself runs and files its manifest under. That
#: manifest records every stage it completed as an input, which is what
#: `crucible.alerts.evaluate_absence` reads to grade an arc member against the
#: declaration in force on the day the arc ran (`alpha-engine-config-I10711`).
ARC_JOB = "weekly"

#: The arc jobs that are run once per slot rather than once. Derived from the
#: CLI's own `--slot` requirement in `tests/test_weekly.py`, so a new
#: slot-scoped job cannot join the arc and silently run for one slot.
#:
#: `promote` joined at `alpha-engine-config-I9759`. `experiment.grade` decides
#: the pointer and writes the `arena_cycle`; `promote` is what ACTS on that
#: decision, and it was dispatched by nothing — so the v2 arena had graded
#: every week since phase 1 and never once written a champion pointer or a
#: promote run. Every §6 phase-3 promotion clause therefore read UNMEASURABLE:
#: not "the slot held its pointer", but "nothing looked".
#: `experiment.register` joined at `alpha-engine-config-I10927`, at 11:00 —
#: before `experiment.run`'s 12:00, because the arm it registers is the arm
#: that stage is supposed to score. Until it did, `experiment.new` was the
#: only path from "a recipe is merged" to "an arm is scored" and it was
#: scheduled nowhere: ten recipes merged into the strategy tree had never
#: been registered, the U slot had graded the same three arms every cycle
#: since 2026-09-12, and no surface said so. The same defect this module's
#: header names for the arc, one step upstream.
ARC_SLOT_JOBS: frozenset[str] = frozenset(
    {"experiment.register", "experiment.run", "experiment.grade", "promote"}
)

#: The arc jobs that read ArcticDB directly and therefore accept
#: `--arctic-library` (`crucible.track_a._source`). Only `data.weekly` is a
#: declared arc stage today; `data.daily` is named alongside it because it
#: takes the identical flag for the identical reason and a future arc row
#: for it must not need a second edit here (`alpha-engine-config-I10633`).
ARCTIC_LIBRARY_JOBS: frozenset[str] = frozenset({"data.daily", "data.weekly"})

#: The arc jobs that carry `--select-newest-verdict` rather than a positional
#: target (`alpha-engine-config-I10858`). `explain` takes `target
#: RUN_ID|VERDICT_KEY` as an on-demand CLI job, but the arc dispatches no
#: operator holding a specific target — it names the flag that makes the
#: selection deterministic instead, so `explain_walks_a_verdict` (a ROLLING
#: phase-1 window) has a producer on a cadence shorter than its window rather
#: than depending on an operator's memory, the way its one qualifying
#: manifest (2026-08-07) did before it aged out.
SELECT_NEWEST_VERDICT_JOBS: frozenset[str] = frozenset({"explain"})


@dataclass(frozen=True)
class Stage:
    """One invocation of the arc: a job, its slot if it has one, its argv."""

    job: str
    slot: str | None
    due_at: dt.datetime

    def argv(
        self,
        *,
        trading_day: dt.date,
        store: str | None,
        run_mode: str,
        dry_run: bool = False,
        arctic_library: str | None = None,
    ) -> list[str]:
        """The exact argv an operator would type for this stage.

        ``run_mode`` is required and always emitted. Each stage re-enters
        `crucible.cli.main` as its own process-equivalent invocation and
        therefore resolves the mode independently: without it on the argv, a
        stage falls back to `$CRUCIBLE_RUN_MODE`, which means a replay arc
        (`crucible weekly --run-mode replay`) launched on a box whose
        environment declares `live` would file twelve LIVE stage manifests
        under an arc manifest that says `replay` — the false-liveness claim
        the field exists to prevent, and `_clause_replays_ok` reads the STAGE
        manifests. With the environment unset it is worse and better at once:
        every stage refuses, so the arc fails loudly at stage one.

        ``dry_run=True`` appends `--dry-run` for the same reason
        (alpha-engine-config-I9922 N1): the arc's own `--dry-run` reaches a
        stage ONLY via that stage's own argv, since there is no `args` object
        shared between this call and the stage's.

        ``arctic_library`` appends `--arctic-library <name>` for a stage in
        `ARCTIC_LIBRARY_JOBS` only (`alpha-engine-config-I10633`), mirroring
        `crucible.track_a._source`'s own additive flag: absent, a stage's argv
        is byte-for-byte unchanged and the job reads the production `universe`
        library exactly as before this parameter existed. Without this, the
        arc run inside `tests/integration/` would silently read the
        PRODUCTION library within the shared integration bucket rather than
        the dedicated one — the opposite of this tier's isolation guarantee
        (see `tests/integration/README.md`, "Not exercised, and why").
        """
        argv = [self.job, "--date", trading_day.isoformat(), "--run-mode", run_mode]
        if store:
            argv += ["--store", store]
        if self.slot:
            argv += ["--slot", self.slot]
        if arctic_library and self.job in ARCTIC_LIBRARY_JOBS:
            argv += ["--arctic-library", arctic_library]
        if self.job in SELECT_NEWEST_VERDICT_JOBS:
            argv.append("--select-newest-verdict")
        if dry_run:
            # alpha-engine-config-I9922 N1: `weekly --dry-run` used to ignore
            # the flag entirely and dispatch every stage for real. Each stage
            # runs through `crucible.cli.main` as a fresh process-in-process
            # invocation (this module's own docstring), so the ONLY way for
            # the arc's own `--dry-run` to reach a stage is to hand it back
            # down on that stage's own argv — there is no shared `args`
            # object between this call and the stage's.
            argv.append("--dry-run")
        return argv

    @property
    def label(self) -> str:
        return f"{self.job}[{self.slot}]" if self.slot else self.job


def _arc_rows(registry: dict[str, Component]) -> list[Component]:
    rows = [c for c in registry.values() if c.dispatch == "arc" and c.lifecycle == "ACTIVE"]
    if not rows:
        raise ValueError(
            "no component declares `dispatch: arc`, so the weekly arc would run "
            "nothing and report `ok`. An empty sequence is a broken registry, not "
            "a quiet week."
        )
    return rows


def arc_stages(trading_day: dt.date, registry: dict[str, Component] | None = None) -> list[Stage]:
    """The arc for ``trading_day``, in the order the deadlines already imply.

    Ordering is `(deadline, job name)`. The deadline table is the dependency
    order — `data.weekly` at 09:00 is what `experiment.run` at 12:00 reads,
    and `report` at 16:00 reduces both — so deriving the order from it means
    there is one declaration of the arc's shape rather than two that can
    disagree. The name is the tie-break, so two stages sharing a deadline
    have a deterministic order rather than a dict-insertion one.

    A slot-scoped job expands into one stage per DISPATCHABLE slot
    (`crucible.slots.dispatchable_slots`), in `SLOTS` order (u then r then m
    then s), which is the data dependency: the universe cut feeds the
    signal, the signal feeds the model, the model feeds the strategy. A slot
    whose entry points do not exist yet is not a stage: expanding over all
    of `SLOTS` made every arc fail at `experiment.run[m]` until phase 3, and
    the phase-1 gate — which derives its expected set from this function —
    unreadable for the same reason.
    """
    registry = registry or load_registry()
    slots = [slot for slot in SLOTS if slot in dispatchable_slots()]
    stages: list[Stage] = []
    for row in sorted(_arc_rows(registry), key=lambda c: (c.deadline.due_at(trading_day), c.name)):
        due = row.deadline.due_at(trading_day)
        if row.name in ARC_SLOT_JOBS:
            stages.extend(Stage(row.name, slot, due) for slot in slots)
        else:
            stages.append(Stage(row.name, None, due))
    return stages


def assert_arc_champions_producible(
    trading_day: dt.date, stages: list[Stage], *, store: str | None
) -> None:
    """Raise if any slot the arc dispatches has a champion pointer naming an arm
    the release in force cannot produce. See `crucible.slots.producibility`.

    The slots are the ones the arc actually DISPATCHES, read off ``stages`` —
    never `SLOTS`, and never a list: a slot with no stage serves nothing this
    arc, and a slot added to the arc is checked the day it joins.

    The store is opened read-only (`open_store(..., dry_run=True)`: the
    capturing wrapper has no code path to a write), from the same ``store``
    argument, or `$CRUCIBLE_STORE`, that every stage resolves; the recipe tree
    is the one `crucible.config.settings` resolves, so the check reads what
    `experiment.run` will read. It never degrades: an unreadable pointer or
    recipe tree raises here, before stage 1, rather than being reported as
    producible.
    """
    from crucible.config import settings as resolve_settings  # noqa: PLC0415 - light, one site
    from crucible.slots.producibility import (  # noqa: PLC0415 - avoids a cycle
        assert_champions_producible,
    )
    from crucible.store import open_store  # noqa: PLC0415 - avoids a cycle

    dispatched = [slot for slot in SLOTS if any(stage.slot == slot for stage in stages)]
    if not dispatched:
        return
    assert_champions_producible(
        open_store(store, dry_run=True),
        dispatched,
        strategy_dir=resolve_settings(store_uri=store).strategy_dir,
        trading_day=trading_day.isoformat(),
        context=f"weekly arc for {trading_day.isoformat()} refused before stage 1",
    )


class ArcStageFailed(RuntimeError):
    """A stage exited non-zero. Carries which one, so the arc's own manifest
    `reason` names a job rather than a traceback in a module nobody opens."""


def run_arc(
    trading_day: dt.date,
    *,
    store: str | None,
    run_mode: str,
    registry: dict[str, Component] | None = None,
    main: object | None = None,
    dry_run: bool = False,
    arctic_library: str | None = None,
) -> list[Stage]:
    """Run every stage for ``trading_day``. Raises on the first failure.

    ``run_mode`` is required and threaded onto every stage's argv. It is the
    ARC's own resolved mode, passed down rather than re-resolved per stage:
    an arc and its stages are one invocation, and letting twelve stages each
    consult the environment is how an arc's manifest and its stages' manifests
    end up disagreeing about whether the week was live.

    ``main`` is injectable for tests only; it defaults to `crucible.cli.main`,
    imported lazily because `cli` imports this module's handler. A test that
    passed a fake would be testing its fake, so `tests/test_weekly.py` also
    asserts the real default is `crucible.cli.main` itself.

    ``dry_run=True`` passes `--dry-run` down to every stage's own argv
    (alpha-engine-config-I9922 N1) — the arc previously ignored the flag and
    dispatched every stage for real regardless of it.

    ``arctic_library`` passes `--arctic-library <name>` down to every stage in
    `ARCTIC_LIBRARY_JOBS` (`alpha-engine-config-I10633`) — additive and
    production-inert like `dry_run` above: absent, no stage's argv changes.
    The arc's own CLI job takes no `--arctic-library` flag itself (only the
    dedicated integration-test tier calls this parameter directly, resolving
    the value from `CRUCIBLE_INTEGRATION_ARCTIC_LIBRARY` via
    `crucible.required.require_env` in its own conftest — RAISE-on-absent
    already lives there, not here).

    **Before stage 1, every dispatched slot's champion pointer must name an arm
    the release can produce** (:func:`assert_arc_champions_producible`,
    `alpha-engine-config-I11085`). A pointer at an arm that cannot produce is
    not a stage failure waiting to happen, it is a certainty — and discovering
    it at `experiment.run[r]` took nine stages down with it on 2026-09-19. It
    runs under `dry_run` too, because a rehearsal that skipped it would pass
    the check the scheduled arc then fails.
    """
    if main is None:
        from crucible.cli import main as cli_main  # noqa: PLC0415 - cycle; see docstring

        main = cli_main
    stages = arc_stages(trading_day, registry)
    assert_arc_champions_producible(trading_day, stages, store=store)
    ran: list[Stage] = []
    for stage in stages:
        try:
            code = main(  # type: ignore[operator]
                stage.argv(
                    trading_day=trading_day,
                    store=store,
                    run_mode=run_mode,
                    dry_run=dry_run,
                    arctic_library=arctic_library,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised on the next line
            # NOT a swallow: re-raised immediately, chained to the original.
            # A stage that RAISES rather than returning non-zero would
            # otherwise leave the arc's own manifest reason naming the
            # underlying error with no clue which of twelve stages produced
            # it — and the stage name is the first thing an operator needs at
            # 09:00 on a Saturday. The stage's own manifest already carries
            # the full cause.
            raise ArcStageFailed(
                f"weekly arc stage {stage.label} raised for {trading_day.isoformat()}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if code != 0:
            raise ArcStageFailed(
                f"weekly arc stage {stage.label} exited {code} for {trading_day.isoformat()}; "
                "the stages after it read what it was to write, so the arc stops here "
                "rather than producing a report card over an absent input."
            )
        ran.append(stage)
    return ran
