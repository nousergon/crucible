"""One document per weekly arc: what the console's weekly-run page reads.

Director, 2026-10-31 (Brian agreed a "Crucible weekly run" console page for
M3): version 1 of the page reads one document per week rather than
re-deriving it from ~3,400 verdicts, 23 stage manifests and four arena cycles.
This module builds that document from artifacts the arc's stages already
wrote. It computes nothing the arena does not already compute, except the one
comparison the arena does not file: every arm beside the champion over the
SAME dates, so an arm with no decision this week still has a row.

**Compute cost is an ESTIMATE and says so.** Every manifest's ``cost_usd`` is
LLM spend only and reads 0.0 for a run with no model calls; the box itself is
priced nowhere. ``compute`` multiplies the arc's wall time by the declared
on-demand list price of the instance type (:data:`COMPUTE_PRICES`), marked
``estimated: true`` with its source. It excludes boot and install time before
the arc started, and a spot box is billed below list, so the figure is an
upper bound on the arc's own compute, not an invoice. LLM cost stays its own
field.

**Pure over its reads.** :func:`build_summary` takes a ``read`` callable, so
the arc (reading the store) and a backfill (reading the same keys later) build
byte-comparable documents, and tests need no store.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import cast

from crucible.keys import (
    arena_cycle_key,
    arm_series_key,
    attribution_key,
    coverage_key,
    experiments_key,
    parse_manifest_key,
)

__all__ = [
    "COMPARISON_DATES",
    "COMPUTE_PRICES",
    "SLOTS",
    "WEEKLY_SUMMARY_SCHEMA",
    "build_summary",
    "compute_estimate",
]

WEEKLY_SUMMARY_SCHEMA = "weekly_summary.v1"
SLOTS = ("u", "r", "m", "s")

#: The arm-vs-champion comparison window: the most recent scored dates both
#: series share, up to this many (about four trading weeks).
COMPARISON_DATES = 20

#: Declared on-demand list prices, USD per instance-hour, us-east-1 Linux.
#: A type missing here gives a ``null`` cost with the reason, never a guess.
COMPUTE_PRICES: Mapping[str, float] = {
    "r5a.large": 0.113,
    "r5a.xlarge": 0.226,
    "r5a.2xlarge": 0.452,
}
COMPUTE_PRICE_SOURCE = "AWS EC2 on-demand list price, us-east-1 Linux (declared in crucible)"

Doc = Mapping[str, object]
Reader = Callable[[str], "Doc | None"]
EventsReader = Callable[[str], "list[Doc]"]


def _ts(value: object) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _seconds(started: object, finished: object) -> float | None:
    start, end = _ts(started), _ts(finished)
    if start is None or end is None:
        return None
    return round((end - start).total_seconds(), 1)


def _dict(value: object) -> dict[str, object]:
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def compute_estimate(
    instance_type: str | None, wall_seconds: float | None, spot: object
) -> dict[str, object]:
    """Instance-hours x declared list price, marked estimated."""
    price = COMPUTE_PRICES.get(instance_type or "")
    hours = None if wall_seconds is None else round(wall_seconds / 3600, 4)
    cost = None if price is None or hours is None else round(hours * price, 4)
    reason = ""
    if price is None:
        reason = f"no declared price for instance type {instance_type!r}"
    elif hours is None:
        reason = "arc wall time unknown"
    return {
        "estimated": True,
        "instance_type": instance_type,
        "spot": bool(spot),
        "instance_hours": hours,
        "price_usd_per_hour": price,
        "price_source": COMPUTE_PRICE_SOURCE,
        "cost_usd": cost,
        "basis": (
            "arc wall time only (excludes boot/install); list price is an upper bound for spot"
        ),
        "reason": reason,
    }


def _stage_row(key: str, manifest: Doc) -> dict[str, object]:
    parsed = parse_manifest_key(key)
    job, _, slot = parsed if parsed else (key, "", None)
    attempts = manifest.get("attempts")
    return {
        "job": job,
        "slot": slot,
        "manifest_key": key,
        "status": manifest.get("status"),
        "started": manifest.get("started"),
        "finished": manifest.get("finished"),
        "wall_seconds": _seconds(manifest.get("started"), manifest.get("finished")),
        "attempts": len(cast("list[object]", attempts)) if isinstance(attempts, list) else None,
        "llm_cost_usd": manifest.get("cost_usd"),
    }


def _scores(series: Doc | None) -> dict[str, float]:
    raw = _dict(series.get("scores")) if series else {}
    return {d: float(cast("float", v)) for d, v in raw.items() if isinstance(v, int | float)}


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _diff(arm: float | None, champion: float | None) -> float | None:
    return None if arm is None or champion is None else round(arm - champion, 6)


def _slot(slot: str, trading_day: str, read: Reader) -> dict[str, object]:
    cycle = read(arena_cycle_key(slot, trading_day))
    if cycle is None:
        return {"cycle_key": arena_cycle_key(slot, trading_day), "present": False}
    decision = _dict(cycle.get("decision"))
    champion = str(decision.get("champion") or decision.get("incumbent") or "")
    scored = [str(a) for a in cast("list[object]", cycle.get("scored_arms") or [])]
    ineligible = _dict(decision.get("ineligible"))
    controls = {
        arm
        for arm, checks in ineligible.items()
        if any(
            _dict(c).get("name") == "not_a_control_arm"
            for c in cast("list[object]", checks if isinstance(checks, list) else [])
        )
    }
    champion_scores = _scores(read(arm_series_key(slot, champion))) if champion else {}
    arms: list[dict[str, object]] = []
    for arm in scored:
        own = champion_scores if arm == champion else _scores(read(arm_series_key(slot, arm)))
        common = sorted(set(own) & set(champion_scores))[-COMPARISON_DATES:]
        arm_mean = _mean([own[d] for d in common])
        champ_mean = _mean([champion_scores[d] for d in common])
        arms.append(
            {
                "arm_id": arm,
                "is_champion": arm == champion,
                "control": arm in controls,
                "window_start": common[0] if common else None,
                "window_end": common[-1] if common else None,
                "n_dates": len(common),
                "arm_mean_score": arm_mean,
                "champion_mean_score": champ_mean,
                "diff_vs_champion": _diff(arm_mean, champ_mean),
            }
        )
    return {
        "cycle_key": arena_cycle_key(slot, trading_day),
        "present": True,
        "benchmark": cycle.get("benchmark"),
        "champion": champion or None,
        "decision_status": decision.get("status"),
        "pointer_moved": decision.get("moved"),
        "decision_reason": decision.get("reason"),
        "challengers_scored": sum(1 for a in scored if a != champion and a not in controls),
        "controls_scored": sum(1 for a in scored if a in controls),
        "arms": arms,
    }


def build_summary(
    *,
    trading_day: str,
    run_id: str,
    arc_started: str,
    arc_finished: str,
    stages: Iterable[tuple[str, Doc]],
    read: Reader,
    read_events: EventsReader,
    generated_utc: str,
    backfilled: bool = False,
) -> dict[str, object]:
    """The `weekly_summary.v1` document for one arc."""
    stage_list = list(stages)
    first: Doc = stage_list[0][1] if stage_list else {}
    resource = _dict(first.get("resource"))
    coverage: Doc = read(coverage_key(trading_day)) or {}
    attribution: Doc = read(attribution_key(trading_day)) or {}
    wall = _seconds(arc_started, arc_finished)
    llm = [
        float(cast("float", m.get("cost_usd")))
        for _, m in stage_list
        if isinstance(m.get("cost_usd"), int | float)
    ]
    return {
        "schema_version": WEEKLY_SUMMARY_SCHEMA,
        "trading_day": trading_day,
        "run_id": run_id,
        "generated_utc": generated_utc,
        "backfilled": backfilled,
        "arc": {"started": arc_started, "finished": arc_finished, "wall_seconds": wall},
        "release": {
            "release_sha": first.get("release_sha"),
            "code_sha": first.get("code_sha"),
            "wheelhouse_digest": first.get("wheelhouse_digest"),
        },
        "data": {
            "coverage_key": coverage_key(trading_day),
            "data_snapshot_id": coverage.get("data_snapshot_id"),
            "feature_version": coverage.get("feature_version"),
            "coverage_ratio": coverage.get("coverage_ratio"),
        },
        "stages": [_stage_row(k, m) for k, m in stage_list],
        "compute": compute_estimate(
            cast("str | None", resource.get("instance_type")), wall, resource.get("spot")
        ),
        "llm_cost_usd": round(sum(llm), 6),
        "slots": {slot: _slot(slot, trading_day, read) for slot in SLOTS},
        "decisions": read_events(experiments_key(trading_day)),
        "attribution": {
            "key": attribution_key(trading_day),
            "grade": attribution.get("grade"),
            "grade_reason": attribution.get("grade_reason"),
            "members": attribution.get("members"),
        },
    }


def encode(summary: Mapping[str, object]) -> bytes:
    return json.dumps(summary, indent=2, sort_keys=True).encode("utf-8")
