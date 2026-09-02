"""`crucible explain <run_id|key>` — the lineage walk, from manifests alone.

Normative source: plan §10 component 8; principle 1 (Transparency).

Principle 1 asks whether someone can reconstruct *why* an unattended run did
what it did, **from durable artifacts alone, without asking Brian**. The
manifests already make that possible; this command makes it a five-second
operation instead of an S3 archaeology session.

**It reads manifests and nothing else.** No index, no graph store, no
side-table that could drift from the runs it describes. The chain is
recovered by the one relation the manifest schema already carries: a run's
`inputs[].key` is some other run's `outputs[].key`. Walking that backwards
from a verdict reaches the arena cycle, the shadows, the feature layer, the
price panel and the data source — with each hop's `code_sha`, `seed`,
`cost_usd` and `llm_calls` attached, because "why" includes which code, on
which seed, at what cost.

**SOTA:** a lineage graph (OpenLineage, MLflow). **Delta:** a manifest walk
with no graph store — at one producer per key the edges are already in the
artifacts, and a second store of the same edges is a thing to keep in sync.

**An unresolvable hop is reported, never elided.** A key nobody claims as an
output is printed as `produced by: UNKNOWN`, with the key. A chain that
quietly dropped its unexplained hops would read as complete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from crucible.keys import manifest_key
from crucible.manifest import validate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from crucible.store import Store

__all__ = ["Lineage", "explain", "load_manifests", "render"]

#: How deep the walk goes before it stops. A cycle in the input/output graph
#: is impossible by construction (a key has one producer, and a run cannot
#: read its own output), so a limit reached is a bug worth surfacing rather
#: than a depth to raise.
MAX_DEPTH = 12


@dataclass
class Lineage:
    """One node of the chain: a run, the key it explains, and what fed it.

    ``collisions`` is the OTHER run ids that also claimed this key as an
    output, oldest first — empty for the overwhelming majority of keys,
    which have exactly one producer. It is never used to pick a different
    winner; :func:`_index` already decided that (the later run, by
    `finished`). It exists so the caller sees both run_ids rather than one,
    per this module's own docstring: a shared output key is the
    last-writer-wins shape that gave a cycle's verdict to its
    worst-informed author, and a walk that silently kept only the winner
    would hide the exact defect it is meant to surface.
    """

    key: str
    manifest: dict[str, Any] | None
    depth: int
    parents: list[Lineage] = field(default_factory=list)
    collisions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        run = self.manifest or {}
        return {
            "key": self.key,
            "produced_by": run.get("run_id", "UNKNOWN"),
            "job": run.get("job"),
            "trading_day": run.get("trading_day"),
            "status": run.get("status"),
            "reason": run.get("reason"),
            "code_sha": run.get("code_sha"),
            "release_sha": run.get("release_sha"),
            "seed": run.get("seed"),
            "cost_usd": run.get("cost_usd"),
            "llm_calls": len(run.get("llm_calls") or []),
            "inputs": [i["key"] for i in run.get("inputs") or []],
            "also_claimed_by": list(self.collisions),
            "parents": [p.to_dict() for p in self.parents],
        }


def load_manifests(store: Store) -> list[dict[str, Any]]:
    """Every run manifest in the store, validated on read.

    Validated because a manifest written by an older release is still
    refused if it does not conform — an explanation assembled from a
    document nobody checked is an explanation nobody should act on.
    """
    out: list[dict[str, Any]] = []
    for key in store.list_keys("runs/"):
        if not key.endswith("/run.json"):
            continue
        document = json.loads(store.get_bytes(key).decode("utf-8"))
        validate(document)
        out.append(document)
    return out


def _index(
    manifests: list[dict[str, Any]],
) -> tuple[dict[str, dict], dict[str, dict], dict[str, tuple[str, ...]]]:
    """`{run_id: manifest}`, `{output key: manifest}`, and `{output key: other claimants}`.

    A key claimed as an output by two runs keeps the LATER one (by
    `finished`) as the answer `by_output` gives; that part was already true.
    What was missing is the third return value: every OTHER run_id that also
    claimed the key, oldest first, so `explain`/`render` can attach them to
    the node instead of the collision vanishing when the later manifest
    overwrites the earlier one in a plain dict.
    """
    by_run: dict[str, dict] = {}
    by_output: dict[str, dict] = {}
    claimants: dict[str, list[str]] = {}
    for manifest in sorted(manifests, key=lambda m: m["finished"]):
        by_run[manifest["run_id"]] = manifest
        for output in manifest.get("outputs") or []:
            key = output["key"]
            by_output[key] = manifest
            claimants.setdefault(key, []).append(manifest["run_id"])
    collisions = {
        key: tuple(run_ids[:-1]) for key, run_ids in claimants.items() if len(run_ids) > 1
    }
    return by_run, by_output, collisions


def explain(store: Store, target: str) -> Lineage:
    """Walk backwards from a run id or an artifact key.

    A `run_id` resolves to that run and then to everything it read; an
    artifact key resolves to the run that wrote it. Both are accepted
    because an operator holding a verdict has a key, and an operator holding
    a page has a run id.
    """
    manifests = load_manifests(store)
    if not manifests:
        raise FileNotFoundError(
            "no run manifests under `runs/` in this store. Nothing has run here, so "
            "there is no lineage — which is a different answer from 'this verdict has "
            "no explanation'."
        )
    by_run, by_output, collisions = _index(manifests)

    if target in by_run:
        root_manifest: dict[str, Any] | None = by_run[target]
        root_key = manifest_key(
            root_manifest["job"],
            root_manifest["trading_day"],
            discriminator=root_manifest.get("discriminator"),
        )
    elif target in by_output:
        root_manifest = by_output[target]
        root_key = target
    else:
        raise KeyError(
            f"{target!r} is neither a run_id nor a key any run claims as an output. "
            f"The store holds {len(manifests)} manifest(s) claiming "
            f"{len(by_output)} output key(s); `crucible explain` never guesses a near "
            "match, because an explanation of the wrong run is worse than none."
        )

    return _walk(root_key, root_manifest, by_output, collisions, depth=0, seen=set())


def _walk(
    key: str,
    manifest: dict[str, Any] | None,
    by_output: dict[str, dict],
    collisions: dict[str, tuple[str, ...]],
    *,
    depth: int,
    seen: set[str],
) -> Lineage:
    node = Lineage(key=key, manifest=manifest, depth=depth, collisions=collisions.get(key, ()))
    if manifest is None or depth >= MAX_DEPTH:
        return node
    for entry in manifest.get("inputs") or []:
        input_key = entry["key"]
        if input_key in seen:
            continue
        seen.add(input_key)
        node.parents.append(
            _walk(
                input_key,
                by_output.get(input_key),
                by_output,
                collisions,
                depth=depth + 1,
                seen=seen,
            )
        )
    return node


def render(node: Lineage) -> str:
    """The chain, as an operator reads it. One line per hop, indented by depth."""
    lines: list[str] = []

    def emit(current: Lineage) -> None:
        pad = "  " * current.depth
        run = current.manifest
        if run is None:
            lines.append(
                f"{pad}- {current.key}\n{pad}    produced by: UNKNOWN — no run in this "
                "store claims this key as an output"
            )
        else:
            lines.append(
                f"{pad}- {current.key}\n"
                f"{pad}    run {run['run_id']}  job={run['job']}  "
                f"trading_day={run['trading_day']}  status={run['status']}"
                + (f"  reason={run['reason']}" if run["reason"] else "")
                + f"\n{pad}    code_sha={run['code_sha'][:12]}  release_sha="
                f"{run['release_sha'][:12]}  seed={run['seed']}  "
                f"cost_usd={run['cost_usd']}  llm_calls={len(run.get('llm_calls') or [])}"
            )
            if current.collisions:
                lines.append(
                    f"{pad}    ALSO CLAIMED BY: {', '.join(current.collisions)} — "
                    "this key was written by more than one run; the run above is only "
                    "the LATEST by `finished`, and the others are not shown in this walk"
                )
        for parent in current.parents:
            emit(parent)

    emit(node)
    return "\n".join(lines)
