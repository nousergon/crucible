"""The `krepis` pin must carry the `openai` extra, or no box can make a call
in that wire format.

`alpha-engine-config-I10473`, measured 2026-09-10 on `i-0fcd1c9649a27aa62`,
release `ab8ed65`:

    FaultProbeFailure: fault_probe_outcome=routing_refusal: … Underlying
    ModuleNotFoundError: No module named 'openai' at fault_probe.py:320

`krepis` declares the client as an EXTRA (`openai>=1.40; extra == "openai"`,
imported at `krepis/llm.py`'s `from openai import OpenAI`). This repo pinned
bare `krepis`, and the box installs `"/tmp/$WHEEL[arcticdb]"` — crucible's
own extras, which cannot add one to a transitive dependency. So the client
was absent on every v2 box.

**Why this assertion and not `import openai`.** An import check passes in CI
for the wrong reason the moment anything in the dev tree pulls the package
in, and CI is exactly where it is most likely to be present incidentally.
What actually has to hold is a property of the DECLARATION: the pin this
repo ships names the extra, so a version bump cannot quietly drop it — which
is the realistic way this regresses, since `krepis==X` and
`krepis[openai]==X` differ by six characters in a line that gets rewritten
by hand on every bump.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"


def _krepis_requirement() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    for requirement in data["project"]["dependencies"]:
        if requirement.split("[")[0].split("=")[0].split(">")[0].strip() == "krepis":
            return requirement
    raise AssertionError("krepis is not in [project.dependencies]")


def test_the_krepis_pin_carries_the_openai_extra() -> None:
    requirement = _krepis_requirement()
    assert "[openai]" in requirement, (
        f"{requirement!r} does not carry the `openai` extra. krepis declares the "
        "OpenAI client as an extra and imports it in `krepis/llm.py`; without it "
        "every LLM call in the `openai` wire format dies "
        "`ModuleNotFoundError: No module named 'openai'` on the box, which is "
        "alpha-engine-config-I10473."
    )


def test_the_pin_is_still_an_exact_version() -> None:
    """The extra must not arrive at the cost of the pin. Fleet §139:
    first-party dependencies are PINNED, not floored — a floor would let a
    box install a krepis the lockfile never resolved.
    """
    requirement = _krepis_requirement()
    assert "==" in requirement, f"{requirement!r} is not an exact pin"
    assert ">=" not in requirement.split("==")[0], f"{requirement!r} floors rather than pins"


def test_the_lockfile_actually_resolved_the_client() -> None:
    """The declaration and the lock have to agree. A pyproject that names the
    extra over a lock that never resolved it would install nothing new and
    reproduce the original failure with a green test above it.
    """
    lock = (REPO / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "openai"' in lock, (
        "uv.lock does not resolve `openai`. Regenerate it in the same commit as "
        "the pyproject change — a pyproject.toml change lands with its lock."
    )
