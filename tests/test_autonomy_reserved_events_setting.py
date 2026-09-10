"""`CRUCIBLE_AUTONOMY_RESERVED_EVENTS` — `alpha-engine-config-I10416` §11 row 9.

The reserved-action exclusion list (IB Gateway paper re-auth, privileged SSO
actions, rulings on holdout unseal and trader release pin) is declared in
CONFIG, never a Python literal in this public-at-phase-1-exit tree. These pin
the resolution order and the provenance record, the same shape
`test_console_url_setting.py` already pins for `CRUCIBLE_CONSOLE_URL`.
"""

from __future__ import annotations

from crucible.config import DEFAULT_AUTONOMY_RESERVED_EVENTS, settings


def test_the_default_is_empty():
    assert DEFAULT_AUTONOMY_RESERVED_EVENTS == ""
    assert settings().autonomy_reserved_events == ()


def test_unset_resolves_to_the_empty_default(monkeypatch):
    monkeypatch.delenv("CRUCIBLE_AUTONOMY_RESERVED_EVENTS", raising=False)
    resolved = settings()
    assert resolved.autonomy_reserved_events == ()
    assert resolved.origins["autonomy_reserved_events"] == "default"


def test_the_environment_variable_is_parsed_as_a_comma_separated_set(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_AUTONOMY_RESERVED_EVENTS", "EventA,EventB, EventC ")
    resolved = settings()
    assert resolved.autonomy_reserved_events == ("EventA", "EventB", "EventC")
    assert (
        resolved.origins["autonomy_reserved_events"] == "environ:CRUCIBLE_AUTONOMY_RESERVED_EVENTS"
    )


def test_an_explicit_argument_outranks_the_environment(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_AUTONOMY_RESERVED_EVENTS", "FromEnv")
    resolved = settings(autonomy_reserved_events="FromArgument")
    assert resolved.autonomy_reserved_events == ("FromArgument",)
    assert resolved.origins["autonomy_reserved_events"] == "argument"


def test_the_value_is_reported_in_to_dict(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_AUTONOMY_RESERVED_EVENTS", "EventA")
    assert settings().to_dict()["autonomy_reserved_events"] == ["EventA"]
