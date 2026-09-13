"""`crucible.required.optional_env` — the override counterpart to
`require_env` (`alpha-engine-config-I10458`).

No existing test file exercised `crucible.required` directly; `require_env`
is covered indirectly through its many consumers (`crucible.alerts`,
`crucible.gate`, ...). `optional_env` is new and gets its own direct
coverage rather than relying on `crucible/morning.py` alone to prove it.
"""

from __future__ import annotations

from crucible.required import optional_env

VAR = "CRUCIBLE_TEST_OPTIONAL_ENV_VAR"


def test_unset_returns_the_default(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)
    assert optional_env(VAR, default="prod-value") == "prod-value"


def test_empty_string_returns_the_default(monkeypatch):
    """Empty is treated the same as unset — a blank override must not win
    over a safe, named default."""
    monkeypatch.setenv(VAR, "")
    assert optional_env(VAR, default="prod-value") == "prod-value"


def test_whitespace_only_returns_the_default(monkeypatch):
    monkeypatch.setenv(VAR, "   ")
    assert optional_env(VAR, default="prod-value") == "prod-value"


def test_a_set_value_wins_over_the_default(monkeypatch):
    monkeypatch.setenv(VAR, "override-value")
    assert optional_env(VAR, default="prod-value") == "override-value"


def test_a_set_value_is_stripped(monkeypatch):
    monkeypatch.setenv(VAR, "  override-value  ")
    assert optional_env(VAR, default="prod-value") == "override-value"
