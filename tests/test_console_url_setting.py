"""`CRUCIBLE_CONSOLE_URL` — the one outbound link this tree carries, as config.

alpha-engine-config-I9926. The fleet console's host is an infrastructure
identifier, and this repository goes public at phase-1 exit, so the default
is EMPTY and the morning report falls back to a presigned page when nothing
resolves it. These pin the resolution order, the provenance record and the
"no default" fact — the same three properties `CRUCIBLE_CLOUDTRAIL_ARCHIVE`
already pins.
"""

from __future__ import annotations

import ast
import inspect

from crucible import config
from crucible.config import DEFAULT_CONSOLE_URL, settings


def test_the_default_is_empty_and_the_tree_names_no_host():
    assert DEFAULT_CONSOLE_URL == ""
    # No web host literal anywhere in config.py — an AST walk over string
    # constants, so a docstring MENTIONING `s3://` or the variable is fine and
    # an `http(s)://` value is not.
    tree = ast.parse(inspect.getsource(config))
    literals = [
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    hosts = [s for s in literals if s.strip().lower().startswith(("http://", "https://"))]
    assert not hosts, hosts


def test_unset_resolves_to_the_empty_default(monkeypatch):
    monkeypatch.delenv("CRUCIBLE_CONSOLE_URL", raising=False)
    resolved = settings()
    assert resolved.console_url == ""
    assert resolved.origins["console_url"] == "default"


def test_the_environment_variable_is_read_with_provenance(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_CONSOLE_URL", "https://console.example.test")
    resolved = settings()
    assert resolved.console_url == "https://console.example.test"
    assert resolved.origins["console_url"] == "environ:CRUCIBLE_CONSOLE_URL"


def test_an_explicit_argument_outranks_the_environment(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_CONSOLE_URL", "https://env.example.test")
    resolved = settings(console_url="https://arg.example.test")
    assert resolved.console_url == "https://arg.example.test"
    assert resolved.origins["console_url"] == "argument"


def test_a_trailing_slash_is_normalised_away(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_CONSOLE_URL", "https://console.example.test/")
    assert settings().console_url == "https://console.example.test"


def test_the_value_is_reported_in_to_dict(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_CONSOLE_URL", "https://console.example.test")
    assert settings().to_dict()["console_url"] == "https://console.example.test"
