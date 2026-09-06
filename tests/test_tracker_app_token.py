"""The tracker credential is minted from the fleet GitHub App, not stored.

`alpha-engine-config-I9967`, revised 2026-09-06 on Brian's question "do we
really need to make another one?" — no. The `ne-groomer` App is installed
org-wide with Issues: write and its credentials are already in SSM;
`nousergon_lib.github_app.installation_token` mints a short-lived token from
them, narrowed at mint time to the two calls this adapter makes.
"""

from __future__ import annotations

import pytest

import crucible.tracker as tracker_module
from crucible.tracker import (
    TRACKER_APP_PERMISSIONS,
    TRACKER_APP_SSM_PREFIX_VAR,
    TRACKER_TOKEN_VAR,
    TrackerCredentialError,
    TrackerError,
    credential,
    post_comment,
    read_issue,
)

PREFIX = "/example/app/"
REPO = "nousergon/alpha-engine-config"


class _Minted:
    def __init__(self, token: str = "ghs_minted", *, fail: Exception | None = None) -> None:
        self.token = token
        self.fail = fail
        self.calls: list[dict] = []

    def __call__(self, **kwargs) -> str:
        self.calls.append(kwargs)
        if self.fail is not None:
            raise self.fail
        return self.token


def _install(monkeypatch, minted: _Minted) -> None:
    import nousergon_lib.github_app as app_module  # noqa: PLC0415 - local to the test

    monkeypatch.setattr(app_module, "installation_token", minted)


class TestTheCredentialIsMintedFromTheApp:
    def test_the_prefix_mints_a_token_narrowed_to_issues_write(self, monkeypatch) -> None:
        monkeypatch.setenv(TRACKER_APP_SSM_PREFIX_VAR, PREFIX)
        minted = _Minted()
        _install(monkeypatch, minted)
        assert credential() == "ghs_minted"
        assert minted.calls == [{"ssm_prefix": PREFIX, "permissions": TRACKER_APP_PERMISSIONS}]

    def test_the_narrowing_is_exactly_issues_write_and_nothing_wider(self) -> None:
        assert TRACKER_APP_PERMISSIONS == {"issues": "write"}

    def test_an_explicit_token_beats_the_app(self, monkeypatch) -> None:
        monkeypatch.setenv(TRACKER_APP_SSM_PREFIX_VAR, PREFIX)
        minted = _Minted()
        _install(monkeypatch, minted)
        assert credential("explicit") == "explicit"
        assert minted.calls == [], "no mint when a token is in hand"

    def test_the_environment_token_beats_the_app(self, monkeypatch) -> None:
        monkeypatch.setenv(TRACKER_APP_SSM_PREFIX_VAR, PREFIX)
        monkeypatch.setenv(TRACKER_TOKEN_VAR, "from-env")
        minted = _Minted()
        _install(monkeypatch, minted)
        assert credential() == "from-env"
        assert minted.calls == []

    def test_neither_configured_is_a_named_absence_not_a_mint(self, monkeypatch) -> None:
        monkeypatch.delenv(TRACKER_APP_SSM_PREFIX_VAR, raising=False)
        minted = _Minted(fail=AssertionError("must not be called"))
        _install(monkeypatch, minted)
        assert credential() is None

    def test_a_failed_mint_is_configured_and_broken_never_absent(self, monkeypatch) -> None:
        from nousergon_lib.github_app import GitHubAppTokenError  # noqa: PLC0415

        monkeypatch.setenv(TRACKER_APP_SSM_PREFIX_VAR, PREFIX)
        _install(monkeypatch, _Minted(fail=GitHubAppTokenError("AccessDenied on SSM")))
        with pytest.raises(TrackerCredentialError, match="AccessDenied on SSM"):
            credential()

    def test_any_other_mint_fault_is_reported_with_its_cause(self, monkeypatch) -> None:
        monkeypatch.setenv(TRACKER_APP_SSM_PREFIX_VAR, PREFIX)
        _install(monkeypatch, _Minted(fail=RuntimeError("boom")))
        with pytest.raises(TrackerCredentialError, match="RuntimeError: boom"):
            credential()


class TestTheSurfacesRenderAMintFailureHonestly:
    def test_a_read_returns_the_mint_fault_as_an_access_problem(self, monkeypatch) -> None:
        from nousergon_lib.github_app import GitHubAppTokenError  # noqa: PLC0415

        monkeypatch.setenv(TRACKER_APP_SSM_PREFIX_VAR, PREFIX)
        _install(monkeypatch, _Minted(fail=GitHubAppTokenError("unreadable at SSM")))
        read = read_issue(REPO, 9757)
        assert read.state is None
        assert read.access_problem is True
        assert "unreadable at SSM" in read.problem
        assert TRACKER_APP_SSM_PREFIX_VAR in read.problem

    def test_a_write_raises_the_mint_fault_before_touching_github(self, monkeypatch) -> None:
        from nousergon_lib.github_app import GitHubAppTokenError  # noqa: PLC0415

        monkeypatch.setenv(TRACKER_APP_SSM_PREFIX_VAR, PREFIX)
        _install(monkeypatch, _Minted(fail=GitHubAppTokenError("no App")))

        def opener(_request):
            raise AssertionError("GitHub must not be reached without a credential")

        with pytest.raises(TrackerError, match="no App"):
            post_comment(REPO, 9757, "reading", opener=opener)

    def test_the_grant_names_the_variable_and_the_stack_not_a_pat(self) -> None:
        text = tracker_module.grant_command(REPO)
        assert TRACKER_APP_SSM_PREFIX_VAR in text
        assert "crucible-v2 stack" in text
        assert "personal access token" not in text
