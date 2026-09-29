"""The gate's lazily-constructed AWS clients, exercised without a network.

`_s3_client` and `_ce_client` carried `pragma: no cover - constructed only
outside tests`, which is the coverage ratchet being narrowed by hand at the
exact two functions that touch a credential chain. They are constructible
offline: a boto3 client needs a region and nothing else until it is called.
"""

from __future__ import annotations

import pytest

import crucible.gate as gate_module


@pytest.fixture(autouse=True)
def _region_only(monkeypatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


def test_the_s3_client_is_an_s3_client() -> None:
    client = gate_module._s3_client()
    assert client.meta.service_model.service_name == "s3"


def test_the_spend_client_reads_the_collector_not_cost_explorer(
    no_live_cost_explorer, monkeypatch
) -> None:
    # The autouse fixture replaces `_ce_client` with a refusal; it hands back
    # the real constructor for exactly this assertion, which builds a client
    # and reads nothing (alpha-engine-config-I11707).
    from crucible.cost import EXPENSES_URI_VAR, CollectorSpendClient

    monkeypatch.setenv(EXPENSES_URI_VAR, "s3://test-bucket/expenses/latest.json")
    assert isinstance(no_live_cost_explorer(), CollectorSpendClient)


def test_an_unset_spend_source_raises_rather_than_reading_cost_explorer(
    no_live_cost_explorer, monkeypatch
) -> None:
    from crucible.cost import EXPENSES_URI_VAR, CostUnreadableError

    monkeypatch.delenv(EXPENSES_URI_VAR, raising=False)
    with pytest.raises(CostUnreadableError, match=EXPENSES_URI_VAR):
        no_live_cost_explorer()


def test_the_typing_shim_returns_none_and_touches_nothing() -> None:
    assert gate_module._unused((object(), object())) is None
