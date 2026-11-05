"""A tuned ranker parameter has no public default (codex b373ab794ede195d).

`crucible.slots.rankers._required_param` refuses a recipe that omits a tuned
value, or supplies one that is not a number, instead of running on a number
nobody declared. Nothing exercised those refusals.
"""

from __future__ import annotations

import pytest

from crucible.slots.rankers import _required_param  # pyright: ignore[reportPrivateUsage]


def test_a_missing_tuned_parameter_is_refused_by_name():
    with pytest.raises(ValueError, match="needs parameter 'cut_top_n'.*carries no default"):
        _required_param({}, "predicted_alpha_within_cut", "cut_top_n")


@pytest.mark.parametrize("value", [True, False, "10", None, [10]])
def test_a_non_numeric_tuned_parameter_is_refused(value):
    with pytest.raises(ValueError, match="must be a number"):
        _required_param({"cut_top_n": value}, "predicted_alpha_within_cut", "cut_top_n")


@pytest.mark.parametrize("value, expected", [(10, 10.0), (0.25, 0.25)])
def test_a_numeric_tuned_parameter_is_returned_as_float(value, expected):
    assert _required_param({"w": value}, "quant_composite", "w") == expected
