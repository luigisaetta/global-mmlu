"""
Author: L. Saetta
Date last modified: 2026-10-18
License: MIT
Description: Tests of the Wilson score interval.
"""

# pylint: disable=missing-function-docstring

import pytest

from app.stats import wilson_interval


@pytest.mark.parametrize(
    "correct,total,low,high",
    [
        (80, 100, 0.7112, 0.8666),  # reference values of the Wilson interval
        (10, 10, 0.7225, 1.0),
        (0, 10, 0.0, 0.2775),
        (200, 250, 0.7458, 0.8446),
    ],
)
def test_known_values(correct, total, low, high):
    ci_low, ci_high = wilson_interval(correct, total)
    assert ci_low == pytest.approx(low, abs=1e-3)
    assert ci_high == pytest.approx(high, abs=1e-3)


def test_interval_contains_the_estimate_and_shrinks_with_n():
    small = wilson_interval(8, 10)
    large = wilson_interval(800, 1000)
    assert small[0] < 0.8 < small[1]
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_empty_sample_is_uninformative():
    assert wilson_interval(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize("correct,total", [(-1, 10), (11, 10)])
def test_invalid_counts_are_rejected(correct, total):
    with pytest.raises(ValueError):
        wilson_interval(correct, total)
