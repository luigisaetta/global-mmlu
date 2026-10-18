"""
Author: L. Saetta
Date last modified: 2026-10-18
License: MIT
Description: Statistics helpers: Wilson score interval for an accuracy (binomial proportion).
"""

import math

Z_95 = 1.959964  # two-sided 95% normal quantile


def wilson_interval(correct: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Return the Wilson score interval of ``correct / total``.

    Unlike the normal approximation, the Wilson interval stays inside [0, 1]
    and remains meaningful for small samples and for accuracies near 0% or
    100% (for example 10/10 gives about 0.72-1.00).

    Args:
        correct: number of correct answers.
        total: number of answered questions.
        z: normal quantile of the confidence level (default 95%).

    Returns:
        tuple[float, float]: lower and upper bound; ``(0.0, 1.0)`` when
        ``total`` is 0.

    Raises:
        ValueError: if ``correct`` is negative or greater than ``total``.
    """
    if not 0 <= correct <= total:
        raise ValueError(f"Invalid counts: correct={correct}, total={total}")
    if total == 0:
        return 0.0, 1.0
    p = correct / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
    half_width /= denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)
