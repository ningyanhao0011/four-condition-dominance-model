"""Closure aggregation rules. C = aggregate(S1, S2, S3).

The v2 paper used min-aggregation as the primary rule and weighted
average as a sensitivity check. Module A treats the rule itself as a
sampled categorical to address the 'aggregation-rule baked the answer'
critique from internal review.
"""
from __future__ import annotations

import numpy as np

DEFAULT_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)


def aggregate_min(S1, S2, S3):
    return np.minimum(np.minimum(S1, S2), S3)


def aggregate_geomean(S1, S2, S3, eps: float = 1e-9):
    """Geometric mean with epsilon guard for zeros."""
    S1 = np.maximum(np.asarray(S1), eps)
    S2 = np.maximum(np.asarray(S2), eps)
    S3 = np.maximum(np.asarray(S3), eps)
    return np.cbrt(S1 * S2 * S3)


def aggregate_wavg(S1, S2, S3, weights=DEFAULT_WEIGHTS):
    w1, w2, w3 = weights
    return w1 * np.asarray(S1) + w2 * np.asarray(S2) + w3 * np.asarray(S3)


def aggregate_max(S1, S2, S3):
    return np.maximum(np.maximum(S1, S2), S3)


RULES = {
    "min": aggregate_min,
    "geomean": aggregate_geomean,
    "wavg": aggregate_wavg,
    "max": aggregate_max,
}


def aggregate(S1, S2, S3, rule: str = "min"):
    if rule not in RULES:
        raise ValueError(f"Unknown aggregation rule: {rule!r}. Valid: {list(RULES)}")
    return RULES[rule](S1, S2, S3)
