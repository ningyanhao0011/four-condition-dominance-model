"""Module A - prior distributions for Monte Carlo sampling.

Each `sample_<name>(n, rng)` returns n draws from the prior. Priors are
documented with their literature anchor in the function docstring so the
choice is auditable.

All Beta priors are scaled with `_scaled_beta(a, b, lo, hi)` so that the
[0, 1] Beta variable is mapped to [lo, hi].
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _scaled_beta(a: float, b: float, lo: float, hi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    raw = stats.beta(a, b).rvs(n, random_state=rng)
    return lo + (hi - lo) * raw


# ---------------------------------------------------------------------
# Closure subcomponents for A4
# ---------------------------------------------------------------------

def sample_S1(n: int, rng: np.random.Generator) -> np.ndarray:
    """Operational autonomy.

    Anchored by GAIA benchmark (Mialon et al. 2023, ~10-15% multi-step
    autonomy lower bound) and SWE-bench-Verified (~50% upper bound for
    isolated tasks). Symmetric Beta(2.5, 2.5) on [0.30, 0.60]; mean 0.45.
    """
    return _scaled_beta(2.5, 2.5, 0.30, 0.60, n, rng)


def sample_S2(n: int, rng: np.random.Generator) -> np.ndarray:
    """Maintenance autonomy.

    Skewed low: automated retraining is common but architecture redesign
    and security patching remain human-led (Uptime Institute 2024).
    Beta(2, 4) on [0.10, 0.35]; mean 0.187.
    """
    return _scaled_beta(2.0, 4.0, 0.10, 0.35, n, rng)


def sample_S3(n: int, rng: np.random.Generator) -> np.ndarray:
    """Dependency independence.

    Anchored by Kinniment et al. 2024 ARA evaluations. LogNormal with
    median 0.02 and sigma 0.5 in log-space, clipped to [0.001, 0.10].
    Heavy right tail allows for upside scenarios without forcing them.
    """
    mu = np.log(0.02)
    sigma = 0.5
    draws = stats.lognorm(s=sigma, scale=np.exp(mu)).rvs(n, random_state=rng)
    return np.clip(draws, 0.001, 0.10)


def sample_R_A4(n: int, rng: np.random.Generator) -> np.ndarray:
    """Replication score for A4.

    Strongly skewed low because contemporary model succession is human-
    driven (Pan et al. 2024 self-replication is sandbox-only). Beta(1.5, 5)
    scaled to [0.05, 0.80]; mean 0.222.
    """
    return _scaled_beta(1.5, 5.0, 0.05, 0.80, n, rng)


# ---------------------------------------------------------------------
# Closure and replication for H3
# ---------------------------------------------------------------------

def sample_C_H3(n: int, rng: np.random.Generator) -> np.ndarray:
    """Civilisation closure.

    Strong prior on saturation: H3 is the outermost organised system in
    the framework. Beta(8, 2) on [0.80, 0.99]; mean ~0.952.
    """
    return _scaled_beta(8.0, 2.0, 0.80, 0.99, n, rng)


def sample_R_H3(n: int, rng: np.random.Generator) -> np.ndarray:
    """Civilisation replication.

    Demographic and cultural transmission across generations. Held at
    Beta(6, 2) on [0.55, 0.85]; mean ~0.775.
    """
    return _scaled_beta(6.0, 2.0, 0.55, 0.85, n, rng)


# ---------------------------------------------------------------------
# Dissipation
# ---------------------------------------------------------------------

def sample_E_A4(n: int, rng: np.random.Generator) -> np.ndarray:
    """A4 functional dissipation in TW.

    LogNormal median 0.06, sigma 0.35: P05 ~0.034, P95 ~0.106.
    Envelope spans de Vries 2023/2025, IEA 2025a/b, Shehabi et al. 2024.
    """
    mu = np.log(0.06)
    sigma = 0.35
    return stats.lognorm(s=sigma, scale=np.exp(mu)).rvs(n, random_state=rng)


def sample_E_H3(n: int, rng: np.random.Generator) -> np.ndarray:
    """H3 functional dissipation in TW.

    Energy Institute 2023 total primary energy (~18.4 TW) times functional
    fraction 0.30-0.55. Normal(7.6, 1.0), clipped positive.
    """
    raw = stats.norm(loc=7.6, scale=1.0).rvs(n, random_state=rng)
    return np.clip(raw, 3.0, 12.0)


# ---------------------------------------------------------------------
# Sigmoid parameters
# ---------------------------------------------------------------------

def sample_k(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sigmoid steepness. Uniform(2, 20)."""
    return stats.uniform(loc=2.0, scale=18.0).rvs(n, random_state=rng)


def sample_x_star(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sigmoid midpoint. Uniform(0.3, 0.6)."""
    return stats.uniform(loc=0.3, scale=0.3).rvs(n, random_state=rng)


# ---------------------------------------------------------------------
# Aggregation rule as a sampled categorical
# ---------------------------------------------------------------------

AGG_RULES = ("min", "geomean", "wavg")


def sample_agg_rule(n: int, rng: np.random.Generator) -> np.ndarray:
    """Discrete uniform over aggregation rules.

    Treating the aggregation rule itself as an uncertain modelling
    decision is the core defence against the 'min-aggregation baked the
    answer' critique. If the substantive result holds across the three
    rules, the conclusion is not artefactual.
    """
    idx = rng.integers(0, len(AGG_RULES), size=n)
    return np.array(AGG_RULES)[idx]
