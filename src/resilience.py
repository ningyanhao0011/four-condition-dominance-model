"""Module B - Resilience h(Z) operationalisation.

Replaces the v2 placeholder h(Z) = 1 with empirically grounded values for
human civilisation (H3) and AI infrastructure (A4). Reports two
A4 resilience values - one for short-timescale disturbances and one for
long-timescale supply-chain disturbances - because the asymmetry between
them is itself a substantive finding.

Z is defined as the *persistence fraction* of system function that
survives an indefinite or characteristic-timescale disturbance, with
empirical anchors in `data/recovery_data.csv`.

The function `sample_Z(system, n, rng)` returns Monte Carlo draws from a
Beta posterior fitted to the empirical (low, mid, high) range per system.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(__file__).resolve().parents[1] / "data" / "recovery_data.csv"

VALID_SYSTEMS = ("H3", "A4_short", "A4_long")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    return df


def beta_from_quantiles(low: float, mid: float, high: float) -> tuple[float, float]:
    """Fit a Beta(alpha, beta) such that median ~= mid and P10 ~= low, P90 ~= high.

    We solve a small optimisation. Beta is parameterised so that
    alpha/(alpha+beta) is close to mid and the spread matches the
    low/high interval at the 10/90 quantiles.
    """
    from scipy.optimize import minimize

    def loss(params):
        a, b = np.exp(params)  # ensure positive
        rv = stats.beta(a, b)
        p10 = rv.ppf(0.10)
        p50 = rv.ppf(0.50)
        p90 = rv.ppf(0.90)
        return (p10 - low) ** 2 + (p50 - mid) ** 2 + (p90 - high) ** 2

    # Initial guess: method-of-moments from mid and approximate variance
    var = ((high - low) / 2.56) ** 2  # 80% CI -> 2.56 sigma in normal approx
    mu = mid
    var = min(var, mu * (1 - mu) * 0.99)  # ensure valid
    a0 = mu * ((mu * (1 - mu) / var) - 1)
    b0 = (1 - mu) * ((mu * (1 - mu) / var) - 1)
    x0 = np.log([max(a0, 1e-2), max(b0, 1e-2)])
    result = minimize(loss, x0, method="Nelder-Mead")
    a, b = np.exp(result.x)
    return float(a), float(b)


def system_posterior(df: pd.DataFrame, system: str) -> tuple[float, float]:
    """Aggregate Beta posterior for a system by pooling its disturbances.

    We treat each disturbance as a noisy observation of the underlying
    system-level resilience. We compute a Beta prior per disturbance,
    then take a weighted-equal mixture to produce the marginal.

    To return a single Beta parameterisation we re-fit a Beta to the
    pooled (P10, P50, P90) of the mixture.
    """
    sub = df[df["system"] == system].copy()
    if sub.empty:
        raise ValueError(f"No data for system {system!r}")
    # Build per-disturbance Beta and sample
    rng = np.random.default_rng(0)
    samples_per = 5000
    pooled = []
    for _, row in sub.iterrows():
        a, b = beta_from_quantiles(
            row["persistence_fraction_low"],
            row["persistence_fraction_mid"],
            row["persistence_fraction_high"],
        )
        pooled.append(stats.beta(a, b).rvs(samples_per, random_state=rng))
    mix = np.concatenate(pooled)
    p10, p50, p90 = np.quantile(mix, [0.10, 0.50, 0.90])
    a_mix, b_mix = beta_from_quantiles(float(p10), float(p50), float(p90))
    return a_mix, b_mix


def fit_all() -> dict[str, tuple[float, float]]:
    df = load_data()
    return {s: system_posterior(df, s) for s in VALID_SYSTEMS}


def sample_Z(system: str, n: int, rng: np.random.Generator,
             cached: dict[str, tuple[float, float]] | None = None) -> np.ndarray:
    """Sample n draws from the Z posterior for a system.

    Pass `cached=fit_all()` to avoid re-fitting on every call.
    """
    if cached is None:
        cached = fit_all()
    a, b = cached[system]
    return stats.beta(a, b).rvs(n, random_state=rng)


def summary_table(cached: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    if cached is None:
        cached = fit_all()
    rows = []
    for system, (a, b) in cached.items():
        rv = stats.beta(a, b)
        rows.append(
            dict(
                system=system,
                alpha=a,
                beta=b,
                mean=rv.mean(),
                p05=rv.ppf(0.05),
                p50=rv.ppf(0.50),
                p95=rv.ppf(0.95),
            )
        )
    return pd.DataFrame(rows).set_index("system")
