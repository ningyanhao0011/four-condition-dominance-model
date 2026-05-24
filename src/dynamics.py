"""Module C - time-series dynamics for S1/S2/S3 and 2030/2035 projection.

S1 has the richest longitudinal data and is fitted with a logistic
trajectory plus bootstrap. S2 and S3 are sparser and are fitted as
linear (S2) or exponential (S3) trajectories with wide uncertainty.

The output is a per-year posterior of (S1, S2, S3) at any target year y,
which Module A's MC sampler then consumes to produce posterior
distributions of D(A4)/D(H3) at year y.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

DATA = Path(__file__).resolve().parents[1] / "data" / "longitudinal.csv"


def load_longitudinal() -> pd.DataFrame:
    return pd.read_csv(DATA)


# ----------------------------------------------------------------------
# Functional forms
# ----------------------------------------------------------------------

def logistic(t, K, r, t0):
    """Logistic curve with ceiling K, growth rate r, midpoint t0."""
    return K / (1.0 + np.exp(-r * (t - t0)))


def linear(t, m, c):
    return m * (t - 2020.0) + c


def exponential(t, a, b, c):
    """y = a + b * exp(c * (t - 2023))."""
    return a + b * np.exp(c * (t - 2023.0))


# ----------------------------------------------------------------------
# Per-subcomponent fitters
# ----------------------------------------------------------------------

@dataclass
class FitResult:
    name: str  # "S1" | "S2" | "S3"
    func_kind: str  # "logistic" | "linear" | "exponential"
    params: np.ndarray  # central fit
    boot_params: np.ndarray  # shape (B, n_params)
    years_obs: np.ndarray  # observation years
    obs_low: np.ndarray
    obs_mid: np.ndarray
    obs_high: np.ndarray


def _bootstrap_samples(df_sub: pd.DataFrame, B: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Generate B noisy resamples of the observed mid values within
    [low, high] bounds. We use a triangular distribution as a simple
    asymmetric noise model around (low, mid, high)."""
    samples = []
    for _ in range(B):
        ys = []
        for _, row in df_sub.iterrows():
            y = rng.triangular(row["value_low"], row["value_mid"], row["value_high"])
            ys.append(y)
        samples.append(np.array(ys))
    return samples


def fit_S1(df: pd.DataFrame, B: int = 500, rng: np.random.Generator | None = None) -> FitResult:
    """Logistic fit with bootstrap. K bounded to [0.7, 0.99]."""
    rng = rng or np.random.default_rng(0)
    sub = df[df["subcomponent"] == "S1"].sort_values("year").reset_index(drop=True)
    t_obs = sub["year"].to_numpy(dtype=float)
    y_obs = sub["value_mid"].to_numpy(dtype=float)

    p0 = [0.9, 0.6, 2025.0]
    bounds = ([0.7, 0.1, 2018.0], [0.99, 2.0, 2032.0])
    popt, _ = curve_fit(logistic, t_obs, y_obs, p0=p0, bounds=bounds)

    boot_params = []
    for sample in _bootstrap_samples(sub, B, rng):
        try:
            p, _ = curve_fit(logistic, t_obs, sample, p0=popt, bounds=bounds, maxfev=4000)
            boot_params.append(p)
        except Exception:
            continue
    boot = np.array(boot_params)

    return FitResult(
        name="S1", func_kind="logistic", params=popt, boot_params=boot,
        years_obs=t_obs, obs_low=sub["value_low"].to_numpy(),
        obs_mid=sub["value_mid"].to_numpy(), obs_high=sub["value_high"].to_numpy(),
    )


def fit_S2(df: pd.DataFrame, B: int = 500, rng: np.random.Generator | None = None) -> FitResult:
    """Linear fit (sparse data)."""
    rng = rng or np.random.default_rng(1)
    sub = df[df["subcomponent"] == "S2"].sort_values("year").reset_index(drop=True)
    t_obs = sub["year"].to_numpy(dtype=float)
    y_obs = sub["value_mid"].to_numpy(dtype=float)

    p0 = [0.03, 0.05]
    bounds = ([0.0, 0.0], [0.10, 0.30])
    popt, _ = curve_fit(linear, t_obs, y_obs, p0=p0, bounds=bounds)

    boot_params = []
    for sample in _bootstrap_samples(sub, B, rng):
        try:
            p, _ = curve_fit(linear, t_obs, sample, p0=popt, bounds=bounds, maxfev=4000)
            boot_params.append(p)
        except Exception:
            continue
    boot = np.array(boot_params)

    return FitResult(
        name="S2", func_kind="linear", params=popt, boot_params=boot,
        years_obs=t_obs, obs_low=sub["value_low"].to_numpy(),
        obs_mid=sub["value_mid"].to_numpy(), obs_high=sub["value_high"].to_numpy(),
    )


def fit_S3(df: pd.DataFrame, B: int = 500, rng: np.random.Generator | None = None) -> FitResult:
    """Linear fit. Only three observations spanning 2023-2025 — an
    exponential would over-extrapolate. Linear is the most defensible
    functional form under this sparsity (and the most conservative for
    projecting whether the closure gate will open).
    """
    rng = rng or np.random.default_rng(2)
    sub = df[df["subcomponent"] == "S3"].sort_values("year").reset_index(drop=True)
    t_obs = sub["year"].to_numpy(dtype=float)
    y_obs = sub["value_mid"].to_numpy(dtype=float)

    p0 = [0.010, 0.010]
    bounds = ([0.0, 0.0], [0.05, 0.10])  # slope <= 0.05/year, intercept <= 0.10
    popt, _ = curve_fit(linear, t_obs, y_obs, p0=p0, bounds=bounds)

    boot_params = []
    for sample in _bootstrap_samples(sub, B, rng):
        try:
            p, _ = curve_fit(linear, t_obs, sample, p0=popt, bounds=bounds, maxfev=4000)
            boot_params.append(p)
        except Exception:
            continue
    boot = np.array(boot_params)

    return FitResult(
        name="S3", func_kind="linear", params=popt, boot_params=boot,
        years_obs=t_obs, obs_low=sub["value_low"].to_numpy(),
        obs_mid=sub["value_mid"].to_numpy(), obs_high=sub["value_high"].to_numpy(),
    )


# ----------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------

def project(fit: FitResult, year: float | np.ndarray) -> np.ndarray:
    """Evaluate fit at year(s) using bootstrap parameters.

    Returns shape (n_bootstrap, n_year) array of projected values.
    """
    year = np.atleast_1d(np.asarray(year, dtype=float))
    if fit.func_kind == "logistic":
        return np.array([logistic(year, *p) for p in fit.boot_params])
    if fit.func_kind == "linear":
        return np.array([linear(year, *p) for p in fit.boot_params])
    if fit.func_kind == "exponential":
        return np.array([exponential(year, *p) for p in fit.boot_params])
    raise ValueError(fit.func_kind)


def project_quantiles(fit: FitResult, years: np.ndarray, qs=(0.05, 0.50, 0.95)) -> pd.DataFrame:
    arr = project(fit, years)  # (B, T)
    rows = []
    for j, y in enumerate(years):
        q = np.quantile(arr[:, j], qs)
        clip = lambda v: float(np.clip(v, 0.0, 1.0))  # noqa: E731
        rows.append(dict(year=float(y), p05=clip(q[0]), p50=clip(q[1]), p95=clip(q[2])))
    return pd.DataFrame(rows)
