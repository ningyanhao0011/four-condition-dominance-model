"""Time-varying resilience Z(t) for H3, A4_short, A4_long.

Snapshot Z (Module B) was a single Beta posterior fitted to all
disturbances pooled across time. The dynamic version fits a year-stamped
trajectory and projects forward.

Functional forms:
- Z_H3(t): essentially constant with small noise — linear with small slope.
- Z_A4_short(t): improving — logistic with ceiling near 1.0.
- Z_A4_long(t): improving slowly with structural ceiling — logistic with
  ceiling around 0.4-0.5 (semiconductor manufacturing remains entirely
  human-dependent even with full geographic diversification).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DATA = Path(__file__).resolve().parents[1] / "data" / "resilience_longitudinal.csv"


def linear(t, m, c):
    return m * (t - 2020.0) + c


def logistic_ceiling(t, K, r, t0):
    return K / (1.0 + np.exp(-r * (t - t0)))


@dataclass
class ZFitResult:
    system: str
    kind: str         # "linear" | "logistic"
    params: np.ndarray
    boot_params: np.ndarray
    t_obs: np.ndarray
    z_obs_mid: np.ndarray


def load_z_longitudinal() -> pd.DataFrame:
    return pd.read_csv(DATA)


def _bootstrap(df_sub: pd.DataFrame, B: int, rng: np.random.Generator):
    """Triangular noise around (Z_low, Z_mid, Z_high) per observation."""
    out = []
    for _ in range(B):
        ys = [rng.triangular(r.Z_low, r.Z_mid, r.Z_high) for r in df_sub.itertuples()]
        out.append(np.array(ys))
    return out


def fit_Z_H3(df: pd.DataFrame, B: int = 400, rng=None) -> ZFitResult:
    """H3 fitted linearly with tight slope prior (slope small)."""
    rng = rng or np.random.default_rng(10)
    sub = df[df["system"] == "H3"].sort_values("year").reset_index(drop=True)
    t = sub["year"].to_numpy(dtype=float)
    y = sub["Z_mid"].to_numpy(dtype=float)
    p0 = [0.0, 0.93]
    bounds = ([-0.005, 0.80], [0.005, 1.00])  # tiny slope allowed
    popt, _ = curve_fit(linear, t, y, p0=p0, bounds=bounds)
    boot = []
    for sample in _bootstrap(sub, B, rng):
        try:
            p, _ = curve_fit(linear, t, sample, p0=popt, bounds=bounds, maxfev=4000)
            boot.append(p)
        except Exception:
            continue
    return ZFitResult("H3", "linear", popt, np.array(boot), t, y)


def fit_Z_A4_short(df: pd.DataFrame, B: int = 400, rng=None) -> ZFitResult:
    """Logistic with K ceiling near 0.99."""
    rng = rng or np.random.default_rng(11)
    sub = df[df["system"] == "A4_short"].sort_values("year").reset_index(drop=True)
    t = sub["year"].to_numpy(dtype=float)
    y = sub["Z_mid"].to_numpy(dtype=float)
    p0 = [0.99, 0.25, 2015.0]
    bounds = ([0.94, 0.05, 2000.0], [1.00, 1.0, 2025.0])
    popt, _ = curve_fit(logistic_ceiling, t, y, p0=p0, bounds=bounds)
    boot = []
    for sample in _bootstrap(sub, B, rng):
        try:
            p, _ = curve_fit(logistic_ceiling, t, sample, p0=popt, bounds=bounds, maxfev=4000)
            boot.append(p)
        except Exception:
            continue
    return ZFitResult("A4_short", "logistic", popt, np.array(boot), t, y)


def fit_Z_A4_long(df: pd.DataFrame, B: int = 400, rng=None) -> ZFitResult:
    """Logistic with structural ceiling K capped at ~0.5.

    Semiconductor manufacturing remains entirely human-dependent even
    with full geographic diversification, so the ceiling reflects 'best
    case under achievable diversification', not 'A4 substrate
    self-sufficiency'.
    """
    rng = rng or np.random.default_rng(12)
    sub = df[df["system"] == "A4_long"].sort_values("year").reset_index(drop=True)
    t = sub["year"].to_numpy(dtype=float)
    y = sub["Z_mid"].to_numpy(dtype=float)
    p0 = [0.40, 0.20, 2030.0]
    bounds = ([0.15, 0.05, 2020.0], [0.50, 0.80, 2040.0])
    popt, _ = curve_fit(logistic_ceiling, t, y, p0=p0, bounds=bounds)
    boot = []
    for sample in _bootstrap(sub, B, rng):
        try:
            p, _ = curve_fit(logistic_ceiling, t, sample, p0=popt, bounds=bounds, maxfev=4000)
            boot.append(p)
        except Exception:
            continue
    return ZFitResult("A4_long", "logistic", popt, np.array(boot), t, y)


def project_Z(fit: ZFitResult, years: np.ndarray) -> np.ndarray:
    """Returns shape (n_boot, n_year) of projected Z values, clipped to [0,1]."""
    years = np.atleast_1d(np.asarray(years, dtype=float))
    if fit.kind == "linear":
        arr = np.array([linear(years, *p) for p in fit.boot_params])
    else:
        arr = np.array([logistic_ceiling(years, *p) for p in fit.boot_params])
    return np.clip(arr, 0.0, 1.0)


def fit_all_Z() -> dict[str, ZFitResult]:
    df = load_z_longitudinal()
    return {
        "H3":       fit_Z_H3(df),
        "A4_short": fit_Z_A4_short(df),
        "A4_long":  fit_Z_A4_long(df),
    }
