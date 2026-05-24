"""Module A - Monte Carlo sampler that composes priors + Z + model.

Returns a pandas DataFrame with one row per Monte Carlo draw containing
all sampled parameters and the derived D(A4), D(H3), D(A4)/D(H3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import priors
from src.model import dominance
from src.aggregation import RULES
from src.resilience import sample_Z, fit_all


def run_mc(
    n: int = 10_000,
    seed: int = 42,
    z_kind: str = "long",  # "long" | "short" | "none"
    cached_z: dict | None = None,
) -> pd.DataFrame:
    """Run n Monte Carlo draws and return a tidy DataFrame.

    z_kind selects which resilience class to use for A4:
      - "long":  Z_A4_long  (supply-chain disturbance class; primary)
      - "short": Z_A4_short (engineering disturbance class)
      - "none":  Z_A4 = 1   (reproduces v2 deterministic Z)
    H3 always uses Z_H3 unless z_kind == 'none'.
    """
    rng = np.random.default_rng(seed)

    s1 = priors.sample_S1(n, rng)
    s2 = priors.sample_S2(n, rng)
    s3 = priors.sample_S3(n, rng)
    r_a4 = priors.sample_R_A4(n, rng)
    c_h3 = priors.sample_C_H3(n, rng)
    r_h3 = priors.sample_R_H3(n, rng)
    e_a4 = priors.sample_E_A4(n, rng)
    e_h3 = priors.sample_E_H3(n, rng)
    k = priors.sample_k(n, rng)
    xs = priors.sample_x_star(n, rng)
    rule = priors.sample_agg_rule(n, rng)

    if z_kind == "none":
        z_a4 = np.ones(n)
        z_h3 = np.ones(n)
    else:
        if cached_z is None:
            cached_z = fit_all()
        z_a4 = sample_Z("A4_long" if z_kind == "long" else "A4_short", n, rng, cached=cached_z)
        z_h3 = sample_Z("H3", n, rng, cached=cached_z)

    # Aggregate closure per row according to the sampled rule
    c_a4 = np.empty(n)
    for r_name, agg_fn in RULES.items():
        mask = rule == r_name
        if mask.any():
            c_a4[mask] = agg_fn(s1[mask], s2[mask], s3[mask])

    # Dominance per draw (vectorised because k and xs vary per draw)
    d_a4 = dominance(c_a4, r_a4, e_a4, z_a4, k=k, x_star=xs, g_form="linear")
    d_h3 = dominance(c_h3, r_h3, e_h3, z_h3, k=k, x_star=xs, g_form="linear")
    ratio = d_a4 / d_h3

    df = pd.DataFrame(
        dict(
            S1=s1, S2=s2, S3=s3, R_A4=r_a4, C_A4=c_a4,
            C_H3=c_h3, R_H3=r_h3,
            E_A4=e_a4, E_H3=e_h3,
            k=k, x_star=xs, rule=rule,
            Z_A4=z_a4, Z_H3=z_h3,
            D_A4=d_a4, D_H3=d_h3, ratio=ratio,
            log10_ratio=np.log10(ratio),
        )
    )
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute quantile summary of dominance ratio overall and per rule."""
    qs = [0.05, 0.25, 0.50, 0.75, 0.95]

    def _row(values: pd.Series, stratum: str) -> dict:
        q = values.quantile(qs)
        return {
            "stratum": stratum,
            "P05": float(q.iloc[0]),
            "P25": float(q.iloc[1]),
            "P50": float(q.iloc[2]),
            "P75": float(q.iloc[3]),
            "P95": float(q.iloc[4]),
            "mean": float(values.mean()),
            "n": int(len(values)),
        }

    rows = [_row(df["ratio"], "overall")]
    for rule, sub in df.groupby("rule"):
        rows.append(_row(sub["ratio"], f"rule={rule}"))
    return pd.DataFrame(rows).set_index("stratum")
