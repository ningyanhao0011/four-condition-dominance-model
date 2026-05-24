"""Module A - sensitivity analysis utilities.

Three methods:

1) Spearman rank correlations - cheap, model-free, model-agnostic.
2) First-order Sobol indices via the Pick-Freeze estimator
   (Saltelli 2002, refined by Saltelli et al. 2010).
3) Total Sobol indices (capture interactions).

The Sobol scheme uses (d + 2) * N model evaluations to estimate both
first-order S_i and total ST_i indices, where d is the number of
continuous inputs and N is the base sample size. With N = 2048 and
d = 10 the total cost is 24,576 evaluations - fast for the dominance
model.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


CONTINUOUS_INPUTS = ("S1", "S2", "S3", "R_A4", "C_A4", "C_H3", "R_H3", "E_A4", "E_H3", "k", "x_star", "Z_A4", "Z_H3")


def rank_correlations(df: pd.DataFrame, target: str = "log10_ratio") -> pd.DataFrame:
    rows = []
    for col in CONTINUOUS_INPUTS:
        if col not in df.columns:
            continue
        rho, p = stats.spearmanr(df[col], df[target])
        rows.append(dict(input=col, spearman_rho=rho, p_value=p))
    out = pd.DataFrame(rows).set_index("input")
    out["abs_rho"] = out["spearman_rho"].abs()
    return out.sort_values("abs_rho", ascending=False)


def stratum_means(df: pd.DataFrame, by: str = "rule", target: str = "log10_ratio") -> pd.DataFrame:
    return df.groupby(by)[target].agg(["mean", "std", "count"])


def falsifier_check(df: pd.DataFrame, threshold: float = 0.1) -> dict:
    """Pre-registered: framework is falsified if P95(ratio) > threshold."""
    p95 = df["ratio"].quantile(0.95)
    p99 = df["ratio"].quantile(0.99)
    exceed = (df["ratio"] > threshold).mean()
    falsified = p95 > threshold
    return dict(
        threshold=threshold, p95=float(p95), p99=float(p99),
        exceedance_fraction=float(exceed),
        falsified=bool(falsified),
    )


# ---------------------------------------------------------------------
# Sobol indices via Pick-Freeze (Saltelli 2002 / 2010)
# ---------------------------------------------------------------------

@dataclass
class SobolResult:
    inputs: list[str]
    S1: np.ndarray      # first-order indices, shape (d,)
    ST: np.ndarray      # total-order indices,  shape (d,)
    S1_conf: np.ndarray # bootstrap 95% CI half-width
    ST_conf: np.ndarray
    var_Y: float
    N: int


# Inverse-CDF transforms for each prior (must match priors.py).
def _u_to_param(u: np.ndarray, name: str) -> np.ndarray:
    """Map uniform u in [0, 1] to the corresponding prior draw."""
    if name == "S1":
        return stats.beta(2.5, 2.5).ppf(u) * 0.30 + 0.30   # [0.30, 0.60]
    if name == "S2":
        return stats.beta(2.0, 4.0).ppf(u) * 0.25 + 0.10   # [0.10, 0.35]
    if name == "S3":
        x = stats.lognorm(s=0.5, scale=np.exp(np.log(0.02))).ppf(u)
        return np.clip(x, 0.001, 0.10)
    if name == "R_A4":
        return stats.beta(1.5, 5.0).ppf(u) * 0.75 + 0.05
    if name == "C_H3":
        return stats.beta(8.0, 2.0).ppf(u) * 0.19 + 0.80
    if name == "R_H3":
        return stats.beta(6.0, 2.0).ppf(u) * 0.30 + 0.55
    if name == "E_A4":
        return stats.lognorm(s=0.35, scale=np.exp(np.log(0.06))).ppf(u)
    if name == "E_H3":
        return np.clip(stats.norm(7.6, 1.0).ppf(u), 3.0, 12.0)
    if name == "k":
        return 2.0 + u * 18.0
    if name == "x_star":
        return 0.3 + u * 0.3
    raise ValueError(name)


SOBOL_INPUTS = (
    "S1", "S2", "S3", "R_A4", "C_H3", "R_H3", "E_A4", "E_H3", "k", "x_star",
)


def _evaluate_model(U: np.ndarray, agg_rule: str = "min") -> np.ndarray:
    """Evaluate the dominance ratio for a (N x d) uniform sample.

    Returns Y = log10(D(A4) / D(H3)).
    """
    from src.model import dominance
    from src.aggregation import RULES
    cols = {name: _u_to_param(U[:, j], name) for j, name in enumerate(SOBOL_INPUTS)}

    s1, s2, s3 = cols["S1"], cols["S2"], cols["S3"]
    c_a4 = RULES[agg_rule](s1, s2, s3)
    r_a4 = cols["R_A4"]
    c_h3 = cols["C_H3"]
    r_h3 = cols["R_H3"]
    e_a4 = cols["E_A4"]
    e_h3 = cols["E_H3"]
    k = cols["k"]
    xs = cols["x_star"]
    z_a4 = np.ones_like(s1)  # Sobol holds Z fixed; vary independently if needed
    z_h3 = np.ones_like(s1)

    d_a4 = dominance(c_a4, r_a4, e_a4, z_a4, k=k, x_star=xs)
    d_h3 = dominance(c_h3, r_h3, e_h3, z_h3, k=k, x_star=xs)
    ratio = d_a4 / d_h3
    return np.log10(np.maximum(ratio, 1e-30))


def sobol_indices(
    N: int = 2048,
    agg_rule: str = "min",
    seed: int = 42,
    bootstrap: int = 200,
) -> SobolResult:
    """Pick-Freeze (Saltelli 2010) first-order and total Sobol indices.

    Method: draw two independent N x d uniform matrices A, B. For each
    parameter i, build AB_i = B with column i copied from A. Run model
    on A, B, AB_1, ..., AB_d. The total cost is (d + 2) * N evaluations.
    """
    rng = np.random.default_rng(seed)
    d = len(SOBOL_INPUTS)

    A = rng.uniform(0, 1, size=(N, d))
    B = rng.uniform(0, 1, size=(N, d))

    Y_A = _evaluate_model(A, agg_rule=agg_rule)
    Y_B = _evaluate_model(B, agg_rule=agg_rule)

    # AB_i = A with column i replaced by B[:, i]
    # (SALib convention; matches Saltelli 2010 Table 2 estimator b).
    Y_ABi = np.zeros((d, N))
    for i in range(d):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        Y_ABi[i] = _evaluate_model(AB_i, agg_rule=agg_rule)

    # Saltelli 2010 / Jansen 1999 estimators
    Y_all = np.concatenate([Y_A, Y_B])
    var_Y = float(np.var(Y_all, ddof=1))

    # First-order (Saltelli 2010, Table 2, estimator b):
    #   S_i = (1/N) sum  Y_B * (Y_AB_i - Y_A)  /  V(Y)
    S1 = np.array([
        np.mean(Y_B * (Y_ABi[i] - Y_A)) / var_Y
        for i in range(d)
    ])
    # Total-order (Jansen 1999):
    #   ST_i = (1/2N) sum  (Y_B - Y_AB_i)^2  /  V(Y)
    ST = np.array([
        0.5 * np.mean((Y_A - Y_ABi[i]) ** 2) / var_Y
        for i in range(d)
    ])

    # Bootstrap CI on indices
    S1_boot = np.zeros((bootstrap, d))
    ST_boot = np.zeros((bootstrap, d))
    for b in range(bootstrap):
        idx = rng.integers(0, N, size=N)
        Y_A_b = Y_A[idx]
        Y_B_b = Y_B[idx]
        Y_ABi_b = Y_ABi[:, idx]
        var_b = float(np.var(np.concatenate([Y_A_b, Y_B_b]), ddof=1))
        for i in range(d):
            S1_boot[b, i] = np.mean(Y_B_b * (Y_ABi_b[i] - Y_A_b)) / var_b
            ST_boot[b, i] = 0.5 * np.mean((Y_B_b - Y_ABi_b[i]) ** 2) / var_b
    S1_conf = 1.96 * np.std(S1_boot, axis=0)
    ST_conf = 1.96 * np.std(ST_boot, axis=0)

    return SobolResult(
        inputs=list(SOBOL_INPUTS),
        S1=S1, ST=ST, S1_conf=S1_conf, ST_conf=ST_conf,
        var_Y=var_Y, N=N,
    )


def sobol_table(res: SobolResult) -> pd.DataFrame:
    return pd.DataFrame(
        dict(
            input=res.inputs,
            S1=res.S1,
            S1_95ci=res.S1_conf,
            ST=res.ST,
            ST_95ci=res.ST_conf,
            interaction=res.ST - res.S1,
        )
    ).set_index("input").sort_values("ST", ascending=False)
