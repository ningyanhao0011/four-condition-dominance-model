"""Reproduce the headline numerical claims of the paper.

Run from the repository root:
    python examples/run_analysis.py

This script demonstrates the core model functionality without producing
figures. Figures are generated separately in a companion plotting package
not included here (the model code is journal-replication-focused).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow importing src.* from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.model import dominance_ratio
from src.aggregation import aggregate
from src.nested_fitness import phi, quadrant
from src.resilience import fit_all as fit_Z_snapshot, sample_Z
from src.resilience_dynamics import fit_all_Z, project_Z
from src.dynamics import load_longitudinal, fit_S1, fit_S2, fit_S3, project
from src.priors import (
    sample_S1, sample_S2, sample_S3, sample_R_A4,
    sample_C_H3, sample_R_H3, sample_E_A4, sample_E_H3,
    sample_k, sample_x_star, sample_agg_rule,
)
from src.sensitivity import sobol_indices, sobol_table
from src.mc_sampler import run_mc, summary


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------
# 1. v2 deterministic baseline
# ---------------------------------------------------------------------
section("1. v2 deterministic Strict-current baseline")
ratio_v2 = dominance_ratio(
    C_A=0.02, R_A=0.08, E_A=0.06, Z_A=1.0,
    C_B=0.90, R_B=0.70, E_B=7.6, Z_B=1.0,
    k=10.0, x_star=0.5, g_form="linear",
)
print(f"D(A4) / D(H3) = {ratio_v2:.3e}")
print("(Manuscript reports 1.10e-6 for this scenario.)")


# ---------------------------------------------------------------------
# 2. Snapshot resilience Z (Module B)
# ---------------------------------------------------------------------
section("2. Snapshot resilience Z (Module B)")
cached_Z = fit_Z_snapshot()
for system, (a, b) in cached_Z.items():
    mean = a / (a + b)
    print(f"  {system:9s}  Beta(alpha={a:7.3f}, beta={b:6.3f}), mean={mean:.3f}")


# ---------------------------------------------------------------------
# 3. Monte Carlo over 12 priors (Module A)
# ---------------------------------------------------------------------
section("3. Monte Carlo posterior of D(A4)/D(H3), N=10,000 (Module A)")
df = run_mc(n=10_000, seed=42, z_kind="long", cached_z=cached_Z)
s = summary(df)
print(s.round(5))
p95 = float(df["ratio"].quantile(0.95))
print(f"\n  Pre-registered falsifier check:")
print(f"  P95 of D(A4)/D(H3) = {p95:.3e}")
print(f"  Falsifier threshold (strict): 0.01")
print(f"  Falsified: {p95 > 0.01}")


# ---------------------------------------------------------------------
# 4. Sobol indices (Module A continued)
# ---------------------------------------------------------------------
section("4. Saltelli (2010) Pick-Freeze Sobol indices (min aggregation)")
sob = sobol_indices(N=512, agg_rule="min", seed=42, bootstrap=50)
tab = sobol_table(sob)
print(tab.round(3))


# ---------------------------------------------------------------------
# 5. Dynamic projection of S1, S2, S3 (Module C)
# ---------------------------------------------------------------------
section("5. Longitudinal projection of S1/S2/S3 to 2030 and 2035")
df_long = load_longitudinal()
fS1 = fit_S1(df_long, B=200)
fS2 = fit_S2(df_long, B=200)
fS3 = fit_S3(df_long, B=200)
for year in [2025.0, 2030.0, 2035.0]:
    s1 = float(np.median(project(fS1, np.array([year])).flatten()))
    s2 = float(np.median(project(fS2, np.array([year])).flatten()))
    s3 = float(np.median(project(fS3, np.array([year])).flatten()))
    print(f"  Year {int(year)}: S1={s1:.3f}  S2={s2:.3f}  S3={s3:.3f}")


# ---------------------------------------------------------------------
# 6. Time-varying resilience Z(t) (Module B-prime)
# ---------------------------------------------------------------------
section("6. Time-varying Z trajectories")
z_fits = fit_all_Z()
for year in [2010.0, 2025.0, 2035.0]:
    print(f"  Year {int(year)}:")
    for system, fit in z_fits.items():
        med = float(np.median(project_Z(fit, np.array([year])).flatten()))
        print(f"    Z_{system:9s} = {med:.3f}")


# ---------------------------------------------------------------------
# 7. Nested fitness phi(A4|H3) (Module D)
# ---------------------------------------------------------------------
section("7. Nested fitness phi(A4|H3) = sigma(S3; k, x*)")
scenarios = [
    ("Strict current", 0.02),
    ("Semi-autonomous", 0.03),
    ("Org. propagating", 0.05),
    ("Technogenesis*", 0.30),
    ("Full autonomy*", 0.80),
]
for name, s3 in scenarios:
    p = float(phi(s3, k=10.0, x_star=0.5))
    q = quadrant(0.20, s3, gate_threshold=0.5)  # current AI R_auto median
    print(f"  {name:18s} S3={s3:.2f}  phi={p:.4f}  quadrant={q}")


# ---------------------------------------------------------------------
# 8. Selection-pressure dynamics (Module D continued)
# ---------------------------------------------------------------------
section("8. Selection-pressure 25-year trajectories from current AI position")
from src.selection_pressure import integrate_trajectory

R0, S0 = 0.20, 0.03
for alpha, label in [(0.95, "breeder"), (0.50, "mixed"), (0.05, "ecosystem")]:
    traj = integrate_trajectory(R0=R0, S0=S0, alpha=alpha, years=25.0)
    R_end, S_end = float(traj[-1, 0]), float(traj[-1, 1])
    print(f"  alpha={alpha:.2f} ({label}):  25-yr end -> R_auto={R_end:.3f}, S3={S_end:.3f}")


print()
print("=" * 72)
print("  END — all headline numbers reproduced.")
print("=" * 72)
