"""Unit tests for the four-condition dominance model.

Run from v3_code/ as:
    python -m pytest tests/
or:
    python tests/test_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import math
import numpy as np

from src.model import sigmoid, g_dissipation, dominance, dominance_ratio
from src.aggregation import (
    aggregate_min, aggregate_geomean, aggregate_wavg, aggregate_max,
    aggregate, RULES,
)
from src.nested_fitness import phi, joint_gate, quadrant


# ------------------------------------------------------------
# sigmoid()
# ------------------------------------------------------------

def test_sigmoid_midpoint():
    """sigmoid at x = x* must equal 0.5 exactly."""
    assert math.isclose(sigmoid(0.5, k=10, x_star=0.5), 0.5, abs_tol=1e-12)
    assert math.isclose(sigmoid(0.2, k=5, x_star=0.2), 0.5, abs_tol=1e-12)


def test_sigmoid_monotone():
    """sigmoid must be monotone increasing in x."""
    xs = np.linspace(0, 1, 100)
    ys = sigmoid(xs, k=10, x_star=0.5)
    assert np.all(np.diff(ys) >= 0)


def test_sigmoid_steepness():
    """Larger k must produce a steeper transition."""
    s_low = sigmoid(0.6, k=2, x_star=0.5) - sigmoid(0.4, k=2, x_star=0.5)
    s_high = sigmoid(0.6, k=20, x_star=0.5) - sigmoid(0.4, k=20, x_star=0.5)
    assert s_high > s_low


def test_sigmoid_bounded():
    """sigmoid output must lie in [0, 1]. At large positive x the
    float64 representation saturates at exactly 1.0, which is fine.
    """
    for x in [-5, 0, 0.5, 1, 10]:
        y = float(sigmoid(x, k=10, x_star=0.5))
        assert 0.0 <= y <= 1.0


# ------------------------------------------------------------
# g_dissipation()
# ------------------------------------------------------------

def test_g_dissipation_forms():
    assert g_dissipation(4.0, "linear") == 4.0
    assert math.isclose(g_dissipation(4.0, "sqrt"), 2.0, abs_tol=1e-12)
    assert math.isclose(g_dissipation(9.0, "log"), math.log10(10.0), abs_tol=1e-12)


def test_g_dissipation_invalid():
    try:
        g_dissipation(1.0, "cubic")
        raise AssertionError("Expected ValueError for invalid form")
    except ValueError:
        pass


# ------------------------------------------------------------
# dominance()
# ------------------------------------------------------------

def test_dominance_zero_closure_relative():
    """If C is near zero, the gated D must be at least 100x smaller
    than the open-gate equivalent at the same E.
    """
    d_closed = dominance(C=0.01, R=0.7, E=10.0, Z=1.0, k=10, x_star=0.5)
    d_open = dominance(C=0.95, R=0.7, E=10.0, Z=1.0, k=10, x_star=0.5)
    assert d_open / d_closed > 100  # gate suppresses by >2 orders


def test_dominance_open_gates():
    """If C and R are near 1, D approaches E * Z * 1 * 1."""
    d = dominance(C=0.95, R=0.95, E=10.0, Z=0.9, k=10, x_star=0.5)
    assert d > 8.0  # close to 10 * 0.9 = 9


def test_dominance_ratio_v2_replication():
    """Reproduce the v2 Strict-current deterministic baseline."""
    ratio = dominance_ratio(
        C_A=0.02, R_A=0.08, E_A=0.06, Z_A=1.0,
        C_B=0.90, R_B=0.70, E_B=7.6, Z_B=1.0,
        k=10, x_star=0.5,
    )
    # v2 manuscript reports 1.10e-6 for this scenario
    assert 1e-7 < ratio < 1e-5
    assert math.isclose(ratio, 1.10e-6, rel_tol=0.10)  # within 10%


# ------------------------------------------------------------
# aggregation rules
# ------------------------------------------------------------

def test_aggregate_min():
    assert aggregate_min(0.1, 0.5, 0.3) == 0.1
    assert aggregate_min(0.5, 0.2, 0.8) == 0.2


def test_aggregate_geomean():
    g = aggregate_geomean(0.1, 0.5, 0.3)
    expected = (0.1 * 0.5 * 0.3) ** (1.0 / 3.0)
    assert math.isclose(float(g), expected, rel_tol=1e-9)


def test_aggregate_wavg_default():
    w = aggregate_wavg(0.1, 0.5, 0.3)
    expected = (0.1 + 0.5 + 0.3) / 3.0
    assert math.isclose(float(w), expected, rel_tol=1e-9)


def test_aggregate_geomean_with_zero():
    """Geomean must guard against zeros, not produce NaN/inf."""
    g = aggregate_geomean(0.0, 0.5, 0.3)
    assert np.isfinite(g) and g > 0


def test_aggregate_rules_ordering():
    """For positive inputs: min <= geomean <= wavg <= max."""
    s1, s2, s3 = 0.10, 0.30, 0.05
    m = float(aggregate_min(s1, s2, s3))
    g = float(aggregate_geomean(s1, s2, s3))
    w = float(aggregate_wavg(s1, s2, s3))
    mx = float(aggregate_max(s1, s2, s3))
    assert m <= g <= w <= mx


def test_aggregate_dispatcher():
    for name in ("min", "geomean", "wavg", "max"):
        assert callable(RULES[name])
        val = aggregate(0.1, 0.2, 0.3, rule=name)
        assert np.isfinite(val)


def test_aggregate_invalid_rule():
    try:
        aggregate(0.1, 0.2, 0.3, rule="median")
        raise AssertionError("Expected ValueError for invalid rule")
    except ValueError:
        pass


# ------------------------------------------------------------
# Nested fitness phi identity
# ------------------------------------------------------------

def test_phi_equals_sigmoid_S3():
    """The key identity: phi(A4|H3) = sigma(S3; k, x*)."""
    for s3 in [0.01, 0.05, 0.20, 0.50, 0.80, 0.95]:
        for k in [5, 10, 20]:
            for xs in [0.3, 0.5, 0.7]:
                assert math.isclose(
                    float(phi(s3, k=k, x_star=xs)),
                    float(sigmoid(s3, k=k, x_star=xs)),
                    rel_tol=1e-12,
                )


def test_phi_monotone_in_S3():
    """As S3 increases from 0 to 1, phi increases monotonically."""
    s3_grid = np.linspace(0, 1, 50)
    phi_vals = phi(s3_grid)
    assert np.all(np.diff(phi_vals) > 0)


def test_joint_gate_factorisation():
    """joint_gate(R, S3) = sigma(R) * sigma(S3)."""
    for r in [0.1, 0.5, 0.9]:
        for s in [0.05, 0.5, 0.85]:
            jg = float(joint_gate(r, s, k=10, x_star=0.5))
            sr = float(sigmoid(r, k=10, x_star=0.5))
            ss = float(sigmoid(s, k=10, x_star=0.5))
            assert math.isclose(jg, sr * ss, rel_tol=1e-12)


# ------------------------------------------------------------
# quadrant() classifier
# ------------------------------------------------------------

def test_quadrant_dependent_tool():
    assert "dependent tool" in quadrant(0.2, 0.05).lower()


def test_quadrant_mueller_ecosystem():
    q = quadrant(0.85, 0.10)
    assert "mueller" in q.lower() or "ecosystem" in q.lower()


def test_quadrant_a5_transition():
    assert "a5" in quadrant(0.9, 0.95).lower()


def test_quadrant_impossible():
    q = quadrant(0.1, 0.9)
    assert "impossible" in q.lower()


# ------------------------------------------------------------
# Resilience priors smoke test
# ------------------------------------------------------------

def test_resilience_module_smoke():
    from src.resilience import fit_all
    cached = fit_all()
    assert set(cached.keys()) == {"H3", "A4_short", "A4_long"}
    # Posterior means should respect H3 high, A4_short high, A4_long low
    from src.resilience import summary_table
    s = summary_table(cached)
    assert s.loc["H3", "p50"] > 0.85
    assert s.loc["A4_short", "p50"] > 0.85
    assert s.loc["A4_long", "p50"] < 0.50


# ------------------------------------------------------------
# Module C dynamics smoke test
# ------------------------------------------------------------

def test_dynamics_S1_fit_smoke():
    from src.dynamics import load_longitudinal, fit_S1, fit_S2, fit_S3, project
    df = load_longitudinal()
    f1 = fit_S1(df, B=50)
    proj = project(f1, np.array([2025.0, 2030.0]))
    assert proj.shape[1] == 2
    # S1 should be higher in 2030 than 2025 (logistic growth)
    assert np.median(proj[:, 1]) > np.median(proj[:, 0])


def test_dynamics_S3_below_threshold_2035():
    """A core finding: S3 trajectory stays below the closure gate
    threshold of 0.5 well past 2035 under defensible priors.
    """
    from src.dynamics import load_longitudinal, fit_S3, project
    df = load_longitudinal()
    f3 = fit_S3(df, B=50)
    proj = project(f3, np.array([2035.0]))
    assert float(np.quantile(proj.flatten(), 0.95)) < 0.5


# ------------------------------------------------------------
# Selection-pressure dynamics smoke test
# ------------------------------------------------------------

def test_selection_pressure_breeder_stays_dependent():
    """Under a fully breeder regime (alpha=1), the system should stay
    in the dependent-tool quadrant over 25 years.
    """
    from src.selection_pressure import integrate_trajectory
    traj = integrate_trajectory(R0=0.20, S0=0.03, alpha=1.0, years=25.0)
    final_R, final_S = traj[-1]
    assert final_S < 0.30  # well below A5 threshold


def test_selection_pressure_ecosystem_crosses_a5():
    """Under a fully ecosystem regime (alpha=0), the system should
    cross into the A5 quadrant within 30 years.
    """
    from src.selection_pressure import integrate_trajectory
    traj = integrate_trajectory(R0=0.20, S0=0.03, alpha=0.0, years=30.0)
    in_a5 = (traj[:, 0] >= 0.5) & (traj[:, 1] >= 0.5)
    assert in_a5.any()


# ------------------------------------------------------------
# Time-varying Z (Module B-prime) smoke tests
# ------------------------------------------------------------

def test_Z_trajectory_H3_roughly_constant():
    """H3 resilience should stay between 0.85 and 0.97 across 2010-2035."""
    from src.resilience_dynamics import fit_all_Z, project_Z
    fits = fit_all_Z()
    years = np.array([2010, 2020, 2025, 2030, 2035], dtype=float)
    arr = project_Z(fits["H3"], years)  # (B, T)
    median_traj = np.median(arr, axis=0)
    assert (median_traj >= 0.85).all() and (median_traj <= 0.97).all()


def test_Z_trajectory_A4_short_improves():
    """A4_short resilience should be monotonically non-decreasing
    over 2010-2035 (logistic-saturation expected).
    """
    from src.resilience_dynamics import fit_all_Z, project_Z
    fits = fit_all_Z()
    years = np.arange(2010, 2036, 5, dtype=float)
    arr = project_Z(fits["A4_short"], years)
    med = np.median(arr, axis=0)
    # Allow small numerical noise (max-violation tolerance 0.01)
    diffs = np.diff(med)
    assert (diffs >= -0.01).all()


def test_Z_trajectory_A4_long_bounded():
    """A4_long resilience must respect its 0.5 structural ceiling."""
    from src.resilience_dynamics import fit_all_Z, project_Z
    fits = fit_all_Z()
    years = np.array([2050.0], dtype=float)  # well past projection horizon
    arr = project_Z(fits["A4_long"], years)
    assert float(np.quantile(arr.flatten(), 0.95)) <= 0.55


def test_Z_trajectory_A4_long_below_H3():
    """At every projection year, median Z_A4_long must be below median Z_H3 —
    the substantive asymmetry must persist across the trajectory."""
    from src.resilience_dynamics import fit_all_Z, project_Z
    fits = fit_all_Z()
    years = np.arange(2010, 2036, 5, dtype=float)
    z_long = np.median(project_Z(fits["A4_long"], years), axis=0)
    z_h3 = np.median(project_Z(fits["H3"], years), axis=0)
    assert (z_long < z_h3).all()


# ------------------------------------------------------------
# Sobol indices basic sanity (small N for speed)
# ------------------------------------------------------------

def test_sobol_first_order_within_bounds():
    """First-order Sobol must be in [-eps, 1+eps] for any input
    (small negative values from MC noise are acceptable).
    """
    from src.sensitivity import sobol_indices
    res = sobol_indices(N=128, bootstrap=20, seed=1, agg_rule="min")
    assert (res.S1 > -0.05).all() and (res.S1 < 1.10).all()
    assert (res.ST > -0.05).all() and (res.ST < 1.10).all()


def test_sobol_k_dominates():
    """The k input must dominate first-order sensitivity under all
    rules — this is the central interpretive finding."""
    from src.sensitivity import sobol_indices, SOBOL_INPUTS
    res = sobol_indices(N=512, bootstrap=20, seed=1, agg_rule="min")
    k_idx = list(SOBOL_INPUTS).index("k")
    assert res.S1[k_idx] == max(res.S1)


# ------------------------------------------------------------
# Run as script
# ------------------------------------------------------------

if __name__ == "__main__":
    import traceback
    tests = [
        v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}  ({type(e).__name__}: {e})")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed (out of {passed + failed})")
    sys.exit(0 if failed == 0 else 1)
