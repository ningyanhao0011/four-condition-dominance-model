"""
collect_results.py
Run all model computations and print structured results for table population.

Usage (from simulation_code/ root):
    python scripts/collect_results.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.dominance_model import (
    load_base_case, load_scenario_grid, params_from_base_case,
    SystemInputs, dominance_score, threshold_gate
)

bc   = load_base_case(PROJECT_ROOT / 'data' / 'assumptions' / 'base_case.yaml')
grid = load_scenario_grid(PROJECT_ROOT / 'data' / 'assumptions' / 'scenario_grid.yaml')
params = params_from_base_case(bc)

A4_E = bc['dissipation']['A4']['value']
H3_C = bc['closure']['H3']['value']
H3_R = bc['replication']['H3']['value']
H3_E = bc['dissipation']['H3']['value']

# ── H3 reference ─────────────────────────────────────────────────────────────
print("=== H3 reference ===")
h3_by_level = {}
for el in ['low', 'mid', 'high']:
    c, r, e = H3_C[el], H3_R[el], H3_E[el]
    h3 = SystemInputs('H3', C=c, R=r, E=e, Z=1.0)
    g  = threshold_gate(c, r, params.C_star, params.R_star, params.k_C, params.k_R)
    D  = dominance_score(h3, params)
    h3_by_level[el] = (h3, D, g)
    print(f"  {el}: C={c} R={r} E={e} TW  gate={g:.4f}  D={D:.4f}")

h3_mid, D_h3_mid, g_h3_mid = h3_by_level['mid']
print(f"\nH3 mid reference: gate={g_h3_mid:.4f}  D={D_h3_mid:.4f}")

# ── A4 scenario grid ─────────────────────────────────────────────────────────
print("\n=== A4 scenario results (ratio vs H3 mid) ===")
header = f"{'Scenario':<26} {'C':>5} {'R':>5} {'Gate':>8} {'ratio_low':>10} {'ratio_mid':>10} {'ratio_hi':>10} {'cross?':>6}"
print(header)
scenario_rows = []
for sname, sdata in grid['scenarios'].items():
    c, r = sdata['A4_closure'], sdata['A4_replication']
    g    = threshold_gate(c, r, params.C_star, params.R_star, params.k_C, params.k_R)
    ratios = {}
    for el in ['low', 'mid', 'high']:
        a4 = SystemInputs('A4', C=c, R=r, E=A4_E[el], Z=1.0)
        ratios[el] = dominance_score(a4, params) / D_h3_mid
    cross = ratios['high'] >= 1.0
    row = dict(scenario=sname, C=c, R=r, gate=g,
               ratio_low=ratios['low'], ratio_mid=ratios['mid'], ratio_high=ratios['high'],
               crossover=cross)
    scenario_rows.append(row)
    print(f"  {sname:<26} {c:>5.2f} {r:>5.2f} {g:>8.5f} "
          f"{ratios['low']:>10.3e} {ratios['mid']:>10.3e} {ratios['high']:>10.3e} {str(cross):>6}")

# ── Inverse crossover ─────────────────────────────────────────────────────────
print("\n=== Inverse crossover ===")
inverse_rows = []
for row in scenario_rows:
    g = row['gate']
    if g > 1e-12:
        e_need = D_h3_mid / g
        mult   = e_need / A4_E['mid']
    else:
        e_need, mult = float('inf'), float('inf')
    inverse_rows.append(dict(scenario=row['scenario'], gate=g,
                             E_crossover_TW=e_need, multiple_of_current=mult))
    mult_str = f"{mult:>10.0f}" if mult != float('inf') else "         inf"
    print(f"  {row['scenario']:<26} gate={g:.5f}  E_need={e_need:>10.1f} TW  {mult_str}x")

# ── A5 speculative ────────────────────────────────────────────────────────────
print("\n=== A5 speculative ===")
a5_specs = [
    ('conservative', 0.70, 0.60, 1.0),
    ('moderate',     0.85, 0.75, 5.0),
    ('strong',       0.95, 0.90, 10.0),
]
a5_rows = []
for label, c, r, e in a5_specs:
    a5 = SystemInputs('A5', C=c, R=r, E=e, Z=1.0)
    D  = dominance_score(a5, params)
    g  = threshold_gate(c, r, params.C_star, params.R_star, params.k_C, params.k_R)
    ratio = D / D_h3_mid
    cross = D > D_h3_mid
    a5_rows.append(dict(label=label, C=c, R=r, E_TW=e, gate=g, D=D, ratio=ratio, crossover=cross))
    print(f"  A5_{label:<12}: C={c} R={r} E={e} TW  gate={g:.4f}  D={D:.4f}  ratio={ratio:.4f}  cross={cross}")

# ── Layer A vs B ──────────────────────────────────────────────────────────────
print("\n=== Layer A vs Layer B (strict_current_ai, mid energy) ===")
strict = next(r for r in scenario_rows if r['scenario'] == 'strict_current_ai')
g_a4  = strict['gate']
e_a4  = A4_E['mid']
e_h3  = H3_E['mid']
layerA = e_a4 / e_h3
layerB = (g_a4 * e_a4) / (g_h3_mid * e_h3)
print(f"  Layer A ratio (energy only): {layerA:.4f}")
print(f"  Layer B ratio (gated):       {layerB:.2e}")
print(f"  Gate suppression factor:     {layerA/layerB:.0f}x")

# ── Closure decomposition ─────────────────────────────────────────────────────
print("\n=== Closure decomposition (min-aggregation) ===")
decomp = {
    'strict_current_ai':       (0.35, 0.20, 0.02),
    'semi_autonomous_infra':   (0.45, 0.25, 0.03),
    'propagating_ai':          (0.55, 0.30, 0.05),
    'technogenesis':           (0.60, 0.40, 0.30),
    'full_autonomy':           (0.60, 0.55, 0.80),
}
wts = (0.2, 0.3, 0.5)
for sname, (s1, s2, s3) in decomp.items():
    c_min = min(s1, s2, s3)
    c_wt  = wts[0]*s1 + wts[1]*s2 + wts[2]*s3
    c_scl = grid['scenarios'].get(sname, {}).get('A4_closure', '?')
    print(f"  {sname:<26}: S1={s1} S2={s2} S3={s3}  C_min={c_min:.2f}  C_wt={c_wt:.2f}  C_scalar={c_scl}")

print("\nDone.")
