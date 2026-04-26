"""
Scenario runner: loads from YAML, runs all scenarios, outputs comparison table.

Usage:
    python models/scenario_runner.py
    python models/scenario_runner.py --energy mid
    python models/scenario_runner.py --energy all
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models"))

from dominance_model import (
    ModelParams,
    SystemInputs,
    ScenarioResult,
    run_scenario,
    energy_score,
    gate_value,
    load_base_case,
    load_scenario_grid,
    params_from_base_case,
    build_system_inputs,
)


def run_all_scenarios(
    base_case_path: Path,
    grid_path: Path,
    energy_levels: list[str] | None = None,
    pair: str = "primary",
) -> list[ScenarioResult]:
    """Run all scenario × energy-level combinations.

    Args:
        base_case_path: path to base_case.yaml
        grid_path: path to scenario_grid.yaml
        energy_levels: subset of ["low", "mid", "high"]; None = all three
        pair: "primary" (H3 vs A4) or "robustness" (H1 vs A3)

    Returns:
        List of ScenarioResult objects.
    """
    bc = load_base_case(base_case_path)
    grid = load_scenario_grid(grid_path)
    params = params_from_base_case(bc)

    if energy_levels is None:
        energy_levels = ["low", "mid", "high"]

    if pair == "primary":
        human_sys, ai_sys = "H3", "A4"
    elif pair == "robustness":
        human_sys, ai_sys = "H1", "A3"
    else:
        raise ValueError(f"Unknown pair: {pair}")

    results = []
    scenarios = grid["scenarios"]

    for scenario_name, scenario_data in scenarios.items():
        # Scenario overrides apply to A4 only; robustness pair (A3) uses YAML values
        c_override = scenario_data.get("A4_closure") if ai_sys == "A4" else None
        r_override = scenario_data.get("A4_replication") if ai_sys == "A4" else None

        for elevel in energy_levels:
            human = build_system_inputs(bc, human_sys, energy_level=elevel)
            ai = build_system_inputs(
                bc, ai_sys, energy_level=elevel,
                closure_override=c_override,
                replication_override=r_override,
            )
            label = f"{scenario_name}|{elevel}"
            result = run_scenario(human, ai, params, label=label)
            results.append(result)

    return results


def print_results(results: list[ScenarioResult]) -> None:
    """Print a formatted comparison table."""
    header = (
        f"{'Scenario':<35} {'H_energy':>8} {'A_energy':>8} "
        f"{'H_gate':>7} {'A_gate':>7} {'H_D':>9} {'A_D':>10} "
        f"{'Ratio':>10} {'Cross':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.label:<35} {r.human_energy:>8.3f} {r.ai_energy:>8.4f} "
            f"{r.human_gate:>7.4f} {r.ai_gate:>7.4f} {r.human_score:>9.4f} "
            f"{r.ai_score:>10.6f} {r.ratio:>10.6f} {str(r.crossover):>6}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run dominance model scenarios")
    parser.add_argument(
        "--energy", default="all",
        help="Energy level: low, mid, high, or all (default: all)"
    )
    parser.add_argument(
        "--pair", default="primary",
        choices=["primary", "robustness"],
        help="System pair to compare (default: primary = H3 vs A4)"
    )
    args = parser.parse_args()

    base_path = PROJECT_ROOT / "data" / "assumptions" / "base_case.yaml"
    grid_path = PROJECT_ROOT / "data" / "assumptions" / "scenario_grid.yaml"

    if args.energy == "all":
        energy_levels = ["low", "mid", "high"]
    else:
        energy_levels = [args.energy]

    results = run_all_scenarios(base_path, grid_path, energy_levels, pair=args.pair)
    print_results(results)


if __name__ == "__main__":
    main()
