"""
Dominance model for the four-condition framework — v2.

Two-layer architecture:
  Layer A (energy backbone): D_energy = E
  Layer B (threshold gate):  D_gated = Gate(C, R) × E

Base case: h(Z) = 1 (resilience disabled).
Full model: D = Gate(C, R) × g(E) × h(Z)

Key design decisions:
- Sigmoid thresholds for closure and replication (threshold_design.md)
- A4 closure and replication are scenario variables, not point estimates
- Three combination forms implemented for A9 comparison
- Three temporal modes implemented for A10 comparison
- All empirical values read from YAML; nothing hard-coded

See: definitions.md, threshold_design.md, proxy_design.md, base_case.yaml, scenario_grid.yaml
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Threshold functions (Layer B)
# ---------------------------------------------------------------------------

def sigmoid_threshold(x: float, x_star: float, k: float) -> float:
    """Sigmoid gate for a single condition.

    σ(x) = 1 / (1 + exp(-k(x - x*)))
    """
    exponent = -k * (x - x_star)
    exponent = max(min(exponent, 500), -500)
    return 1.0 / (1.0 + math.exp(exponent))


def binary_threshold(x: float, x_star: float) -> float:
    """Hard binary threshold (robustness alternative)."""
    return 1.0 if x >= x_star else 0.0


def threshold_gate(
    C: float,
    R: float,
    C_star: float,
    R_star: float,
    k_C: float,
    k_R: float,
    mode: Literal["sigmoid", "binary"] = "sigmoid",
) -> float:
    """Combined threshold gate for closure and replication."""
    if mode == "sigmoid":
        return sigmoid_threshold(C, C_star, k_C) * sigmoid_threshold(R, R_star, k_R)
    elif mode == "binary":
        return binary_threshold(C, C_star) * binary_threshold(R, R_star)
    else:
        raise ValueError(f"Unknown threshold mode: {mode}")


# ---------------------------------------------------------------------------
# Core functions (Layer A)
# ---------------------------------------------------------------------------

def energy_only(E: float) -> float:
    """D_energy = E. No gate, no resilience. Pure energy backbone."""
    return E


def core_multiplicative(E: float, Z: float) -> float:
    """D_core = E × Z."""
    return E * Z


def core_additive(E: float, Z: float, w_E: float = 0.7, w_Z: float = 0.3) -> float:
    """D_core = w_E × E + w_Z × Z."""
    return w_E * E + w_Z * Z


def core_log_multiplicative(E: float, Z: float) -> float:
    """D_core = log(1 + E) × Z. Diminishing returns on dissipation."""
    return math.log1p(E) * Z


CORE_FORMS = {
    "multiplicative": core_multiplicative,
    "additive_weighted": core_additive,
    "multiplicative_log_dissipation": core_log_multiplicative,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    """All tunable parameters for one model run."""
    C_star: float = 0.5
    R_star: float = 0.5
    k_C: float = 10.0
    k_R: float = 10.0
    threshold_mode: Literal["sigmoid", "binary"] = "sigmoid"
    core_form: str = "multiplicative"
    resilience_enabled: bool = False   # h(Z) = 1 when False
    w_E: float = 0.7
    w_Z: float = 0.3


@dataclass
class SystemInputs:
    """Observed/inferred proxy values for one system."""
    label: str
    C: float = 0.0     # closure [0, 1]
    R: float = 0.0     # replication [0, 1]
    E: float = 0.0     # functional dissipation (absolute units)
    Z: float = 1.0     # resilience [0, 1]; default 1 = disabled


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def dominance_score(inputs: SystemInputs, params: ModelParams) -> float:
    """Full model: D = Gate(C, R) × Core(E, Z)."""
    gate = threshold_gate(
        inputs.C, inputs.R,
        params.C_star, params.R_star,
        params.k_C, params.k_R,
        mode=params.threshold_mode,
    )
    Z = inputs.Z if params.resilience_enabled else 1.0
    core_fn = CORE_FORMS.get(params.core_form)
    if core_fn is None:
        raise ValueError(f"Unknown core form: {params.core_form}")
    if params.core_form == "additive_weighted":
        core = core_fn(inputs.E, Z, params.w_E, params.w_Z)
    else:
        core = core_fn(inputs.E, Z)
    return gate * core


def energy_score(inputs: SystemInputs) -> float:
    """Layer A only: D_energy = E. No gate."""
    return inputs.E


def gate_value(inputs: SystemInputs, params: ModelParams) -> float:
    """Return just the gate multiplier for diagnostic purposes."""
    return threshold_gate(
        inputs.C, inputs.R,
        params.C_star, params.R_star,
        params.k_C, params.k_R,
        mode=params.threshold_mode,
    )


# ---------------------------------------------------------------------------
# Temporal modes (A10)
# ---------------------------------------------------------------------------

def rolling_average_score(
    trajectory: list[SystemInputs],
    params: ModelParams,
    window: int | None = None,
) -> float:
    if not trajectory:
        return 0.0
    subset = trajectory[-window:] if window else trajectory
    return sum(dominance_score(s, params) for s in subset) / len(subset)


def trajectory_integrated_score(
    trajectory: list[SystemInputs],
    params: ModelParams,
) -> float:
    return sum(dominance_score(s, params) for s in trajectory)


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """Output of a single scenario comparison."""
    label: str
    human_score: float
    ai_score: float
    human_energy: float
    ai_energy: float
    human_gate: float
    ai_gate: float
    ratio: float
    crossover: bool


def run_scenario(
    human: SystemInputs,
    ai: SystemInputs,
    params: ModelParams,
    label: str = "unnamed",
) -> ScenarioResult:
    """Run one comparison: both energy-only and gated."""
    h = dominance_score(human, params)
    a = dominance_score(ai, params)
    ratio = a / h if h > 0 else float("inf")
    return ScenarioResult(
        label=label,
        human_score=h,
        ai_score=a,
        human_energy=energy_score(human),
        ai_energy=energy_score(ai),
        human_gate=gate_value(human, params),
        ai_gate=gate_value(ai, params),
        ratio=ratio,
        crossover=a > h,
    )


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_base_case(path: str | Path) -> dict:
    """Load base_case.yaml. Requires PyYAML."""
    if not HAS_YAML:
        raise ImportError("PyYAML required: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scenario_grid(path: str | Path) -> dict:
    """Load scenario_grid.yaml. Requires PyYAML."""
    if not HAS_YAML:
        raise ImportError("PyYAML required: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def params_from_base_case(bc: dict) -> ModelParams:
    """Extract ModelParams from a loaded base_case dict."""
    m = bc["model"]
    return ModelParams(
        C_star=m.get("C_star", 0.5),
        R_star=m.get("R_star", 0.5),
        k_C=m.get("k_C", 10),
        k_R=m.get("k_R", 10),
        threshold_mode=m.get("threshold_type", "sigmoid"),
        core_form=m.get("functional_form", "multiplicative"),
        resilience_enabled=(m.get("resilience_mode", "disabled") == "enabled"),
    )


def build_system_inputs(
    bc: dict,
    system: str,
    energy_level: str = "mid",
    closure_override: float | None = None,
    replication_override: float | None = None,
) -> SystemInputs:
    """Build SystemInputs from base_case.yaml for a given system and energy level.

    For A4, closure_override and replication_override should come from scenario_grid.
    """
    diss = bc["dissipation"][system]
    clos = bc["closure"][system]
    repl = bc["replication"][system]

    # Dissipation
    if isinstance(diss["value"], dict):
        E = float(diss["value"][energy_level])
    else:
        E = float(diss["value"])

    # Closure
    if closure_override is not None:
        C = closure_override
    elif isinstance(clos.get("value"), dict):
        C = float(clos["value"].get(energy_level, clos["value"].get("mid", 0)))
    elif clos.get("value") is not None:
        C = float(clos["value"])
    else:
        C = 0.0

    # Replication
    if replication_override is not None:
        R = replication_override
    elif isinstance(repl.get("value"), dict):
        R = float(repl["value"].get(energy_level, repl["value"].get("mid", 0)))
    elif repl.get("value") is not None:
        R = float(repl["value"])
    else:
        R = 0.0

    return SystemInputs(label=system, C=C, R=R, E=E, Z=1.0)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    params = ModelParams(C_star=0.5, R_star=0.5, k_C=10, k_R=10)

    # H3 mid estimates
    h3 = SystemInputs(label="H3", C=0.90, R=0.70, E=7.6, Z=1.0)

    # A4 under each scenario (mid energy = 0.06 TW)
    scenarios = {
        "strict_current":  (0.10, 0.08),
        "semi_autonomous":  (0.20, 0.20),
        "propagating":      (0.40, 0.35),
        "technogenesis":    (0.65, 0.55),
        "full_autonomy":    (0.90, 0.80),
    }

    print("=" * 72)
    print(f"{'Scenario':<20} {'H3_D':>8} {'A4_D':>10} {'A4_gate':>8} {'Ratio':>10} {'Cross':>6}")
    print("-" * 72)
    for name, (c, r) in scenarios.items():
        a4 = SystemInputs(label="A4", C=c, R=r, E=0.06, Z=1.0)
        res = run_scenario(h3, a4, params, label=name)
        print(f"{name:<20} {res.human_score:>8.4f} {res.ai_score:>10.6f} "
              f"{res.ai_gate:>8.4f} {res.ratio:>10.6f} {str(res.crossover):>6}")

    print(f"\n{'Energy-only comparison (no gate):'}")
    print(f"  H3 = {energy_score(h3):.2f} TW")
    print(f"  A4 = {energy_score(SystemInputs('A4', E=0.06)):.2f} TW")
    print(f"  Ratio = {0.06/7.6:.4f}")
