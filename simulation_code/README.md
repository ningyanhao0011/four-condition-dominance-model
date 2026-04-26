# Simulation Code — Dominance Model for AI Safety Analysis

Replication code for the four-condition dominance framework comparing organizational trajectories of human civilization (H3) and AI-linked technospheric subsystems (A4).

## Repository Structure

```
simulation_code/
├── models/
│   ├── dominance_model.py        # Core model: sigmoid gate, scoring, YAML loader
│   └── scenario_runner.py        # CLI runner: loads YAML, runs all scenarios
├── scripts/
│   └── collect_results.py        # Print all numerical results (tables)
└── data/
    └── assumptions/
        ├── base_case.yaml        # Empirical proxy values and model parameters
        └── scenario_grid.yaml    # A4 scenario definitions (closure × replication)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all scenario comparisons (prints table)
python models/scenario_runner.py

# Run with specific energy level
python models/scenario_runner.py --energy mid
python models/scenario_runner.py --energy all --pair robustness

# Print full numerical results for all tables
python scripts/collect_results.py
```

All commands should be run from the `simulation_code/` directory.

## Model Overview

The dominance score D is computed as:

```
D = Gate(C, R) × g(E) × h(Z)
```

Where:
- **C** — Organizational closure [0, 1]
- **R** — Replication [0, 1]
- **E** — Functional dissipation (TW)
- **Z** — Resilience (h(Z) = 1 in base case; deferred)
- **Gate(C, R)** = σ(C) × σ(R), sigmoid thresholds at C* = R* = 0.5

### Two-layer architecture
- **Layer A** (energy backbone): D_energy = E
- **Layer B** (threshold gate): D_gated = Gate(C, R) × E

### System pairs
| Pair | Human system | AI system | Purpose |
|------|-------------|-----------|---------|
| Primary | H3 — Human Civilization | A4 — AI-linked Technospheric Subsystem | Main analysis |
| Robustness | H1 — Individual Organism | A3 — Data Center + Stack | Physical-scale check |

### Scenarios (A4)
| Scenario | C | R | Plausibility |
|----------|---|---|-------------|
| strict_current_ai | 0.10 | 0.08 | High |
| semi_autonomous_infra | 0.20 | 0.20 | Medium-high |
| propagating_ai | 0.40 | 0.35 | Medium |
| technogenesis | 0.65 | 0.55 | Speculative |
| full_autonomy | 0.90 | 0.80 | Highly speculative |

## Key caveat

A4 is currently nested within H3. The primary comparison is an **intra-system trajectory analysis**, not inter-system competition. Results should not be interpreted as predictions of AI dominance over human civilization.

## Requirements

See `requirements.txt`. Python ≥ 3.10 required (uses `list[str] | None` type hints).
