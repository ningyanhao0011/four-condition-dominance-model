# four-condition-model

Code, data, and tests for the paper
**"Organisational Closure Conditions for an AI Evolutionary Transition: A Diagnostic Framework for Anticipatory Energy Governance"** (submitted to *Futures*).

The model evaluates whether AI-linked infrastructure satisfies the
organisational preconditions for an evolutionary transition relative to
human civilisation. The framework is implemented in nine Python modules
under `src/`, each focused on one constituent of the diagnostic.

## Modules

| Module | File | Role |
|---|---|---|
| Core | `src/model.py` | Sigmoid gate, dominance, dominance-ratio |
| Aggregation | `src/aggregation.py` | min / geomean / weighted-average / max rules |
| Priors | `src/priors.py` | Twelve prior distributions over uncertain inputs |
| Nested fitness | `src/nested_fitness.py` | phi(A4 \| H3) = sigma(S3; k, x*) identity and quadrant classifier |
| Selection pressure | `src/selection_pressure.py` | Breeder/ecosystem dynamics and (R_auto, S3) trajectory integration |
| Resilience (snapshot) | `src/resilience.py` | Beta posteriors fitted to civilisational and AI-infrastructure disturbance data |
| Resilience (time-varying) | `src/resilience_dynamics.py` | Logistic / linear trajectories of Z over 1990–2035 |
| Dynamics | `src/dynamics.py` | Longitudinal fits of S1 / S2 / S3 over 2019–2025 with projection to 2030 / 2035 |
| Sensitivity | `src/sensitivity.py` | Saltelli (2010) Pick-Freeze Sobol decomposition |
| Monte Carlo | `src/mc_sampler.py` | Prior-predictive sampler orchestrating priors + Z + model |

## Layout

```
four-condition-model/
├── README.md                    (this file)
├── LICENSE                      (MIT)
├── requirements.txt             (numpy, scipy, pandas)
├── .gitignore
├── src/                         (10 Python modules — the model itself)
├── data/                        (4 CSV files — disturbance and longitudinal data)
│   ├── recovery_data.csv
│   ├── longitudinal.csv
│   ├── resilience_longitudinal.csv
│   └── historical_systems.csv
├── examples/
│   └── run_analysis.py          (reproduce all headline numbers in one script)
└── tests/
    └── test_model.py            (34 unit tests; all passing)
```

## Installation

```bash
git clone https://github.com/ningyanhao0011/four-condition-model.git
cd four-condition-model
pip install -r requirements.txt
```

Tested with Python 3.11+. Minimum versions: numpy>=1.26, scipy>=1.11, pandas>=2.0.

## Reproducing the headline numbers

```bash
python examples/run_analysis.py
```

This single script reproduces the v2 deterministic baseline, the
Module B resilience posteriors, the 10,000-draw Monte Carlo with
pre-registered falsifier, the Sobol decomposition (k = S1 = 0.71 under
min-rule), the dynamic projection of S1/S2/S3 to 2035, the time-varying
Z trajectories, the phi(A4 | H3) identity at five scenarios, and the
selection-pressure 25-year trajectories under three alpha regimes.

## Tests

```bash
python tests/test_model.py
```

34 unit tests cover the sigmoid behaviour, dissipation functional forms,
dominance under gated and open conditions, all four aggregation rules,
the phi identity, the quadrant classifier, snapshot and time-varying Z
posteriors, the longitudinal fits, the selection-pressure dynamics, and
the Sobol first-order/total-order bounds. All currently pass.

## Headline numerical results

| Metric | Value | Source module |
|---|---|---|
| Monte Carlo draws | 10,000 | mc_sampler |
| Pre-registered strict falsifier threshold | 0.01 | sensitivity |
| P95 of D(A4)/D(H3), primary specification | 7.7 x 10^-4 | mc_sampler + sensitivity |
| Falsifier exceedance | 0.000 % | sensitivity |
| Sobol S1(k) under min-rule | 0.71 | sensitivity |
| Sobol S1(x*) under min-rule | 0.24 | sensitivity |
| Median Z_H3 (civilisation) | 0.95 | resilience |
| Median Z_A4_long (supply chain) | 0.36 | resilience |
| S3 trajectory slope 2023-2025 | ~ 0.01 / year | dynamics |
| Dominance ratio P95 in 2035 | 1.7 x 10^-3 | mc_sampler + dynamics |
| phi(A4 \| H3) median, current AI | 0.011 | nested_fitness |
| P(current AI in dependent-tool quadrant) | 97.7 % | selection_pressure |
| Critical alpha for A5 crossover within 50 yr | alpha < 0.5 | selection_pressure |
| Historical back-test agreement (10 systems) | 90 % | (see data/historical_systems.csv) |

## Citation

```bibtex
@article{ning2026,
  author  = {Ning, Yanhao and Loh, Tzu Yang},
  title   = {Organisational Closure Conditions for an {AI} Evolutionary
             Transition: A Diagnostic Framework for Anticipatory Energy
             Governance},
  journal = {Futures},
  year    = {2026},
  note    = {under review},
}
```

## License

MIT.

## Notes

This repository contains the model code only. Figure-generation,
manuscript-building, and report-generation scripts are not included
here. They were used during the paper's preparation and are available
from the authors on request, but they are not necessary for verifying or
extending the model itself.
