"""Core dominance-score model.

D = sigmoid(C; k, x*) * sigmoid(R; k, x*) * g(E) * h(Z)

This file deliberately stays compact — it implements the bare equations
from the v2 manuscript so other modules (priors, resilience, dynamics,
nested-fitness) can compose on top of it.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray | float, k: float = 10.0, x_star: float = 0.5) -> np.ndarray | float:
    """Sigmoid gate. Maps [0, 1] -> (0, 1) with midpoint x_star and steepness k."""
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(x) - x_star)))


def g_dissipation(E: np.ndarray | float, form: str = "linear") -> np.ndarray | float:
    """Dissipation scaling function. Three forms tested in v2.

    - "linear": g(E) = E
    - "log":    g(E) = log10(1 + E)
    - "sqrt":   g(E) = sqrt(E)
    """
    E = np.asarray(E)
    if form == "linear":
        return E
    if form == "log":
        return np.log10(1.0 + E)
    if form == "sqrt":
        return np.sqrt(E)
    raise ValueError(f"Unknown dissipation form: {form!r}")


def dominance(
    C: np.ndarray | float,
    R: np.ndarray | float,
    E: np.ndarray | float,
    Z: np.ndarray | float = 1.0,
    k: float = 10.0,
    x_star: float = 0.5,
    g_form: str = "linear",
) -> np.ndarray | float:
    """Composite dominance score.

    All inputs can be scalars or numpy arrays of identical shape (Monte
    Carlo samples). Z defaults to 1.0 to reproduce the v2 deterministic
    behaviour when called without an explicit resilience term.
    """
    gate_C = sigmoid(C, k=k, x_star=x_star)
    gate_R = sigmoid(R, k=k, x_star=x_star)
    return gate_C * gate_R * g_dissipation(E, form=g_form) * np.asarray(Z)


def dominance_ratio(
    C_A: np.ndarray | float, R_A: np.ndarray | float, E_A: np.ndarray | float, Z_A: np.ndarray | float,
    C_B: np.ndarray | float, R_B: np.ndarray | float, E_B: np.ndarray | float, Z_B: np.ndarray | float,
    k: float = 10.0, x_star: float = 0.5, g_form: str = "linear",
) -> np.ndarray | float:
    """Compute D(A) / D(B). Convenience for cross-system comparison."""
    D_A = dominance(C_A, R_A, E_A, Z_A, k=k, x_star=x_star, g_form=g_form)
    D_B = dominance(C_B, R_B, E_B, Z_B, k=k, x_star=x_star, g_form=g_form)
    return D_A / D_B
