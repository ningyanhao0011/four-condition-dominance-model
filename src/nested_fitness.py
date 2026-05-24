"""Module D - nested fitness metric phi(A4 | H3) and the breeder/ecosystem
phase diagram.

Derivation of phi:

  Decompose A4's total dissipation D(A4) into two parts:
    D_autonomous(A4)        = energy use that A4 sustains by itself
    D_constructed(A4 | H3)  = energy use that H3 sustains for A4

  Total:  D(A4) = D_autonomous(A4) + D_constructed(A4 | H3)

  Define the nested fitness share:
    phi(A4 | H3) := D_autonomous(A4) / D(A4)

  The autonomous fraction is exactly the dependency-independence value
  S3, passed through the closure gate sigma(.) that already governs
  evolutionary standing in the v2 model:

    D_autonomous(A4) = D(A4) * sigma(S3; k, x*)

  Therefore:

    phi(A4 | H3) = sigma(S3; k, x*)

This identity has three important consequences:

1. phi is a one-dimensional function of a single empirical quantity (S3),
   not a multiplicative product of multiple sampled inputs. The
   'tautology' critique of D(A4)/D(H3) does not apply to phi because phi
   is not a ratio between a system and its container; it is a property
   of A4's own internal structure.

2. phi can be measured against existing capability evaluations (ARA
   suite, Kinniment 2024, AISI 2025) directly, and updated annually.

3. phi maps onto Mueller, Steels & Szathmary (2026)'s breeder vs
   ecosystem dichotomy via the joint position in (R_autonomy, S3)
   space. The phase diagram has four quadrants:

      |  Low R_auto, Low S3   | Low R_auto, High S3
      |  (dependent tool)     | (impossible: requires R)
   ---+----------------------+---------------------
      |  High R_auto, Low S3 | High R_auto, High S3
      |  (Mueller ecosystem) | (A4 -> A5 transition)
"""
from __future__ import annotations

import numpy as np

from src.model import sigmoid


def phi(S3: np.ndarray | float, k: float = 10.0, x_star: float = 0.5) -> np.ndarray | float:
    """Nested fitness share phi(A4 | H3) = sigma(S3; k, x*)."""
    return sigmoid(S3, k=k, x_star=x_star)


def joint_gate(R_auto: np.ndarray | float, S3: np.ndarray | float,
               k: float = 10.0, x_star: float = 0.5) -> np.ndarray | float:
    """Joint gate sigma(R_auto) * sigma(S3). When both > 0.5, both gates
    are mostly open (the 'evolutionary agent' quadrant).
    """
    return sigmoid(R_auto, k=k, x_star=x_star) * sigmoid(S3, k=k, x_star=x_star)


def quadrant(R_auto: float, S3: float, gate_threshold: float = 0.5) -> str:
    """Classify a system into one of the four quadrants of the
    breeder/ecosystem/A5 phase diagram.
    """
    r_open = R_auto >= gate_threshold
    s_open = S3 >= gate_threshold
    if not r_open and not s_open:
        return "dependent tool"
    if not r_open and s_open:
        return "impossible (R-gated)"
    if r_open and not s_open:
        return "Mueller ecosystem (variant selection on human substrate)"
    return "A5 transition (independent evolutionary agent)"
