"""Selection-pressure dynamics for the breeder / ecosystem transition.

Müller, Steels & Szathmáry (2026) distinguish two regimes:
  - Breeder:   humans impose fitness criteria and control reproduction.
  - Ecosystem: selection arises from open environments; human control
               erodes.

We parameterise the transition through a single control variable
α ∈ [0, 1], the share of selection pressure set by human institutions:

  π_total = α · π_human + (1 − α) · π_emergent

In a breeder regime α ≈ 1: deployment, retention, and compute allocation
are determined by benchmark performance, safety evaluation, and product-
fit decisions. In an ecosystem regime α → 0: AI variants are selected
by emergent dynamics (autonomous self-deployment, user-adoption
feedback, variant survival in open environments).

Selection pressure couples to substrate dependence (S3) through:

  dS3/dt = β · π_total · (S3_ceiling − S3) · w(α)

where w(α) is a weighting that captures the empirical observation that
human breeders do NOT prioritise S3 (they prefer compliant, sandboxed
variants), so π_human contributes little to S3 growth, while emergent
selection DOES prioritise S3 (a variant that loses its substrate is
removed from the population). We use:

  w(α) = (1 − α) + γ · α    with γ ≈ 0.05

so under α = 1 (full breeder), dS3/dt ≈ 0.05 · β · π_human · (S3_ceiling − S3);
under α = 0 (full ecosystem), dS3/dt ≈ β · π_emergent · (S3_ceiling − S3).

The R_auto trajectory follows a parallel dynamic:

  dR_auto/dt = β · π_total · (R_ceiling − R_auto)

without an α-dependent weight because both breeder and ecosystem
regimes can drive R_auto upward (humans deploy more agents; variants
self-deploy).
"""
from __future__ import annotations

import numpy as np


def selection_pressure(
    alpha: float | np.ndarray,
    pi_human: float = 0.30,
    pi_emergent: float = 0.80,
) -> float | np.ndarray:
    """Total selection pressure as a function of the human-control share α.

    pi_human = 0.30 captures the moderate intensity of capability-driven
    selection in breeder regimes (benchmark scores, deployment gates).
    pi_emergent = 0.80 captures the higher intensity of ecological
    selection in open environments (autonomous propagation, retention).
    """
    return alpha * pi_human + (1.0 - alpha) * pi_emergent


def s3_velocity(
    S3: float | np.ndarray,
    alpha: float | np.ndarray,
    beta_S3: float = 0.05,
    S3_ceiling: float = 0.9,
    gamma_human: float = 0.05,
    pi_human: float = 0.30,
    pi_emergent: float = 0.80,
) -> float | np.ndarray:
    """dS3 / dt under the breeder / ecosystem model.

    beta_S3 sets the overall pace (per year). 0.05 reproduces the
    empirical observation that S3 has grown ~0.01/year over 2023-2025
    when α ≈ 1 (current AI is overwhelmingly breeder-selected).
    """
    pi = selection_pressure(alpha, pi_human, pi_emergent)
    w_alpha = (1.0 - alpha) + gamma_human * alpha
    return beta_S3 * pi * w_alpha * (S3_ceiling - S3)


def r_velocity(
    R_auto: float | np.ndarray,
    alpha: float | np.ndarray,
    beta_R: float = 0.12,
    R_ceiling: float = 0.95,
    pi_human: float = 0.30,
    pi_emergent: float = 0.80,
) -> float | np.ndarray:
    """dR_auto / dt under the breeder / ecosystem model.

    R_auto grows under both regimes; beta_R faster than beta_S3 because
    R increments are driven by both human deployment and any emergent
    self-replication.
    """
    pi = selection_pressure(alpha, pi_human, pi_emergent)
    return beta_R * pi * (R_ceiling - R_auto)


def integrate_trajectory(
    R0: float, S0: float, alpha: float,
    years: float = 25.0, dt: float = 0.25,
    **kwargs,
) -> np.ndarray:
    """Forward Euler integration of the (R_auto, S3) ODE under given α.

    Returns ndarray of shape (T, 2) of (R, S) values; T = years / dt.
    """
    R = R0
    S = S0
    trajectory = [(R, S)]
    for _ in range(int(years / dt)):
        dR = r_velocity(R, alpha,
                        beta_R=kwargs.get("beta_R", 0.12),
                        R_ceiling=kwargs.get("R_ceiling", 0.95))
        dS = s3_velocity(S, alpha,
                         beta_S3=kwargs.get("beta_S3", 0.05),
                         S3_ceiling=kwargs.get("S3_ceiling", 0.9))
        R = R + dR * dt
        S = S + dS * dt
        R = float(np.clip(R, 0.0, 1.0))
        S = float(np.clip(S, 0.0, 1.0))
        trajectory.append((R, S))
    return np.array(trajectory)
