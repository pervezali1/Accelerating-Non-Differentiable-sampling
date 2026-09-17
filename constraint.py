"""Smoothed l^p constraint set and its geometry.

Mathematics
-----------
For ``w in R^d`` define

    g(w) = sum_{i=1}^{d} (w_i^2 + epsilon_constraint^2)^(p_constraint/2),

    K    = {w : g(w) <= Lambda_constraint},

    Lambda_constraint = d * epsilon_constraint**p_constraint
                        + radius_budget**p_constraint.

``g`` is a smooth (C^infinity for epsilon > 0) surrogate for the l^p ball
``||w||_p^p <= radius_budget^p``: the offset ``epsilon_constraint^2`` removes
the kink of ``|w_i|^p`` at the origin, and the ``d * epsilon^p`` term in the
threshold exactly compensates the resulting floor ``g(0) = d * epsilon^p``.

The gradient is separable,

    grad_g[i] = p_constraint * w_i * (w_i^2 + epsilon_constraint^2)^(p/2 - 1),

and the "slack" function and its gradient are

    psi(w)      = Lambda_constraint - g(w),
    grad_psi(w) = -grad_g(w).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import brentq

from config import ConstraintConfig


def g_value(
    w: np.ndarray,
    p_constraint: float,
    epsilon_constraint: float,
) -> float | np.ndarray:
    """Evaluate ``g(w) = sum_i (w_i^2 + eps^2)^(p/2)``.

    Accepts a single point of shape ``(d,)`` or a batch of shape ``(..., d)``;
    the sum is taken over the last axis.
    """
    w = np.asarray(w, dtype=float)
    terms = (w * w + epsilon_constraint ** 2) ** (p_constraint / 2.0)
    return terms.sum(axis=-1)


def grad_g(
    w: np.ndarray,
    p_constraint: float,
    epsilon_constraint: float,
) -> np.ndarray:
    """Gradient ``grad_g[i] = p * w_i * (w_i^2 + eps^2)^(p/2 - 1)``."""
    w = np.asarray(w, dtype=float)
    return p_constraint * w * (w * w + epsilon_constraint ** 2) ** (p_constraint / 2.0 - 1.0)


def psi_value(
    w: np.ndarray,
    p_constraint: float,
    epsilon_constraint: float,
    Lambda_constraint: float,
) -> float | np.ndarray:
    """``psi(w) = Lambda_constraint - g(w)`` (non-negative exactly on ``K``)."""
    return Lambda_constraint - g_value(w, p_constraint, epsilon_constraint)


def grad_psi(
    w: np.ndarray,
    p_constraint: float,
    epsilon_constraint: float,
) -> np.ndarray:
    """``grad_psi(w) = -grad_g(w)``."""
    return -grad_g(w, p_constraint, epsilon_constraint)


def unit_normal(
    w: np.ndarray,
    p_constraint: float,
    epsilon_constraint: float,
) -> np.ndarray:
    """Outward unit normal ``grad_g(w) / ||grad_g(w)||`` of the level set.

    Returns the zero vector at ``w = 0``, where ``grad_g`` vanishes and the
    normal is undefined.
    """
    gradient = grad_g(w, p_constraint, epsilon_constraint)
    norm = float(np.linalg.norm(gradient))
    if norm == 0.0:
        return np.zeros_like(gradient)
    return gradient / norm


def is_feasible(
    w: np.ndarray,
    constraint: ConstraintConfig,
    tol: float = 1e-8,
) -> bool | np.ndarray:
    """Test ``g(w) <= Lambda_constraint + tol``."""
    value = g_value(w, constraint.p_constraint, constraint.epsilon_constraint)
    return value <= constraint.Lambda_constraint + tol


def distance_to_boundary(
    w: np.ndarray,
    constraint: ConstraintConfig,
) -> float | np.ndarray:
    """Level-set slack ``Lambda_constraint - g(w)``.

    This is a *constraint-value* distance, not a Euclidean distance to the
    surface; it is the quantity the specification asks to be recorded.
    """
    return psi_value(
        w,
        constraint.p_constraint,
        constraint.epsilon_constraint,
        constraint.Lambda_constraint,
    )


def rescale_into_constraint(
    beta_raw: np.ndarray,
    constraint: ConstraintConfig,
    xtol: float = 1e-14,
) -> tuple[np.ndarray, float]:
    """Shrink ``beta_raw`` so that it sits safely inside ``K``.

    If ``g(beta_raw)`` already lies at or below the mid-point level

        target = d * eps^p + 0.5 * (Lambda_constraint - d * eps^p),

    the vector is returned unchanged with ``t = 1``.  Otherwise a bisection
    (``scipy.optimize.brentq``) finds ``t in (0, 1)`` with

        g(t * beta_raw) = target,

    and ``beta_true = t * beta_raw`` is returned.  The map ``t -> g(t*beta)``
    is strictly increasing on ``[0, 1]`` (each summand is), so the root is
    unique and the bracketing ``g(0) = d*eps^p < target < g(beta_raw)`` is
    valid.

    Returns
    -------
    (beta_true, t)
    """
    beta_raw = np.asarray(beta_raw, dtype=float)
    p, eps = constraint.p_constraint, constraint.epsilon_constraint
    target = constraint.interior_level

    if float(g_value(beta_raw, p, eps)) <= target:
        return beta_raw.copy(), 1.0

    def objective(t: float) -> float:
        return float(g_value(t * beta_raw, p, eps)) - target

    # objective(0) = d*eps^p - target < 0 and objective(1) > 0 by construction.
    t_star = brentq(objective, 0.0, 1.0, xtol=xtol, rtol=8.9e-16, maxiter=200)
    return t_star * beta_raw, float(t_star)


def numerical_grad_g(
    w: np.ndarray,
    p_constraint: float,
    epsilon_constraint: float,
    h: float = 1e-6,
) -> np.ndarray:
    """Centred finite-difference gradient of ``g``, for verification only."""
    w = np.asarray(w, dtype=float)
    out = np.zeros_like(w)
    for i in range(w.size):
        plus, minus = w.copy(), w.copy()
        plus[i] += h
        minus[i] -= h
        out[i] = (
            g_value(plus, p_constraint, epsilon_constraint)
            - g_value(minus, p_constraint, epsilon_constraint)
        ) / (2.0 * h)
    return out


def boundary_point(
    direction: np.ndarray,
    constraint: ConstraintConfig,
    xtol: float = 1e-14,
) -> np.ndarray:
    """Return the point ``t * direction`` lying exactly on ``{g = Lambda}``.

    Used by the unit tests to generate genuine boundary points on which the
    identity ``J(w) @ normal(w) = 0`` is checked.
    """
    direction = np.asarray(direction, dtype=float)
    p, eps = constraint.p_constraint, constraint.epsilon_constraint
    Lambda = constraint.Lambda_constraint

    def objective(t: float) -> float:
        return float(g_value(t * direction, p, eps)) - Lambda

    upper = 1.0
    for _ in range(200):
        if objective(upper) > 0.0:
            break
        upper *= 2.0
    else:  # pragma: no cover - unreachable for finite Lambda and direction != 0
        raise RuntimeError("could not bracket the boundary along this direction")
    t_star = brentq(objective, 0.0, upper, xtol=xtol, rtol=8.9e-16, maxiter=200)
    return t_star * direction


def make_constraint_functions(
    constraint: ConstraintConfig,
) -> dict[str, Callable[..., np.ndarray | float]]:
    """Bundle ``g``, ``grad_g``, ``psi`` and ``grad_psi`` with fixed parameters."""
    p, eps, Lambda = (
        constraint.p_constraint,
        constraint.epsilon_constraint,
        constraint.Lambda_constraint,
    )
    return {
        "g": lambda w: g_value(w, p, eps),
        "grad_g": lambda w: grad_g(w, p, eps),
        "psi": lambda w: psi_value(w, p, eps, Lambda),
        "grad_psi": lambda w: grad_psi(w, p, eps),
        "normal": lambda w: unit_normal(w, p, eps),
    }
