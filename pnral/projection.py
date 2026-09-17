"""Euclidean projection onto ``K = {w : g(w) <= Lambda_constraint}``.

The projection solves

.. math::

    \\Pi_K(y) = \\arg\\min_z \\tfrac12 \\|z - y\\|^2
    \\quad\\text{subject to}\\quad g(z) \\le \\Lambda_{\\mathrm{constraint}} .

If ``g(y) <= Lambda_constraint`` the point is returned unchanged.  Otherwise
the constraint is active and the KKT conditions give, for some ``eta > 0``,

.. math::

    z_i + \\eta\\, p\\, z_i\\,(z_i^2 + \\epsilon^2)^{p/2-1} = y_i
    \\quad (i = 0,\\dots,d-1),
    \\qquad g(z) = \\Lambda_{\\mathrm{constraint}} .

Structure exploited by the solver
---------------------------------
* the multiplier is a *scalar*, so the ``d`` equations decouple given ``eta``;
* the factor multiplying ``z_i`` is ``> 1``, hence ``sign(z_i) = sign(y_i)``
  and ``|z_i| <= |y_i|``: the inner root lives in ``[0, |y_i|]``;
* ``d/dt [ t (1 + eta p (t^2+eps^2)^{p/2-1}) ]
        = 1 + eta p (t^2+eps^2)^{p/2-2} [ (p-1) t^2 + eps^2 ] > 0`` for
  ``p >= 1``, so the inner root is unique;
* ``eta -> |z_i(eta)|`` is decreasing, hence ``eta -> g(z(eta))`` is
  decreasing and the outer scalar equation
  ``g(z(eta)) - Lambda_constraint = 0`` has a unique root, bracketed by
  geometrically inflating ``eta_high``.

Two inner solvers are provided and agree to ~1e-12 (checked in the
test-suite): the literal ``scipy.optimize.brentq`` implementation and a
vectorised safeguarded Newton/bisection solver used by default inside the
sampler for speed.

For ``p_constraint == 2`` the constraint reduces to a Euclidean ball,
``g(w) = ||w||_2^2 + d eps^2`` and ``K = {||w||_2 <= sqrt(Lambda - d eps^2)}``,
and an exact radial projection is used.  Radial projection is *never* used
for ``p != 2``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import brentq

from .constraint import g_constraint, g_minimum

__all__ = ["ProjectionResult", "project_K", "kkt_residual", "radial_projection"]


@dataclass
class ProjectionResult:
    """Outcome of one call to :func:`project_K`."""

    z: np.ndarray
    projected: bool
    distance: float
    eta: float
    g_before: float
    g_after: float
    n_outer_iterations: int = 0

    def __iter__(self):
        """Allow ``z, info = project_K(...)``-style unpacking of ``(z, self)``."""
        yield self.z
        yield self


# ----------------------------------------------------------------------
# inner solve:  t (1 + eta p (t^2+eps^2)^{p/2-1}) = |y_i|,  t in [0, |y_i|]
# ----------------------------------------------------------------------
def _inner_residual(t: np.ndarray, y_abs: np.ndarray, eta: float,
                    p: float, eps: float) -> np.ndarray:
    return t * (1.0 + eta * p * (t * t + eps * eps) ** (0.5 * p - 1.0)) - y_abs


def _inner_derivative(t: np.ndarray, eta: float, p: float,
                      eps: float) -> np.ndarray:
    s = t * t + eps * eps
    return 1.0 + eta * p * s ** (0.5 * p - 2.0) * ((p - 1.0) * t * t + eps * eps)


def _inner_solve_vectorized(y_abs: np.ndarray, eta: float, p: float, eps: float,
                            tolerance: float = 1e-13,
                            max_iterations: int = 60) -> np.ndarray:
    """Safeguarded Newton/bisection for every coordinate simultaneously."""
    lo = np.zeros_like(y_abs)
    hi = y_abs.copy()
    t = y_abs.copy()                      # phi(|y|) >= 0 -> valid upper bound
    for _ in range(max_iterations):
        phi = _inner_residual(t, y_abs, eta, p, eps)
        hi = np.where(phi >= 0.0, np.minimum(hi, t), hi)
        lo = np.where(phi < 0.0, np.maximum(lo, t), lo)
        if np.all(np.abs(phi) <= tolerance * (1.0 + y_abs)) or np.all(hi - lo <= tolerance):
            break
        derivative = _inner_derivative(t, eta, p, eps)
        newton = t - phi / derivative
        inside = (newton > lo) & (newton < hi) & np.isfinite(newton)
        t = np.where(inside, newton, 0.5 * (lo + hi))
    return np.clip(t, 0.0, y_abs)


def _inner_solve_brentq(y_abs: np.ndarray, eta: float, p: float, eps: float,
                        xtol: float = 1e-14) -> np.ndarray:
    """Literal per-coordinate ``scipy.optimize.brentq`` implementation."""
    out = np.zeros_like(y_abs)
    for i, magnitude in enumerate(y_abs):
        if magnitude <= 0.0:
            out[i] = 0.0
            continue
        scalar = np.asarray([magnitude])

        def objective(t: float, magnitude=magnitude) -> float:
            return float(_inner_residual(np.asarray([t]), scalar, eta, p, eps)[0])

        out[i] = brentq(objective, 0.0, magnitude, xtol=xtol, rtol=8.9e-16,
                        maxiter=200)
    return out


def _solve_z_given_eta(y: np.ndarray, eta: float, p: float, eps: float,
                       use_brentq: bool, tolerance: float) -> np.ndarray:
    """Solve the ``d`` decoupled KKT equations for a fixed multiplier ``eta``."""
    y_abs = np.abs(y)
    if use_brentq:
        magnitudes = _inner_solve_brentq(y_abs, eta, p, eps, xtol=max(tolerance, 1e-15))
    else:
        magnitudes = _inner_solve_vectorized(y_abs, eta, p, eps, tolerance=tolerance)
    return np.sign(y) * magnitudes          # z_i inherits the sign of y_i


# ----------------------------------------------------------------------
def radial_projection(y: np.ndarray, epsilon_constraint: float,
                      Lambda_constraint: float) -> np.ndarray:
    """Exact projection for ``p_constraint == 2`` (Euclidean ball)."""
    d = y.shape[0]
    radius_squared = Lambda_constraint - d * epsilon_constraint ** 2
    if radius_squared <= 0.0:
        raise ValueError("empty constraint set for p_constraint = 2")
    radius = float(np.sqrt(radius_squared))
    norm = float(np.linalg.norm(y))
    if norm <= radius:
        return y.copy()
    return y * (radius / norm)


def project_K(
    y: np.ndarray,
    p_constraint: float,
    epsilon_constraint: float,
    Lambda_constraint: float,
    use_brentq: bool = False,
    tolerance: float = 1e-13,
    max_bracket_expansions: int = 200,
) -> ProjectionResult:
    """Euclidean projection of ``y`` onto ``K``.

    Parameters
    ----------
    use_brentq:
        ``True`` uses the per-coordinate ``brentq`` inner solve of the
        specification; ``False`` (default) the equivalent vectorised solver.
    tolerance:
        Root-finder tolerance for both the inner and the outer equation.

    Returns
    -------
    ProjectionResult
        Carrying the projected point, whether a projection actually took
        place, the projection distance ``||z - y||`` and the multiplier
        ``eta`` (``0`` when the point was already feasible).
    """
    y = np.asarray(y, dtype=np.float64)
    d = y.shape[0]
    if p_constraint < 1.0:
        raise ValueError(
            "the projection solver assumes p_constraint >= 1 (uniqueness of "
            "the inner root); p < 1 gives a non-convex constraint set")
    minimum = g_minimum(d, p_constraint, epsilon_constraint)
    if Lambda_constraint <= minimum:
        raise ValueError("Lambda_constraint must exceed g(0)")

    g_before = g_constraint(y, p_constraint, epsilon_constraint)
    if g_before <= Lambda_constraint:
        return ProjectionResult(z=y.copy(), projected=False, distance=0.0,
                                eta=0.0, g_before=g_before, g_after=g_before)

    # --- p = 2: exact radial projection (never used for p != 2) ---------
    if p_constraint == 2.0:
        z = radial_projection(y, epsilon_constraint, Lambda_constraint)
        g_after = g_constraint(z, p_constraint, epsilon_constraint)
        return ProjectionResult(
            z=z, projected=True, distance=float(np.linalg.norm(z - y)),
            # eta from z_i (1 + 2 eta) = y_i  =>  eta = (||y||/||z|| - 1)/2
            eta=float((np.linalg.norm(y) / np.linalg.norm(z) - 1.0) / 2.0),
            g_before=g_before, g_after=g_after)

    # --- general p: outer scalar equation g(z(eta)) = Lambda -------------
    def outer(eta: float) -> float:
        z = _solve_z_given_eta(y, eta, p_constraint, epsilon_constraint,
                               use_brentq, tolerance)
        return g_constraint(z, p_constraint, epsilon_constraint) - Lambda_constraint

    # First-order guess: g(y - eta grad_g(y)) ~ g(y) - eta ||grad_g(y)||^2.
    gradient = p_constraint * y * (y * y + epsilon_constraint ** 2) ** (0.5 * p_constraint - 1.0)
    gradient_norm_squared = float(gradient @ gradient)
    eta_high = max((g_before - Lambda_constraint) / max(gradient_norm_squared, 1e-300),
                   1e-8)

    expansions = 0
    while outer(eta_high) > 0.0:            # inflate geometrically until feasible
        eta_high *= 4.0
        expansions += 1
        if expansions > max_bracket_expansions:
            raise RuntimeError("failed to bracket the projection multiplier eta")
    eta_low = 0.0

    # xtol is scaled by the bracket: eta is an O(1e-3 .. 1e0) multiplier and
    # a relative accuracy of 1e-12 is far below the constraint tolerance.
    eta = brentq(outer, eta_low, eta_high, xtol=1e-14 * max(1.0, eta_high),
                 rtol=1e-12, maxiter=200)
    z = _solve_z_given_eta(y, eta, p_constraint, epsilon_constraint,
                           use_brentq, tolerance)
    g_after = g_constraint(z, p_constraint, epsilon_constraint)

    # Guard against a root returned marginally on the infeasible side.
    if g_after > Lambda_constraint:
        eta_safe = eta
        for _ in range(60):
            eta_safe *= 1.0 + 1e-9
            z = _solve_z_given_eta(y, eta_safe, p_constraint, epsilon_constraint,
                                   use_brentq, tolerance)
            g_after = g_constraint(z, p_constraint, epsilon_constraint)
            if g_after <= Lambda_constraint:
                break
        eta = eta_safe

    return ProjectionResult(z=z, projected=True,
                            distance=float(np.linalg.norm(z - y)), eta=float(eta),
                            g_before=g_before, g_after=g_after,
                            n_outer_iterations=expansions)


def kkt_residual(y: np.ndarray, result: ProjectionResult, p_constraint: float,
                 epsilon_constraint: float,
                 Lambda_constraint: float) -> dict:
    """Numerical KKT check for a projection.

    Returns the stationarity residual
    ``max_i |z_i - y_i + eta p z_i (z_i^2+eps^2)^{p/2-1}|``, the primal
    feasibility violation ``max(0, g(z) - Lambda)``, the dual feasibility
    ``min(0, eta)`` and the complementary-slackness product
    ``eta * (Lambda - g(z))``.
    """
    z = result.z
    stationarity = z - y + result.eta * p_constraint * z * (
        z * z + epsilon_constraint ** 2) ** (0.5 * p_constraint - 1.0)
    g_z = g_constraint(z, p_constraint, epsilon_constraint)
    return {
        "stationarity": float(np.max(np.abs(stationarity))),
        "primal_violation": float(max(0.0, g_z - Lambda_constraint)),
        "dual_violation": float(min(0.0, result.eta)),
        "complementary_slackness": float(result.eta * (Lambda_constraint - g_z)),
        "g_after": float(g_z),
    }
