"""Exact Euclidean projection onto the smoothed l^p ball ``K``.

Problem
-------
    projection_K(y) = argmin_z  0.5 * ||z - y||^2   subject to  g(z) <= Lambda,

with ``g(z) = sum_i (z_i^2 + eps^2)^(p/2)``.

If ``g(y) <= Lambda`` the point is already feasible and is returned unchanged.
Otherwise the constraint is active and the KKT conditions give a multiplier
``eta > 0`` such that, coordinatewise,

    z_i + eta * p * z_i * (z_i^2 + eps^2)^(p/2 - 1) = y_i,        (stationarity)
    g(z) = Lambda.                                                (activity)

Solution strategy
-----------------
*Inner solve.*  For fixed ``eta`` the scalar map

    h_eta(z) = z + eta * p * z * (z^2 + eps^2)^(p/2 - 1)

is odd and, for ``p >= 1``, strictly increasing:

    h_eta'(z) = 1 + eta * p * (z^2 + eps^2)^(p/2 - 2) * ((p-1) z^2 + eps^2) > 0.

Hence ``z_i = sign(y_i) * h_eta^{-1}(|y_i|)`` and, because ``h_eta(u) >= u``
for ``u >= 0``, the root ``|z_i|`` lies in ``[0, |y_i|]``.  That bracket is
handed to :func:`scipy.optimize.brentq`.

*Outer solve.*  ``eta -> g(z(eta))`` is non-increasing (every ``|z_i|``
shrinks as ``eta`` grows) with ``g(z(0)) = g(y) > Lambda`` and
``g(z(eta)) -> d * eps^p < Lambda`` as ``eta -> infinity``.  Starting from
``eta_low = 0`` the upper end is grown geometrically until
``g(z(eta_high)) <= Lambda`` and ``brentq`` then solves
``g(z(eta)) - Lambda = 0``.

*``p = 2`` shortcut.*  Here ``g(w) = ||w||^2 + d * eps^2`` and ``K`` is the
Euclidean ball of radius ``sqrt(Lambda - d * eps^2)``, whose projection is a
radial rescaling.  This is the **only** case in which radial scaling is used:
for ``p != 2`` radial scaling is *not* the Euclidean projection, so it is
never applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from config import ConstraintConfig, ProjectionConfig
from constraint import g_value, grad_g


@dataclass
class ProjectionResult:
    """Outcome of a single projection call."""

    z: np.ndarray
    projected: bool          # was the input infeasible?
    eta: float               # KKT multiplier (0 when no projection occurred)
    distance: float          # ||z - y||
    g_value: float           # g(z)
    kkt_residual: float      # max_i |z_i + eta*grad_g(z)_i - y_i|


def _scalar_map(z: float, eta: float, p: float, eps: float) -> float:
    """``h_eta(z) = z + eta * p * z * (z^2 + eps^2)^(p/2 - 1)``."""
    return z + eta * p * z * (z * z + eps ** 2) ** (p / 2.0 - 1.0)


def _solve_coordinates(
    y_abs: np.ndarray,
    eta: float,
    p: float,
    eps: float,
    xtol: float,
) -> np.ndarray:
    """Solve ``h_eta(u) = |y_i|`` for ``u in [0, |y_i|]``, coordinatewise."""
    out = np.zeros_like(y_abs)
    for i, target in enumerate(y_abs):
        if target == 0.0:
            out[i] = 0.0
            continue
        # h_eta(0) = 0 <= target and h_eta(target) >= target: valid bracket.
        upper_value = _scalar_map(target, eta, p, eps)
        if upper_value <= target:
            # Only possible when eta == 0 (then h is the identity).
            out[i] = target
            continue
        out[i] = brentq(
            lambda u: _scalar_map(u, eta, p, eps) - target,
            0.0,
            target,
            xtol=xtol,
            rtol=8.9e-16,
            maxiter=200,
        )
    return out


def _z_of_eta(
    y: np.ndarray,
    eta: float,
    p: float,
    eps: float,
    xtol: float,
) -> np.ndarray:
    """``z(eta)``: the stationarity solution for a given multiplier."""
    signs = np.sign(y)
    magnitudes = _solve_coordinates(np.abs(y), eta, p, eps, xtol)
    return signs * magnitudes


def kkt_residual(
    z: np.ndarray,
    y: np.ndarray,
    eta: float,
    p: float,
    eps: float,
) -> float:
    """Max absolute stationarity residual ``z + eta * grad_g(z) - y``."""
    return float(np.abs(z + eta * grad_g(z, p, eps) - y).max())


def project_onto_K(
    y: np.ndarray,
    constraint: ConstraintConfig,
    projection_cfg: ProjectionConfig | None = None,
) -> ProjectionResult:
    """Euclidean projection of ``y`` onto ``K = {g <= Lambda_constraint}``."""
    if projection_cfg is None:
        projection_cfg = ProjectionConfig()

    y = np.asarray(y, dtype=float)
    p = constraint.p_constraint
    eps = constraint.epsilon_constraint
    Lambda = constraint.Lambda_constraint

    g_y = float(g_value(y, p, eps))
    if g_y <= Lambda:
        return ProjectionResult(
            z=y.copy(),
            projected=False,
            eta=0.0,
            distance=0.0,
            g_value=g_y,
            kkt_residual=0.0,
        )

    # ---- p = 2 shortcut: K is the Euclidean ball, projection is radial. ----
    if p == 2.0:
        radius = np.sqrt(max(Lambda - constraint.d * eps ** 2, 0.0))
        norm_y = float(np.linalg.norm(y))
        z = y * (radius / norm_y)
        # Stationarity z(1 + 2 eta) = y gives eta = (||y||/radius - 1) / 2.
        eta = 0.5 * (norm_y / radius - 1.0)
        return ProjectionResult(
            z=z,
            projected=True,
            eta=eta,
            distance=float(np.linalg.norm(z - y)),
            g_value=float(g_value(z, p, eps)),
            kkt_residual=kkt_residual(z, y, eta, p, eps),
        )

    # ---- general p: outer search on the KKT multiplier eta ----
    def g_gap(eta: float) -> float:
        z_eta = _z_of_eta(y, eta, p, eps, projection_cfg.xtol)
        return float(g_value(z_eta, p, eps)) - Lambda

    eta_low = 0.0                                  # g_gap(0) = g(y) - Lambda > 0
    eta_high = projection_cfg.eta_high_init
    for _ in range(projection_cfg.max_eta_expansions):
        if g_gap(eta_high) <= 0.0:
            break
        eta_high *= projection_cfg.eta_growth
    else:  # pragma: no cover - g(z(eta)) -> d*eps^p < Lambda guarantees exit
        raise RuntimeError(
            "failed to bracket the KKT multiplier eta; check Lambda_constraint"
        )

    eta_star = brentq(
        g_gap,
        eta_low,
        eta_high,
        xtol=projection_cfg.eta_xtol,
        rtol=8.9e-16,
        maxiter=300,
    )
    z = _z_of_eta(y, eta_star, p, eps, projection_cfg.xtol)

    g_z = float(g_value(z, p, eps))
    if g_z > Lambda + projection_cfg.feasibility_tol:  # pragma: no cover
        raise RuntimeError(
            f"projection infeasible: g(z) = {g_z} > Lambda + tol = "
            f"{Lambda + projection_cfg.feasibility_tol}"
        )

    return ProjectionResult(
        z=z,
        projected=True,
        eta=float(eta_star),
        distance=float(np.linalg.norm(z - y)),
        g_value=g_z,
        kkt_residual=kkt_residual(z, y, float(eta_star), p, eps),
    )


def check_kkt(
    result: ProjectionResult,
    y: np.ndarray,
    constraint: ConstraintConfig,
    tol: float = 1e-6,
) -> dict[str, float | bool]:
    """Full KKT audit of a projection: primal, dual and complementary slackness."""
    Lambda = constraint.Lambda_constraint
    primal = Lambda - result.g_value                      # >= 0 required
    dual = result.eta                                     # >= 0 required
    complementary = abs(result.eta * (result.g_value - Lambda))
    return {
        "stationarity_residual": result.kkt_residual,
        "primal_feasibility": float(primal),
        "dual_feasibility": float(dual),
        "complementary_slackness": float(complementary),
        "satisfied": bool(
            result.kkt_residual < tol
            and primal > -tol
            and dual >= -tol
            and complementary < tol
        ),
    }
