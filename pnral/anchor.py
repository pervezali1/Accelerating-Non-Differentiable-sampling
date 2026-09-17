"""The smooth anchor potential U0, its gradient and the anchoring coefficient.

The non-smooth L1 term of the target ``U`` is replaced by the smooth surrogate
``sqrt(w_j^2 + delta_anchor^2)``:

.. math::

    U_0(w) = \\sum_i [\\mathrm{softplus}(z_i) - y_i z_i]
             + \\frac{w_0^2}{2\\sigma_{\\mathrm{intercept}}^2}
             + \\lambda_{\\mathrm{lasso}}
               \\sum_{j\\ge 1}\\sqrt{w_j^2 + \\delta_{\\mathrm{anchor}}^2}.

The two potentials differ by the *log anchoring coefficient*

.. math::

    \\log a(w) = \\lambda_{\\mathrm{lasso}} \\sum_{j\\ge 1}
        \\Big(|w_j| - \\sqrt{w_j^2 + \\delta_{\\mathrm{anchor}}^2}\\Big)
        = U(w) - U_0(w),

which is evaluated **directly** from this formula and never as a ratio of two
exponentials.  Because ``0 <= sqrt(w^2 + delta^2) - |w| <= delta`` we have the
sharp bounds

    ``-n_slopes * lambda_lasso * delta_anchor <= log a(w) <= 0``,
    ``exp(-n_slopes * lambda_lasso * delta_anchor) <= a(w) <= 1``,

with ``n_slopes = d - 1 = 10`` for the MAGIC design.

Why ``a`` appears in the dynamics
---------------------------------
The anchored Langevin diffusion

    ``dW = -a(W) grad U0(W) dt + sqrt(2 a(W)) dB``

has stationary density ``pi ∝ exp(-U0)/a = exp(-U)`` -- substituting
``pi = e^{-U_0}/a`` into the stationary Fokker--Planck equation

    ``div( a grad(U0) pi + grad(a pi) ) = div( -grad e^{-U_0} + grad e^{-U_0} ) = 0``

shows this exactly, and in particular no ``grad a`` correction term is needed
in the drift.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import expit

from .target import LogisticTarget, softplus

__all__ = [
    "smooth_abs",
    "potential_U0",
    "grad_U0",
    "value_and_grad_U0",
    "log_anchor_coefficient",
    "anchor_coefficient",
    "anchor_bounds",
    "check_anchor_bounds",
]


def smooth_abs(w: np.ndarray, delta_anchor: float) -> np.ndarray:
    """``sqrt(w^2 + delta_anchor^2)``: the smoothed absolute value."""
    return np.sqrt(w * w + delta_anchor * delta_anchor)


def potential_U0(target: LogisticTarget, w: np.ndarray,
                 z: np.ndarray | None = None) -> float:
    """Smooth anchor potential ``U0(w)``."""
    w = np.asarray(w, dtype=np.float64)
    slopes = w[target.slope_slice]
    return (target.negative_log_likelihood(w, z)
            + target.intercept_prior_potential(w)
            + float(target.lambda_lasso
                    * np.sum(smooth_abs(slopes, target.delta_anchor))))


def grad_U0(target: LogisticTarget, w: np.ndarray) -> np.ndarray:
    """Gradient of the anchor potential.

    ``probabilities = expit(X @ w)``,
    ``grad_likelihood = X.T @ (probabilities - y)``,
    ``grad_prior[0] = w[0] / sigma_intercept^2`` and, for ``j >= 1``,
    ``grad_prior[j] = lambda_lasso * w[j] / sqrt(w[j]^2 + delta_anchor^2)``.
    """
    return value_and_grad_U0(target, w)[1]


def value_and_grad_U0(target: LogisticTarget,
                      w: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return ``(U0(w), grad U0(w))`` sharing a single pass over the data.

    The linear predictor ``z = X @ w`` is by far the dominant cost, so the
    value and the gradient are always produced together inside the sampler.
    """
    w = np.asarray(w, dtype=np.float64)
    z = target.X @ w
    probabilities = expit(z)

    # --- value ---------------------------------------------------------
    negative_log_likelihood = (float(np.sum(softplus(z)))
                               - float(target.y @ z)) / target.temperature
    slopes = w[target.slope_slice]
    smoothed = smooth_abs(slopes, target.delta_anchor)
    value = (negative_log_likelihood
             + float(w[0] * w[0] / (2.0 * target.sigma_intercept ** 2))
             + float(target.lambda_lasso * np.sum(smoothed)))

    # --- gradient ------------------------------------------------------
    grad_likelihood = (target.X.T @ (probabilities - target.y)) / target.temperature
    grad_prior = np.empty_like(grad_likelihood)
    grad_prior[0] = w[0] / target.sigma_intercept ** 2          # Gaussian prior
    grad_prior[target.slope_slice] = (target.lambda_lasso * slopes / smoothed)
    gradient = grad_likelihood + grad_prior

    target.gradient_evaluations += 1
    return float(value), np.asarray(gradient, dtype=np.float64)


def log_anchor_coefficient(target: LogisticTarget, w: np.ndarray) -> float:
    """``log a(w)`` evaluated directly (never as ``log(exp(U)/exp(U0))``)."""
    w = np.asarray(w, dtype=np.float64)
    slopes = w[target.slope_slice]
    return float(target.lambda_lasso
                 * np.sum(np.abs(slopes) - smooth_abs(slopes, target.delta_anchor)))


def anchor_coefficient(target: LogisticTarget, w: np.ndarray) -> float:
    """``a(w) = exp(log a(w)) in (0, 1]``."""
    return float(np.exp(log_anchor_coefficient(target, w)))


def anchor_bounds(target: LogisticTarget) -> Tuple[float, float, float, float]:
    """Theoretical bounds ``(log_a_min, log_a_max, a_min, a_max)``.

    ``log_a_min = -(d - 1) * lambda_lasso * delta_anchor`` (ten slopes for the
    MAGIC design, hence the ``-10 * lambda_lasso * delta_anchor`` of the
    specification) and ``log_a_max = 0``.
    """
    n_slopes = target.d - 1
    log_a_min = -n_slopes * target.lambda_lasso * target.delta_anchor
    return log_a_min, 0.0, float(np.exp(log_a_min)), 1.0


def check_anchor_bounds(target: LogisticTarget, w: np.ndarray,
                        tolerance: float = 1e-10) -> None:
    """Raise :class:`AssertionError` if ``log a(w)`` leaves its interval."""
    log_a = log_anchor_coefficient(target, w)
    log_a_min, log_a_max, a_min, a_max = anchor_bounds(target)
    if not (log_a_min - tolerance <= log_a <= log_a_max + tolerance):
        raise AssertionError(
            f"log a(w) = {log_a!r} outside [{log_a_min}, {log_a_max}]")
    a = float(np.exp(log_a))
    if not (a_min - tolerance <= a <= a_max + tolerance):
        raise AssertionError(f"a(w) = {a!r} outside [{a_min}, {a_max}]")
