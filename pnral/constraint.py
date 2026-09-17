"""The smoothed l_p constraint and the pilot rule for its threshold.

Constraint functional
---------------------
.. math::

    g(w) = \\sum_{i=0}^{d-1}
           \\big(w_i^2 + \\epsilon_{\\mathrm{constraint}}^2\\big)^{p/2},
    \\qquad
    K = \\{w \\in \\mathbb{R}^d : g(w) \\le \\Lambda_{\\mathrm{constraint}}\\}.

``g`` is a smooth surrogate of ``||w||_p^p``; it is separable, strictly
convex for ``p >= 1`` and its minimum is attained at the origin with

    ``g(0) = d * epsilon_constraint ** p_constraint``,

so ``Lambda_constraint`` must strictly exceed that value for ``K`` to be
non-empty.  The gradient is

    ``grad_g[i] = p * w_i * (w_i^2 + eps^2) ** (p/2 - 1)``,

and the slack function used by the sampler is ``psi = Lambda - g`` with
``grad_psi = -grad_g``.

Interpretation
--------------
The constrained posterior is the **truncation** of the unconstrained
posterior ``exp(-U)`` to ``K``:

    ``pi_K(w) ∝ exp(-U(w)) 1{g(w) <= Lambda_constraint}``.

It is a different probability measure from the unconstrained posterior, and
comparisons across different thresholds compare *different targets*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "g_constraint",
    "grad_g_constraint",
    "psi_constraint",
    "grad_psi_constraint",
    "g_minimum",
    "validate_threshold",
    "choose_threshold_from_pilot",
    "outward_normal",
    "ThresholdSelection",
]


def g_constraint(w: np.ndarray, p_constraint: float,
                 epsilon_constraint: float) -> float:
    """``g(w) = sum_i (w_i^2 + eps^2) ** (p/2)``.

    Accepts a single ``(d,)`` vector or a ``(n, d)`` batch, in which case a
    ``(n,)`` array of values is returned.
    """
    w = np.asarray(w, dtype=np.float64)
    terms = (w * w + epsilon_constraint ** 2) ** (0.5 * p_constraint)
    if w.ndim == 1:
        return float(np.sum(terms))
    return np.sum(terms, axis=-1)


def grad_g_constraint(w: np.ndarray, p_constraint: float,
                      epsilon_constraint: float) -> np.ndarray:
    """``grad_g[i] = p * w_i * (w_i^2 + eps^2) ** (p/2 - 1)``."""
    w = np.asarray(w, dtype=np.float64)
    return p_constraint * w * (w * w + epsilon_constraint ** 2) ** (0.5 * p_constraint - 1.0)


def psi_constraint(w: np.ndarray, p_constraint: float,
                   epsilon_constraint: float,
                   Lambda_constraint: float) -> float:
    """Slack ``psi(w) = Lambda_constraint - g(w)`` (positive inside ``K``)."""
    return float(Lambda_constraint - g_constraint(w, p_constraint, epsilon_constraint))


def grad_psi_constraint(w: np.ndarray, p_constraint: float,
                        epsilon_constraint: float) -> np.ndarray:
    """``grad_psi(w) = -grad_g(w)``; it defines the axes of the swirl blocks."""
    return -grad_g_constraint(w, p_constraint, epsilon_constraint)


def g_minimum(d: int, p_constraint: float, epsilon_constraint: float) -> float:
    """``g(0) = d * epsilon_constraint ** p_constraint``, the global minimum."""
    return float(d) * float(epsilon_constraint) ** float(p_constraint)


def validate_threshold(Lambda_constraint: float, d: int, p_constraint: float,
                       epsilon_constraint: float) -> float:
    """Check ``Lambda_constraint > g(0)`` and return ``g(0)``.

    Raises
    ------
    ValueError
        If the constraint set would be empty or a single point.
    """
    minimum = g_minimum(d, p_constraint, epsilon_constraint)
    if not np.isfinite(Lambda_constraint):
        raise ValueError("Lambda_constraint must be finite")
    if Lambda_constraint <= minimum:
        raise ValueError(
            f"Lambda_constraint = {Lambda_constraint} must exceed "
            f"g(0) = d * epsilon**p = {minimum}; the constraint set would be empty"
        )
    return minimum


def outward_normal(w: np.ndarray, p_constraint: float,
                   epsilon_constraint: float) -> np.ndarray:
    """Unit outward normal ``grad_g(w) / ||grad_g(w)||`` of the level set."""
    gradient = grad_g_constraint(w, p_constraint, epsilon_constraint)
    norm = float(np.linalg.norm(gradient))
    if norm == 0.0:
        raise ValueError("grad_g vanishes at w = 0; the normal is undefined")
    return gradient / norm


@dataclass
class ThresholdSelection:
    """Outcome of the pilot-based choice of ``Lambda_constraint``."""

    Lambda_constraint: float
    g_min: float
    pilot_quantile_level: float
    pilot_quantile_value: float
    inflation: float
    n_pilot_samples: int
    exceedance_fraction: float
    Lambda_small: float
    Lambda_large: float
    from_pilot: bool = True

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in asdict(self).items()}


def choose_threshold_from_pilot(
    pilot_g_values: np.ndarray,
    d: int,
    p_constraint: float,
    epsilon_constraint: float,
    quantile: float = 0.999,
    inflation: float = 1.10,
    lambda_small_factor: float = 0.90,
    lambda_large_factor: float = 1.10,
) -> ThresholdSelection:
    """Pilot rule of section 6.

    ``Lambda_constraint = g_min + inflation * (q - g_min)`` where ``q`` is the
    empirical ``quantile`` of ``g(w_k)`` over the retained samples of an
    *unconstrained* reversible (``alpha = 0``) anchored pilot chain.

    The returned object also carries the radii used by the constraint
    sensitivity study,

        ``Lambda_small = g_min + 0.90 * (Lambda_constraint - g_min)``,
        ``Lambda_large = g_min + 1.10 * (Lambda_constraint - g_min)``,

    and the fraction of pilot samples that violate the chosen threshold.
    Once returned, ``Lambda_constraint`` is frozen: the *same* value is used
    by every alpha of the primary comparison.
    """
    pilot_g_values = np.asarray(pilot_g_values, dtype=np.float64).ravel()
    if pilot_g_values.size == 0:
        raise ValueError("no pilot samples supplied")
    minimum = g_minimum(d, p_constraint, epsilon_constraint)
    quantile_value = float(np.quantile(pilot_g_values, quantile))
    Lambda_constraint = float(minimum + inflation * (quantile_value - minimum))
    validate_threshold(Lambda_constraint, d, p_constraint, epsilon_constraint)

    exceedance = float(np.mean(pilot_g_values > Lambda_constraint))
    radius = Lambda_constraint - minimum
    return ThresholdSelection(
        Lambda_constraint=Lambda_constraint,
        g_min=minimum,
        pilot_quantile_level=float(quantile),
        pilot_quantile_value=quantile_value,
        inflation=float(inflation),
        n_pilot_samples=int(pilot_g_values.size),
        exceedance_fraction=exceedance,
        Lambda_small=float(minimum + lambda_small_factor * radius),
        Lambda_large=float(minimum + lambda_large_factor * radius),
        from_pilot=True,
    )


def fixed_threshold(Lambda_constraint: float, d: int, p_constraint: float,
                    epsilon_constraint: float,
                    lambda_small_factor: float = 0.90,
                    lambda_large_factor: float = 1.10) -> ThresholdSelection:
    """Wrap a user-supplied ``Lambda_constraint``, bypassing the pilot run."""
    minimum = validate_threshold(Lambda_constraint, d, p_constraint,
                                 epsilon_constraint)
    radius = Lambda_constraint - minimum
    return ThresholdSelection(
        Lambda_constraint=float(Lambda_constraint),
        g_min=minimum,
        pilot_quantile_level=float("nan"),
        pilot_quantile_value=float("nan"),
        inflation=float("nan"),
        n_pilot_samples=0,
        exceedance_fraction=float("nan"),
        Lambda_small=float(minimum + lambda_small_factor * radius),
        Lambda_large=float(minimum + lambda_large_factor * radius),
        from_pilot=False,
    )
