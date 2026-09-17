r"""The logistic-LASSO target ``U``, its smooth anchor ``U0``, and the clock ``a``.

The target potential is

.. math::
    U(w) = \sum_i [\mathrm{softplus}(x_i'w) - y_i x_i'w]
           + \frac{\beta_0^2}{2\sigma_0^2}
           + \lambda \sum_{j\ge 1} |\beta_j|,

a sum -- not an average -- over the training observations, with the intercept
excluded from the L1 penalty and given a weak Gaussian prior instead.  The
smooth anchor replaces each ``|beta_j|`` by ``sqrt(beta_j^2 + delta^2)``:

.. math::
    U_0(w) = \sum_i [\mathrm{softplus}(x_i'w) - y_i x_i'w]
             + \frac{\beta_0^2}{2\sigma_0^2}
             + \lambda \sum_{j\ge 1} \sqrt{\beta_j^2 + \delta^2}.

Anchored Langevin runs the smooth dynamics of ``U0`` but slows its own clock by

.. math::
    a(w) = e^{U(w) - U_0(w)}
         = \exp\Big(\lambda \sum_{j\ge1}\big(|\beta_j| - \sqrt{\beta_j^2+\delta^2}\big)\Big),

which leaves ``e^{-U} \propto e^{-U_0}/a`` invariant.  Two things are worth
noticing about that expression, and both are used below.  First the likelihood
and the intercept prior *cancel identically*, so the clock costs ``O(p)`` and
never touches the data -- computing it as a ratio of two exponentiated
potentials would be both wasteful and numerically hopeless.  Second, since
``|b| <= sqrt(b^2 + delta^2) <= |b| + delta`` termwise,

.. math::
    -\lambda p \delta \;\le\; \log a(w) \;\le\; 0,
    \qquad e^{-\lambda p \delta} \;\le\; a(w) \;\le\; 1,

so the clock can be bounded a priori: it never speeds the dynamics up and never
stalls them completely.  :func:`check_anchor_bounds` asserts exactly this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

__all__ = [
    "LogisticTarget",
    "target_potential",
    "anchor_potential",
    "grad_anchor",
    "grad_target_likelihood",
    "log_anchor_coefficient",
    "anchor_coefficient",
    "anchor_bounds",
    "check_anchor_bounds",
    "anchor_hessian",
    "delta_for_clock_budget",
    "sigmoid",
    "softplus",
]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return expit(z)


def softplus(z: np.ndarray) -> np.ndarray:
    """``log(1 + exp(z))``, stable for large ``|z|``."""
    return np.logaddexp(0.0, z)


def _as_vector(w: np.ndarray, dim: int | None = None) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).ravel()
    if dim is not None and w.size != dim:
        raise ValueError(f"expected a parameter vector of length {dim}, got {w.size}")
    return w


def _negative_log_likelihood(
    w: np.ndarray, X: np.ndarray, y: np.ndarray, weight: float
) -> float:
    z = X @ w
    return float(weight * np.sum(softplus(z) - y * z))


def _penalty_terms(w: np.ndarray) -> np.ndarray:
    """The penalised coefficients ``beta_1, ..., beta_p`` -- the intercept is out."""
    return w[1:]


def target_potential(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    sigma0: float,
    tempered: bool = False,
) -> float:
    r"""``U(w)``: logistic negative log-likelihood, intercept prior, L1 penalty."""
    w = _as_vector(w, X.shape[1])
    weight = 1.0 / X.shape[0] if tempered else 1.0
    nll = _negative_log_likelihood(w, X, y, weight)
    intercept = 0.5 * w[0] ** 2 / sigma0**2
    l1 = lambda_ * float(np.sum(np.abs(_penalty_terms(w))))
    return nll + intercept + l1


def anchor_potential(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    sigma0: float,
    delta: float,
    tempered: bool = False,
) -> float:
    r"""``U0(w)``: the same, with ``|beta_j|`` replaced by ``sqrt(beta_j^2+delta^2)``."""
    w = _as_vector(w, X.shape[1])
    weight = 1.0 / X.shape[0] if tempered else 1.0
    nll = _negative_log_likelihood(w, X, y, weight)
    intercept = 0.5 * w[0] ** 2 / sigma0**2
    beta = _penalty_terms(w)
    smooth_l1 = lambda_ * float(np.sum(np.sqrt(beta * beta + delta * delta)))
    return nll + intercept + smooth_l1


def grad_target_likelihood(
    w: np.ndarray, X: np.ndarray, y: np.ndarray, tempered: bool = False
) -> np.ndarray:
    r"""``X' (sigmoid(Xw) - y)``, the gradient of the negative log-likelihood."""
    w = _as_vector(w, X.shape[1])
    weight = 1.0 / X.shape[0] if tempered else 1.0
    return weight * (X.T @ (sigmoid(X @ w) - y))


def grad_anchor(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    sigma0: float,
    delta: float,
    tempered: bool = False,
) -> np.ndarray:
    r"""``grad U0(w)``.

    ``grad_likelihood = X'(sigmoid(Xw) - y)``,
    ``grad_prior[0] = beta_0 / sigma0^2`` and, for ``j >= 1``,
    ``grad_prior[j] = lambda beta_j / sqrt(beta_j^2 + delta^2)``.  The ``j = 0``
    entry of the penalty gradient is exactly zero by construction, which is
    validation check 7.
    """
    w = _as_vector(w, X.shape[1])
    grad = grad_target_likelihood(w, X, y, tempered)
    prior = np.zeros_like(grad)
    prior[0] = w[0] / sigma0**2
    beta = _penalty_terms(w)
    prior[1:] = lambda_ * beta / np.sqrt(beta * beta + delta * delta)
    return grad + prior


def log_anchor_coefficient(w: np.ndarray, lambda_: float, delta: float) -> float:
    r"""``log a(w) = lambda sum_{j>=1} (|beta_j| - sqrt(beta_j^2 + delta^2))``.

    Computed directly from the penalty, never as ``U(w) - U0(w)`` with the two
    potentials evaluated separately: the likelihood cancels analytically, so
    subtracting two numbers of size ``O(n)`` to get one of size ``O(lambda p
    delta)`` would throw away most of the available precision.
    """
    beta = _penalty_terms(_as_vector(w))
    return float(
        lambda_ * np.sum(np.abs(beta) - np.sqrt(beta * beta + delta * delta))
    )


def anchor_coefficient(w: np.ndarray, lambda_: float, delta: float) -> float:
    r"""``a(w) = exp(log a(w))``, in ``(0, 1]``."""
    return float(np.exp(log_anchor_coefficient(w, lambda_, delta)))


def anchor_bounds(dim: int, lambda_: float, delta: float) -> tuple[float, float]:
    r"""``(log_a_min, a_min) = (-lambda p delta, exp(-lambda p delta))``."""
    n_penalised = dim - 1
    log_a_min = -lambda_ * n_penalised * delta
    return log_a_min, float(np.exp(log_a_min))


def check_anchor_bounds(
    log_a: np.ndarray, dim: int, lambda_: float, delta: float, tol: float = 1e-9
) -> None:
    r"""Assert ``-lambda p delta <= log a <= 0`` and the matching bound on ``a``."""
    log_a = np.asarray(log_a, dtype=np.float64)
    log_a_min, a_min = anchor_bounds(dim, lambda_, delta)
    if not np.isfinite(log_a).all():
        raise AssertionError("log a(w) contains non-finite values")
    if log_a.max() > tol:
        raise AssertionError(
            f"log a(w) must be <= 0, saw {log_a.max():.3e}; the anchor is not a "
            "majoriser of the target"
        )
    if log_a.min() < log_a_min - tol:
        raise AssertionError(
            f"log a(w) must be >= -lambda p delta = {log_a_min:.6f}, "
            f"saw {log_a.min():.6f}"
        )
    a = np.exp(log_a)
    if a.min() < a_min - tol or a.max() > 1.0 + tol:
        raise AssertionError(
            f"a(w) must lie in [{a_min:.3e}, 1], saw "
            f"[{a.min():.3e}, {a.max():.6f}]"
        )


def delta_for_clock_budget(
    dim: int, lambda_: float, budget: float = 0.5
) -> float:
    r"""The ``delta`` whose worst-case clock is ``exp(-budget)``.

    From ``log a >= -lambda p delta``, choosing ``delta = budget / (lambda p)``
    guarantees ``a(w) >= exp(-budget)`` everywhere.  Offered as a helper rather
    than a default because it trades the other way too: the anchor's curvature
    at the kink is ``lambda / delta``, so a tight clock bound forces a small
    step size.
    """
    return float(budget / (lambda_ * max(dim - 1, 1)))


def anchor_hessian(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    sigma0: float,
    delta: float,
    tempered: bool = False,
) -> np.ndarray:
    r"""``Hess U0(w)``: ``X' diag(p(1-p)) X`` plus the diagonal prior curvature.

    The smoothed penalty contributes ``lambda delta^2 (beta_j^2 +
    delta^2)^{-3/2}`` on the diagonal, which is ``lambda / delta`` at a
    coefficient sitting exactly on the kink.  That is the largest curvature in
    the problem and hence what limits the step size, so this is used to
    calibrate ``h``.
    """
    w = _as_vector(w, X.shape[1])
    weight = 1.0 / X.shape[0] if tempered else 1.0
    p = sigmoid(X @ w)
    H = weight * (X.T @ (X * (p * (1.0 - p))[:, None]))
    diag = np.zeros(w.size)
    diag[0] = 1.0 / sigma0**2
    beta = _penalty_terms(w)
    diag[1:] = lambda_ * delta**2 / (beta * beta + delta**2) ** 1.5
    return H + np.diag(diag)


@dataclass(frozen=True)
class LogisticTarget:
    """``U``, ``U0``, ``grad U0`` and ``a`` bound to one dataset and one prior.

    Bundling them keeps the sampler's inner loop free of parameter passing and
    makes it impossible for a chain to be run against a different target than
    the one it is compared with (validation check 8).
    """

    X: np.ndarray
    y: np.ndarray
    lambda_: float
    sigma0: float
    delta: float
    tempered: bool = False

    @property
    def dim(self) -> int:
        return self.X.shape[1]

    @property
    def n_obs(self) -> int:
        return self.X.shape[0]

    def U(self, w: np.ndarray) -> float:
        return target_potential(
            w, self.X, self.y, self.lambda_, self.sigma0, self.tempered
        )

    def U0(self, w: np.ndarray) -> float:
        return anchor_potential(
            w, self.X, self.y, self.lambda_, self.sigma0, self.delta, self.tempered
        )

    def grad_U0(self, w: np.ndarray) -> np.ndarray:
        return grad_anchor(
            w, self.X, self.y, self.lambda_, self.sigma0, self.delta, self.tempered
        )

    def log_a(self, w: np.ndarray) -> float:
        return log_anchor_coefficient(w, self.lambda_, self.delta)

    def a(self, w: np.ndarray) -> float:
        return anchor_coefficient(w, self.lambda_, self.delta)

    def hessian_U0(self, w: np.ndarray) -> np.ndarray:
        return anchor_hessian(
            w, self.X, self.y, self.lambda_, self.sigma0, self.delta, self.tempered
        )

    def bounds(self) -> tuple[float, float]:
        return anchor_bounds(self.dim, self.lambda_, self.delta)

    def describe(self) -> dict[str, float | int | bool]:
        log_a_min, a_min = self.bounds()
        return {
            "dim": self.dim,
            "n_obs": self.n_obs,
            "lambda": self.lambda_,
            "sigma0": self.sigma0,
            "delta": self.delta,
            "tempered": self.tempered,
            "log_a_min": log_a_min,
            "a_min": a_min,
        }
