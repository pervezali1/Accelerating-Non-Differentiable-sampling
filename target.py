"""Target potential ``U``, smooth anchor ``U0`` and the anchor coefficient ``a``.

Target
------
    U(w) = sum_i [ softplus(X_i @ w) - y_i * (X_i @ w) ]
           + w[0]^2 / (2 sigma_intercept^2)
           + lambda_lasso * sum_{j=1}^{d-1} |w[j]|

The likelihood is a **sum** of per-observation contributions (never an
average).  The intercept ``w[0]`` carries a weak Gaussian prior and is
**excluded** from the L1 penalty.

Smooth anchor
-------------
    U0(w) = sum_i [ softplus(X_i @ w) - y_i * (X_i @ w) ]
            + w[0]^2 / (2 sigma_intercept^2)
            + lambda_lasso * sum_{j=1}^{d-1} sqrt(w[j]^2 + delta_anchor^2)

    grad_likelihood = X.T @ (sigmoid(X @ w) - y)
    grad_prior[0]   = w[0] / sigma_intercept^2
    grad_prior[j]   = lambda_lasso * w[j] / sqrt(w[j]^2 + delta_anchor^2),  j >= 1
    grad_U0         = grad_likelihood + grad_prior

Anchor coefficient
------------------
    log_a(w) = lambda_lasso * sum_{j=1}^{d-1} ( |w[j]| - sqrt(w[j]^2 + delta^2) )
    a(w)     = exp(log_a(w))

This is computed directly from the difference of the two penalties -- never as
``exp(U) / exp(U0)``, which would overflow for a sum-likelihood of 2000 terms.
Because ``0 <= sqrt(t^2 + delta^2) - |t| <= delta``,

    -(d - 1) * lambda_lasso * delta <= log_a(w) <= 0,
    exp(-(d - 1) * lambda_lasso * delta) <= a(w) <= 1,

which for ``d = 9`` is the ``-8 * lambda_lasso * delta_anchor`` bound of the
specification.

Why ``a`` appears in the sampler
--------------------------------
Note that ``log_a = U0 - U + (U - U0) = U - U0`` by construction, i.e.
``a = exp(U - U0)``.  The anchored diffusion

    dw = -a(w) grad_U0(w) dt + sqrt(2 a(w)) dW

has stationary density proportional to ``exp(-U0) / a = exp(-U)``: for
``pi ∝ exp(-U0)/a`` one gets ``a pi ∝ exp(-U0)`` and the Fokker-Planck flux
``-b pi + grad(a pi) = a grad_U0 pi - grad_U0 a pi = 0``.  So the *smooth*
gradient ``grad_U0`` is all that is ever evaluated, while the sampled
distribution is the *non-differentiable* target ``exp(-U)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from config import TargetConfig


@dataclass
class LogisticTarget:
    """Callable bundle for ``U``, ``U0``, ``grad_U0`` and ``log_a`` ."""

    X: np.ndarray                 # (n, d) design matrix, column 0 = intercept
    y: np.ndarray                 # (n,)
    lambda_lasso: float
    sigma_intercept: float
    delta_anchor: float

    @classmethod
    def from_config(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        target_cfg: TargetConfig,
    ) -> "LogisticTarget":
        return cls(
            X=np.asarray(X, dtype=float),
            y=np.asarray(y, dtype=float),
            lambda_lasso=target_cfg.lambda_lasso,
            sigma_intercept=target_cfg.sigma_intercept,
            delta_anchor=target_cfg.delta_anchor,
        )

    @property
    def d(self) -> int:
        return self.X.shape[1]

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------
    def negative_log_likelihood(self, w: np.ndarray) -> float:
        """``sum_i [softplus(X_i @ w) - y_i (X_i @ w)]`` (a sum, not a mean)."""
        linear = self.X @ w
        # numpy.logaddexp(0, z) is the numerically stable softplus.
        return float(np.sum(np.logaddexp(0.0, linear) - self.y * linear))

    def grad_likelihood(self, w: np.ndarray) -> np.ndarray:
        """``X.T @ (sigmoid(X @ w) - y)``."""
        return self.X.T @ (expit(self.X @ w) - self.y)

    # ------------------------------------------------------------------
    # Potentials
    # ------------------------------------------------------------------
    def intercept_prior(self, w: np.ndarray) -> float:
        """``w[0]^2 / (2 sigma_intercept^2)``."""
        return float(w[0] ** 2 / (2.0 * self.sigma_intercept ** 2))

    def U(self, w: np.ndarray) -> float:
        """Non-differentiable target potential."""
        w = np.asarray(w, dtype=float)
        l1 = self.lambda_lasso * float(np.abs(w[1:]).sum())
        return self.negative_log_likelihood(w) + self.intercept_prior(w) + l1

    def U0(self, w: np.ndarray) -> float:
        """Smooth anchor potential (``|w_j|`` replaced by ``sqrt(w_j^2+delta^2)``)."""
        w = np.asarray(w, dtype=float)
        smooth_l1 = self.lambda_lasso * float(
            np.sqrt(w[1:] ** 2 + self.delta_anchor ** 2).sum()
        )
        return self.negative_log_likelihood(w) + self.intercept_prior(w) + smooth_l1

    def grad_prior(self, w: np.ndarray) -> np.ndarray:
        """Gradient of the intercept prior plus the smoothed L1 anchor."""
        w = np.asarray(w, dtype=float)
        out = np.empty_like(w)
        out[0] = w[0] / self.sigma_intercept ** 2
        out[1:] = (
            self.lambda_lasso
            * w[1:]
            / np.sqrt(w[1:] ** 2 + self.delta_anchor ** 2)
        )
        return out

    def grad_U0(self, w: np.ndarray) -> np.ndarray:
        """``grad_U0 = grad_likelihood + grad_prior``."""
        return self.grad_likelihood(w) + self.grad_prior(w)

    # ------------------------------------------------------------------
    # Anchor coefficient
    # ------------------------------------------------------------------
    def log_a(self, w: np.ndarray) -> float:
        """``lambda_lasso * sum_{j>=1} (|w_j| - sqrt(w_j^2 + delta^2))``.

        Computed from the penalty difference directly, never as a ratio of
        exponentials of the (large) potentials.
        """
        w = np.asarray(w, dtype=float)
        slopes = w[1:]
        return float(
            self.lambda_lasso
            * np.sum(np.abs(slopes) - np.sqrt(slopes ** 2 + self.delta_anchor ** 2))
        )

    def a(self, w: np.ndarray) -> float:
        """``a(w) = exp(log_a(w)) in (0, 1]``."""
        return float(np.exp(self.log_a(w)))

    @property
    def log_a_lower_bound(self) -> float:
        """``-(d - 1) * lambda_lasso * delta_anchor`` (= -8*lambda*delta for d=9)."""
        return -(self.d - 1) * self.lambda_lasso * self.delta_anchor

    @property
    def a_lower_bound(self) -> float:
        """``exp(-(d - 1) * lambda_lasso * delta_anchor)``."""
        return float(np.exp(self.log_a_lower_bound))

    def check_anchor_bounds(self, w: np.ndarray, tol: float = 1e-12) -> bool:
        """Verify the theoretical bounds on ``log_a`` and ``a``."""
        log_a_value = self.log_a(w)
        a_value = np.exp(log_a_value)
        return bool(
            self.log_a_lower_bound - tol <= log_a_value <= tol
            and self.a_lower_bound - tol <= a_value <= 1.0 + tol
        )

    # ------------------------------------------------------------------
    # Curvature and the smooth MAP
    # ------------------------------------------------------------------
    def lipschitz_constant(self) -> float:
        """Upper bound on the largest eigenvalue of ``Hess U0``.

        ``Hess = X^T D X + diag(prior curvature)`` with ``D_ii = s(1-s) <= 1/4``
        and the anchor curvature ``lambda * delta^2 / (w^2 + delta^2)^{3/2}``
        bounded by ``lambda / delta``.
        """
        eig_max = float(np.linalg.eigvalsh(self.X.T @ self.X).max())
        prior_curvature = max(
            1.0 / self.sigma_intercept ** 2,
            self.lambda_lasso / self.delta_anchor,
        )
        return 0.25 * eig_max + prior_curvature

    def smooth_map(self, w0: np.ndarray | None = None) -> np.ndarray:
        """Minimiser of the smooth anchor ``U0`` (the 'smooth MAP')."""
        if w0 is None:
            w0 = np.zeros(self.d)
        result = minimize(
            fun=lambda w: self.U0(w),
            x0=np.asarray(w0, dtype=float),
            jac=lambda w: self.grad_U0(w),
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10},
        )
        return result.x


@dataclass
class WeightedL1Target:
    """Section 13 validation target ``U(x) = sum_i omega_i |x_i|``.

    Smooth anchor ``U0(x) = sum_i omega_i sqrt(x_i^2 + delta_anchor^2)`` with

        grad_U0[i] = omega_i * x_i / sqrt(x_i^2 + delta_anchor^2),
        log_a(x)   = sum_i omega_i ( |x_i| - sqrt(x_i^2 + delta_anchor^2) ).

    Every coordinate is penalised here (there is no intercept), so the bound is
    ``-delta_anchor * sum_i omega_i <= log_a <= 0``.
    """

    omega: np.ndarray
    delta_anchor: float

    @property
    def d(self) -> int:
        return int(self.omega.size)

    def U(self, x: np.ndarray) -> float:
        return float(np.sum(self.omega * np.abs(np.asarray(x, dtype=float))))

    def U0(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.sum(self.omega * np.sqrt(x ** 2 + self.delta_anchor ** 2)))

    def grad_U0(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.omega * x / np.sqrt(x ** 2 + self.delta_anchor ** 2)

    def log_a(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(
            np.sum(self.omega * (np.abs(x) - np.sqrt(x ** 2 + self.delta_anchor ** 2)))
        )

    def a(self, x: np.ndarray) -> float:
        return float(np.exp(self.log_a(x)))

    @property
    def log_a_lower_bound(self) -> float:
        return float(-self.delta_anchor * self.omega.sum())

    @property
    def a_lower_bound(self) -> float:
        return float(np.exp(self.log_a_lower_bound))

    def check_anchor_bounds(self, x: np.ndarray, tol: float = 1e-12) -> bool:
        log_a_value = self.log_a(x)
        return bool(
            self.log_a_lower_bound - tol <= log_a_value <= tol
            and self.a_lower_bound - tol <= np.exp(log_a_value) <= 1.0 + tol
        )

    def lipschitz_constant(self) -> float:
        """``max_i omega_i / delta_anchor`` bounds the curvature of ``U0``."""
        return float(self.omega.max() / self.delta_anchor)
