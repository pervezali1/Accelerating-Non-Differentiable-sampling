"""The constrained target potential U (negative log posterior).

For labels ``y_i in {0, 1}`` and ``z = X @ w`` the target potential is

.. math::

    U(w) = \\sum_i [\\mathrm{softplus}(z_i) - y_i z_i]
           + \\frac{w_0^2}{2\\sigma_{\\mathrm{intercept}}^2}
           + \\lambda_{\\mathrm{lasso}} \\sum_{j\\ge 1} |w_j| ,

i.e. a Bayesian logistic regression with a Laplace (LASSO) prior on the ten
slopes and a weak Gaussian prior on the intercept.  The intercept is **never**
L1-penalised.

The likelihood contribution is a *sum* over the training rows (not an
average); a tempered variant is available but must be requested explicitly
through ``temperature`` and is always labelled as such in the outputs.

Numerics
--------
``softplus(z) = log(1 + e^z)`` is evaluated with :func:`numpy.logaddexp` and
the sigmoid with :func:`scipy.special.expit`, both of which are stable for
``|z|`` of several hundred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.special import expit

__all__ = ["LogisticTarget", "softplus"]


def softplus(z: np.ndarray) -> np.ndarray:
    """Numerically stable ``log(1 + exp(z))``."""
    return np.logaddexp(0.0, z)


@dataclass
class LogisticTarget:
    """Bundles the training data with the potentials U and U0.

    Parameters
    ----------
    X, y:
        Training design matrix (intercept in column 0) and ``{0,1}`` labels.
    lambda_lasso:
        L1 penalty applied to coordinates ``1 .. d-1`` only.
    sigma_intercept:
        Standard deviation of the Gaussian prior on ``w[0]``.
    delta_anchor:
        Smoothing width of the anchor potential (see :mod:`pnral.anchor`).
    temperature:
        ``1.0`` reproduces the specification exactly.  Any other value
        multiplies the *likelihood* by ``1/temperature`` and marks the run as
        a clearly-labelled tempered-posterior experiment.
    """

    X: np.ndarray
    y: np.ndarray
    lambda_lasso: float
    sigma_intercept: float
    delta_anchor: float
    temperature: float = 1.0

    def __post_init__(self) -> None:
        self.X = np.ascontiguousarray(self.X, dtype=np.float64)
        self.y = np.ascontiguousarray(self.y, dtype=np.float64)
        if self.X.ndim != 2:
            raise ValueError("X must be a 2-d design matrix")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y have inconsistent numbers of rows")
        if not np.all(np.isin(self.y, (0.0, 1.0))):
            raise ValueError("labels must be encoded in {0, 1}")
        if self.lambda_lasso < 0.0:
            raise ValueError("lambda_lasso must be non-negative")
        if self.sigma_intercept <= 0.0:
            raise ValueError("sigma_intercept must be positive")
        if self.delta_anchor <= 0.0:
            raise ValueError("delta_anchor must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        #: Number of evaluations of grad U0 (reported as a cost measure).
        self.gradient_evaluations: int = 0

    # ------------------------------------------------------------------
    @property
    def d(self) -> int:
        """Parameter dimension, read from the design matrix."""
        return int(self.X.shape[1])

    @property
    def n(self) -> int:
        """Number of training observations."""
        return int(self.X.shape[0])

    @property
    def slope_slice(self) -> slice:
        """Coordinates carrying the L1 penalty: everything but the intercept."""
        return slice(1, self.d)

    # ------------------------------------------------------------------
    def linear_predictor(self, w: np.ndarray) -> np.ndarray:
        """``z = X @ w``."""
        return self.X @ w

    def negative_log_likelihood(self, w: np.ndarray,
                                z: np.ndarray | None = None) -> float:
        """``sum_i [softplus(z_i) - y_i z_i]``, divided by the temperature."""
        z = self.linear_predictor(w) if z is None else z
        value = float(np.sum(softplus(z)) - float(self.y @ z))
        return value / self.temperature

    def intercept_prior_potential(self, w: np.ndarray) -> float:
        """``w[0]^2 / (2 sigma_intercept^2)`` -- the intercept is not L1 penalised."""
        return float(w[0] * w[0] / (2.0 * self.sigma_intercept ** 2))

    def l1_penalty(self, w: np.ndarray) -> float:
        """``lambda_lasso * sum_{j>=1} |w_j|`` (intercept excluded)."""
        return float(self.lambda_lasso * np.sum(np.abs(w[self.slope_slice])))

    def potential_U(self, w: np.ndarray, z: np.ndarray | None = None) -> float:
        """The non-smooth constrained target potential ``U(w)``.

        ``U`` is only ever *evaluated* (for diagnostics and for the identity
        ``U = U0 + log a``); the sampler differentiates the smooth anchor
        ``U0`` instead, which is what makes the scheme well defined despite
        the non-differentiable L1 term.
        """
        w = np.asarray(w, dtype=np.float64)
        return (self.negative_log_likelihood(w, z)
                + self.intercept_prior_potential(w)
                + self.l1_penalty(w))

    def predict_probabilities(self, w: np.ndarray,
                              X: np.ndarray | None = None) -> np.ndarray:
        """``expit(X @ w)`` for the training matrix or any other design."""
        design = self.X if X is None else X
        return expit(design @ np.asarray(w, dtype=np.float64))

    # ------------------------------------------------------------------
    def likelihood_smoothness_guide(self) -> float:
        """Guide value ``L_likelihood <= 0.25 * ||X||_2^2``.

        The logistic Hessian is ``X^T diag(p(1-p)) X`` with
        ``p(1-p) <= 1/4``, so the largest eigenvalue is bounded by
        ``0.25 * ||X||_2^2``.  This is used *only* to pick an initial step
        size; the step size is then validated empirically.
        """
        spectral_norm = float(np.linalg.norm(self.X, ord=2))
        return 0.25 * spectral_norm ** 2 / self.temperature
