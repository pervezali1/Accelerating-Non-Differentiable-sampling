"""Bayesian logistic-regression posteriors, evaluated without derivatives.

The sampler in :mod:`nds.sampler` is derivative free: it only ever calls
:meth:`LogisticPosterior.potential` and :meth:`LogisticPosterior.fd_grad`, the
latter being a central-difference surrogate built from potential evaluations
alone.  :meth:`LogisticPosterior.grad` is the analytic gradient and is used
*only* to build the gold-standard reference posterior in :mod:`nds.reference`.

All methods are vectorised over an ensemble of walkers: a parameter block ``W``
has shape ``(d, m)`` for ``m`` walkers and returns per-walker quantities.
"""

from __future__ import annotations

import numpy as np


def softplus(Z: np.ndarray) -> np.ndarray:
    """log(1 + exp(Z)), evaluated stably."""
    return np.logaddexp(0.0, Z)


class LogisticPosterior:
    r"""Potential :math:`U(w) = -\log p(y \mid X, w) - \log p(w)`.

    The likelihood is logistic and the prior Gaussian, with a wider scale on
    the intercept so that it is effectively unpenalised.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prior_scale: float = 2.0,
        intercept_scale: float = 10.0,
    ) -> None:
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.n, self.d = self.X.shape
        self.prior_precision = np.full(self.d, 1.0 / prior_scale**2)
        self.prior_precision[0] = 1.0 / intercept_scale**2
        # Sufficient statistics reused by the potential and its surrogate.
        self._Xty = self.X.T @ self.y  # (d,)

    # ------------------------------------------------------------------ core
    def linear(self, W: np.ndarray) -> np.ndarray:
        """Linear predictors ``X @ W``, shape (n, m)."""
        return self.X @ W

    def potential(self, W: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        """U(W) for each walker, shape (m,)."""
        Z = self.linear(W) if Z is None else Z
        nll = softplus(Z).sum(axis=0) - self.y @ Z
        prior = 0.5 * (self.prior_precision[:, None] * W**2).sum(axis=0)
        return nll + prior

    def grad(self, W: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        """Analytic gradient of U, shape (d, m).  Reference use only."""
        Z = self.linear(W) if Z is None else Z
        resid = _sigmoid(Z) - self.y[:, None]
        return self.X.T @ resid + self.prior_precision[:, None] * W

    # ------------------------------------------------------- derivative free
    def fd_grad(
        self, W: np.ndarray, eps: float = 1e-2, Z: np.ndarray | None = None
    ) -> np.ndarray:
        r"""Central-difference surrogate of :math:`\nabla U`, shape (d, m).

        Coordinate ``j`` uses ``(U(w + eps e_j) - U(w - eps e_j)) / (2 eps)``,
        so the surrogate is a deterministic function of ``w`` -- which is what
        keeps the Metropolis--Hastings correction in :mod:`nds.sampler` exact.
        Cost is ``2 d`` likelihood evaluations per walker; the shift only enters
        through the linear predictor, ``X (w + eps e_j) = Z + eps X_j``, so no
        matrix product is repeated.  The quadratic prior term is differenced in
        closed form, which is exactly what central differences return for it.
        """
        Z = self.linear(W) if Z is None else Z
        out = np.empty((self.d, W.shape[1]))
        inv = 0.5 / eps
        for j in range(self.d):
            col = self.X[:, j : j + 1]
            shift = eps * col
            plus = softplus(Z + shift).sum(axis=0)
            minus = softplus(Z - shift).sum(axis=0)
            out[j] = (plus - minus) * inv - self._Xty[j]
        out += self.prior_precision[:, None] * W
        return out

    # ------------------------------------------------------------ prediction
    def predict_proba(self, W: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Per-walker predictive probabilities, shape (n_eval, m)."""
        return _sigmoid(X @ W)


try:  # scipy's expit is a fast, stable C implementation
    from scipy.special import expit as _sigmoid
except ImportError:  # pragma: no cover - fallback keeps scipy optional

    def _sigmoid(Z: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.tanh(0.5 * Z))
