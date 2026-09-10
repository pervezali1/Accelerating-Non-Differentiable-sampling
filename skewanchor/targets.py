r"""Heavy-tailed target distributions and their anchoring (reference) potentials.

The whole experiment lives inside one parametric family, the *log-quadratic*
family, which contains both heavy-tailed targets used in the paper.

    q(x)  = 1 + (1 / nu) (x - mu)^T Sigma^{-1} (x - mu)
    U(x)  = iota * log q(x),            iota = (d + nu) / 2
    U0(x) = beta * log q(x),            beta free, default beta = iota - 1

so that

    pi(x) \propto exp(-U(x)) = q(x)^{-(d+nu)/2},

which is exactly the d-dimensional Student-t distribution with ``nu`` degrees of
freedom, location ``mu`` and scale matrix ``Sigma`` (paper, Example 1 and 2).

Two named instances:

* ``isotropic_polynomial(d, iota)`` -- the target of Section 6.4 of the paper,
  pi(x) \propto (1 + ||x||^2)^{-iota}.  Setting nu = 2 iota - d and
  Sigma = I / nu makes q(x) = 1 + ||x||^2 exactly.
* ``anisotropic_student_t(d, nu, kappa, ...)`` -- an ill-conditioned Student-t,
  the regime in which a skew-symmetric perturbation can actually accelerate.

Anchored quantities (paper, Eq. 6-7 and Example 2), with s := iota - beta:

    e^{U - U0}(x) = q(x)^s
    grad U0(x)    = (2 beta / nu) Sigma^{-1} (x - mu) / q(x)
    b(x)          = -grad U0(x) e^{U-U0}(x) = -(2 beta / nu) q(x)^{s-1} Sigma^{-1}(x - mu)
    sigma(x)      = q(x)^{s/2}

For the canonical choice beta = iota - 1 we get s = 1, so ``b`` is *linear* and
only the (scalar) diffusion coefficient is state dependent.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "LogQuadraticTarget",
    "isotropic_polynomial",
    "anisotropic_student_t",
]


def _as_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x[None, :], True
    return x, False


class LogQuadraticTarget:
    """Multivariate Student-t target together with its log-quadratic anchor.

    Parameters
    ----------
    nu : float
        Degrees of freedom.  Finite mean requires ``nu > 1``, finite variance
        ``nu > 2``.
    mu : array_like, shape (d,)
        Location.
    Sigma : array_like, shape (d, d)
        Positive definite scale matrix.
    beta : float, optional
        Anchoring exponent, ``U0 = beta log q``.  Defaults to ``iota - 1``,
        which is the choice of Example 2 in the paper and makes the anchored
        drift linear.  Must satisfy ``beta > d / 2`` for Theorem 3 to apply.
    """

    def __init__(self, nu, mu, Sigma, beta=None):
        self.Sigma = np.atleast_2d(np.asarray(Sigma, dtype=np.float64))
        self.d = self.Sigma.shape[0]
        if self.Sigma.shape != (self.d, self.d):
            raise ValueError("Sigma must be square")
        self.mu = np.asarray(mu, dtype=np.float64).reshape(self.d)
        self.nu = float(nu)
        if self.nu <= 0:
            raise ValueError("nu must be positive")

        evals, evecs = np.linalg.eigh(self.Sigma)
        if evals.min() <= 0:
            raise ValueError("Sigma must be positive definite")
        self.Sigma_evals = evals
        self.Sigma_evecs = evecs
        self.Sigma_inv = (evecs * (1.0 / evals)) @ evecs.T
        self.Sigma_half = (evecs * np.sqrt(evals)) @ evecs.T
        self.Sigma_inv_half = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T
        self.cond = float(evals.max() / evals.min())

        self.iota = 0.5 * (self.d + self.nu)
        self.beta = float(self.iota - 1.0) if beta is None else float(beta)
        self.s = self.iota - self.beta  # exponent of q in e^{U - U0}

    # ---------------------------------------------------------------- basics

    def __repr__(self):  # pragma: no cover - cosmetic
        return (
            f"LogQuadraticTarget(d={self.d}, nu={self.nu:g}, iota={self.iota:g}, "
            f"beta={self.beta:g}, s={self.s:g}, cond(Sigma)={self.cond:.3g})"
        )

    def q(self, x):
        """q(x) = 1 + (1/nu) (x-mu)^T Sigma^{-1} (x-mu).  Shape (n,) or ()."""
        x2, squeeze = _as_2d(x)
        z = x2 - self.mu
        val = 1.0 + np.einsum("ni,ij,nj->n", z, self.Sigma_inv, z) / self.nu
        return val[0] if squeeze else val

    def U(self, x):
        return self.iota * np.log(self.q(x))

    def U0(self, x):
        return self.beta * np.log(self.q(x))

    def grad_U(self, x):
        """grad U = (2 iota / nu) Sigma^{-1}(x-mu) / q(x).  Bounded in ||x||."""
        x2, squeeze = _as_2d(x)
        z = x2 - self.mu
        g = (2.0 * self.iota / self.nu) * (z @ self.Sigma_inv) / self.q(x2)[:, None]
        return g[0] if squeeze else g

    def grad_U0(self, x):
        x2, squeeze = _as_2d(x)
        z = x2 - self.mu
        g = (2.0 * self.beta / self.nu) * (z @ self.Sigma_inv) / self.q(x2)[:, None]
        return g[0] if squeeze else g

    # ------------------------------------------------- anchored coefficients

    def anchor_scale(self, x):
        """e^{(U - U0)(x)} = q(x)^s, the multiplicative acceleration factor."""
        return self.q(x) ** self.s

    def sigma(self, x):
        """Scalar diffusion coefficient sigma(x) = e^{(U-U0)(x)/2} = q(x)^{s/2}."""
        return self.q(x) ** (0.5 * self.s)

    def anchored_drift(self, x):
        """b(x) = -grad U0(x) e^{(U-U0)(x)} = -(2 beta/nu) q^{s-1} Sigma^{-1}(x-mu)."""
        x2, squeeze = _as_2d(x)
        z = x2 - self.mu
        fac = (2.0 * self.beta / self.nu) * self.q(x2) ** (self.s - 1.0)
        b = -fac[:, None] * (z @ self.Sigma_inv)
        return b[0] if squeeze else b

    def skew_anchored_drift(self, x, J):
        """b_J(x) = e^{(U-U0)(x)} (J - I) grad U0(x).

        Reduces to :meth:`anchored_drift` when ``J`` is zero.
        """
        x2, squeeze = _as_2d(x)
        z = x2 - self.mu
        fac = (2.0 * self.beta / self.nu) * self.q(x2) ** (self.s - 1.0)
        # (J - I) Sigma^{-1} z, computed row-wise for a batch of points
        M = (J - np.eye(self.d)) @ self.Sigma_inv
        b = fac[:, None] * (z @ M.T)
        return b[0] if squeeze else b

    # ------------------------------------------------------------- sampling

    def sample(self, n, rng):
        """Exact i.i.d. draws via the Gaussian scale-mixture representation."""
        z = rng.standard_normal((n, self.d))
        g = rng.chisquare(self.nu, size=n)
        return self.mu + (z @ self.Sigma_half) * np.sqrt(self.nu / g)[:, None]

    # -------------------------------------------------- exact 1-d quantiles

    def projected_scale(self, theta):
        """Scale of the univariate Student-t law of <theta, X>."""
        theta = np.asarray(theta, dtype=np.float64)
        return float(np.sqrt(theta @ self.Sigma @ theta))

    def projected_ppf(self, u, theta):
        """Exact quantile function of <theta, X>, a univariate t_nu."""
        theta = np.asarray(theta, dtype=np.float64)
        loc = float(theta @ self.mu)
        scale = self.projected_scale(theta)
        return stats.t.ppf(u, df=self.nu, loc=loc, scale=scale)

    def projected_isf(self, w, theta):
        """Inverse survival function of <theta, X>: ``ppf(1 - w)`` computed stably."""
        theta = np.asarray(theta, dtype=np.float64)
        loc = float(theta @ self.mu)
        scale = self.projected_scale(theta)
        return stats.t.isf(w, df=self.nu, loc=loc, scale=scale)

    def marginal_ppf(self, u, i):
        e = np.zeros(self.d)
        e[i] = 1.0
        return self.projected_ppf(u, e)

    # ------------------------------------------------------------- moments

    def mean(self):
        if self.nu <= 1:
            return np.full(self.d, np.nan)
        return self.mu.copy()

    def cov(self):
        if self.nu <= 2:
            return np.full((self.d, self.d), np.nan)
        return (self.nu / (self.nu - 2.0)) * self.Sigma

    # ------------------------------------------------- theory diagnostics

    def theory(self):
        """Constants from the paper's Assumption 12 / Example 2, plus the
        anchored-drift matrix used by the linear (s = 1) analysis."""
        lam_min = 1.0 / self.Sigma_evals.max()   # lambda_min(Sigma^{-1})
        lam_max = 1.0 / self.Sigma_evals.min()   # lambda_max(Sigma^{-1})
        return {
            "lambda_min_Sigma_inv": lam_min,
            "lambda_max_Sigma_inv": lam_max,
            "kappa": self.cond,
            # Example 2 of the paper: c0 = 2 sqrt(lam_max/nu), c1 = 2 lam_max/nu,
            # c2 = 2 lam_min/nu; Corollary 13 needs beta > d c0^2 / (4 c2).
            "corollary13_beta_threshold": 0.5 * self.d * self.cond,
            "corollary13_satisfied": self.beta > 0.5 * self.d * self.cond,
            # Assumption 12 constants for the s = 1 (linear drift) case.
            "m": (2.0 * self.beta / self.nu) * lam_min,
            "L": (2.0 * self.beta / self.nu) * lam_max,
            "alpha": self.d * lam_max / self.nu,
            "theorem3_beta_threshold": 0.5 * self.d,
            "theorem3_satisfied": self.beta > 0.5 * self.d,
            "finite_mean": self.nu > 1,
            "finite_variance": self.nu > 2,
        }


# --------------------------------------------------------------- factories


def isotropic_polynomial(d, iota, beta=None):
    r"""The target of Section 6.4: pi(x) \propto (1 + ||x||^2)^{-iota}.

    Integrability needs ``iota > d/2``; the paper additionally asks
    ``iota > 1 + d/2`` so that the mean exists.  Written as a Student-t this is
    ``nu = 2 iota - d`` degrees of freedom with ``Sigma = I / nu``.
    """
    nu = 2.0 * iota - d
    if nu <= 0:
        raise ValueError(f"iota={iota} too small for d={d}: need iota > d/2")
    Sigma = np.eye(d) / nu
    return LogQuadraticTarget(nu=nu, mu=np.zeros(d), Sigma=Sigma, beta=beta)


def anisotropic_student_t(d, nu, kappa, beta=None, spectrum="geometric", rotate=False, seed=0):
    """Ill-conditioned Student-t with ``cond(Sigma) = kappa``.

    ``spectrum='geometric'`` spreads the eigenvalues log-uniformly in
    ``[1/kappa, 1]``; ``spectrum='two_point'`` uses only the two extremes,
    which is the classical hard case for reversible samplers.
    ``rotate=True`` applies a fixed random orthogonal change of basis so that
    ``Sigma`` is not diagonal (guards against axis-aligned artefacts).
    """
    if spectrum == "geometric":
        evals = np.geomspace(1.0 / kappa, 1.0, d)
    elif spectrum == "two_point":
        evals = np.concatenate(
            [np.full(d // 2, 1.0 / kappa), np.full(d - d // 2, 1.0)]
        )
    else:
        raise ValueError(f"unknown spectrum {spectrum!r}")
    Sigma = np.diag(evals)
    if rotate:
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        Sigma = Q @ Sigma @ Q.T
        Sigma = 0.5 * (Sigma + Sigma.T)
    return LogQuadraticTarget(nu=nu, mu=np.zeros(d), Sigma=Sigma, beta=beta)
