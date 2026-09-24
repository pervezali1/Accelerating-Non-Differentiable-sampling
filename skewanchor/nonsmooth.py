r"""A target that is heavy tailed *and* non-differentiable at the same time.

The paper handles the two difficulties in separate experiments: non-smooth
potentials in Sections 6.1-6.3, heavy tails in Section 6.4.  Combining them is
the case the anchored framework is built for, so it is the natural place to ask
whether the skew perturbation still accelerates.

    q(x)  = 1 + (1/nu) x^T Sigma^{-1} x
    U(x)  = iota log q(x) + sum_i p_lambda(x_i)
    U0(x) = beta log q(x) + sum_i p^eps_lambda(x_i)

``p_lambda`` is the minimax concave penalty (MCP) of Zhang (2010), the paper's
own Section 6.2 regulariser, and ``p^eps_lambda`` is exactly the paper's smoothed
version (its Eq. 67):

    p_lambda(t)     = lambda|t| - t^2/(2a)                   if |t| <= a lambda
                    = a lambda^2 / 2                          otherwise
    p^eps_lambda(t) = lambda sqrt(t^2+eps^2)
                        - lambda t^2 / (2 sqrt(a^2 lambda^2 + eps^2))   if |t| <= a lambda
                    = lambda (a^2 lambda^2 + 2 eps^2)
                        / (2 sqrt(a^2 lambda^2 + eps^2))                otherwise

Two properties make this a good test object.  ``p_lambda`` is non-differentiable
at the origin, so ``grad U`` does not exist there and plain ULA is ill defined;
and ``p_lambda`` is *bounded*, so ``e^{-U}`` keeps the polynomial tail of the
Student-t.  That also gives an exact i.i.d. sampler: rejection from the
Student-t with acceptance probability ``exp(-sum_i p_lambda(x_i))``.
"""

from __future__ import annotations

import numpy as np

from .targets import LogQuadraticTarget

__all__ = ["mcp", "mcp_smoothed", "mcp_smoothed_grad", "HeavyTailedMCP",
           "smoothed_curvature", "gaussianised"]


def mcp(t, lam, a):
    t = np.abs(np.asarray(t, dtype=np.float64))
    inner = lam * t - t * t / (2.0 * a)
    return np.where(t <= a * lam, inner, 0.5 * a * lam * lam)


def mcp_smoothed(t, lam, a, eps):
    t = np.asarray(t, dtype=np.float64)
    r = np.sqrt(a * a * lam * lam + eps * eps)
    inner = lam * np.sqrt(t * t + eps * eps) - lam * t * t / (2.0 * r)
    outer = lam * (a * a * lam * lam + 2.0 * eps * eps) / (2.0 * r)
    return np.where(np.abs(t) <= a * lam, inner, outer)


def mcp_smoothed_grad(t, lam, a, eps):
    t = np.asarray(t, dtype=np.float64)
    r = np.sqrt(a * a * lam * lam + eps * eps)
    inner = lam * t / np.sqrt(t * t + eps * eps) - lam * t / r
    return np.where(np.abs(t) <= a * lam, inner, 0.0)


def smoothed_curvature(lam, a, eps):
    r"""``p^eps_lambda''(0)``: the curvature the smoothed penalty adds at the origin.

        p^eps_lambda''(0) = lambda / eps - lambda / sqrt(a^2 lambda^2 + eps^2)

    The penalty is separable and even, so this is the same number in every
    coordinate: the regulariser adds an **isotropic** term to the anchor's
    Hessian.  That is the whole story of why a skew perturbation does less on
    the regularised target than on the Student-t core.  A skew field buys speed
    by moving mass between directions of *different* stiffness, so what it can
    buy is governed by the anisotropy; adding ``lambda/eps`` equally to all
    directions makes the target rounder and takes that away.

    With the paper's ``a = 2``, ``eps = 0.1`` and the ``kappa = 100``,
    ``nu = 6``, ``d = 3`` core, the effective condition number falls

        lambda        0     0.25    0.5    0.75      1
        cond      100.0     37.4   21.4    15.1   11.8

    and the best speedup any skew field can reach at equal discretisation bias
    falls with it, from 4.37x at ``lambda = 0`` to 1.16x at ``lambda = 1``.
    Note ``eps`` enters as ``lambda/eps``, so it is the stronger of the two
    knobs: at ``lambda = 0.5`` the condition number is 11.8 for ``eps = 0.05``
    but 89.0 for ``eps = 0.5``.
    """
    lam, a, eps = float(lam), float(a), float(eps)
    return lam / eps - lam / np.sqrt(a * a * lam * lam + eps * eps)


def gaussianised(target):
    """Log-quadratic stand-in with the same anchor Hessian at the origin.

    The exact second-moment machinery of :mod:`skewanchor.analysis` needs the
    anchored drift to be linear, which the MCP target's is not.  Replacing the
    penalty by its curvature at the origin restores a log-quadratic target,

        Sigma_eff^{-1} = Sigma^{-1} + (nu / 2 beta) p^eps_lambda''(0) I,

    on which every exact quantity -- stable stepsize, discretisation bias,
    convergence rate, equal-bias speedup -- is available in closed form.  It is
    an approximation of the target, not of the dynamics: it is accurate where
    the chain spends its time, because ``|x_i| <= a lambda`` holds for the bulk
    of the mass in every coordinate, and outside that band the penalty's
    gradient is exactly zero anyway.

    Measured against the ``J = 0`` chain on the ``lambda = 1`` MCP target, with
    the stepsize written as a multiple of the ``bias = 0.02`` value:

        eta/eta0             1      2      4      8
        predicted rate    1.00   1.98   3.87   6.65
        measured speed    1.00   2.13   3.11      -
        predicted bias    0.032  0.069  0.167  0.654

    The last column is the useful one: the stand-in says the bias at ``8 eta0``
    is two thirds of the covariance, and the chain indeed never reaches the
    target accuracy there, so the stand-in picks the usable stepsize correctly.
    """
    from .targets import LogQuadraticTarget

    kappa = smoothed_curvature(target.lam, target.a, target.eps)
    Sigma_inv = target.Sigma_inv + (target.nu / (2.0 * target.beta)) * kappa * np.eye(target.d)
    return LogQuadraticTarget(nu=target.nu, mu=target.mu,
                              Sigma=np.linalg.inv(Sigma_inv), beta=target.beta)


class HeavyTailedMCP:
    """Heavy-tailed Student-t core with a non-smooth bounded MCP penalty."""

    def __init__(self, nu, Sigma, beta=None, lam=1.0, a=2.0, eps=0.1):
        self.base = LogQuadraticTarget(nu=nu, mu=np.zeros(np.atleast_2d(Sigma).shape[0]),
                                       Sigma=Sigma, beta=beta)
        self.d = self.base.d
        self.nu = self.base.nu
        self.iota = self.base.iota
        self.beta = self.base.beta
        self.s = self.base.s
        self.Sigma = self.base.Sigma
        self.Sigma_inv = self.base.Sigma_inv
        self.Sigma_inv_half = self.base.Sigma_inv_half
        self.Sigma_half = self.base.Sigma_half
        self.Sigma_evals = self.base.Sigma_evals
        self.Sigma_evecs = self.base.Sigma_evecs
        self.mu = self.base.mu
        self.cond = self.base.cond
        self.lam = float(lam)
        self.a = float(a)
        self.eps = float(eps)
        self._cov = None

    def __repr__(self):  # pragma: no cover - cosmetic
        return (f"HeavyTailedMCP(d={self.d}, nu={self.nu:g}, beta={self.beta:g}, "
                f"lam={self.lam:g}, a={self.a:g}, eps={self.eps:g}, "
                f"cond(Sigma)={self.cond:.3g})")

    # ------------------------------------------------------------ potentials

    def penalty(self, x):
        return np.sum(mcp(x, self.lam, self.a), axis=-1)

    def penalty_smoothed(self, x):
        return np.sum(mcp_smoothed(x, self.lam, self.a, self.eps), axis=-1)

    def q(self, x):
        return self.base.q(x)

    def U(self, x):
        return self.base.U(x) + self.penalty(np.atleast_2d(x))

    def U0(self, x):
        return self.base.U0(x) + self.penalty_smoothed(np.atleast_2d(x))

    def grad_U(self, x):
        """A.e. gradient of ``U``; the MCP part is undefined at ``x_i = 0``.

        ``np.sign(0) == 0`` picks the zero element of the subdifferential there,
        which is what a subgradient ULA would use.
        """
        x2 = np.atleast_2d(x)
        t = np.abs(x2)
        sub = np.where(t <= self.a * self.lam,
                       np.sign(x2) * self.lam - x2 / self.a, 0.0)
        return self.base.grad_U(x2) + sub

    def grad_U0(self, x):
        x2 = np.atleast_2d(x)
        return self.base.grad_U0(x2) + mcp_smoothed_grad(x2, self.lam, self.a, self.eps)

    def anchor_scale(self, x):
        """``e^{(U-U0)(x)}``: the Student-t factor ``q^s`` times a bounded factor."""
        x2 = np.atleast_2d(x)
        return self.base.anchor_scale(x2) * np.exp(
            self.penalty(x2) - self.penalty_smoothed(x2)
        )

    def sigma(self, x):
        return np.sqrt(self.anchor_scale(x))

    def skew_anchored_drift(self, x, J=None):
        x2 = np.atleast_2d(x)
        d = self.d
        J = np.zeros((d, d)) if J is None else np.asarray(J, dtype=np.float64)
        g = self.grad_U0(x2)
        return self.anchor_scale(x2)[:, None] * (g @ (J - np.eye(d)).T)

    # -------------------------------------------------------------- sampling

    def sample(self, n, rng, max_rounds=200):
        """Exact i.i.d. draws by rejection from the Student-t core.

        ``exp(-penalty) <= 1`` bounds the ratio, so acceptance is exact.
        """
        out = []
        got = 0
        for _ in range(max_rounds):
            m = max(int(1.6 * (n - got)), 1024)
            prop = self.base.sample(m, rng)
            keep = rng.random(m) < np.exp(-self.penalty(prop))
            acc = prop[keep]
            out.append(acc)
            got += acc.shape[0]
            if got >= n:
                break
        else:
            raise RuntimeError("rejection sampler did not reach n draws")
        return np.concatenate(out, axis=0)[:n]

    def acceptance_rate(self, rng, n=200_000):
        prop = self.base.sample(n, rng)
        return float(np.mean(np.exp(-self.penalty(prop))))

    # --------------------------------------------------------------- moments

    def cov(self, rng=None, n=2_000_000):
        """Covariance, estimated once from a large exact sample and cached."""
        if self._cov is None:
            rng = np.random.default_rng(20240917) if rng is None else rng
            x = self.sample(n, rng)
            self._cov = np.cov(x, rowvar=False).reshape(self.d, self.d)
        return self._cov

    def mean(self):
        return np.zeros(self.d)


def make(d=2, nu=5.0, kappa=100.0, lam=1.0, a=2.0, eps=0.1, beta=None):
    from .targets import anisotropic_student_t
    base = anisotropic_student_t(d, nu, kappa, beta=beta)
    return HeavyTailedMCP(nu=nu, Sigma=base.Sigma, beta=beta, lam=lam, a=a, eps=eps)
