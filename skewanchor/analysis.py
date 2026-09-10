r"""Exact second-moment analysis of the discretised skew-anchored dynamics.

For a log-quadratic target with the canonical anchor (``s = 1``) the anchored
drift is *linear* and the diffusion coefficient is the scalar ``q(x)^{1/2}``:

    x_{k+1} = (I - eta B) x_k + sqrt(2 eta) q(x_k)^{1/2} xi_{k+1},
    B := (2 beta / nu) (I - J) Sigma^{-1},   q(x) = 1 + x^T Sigma^{-1} x / nu.

Because ``q`` is exactly quadratic and the noise is isotropic given ``x``, the
second-moment matrix ``C_k = E[x_k x_k^T]`` follows an **exact affine
recursion** (no closure approximation):

    C_{k+1} = M C_k M^T + 2 eta (1 + Tr(Sigma^{-1} C_k) / nu) I,   M = I - eta B.

Everything about the discretisation's second-order behaviour then follows in
closed form:

* the per-iteration convergence factor is the spectral radius ``rho(L)`` of the
  linear part ``L(C) = M C M^T + (2 eta / nu) Tr(Sigma^{-1} C) I``;
* the scheme is mean-square stable iff ``rho(L) < 1``, which gives an exact
  maximum stepsize -- much smaller than the drift-only (deterministic) limit,
  because the multiplicative noise ``q^{1/2}`` grows linearly in ``||x||``;
* the stationary covariance of the *chain* is the fixed point of the affine
  map, so the discretisation bias is exact rather than an ``O(eta)`` bound.

The continuous-time SDE has stationary covariance ``nu/(nu-2) Sigma`` for every
skew ``J`` (the ``J`` terms cancel in ``B C + C B^T``), which is what makes an
equal-bias comparison between different ``J`` meaningful.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "drift_matrix",
    "ou_gap",
    "ou_gap_ceiling",
    "second_moment_operator",
    "ms_factor",
    "ms_rate",
    "is_ms_stable",
    "max_stable_stepsize",
    "stationary_covariance",
    "covariance_bias",
    "eta_for_bias",
    "deterministic_stepsize_limit",
    "summary",
]


def drift_matrix(target, J=None):
    """``B = (2 beta / nu)(I - J) Sigma^{-1}``, the anchored drift matrix (s = 1)."""
    d = target.d
    J = np.zeros((d, d)) if J is None else np.asarray(J, dtype=np.float64)
    return (2.0 * target.beta / target.nu) * (np.eye(d) - J) @ target.Sigma_inv


def ou_gap(target, J=None):
    """Continuous-time rate ``min_i Re lambda_i(B)``."""
    return float(np.min(np.linalg.eigvals(drift_matrix(target, J)).real))


def ou_gap_ceiling(target):
    """``Tr(B)/d``: the best rate any skew ``J`` can reach, since ``Tr(JA) = 0``."""
    B = drift_matrix(target, None)
    return float(np.trace(B) / target.d)


# ------------------------------------------------- second-moment operator


def _basis(d):
    """Orthonormal basis of the symmetric matrices, as a (d*(d+1)/2, d, d) array."""
    mats = []
    for i in range(d):
        for j in range(i, d):
            E = np.zeros((d, d))
            if i == j:
                E[i, i] = 1.0
            else:
                E[i, j] = E[j, i] = 1.0 / np.sqrt(2.0)
            mats.append(E)
    return np.array(mats)


def second_moment_operator(target, J, eta):
    """Matrix of ``L(C) = M C M^T + (2 eta/nu) Tr(Sigma^{-1} C) I`` on symmetric ``C``."""
    d = target.d
    M = np.eye(d) - eta * drift_matrix(target, J)
    A = target.Sigma_inv
    basis = _basis(d)
    k = basis.shape[0]
    out = np.zeros((k, k))
    for c, E in enumerate(basis):
        img = M @ E @ M.T + (2.0 * eta / target.nu) * np.trace(A @ E) * np.eye(d)
        out[:, c] = np.einsum("kij,ij->k", basis, img)
    return out


def ms_factor(target, J, eta):
    """Per-iteration second-moment contraction factor ``rho(L)``."""
    return float(np.max(np.abs(np.linalg.eigvals(second_moment_operator(target, J, eta)))))


def ms_rate(target, J, eta):
    """Per-iteration convergence rate ``-log rho(L)``.  Larger is faster.

    This is the quantity to compare across methods at equal cost: the number of
    iterations to reduce the second-moment error by a factor ``e`` is ``1/rate``.
    """
    rho = ms_factor(target, J, eta)
    return float(-np.log(rho)) if 0 < rho < 1 else float("-inf")


def is_ms_stable(target, J, eta):
    return ms_factor(target, J, eta) < 1.0


def max_stable_stepsize(target, J=None, hi=10.0, tol=1e-12):
    """Largest ``eta`` with ``rho(L) < 1``, by bisection.

    Returns 0.0 when no positive stepsize is stable, which happens when the
    continuous-time dynamics itself fails the ``m > alpha`` balance.
    """
    if not is_ms_stable(target, J, 1e-12):
        return 0.0
    lo = 1e-12
    while is_ms_stable(target, J, hi):
        hi *= 2.0
        if hi > 1e6:
            return hi
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if is_ms_stable(target, J, mid):
            lo = mid
        else:
            hi = mid
    return lo


def deterministic_stepsize_limit(target, J=None):
    """Drift-only Euler limit ``min_i 2 Re(lambda_i)/|lambda_i|^2``.

    Reported alongside :func:`max_stable_stepsize` to show how much stricter the
    multiplicative noise makes the real constraint.
    """
    lam = np.linalg.eigvals(drift_matrix(target, J))
    return float(np.min(2.0 * lam.real / np.abs(lam) ** 2))


def stationary_covariance(target, J, eta):
    """Fixed point ``C_inf`` of the affine recursion, or ``None`` if unstable."""
    d = target.d
    if not is_ms_stable(target, J, eta):
        return None
    basis = _basis(d)
    Lmat = second_moment_operator(target, J, eta)
    rhs = np.einsum("kij,ij->k", basis, 2.0 * eta * np.eye(d))
    coef = np.linalg.solve(np.eye(Lmat.shape[0]) - Lmat, rhs)
    return np.einsum("k,kij->ij", coef, basis)


def covariance_bias(target, J, eta, relative=True):
    """Relative Frobenius distance between ``C_inf`` and ``nu/(nu-2) Sigma``."""
    C = stationary_covariance(target, J, eta)
    if C is None:
        return float("inf")
    truth = target.cov()
    err = np.linalg.norm(C - truth)
    return float(err / np.linalg.norm(truth)) if relative else float(err)


def eta_for_bias(target, J, bias, tol=1e-10):
    """Largest ``eta`` whose stationary covariance bias equals ``bias``.

    The equal-bias stepsize is the basis of the fair comparison: methods are run
    at the stepsize that puts them all at the same asymptotic accuracy, and only
    then are their convergence rates compared.
    """
    hi = max_stable_stepsize(target, J)
    if hi <= 0:
        return 0.0
    if covariance_bias(target, J, hi * (1 - 1e-9)) < bias:
        return hi
    lo = 1e-14
    hi = hi * (1 - 1e-9)
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if covariance_bias(target, J, mid) < bias:
            lo = mid
        else:
            hi = mid
    return lo


def summary(target, J=None, eta=None, bias=None):
    """All of the above for one ``(target, J)`` pair, as a dict."""
    out = {
        "gap": ou_gap(target, J),
        "gap_ceiling": ou_gap_ceiling(target),
        "eta_max_ms": max_stable_stepsize(target, J),
        "eta_max_drift": deterministic_stepsize_limit(target, J),
        "J_norm": 0.0 if J is None else float(np.linalg.norm(J, 2)),
    }
    if eta is not None:
        out.update(
            eta=eta,
            ms_factor=ms_factor(target, J, eta),
            ms_rate=ms_rate(target, J, eta),
            cov_bias=covariance_bias(target, J, eta),
        )
    if bias is not None:
        e = eta_for_bias(target, J, bias)
        out.update(
            eta_equal_bias=e,
            ms_rate_equal_bias=ms_rate(target, J, e) if e > 0 else float("-inf"),
        )
    return out
