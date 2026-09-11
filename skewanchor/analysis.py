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
    "rho_J",
    "continuous_ms_rate",
    "continuous_second_moment_operator",
    "assumption12",
    "optimize_J",
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


def rho_J(target, J=None):
    r"""``rho_J = lambda_max(Sigma^{1/2} S Sigma^{1/2})`` with ``S = sym(J Sigma^{-1})``.

    One number controls two separate things for the anisotropic Student-t.

    * The Euclidean one-sided Lipschitz constant of the anchored drift is
      ``m_J = (2 beta / nu) lambda_min(Sigma^{-1} - S)``, so the paper's
      Assumption 12(18) holds iff ``rho_J < 1``.
    * The Euclidean Lyapunov drift condition (Assumption 1 of the paper) picks
      up the same term.

    ``S = (1/2)[J, Sigma^{-1}]``, so ``rho_J = 0`` exactly when ``J`` commutes
    with ``Sigma`` -- and a commuting ``J`` is also the one that cannot
    accelerate.  Acceleration and the Euclidean contraction constant are
    therefore in direct tension.

    The tension is an artefact of measuring in the Euclidean metric: for any
    Lyapunov function of the form ``V = Psi(U_0)`` the skew drift contributes
    exactly zero, because ``<J grad U0, grad V> = Psi'(U0) <J grad U0, grad U0>
    = 0``.  Geometric ergodicity of the continuous-time dynamics therefore holds
    for every skew ``J``, with no smallness condition; it is only the
    synchronous-coupling W2 constants that degrade.
    """
    d = target.d
    if J is None:
        return 0.0
    J = np.asarray(J, dtype=np.float64)
    A = target.Sigma_inv
    S = 0.5 * (J @ A + (J @ A).T)
    H = target.Sigma_half @ S @ target.Sigma_half
    return float(np.max(np.linalg.eigvalsh(0.5 * (H + H.T))))


def assumption12(target, J=None):
    """The paper's Assumption 12 constants ``(m, L, alpha)`` for the skew drift."""
    d = target.d
    A = target.Sigma_inv
    c = 2.0 * target.beta / target.nu
    Jm = np.zeros((d, d)) if J is None else np.asarray(J, dtype=np.float64)
    S = 0.5 * (Jm @ A + (Jm @ A).T)
    m = c * float(np.min(np.linalg.eigvalsh(A - S)))
    L = c * float(np.linalg.norm((Jm - np.eye(d)) @ A, 2))
    alpha = target.d * float(np.max(np.linalg.eigvalsh(A))) / target.nu
    return {"m": m, "L": L, "alpha": alpha, "rho_J": rho_J(target, J),
            "assumption12_holds": bool(0.0 < alpha < m)}


def propagator(target, J, eta, integrator="euler"):
    r"""One-step linear map ``M`` applied to the state by the chosen integrator.

    ``euler``   ``I - eta B`` -- the explicit scheme of the paper.  A rotation is
                amplified by ``sqrt(1 + theta^2)`` each step, which is where the
                ``J``-dependent bias comes from.
    ``cayley``  ``(I + eta B/2)^{-1}(I - eta B/2)`` -- unit modulus on the skew
                part, so the rotation is no longer amplified.
    ``expm``    ``exp(-eta B)`` -- the linear drift advanced exactly.
    """
    from scipy.linalg import expm as _expm

    d = target.d
    B = drift_matrix(target, J)
    I = np.eye(d)
    if integrator == "euler":
        return I - eta * B
    if integrator == "cayley":
        return np.linalg.solve(I + 0.5 * eta * B, I - 0.5 * eta * B)
    if integrator == "expm":
        return _expm(-eta * B)
    raise ValueError(f"unknown integrator {integrator!r}")


def second_moment_operator(target, J, eta, integrator="euler"):
    r"""Matrix of ``L(C) = M C M^T + (2 eta/nu) Tr(Sigma^{-1} C) I``.

    Vectorised, ``L = M (x) M + (2 eta / nu) vec(I) vec(Sigma^{-1})^T``, a
    ``d^2 x d^2`` matrix.  The operator maps symmetric matrices to symmetric
    matrices and the rank-one term annihilates the antisymmetric part, so the
    spectral radius over the whole space equals the one over the symmetric part:
    the antisymmetric block is a sub-block of ``M (x) M``, whose eigenvalues
    ``lambda_i lambda_j`` are dominated by the ``lambda_i^2`` already present in
    the symmetric block.
    """
    d = target.d
    M = propagator(target, J, eta, integrator)
    A = target.Sigma_inv
    return np.kron(M, M) + (2.0 * eta / target.nu) * np.outer(
        np.eye(d).ravel(), A.ravel()
    )


def continuous_second_moment_operator(target, J=None):
    r"""Generator of ``d/dt C = -B C - C B^T + 2(1 + Tr(Sigma^{-1} C)/nu) I``.

    The continuous-time counterpart of :func:`second_moment_operator`.  Its
    spectral abscissa gives the exact exponential rate at which the second
    moment of the *SDE* (no discretisation) reaches equilibrium, so comparing
    it across ``J`` isolates the acceleration of the dynamics from the smaller
    stepsize the skew drift forces on the scheme.
    """
    d = target.d
    B = drift_matrix(target, J)
    I = np.eye(d)
    return -(np.kron(I, B) + np.kron(B, I)) + (2.0 / target.nu) * np.outer(
        I.ravel(), target.Sigma_inv.ravel()
    )


def continuous_ms_rate(target, J=None):
    """Exponential rate of second-moment convergence for the SDE itself.

    Positive iff the second moment converges; this is the honest statement of
    ergodicity in the regime where the paper's Assumption 12 fails.
    """
    ev = np.linalg.eigvals(continuous_second_moment_operator(target, J))
    return float(-np.max(ev.real))


def ms_factor(target, J, eta, integrator="euler"):
    """Per-iteration second-moment contraction factor ``rho(L)``."""
    return float(np.max(np.abs(np.linalg.eigvals(
        second_moment_operator(target, J, eta, integrator)))))


def ms_rate(target, J, eta, integrator="euler"):
    """Per-iteration convergence rate ``-log rho(L)``.  Larger is faster.

    This is the quantity to compare across methods at equal cost: the number of
    iterations to reduce the second-moment error by a factor ``e`` is ``1/rate``.
    """
    rho = ms_factor(target, J, eta, integrator)
    return float(-np.log(rho)) if 0 < rho < 1 else float("-inf")


def is_ms_stable(target, J, eta, integrator="euler"):
    return ms_factor(target, J, eta, integrator) < 1.0


def max_stable_stepsize(target, J=None, hi=10.0, tol=1e-8, integrator="euler"):
    """Largest ``eta`` with ``rho(L) < 1``, by bisection.

    Returns 0.0 when no positive stepsize is stable, which happens when the
    continuous-time dynamics itself fails the ``m > alpha`` balance.
    """
    if not is_ms_stable(target, J, 1e-12, integrator):
        return 0.0
    lo = 1e-12
    while is_ms_stable(target, J, hi, integrator):
        hi *= 2.0
        if hi > 1e6:
            return hi
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if is_ms_stable(target, J, mid, integrator):
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


def stationary_covariance(target, J, eta, integrator="euler"):
    """Fixed point ``C_inf`` of the affine recursion, or ``None`` if unstable."""
    d = target.d
    if not is_ms_stable(target, J, eta, integrator):
        return None
    K = second_moment_operator(target, J, eta, integrator)
    rhs = (2.0 * eta * np.eye(d)).ravel()
    C = np.linalg.solve(np.eye(d * d) - K, rhs).reshape(d, d)
    return 0.5 * (C + C.T)


def covariance_bias(target, J, eta, relative=True, integrator="euler"):
    """Relative Frobenius distance between ``C_inf`` and ``nu/(nu-2) Sigma``."""
    C = stationary_covariance(target, J, eta, integrator)
    if C is None:
        return float("inf")
    truth = target.cov()
    err = np.linalg.norm(C - truth)
    return float(err / np.linalg.norm(truth)) if relative else float(err)


def eta_for_bias(target, J, bias, tol=1e-6, integrator="euler"):
    """Largest ``eta`` whose stationary covariance bias equals ``bias``.

    The equal-bias stepsize is the basis of the fair comparison: methods are run
    at the stepsize that puts them all at the same asymptotic accuracy, and only
    then are their convergence rates compared.
    """
    hi = max_stable_stepsize(target, J, integrator=integrator)
    if hi <= 0:
        return 0.0
    if covariance_bias(target, J, hi * (1 - 1e-9), integrator=integrator) < bias:
        return hi
    lo = 1e-14
    hi = hi * (1 - 1e-9)
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if covariance_bias(target, J, mid, integrator=integrator) < bias:
            lo = mid
        else:
            hi = mid
    return lo


def optimize_J(target, bias, n_restarts=4, seed=0, maxiter=400, delta_cap=None):
    """Maximise the equal-bias per-iteration rate over all skew ``J``.

    This optimises the quantity the experiment actually reports -- the
    convergence rate at a *fixed* discretisation bias -- rather than the
    continuous-time spectral gap, which ignores the stepsize the bias allows.
    Returns ``(J, rate, eta)``.
    """
    from scipy.optimize import minimize

    d = target.d
    if d < 2:
        return np.zeros((d, d)), ms_rate(target, None, eta_for_bias(target, None, bias)), \
            eta_for_bias(target, None, bias)
    n_par = d * (d - 1) // 2
    rng = np.random.default_rng(seed)

    def objective(v):
        J = _vec_to_skew(v, d)
        if delta_cap is not None and np.linalg.norm(J, 2) > delta_cap:
            return 1e6
        e = eta_for_bias(target, J, bias, tol=1e-4)
        if e <= 0:
            return 1e6
        r = ms_rate(target, J, e)
        return -r if np.isfinite(r) else 1e6

    best_J = np.zeros((d, d))
    best_eta = eta_for_bias(target, None, bias)
    best_rate = ms_rate(target, None, best_eta) if best_eta > 0 else float("-inf")
    scale = float(np.linalg.norm(target.Sigma_inv, 2)) ** 0.5
    for r in range(n_restarts):
        v0 = np.zeros(n_par) if r == 0 else rng.standard_normal(n_par) * scale * 0.5 * r
        res = minimize(objective, v0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-8, "fatol": 1e-10})
        J = _vec_to_skew(res.x, d)
        if delta_cap is not None:
            nrm = np.linalg.norm(J, 2)
            if nrm > delta_cap:
                J = J * (delta_cap / nrm)
        e = eta_for_bias(target, J, bias)
        if e <= 0:
            continue
        rate = ms_rate(target, J, e)
        if rate > best_rate:
            best_J, best_rate, best_eta = J, rate, e
    return best_J, float(best_rate), float(best_eta)


def _vec_to_skew(v, d):
    J = np.zeros((d, d))
    iu = np.triu_indices(d, 1)
    J[iu] = v
    return J - J.T


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
