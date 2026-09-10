r"""Constructions of the skew-symmetric matrix ``J`` used to break reversibility.

For the anchored dynamics on a log-quadratic target the drift is linear,

    b_J(x) = (2 beta / nu) (J - I) Sigma^{-1} (x - mu),

so the natural surrogate for tuning ``J`` is the Ornstein-Uhlenbeck process
``dX = -(I - J) A x dt + sqrt(2) dW`` with ``A = Sigma^{-1}``, whose rate of
convergence is the spectral abscissa

    gap(J) = min_i Re lambda_i( (I - J) A ).

Two facts drive every construction here.

1. ``Tr(J A) = 0`` for skew ``J`` and symmetric ``A``, hence
   ``Tr((I - J)A) = Tr(A)`` is invariant.  The eigenvalues of ``(I-J)A`` always
   sum to ``Tr(A)``, so ``gap(J) <= Tr(A)/d`` -- the mean eigenvalue of ``A``.
   The reversible choice ``J = 0`` gives ``gap = lambda_min(A)``.  The best
   possible speed-up is therefore ``mean(lambda(A)) / min(lambda(A))``, which is
   large exactly when ``Sigma`` is ill conditioned.  (Hwang, Hwang-Sheu and
   Sheu 1993/2005; Lelievre, Nier and Pavliotis 2013.)
2. If ``J`` commutes with ``A`` then ``(I-J)A`` and ``A`` share eigenvectors and
   ``Re lambda_i((I-J)A) = lambda_i(A)``: the gap is *unchanged*.  Commuting
   ``J`` is the built-in negative control.  In particular for ``A = I``
   (isotropic target) every skew ``J`` commutes with ``A`` and no acceleration
   is possible.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "zero",
    "cyclic",
    "pairwise",
    "random_skew",
    "commuting",
    "eigenbasis_cyclic",
    "optimal",
    "scale_to_norm",
    "spectral_abscissa",
    "gap_upper_bound",
    "is_skew",
    "vec_to_skew",
    "skew_to_vec",
    "build",
]


def is_skew(J, tol=1e-10):
    J = np.asarray(J, dtype=np.float64)
    return bool(np.max(np.abs(J + J.T)) <= tol * max(1.0, np.max(np.abs(J))))


def zero(d):
    return np.zeros((d, d))


def scale_to_norm(J, delta):
    """Rescale ``J`` so that its spectral norm equals ``delta``."""
    J = np.asarray(J, dtype=np.float64)
    n = np.linalg.norm(J, 2)
    if n == 0.0:
        return J
    return (delta / n) * J


def cyclic(d, delta=1.0):
    """Nearest-neighbour cyclic skew matrix, normalised to spectral norm ``delta``.

    ``J[i, i+1] = +1``, ``J[i+1, i] = -1`` (indices mod ``d``).  For ``d = 1``
    this is necessarily zero -- the only skew 1x1 matrix.
    """
    J = np.zeros((d, d))
    if d < 2:
        return J
    for i in range(d):
        j = (i + 1) % d
        J[i, j] += 1.0
        J[j, i] -= 1.0
    if d == 2:  # the cyclic wrap doubles the single entry; harmless after scaling
        J = np.array([[0.0, 1.0], [-1.0, 0.0]])
    return scale_to_norm(J, delta)


def pairwise(d, delta=1.0):
    """Block-diagonal rotations in the coordinate planes (0,1), (2,3), ..."""
    J = np.zeros((d, d))
    for i in range(0, d - 1, 2):
        J[i, i + 1] = 1.0
        J[i + 1, i] = -1.0
    return scale_to_norm(J, delta)


def random_skew(d, delta=1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    M = rng.standard_normal((d, d))
    J = M - M.T
    return scale_to_norm(J, delta)


def eigenbasis_cyclic(Sigma, delta=1.0):
    """Cyclic skew matrix expressed in the eigenbasis of ``Sigma``.

    Sorting the eigenvectors by eigenvalue and coupling neighbours in that
    order mixes slow and fast directions, which is what breaks reversibility
    usefully.
    """
    evals, evecs = np.linalg.eigh(np.asarray(Sigma, dtype=np.float64))
    order = np.argsort(evals)
    Q = evecs[:, order]
    d = Q.shape[0]
    return scale_to_norm(Q @ cyclic(d, 1.0) @ Q.T, delta)


def commuting(Sigma, delta=1.0, tol=1e-9):
    """A skew matrix that commutes with ``Sigma`` -- the negative control.

    Such a matrix exists only when ``Sigma`` has a repeated eigenvalue: it is a
    rotation *inside* one eigenspace.  Returns the zero matrix (and therefore a
    trivially inert control) when the spectrum is simple.
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    evals, evecs = np.linalg.eigh(Sigma)
    d = Sigma.shape[0]
    J = np.zeros((d, d))
    i = 0
    while i < d:
        j = i
        while j + 1 < d and abs(evals[j + 1] - evals[i]) <= tol * max(1.0, abs(evals[i])):
            j += 1
        k = j - i + 1  # size of this eigenspace
        if k >= 2:
            block = np.zeros((k, k))
            block[0, 1] = 1.0
            block[1, 0] = -1.0
            V = evecs[:, i : j + 1]
            J = J + V @ block @ V.T
        i = j + 1
    if np.max(np.abs(J)) == 0.0:
        return J
    return scale_to_norm(J, delta)


# ------------------------------------------------------------- optimisation


def vec_to_skew(v, d):
    J = np.zeros((d, d))
    iu = np.triu_indices(d, 1)
    J[iu] = v
    return J - J.T


def skew_to_vec(J):
    d = J.shape[0]
    iu = np.triu_indices(d, 1)
    return J[iu]


def spectral_abscissa(J, A):
    """min_i Re lambda_i((I - J) A) -- the OU convergence rate."""
    d = A.shape[0]
    ev = np.linalg.eigvals((np.eye(d) - J) @ A)
    return float(np.min(ev.real))


def gap_upper_bound(A):
    """``Tr(A)/d``: the largest rate any skew perturbation can achieve."""
    return float(np.trace(A) / A.shape[0])


def optimal(A, delta=None, n_restarts=8, seed=0, maxiter=2000):
    """Numerically maximise ``spectral_abscissa(J, A)`` over skew ``J``.

    ``delta`` optionally constrains ``||J||_2 <= delta`` (a soft barrier); with
    ``delta=None`` the search is unconstrained and should approach the
    theoretical ceiling ``Tr(A)/d``.
    """
    A = np.asarray(A, dtype=np.float64)
    d = A.shape[0]
    if d < 2:
        return np.zeros((d, d))
    rng = np.random.default_rng(seed)
    n_par = d * (d - 1) // 2

    def negative_gap(v):
        J = vec_to_skew(v, d)
        val = spectral_abscissa(J, A)
        if delta is not None:
            excess = np.linalg.norm(J, 2) - delta
            if excess > 0:
                val -= 1e3 * excess ** 2
        return -val

    best, best_val = np.zeros((d, d)), spectral_abscissa(np.zeros((d, d)), A)
    scale = np.trace(A) / d
    for r in range(n_restarts):
        v0 = np.zeros(n_par) if r == 0 else rng.standard_normal(n_par) * scale * (0.3 * r)
        if delta is not None and np.linalg.norm(v0) > 0:
            J0 = vec_to_skew(v0, d)
            n0 = np.linalg.norm(J0, 2)
            if n0 > delta:
                v0 = v0 * (delta / n0)
        res = minimize(negative_gap, v0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-10, "fatol": 1e-12})
        J = vec_to_skew(res.x, d)
        if delta is not None:
            n = np.linalg.norm(J, 2)
            if n > delta:
                J = J * (delta / n)
        val = spectral_abscissa(J, A)
        if val > best_val:
            best, best_val = J, val
    return best


# ------------------------------------------------------------------ facade


def build(kind, target, delta=1.0, rng=None):
    """Build ``J`` for a :class:`~skewanchor.targets.LogQuadraticTarget`.

    ``kind`` is one of ``zero``, ``cyclic``, ``pairwise``, ``random``,
    ``eigen``, ``commuting``, ``optimal``.
    """
    d = target.d
    A = target.Sigma_inv
    if kind == "zero":
        return zero(d)
    if kind == "cyclic":
        return cyclic(d, delta)
    if kind == "pairwise":
        return pairwise(d, delta)
    if kind == "random":
        return random_skew(d, delta, rng)
    if kind == "eigen":
        return eigenbasis_cyclic(target.Sigma, delta)
    if kind == "commuting":
        return commuting(target.Sigma, delta)
    if kind == "optimal":
        return optimal(A, delta=None if delta is None else delta)
    raise ValueError(f"unknown J kind {kind!r}")
