"""Skew-symmetric fields J(w) that make the sampler irreversible.

The sampled dynamics are the irreversible Langevin diffusion

    dw = [-(D + J(w)) grad U(w) + Gamma(w)] dt + sqrt(2 D) dB,
    Gamma_i(w) = sum_j d/dw_j (D_ij + J_ij(w)),

which leaves ``exp(-U)`` invariant for any skew-symmetric ``J`` (Ma, Chen and
Fox, 2015, *A complete recipe for stochastic gradient MCMC*).  For a constant
``J`` the divergence term ``Gamma`` vanishes, because ``sum_ij J_ij d_i d_j U``
is the contraction of an antisymmetric matrix with a symmetric Hessian.  For a
state-dependent ``J`` it does not, and dropping it changes the invariant
measure of the *diffusion* -- though not of the Metropolis-corrected chain in
:mod:`nds.sampler`, which only loses efficiency.

Each field is vectorised over walkers: ``W`` and ``G`` have shape ``(d, m)``.
Note that ``J`` is chosen by the user, so its divergence is available in closed
form; nothing here differentiates the target.
"""

from __future__ import annotations

import numpy as np


def random_skew(d: int, seed: int = 0) -> np.ndarray:
    """A random skew-symmetric matrix normalised to unit spectral norm."""
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(d, d))
    A = G - G.T
    return A / np.linalg.norm(A, ord=2)


def whiten(A: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Express a skew matrix in the geometry of ``D = L L^T``: ``L A L^T``.

    ``L A L^T`` is skew whenever ``A`` is, and it carries the same scale as
    ``D``, so an irreversibility strength ``alpha = 1`` means "a rotation as
    strong as the reversible part" in whitened coordinates rather than in the
    arbitrary units of the parameters.
    """
    return L @ A @ L.T


def cyclic_skew(d: int) -> np.ndarray:
    """The nearest-neighbour cyclic rotation ``e_i e_{i+1}^T - e_{i+1} e_i^T``."""
    A = np.zeros((d, d))
    idx = np.arange(d)
    A[idx, (idx + 1) % d] = 1.0
    A -= A.T
    return A / np.linalg.norm(A, ord=2)


class SkewField:
    """Interface: apply ``J(w)`` to a vector and return its divergence."""

    label = "J"

    def apply(self, W: np.ndarray, G: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def divergence(self, W: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ZeroSkew(SkewField):
    """``J = 0``: the reversible baseline."""

    label = "J = 0"

    def apply(self, W: np.ndarray, G: np.ndarray) -> np.ndarray:
        return np.zeros_like(G)

    def divergence(self, W: np.ndarray) -> np.ndarray:
        return np.zeros_like(W)


class ConstantSkew(SkewField):
    """``J(w) = alpha A`` with ``A`` skew-symmetric and of unit spectral norm."""

    label = "constant $J_a$"

    def __init__(self, A: np.ndarray, alpha: float = 1.0) -> None:
        self.A = np.ascontiguousarray(A)
        self.alpha = float(alpha)

    def apply(self, W: np.ndarray, G: np.ndarray) -> np.ndarray:
        return self.alpha * (self.A @ G)

    def divergence(self, W: np.ndarray) -> np.ndarray:
        return np.zeros_like(W)


class LocalizedSkew(SkewField):
    r"""``J(w) = alpha s(w) A`` with a radial profile ``s``.

    Distances use the metric ``M`` (pass ``metric=D^{-1}`` to measure them in
    whitened coordinates): ``r(w)^2 = w^T M w``.
    With ``s(w) = 1 - exp(-r(w)^2 / 2 rho^2)`` the rotation is switched off at
    the ``w = 0`` start and switched on once the walker has travelled a
    distance of order ``rho``; ``profile="decay"`` flips that around.  The
    divergence is available in closed form,

        Gamma(w) = alpha A grad s(w),   grad s(w) = +/- (M w / rho^2) exp(-r(w)^2 / 2 rho^2),

    so the correction term costs one matrix-vector product.
    """

    label = "state-dependent $J_s$"

    def __init__(
        self,
        A: np.ndarray,
        alpha: float = 1.0,
        rho: float = 1.0,
        profile: str = "grow",
        drop_correction: bool = False,
        metric: np.ndarray | None = None,
    ) -> None:
        if profile not in ("grow", "decay"):
            raise ValueError("profile must be 'grow' or 'decay'")
        self.A = np.ascontiguousarray(A)
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.profile = profile
        self.drop_correction = bool(drop_correction)
        self.metric = None if metric is None else np.ascontiguousarray(metric)
        if drop_correction:
            self.label = "$J_s$, correction dropped"

    def _metric_apply(self, W: np.ndarray) -> np.ndarray:
        return W if self.metric is None else self.metric @ W

    def _bump(self, W: np.ndarray) -> np.ndarray:
        """exp(-r(w)^2 / 2 rho^2) per walker, shape (1, m), r^2 = w^T M w."""
        MW = self._metric_apply(W)
        r2 = (W * MW).sum(axis=0, keepdims=True)
        return np.exp(-0.5 * r2 / self.rho**2)

    def scale(self, W: np.ndarray) -> np.ndarray:
        bump = self._bump(W)
        return 1.0 - bump if self.profile == "grow" else bump

    def apply(self, W: np.ndarray, G: np.ndarray) -> np.ndarray:
        return self.alpha * self.scale(W) * (self.A @ G)

    def divergence(self, W: np.ndarray) -> np.ndarray:
        if self.drop_correction:
            return np.zeros_like(W)
        sign = 1.0 if self.profile == "grow" else -1.0
        grad_s = sign * self._bump(W) * self._metric_apply(W) / self.rho**2
        return self.alpha * (self.A @ grad_s)
