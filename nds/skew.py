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

    Distances are measured from ``center`` in the metric ``M`` (pass
    ``metric=D^{-1}`` for whitened coordinates): ``r(w)^2 = (w - c)^T M (w - c)``.
    Three profiles:

    * ``"grow"``: ``1 - exp(-r^2 / 2 rho^2)`` -- off at the centre, on far away;
    * ``"decay"``: ``exp(-r^2 / 2 rho^2)`` -- the reverse;
    * ``"taper"``: ``1 / (1 + r^2 / rho^2)`` -- a heavy-tailed version of
      ``"decay"``, which keeps the field at half strength a distance ``rho``
      from the centre instead of extinguishing it.

    The divergence is available in closed form,

        Gamma(w) = alpha A grad s(w),

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
        center: np.ndarray | None = None,
    ) -> None:
        if profile not in ("grow", "decay", "taper"):
            raise ValueError("profile must be 'grow', 'decay' or 'taper'")
        self.A = np.ascontiguousarray(A)
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.profile = profile
        self.drop_correction = bool(drop_correction)
        self.metric = None if metric is None else np.ascontiguousarray(metric)
        d = len(self.A)
        self.center = (
            np.zeros((d, 1)) if center is None else np.asarray(center, float).reshape(d, 1)
        )
        if drop_correction:
            self.label = "$J_s$, correction dropped"

    def _offset(self, W: np.ndarray) -> tuple:
        """``w - c`` and ``M (w - c)``."""
        Z = W - self.center
        return Z, (Z if self.metric is None else self.metric @ Z)

    def _profile(self, W: np.ndarray) -> tuple:
        """``s(w)`` of shape (1, m) and its gradient of shape (d, m)."""
        Z, MZ = self._offset(W)
        r2 = (Z * MZ).sum(axis=0, keepdims=True)
        if self.profile == "taper":
            denom = 1.0 + r2 / self.rho**2
            return 1.0 / denom, -2.0 * MZ / (self.rho**2 * denom**2)
        bump = np.exp(-0.5 * r2 / self.rho**2)
        grad_bump = -bump * MZ / self.rho**2
        if self.profile == "grow":
            return 1.0 - bump, -grad_bump
        return bump, grad_bump

    def scale(self, W: np.ndarray) -> np.ndarray:
        return self._profile(W)[0]

    def apply(self, W: np.ndarray, G: np.ndarray) -> np.ndarray:
        return self.alpha * self.scale(W) * (self.A @ G)

    def divergence(self, W: np.ndarray) -> np.ndarray:
        if self.drop_correction:
            return np.zeros_like(W)
        return self.alpha * (self.A @ self._profile(W)[1])


class DirectionalSkew(SkewField):
    r"""``J(w) = alpha s(w) A`` with ``s(w) = tanh(c . w / ell)``.

    ``s(w) = offset + gain * tanh(c . (w - center) / ell)``, so ``offset`` and
    ``gain`` place the modulation range: the default ``(0, 1)`` sweeps the field
    from ``-A`` to ``+A``, while ``(0.75, -0.25)`` keeps it between half and full
    strength -- useful when the field is already tuned and only its *variation*
    is under test.
    Centring matters: the posterior bulk of these problems sits many whitened
    standard deviations away from ``w = 0``, so any profile centred at the origin
    is saturated where the chain actually lives, its gradient is ~0, and
    ``Gamma`` is numerically negligible.  Centred on the bulk with ``ell`` of the
    order of one posterior standard deviation, ``s`` varies by order one across
    the posterior and ``Gamma`` is comparable to the reversible drift -- which is
    the regime in which dropping it can be seen to bias the uncorrected
    diffusion.

        Gamma(w) = alpha A grad s(w),
        grad s(w) = gain (1 - tanh^2(c . (w - center) / ell)) c / ell.
    """

    label = "directional $J_c$"

    def __init__(
        self,
        A: np.ndarray,
        alpha: float = 1.0,
        direction: np.ndarray | None = None,
        length_scale: float = 1.0,
        drop_correction: bool = False,
        center: np.ndarray | None = None,
        offset: float = 0.0,
        gain: float = 1.0,
    ) -> None:
        self.A = np.ascontiguousarray(A)
        self.alpha = float(alpha)
        self.offset = float(offset)
        self.gain = float(gain)
        d = len(self.A)
        c = np.zeros(d) if direction is None else np.asarray(direction, float).ravel()
        if direction is None:
            c[0] = 1.0
        self.c = c.reshape(d, 1)
        self.length_scale = float(length_scale)
        self.center = (
            np.zeros((d, 1)) if center is None else np.asarray(center, float).reshape(d, 1)
        )
        self.drop_correction = bool(drop_correction)
        if drop_correction:
            self.label = "$J_c$, correction dropped"

    def _tanh(self, W: np.ndarray) -> np.ndarray:
        proj = (self.c * (W - self.center)).sum(axis=0, keepdims=True)
        return np.tanh(proj / self.length_scale)

    def scale(self, W: np.ndarray) -> np.ndarray:
        return self.offset + self.gain * self._tanh(W)

    def apply(self, W: np.ndarray, G: np.ndarray) -> np.ndarray:
        return self.alpha * self.scale(W) * (self.A @ G)

    def divergence(self, W: np.ndarray) -> np.ndarray:
        if self.drop_correction:
            return np.zeros_like(W)
        t = self._tanh(W)
        grad_s = self.gain * (1.0 - t**2) * self.c / self.length_scale
        return self.alpha * (self.A @ grad_s)


class GatedSkew(SkewField):
    r"""``J(w) = alpha s(w) A`` with a sharp radial gate around the bulk.

    ``s(w) = 1 / (1 + exp((r(w) - radius) / width))`` with
    ``r(w)^2 = (w - center)^T M (w - center)``: the field is on inside the gate
    and off outside it, with a transition of width ``width``.

    :class:`LocalizedSkew`'s Gaussian profiles decay over a scale comparable to
    the distance they are centred on, which is too gradual when the ``w = 0``
    start and the bulk are only a factor of two apart in whitened distance --
    the rotation is then still a quarter of its strength during the descent,
    where the quadratic model that designed it does not hold.  A gate separates
    the two regimes properly: off while the chain descends, full strength once
    it is in the bulk.

        Gamma(w) = -alpha s(w) (1 - s(w)) / (width r(w)) * A M (w - center).
    """

    label = "gated $J_s$"

    def __init__(
        self,
        A: np.ndarray,
        alpha: float = 1.0,
        radius: float = 1.0,
        width: float = 0.1,
        metric: np.ndarray | None = None,
        center: np.ndarray | None = None,
        drop_correction: bool = False,
    ) -> None:
        self.A = np.ascontiguousarray(A)
        self.alpha = float(alpha)
        self.radius = float(radius)
        self.width = float(width)
        self.metric = None if metric is None else np.ascontiguousarray(metric)
        d = len(self.A)
        self.center = (
            np.zeros((d, 1)) if center is None else np.asarray(center, float).reshape(d, 1)
        )
        self.drop_correction = bool(drop_correction)
        if drop_correction:
            self.label = "gated $J_s$, correction dropped"

    def _radius(self, W: np.ndarray) -> tuple:
        Z = W - self.center
        MZ = Z if self.metric is None else self.metric @ Z
        r2 = (Z * MZ).sum(axis=0, keepdims=True)
        return np.sqrt(np.maximum(r2, 1e-300)), MZ

    def scale(self, W: np.ndarray) -> np.ndarray:
        r, _ = self._radius(W)
        return 1.0 / (1.0 + np.exp(np.clip((r - self.radius) / self.width, -500, 500)))

    def apply(self, W: np.ndarray, G: np.ndarray) -> np.ndarray:
        return self.alpha * self.scale(W) * (self.A @ G)

    def divergence(self, W: np.ndarray) -> np.ndarray:
        if self.drop_correction:
            return np.zeros_like(W)
        r, MZ = self._radius(W)
        s = self.scale(W)
        grad_s = -(s * (1.0 - s) / (self.width * r)) * MZ
        return self.alpha * (self.A @ grad_s)
