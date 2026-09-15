"""Skew (antisymmetric) drift fields for anchored Langevin on the Euclidean ball.

Sampling ``pi(x) \\propto e^{-U(x)} 1_K(x)`` on ``K = {||x||_2 <= R}`` by projection
puts two conditions on a skew field ``J``:

1. *Invariance.*  For a state-dependent ``J`` the divergence-free stationary flux is
   ``F_i = sum_j d_j (J_ij phi)`` with ``phi = e^{-U_0}``, which turns the anchored
   dynamics into

       dX = e^{Delta} [ -(I + J(X)) grad U_0(X) + div J(X) ] dt + sqrt(2) e^{Delta/2} dW,

   with ``(div J)_i = sum_j d_j J_ij``.  The correction ``div J`` vanishes for constant
   ``J`` but *not* in general -- dropping it biases the target.

2. *Boundary.*  The skew drift must be tangential to ``dK``, i.e. ``J(x) nu(x) = 0`` for
   ``||x|| = R``.  Otherwise the skew term pushes mass into the wall, the projection
   absorbs it, and the stationary law is distorted by an amount no stepsize refinement
   removes.

Three fields are provided:

``none``
    ``J = 0``.  Reversible anchored Langevin.

``const``
    The tridiagonal constant skew ``J_a``.  ``div J_a = 0`` but ``J_a x != 0``, so it
    violates the boundary condition.

``axial``
    The state-dependent field

        J_s(x) = (s / R^2) [ ||x||^2 A + x (Ax)^T - (Ax) x^T ],   A^T = -A constant,

    which satisfies ``J_s(x) x = 0`` *identically* (hence on ``dK``), and has the closed
    form divergence

        div J_s(x) = -(s / R^2) (d - 2) A x.

    The ``1/R^2`` normalisation makes ``||J_s||`` on ``dK`` comparable to ``||J_a||``, so
    the strength ``s`` means the same thing for both fields at any radius.

    In d = 3 this correction is a single term; in the d >= 4 posteriors of this repo it
    is genuinely non-zero, so the correction term has to be carried.  (The 3-d axial
    field ``s x cross w`` of the original ball notebook is divergence-free, which is a
    property of that field, not a general licence.)
"""

import numpy as np
import torch


def tridiagonal_skew(dim: int, dtype=torch.float64) -> torch.Tensor:
    """The constant skew ``J_a``: +1 on the first superdiagonal, -1 below."""
    a = torch.ones(dim - 1, dtype=dtype)
    return torch.diag(a, diagonal=1) + torch.diag(-a, diagonal=-1)


def unit_skew(dim: int, dtype=torch.float64) -> torch.Tensor:
    """``J_a`` rescaled to unit spectral norm, so ``s`` means the same thing for
    every field and every dimension."""
    A = tridiagonal_skew(dim, dtype)
    return A / torch.linalg.matrix_norm(A, ord=2)


class SkewField:
    """Base class: a field must know how to apply ``J(x)`` and its divergence."""

    name = "base"
    state_dependent = False

    def apply(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Return ``J(x) g`` row-wise for batches ``x, g`` of shape ``(N, d)``."""
        raise NotImplementedError

    def divergence(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(div J)(x)`` of shape ``(N, d)``."""
        raise NotImplementedError

    def matrix(self, x_row: torch.Tensor) -> torch.Tensor:
        """``J(x)`` as a dense ``(d, d)`` matrix for a single point (checks/plots)."""
        d = x_row.shape[0]
        eye = torch.eye(d, dtype=x_row.dtype)
        cols = self.apply(x_row.expand(d, d), eye)          # column j is J(x) e_j
        return cols.T


class ZeroSkew(SkewField):
    name = "none"

    def __init__(self, dim: int, s: float = 0.0):
        self.dim, self.s = dim, 0.0

    def apply(self, x, g):
        return torch.zeros_like(g)

    def divergence(self, x):
        return torch.zeros_like(x)


class ConstantSkew(SkewField):
    """``J_a`` scaled by ``s``.  Divergence-free, but ``J_a x != 0`` on the sphere."""

    name = "const"

    def __init__(self, dim: int, s: float, normalise: bool = True, dtype=torch.float64):
        self.dim, self.s = dim, float(s)
        # ``normalise=False`` is the literal matrix of the paper: superdiagonal ``a``,
        # subdiagonal ``-a``.  The default rescales to unit spectral norm so that ``s``
        # means the same push for every field and every dimension; the two differ by
        # ||J_a||_2 = 2 cos(pi/(d+1)), which is 1.90 at d = 9.
        self.normalise = bool(normalise)
        self.A = unit_skew(dim, dtype) if normalise else tridiagonal_skew(dim, dtype)
        self.J = self.s * self.A

    def apply(self, x, g):
        return g @ self.J.T

    def divergence(self, x):
        return torch.zeros_like(x)


class AxialSkew(SkewField):
    """``J_s(x) = s [ r^2 A + x (Ax)^T - (Ax) x^T ]`` with ``r = ||x||``.

    Antisymmetric for every ``x``; annihilates ``x`` exactly, so ``J_s nu = 0`` on the
    sphere; polynomial in ``x`` (no singularity at the origin); and

        div J_s(x) = -s (d - 2) A x.

    ``drop_correction=True`` deliberately omits that term -- the ablation that shows the
    correction is not optional.
    """

    name = "axial"
    state_dependent = True

    def __init__(self, dim: int, s: float, radius: float = 1.0,
                 drop_correction: bool = False, dtype=torch.float64):
        self.dim, self.s = dim, float(s)
        self.A = unit_skew(dim, dtype)
        self.c = float(s) / float(radius) ** 2
        self.drop_correction = bool(drop_correction)

    def apply(self, x, g):
        Ax = x @ self.A.T                                   # (N, d)
        Ag = g @ self.A.T
        r2 = (x * x).sum(dim=1, keepdim=True)
        xg = (x * g).sum(dim=1, keepdim=True)
        Axg = (Ax * g).sum(dim=1, keepdim=True)
        return self.c * (r2 * Ag + Axg * x - xg * Ax)

    def divergence(self, x):
        if self.drop_correction:
            return torch.zeros_like(x)
        return -self.c * (self.dim - 2) * (x @ self.A.T)


class BlockCrossSkew(SkewField):
    """Block-diagonal 3x3 cross-product blocks -- the field ``J_s(x)`` of the paper.

    For ``d = 3B`` and scales ``s = [s_1, ..., s_B]``, block ``k`` acts on the coordinate
    triple ``I_k = (3k, 3k+1, 3k+2)`` as ``v -> s_k (x_{I_k} times v_{I_k})``, i.e.

        [[0, -s_k x_3,  s_k x_2],
         [ s_k x_3, 0, -s_k x_1],
         [-s_k x_2,  s_k x_1, 0]].

    This is the tensor ``J_I`` of the higher-dimensional ball construction, one block per
    coordinate triple.  Each block is skew-symmetric, divergence-free, and annihilates its
    own sub-vector, so

        div J_s(x) = 0,        J_s(x) x = 0.

    The first two make it admissible.  The third is what makes it *tangent* to a centred
    sphere, which is why it wins under a ball constraint -- and, unconstrained, is exactly
    what costs it: it annihilates any part of the drift parallel to ``x``, the isotropic
    Gaussian prior's gradient included.
    """

    name = "block"
    state_dependent = True

    def __init__(self, dim: int, s, dtype=torch.float64):
        if dim % 3:
            raise ValueError("block cross field needs dim divisible by 3, got {}".format(dim))
        self.dim = dim
        self.n_blocks = dim // 3
        vals = [float(s)] * self.n_blocks if np.isscalar(s) else [float(v) for v in s]
        if len(vals) != self.n_blocks:
            raise ValueError("need one scale per 3-block, got {} for {} blocks".format(
                len(vals), self.n_blocks))
        self.s = torch.tensor(vals, dtype=dtype)

    def apply(self, x, g):
        n = x.shape[0]
        xb = x.reshape(n, self.n_blocks, 3)
        gb = g.reshape(n, self.n_blocks, 3)
        out = torch.linalg.cross(xb, gb, dim=2) * self.s.view(1, -1, 1)
        return out.reshape(n, self.dim)

    def divergence(self, x):
        return torch.zeros_like(x)


def build_skew(kind: str, dim: int, s: float, radius: float = 1.0,
               drop_correction: bool = False) -> SkewField:
    if kind == "none":
        return ZeroSkew(dim)
    if kind == "const":
        return ConstantSkew(dim, s)
    if kind == "axial":
        return AxialSkew(dim, s, radius=radius, drop_correction=drop_correction)
    if kind == "block":
        return BlockCrossSkew(dim, s)
    raise ValueError("unknown skew field {!r}".format(kind))


# ----------------------------------------------------------------------------------
# numerical checks used in the notebook (and in tests)
# ----------------------------------------------------------------------------------
def autograd_divergence(field: SkewField, x: torch.Tensor) -> torch.Tensor:
    """``(div J)_i = sum_j d_j J_ij`` by autograd, as an independent check of the
    closed forms above."""
    n, d = x.shape
    x = x.detach().clone().requires_grad_(True)
    div = torch.zeros(n, d, dtype=x.dtype)
    eye = torch.eye(d, dtype=x.dtype)
    for j in range(d):
        col = field.apply(x, eye[j].expand(n, d))            # (J(x) e_j)_i, all i
        if not col.requires_grad:                        # constant field: J_ij is x-free
            continue
        for i in range(d):
            gi = torch.autograd.grad(col[:, i].sum(), x, retain_graph=True,
                                     allow_unused=True)[0]
            if gi is not None:
                div[:, i] += gi[:, j]
    return div.detach()


def antisymmetry_error(field: SkewField, x: torch.Tensor) -> float:
    """max ``||J(x) + J(x)^T||_inf`` over the batch."""
    err = 0.0
    for row in x:
        J = field.matrix(row)
        err = max(err, float((J + J.T).abs().max()))
    return err


def boundary_flux(field: SkewField, x: torch.Tensor) -> torch.Tensor:
    """``||J(x) nu(x)||`` with ``nu = x / ||x||``; must be 0 on ``dK`` for admissibility."""
    nu = x / x.norm(dim=1, keepdim=True)
    return field.apply(x, nu).norm(dim=1)


def mean_operator_norm(field: SkewField, x: torch.Tensor) -> float:
    """Average ``||J(x)||_2`` over a batch -- lets the reader check that two fields at
    the same ``s`` really are pushing comparably hard."""
    return float(np.mean([float(torch.linalg.matrix_norm(field.matrix(row), ord=2))
                          for row in x]))
