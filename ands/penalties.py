"""Piecewise-quadratic regularizers and their exact Gaussian smoothing.

Lasso, MCP and SCAD are all continuous and piecewise quadratic, so the Gaussian
smoothing ``p_0(x) = E p(x + mu Z)`` and its derivative have a closed form and never need
a Monte Carlo step.  With ``p(t) = c_0 + c_1 t + c_2 t^2`` on ``(t_j, t_{j+1})`` and
``u_j = (t_j - x)/mu``,

    P_j = Phi(u_{j+1}) - Phi(u_j)
    M_j = phi(u_j) - phi(u_{j+1})
    Q_j = P_j + u_j phi(u_j) - u_{j+1} phi(u_{j+1})

    p_0 (x) = sum_j { (c_0 + c_1 x + c_2 x^2) P_j + mu (c_1 + 2 c_2 x) M_j + c_2 mu^2 Q_j }
    p_0'(x) = sum_j { (c_1 + 2 c_2 x) P_j + 2 c_2 mu M_j }

**The anchor does not always dominate.**  The anchored sampler's time change is
``a = e^{U - U_0} = e^{g - g_0}``, and the convenient claim ``a <= 1`` is Jensen -- which
needs ``p`` convex.  MCP and SCAD are non-convex by construction: tapering the penalty to
zero for large coefficients requires negative curvature, and on those pieces the anchor
sits *below* the penalty.

How far below is bounded, and the bound is a theorem rather than an observation.  Write
``u(x, t) = E p(x + sqrt(2t) Z)``, which solves the heat equation ``d_t u = d_xx u`` with
``u(., 0) = p``, so that ``p_0 = u(., mu^2/2)`` and

    p_0(x) - p(x) = int_0^{mu^2/2} (d_xx u)(x, s) ds.

The distributional second derivative of a continuous piecewise quadratic is ``2 c_2`` on
each piece plus a non-negative atom at every convex kink, so ``d_xx u >= 2 c_2^min``
pointwise and

    p_0(x) - p(x)  >=  c_2^min mu^2        for every x
    max_t (p - p_0)  <=  mu^2 max(0, -c_2^min)
    a(x)  <=  exp( d mu^2 max(0, -c_2^min) )   for a separable penalty in d coordinates.

The bound is *attained* whenever the smoothing kernel fits inside the most concave piece
(numerically, to five digits for mu <= 0.4 with these penalties at lam = 1), and turns
conservative once mu is comparable to the piece width -- at mu = 1 the true deficit is
about 0.8 of it.

Convex penalties have ``c_2^min = 0`` and recover ``a <= 1``.  The tempering argument
survives for MCP and SCAD -- ``a`` is still bounded by a constant of order 1 -- but it
survives because ``g - g_0`` is uniformly bounded, not because the anchor dominates.  A
penalty with unbounded negative curvature would break it.
"""

import math

import torch

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _Phi(u):
    return 0.5 * (1.0 + torch.erf(u / SQRT2))


def _phi(u):
    return INV_SQRT_2PI * torch.exp(-0.5 * u * u)


# The outermost pieces have infinite breakpoints, so the standard normal helpers are
# evaluated on a masked copy and the limits substituted back.
def _PhiX(u):
    fin = torch.isfinite(u)
    step = torch.where(u > 0, torch.ones_like(u), torch.zeros_like(u))
    return torch.where(fin, _Phi(torch.where(fin, u, torch.zeros_like(u))), step)


def _phiX(u):
    fin = torch.isfinite(u)
    return torch.where(fin, _phi(torch.where(fin, u, torch.zeros_like(u))), torch.zeros_like(u))


def _uphiX(u):
    fin = torch.isfinite(u)
    uu = torch.where(fin, u, torch.zeros_like(u))
    return torch.where(fin, uu * _phi(uu), torch.zeros_like(u))


class PiecewiseQuadratic:
    """A continuous piecewise-quadratic penalty and its exact Gaussian smoothing."""

    def __init__(self, breaks, coefs, name=""):
        self.t = list(breaks)
        self.c = [tuple(map(float, c)) for c in coefs]
        self.name = name
        if len(self.c) != len(self.t) + 1:
            raise ValueError("need one coefficient triple per piece")

    def raw(self, x):
        """The exact, non-differentiable penalty."""
        out = torch.zeros_like(x)
        ts = [-math.inf] + self.t + [math.inf]
        for j, (c0, c1, c2) in enumerate(self.c):
            m = (x > ts[j]) & (x <= ts[j + 1])
            out = torch.where(m, c0 + c1 * x + c2 * x * x, out)
        return out

    def smooth(self, x, mu):
        """``(p_0, p_0')`` in closed form."""
        val = torch.zeros_like(x)
        grad = torch.zeros_like(x)
        ts = [-math.inf] + self.t + [math.inf]
        for j, (c0, c1, c2) in enumerate(self.c):
            uj = torch.full_like(x, -math.inf) if ts[j] == -math.inf else (ts[j] - x) / mu
            uj1 = torch.full_like(x, math.inf) if ts[j + 1] == math.inf else (ts[j + 1] - x) / mu
            P = _PhiX(uj1) - _PhiX(uj)
            M = _phiX(uj) - _phiX(uj1)
            Q = P + _uphiX(uj) - _uphiX(uj1)
            val = val + (c0 + c1 * x + c2 * x * x) * P + mu * (c1 + 2 * c2 * x) * M + c2 * mu ** 2 * Q
            grad = grad + (c1 + 2 * c2 * x) * P + 2 * c2 * mu * M
        return val, grad

    def min_curvature(self):
        """The most negative quadratic coefficient; 0 for a convex penalty."""
        return min(c2 for _, _, c2 in self.c)

    def is_convex(self):
        return self.min_curvature() >= 0.0

    def max_anchor_deficit(self, mu):
        """Upper bound on ``max_t (p - p_0)``, namely ``mu^2 max(0, -c_2^min)``.

        Always valid (see the module docstring); attained when the smoothing kernel fits
        inside the most concave piece, conservative when ``mu`` approaches the piece
        width.
        """
        return mu ** 2 * max(0.0, -self.min_curvature())

    def anchor_bound(self, mu, dim=1):
        """The bound on ``a = e^{g - g_0}`` for this penalty applied to ``dim`` coordinates.

        Equals 1 exactly when the penalty is convex.
        """
        return math.exp(dim * self.max_anchor_deficit(mu))


def lasso(lam):
    return PiecewiseQuadratic([0.0], [(0., -lam, 0.), (0., lam, 0.)], "Lasso")


def mcp(lam, a):
    return PiecewiseQuadratic(
        [-a * lam, 0.0, a * lam],
        [(a * lam ** 2 / 2, 0., 0.), (0., -lam, -1 / (2 * a)),
         (0., lam, -1 / (2 * a)), (a * lam ** 2 / 2, 0., 0.)], "MCP")


def scad(lam, a):
    c = 1.0 / (2 * (a - 1))
    return PiecewiseQuadratic(
        [-a * lam, -lam, 0.0, lam, a * lam],
        [(lam ** 2 * (a + 1) / 2, 0., 0.), (-lam ** 2 * c, -2 * a * lam * c, -c), (0., -lam, 0.),
         (0., lam, 0.), (-lam ** 2 * c, 2 * a * lam * c, -c), (lam ** 2 * (a + 1) / 2, 0., 0.)], "SCAD")
