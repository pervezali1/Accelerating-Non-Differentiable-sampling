"""
nald.py -- Non-reversible anchored Langevin dynamics (NALD / NRALD).

Reference: Ali, Wang, Wang, Zhu, "Accelerating Non-Smooth and Heavy-Tailed Sampling" (2026).

Everything here is derived from the paper:

*   anchored speed            a(x) = exp(U(x) - U_0(x))                                   (1.3)
*   canonical NALD            dX = -a (I + alpha J) grad U_0 dt + sqrt(2a) dW              (2.9)
*   stream construction       c = e^U J grad psi,  psi constant on dK  =>  c.n = 0        (2.6), Lemma 2.3(i)
*   Euler / projected Euler   x_{k+1} = Pi_K[x_k + eta b(x_k) + sqrt(2 eta a(x_k)) xi]     (7.1)-(7.2)
*   closed-form Gaussian smoothing of piecewise-quadratic penalties                        Lemma 7.1
*   quadratic-clock heavy-tailed example with exact asymptotic variance                    Prop. 6.7
*   Student-t superquadratic clock                                                         Cor. 6.2

and one construction that the paper leaves open ("how to choose J in practice is a challenging
problem", Section 1): a *spectrally matched* constant skew matrix built from a curvature matrix
H, which pairs the stiffest and softest directions of H and rotates each pair with exactly the
strength that equalises the real parts of the drift spectrum (see `spectral_J`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, ndtr
from scipy.stats import norm

Array = np.ndarray

# ----------------------------------------------------------------------------------------------
# 1. Piecewise-quadratic penalties and their closed-form Gaussian smoothing (Lemma 7.1)
# ----------------------------------------------------------------------------------------------


def _phi(u: Array) -> Array:
    """Standard normal pdf with phi(+-inf) = 0."""
    out = np.zeros_like(u, dtype=float)
    fin = np.isfinite(u)
    out[fin] = np.exp(-0.5 * u[fin] ** 2) / math.sqrt(2 * math.pi)
    return out


def _uphi(u: Array) -> Array:
    """u * phi(u) with the convention u*phi(u) -> 0 at +-inf."""
    out = np.zeros_like(u, dtype=float)
    fin = np.isfinite(u)
    out[fin] = u[fin] * np.exp(-0.5 * u[fin] ** 2) / math.sqrt(2 * math.pi)
    return out


class PiecewiseQuadratic:
    """p(t) = c0_j + c1_j t + c2_j t^2 on (t_j, t_{j+1}), continuous, with breakpoints t_1<...<t_n.

    Provides p, p', the Gaussian smoothing p_0(x) = E p(x + mu Z), p_0', p_0'' in closed form."""

    def __init__(self, breaks: Sequence[float], coefs: Sequence[Sequence[float]], name: str = ""):
        self.t = np.asarray(breaks, dtype=float)
        self.c = np.asarray(coefs, dtype=float)          # (n+1, 3)
        assert self.c.shape == (len(self.t) + 1, 3)
        assert np.all(np.diff(self.t) > 0)
        self.name = name
        # continuity check at the breakpoints
        for j, tj in enumerate(self.t):
            left = self.c[j] @ [1, tj, tj ** 2]
            right = self.c[j + 1] @ [1, tj, tj ** 2]
            assert abs(left - right) < 1e-9 * (1 + abs(left)), f"{name}: discontinuous at {tj}"
        # jump of p' at each breakpoint (needed for p_0'')
        self.dp_jump = np.array([(self.c[j + 1, 1] + 2 * self.c[j + 1, 2] * tj)
                                 - (self.c[j, 1] + 2 * self.c[j, 2] * tj)
                                 for j, tj in enumerate(self.t)])

    # -- the raw (non-smooth) penalty ----------------------------------------------------------
    def value(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        j = np.searchsorted(self.t, x, side="right")
        c = self.c[j]
        return c[..., 0] + c[..., 1] * x + c[..., 2] * x ** 2

    def deriv(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        j = np.searchsorted(self.t, x, side="right")
        c = self.c[j]
        return c[..., 1] + 2 * c[..., 2] * x

    # -- Gaussian smoothing, Lemma 7.1 ----------------------------------------------------------
    def _pieces(self, x: Array, mu: float):
        x = np.asarray(x, dtype=float)
        u = (self.t[None, :] - x[..., None]) / mu                      # (..., n)
        inf = np.full(x.shape + (1,), np.inf)
        u_lo = np.concatenate([-inf, u], axis=-1)                       # u_0 = -inf
        u_hi = np.concatenate([u, inf], axis=-1)                        # u_{n+1} = +inf
        P = ndtr(u_hi) - ndtr(u_lo)
        M = _phi(u_lo) - _phi(u_hi)
        Q = P + _uphi(u_lo) - _uphi(u_hi)
        return x, P, M, Q, u_lo

    def smoothed(self, x: Array, mu: float) -> Array:
        """p_0(x) = E p(x + mu Z), eq. (7.8)."""
        x, P, M, Q, _ = self._pieces(x, mu)
        c0, c1, c2 = self.c[:, 0], self.c[:, 1], self.c[:, 2]
        xx = x[..., None]
        return np.sum((c0 + c1 * xx + c2 * xx ** 2) * P + mu * (c1 + 2 * c2 * xx) * M
                      + c2 * mu ** 2 * Q, axis=-1)

    def smoothed_deriv(self, x: Array, mu: float) -> Array:
        """p_0'(x) = E p'(x + mu Z), eq. (7.9)."""
        x, P, M, _, _ = self._pieces(x, mu)
        c1, c2 = self.c[:, 1], self.c[:, 2]
        xx = x[..., None]
        return np.sum((c1 + 2 * c2 * xx) * P + 2 * c2 * mu * M, axis=-1)

    def smoothed_second_deriv(self, x: Array, mu: float) -> Array:
        """p_0''(x) = sum_j 2 c2_j P_j + sum_j [p'](t_j) phi(u_j) / mu."""
        x, P, _, _, u_lo = self._pieces(x, mu)
        c2 = self.c[:, 2]
        smooth_part = np.sum(2 * c2 * P, axis=-1)
        jump_part = np.sum(self.dp_jump * _phi(u_lo[..., 1:]), axis=-1) / mu
        return smooth_part + jump_part

    def smoothing_gap_bound(self, mu: float) -> Tuple[float, float]:
        """Numerical bounds of p_0 - p over a wide grid (used to bound a = exp(p - p_0))."""
        g = np.linspace(-50 * mu - 10 * (abs(self.t).max() if len(self.t) else 1),
                        50 * mu + 10 * (abs(self.t).max() if len(self.t) else 1), 200001)
        gap = self.smoothed(g, mu) - self.value(g)
        return float(gap.min()), float(gap.max())


def lasso_penalty(lam: float) -> PiecewiseQuadratic:
    return PiecewiseQuadratic([0.0], [[0, -lam, 0], [0, lam, 0]], name="lasso")


def mcp_penalty(lam: float, a: float) -> PiecewiseQuadratic:
    assert a > 1
    return PiecewiseQuadratic([-a * lam, 0.0, a * lam],
                              [[a * lam ** 2 / 2, 0, 0], [0, -lam, -1 / (2 * a)],
                               [0, lam, -1 / (2 * a)], [a * lam ** 2 / 2, 0, 0]], name="mcp")


def scad_penalty(lam: float, a: float) -> PiecewiseQuadratic:
    assert a > 1
    c = 1.0 / (2 * (a - 1))
    return PiecewiseQuadratic(
        [-a * lam, -lam, 0.0, lam, a * lam],
        [[lam ** 2 * (a + 1) / 2, 0, 0], [-lam ** 2 * c, -2 * a * lam * c, -c], [0, -lam, 0],
         [0, lam, 0], [-lam ** 2 * c, 2 * a * lam * c, -c], [lam ** 2 * (a + 1) / 2, 0, 0]],
        name="scad")


# ----------------------------------------------------------------------------------------------
# 2. Potentials: U (non-smooth), U_0 (smooth anchor), grad U_0, a = exp(U - U_0)
# ----------------------------------------------------------------------------------------------


class Potential:
    """Interface. All methods accept W of shape (R, d) (or (d,)) and are vectorised over R."""
    d: int

    def U(self, W: Array) -> Array: ...
    def U0(self, W: Array) -> Array: ...
    def grad_U0(self, W: Array) -> Array: ...

    def a(self, W: Array) -> Array:
        return np.exp(self.U(W) - self.U0(W))

    def grad_U0_and_a(self, W: Array) -> Tuple[Array, Array]:
        """(grad U_0(W), a(W)) -- subclasses may share work between the two."""
        return self.grad_U0(W), self.a(W)

    def hess_U0(self, w: Array) -> Array:
        """Dense Hessian of U_0 at a single point (finite differences by default)."""
        w = np.asarray(w, dtype=float)
        h = 1e-5
        H = np.zeros((self.d, self.d))
        for i in range(self.d):
            e = np.zeros(self.d); e[i] = h
            H[:, i] = (self.grad_U0(w + e) - self.grad_U0(w - e)) / (2 * h)
        return 0.5 * (H + H.T)

    def mode_U0(self, w0: Optional[Array] = None) -> Array:
        w0 = np.zeros(self.d) if w0 is None else np.asarray(w0, float)
        res = minimize(lambda w: float(self.U0(w)), w0, jac=lambda w: self.grad_U0(w),
                       method="L-BFGS-B", options=dict(maxiter=5000, gtol=1e-9))
        return res.x


def _rows(W: Array) -> Tuple[Array, bool]:
    W = np.asarray(W, dtype=float)
    return (W[None, :], True) if W.ndim == 1 else (W, False)


class PenalisedRegression(Potential):
    """U(w) = |y - X w|^2 / (2 sigma^2) + sum_i p(w_i)  with p a piecewise-quadratic penalty,
    U_0 = same with p -> p_0 (Gaussian smoothing with width mu).  a = exp(sum p - p_0) is bounded."""

    def __init__(self, X: Array, y: Array, sigma2: float, penalty: PiecewiseQuadratic, mu: float):
        self.X, self.y, self.sigma2, self.pen, self.mu = X, y, float(sigma2), penalty, float(mu)
        self.d = X.shape[1]
        self.XtX = X.T @ X
        self.Xty = X.T @ y
        self.yty = float(y @ y)
        lo, hi = penalty.smoothing_gap_bound(mu)
        self.a_range = (math.exp(-self.d * hi), math.exp(-self.d * lo))

    def f(self, W):
        W2, single = _rows(W)
        q = np.einsum("ri,ij,rj->r", W2, self.XtX, W2) - 2 * W2 @ self.Xty + self.yty
        out = 0.5 * q / self.sigma2
        return out[0] if single else out

    def grad_f(self, W):
        W2, single = _rows(W)
        out = (W2 @ self.XtX - self.Xty) / self.sigma2
        return out[0] if single else out

    def U(self, W):
        W2, single = _rows(W)
        out = self.f(W2) + self.pen.value(W2).sum(axis=1)
        return out[0] if single else out

    def U0(self, W):
        W2, single = _rows(W)
        out = self.f(W2) + self.pen.smoothed(W2, self.mu).sum(axis=1)
        return out[0] if single else out

    def grad_U0(self, W):
        W2, single = _rows(W)
        out = self.grad_f(W2) + self.pen.smoothed_deriv(W2, self.mu)
        return out[0] if single else out

    def a(self, W):
        W2, single = _rows(W)
        out = np.exp(np.sum(self.pen.value(W2) - self.pen.smoothed(W2, self.mu), axis=1))
        return out[0] if single else out

    def grad_U0_and_a(self, W):
        """Shares the Phi/phi evaluations between grad U_0 and a (one `_pieces` call per step)."""
        W2, single = _rows(W)
        x, P, M, Q, _ = self.pen._pieces(W2, self.mu)
        c0, c1, c2 = self.pen.c[:, 0], self.pen.c[:, 1], self.pen.c[:, 2]
        xx = x[..., None]
        p0 = np.sum((c0 + c1 * xx + c2 * xx ** 2) * P + self.mu * (c1 + 2 * c2 * xx) * M
                    + c2 * self.mu ** 2 * Q, axis=-1)
        dp0 = np.sum((c1 + 2 * c2 * xx) * P + 2 * c2 * self.mu * M, axis=-1)
        G = self.grad_f(W2) + dp0
        A = np.exp(np.sum(self.pen.value(W2) - p0, axis=1))
        return (G[0], A[0]) if single else (G, A)

    def hess_U0(self, w):
        w = np.asarray(w, dtype=float)
        return self.XtX / self.sigma2 + np.diag(self.pen.smoothed_second_deriv(w, self.mu))


class L1Logistic(Potential):
    """Bayesian logistic regression with intercept prior N(0, sigma^2) and LASSO on w_1..w_{d-1}:
    U  = sum softplus(x.w) - y x.w + w_0^2/(2 sigma^2) + lam sum_{j>=1} |w_j|,
    U_0 replaces |w_j| by sqrt(w_j^2 + delta^2). a = exp(U - U_0) in [exp(-(d-1) lam delta), 1]."""

    def __init__(self, X: Array, y: Array, sigma: float, lam: float, delta: float):
        self.X, self.y = np.ascontiguousarray(X), y.astype(float)
        self.sigma, self.lam, self.delta = float(sigma), float(lam), float(delta)
        self.d = X.shape[1]
        self.a_range = (math.exp(-(self.d - 1) * lam * delta), 1.0)

    def _loglik(self, W2):
        z = W2 @ self.X.T
        return np.sum(np.logaddexp(0.0, z) - self.y[None, :] * z, axis=1)

    def U(self, W):
        W2, single = _rows(W)
        out = (self._loglik(W2) + W2[:, 0] ** 2 / (2 * self.sigma ** 2)
               + self.lam * np.sum(np.abs(W2[:, 1:]), axis=1))
        return out[0] if single else out

    def U0(self, W):
        W2, single = _rows(W)
        out = (self._loglik(W2) + W2[:, 0] ** 2 / (2 * self.sigma ** 2)
               + self.lam * np.sum(np.sqrt(W2[:, 1:] ** 2 + self.delta ** 2), axis=1))
        return out[0] if single else out

    def grad_U0(self, W):
        W2, single = _rows(W)
        g = (expit(W2 @ self.X.T) - self.y[None, :]) @ self.X
        g[:, 0] += W2[:, 0] / self.sigma ** 2
        g[:, 1:] += self.lam * W2[:, 1:] / np.sqrt(W2[:, 1:] ** 2 + self.delta ** 2)
        return g[0] if single else g

    def a(self, W):
        W2, single = _rows(W)
        gap = np.sum(np.sqrt(W2[:, 1:] ** 2 + self.delta ** 2) - np.abs(W2[:, 1:]), axis=1)
        out = np.exp(-self.lam * gap)
        return out[0] if single else out

    def hess_U0(self, w):
        w = np.asarray(w, dtype=float)
        p = expit(self.X @ w)
        H = (self.X * (p * (1 - p))[:, None]).T @ self.X
        H[0, 0] += 1 / self.sigma ** 2
        H[1:, 1:] += np.diag(self.lam * self.delta ** 2 / (w[1:] ** 2 + self.delta ** 2) ** 1.5)
        return H


class QuadraticClock(Potential):
    """Section 6.2: q = 1 + |x|^2, U_0 = beta log q, U = (beta+1) log q, a = q.
    Target pi ~ (1+|x|^2)^{-(beta+1)}, E[x x^T] = I/(2 beta - d).
    Prop. 6.7: for g_u = u.x, sigma^2_{u,alpha} = u^T (I - alpha^2 J^2)^{-1} u / (beta (2 beta - d))."""

    def __init__(self, d: int, beta: float):
        assert beta > d / 2
        self.d, self.beta = d, float(beta)

    def q(self, W):
        W2, single = _rows(W)
        out = 1.0 + np.sum(W2 ** 2, axis=1)
        return out[0] if single else out

    def U(self, W):
        return (self.beta + 1) * np.log(self.q(W))

    def U0(self, W):
        return self.beta * np.log(self.q(W))

    def grad_U0(self, W):
        W2, single = _rows(W)
        out = 2 * self.beta * W2 / self.q(W2)[:, None]
        return out[0] if single else out

    def a(self, W):
        return self.q(W)

    def second_moment(self) -> float:
        return 1.0 / (2 * self.beta - self.d)

    def radial_coefficient(self, W):
        """a grad U_0 = c(x) x with c = 2 beta (exactly linear drift)."""
        W2, _ = _rows(W)
        return np.full(W2.shape[0], 2 * self.beta)

    def sigma2_linear(self, u: Array, alpha: float, J: Array) -> float:
        M = np.eye(self.d) - alpha ** 2 * (J @ J)
        return float(u @ np.linalg.solve(M, u)) / (self.beta * (2 * self.beta - self.d))


class StudentClock(Potential):
    """Corollary 6.2: s = 1 + |x|^2/theta, q = s^zeta, beta = (d+theta)/(2 zeta) - 1, a = q,
    target = isotropic Student-t with theta degrees of freedom."""

    def __init__(self, d: int, theta: float, zeta: float):
        assert 1 < zeta < theta / 2
        self.d, self.theta, self.zeta = d, float(theta), float(zeta)
        self.beta = (d + theta) / (2 * zeta) - 1

    def s(self, W):
        W2, single = _rows(W)
        out = 1.0 + np.sum(W2 ** 2, axis=1) / self.theta
        return out[0] if single else out

    def U(self, W):
        return (self.beta + 1) * self.zeta * np.log(self.s(W))

    def U0(self, W):
        return self.beta * self.zeta * np.log(self.s(W))

    def grad_U0(self, W):
        W2, single = _rows(W)
        out = (2 * self.beta * self.zeta / self.theta) * W2 / self.s(W2)[:, None]
        return out[0] if single else out

    def a(self, W):
        return self.s(W) ** self.zeta

    def second_moment(self) -> float:
        return self.theta / (self.theta - 2)

    def radial_coefficient(self, W):
        """a grad U_0 = c(x) x with c = (2 beta zeta / theta) s^{zeta - 1}."""
        W2, _ = _rows(W)
        return (2 * self.beta * self.zeta / self.theta) * self.s(W2) ** (self.zeta - 1)


# ----------------------------------------------------------------------------------------------
# 3. Skew-symmetric fields
# ----------------------------------------------------------------------------------------------


def block_J2(d: int) -> Array:
    """Block diagonal with d//2 copies of [[0,-1],[1,0]]: J^2 = -I when d is even (Prop. 6.7)."""
    J = np.zeros((d, d))
    for k in range(d // 2):
        J[2 * k, 2 * k + 1] = -1.0
        J[2 * k + 1, 2 * k] = 1.0
    return J


def spectral_J(H: Array, strength: float = 1.0, pairing: str = "outer") -> Tuple[Array, Dict]:
    """Constant skew-symmetric J matched to a symmetric positive-definite curvature matrix H.

    Let H = Q diag(l_1 <= ... <= l_d) Q^T.  For a 2x2 block with curvatures (lo, hi) the drift
    matrix (I + alpha J2) diag(lo, hi) has trace lo+hi and determinant lo*hi*(1+alpha^2); its two
    eigenvalues share the real part (lo+hi)/2 as soon as

            alpha >= alpha* := (hi - lo) / (2 sqrt(lo * hi)).

    With `strength` = 1 each pair uses alpha*, so the spectral gap of the reference dynamics
    -(I + J) H x + sqrt(2) dW rises from l_1 to min_pairs (lo + hi)/2 >= l_d / 2, while the
    spectral radius of each block *falls* from hi to (lo+hi)/2 (Euler becomes more stable, not
    less).  pairing='outer' pairs l_1 with l_d, l_2 with l_{d-1}, ...; the median is left alone
    when d is odd.  Returns J in the original coordinates and diagnostics."""
    H = 0.5 * (np.asarray(H, float) + np.asarray(H, float).T)
    lam, Q = np.linalg.eigh(H)
    assert lam.min() > 0, "H must be positive definite"
    d = len(lam)
    Je = np.zeros((d, d))
    alphas = []
    if pairing == "outer":
        pairs = [(i, d - 1 - i) for i in range(d // 2)]
    elif pairing == "adjacent":
        pairs = [(2 * i, 2 * i + 1) for i in range(d // 2)]
    else:
        raise ValueError(pairing)
    for i, j in pairs:
        lo, hi = lam[i], lam[j]
        alpha = strength * (hi - lo) / (2 * math.sqrt(lo * hi))
        Je[i, j] = -alpha
        Je[j, i] = alpha
        alphas.append(alpha)
    J = Q @ Je @ Q.T
    B = (np.eye(d) + J) @ H
    ev = np.linalg.eigvals(B)
    info = dict(eig_H=lam, alphas=np.array(alphas), pairs=pairs,
                gap_rev=float(lam.min()), gap_nrev=float(ev.real.min()),
                radius_rev=float(lam.max()), radius_nrev=float(np.abs(ev).max()),
                cond_H=float(lam.max() / lam.min()))
    return J, info


# ----------------------------------------------------------------------------------------------
# 4. Geometry for the constrained case (ball) and the stream current
# ----------------------------------------------------------------------------------------------


@dataclass
class Ball:
    """K = {|w|^2 <= r2}. Projection is radial. Normal n = w/|w|."""
    r2: float

    @property
    def radius(self) -> float:
        return math.sqrt(self.r2)

    def feasible(self, W: Array, tol: float = 1e-10) -> Array:
        return np.sum(W ** 2, axis=-1) <= self.r2 * (1 + tol)

    def project(self, W: Array) -> Tuple[Array, Array]:
        n2 = np.sum(W ** 2, axis=-1)
        over = n2 > self.r2
        out = W.copy()
        out[over] *= (self.radius / np.sqrt(n2[over]))[:, None]
        return out, over

    def phi(self, W: Array) -> Array:
        """Defining function phi = (|w|^2 - r2)/2: phi = 0 on dK, phi < 0 inside."""
        return 0.5 * (np.sum(W ** 2, axis=-1) - self.r2)

    def grad_phi(self, W: Array) -> Array:
        return W


def stream_current(pot: Potential, W: Array, G: Array, A: Array, J: Array,
                   geom: Optional[Ball]) -> Array:
    """The circulation c = e^U J grad(psi) for the stream potential

        unconstrained:  psi = e^{-U_0}                  =>  c = -a J grad U_0            (2.9)
        ball:           psi = e^{-U_0} phi,  phi|dK = 0  =>  c = a J (grad phi - phi grad U_0)

    The second choice satisfies Lemma 2.3(i) (psi constant on dK), so c.n = 0 for *any*
    constant skew J, which is what lets a spectrally matched J be used with normal reflection.
    G = grad U_0(W), A = a(W). Returns (R, d)."""
    if geom is None:
        V = -G
    else:
        V = geom.grad_phi(W) - geom.phi(W)[:, None] * G
    return A[:, None] * (V @ J.T)


def cross_current(pot: Potential, W: Array, G: Array, A: Array, scale: float,
                  blocks: Sequence[Sequence[int]]) -> Array:
    """The notebook's / Remark 2.4(iii) state-dependent tensor on the ball: J(w) v = s w_I x v_I
    blockwise over index triples, with the canonical psi = e^{-U_0}: c = -a J(w) grad U_0."""
    out = np.zeros_like(G)
    for blk in blocks:
        sl = list(blk)
        out[:, sl] = -np.cross(scale * W[:, sl], G[:, sl])
    return A[:, None] * out


# ----------------------------------------------------------------------------------------------
# 5. Samplers
# ----------------------------------------------------------------------------------------------


@dataclass
class RunResult:
    traj: Array            # (n_saved, R, d) thinned states (including the initial state)
    steps: Array           # step index of each saved state
    proj_rate: float       # fraction of (step, replicate) pairs that were projected
    n_nonfinite: int
    runtime: float
    extra: Dict


def run_nald(pot: Potential, W0: Array, *, eta: float, n_iter: int, alpha: float = 0.0,
             J: Optional[Array] = None, current: str = "stream", geom: Optional[Ball] = None,
             cross_scale: float = 0.0, cross_blocks: Sequence[Sequence[int]] = (),
             seed: int = 0, thin: int = 10, a_clip: Optional[float] = None,
             integrator: str = "euler") -> RunResult:
    """(Projected) Euler-Maruyama for anchored Langevin (alpha = 0) or its non-reversible version.

        W <- Pi_K[ W - eta a grad U_0 + eta alpha c(W) + sqrt(2 eta a) xi ]

    Vectorised over R replicates. Re-using `seed` gives the same Gaussian stream for both arms,
    so paired comparisons are a genuine common-random-numbers coupling.

    integrator="exp" (only for potentials with a radial reference drift a grad U_0 = c(x) x, i.e.
    the two heavy-tailed clocks) freezes c at x_k and applies the drift flow exactly,
        W <- exp(-eta c(W) (I + alpha J)) W + sqrt(2 eta a) xi,
    so that the circulation is a genuine rotation at every step instead of its Euler chord.  This
    is the remedy for the stiffness noted in Remark 5.4: the Euler chord of a rotation inflates
    |x| by a factor sqrt(1 + (eta c alpha)^2) per step, which in a heavy tail (where a, hence c*eta,
    is large) is what destroys the radial observables."""
    import time
    W = np.array(W0, dtype=float, copy=True)
    R, d = W.shape
    rng = np.random.default_rng(seed)
    if geom is not None:
        assert np.all(geom.feasible(W)), "initial states must lie in K"
    saved = list(range(0, n_iter + 1, thin))
    if saved[-1] != n_iter:
        saved.append(n_iter)
    traj = np.empty((len(saved), R, d))
    traj[0] = W
    si = 1
    n_proj = 0
    n_bad = 0
    t0 = time.perf_counter()
    if integrator == "exp":
        assert geom is None and current == "stream" and hasattr(pot, "radial_coefficient")
        Jm = np.zeros((d, d)) if (J is None or alpha == 0.0) else alpha * J
        # exp(-t (I + alpha J)) = e^{-t} exp(-t alpha J); precompute the rotation on a grid of t is
        # unnecessary: for a block J_2 structure exp(-t alpha J) is a plane rotation, and in general
        # we use the eigen-decomposition of the normal matrix alpha J (skew => unitary diagonalisable).
        evals, evecs = np.linalg.eig(Jm)               # purely imaginary eigenvalues
        evecs_inv_T = np.linalg.inv(evecs).T           # eigenvectors need not be orthonormal (repeated eigenvalues)
    for k in range(1, n_iter + 1):
        G, A = pot.grad_U0_and_a(W)
        if a_clip is not None:
            A = np.minimum(A, a_clip)
        if integrator == "exp":
            c = pot.radial_coefficient(W)                              # (R,)
            t = eta * c                                                # (R,)
            # rows: W_new = exp(-t (I + Jm)) W = e^{-t} (evecs diag(e^{-t lam}) evecs^H) W
            Wc = W.astype(complex) @ evecs_inv_T                       # (R, d) in eigenbasis
            Wc = Wc * np.exp(-t[:, None] * (1.0 + evals[None, :]))
            drift_flow = np.real(Wc @ evecs.T)                         # rows of  V diag(.) V^{-1} w
            prop = drift_flow + np.sqrt(2.0 * eta * A)[:, None] * rng.standard_normal((R, d))
        else:
            drift = -A[:, None] * G
            if alpha != 0.0:
                if current == "stream":
                    drift = drift + alpha * stream_current(pot, W, G, A, J, geom)
                elif current == "cross":
                    drift = drift + alpha * cross_current(pot, W, G, A, cross_scale, cross_blocks)
                else:
                    raise ValueError(current)
            prop = W + eta * drift + np.sqrt(2.0 * eta * A)[:, None] * rng.standard_normal((R, d))
        bad = ~np.isfinite(prop).all(axis=1)
        if bad.any():
            n_bad += int(bad.sum())
            prop[bad] = W[bad]
        if geom is not None:
            W, over = geom.project(prop)
            n_proj += int(over.sum())
        else:
            W = prop
        if si < len(saved) and k == saved[si]:
            traj[si] = W
            si += 1
    return RunResult(traj=traj, steps=np.asarray(saved), proj_rate=n_proj / (n_iter * R),
                     n_nonfinite=n_bad, runtime=time.perf_counter() - t0, extra={})


def anchored_mala(pot: Potential, W0: Array, *, eta: float, n_iter: int, geom: Optional[Ball] = None,
                  seed: int = 0, thin: int = 10) -> RunResult:
    """Metropolis-adjusted anchored Langevin: proposal y ~ N(x - eta a(x) grad U_0(x), 2 eta a(x) I),
    accepted with the exact Metropolis-Hastings ratio for pi ~ e^{-U} 1_K.  Proposals outside K are
    rejected.  This is an *exact* sampler for the same target both Euler arms approximate, and is
    used only to build reference samples.  Requires U to be evaluated (it is, cheaply)."""
    import time
    W = np.array(W0, dtype=float, copy=True)
    R, d = W.shape
    rng = np.random.default_rng(seed)
    saved = list(range(0, n_iter + 1, thin))
    if saved[-1] != n_iter:
        saved.append(n_iter)
    traj = np.empty((len(saved), R, d))
    traj[0] = W
    si = 1
    Ux = pot.U(W)
    Gx = pot.grad_U0(W)
    Ax = pot.a(W)
    acc = 0
    t0 = time.perf_counter()
    for k in range(1, n_iter + 1):
        mx = W - eta * Ax[:, None] * Gx
        Y = mx + np.sqrt(2.0 * eta * Ax)[:, None] * rng.standard_normal((R, d))
        Uy = pot.U(Y)
        Gy = pot.grad_U0(Y)
        Ay = pot.a(Y)
        my = Y - eta * Ay[:, None] * Gy
        log_q_xy = -np.sum((Y - mx) ** 2, axis=1) / (4 * eta * Ax) - 0.5 * d * np.log(Ax)
        log_q_yx = -np.sum((W - my) ** 2, axis=1) / (4 * eta * Ay) - 0.5 * d * np.log(Ay)
        log_r = -Uy + Ux + log_q_yx - log_q_xy
        if geom is not None:
            log_r = np.where(geom.feasible(Y), log_r, -np.inf)
        accept = np.log(rng.random(R)) < log_r
        acc += int(accept.sum())
        W = np.where(accept[:, None], Y, W)
        Ux = np.where(accept, Uy, Ux)
        Gx = np.where(accept[:, None], Gy, Gx)
        Ax = np.where(accept, Ay, Ax)
        if si < len(saved) and k == saved[si]:
            traj[si] = W
            si += 1
    return RunResult(traj=traj, steps=np.asarray(saved), proj_rate=0.0, n_nonfinite=0,
                     runtime=time.perf_counter() - t0, extra=dict(accept_rate=acc / (n_iter * R)))


# ----------------------------------------------------------------------------------------------
# 6. Metrics
# ----------------------------------------------------------------------------------------------


def running_mean(traj: Array, burn_saved: int = 0) -> Array:
    """Running time-average of the states after discarding the first `burn_saved` saved frames.
    traj (T, R, d) -> (T - burn_saved, R, d)."""
    x = traj[burn_saved:]
    return np.cumsum(x, axis=0) / np.arange(1, x.shape[0] + 1)[:, None, None]


def mse_curve(traj: Array, ref_mean: Array, ref_var: Array, burn_saved: int = 0) -> Array:
    """Normalised MSE of the running-mean estimator of E_pi[w], averaged over replicates and
    coordinates: (1/d) sum_i E_r (m_{r,t,i} - mu_i)^2 / var_i.  Includes bias and variance."""
    rm = running_mean(traj, burn_saved)
    return np.mean((rm - ref_mean[None, None, :]) ** 2 / ref_var[None, None, :], axis=(1, 2))


def energy_distance(A: Array, B: Array, max_n: int = 4000, seed: int = 0) -> float:
    """Energy distance 2E|X-Y| - E|X-X'| - E|Y-Y'| between two point clouds (Euclidean)."""
    rng = np.random.default_rng(seed)
    if len(A) > max_n:
        A = A[rng.choice(len(A), max_n, replace=False)]
    if len(B) > max_n:
        B = B[rng.choice(len(B), max_n, replace=False)]

    def mean_dist(X, Y):
        d2 = np.sum(X ** 2, 1)[:, None] + np.sum(Y ** 2, 1)[None, :] - 2 * X @ Y.T
        return float(np.mean(np.sqrt(np.maximum(d2, 0.0))))

    return 2 * mean_dist(A, B) - mean_dist(A, A) - mean_dist(B, B)


def energy_curve(traj: Array, ref: Array, whiten: Optional[Array] = None, every: int = 1) -> Tuple[Array, Array]:
    """Energy distance between the replicate cloud at each saved frame and the reference sample."""
    idx = np.arange(0, traj.shape[0], every)
    if whiten is not None:
        ref = ref @ whiten
    out = np.empty(len(idx))
    for n, t in enumerate(idx):
        X = traj[t] if whiten is None else traj[t] @ whiten
        out[n] = energy_distance(X, ref, seed=t)
    return idx, out


def ess_from_replicates(traj: Array, ref_var: Array, burn_saved: int, thin: int) -> Dict[str, Array]:
    """Asymptotic-variance and effective-sample-size estimates for the time averages of each
    coordinate, from R *independent* replicate chains:

        sigma2_disc_i = n Var_r( mean_t w_{r,t,i} )       (per Euler step)
        ESS_i         = n Var_pi(w_i) / sigma2_disc_i      (per n steps)

    where n is the number of Euler steps in the averaging window.  Because the R replicates are
    independent, no autocorrelation modelling is needed; the estimate's own relative standard
    error is about sqrt(2/(R-1)).  Also returns the batch-means estimate from within each chain
    as a cross-check (averaged over replicates)."""
    x = traj[burn_saved:]                          # (T, R, d)
    T, R, d = x.shape
    n_steps = (T - 1) * thin + 1
    tavg = x.mean(axis=0)                          # (R, d)
    var_of_mean = tavg.var(axis=0, ddof=1)         # (d,)
    sigma2_disc = n_steps * var_of_mean
    ess = n_steps * ref_var / sigma2_disc
    # batch means within each chain
    nb = max(10, int(math.sqrt(T)))
    bl = T // nb
    bm = x[: nb * bl].reshape(nb, bl, R, d).mean(axis=1)       # (nb, R, d)
    sigma2_bm = (bl * thin) * bm.var(axis=0, ddof=1)           # per step, (R, d)
    ess_bm = n_steps * ref_var / sigma2_bm.mean(axis=0)
    bias = tavg.mean(axis=0)                                   # mean over replicates (compare with ref)
    return dict(sigma2=sigma2_disc, ess=ess, ess_per_step=ess / n_steps, ess_bm=ess_bm,
                mean=bias, n_steps=n_steps)


def paired_stats(a: Array, b: Array) -> Tuple[float, float, float]:
    """mean, s.e., t of the paired difference a - b."""
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / math.sqrt(len(d)) + 1e-300
    return float(d.mean()), float(se), float(d.mean() / se)


# ----------------------------------------------------------------------------------------------
# 7. Data
# ----------------------------------------------------------------------------------------------


def make_regression_data(n: int, d: int, rho: float, n_nonzero: int, sigma: float, seed: int,
                         signal: float = 2.0) -> Dict[str, Array]:
    """Correlated Gaussian design (AR(1) correlation rho), sparse truth, Gaussian noise."""
    rng = np.random.default_rng(seed)
    Sigma = rho ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((n, d)) @ L.T
    theta = np.zeros(d)
    idx = rng.choice(d, n_nonzero, replace=False)
    theta[idx] = signal * rng.choice([-1.0, 1.0], n_nonzero) * rng.uniform(0.5, 1.5, n_nonzero)
    y = X @ theta + sigma * rng.standard_normal(n)
    return dict(X=X, y=y, theta=theta, Sigma=Sigma)


def load_titanic(path: str, seed: int = 2027, test_frac: float = 0.2) -> Dict[str, Array]:
    """Nine features (Age, SibSp, Parch, Fare, female, Pclass==2, Pclass==3, Embarked==Q,
    Embarked==S), training-median imputation, training-fitted standardisation, stratified split.
    Returns design matrices *with* a leading column of ones (d = 10)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(path)
    F = pd.DataFrame({
        "Age": df["Age"], "SibSp": df["SibSp"], "Parch": df["Parch"], "Fare": df["Fare"],
        "is_female": (df["Sex"] == "female").astype(float),
        "Pclass_2": (df["Pclass"] == 2).astype(float), "Pclass_3": (df["Pclass"] == 3).astype(float),
        "Embarked_Q": (df["Embarked"] == "Q").astype(float), "Embarked_S": (df["Embarked"] == "S").astype(float),
    })
    y = df["Survived"].to_numpy().astype(float)
    Xtr, Xte, ytr, yte = train_test_split(F, y, test_size=test_frac, random_state=seed, stratify=y)
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    mu, sd = Xtr.mean(), Xtr.std(ddof=0).replace(0, 1.0)
    Xtr, Xte = ((Xtr - mu) / sd).to_numpy(), ((Xte - mu) / sd).to_numpy()
    add1 = lambda X: np.column_stack([np.ones(len(X)), X])
    return dict(X_train=add1(Xtr), y_train=ytr, X_test=add1(Xte), y_test=yte,
                feature_names=["intercept"] + list(F.columns))


# ----------------------------------------------------------------------------------------------
# 8. Self-checks
# ----------------------------------------------------------------------------------------------


def self_test(verbose: bool = True) -> Dict[str, float]:
    """Numerical verification of every identity the samplers rely on."""
    rng = np.random.default_rng(0)
    out: Dict[str, float] = {}
    # 8.1 Lemma 7.1 against Monte Carlo smoothing, for all three penalties
    Z = rng.standard_normal(400_000)
    for pen in (lasso_penalty(1.3), mcp_penalty(0.8, 3.0), scad_penalty(0.7, 3.7)):
        mu = 0.35
        xs = np.array([-3.0, -1.2, -0.4, 0.0, 0.3, 0.9, 2.5])
        mc = np.array([pen.value(x + mu * Z).mean() for x in xs])
        mc_d = np.array([pen.deriv(x + mu * Z).mean() for x in xs])
        err = np.max(np.abs(pen.smoothed(xs, mu) - mc))
        err_d = np.max(np.abs(pen.smoothed_deriv(xs, mu) - mc_d))
        h = 1e-4
        fd2 = (pen.smoothed(xs + h, mu) - 2 * pen.smoothed(xs, mu) + pen.smoothed(xs - h, mu)) / h ** 2
        fd1 = (pen.smoothed(xs + h, mu) - pen.smoothed(xs - h, mu)) / (2 * h)
        err_fd1 = np.max(np.abs(fd1 - pen.smoothed_deriv(xs, mu)))
        err_fd2 = np.max(np.abs(fd2 - pen.smoothed_second_deriv(xs, mu)))
        out[f"smooth_mc[{pen.name}]"] = float(err)
        out[f"smooth_mc_deriv[{pen.name}]"] = float(err_d)
        out[f"smooth_fd1[{pen.name}]"] = float(err_fd1)
        out[f"smooth_fd2[{pen.name}]"] = float(err_fd2)
        assert err < 5e-3 and err_d < 5e-3, pen.name
        assert err_fd1 < 1e-6 and err_fd2 < 1e-4, pen.name
    # Lasso special case (7.10)
    lam, mu = 1.7, 0.2
    pen = lasso_penalty(lam)
    x = np.linspace(-2, 2, 41)
    p0_closed = lam * (x * (2 * ndtr(x / mu) - 1) + 2 * mu * norm.pdf(x / mu))
    out["lasso_7.10"] = float(np.max(np.abs(pen.smoothed(x, mu) - p0_closed)))
    assert out["lasso_7.10"] < 1e-12
    # 8.2 gradient of U_0 vs finite differences, a = exp(U - U_0), Hessian
    data = make_regression_data(60, 8, 0.6, 3, 1.0, 1)
    pot = PenalisedRegression(data["X"], data["y"], 1.0, scad_penalty(2.0, 3.7), 0.1)
    w = rng.standard_normal(8) * 0.5
    g = pot.grad_U0(w)
    fd = np.array([(pot.U0(w + h * e) - pot.U0(w - h * e)) / (2 * h)
                   for e in np.eye(8) for h in [1e-6]])
    out["grad_U0_relerr"] = float(np.max(np.abs(g - fd)) / np.max(np.abs(g)))
    assert out["grad_U0_relerr"] < 1e-6
    W = rng.standard_normal((50, 8))
    G2, A2 = pot.grad_U0_and_a(W)
    out["grad_a_combined"] = float(np.max(np.abs(G2 - pot.grad_U0(W))) + np.max(np.abs(A2 - pot.a(W))))
    assert out["grad_a_combined"] < 1e-9
    out["a_identity"] = float(np.max(np.abs(pot.a(W) - np.exp(pot.U(W) - pot.U0(W)))))
    assert out["a_identity"] < 1e-9
    Hfd = Potential.hess_U0(pot, w)
    out["hess_relerr"] = float(np.max(np.abs(Hfd - pot.hess_U0(w))) / np.max(np.abs(Hfd)))
    assert out["hess_relerr"] < 1e-5
    # 8.3 spectral J: skew, real parts equalised, radius reduced
    H = np.diag([1.0, 3.0, 10.0, 100.0]) ; Qr, _ = np.linalg.qr(rng.standard_normal((4, 4))); H = Qr @ H @ Qr.T
    J, info = spectral_J(H)
    out["J_skew"] = float(np.max(np.abs(J + J.T)))
    ev = np.linalg.eigvals((np.eye(4) + J) @ H)
    out["J_gap_nrev"] = info["gap_nrev"]
    assert out["J_skew"] < 1e-12
    assert abs(info["gap_nrev"] - (1.0 + 100.0) / 2 * 0 - min((1 + 100) / 2, (3 + 10) / 2)) < 1e-6
    assert info["radius_nrev"] < info["radius_rev"]
    # 8.4 stream current on the ball is tangential for a constant J and grad(psi) is what we say
    ball = Ball(2.0)
    Wb = rng.standard_normal((30, 8)); Wb *= ball.radius / np.linalg.norm(Wb, axis=1, keepdims=True)
    Jb, _ = spectral_J(pot.hess_U0(np.zeros(8)) + np.eye(8))
    c = stream_current(pot, Wb, pot.grad_U0(Wb), pot.a(Wb), Jb, ball)
    out["stream_c_dot_n"] = float(np.max(np.abs(np.sum(c * Wb, axis=1)) / np.linalg.norm(c, axis=1)))
    assert out["stream_c_dot_n"] < 1e-12
    # grad psi check: psi = e^{-U0} phi, c = e^U J grad psi
    w1 = Wb[0] * 0.7
    psi = lambda v: math.exp(-float(pot.U0(v))) * float(ball.phi(v))
    gpsi = np.array([(psi(w1 + 1e-6 * e) - psi(w1 - 1e-6 * e)) / 2e-6 for e in np.eye(8)])
    c_direct = math.exp(float(pot.U(w1))) * (Jb @ gpsi)
    c_code = stream_current(pot, w1[None], pot.grad_U0(w1[None]), pot.a(w1[None]), Jb, ball)[0]
    out["stream_identity_relerr"] = float(np.max(np.abs(c_direct - c_code)) / np.max(np.abs(c_code)))
    assert out["stream_identity_relerr"] < 1e-5
    # divergence-free weighted current in the unconstrained canonical case: div(pi c) = 0 where
    # pi c ~ -J grad e^{-U0}; check numerically div(J grad e^{-U0}) = 0 for constant skew J
    f = lambda v: math.exp(-float(pot.U0(v)) + float(pot.U0(np.zeros(8))))
    div = 0.0
    h = 1e-4
    for i in range(8):
        e = np.zeros(8); e[i] = h
        gp = lambda v: np.array([(f(v + 1e-5 * ee) - f(v - 1e-5 * ee)) / 2e-5 for ee in np.eye(8)])
        div += ((Jb @ gp(w1 + e))[i] - (Jb @ gp(w1 - e))[i]) / (2 * h)
    out["div_J_grad_psi"] = float(abs(div) / (np.linalg.norm(Jb @ gp(w1)) + 1e-300))
    assert out["div_J_grad_psi"] < 1e-3
    # 8.5 anchored MALA is exact: 1-d Laplace target through the Lasso potential (no data)
    pen1 = lasso_penalty(1.0)
    pot1 = PenalisedRegression(np.zeros((1, 1)), np.zeros(1), 1.0, pen1, 0.3)
    r = anchored_mala(pot1, np.zeros((2000, 1)), eta=0.5, n_iter=3000, seed=3, thin=10)
    smp = r.traj[100:].reshape(-1)
    out["mala_laplace_E|x|"] = float(abs(np.mean(np.abs(smp)) - 1.0))
    out["mala_laplace_Ex2"] = float(abs(np.mean(smp ** 2) - 2.0))
    assert out["mala_laplace_E|x|"] < 0.03 and out["mala_laplace_Ex2"] < 0.1
    # 8.6 quadratic clock: invariance of the second moment under Euler for both arms (small eta)
    qc = QuadraticClock(4, 4.0)
    J2 = block_J2(4)
    for al in (0.0, 1.5):
        rr = run_nald(qc, np.zeros((4000, 4)), eta=2e-3, n_iter=4000, alpha=al, J=J2, seed=5, thin=40)
        m2 = np.mean(rr.traj[50:] ** 2)
        out[f"qclock_m2_relerr[alpha={al}]"] = float(abs(m2 - qc.second_moment()) / qc.second_moment())
        assert out[f"qclock_m2_relerr[alpha={al}]"] < 0.08
    if verbose:
        for k, v in out.items():
            print(f"  {k:40s} {v:.3e}")
        print("all self-tests passed")
    return out


if __name__ == "__main__":
    self_test()
