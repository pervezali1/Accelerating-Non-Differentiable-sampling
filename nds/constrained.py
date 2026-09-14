r"""Constrained Bayesian logistic regression, sampled with skew-reflected SGLD.

The target is the posterior of a logistic regression restricted to a centred
ball ``K_r = {x : |x| <= r}``,

    pi(x)  \propto  exp(-U(x)) 1_{K_r}(x),
    U(x)   =  sum_i softplus(x . a_i) - y_i (x . a_i),

i.e. a uniform prior on ``K_r`` and no other regularisation: the constraint *is*
the prior.  Two samplers run on it, both with stochastic (mini-batch) gradients
and both projected back into ``K_r`` at every step:

* **PSGLD** -- projected stochastic-gradient Langevin dynamics, ``J = 0``:

      y = x - h ghat(x) + sqrt(2h) xi,     x' = Proj_{K_r}(y).

* **SRNSGLD** -- skew-reflected non-reversible SGLD, ``J != 0``:

      y = x - h (I + J(x)) ghat(x) + div J(x) h + sqrt(2h) xi,
      x' = y - lambda gamma(x_b),   gamma = (I + J(x_b)) n(x_b),

  where ``n`` is the outward unit normal at the radial boundary point ``x_b``
  and ``lambda >= 0`` is the smallest push along ``gamma`` that returns ``y`` to
  the sphere.  The reflection has to be *oblique* like this, not normal: with
  ``J != 0`` the stationary probability current has a tangential component, and
  reflecting along ``n`` would let it leak through the boundary and change the
  invariant law.  ``gamma`` always points inward, because ``n . J n = 0`` for a
  skew ``J`` makes ``n . gamma = 1``.

Two skew fields are used, both in ``d = 9``:

* the constant ``J_a``, superdiagonal ``+a`` and subdiagonal ``-a``, which
  couples the coordinates in one chain ``x_1 - x_2 - ... - x_9``;
* the state-dependent ``J_s(x)``, block-diagonal with three ``3 x 3`` blocks,
  the ``k``-th depending only on its own coordinate triple:

      J_s(x) = diag( J1(x_1,x_2,x_3), J2(x_4,x_5,x_6), J3(x_7,x_8,x_9) ),
      Jk(u,v,w) = a * [[0, w, -v], [-w, 0, u], [v, -u, 0]].

Each block is ``a`` times the hat map of its triple, so ``Jk(z) g = a (g x z)``.
Two properties come for free and are what make the block form the right choice
under a ball constraint:

1. ``div J_s = 0`` exactly -- row ``i`` of the divergence is
   ``sum_j d_j (J_s)_ij``, and every entry of a hat map is independent of the
   coordinate it is differentiated by.  So the Ma-Chen-Fox correction term
   vanishes and the drift is just ``-(I + J_s(x)) ghat(x)``, with no derivative
   of ``J`` to estimate.  (The same holds trivially for the constant ``J_a``.)
2. ``J_s(x) x = 0`` -- each block rotates its triple about itself, so the field
   is tangential to every centred sphere.  At the boundary ``x_b = r n`` this
   gives ``J_s(x_b) n = 0`` and hence ``gamma = n``: for this field skew
   reflection *is* plain projection, and SRNSGLD needs no boundary machinery
   beyond what PSGLD already has.  The constant ``J_a`` does need it.
"""

from __future__ import annotations

import numpy as np

from .target import _sigmoid, softplus


# --------------------------------------------------------------------- fields
def superdiagonal_skew(d: int, amplitude: float = 1.0) -> np.ndarray:
    """``J_a``: ``+a`` on the whole superdiagonal, ``-a`` on the subdiagonal."""
    J = np.zeros((d, d))
    i = np.arange(d - 1)
    J[i, i + 1] = amplitude
    J[i + 1, i] = -amplitude
    return J


class ZeroField:
    """``J = 0``: the reversible, projected chain."""

    label = "$J = 0$ (PSGLD)"
    key = "zero"

    def __init__(self, d: int) -> None:
        self.d = d
        self.amplitude = 0.0

    def apply(self, W: np.ndarray, G: np.ndarray) -> np.ndarray:
        return np.zeros_like(G)

    def divergence(self, W: np.ndarray) -> np.ndarray:
        return np.zeros_like(W)

    def matrix(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((self.d, self.d))


class ConstantField:
    """``J_a``, constant, tridiagonal skew: superdiagonal ``+a``, sub ``-a``."""

    label = "constant $J_a$ (SRNSGLD)"
    key = "constant"

    def __init__(self, d: int, amplitude: float = 1.0) -> None:
        self.d = d
        self.amplitude = float(amplitude)
        self.J = superdiagonal_skew(d, self.amplitude)

    def apply(self, W: np.ndarray, G: np.ndarray) -> np.ndarray:
        return self.J @ G

    def divergence(self, W: np.ndarray) -> np.ndarray:
        return np.zeros_like(W)  # constant field

    def matrix(self, x: np.ndarray) -> np.ndarray:
        return self.J


class BlockHatField:
    """``J_s(x)``: block diagonal, each ``3 x 3`` block the hat map of its triple.

    ``J_s(x) g`` is computed blockwise as ``a (g x z)`` with ``z`` the block's
    own coordinates, so a step costs one cross product per block -- ``O(d)``,
    not ``O(d^2)``, and no derivative of ``J`` at all since ``div J_s = 0``.
    """

    label = "state-dependent $J_s$ (SRNSGLD)"
    key = "state"

    def __init__(self, d: int, amplitude: float = 1.0) -> None:
        if d % 3:
            raise ValueError(f"block hat field needs d divisible by 3, got {d}")
        self.d = d
        self.n_blocks = d // 3
        self.amplitude = float(amplitude)

    def apply(self, W: np.ndarray, G: np.ndarray) -> np.ndarray:
        out = np.empty_like(G)
        for b in range(self.n_blocks):
            s = slice(3 * b, 3 * b + 3)
            # J(z) g = a (g x z), cross products taken down the coordinate axis
            out[s] = self.amplitude * np.cross(G[s].T, W[s].T).T
        return out

    def divergence(self, W: np.ndarray) -> np.ndarray:
        return np.zeros_like(W)  # every hat-map entry is free of the coordinate
                                 # it would be differentiated by

    def matrix(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float).ravel()
        J = np.zeros((self.d, self.d))
        for b in range(self.n_blocks):
            u, v, w = x[3 * b : 3 * b + 3]
            J[3 * b : 3 * b + 3, 3 * b : 3 * b + 3] = self.amplitude * np.array(
                [[0.0, w, -v], [-w, 0.0, u], [v, -u, 0.0]]
            )
        return J


class FramedField:
    r"""The same field, read in a rotated orthonormal frame: ``V J(V^T x) V^T``.

    Which coordinates share a block is a modelling choice, and the feature order
    a dataset happens to arrive in is not obviously the right one.  Conjugating
    by an orthogonal ``V`` keeps everything that made the block form usable:
    the matrix stays skew, the divergence stays zero (a divergence transforms
    covariantly, so ``div (V J(V^T x) V^T) = V (div J)(V^T x) = 0``), the
    tangency ``J(x) x = 0`` survives because ``V J(u) u = 0`` at ``u = V^T x``,
    and the constraint set is a ball, which is invariant under ``V``.  So the
    boundary behaviour is unchanged and only the pairing of directions differs.
    """

    def __init__(self, field, V: np.ndarray, label: str | None = None) -> None:
        self.field = field
        self.V = np.ascontiguousarray(V, dtype=np.float64)
        self.d = field.d
        self.amplitude = field.amplitude
        self.key = field.key
        self.label = label or field.label + ", eigen frame"

    def apply(self, W: np.ndarray, G: np.ndarray) -> np.ndarray:
        return self.V @ self.field.apply(self.V.T @ W, self.V.T @ G)

    def divergence(self, W: np.ndarray) -> np.ndarray:
        return self.V @ self.field.divergence(self.V.T @ W)

    def matrix(self, x: np.ndarray) -> np.ndarray:
        u = self.V.T @ np.asarray(x, float).ravel()
        return self.V @ self.field.matrix(u) @ self.V.T


def eigen_frame(H: np.ndarray) -> np.ndarray:
    r"""Eigenvectors of ``H``, ordered so every ``3``-block mixes slow with fast.

    The relaxation rates of the reversible dynamics are the eigenvalues of
    ``H``, and a skew term can only move rate between directions -- it cannot
    add any, since ``trace((I + J) H) = trace H``.  So a block that contains
    three slow directions has nothing to trade, and a block that contains the
    slowest and the fastest has the most.  Ascending eigenvalues are split into
    thirds (slow, middle, fast) and dealt out one per block, fastest first.
    """
    ev, V = np.linalg.eigh(H)
    d = len(ev)
    k = d // 3
    slow, mid, fast = np.arange(k), np.arange(k, 2 * k), np.arange(2 * k, d)[::-1]
    order = np.concatenate([[slow[b], fast[b], mid[b]] for b in range(k)])
    return V[:, order]


FIELDS = {"zero": ZeroField, "constant": ConstantField, "state": BlockHatField}


def field_from_rho(key: str, d: int, rho: float = 1.0, radius: float = 2.0):
    r"""Build a field whose operator norm is ``rho``, so the two are comparable.

    The two fields have different natural scales, and comparing them at a shared
    ``a`` would compare different rotation strengths.  Both norms are known in
    closed form:

    * the skew tridiagonal ``J_a`` has eigenvalues ``2 i a cos(k pi / (d+1))``,
      so ``|J_a| = 2 a cos(pi / (d+1))`` -- ``1.902 a`` at ``d = 9``;
    * a hat map of ``z`` has eigenvalues ``0, +- i |z|``, so
      ``|J_s(x)| = a max_k |z_k|``; averaged over the sphere of radius ``r``
      that is ``a r c_d`` with ``c_d = E max_k |z_k| / r`` a purely geometric
      constant (``0.747`` at ``d = 9``, against ``1/sqrt(3) = 0.577`` for a
      single block), evaluated once by Monte Carlo with a fixed seed.

    So ``rho`` is the size of the rotational part of the drift relative to the
    gradient part: ``rho = 1`` rotates as hard as it descends.
    """
    if key == "zero":
        return ZeroField(d)
    if key == "constant":
        return ConstantField(d, rho / (2.0 * np.cos(np.pi / (d + 1))))
    if key == "state":
        return BlockHatField(d, rho / (radius * _mean_block_norm(d)))
    raise KeyError(f"unknown field {key!r}; have {sorted(FIELDS)}")


def _mean_block_norm(d: int, n_draws: int = 200000, seed: int = 0) -> float:
    """``E max_k |z_k|`` for a point uniform on the unit sphere of ``R^d``."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((d, n_draws))
    Z /= np.sqrt((Z * Z).sum(axis=0))
    blocks = np.sqrt((Z.reshape(d // 3, 3, n_draws) ** 2).sum(axis=1))
    return float(blocks.max(axis=0).mean())


def constrained_map(model: "MinibatchLogistic", ball: "Ball", n_iter: int = 4000) -> np.ndarray:
    """The MAP of the constrained posterior, by projected gradient descent."""
    L = 0.25 * np.linalg.eigvalsh(model.X.T @ model.X).max() + model.prior_precision
    idx = np.arange(model.n)
    w = np.zeros((model.d, 1))
    for _ in range(n_iter):
        w = ball.project(w - model.grad(w, idx) / L)
    return w


def curvature_step_size(
    model: "MinibatchLogistic", ball: "Ball", safety: float = 0.15
) -> tuple:
    """``h = 2 safety / lambda_max(H)`` at the constrained MAP.

    The same rule as the unconstrained experiments, and the same reason: it
    gives the reversible chain its own classical stability optimum and caps the
    worst-case variance inflation of the unadjusted dynamics at
    ``2 / (2 - 2 safety)``.  Here it also fixes *one* step size for all three
    fields, so nothing in the comparison below can come from a bigger step.
    """
    w = constrained_map(model, ball)
    p = _sigmoid(model.X @ w).ravel()
    H = model.X.T @ (model.X * (p * (1.0 - p))[:, None])
    H += model.prior_precision * np.eye(model.d)
    ev = np.linalg.eigvalsh(H)
    return 2.0 * safety / float(ev.max()), {
        "map": w.ravel(),
        "hessian": H,
        "map_radius": float(np.linalg.norm(w)),
        "hessian_max": float(ev.max()),
        "hessian_min": float(ev.min()),
        "condition": float(ev.max() / max(ev.min(), 1e-12)),
    }


# --------------------------------------------------------------------- domain
class Ball:
    """The centred ball ``K_r``, with projection and oblique skew reflection."""

    def __init__(self, radius: float = 2.0) -> None:
        self.radius = float(radius)

    def norms(self, W: np.ndarray) -> np.ndarray:
        return np.sqrt((W * W).sum(axis=0))

    def project(self, W: np.ndarray) -> np.ndarray:
        nrm = self.norms(W)
        out = nrm > self.radius
        if not out.any():
            return W
        W = W.copy()
        W[:, out] *= self.radius / nrm[out]
        return W

    def uniform(self, d: int, m: int, rng: np.random.Generator, radius=None) -> np.ndarray:
        """``m`` draws from the uniform law on the centred ball of ``radius``."""
        r = self.radius if radius is None else float(radius)
        Z = rng.standard_normal((d, m))
        Z /= np.sqrt((Z * Z).sum(axis=0))
        return Z * (r * rng.random(m) ** (1.0 / d))

    def reflect(self, Y: np.ndarray, field) -> tuple:
        r"""Push ``Y`` back into ``K_r`` along ``gamma = (I + J(x_b)) n(x_b)``.

        With ``n = y / |y|`` the normal at the radial boundary point and
        ``gamma = n + J(r n) n``, the push ``y - lambda gamma`` lands on the
        sphere at the smaller root of ``|y - lambda gamma|^2 = r^2``.  Two facts
        make that root explicit: ``y . gamma = |y|`` and
        ``|gamma|^2 = 1 + |J n|^2``, both because ``n . J n = 0``.  If the
        discriminant is negative the oblique ray misses the sphere entirely --
        only possible for a very large overshoot -- and the closest approach is
        taken and then projected, which is reported so it can be checked.
        """
        nrm = self.norms(Y)
        out = nrm > self.radius
        if not out.any():
            return Y, out, 0
        Y = Y.copy()
        Yo, no = Y[:, out], nrm[out]
        N = Yo / no
        JN = field.apply(self.radius * N, N)
        gamma = N + JN
        A = 1.0 + (JN * JN).sum(axis=0)
        b = (Yo * gamma).sum(axis=0)
        disc = b * b - A * (no * no - self.radius**2)
        missed = disc < 0.0
        lam = np.where(missed, b / A, (b - np.sqrt(np.maximum(disc, 0.0))) / A)
        Z = Yo - lam * gamma
        Y[:, out] = Z
        Y = self.project(Y)  # exact on the hit rows, a fallback on the missed ones
        return Y, out, int(missed.sum())


# --------------------------------------------------------------------- target
class MinibatchLogistic:
    """Logistic likelihood with mini-batch gradients; the prior is the ball.

    ``potential`` is the full-data potential, used for diagnostics and for the
    Metropolis reference only.  ``grad`` is the mini-batch estimate rescaled to
    the full sum, which is what SGLD needs to be unbiased for ``grad U``.
    ``fd_grad`` is the same estimate built from central differences of the
    mini-batch potential, for the derivative-free variant: ``2d`` potential
    evaluations and no gradient formula anywhere.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, prior_scale: float | None = None) -> None:
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.n, self.d = self.X.shape
        self.prior_precision = 0.0 if prior_scale is None else 1.0 / float(prior_scale) ** 2

    def potential(self, W: np.ndarray) -> np.ndarray:
        Z = self.X @ W
        return softplus(Z).sum(axis=0) - self.y @ Z + 0.5 * self.prior_precision * (W * W).sum(axis=0)

    def batch(self, m_size: int, rng: np.random.Generator) -> np.ndarray:
        if m_size >= self.n:
            return np.arange(self.n)
        return rng.integers(0, self.n, size=m_size)

    def grad(self, W: np.ndarray, idx: np.ndarray) -> np.ndarray:
        Xb, yb = self.X[idx], self.y[idx]
        Z = Xb @ W
        resid = _sigmoid(Z) - yb[:, None]
        scale = self.n / len(idx)
        return scale * (Xb.T @ resid) + self.prior_precision * W

    def fd_grad(self, W: np.ndarray, idx: np.ndarray, eps: float = 1e-2) -> np.ndarray:
        Xb, yb = self.X[idx], self.y[idx]
        Z = Xb @ W
        Xty = Xb.T @ yb
        scale = self.n / len(idx)
        out = np.empty((self.d, W.shape[1]))
        inv = 0.5 / eps
        for j in range(self.d):
            shift = eps * Xb[:, j : j + 1]
            out[j] = (softplus(Z + shift).sum(axis=0) - softplus(Z - shift).sum(axis=0)) * inv - Xty[j]
        return scale * out + self.prior_precision * W

    def predict_proba(self, W: np.ndarray, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ W)


# -------------------------------------------------------------------- scoring
def accuracy_of(p: np.ndarray, y: np.ndarray) -> float:
    return float(((p > 0.5) * y + (p < 0.5) * (1.0 - y) + (p == 0.5) * 0.5).mean())


def log_loss_of(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    q = np.clip(p, eps, 1.0 - eps)
    return float(-(y * np.log(q) + (1.0 - y) * np.log1p(-q)).mean())


# -------------------------------------------------------------------- sampler
def run_constrained_sgld(
    model: MinibatchLogistic,
    field,
    ball: Ball,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    n_iter: int = 2000,
    n_walkers: int = 32,
    step_size: float = 1e-5,
    batch_size: int = 1024,
    seed: int = 0,
    start_radius: float = 1.0,
    reference_mean: np.ndarray | None = None,
    reference_metric: np.ndarray | None = None,
    grad: str = "exact",
    fd_eps: float = 1e-2,
) -> dict:
    """Run PSGLD / SRNSGLD and score the running posterior predictive.

    The walkers start from the uniform law on the centred *unit* ball, as in the
    reference experiment, and one mini-batch per iteration is shared by all
    walkers, which keeps the comparison between fields on common random numbers.
    """
    rng = np.random.default_rng(seed)
    d = model.d
    W = ball.uniform(d, n_walkers, rng, radius=start_radius)

    accuracy = np.empty(n_iter)
    loss = np.empty(n_iter)
    mean_error = np.empty(n_iter) if reference_mean is not None else None
    radius = np.empty(n_iter)
    P_sum = np.zeros(len(y_eval))
    w_sum = np.zeros(d)
    hits = 0
    missed = 0
    noise_ratio = []

    sqrt2h = np.sqrt(2.0 * step_size)
    for t in range(n_iter):
        idx = model.batch(batch_size, rng)
        G = (
            model.grad(W, idx)
            if grad == "exact"
            else model.fd_grad(W, idx, eps=fd_eps)
        )
        drift = -(G + field.apply(W, G)) + field.divergence(W)
        Y = W + step_size * drift + sqrt2h * rng.standard_normal((d, n_walkers))
        W, out, miss = ball.reflect(Y, field)
        hits += int(out.sum())
        missed += miss

        if t % 50 == 0:  # how much of the step noise is the mini-batch, not the Brownian part
            G_full = model.grad(W, np.arange(model.n))
            noise_ratio.append(
                float(step_size * np.linalg.norm(G - G_full) / np.sqrt(n_walkers) / sqrt2h)
            )

        P_sum += model.predict_proba(W, X_eval).sum(axis=1)
        w_sum += W.sum(axis=1)
        p_bar = P_sum / ((t + 1) * n_walkers)
        accuracy[t] = accuracy_of(p_bar, y_eval)
        loss[t] = log_loss_of(p_bar, y_eval)
        radius[t] = float(ball.norms(W).mean())
        if mean_error is not None:
            diff = w_sum / ((t + 1) * n_walkers) - reference_mean
            mean_error[t] = float(
                np.sqrt(diff @ (reference_metric @ diff))
                if reference_metric is not None
                else np.linalg.norm(diff)
            )

    return {
        "label": field.label,
        "key": field.key,
        "amplitude": field.amplitude,
        "accuracy": accuracy,
        "loss": loss,
        "mean_error": mean_error,
        "radius": radius,
        "posterior_mean": w_sum / (n_iter * n_walkers),
        "final_state": W,
        "boundary_rate": hits / (n_iter * n_walkers),
        "missed_reflections": missed,
        "minibatch_noise_ratio": float(np.mean(noise_ratio)) if noise_ratio else float("nan"),
        "step_size": float(step_size),
    }


# ------------------------------------------------------------------ reference
def reference_constrained_rwm(
    model: MinibatchLogistic,
    ball: Ball,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    n_iter: int = 200000,
    n_warmup: int = 20000,
    n_chains: int = 8,
    thin: int = 10,
    seed: int = 0,
    start: np.ndarray | None = None,
) -> dict:
    """Exact reference for the constrained posterior: random-walk Metropolis.

    A proposal outside ``K_r`` is rejected, which is the Metropolis-Hastings
    rule for the uniform prior on the ball and leaves ``pi`` restricted to
    ``K_r`` exactly invariant -- no discretisation bias, no projection bias, no
    mini-batch bias.  The proposal covariance is adapted during warm-up in the
    manner of Haario, Saksman and Tamminen, then frozen.  The chains start at
    ``start`` -- the constrained MAP, in practice -- because a reference chain
    has no reason to pay for the transient the samplers under test are being
    measured on, and a proposal covariance estimated from that transient is far
    too wide: on MAGIC the posterior scales like ``1 / sqrt(n)``, and freezing a
    transient-wide proposal drops the acceptance rate to zero.
    """
    rng = np.random.default_rng(seed)
    d = model.d
    if start is None:
        W = ball.uniform(d, n_chains, rng, radius=0.5 * ball.radius)
    else:
        W = ball.project(
            np.asarray(start, float).reshape(d, 1)
            + 0.01 * ball.radius * rng.standard_normal((d, n_chains))
        )
    U = model.potential(W)
    chol = np.eye(d) * 0.05
    log_scale = 0.0
    samples = []
    accepted = 0.0

    P_sum = np.zeros(len(y_eval))
    halves = [np.zeros(len(y_eval)), np.zeros(len(y_eval))]
    half_counts = [0, 0]
    w_sum = np.zeros(d)
    ww_sum = np.zeros((d, d))
    n_kept = 0

    total = n_warmup + n_iter
    for t in range(1, total + 1):
        Wp = W + np.exp(log_scale) * (chol @ rng.standard_normal((d, n_chains)))
        inside = ball.norms(Wp) <= ball.radius
        Up = np.where(inside, model.potential(Wp), np.inf)
        take = inside & (np.log(rng.random(n_chains)) < -(Up - U))
        W = np.where(take, Wp, W)
        U = np.where(take, Up, U)
        rate = float(take.mean())

        if t <= n_warmup:
            log_scale += (rate - 0.234) / (10.0 + t) ** 0.6 * 5.0
            samples.append(W.copy())
            # stop re-shaping the proposal well before warm-up ends, so that the
            # scale that is frozen has been adapted to the final shape
            if t % 2000 == 0 and t <= max(n_warmup - 4000, n_warmup // 2):
                S = np.concatenate(samples[-min(len(samples), 4000) :], axis=1)
                chol = np.linalg.cholesky(np.cov(S) + 1e-12 * np.eye(d))
                log_scale = np.log(2.38 / np.sqrt(d))
        else:
            accepted += rate
            if (t - n_warmup) % thin == 0:
                P = model.predict_proba(W, X_eval)
                P_sum += P.sum(axis=1)
                which = 0 if (t - n_warmup) <= n_iter // 2 else 1
                halves[which] += P.sum(axis=1)
                half_counts[which] += n_chains
                w_sum += W.sum(axis=1)
                ww_sum += W @ W.T
                n_kept += n_chains

    p_mean = P_sum / n_kept
    w_mean = w_sum / n_kept
    cov = ww_sum / n_kept - np.outer(w_mean, w_mean)
    return {
        "accuracy": accuracy_of(p_mean, y_eval),
        "loss": log_loss_of(p_mean, y_eval),
        "accuracy_halves": [accuracy_of(halves[i] / half_counts[i], y_eval) for i in (0, 1)],
        "loss_halves": [log_loss_of(halves[i] / half_counts[i], y_eval) for i in (0, 1)],
        "predictive": p_mean,
        "posterior_mean": w_mean,
        "posterior_cov": cov,
        "mean_radius": float(np.linalg.norm(w_mean)),
        "acceptance": accepted / n_iter,
        "n_samples": n_kept,
    }
