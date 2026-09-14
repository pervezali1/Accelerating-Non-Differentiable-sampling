r"""Constrained sampling with a lasso regularizer, by anchored Langevin dynamics.

This module runs the experiments of the skew-reflected non-reversible SGLD paper
(Section 3.3: constrained Bayesian logistic regression, synthetic and real data)
with the target's regularizer replaced by a **non-differentiable** one, and the
dynamics replaced by the ones that can still sample it exactly.

**The target.**  The paper's target is the likelihood with a uniform prior on a
convex body ``K``, so the constraint is the only regularisation.  Here a lasso
(Laplace) prior is added, which is where the difficulty comes from:

    U(x)  =  sum_j softplus(x . X_j) - y_j (x . X_j)  +  lam |x|_1,     x in K.

``|x|_1`` is not differentiable on any coordinate hyperplane, and the posterior
puts mass there, so ``grad U`` does not exist where the sampler spends its time.

**The dynamics: anchored Langevin.**  Define a smooth *anchor* that majorises
the target,

    U_0(x)  =  sum_j softplus(x . X_j) - y_j (x . X_j)  +  lam sum_j sqrt(x_j^2 + delta^2),

so ``U_0 >= U`` with ``U_0 - U = lam sum_j (sqrt(x_j^2 + delta^2) - |x_j|)`` in
``[0, lam d delta]``, and follow ``grad U_0`` -- which exists everywhere -- while
scaling the diffusion by the *clock*

    a(x) = exp(U(x) - U_0(x)) in [exp(-lam d delta), 1].

Time-changing a diffusion whose invariant density is ``p`` at rate ``a`` leaves
``p / a`` invariant, and here ``p / a = e^{-U_0} e^{U_0 - U} = e^{-U}``: the
anchored chain targets the kinked law exactly, for any ``delta``, without ever
evaluating a subgradient.  The likelihood cancels in ``U - U_0``, so the clock
costs ``O(d)``.

**The dynamics: non-reversible, and reflected.**  Adding the paper's
skew-symmetric field to the anchor's drift and time-changing the result gives

    x_{k+1} = P^J_K( x_k - eta a(x_k) (I + J(x_k)) ghat_0(x_k) + sqrt(2 eta a(x_k)) xi ),

with ``ghat_0`` a mini-batch estimate of ``grad U_0`` and ``P^J_K`` the skew
projection: the push back into ``K`` along ``gamma(x) = (I + J(x)) n(x)`` rather
than along the normal ``n``.  ``J = 0`` is anchored PSGLD; ``J != 0`` is anchored
SRNSGLD.  The time change does not touch the boundary behaviour (it rescales
drift and diffusion by the same positive scalar and leaves the reflection
direction alone), and ``div J = 0`` for every field below, so the invariant law
on ``K`` is exactly ``e^{-U} 1_K``.

**The fields**, all with ``div J = 0``, as the paper constructs them:

* ``J_a``: constant, ``+a`` on the whole superdiagonal, ``-a`` on the
  subdiagonal.  In ``d = 3`` this is the paper's (3.1).  It does *not* satisfy
  ``J n = 0``, so it is the only field whose reflection is genuinely oblique.
* ``J_s(x)``: block diagonal, the ``l``-th ``3 x 3`` block acting as
  ``w -> k_l(x) x w`` with axial vector ``k_l(x) = s_l x_{3l-2:3l}``.  Then
  ``J_s(x) x = 0`` everywhere, so ``J_s n = 0`` on any centred sphere.
* ``J_g(x)``: the same block structure with ``k_l(x) = -s_l grad_l g(x)`` for
  the smoothed ``l_p`` potential ``g(x) = sum_i (x_i^2 + eps^2)^{p/2}``, which
  gives ``J_g(x) n(x) = 0`` on the sublevel set ``{g <= level}`` -- and in fact
  everywhere, since ``k_l`` is parallel to the block of ``grad g``.

Both axial fields are curls of gradients blockwise, hence divergence free.
"""

from __future__ import annotations

import numpy as np

from .constrained import ConstantField, ZeroField  # noqa: F401  (re-exported)
from .target import _sigmoid, softplus


# --------------------------------------------------------------------- fields
class AxialBlockField:
    r"""Block-diagonal skew field acting as ``J(x) w = k(x) x w`` per triple.

    ``axial(W)`` returns the axial vector field stacked like ``W``, block by
    block; the ``l``-th ``3 x 3`` block of ``J`` is then the hat map of
    ``k_l``, i.e. the paper's

        [[0, -k_3, k_2], [k_3, 0, -k_1], [-k_2, k_1, 0]].

    The divergence vanishes whenever each ``k_l`` is a gradient field of its own
    three coordinates, which holds for both fields used here, so
    :meth:`divergence` returns exactly zero.
    """

    def __init__(self, d: int, axial, label: str, key: str, amplitude=1.0) -> None:
        if d % 3:
            raise ValueError(f"block field needs d divisible by 3, got {d}")
        self.d = d
        self.n_blocks = d // 3
        self._axial = axial
        self.label = label
        self.key = key
        arr = np.atleast_1d(np.asarray(amplitude, float)).ravel()
        self.amplitudes = np.repeat(arr, self.n_blocks) if arr.size == 1 else arr
        if self.amplitudes.size != self.n_blocks:
            raise ValueError(f"expected 1 or {self.n_blocks} amplitudes")

    def axial(self, W: np.ndarray) -> np.ndarray:
        return self._axial(W)

    def apply(self, W: np.ndarray, G: np.ndarray) -> np.ndarray:
        K = self.axial(W)
        out = np.empty_like(G)
        for b in range(self.n_blocks):
            s = slice(3 * b, 3 * b + 3)
            out[s] = np.cross(K[s].T, G[s].T).T  # k x g
        return out

    def divergence(self, W: np.ndarray) -> np.ndarray:
        return np.zeros_like(W)

    def matrix(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float).reshape(self.d, 1)
        K = self.axial(x).ravel()
        J = np.zeros((self.d, self.d))
        for b in range(self.n_blocks):
            k1, k2, k3 = K[3 * b : 3 * b + 3]
            J[3 * b : 3 * b + 3, 3 * b : 3 * b + 3] = [
                [0.0, -k3, k2], [k3, 0.0, -k1], [-k2, k1, 0.0]
            ]
        return J


def ball_axial_field(d: int, s) -> AxialBlockField:
    """``J_s(x)``: ``k_l(x) = s_l x_{3l-2:3l}``, the paper's ball construction."""
    field = AxialBlockField(
        d, None, label=r"state-dependent $J_s(x)$", key="state", amplitude=s
    )
    scale = np.repeat(field.amplitudes, 3)[:, None]
    field._axial = lambda W: scale * W
    return field


def sublevel_axial_field(d: int, s, p: float, eps: float) -> AxialBlockField:
    r"""``J_g(x)``: ``k_l(x) = -s_l grad_l g``, ``g = sum_i (x_i^2 + eps^2)^{p/2}``."""
    field = AxialBlockField(
        d, None, label=r"state-dependent $J_g(x)$", key="sublevel", amplitude=s
    )
    scale = np.repeat(field.amplitudes, 3)[:, None]

    def axial(W: np.ndarray) -> np.ndarray:
        return -scale * (p * W * (W * W + eps * eps) ** (0.5 * p - 1.0))

    field._axial = axial
    return field


# -------------------------------------------------------------------- domains
class Ball:
    """The centred ball ``K_r = {|x|^2 <= r}`` (the paper's radius convention).

    The paper writes ``K_r = {x : |x|_2^2 <= r}``, so the geometric radius is
    ``sqrt(r)``; both are kept here to avoid any ambiguity.
    """

    key = "ball"

    def __init__(self, radius: float = 2.0, squared: bool = False) -> None:
        self.radius = float(np.sqrt(radius)) if squared else float(radius)
        self.label = f"centred ball, r = {self.radius:g}"

    def norms(self, W: np.ndarray) -> np.ndarray:
        return np.sqrt((W * W).sum(axis=0))

    def contains(self, W: np.ndarray) -> np.ndarray:
        # a relative tolerance, so that a point the projection has just placed
        # exactly on the sphere is not reported as outside by one ulp
        return self.norms(W) <= self.radius * (1.0 + 1e-12)

    def normal(self, W: np.ndarray) -> np.ndarray:
        return W / np.maximum(self.norms(W), 1e-300)

    def project(self, W: np.ndarray) -> np.ndarray:
        nrm = self.norms(W)
        out = nrm > self.radius
        if not out.any():
            return W
        W = W.copy()
        W[:, out] *= self.radius / nrm[out]
        return W

    def retract(self, Y: np.ndarray, field) -> tuple:
        """Skew projection: push along ``gamma = (I + J) n`` back to the sphere."""
        nrm = self.norms(Y)
        out = nrm > self.radius
        if not out.any():
            return Y, out, 0
        Y = Y.copy()
        Yo, no = Y[:, out], nrm[out]
        N = Yo / no
        JN = field.apply(Yo, N)  # J(y) n; zero for both axial fields, since k || y
        gamma = N + JN
        A = 1.0 + (JN * JN).sum(axis=0)
        b = (Yo * gamma).sum(axis=0)
        disc = b * b - A * (no * no - self.radius**2)
        missed = disc < 0.0
        lam = np.where(missed, b / A, (b - np.sqrt(np.maximum(disc, 0.0))) / A)
        Y[:, out] = Yo - lam * gamma
        return self.project(Y), out, int(missed.sum())

    def uniform(self, d: int, m: int, rng, radius=None) -> np.ndarray:
        r = self.radius if radius is None else float(radius)
        Z = rng.standard_normal((d, m))
        Z /= np.sqrt((Z * Z).sum(axis=0))
        return Z * (r * rng.random(m) ** (1.0 / d))


class SmoothedLpBall:
    r"""The sublevel set ``K = {x : g(x) <= level}``, ``g = sum_i (x_i^2 + eps^2)^{p/2}``.

    The outward normal is ``grad g / |grad g|``.  There is no closed-form
    projection, so the push back into ``K`` is a bisection along
    ``-gamma(y) = -(I + J(y)) n(y)``, which is a descent direction for ``g``
    because ``n . gamma = 1``; if the ray fails to reach the boundary within a
    bounded search the fallback is a bisection along ``-y`` toward the origin,
    which always succeeds since ``g(0) = d eps^p < level``.  Both are counted.
    """

    key = "lp"

    def __init__(self, p: float = 4.0, eps: float = 0.2, level: float = 1.0) -> None:
        self.p, self.eps, self.level = float(p), float(eps), float(level)
        self.label = f"smoothed $\\ell_p$ sublevel set, p = {p:g}, $\\varepsilon$ = {eps:g}, $\\lambda$ = {level:g}"

    def g(self, W: np.ndarray) -> np.ndarray:
        return ((W * W + self.eps**2) ** (0.5 * self.p)).sum(axis=0)

    def grad_g(self, W: np.ndarray) -> np.ndarray:
        return self.p * W * (W * W + self.eps**2) ** (0.5 * self.p - 1.0)

    def contains(self, W: np.ndarray) -> np.ndarray:
        return self.g(W) <= self.level * (1.0 + 1e-12)

    def normal(self, W: np.ndarray) -> np.ndarray:
        G = self.grad_g(W)
        return G / np.maximum(np.sqrt((G * G).sum(axis=0)), 1e-300)

    def _shrink(self, Y: np.ndarray) -> np.ndarray:
        """Bisection along ``-y`` toward the origin; always lands inside."""
        lo, hi = np.zeros(Y.shape[1]), np.ones(Y.shape[1])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            inside = self.g(Y * mid) <= self.level
            lo = np.where(inside, mid, lo)
            hi = np.where(inside, hi, mid)
        return Y * lo

    def retract(self, Y: np.ndarray, field) -> tuple:
        out = ~self.contains(Y)
        if not out.any():
            return Y, out, 0
        Y = Y.copy()
        Yo = Y[:, out]
        N = self.normal(Yo)
        gamma = N + field.apply(Yo, N)
        # bracket: double the push until it lands inside, up to a bounded search
        step = np.full(Yo.shape[1], 0.05)
        inside = self.g(Yo - step * gamma) <= self.level
        for _ in range(20):
            if inside.all():
                break
            step = np.where(inside, step, 2.0 * step)
            inside = self.g(Yo - step * gamma) <= self.level
        lo = np.zeros(Yo.shape[1])
        hi = step
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            hit = self.g(Yo - mid * gamma) <= self.level
            hi = np.where(hit, mid, hi)
            lo = np.where(hit, lo, mid)
        Z = Yo - hi * gamma
        failed = ~(self.g(Z) <= self.level)
        if failed.any():
            Z[:, failed] = self._shrink(Yo[:, failed])
        Y[:, out] = Z
        return Y, out, int(failed.sum())

    def uniform(self, d: int, m: int, rng, radius: float = 1.0) -> np.ndarray:
        """Uniform on the ball of ``radius``, restricted to ``K`` by rejection."""
        cols = []
        while sum(c.shape[1] for c in cols) < m:
            Z = rng.standard_normal((d, 4 * m))
            Z /= np.sqrt((Z * Z).sum(axis=0))
            Z *= radius * rng.random(4 * m) ** (1.0 / d)
            cols.append(Z[:, self.contains(Z)])
        return np.concatenate(cols, axis=1)[:, :m]


# -------------------------------------------------------------------- targets
class LassoLogistic:
    r"""``U(x) = NLL(x) + lam |x|_1`` with the smooth anchor and its clock.

    ``potential`` is the exact, kinked target -- used by the Metropolis
    reference and never by the sampler.  ``anchor_grad`` is the mini-batch
    estimate of ``grad U_0``: the likelihood term is scaled from the batch to
    the full sum, exactly as ``grad f(x, Omega)`` is defined in the paper, and
    the penalty term is exact because it costs ``O(d)``.
    """

    def __init__(
        self, X: np.ndarray, y: np.ndarray, lam: float = 0.0, delta: float = 0.02
    ) -> None:
        self.X = np.ascontiguousarray(X, float)
        self.y = np.ascontiguousarray(y, float)
        self.n, self.d = self.X.shape
        self.lam, self.delta = float(lam), float(delta)

    # ------------------------------------------------------------ potentials
    def nll(self, W: np.ndarray) -> np.ndarray:
        Z = self.X @ W
        return softplus(Z).sum(axis=0) - self.y @ Z

    def potential(self, W: np.ndarray) -> np.ndarray:
        return self.nll(W) + self.lam * np.abs(W).sum(axis=0)

    def anchor_potential(self, W: np.ndarray) -> np.ndarray:
        return self.nll(W) + self.lam * np.sqrt(W * W + self.delta**2).sum(axis=0)

    def log_clock(self, W: np.ndarray) -> np.ndarray:
        """``U - U_0``; the likelihood cancels, so this is ``O(d)`` to evaluate."""
        return -self.lam * (np.sqrt(W * W + self.delta**2) - np.abs(W)).sum(axis=0)

    def clock(self, W: np.ndarray) -> np.ndarray:
        return np.exp(self.log_clock(W))

    @property
    def clock_floor(self) -> float:
        return float(np.exp(-self.lam * self.d * self.delta))

    # ------------------------------------------------------------- gradients
    def batch(self, m_size: int, rng) -> np.ndarray:
        if m_size >= self.n:
            return np.arange(self.n)
        return rng.integers(0, self.n, size=m_size)

    def anchor_grad(self, W: np.ndarray, idx: np.ndarray) -> np.ndarray:
        Xb, yb = self.X[idx], self.y[idx]
        resid = _sigmoid(Xb @ W) - yb[:, None]
        scale = self.n / len(idx)
        smooth = self.lam * W / np.sqrt(W * W + self.delta**2)
        return scale * (Xb.T @ resid) + smooth

    def predict_proba(self, W: np.ndarray, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ W)


# -------------------------------------------------------------------- scoring
def per_walker_accuracy(W: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Accuracy of each walker's own parameter, shape ``(m,)``; ties count a half."""
    P = _sigmoid(X @ W)
    yc = y[:, None]
    return ((P > 0.5) * yc + (P < 0.5) * (1.0 - yc) + (P == 0.5) * 0.5).mean(axis=0)


# -------------------------------------------------------------------- sampler
def run_anchored_srnsgld(
    target: LassoLogistic,
    field,
    domain,
    data: dict,
    n_iter: int = 1000,
    n_walkers: int = 100,
    step_size: float = 1e-4,
    batch_size: int = 30,
    seed: int = 0,
    start_radius: float = 1.0,
    anchored: bool = True,
    reference_mean: np.ndarray | None = None,
    reference_metric: np.ndarray | None = None,
    burn_in: float = 0.5,
) -> dict:
    r"""Anchored SRNSGLD / PSGLD on a constrained lasso posterior.

    One step, with ``a`` the clock and ``ghat_0`` the mini-batch anchor gradient:

        y  = x - eta a(x) (I + J(x)) ghat_0(x) + eta a(x) div J(x)
                 + sqrt(2 eta a(x)) xi,
        x' = P^J_K(y).

    ``anchored=False`` drops the clock (``a = 1``), which is the same drift
    without the correction that makes the kinked law invariant -- it is kept so
    the cost of the clock can be separated from its effect.
    """
    rng = np.random.default_rng(seed)
    d = target.d
    W = domain.uniform(d, n_walkers, rng, radius=start_radius)

    acc_train = np.empty((n_iter, n_walkers))
    acc_test = np.empty((n_iter, n_walkers))
    potential = np.empty((n_iter, n_walkers))
    clocks = np.empty(n_iter)
    hits = 0
    failed = 0
    w_sum = np.zeros(d)
    n_kept = 0
    first_kept = int(burn_in * n_iter)

    for t in range(n_iter):
        idx = target.batch(batch_size, rng)
        G = target.anchor_grad(W, idx)
        scale = target.clock(W)[None, :] if anchored else 1.0
        drift = field.divergence(W) - (G + field.apply(W, G))
        Y = (
            W
            + step_size * scale * drift
            + np.sqrt(2.0 * step_size * scale) * rng.standard_normal((d, n_walkers))
        )
        W, out, bad = domain.retract(Y, field)
        hits += int(out.sum())
        failed += bad

        if t >= first_kept:  # a time average over the second half of the run
            w_sum += W.sum(axis=1)
            n_kept += n_walkers
        acc_train[t] = per_walker_accuracy(W, data["X_train"], data["y_train"])
        acc_test[t] = per_walker_accuracy(W, data["X_test"], data["y_test"])
        potential[t] = target.potential(W)
        clocks[t] = float(np.mean(target.clock(W)))

    time_mean = w_sum / max(n_kept, 1)
    mean_error = float("nan")
    if reference_mean is not None:
        diff = time_mean - np.asarray(reference_mean, float)
        mean_error = float(
            np.sqrt(diff @ (reference_metric @ diff))
            if reference_metric is not None
            else np.linalg.norm(diff)
        )
    return {
        "label": field.label,
        "key": field.key,
        "time_mean": time_mean,
        "mean_error": mean_error,
        "accuracy_train": acc_train,
        "accuracy_test": acc_test,
        "potential": potential,
        "clock": clocks,
        "final_state": W,
        "posterior_mean": W.mean(axis=1),
        "boundary_rate": hits / (n_iter * n_walkers),
        "failed_retractions": failed,
        "step_size": float(step_size),
        "anchored": bool(anchored),
    }


# ------------------------------------------------------------------ reference
def reference_constrained_rwm(
    target: LassoLogistic,
    domain,
    data: dict,
    n_iter: int = 100000,
    n_warmup: int = 20000,
    n_chains: int = 8,
    thin: int = 10,
    seed: int = 0,
    start: np.ndarray | None = None,
) -> dict:
    """Exact reference for the kinked, constrained target: random-walk Metropolis.

    Proposals outside ``K`` are rejected, which is the Metropolis-Hastings rule
    for a uniform prior on ``K``, and the accept/reject step makes ``e^{-U}``
    invariant however rough ``U`` is.  No discretisation, projection, clock or
    mini-batch bias -- this is what the anchored chains are measured against.
    """
    rng = np.random.default_rng(seed)
    d = target.d
    if start is None:
        W = domain.uniform(d, n_chains, rng, radius=0.5)
    else:
        W = np.asarray(start, float).reshape(d, 1) + 0.01 * rng.standard_normal((d, n_chains))
        W, _, _ = domain.retract(W, ZeroField(d))
    U = target.potential(W)
    chol = np.eye(d) * 0.05
    log_scale = 0.0
    samples = []
    accepted = 0.0
    w_sum, ww_sum, n_kept = np.zeros(d), np.zeros((d, d)), 0
    acc_tr, acc_te = [], []

    total = n_warmup + n_iter
    for t in range(1, total + 1):
        Wp = W + np.exp(log_scale) * (chol @ rng.standard_normal((d, n_chains)))
        inside = domain.contains(Wp)
        Up = np.where(inside, target.potential(Wp), np.inf)
        take = inside & (np.log(rng.random(n_chains)) < -(Up - U))
        W = np.where(take, Wp, W)
        U = np.where(take, Up, U)
        rate = float(take.mean())

        if t <= n_warmup:
            log_scale += (rate - 0.234) / (10.0 + t) ** 0.6 * 5.0
            samples.append(W.copy())
            if t % 2000 == 0 and t <= max(n_warmup - 4000, n_warmup // 2):
                S = np.concatenate(samples[-min(len(samples), 4000):], axis=1)
                chol = np.linalg.cholesky(np.cov(S) + 1e-12 * np.eye(d))
                log_scale = np.log(2.38 / np.sqrt(d))
        else:
            accepted += rate
            if (t - n_warmup) % thin == 0:
                w_sum += W.sum(axis=1)
                ww_sum += W @ W.T
                n_kept += n_chains
                acc_tr.append(per_walker_accuracy(W, data["X_train"], data["y_train"]))
                acc_te.append(per_walker_accuracy(W, data["X_test"], data["y_test"]))

    w_mean = w_sum / n_kept
    return {
        "posterior_mean": w_mean,
        "posterior_cov": ww_sum / n_kept - np.outer(w_mean, w_mean),
        "accuracy_train": float(np.mean(acc_tr)),
        "accuracy_test": float(np.mean(acc_te)),
        "accuracy_train_sd": float(np.std(np.concatenate(acc_tr))),
        "accuracy_test_sd": float(np.std(np.concatenate(acc_te))),
        "accuracy_test_halves": [
            float(np.mean(acc_te[: len(acc_te) // 2])),
            float(np.mean(acc_te[len(acc_te) // 2:])),
        ],
        "acceptance": accepted / n_iter,
        "n_samples": n_kept,
        "mean_radius": float(np.linalg.norm(w_mean)),
        "n_near_zero": int((np.abs(w_mean) < 0.01).sum()),
    }
