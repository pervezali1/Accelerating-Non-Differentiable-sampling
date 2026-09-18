"""Exact-gradient anchored Langevin on a NON-differentiable LASSO target.

Target (the split is the user's):

    U(w) = sum_j [softplus(x_j.w) - y_j x_j.w] + w0^2/(2 sigma^2)   <- f, differentiable
           + lambda_lasso * sum_{j>=1} |w_j|                        <- g, NOT differentiable

so ``U = f + g`` with ``g`` kinked at every ``w_j = 0`` (the intercept ``w0`` is
excluded from the penalty and carries a weak Gaussian prior instead).

Anchor.  ``g`` is replaced by its smoothed surrogate

    g_delta(w) = lambda_lasso * sum_{j>=1} sqrt(w_j^2 + delta^2),
    U0 = f + g_delta,
    a(w) = exp(U - U0) = exp(lambda_lasso * sum_{j>=1} (|w_j| - sqrt(w_j^2 + delta^2))),

computed from the penalty difference directly, never as ``exp(U)/exp(U0)``.
Because ``0 <= sqrt(t^2+delta^2) - |t| <= delta``,

    exp(-(d-1) * lambda_lasso * delta) <= a(w) <= 1.

This is a *real* anchor: only ``grad U0`` is ever evaluated, yet the sampled
measure is ``exp(-U)`` with the true kinked ``g``.  (Contrast the earlier
``U0 = U + rho H_K`` construction, where ``U`` was already smooth and the anchor
was a proposed geometric one.)

Update (exact gradient, sign convention as specified):

    x_{k+1} = Pi_K[ x_k - eta a(x_k) grad_U0(x_k)
                        + eta alpha a(x_k) J_s(x_k) grad_U0(x_k)
                        + sqrt(2 eta a(x_k)) xi_{k+1} ]

i.e. the drift is ``-eta a (I - alpha J) grad_U0``.  Note the PLUS on the ``J``
term: since ``J`` enters linearly and is skew, this is the previous
``(I + alpha J)`` convention with the rotation reversed (equivalently
``s -> -s``), and the invariance argument is unchanged by the sign.

Constraints: the Euclidean ball and the smoothed l^p ball, both handled by exact
Euclidean projection.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.model_selection import train_test_split

from anchored_sgld import (  # verified machinery, reused unchanged
    BallGeometry,
    Geometry,
    ProjectionOutcome,
    _as_2d,
    apply_J,
    build_J,
    divergence_J,
    hat,
    paired_difference,
    posterior_condition_number,
)

BETA_TRUE_9 = np.array([0.35, -0.25, 0.15, 0.30, -0.20, 0.10, 0.25, -0.15, 0.20])


# ==========================================================================
# Configuration
# ==========================================================================
@dataclass(frozen=True)
class LassoConfig:
    """Settings for the exact-gradient LASSO experiment."""

    d: int = 9                       # 1 intercept + 8 slopes
    n_total: int = 2000
    test_fraction: float = 0.2
    # target
    lambda_lasso: float = 10.0       # strength of the non-differentiable g
    sigma_intercept: float = 5.0     # weak Gaussian prior on w0
    delta_anchor: float = 0.02       # smoothing of g inside U0
    # sampler
    eta: float = 1e-4                # step size
    n_iterations: int = 1000
    n_repeats: int = 100
    checkpoint_every: int = 10
    block_scales: tuple[float, ...] = (5.0, 5.0, 5.0)
    # l^p ball
    p_constraint: float = 1.0        # smoothed L1 ball
    epsilon: float = 0.2
    l1_radius: float = 1.9           # Lambda = d*eps + l1_radius = 3.7
    # design
    rho_x: float | None = None       # None -> X ~ N(0, 2I); else AR(1) correlation
    # seeds
    data_seed: int = 2026
    split_seed: int = 2027
    sampler_seed: int = 3000

    @property
    def n_train(self) -> int:
        return self.n_total - int(round(self.n_total * self.test_fraction))

    @property
    def n_test(self) -> int:
        return int(round(self.n_total * self.test_fraction))

    @property
    def n_blocks(self) -> int:
        if self.d % 3:
            raise ValueError("d must be a multiple of 3")
        return self.d // 3

    @property
    def scales(self) -> np.ndarray:
        s = np.asarray(self.block_scales, dtype=float)
        if s.size != self.n_blocks:
            raise ValueError(f"block_scales needs {self.n_blocks} entries")
        return s

    def beta_true(self) -> np.ndarray:
        if self.d != 9:
            raise ValueError("beta_true is specified for d = 9")
        return BETA_TRUE_9.copy()


#: (name, alpha) for the two compared methods; both anchored.
METHODS = (("Reversible anchored Langevin", 0.0),
           ("Non-reversible anchored Langevin", 1.0))


# ==========================================================================
# Data — now WITH an intercept column, because U penalises w0 separately
# ==========================================================================
@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    beta_true: np.ndarray

    @property
    def n_train(self) -> int: return self.X_train.shape[0]
    @property
    def n_test(self) -> int: return self.X_test.shape[0]
    @property
    def d(self) -> int: return self.X_train.shape[1]


def make_dataset(cfg: LassoConfig) -> Dataset:
    """Synthetic logistic data with an intercept in column 0.

    ``U`` gives ``w0`` a Gaussian prior and penalises only ``w_1..w_8``, so
    column 0 is a column of ones and ``w0`` is a genuine intercept.  This is a
    deliberate change from the earlier no-intercept design, forced by the form
    of ``U``.
    """
    rng = np.random.default_rng(cfg.data_seed)
    n_features = cfg.d - 1
    if cfg.rho_x is None:
        Z = rng.normal(0.0, math.sqrt(2.0), size=(cfg.n_total, n_features))
    else:
        index = np.arange(n_features)
        Sigma = 2.0 * (cfg.rho_x ** np.abs(index[:, None] - index[None, :]))
        Z = rng.multivariate_normal(np.zeros(n_features), Sigma, size=cfg.n_total)
    X = np.hstack([np.ones((cfg.n_total, 1)), Z])
    beta_true = cfg.beta_true()
    y = (rng.uniform(size=cfg.n_total) <= expit(X @ beta_true)).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_fraction, random_state=cfg.split_seed,
        stratify=y, shuffle=True,
    )
    return Dataset(np.ascontiguousarray(X_train), y_train,
                   np.ascontiguousarray(X_test), y_test, beta_true)


# ==========================================================================
# Target: U = f + g, anchor U0 = f + g_delta
# ==========================================================================
@dataclass
class LassoTarget:
    """``U = f + g`` with a non-differentiable ``g``, and its smooth anchor."""

    X: np.ndarray
    y: np.ndarray
    lambda_lasso: float
    sigma_intercept: float
    delta_anchor: float

    @property
    def d(self) -> int:
        return self.X.shape[1]

    # ---- f: the differentiable part ----
    def f(self, w: np.ndarray) -> np.ndarray:
        w2, squeeze = _as_2d(w)
        eta = w2 @ self.X.T
        value = (np.logaddexp(0.0, eta) - self.y[None, :] * eta).sum(axis=1)
        value = value + w2[:, 0] ** 2 / (2.0 * self.sigma_intercept ** 2)
        return value[0] if squeeze else value

    def grad_f(self, w: np.ndarray) -> np.ndarray:
        w2, squeeze = _as_2d(w)
        gradient = (expit(w2 @ self.X.T) - self.y[None, :]) @ self.X
        gradient = gradient.copy()
        gradient[:, 0] += w2[:, 0] / self.sigma_intercept ** 2
        return gradient[0] if squeeze else gradient

    # ---- g: the NON-differentiable part, and its smoothing ----
    def g(self, w: np.ndarray) -> np.ndarray:
        w2, squeeze = _as_2d(w)
        value = self.lambda_lasso * np.abs(w2[:, 1:]).sum(axis=1)
        return value[0] if squeeze else value

    def g_smooth(self, w: np.ndarray) -> np.ndarray:
        w2, squeeze = _as_2d(w)
        value = self.lambda_lasso * np.sqrt(
            w2[:, 1:] ** 2 + self.delta_anchor ** 2).sum(axis=1)
        return value[0] if squeeze else value

    def grad_g_smooth(self, w: np.ndarray) -> np.ndarray:
        w2, squeeze = _as_2d(w)
        out = np.zeros_like(w2)
        out[:, 1:] = (self.lambda_lasso * w2[:, 1:]
                      / np.sqrt(w2[:, 1:] ** 2 + self.delta_anchor ** 2))
        return out[0] if squeeze else out

    # ---- potentials ----
    def U(self, w: np.ndarray) -> np.ndarray:
        return self.f(w) + self.g(w)

    def U0(self, w: np.ndarray) -> np.ndarray:
        return self.f(w) + self.g_smooth(w)

    def grad_U0(self, w: np.ndarray) -> np.ndarray:
        """The ONLY gradient the sampler evaluates — exact, no mini-batch."""
        return self.grad_f(w) + self.grad_g_smooth(w)

    # ---- anchor coefficient ----
    def log_a(self, w: np.ndarray) -> np.ndarray:
        """``U - U0 = g - g_smooth``, from the penalty difference directly."""
        w2, squeeze = _as_2d(w)
        slopes = w2[:, 1:]
        value = self.lambda_lasso * (
            np.abs(slopes) - np.sqrt(slopes ** 2 + self.delta_anchor ** 2)
        ).sum(axis=1)
        return value[0] if squeeze else value

    def a(self, w: np.ndarray) -> np.ndarray:
        return np.exp(self.log_a(w))

    @property
    def log_a_lower_bound(self) -> float:
        return -(self.d - 1) * self.lambda_lasso * self.delta_anchor

    @property
    def a_lower_bound(self) -> float:
        return float(np.exp(self.log_a_lower_bound))

    def lipschitz_constant(self) -> float:
        """Upper bound on ``||Hess U0||``: data curvature plus anchor curvature."""
        eig_max = float(np.linalg.eigvalsh(self.X.T @ self.X).max())
        return 0.25 * eig_max + max(1.0 / self.sigma_intercept ** 2,
                                    self.lambda_lasso / self.delta_anchor)


def accuracy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    w2, squeeze = _as_2d(w)
    value = (((w2 @ X.T) >= 0.0) == (y[None, :] >= 0.5)).mean(axis=1)
    return value[0] if squeeze else value


# ==========================================================================
# Smoothed l^p ball (general p >= 1; p = 1 and p = 4 reproduce the earlier sets)
# ==========================================================================
class SmoothLpBallGeometry(Geometry):
    """``K = {w : g_p(w) = sum_i (w_i^2 + eps^2)^{p/2} <= Lambda}``.

    ``Lambda = d eps^p + radius^p``, so ``p = 4`` with ``eps = 0.2`` and
    ``radius = 0.9964`` reproduces the quartic set (``Lambda = 1``) and ``p = 1``
    reproduces the L1-smooth ball.  ``grad_g[i] = p w_i (w_i^2+eps^2)^{p/2-1}``,
    and the J blocks use ``-s_l grad_{I_l} g`` since the normal is parallel to
    ``grad g``.
    """

    def __init__(self, d: int, p: float = 4.0, epsilon: float = 0.2,
                 radius: float | None = None, Lambda: float | None = None) -> None:
        if p < 1.0:
            raise ValueError("p < 1 breaks monotonicity of the projection map")
        self.d, self.p, self.epsilon = d, p, epsilon
        self.g_min = d * epsilon ** p
        if (radius is None) == (Lambda is None):
            raise ValueError("give exactly one of radius or Lambda")
        if Lambda is None:
            self.radius, self.Lambda = radius, self.g_min + radius ** p
        else:
            self.Lambda = Lambda
            self.radius = (Lambda - self.g_min) ** (1.0 / p)
        self.D = self.Lambda - self.g_min
        self.name = f"smoothed l^{p:g} ball"

    def constraint_value(self, w: np.ndarray) -> np.ndarray:
        w2, squeeze = _as_2d(w)
        value = ((w2 * w2 + self.epsilon ** 2) ** (self.p / 2.0)).sum(axis=1)
        return value[0] if squeeze else value

    @property
    def threshold(self) -> float:
        return self.Lambda

    def grad_g(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        return self.p * w * (w * w + self.epsilon ** 2) ** (self.p / 2.0 - 1.0)

    def H(self, w: np.ndarray) -> np.ndarray:
        return (self.constraint_value(w) - self.g_min) / self.D

    def grad_H(self, w: np.ndarray) -> np.ndarray:
        return self.grad_g(w) / self.D

    def normal(self, w: np.ndarray) -> np.ndarray:
        gradient, squeeze = _as_2d(self.grad_g(w))
        norm = np.linalg.norm(gradient, axis=1, keepdims=True)
        out = np.divide(gradient, norm, out=np.zeros_like(gradient), where=norm > 0)
        return out[0] if squeeze else out

    def boundary_point(self, direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        objective = lambda t: float(self.constraint_value(t * direction)) - self.Lambda
        upper = 1.0
        while objective(upper) < 0.0:
            upper *= 2.0
        return brentq(objective, 0.0, upper, xtol=1e-14, maxiter=200) * direction

    def j_block_vectors(self, w: np.ndarray) -> np.ndarray:
        gradient, _ = _as_2d(self.grad_g(w))
        return (-gradient).reshape(gradient.shape[0], -1, 3)

    # ---- projection ----
    def _solve_coordinates(self, z_abs: np.ndarray, mu, inner: int = 60) -> np.ndarray:
        """``b + mu p b (b^2+eps^2)^{p/2-1} = |z|``, ``b`` in ``[0, |z|]``.

        Strictly increasing for ``p >= 1`` (derivative
        ``1 + mu p (b^2+eps^2)^{p/2-2}[(p-1)b^2 + eps^2] > 0``), so bisection on
        that bracket is unconditionally reliable for any ``p``.
        """
        mu_array = np.asarray(mu, dtype=float)
        low, high = np.zeros_like(z_abs), z_abs.copy()
        for _ in range(inner):
            mid = 0.5 * (low + high)
            value = mid + mu_array * self.p * mid * (
                mid * mid + self.epsilon ** 2) ** (self.p / 2.0 - 1.0)
            positive = value > z_abs
            high = np.where(positive, mid, high)
            low = np.where(positive, low, mid)
        return 0.5 * (low + high)

    def _project_one(self, z: np.ndarray) -> tuple[np.ndarray, float]:
        sign, z_abs = np.sign(z), np.abs(z)

        def gap(mu: float) -> float:
            b = self._solve_coordinates(z_abs, mu)
            return float(((b * b + self.epsilon ** 2) ** (self.p / 2.0)).sum()) - self.Lambda

        high = 1.0
        while gap(high) > 0.0:
            high *= 2.0
        mu = brentq(gap, 0.0, high, xtol=1e-14, maxiter=300)
        return sign * self._solve_coordinates(z_abs, mu), float(mu)

    def project(self, z: np.ndarray, n_bisect: int = 80) -> ProjectionOutcome:
        z2, squeeze = _as_2d(z)
        beta = z2.copy()
        outside = self.constraint_value(z2) > self.Lambda
        worst = 0.0
        rows = np.nonzero(outside)[0]
        if rows.size:
            z_rows = z2[rows]
            sign, z_abs = np.sign(z_rows), np.abs(z_rows)

            def gap(mu_col: np.ndarray) -> np.ndarray:
                b = self._solve_coordinates(z_abs, mu_col)
                return ((b * b + self.epsilon ** 2) ** (self.p / 2.0)).sum(axis=1) - self.Lambda

            low, high = np.zeros(rows.size), np.ones(rows.size)
            for _ in range(200):
                need = gap(high[:, None]) > 0.0
                if not need.any():
                    break
                high[need] *= 2.0
            for _ in range(n_bisect):
                mid = 0.5 * (low + high)
                positive = gap(mid[:, None]) > 0.0
                low = np.where(positive, mid, low)
                high = np.where(positive, high, mid)
            mu = 0.5 * (low + high)
            b = sign * self._solve_coordinates(z_abs, mu[:, None])
            beta[rows] = b
            worst = float(np.abs(b + mu[:, None] * self.grad_g(b) - z_rows).max())
        excess = float((self.constraint_value(beta) - self.Lambda).max())
        return ProjectionOutcome(beta[0] if squeeze else beta, outside, worst, excess)

    def sample_uniform(self, rng: np.random.Generator, n: int,
                       max_rounds: int = 20_000) -> np.ndarray:
        """Uniform on ``K`` by rejection from the enclosing box."""
        half = math.sqrt(
            max((self.Lambda - (self.d - 1) * self.epsilon ** self.p) ** (2.0 / self.p)
                - self.epsilon ** 2, 0.0))
        accepted, total, kept = [], 0, 0
        for _ in range(max_rounds):
            size = max(n, 512)
            proposals = rng.uniform(-half, half, size=(size, self.d))
            total += size
            good = proposals[self.constraint_value(proposals) <= self.Lambda]
            kept += good.shape[0]
            if good.size:
                accepted.append(good)
            if sum(a.shape[0] for a in accepted) >= n:
                break
        else:  # pragma: no cover
            raise RuntimeError("rejection sampler failed")
        self.last_acceptance_rate = kept / total
        return np.vstack(accepted)[:n]


# ==========================================================================
# Sampler — exact gradient, no mini-batch anywhere
# ==========================================================================
@dataclass
class Streams:
    """Shared randomness: identical starts and increments for both methods."""

    w_init: np.ndarray               # (R, d)
    noise: np.ndarray                # (R, n_iterations, d)


def make_streams(cfg: LassoConfig, geometry: Geometry, seed_offset: int = 0) -> Streams:
    # NOTE: Python's built-in hash() on str is salted per process, so it must not
    # be used to derive a seed -- that silently breaks reproducibility between
    # runs.  A stable digest of the geometry name is used instead.
    tag = int.from_bytes(hashlib.sha256(geometry.name.encode()).digest()[:4], "big")
    base = np.random.SeedSequence([cfg.sampler_seed + seed_offset, tag])
    init_ss, noise_ss = base.spawn(2)
    w_init = geometry.sample_uniform(np.random.default_rng(init_ss), cfg.n_repeats)
    noise = np.empty((cfg.n_repeats, cfg.n_iterations, cfg.d))
    for r, seed in enumerate(noise_ss.spawn(cfg.n_repeats)):
        noise[r] = np.random.default_rng(seed).standard_normal((cfg.n_iterations, cfg.d))
    return Streams(w_init, noise)


@dataclass
class RunResult:
    method: str
    geometry: str
    alpha: float
    eta: float
    block_scales: tuple[float, ...]
    checkpoints: np.ndarray
    train_accuracy: np.ndarray       # (n_ckpt, R)
    test_accuracy: np.ndarray        # (n_ckpt, R)
    w: np.ndarray                    # (n_ckpt, R, d)
    U: np.ndarray                    # (n_ckpt, R) the TRUE non-smooth potential
    anchor: np.ndarray               # (n_ckpt, R) a(w)
    constraint: np.ndarray           # (n_ckpt, R)
    projection_rate: float
    max_kkt_residual: float
    n_nonfinite: int
    runtime: float
    drift_ratio: float = 0.0

    @property
    def n_repeats(self) -> int:
        return self.train_accuracy.shape[1]

    def mean_std(self, which: str = "test_accuracy") -> tuple[np.ndarray, np.ndarray]:
        values = getattr(self, which)
        return values.mean(axis=1), values.std(axis=1, ddof=1)


def run_chain(dataset: Dataset, target: LassoTarget, geometry: Geometry,
              streams: Streams, cfg: LassoConfig, *, method: str,
              alpha: float) -> RunResult:
    """x_{k+1} = Pi_K[ x_k - eta a grad_U0 + eta alpha a J grad_U0 + sqrt(2 eta a) xi ].

    ``grad_U0`` is the EXACT gradient of the smooth anchor; nothing is
    subsampled.  Only ``U0`` is differentiated, while the invariant measure of
    the underlying diffusion is ``exp(-U)`` with the true non-differentiable
    ``g``.
    """
    eta, scales = cfg.eta, cfg.scales
    w = streams.w_init.copy()
    checkpoints = [0]
    train_acc = [accuracy(w, dataset.X_train, dataset.y_train)]
    test_acc = [accuracy(w, dataset.X_test, dataset.y_test)]
    history = [w.copy()]
    potential = [target.U(w)]
    anchor = [target.a(w)]
    constraint = [geometry.constraint_value(w)]

    n_projected = n_nonfinite = 0
    worst_kkt = ratio_sum = 0.0

    start = time.perf_counter()
    for k in range(cfg.n_iterations):
        gradient = target.grad_U0(w)                 # exact
        a = target.a(w)
        if alpha == 0.0:
            drift = -eta * a[:, None] * gradient
        else:
            rotated = apply_J(w, gradient, geometry, scales)
            drift = -eta * a[:, None] * gradient + eta * alpha * a[:, None] * rotated
            ratio_sum += float(np.mean(
                np.linalg.norm(alpha * rotated, axis=1)
                / np.maximum(np.linalg.norm(gradient, axis=1), 1e-300)))
        proposal = w + drift + np.sqrt(2.0 * eta * a)[:, None] * streams.noise[:, k, :]

        bad = ~np.isfinite(proposal).all(axis=1)
        if np.any(bad):
            n_nonfinite += int(bad.sum())
            proposal[bad] = w[bad]

        outcome = geometry.project(proposal)
        w = outcome.beta
        n_projected += int(outcome.projected.sum())
        worst_kkt = max(worst_kkt, outcome.max_kkt_residual)

        if (k + 1) % cfg.checkpoint_every == 0:
            checkpoints.append(k + 1)
            train_acc.append(accuracy(w, dataset.X_train, dataset.y_train))
            test_acc.append(accuracy(w, dataset.X_test, dataset.y_test))
            history.append(w.copy())
            potential.append(target.U(w))
            anchor.append(target.a(w))
            constraint.append(geometry.constraint_value(w))
    runtime = time.perf_counter() - start

    return RunResult(
        method=method, geometry=geometry.name, alpha=alpha, eta=eta,
        block_scales=tuple(float(x) for x in np.atleast_1d(scales)),
        checkpoints=np.asarray(checkpoints),
        train_accuracy=np.asarray(train_acc), test_accuracy=np.asarray(test_acc),
        w=np.asarray(history), U=np.asarray(potential), anchor=np.asarray(anchor),
        constraint=np.asarray(constraint),
        projection_rate=n_projected / (cfg.n_iterations * cfg.n_repeats),
        max_kkt_residual=worst_kkt, n_nonfinite=n_nonfinite, runtime=runtime,
        drift_ratio=ratio_sum / cfg.n_iterations,
    )


def run_both(dataset: Dataset, target: LassoTarget, geometry: Geometry,
             cfg: LassoConfig, seed_offset: int = 0,
             verbose: bool = True) -> dict[str, RunResult]:
    """Both anchored methods on shared randomness (so comparisons are paired)."""
    streams = make_streams(cfg, geometry, seed_offset)
    results = {}
    for name, alpha in METHODS:
        results[name] = run_chain(dataset, target, geometry, streams, cfg,
                                  method=name, alpha=alpha)
        if verbose:
            run = results[name]
            mean, sd = run.mean_std("test_accuracy")
            print(f"    {name:<34} test {mean[-1]:.4f} +/- {sd[-1]:.4f}  "
                  f"proj {run.projection_rate:.3f}  ||aJg||/||g|| {run.drift_ratio:.2f}  "
                  f"nonfinite {run.n_nonfinite}  {run.runtime:5.1f}s", flush=True)
    return results


def make_block_anisotropic_dataset(
    cfg: LassoConfig,
    eigenvalues: tuple[float, ...] = (100.0, 1.0, 0.01),
    seed: int | None = None,
) -> Dataset:
    """Design whose anisotropy lives INSIDE each coordinate triple.

    ``Sigma_X`` is block diagonal on the slope coordinates, matching the block
    structure of ``J``, with each 3x3 block a random rotation of
    ``diag(eigenvalues)``.

    Why this matters.  ``J`` is built from the *constraint* geometry and is block
    diagonal on the triples ``(0,1,2), (3,4,5), (6,7,8)``, so it can only rotate
    *within* a triple.  A non-reversible perturbation accelerates convergence by
    coupling slow and fast directions of the target; if the target's slow
    directions span blocks (as under an AR(1) design) ``J`` cannot reach them.
    Putting the anisotropy inside the blocks aligns the two structures.  The
    linearised per-iteration rate ``-log rho(I - eta a (I - alpha J) H)`` rises
    from a 1.1x speed-up on the isotropic design and 2.2x under AR(1) to over 5x
    here.
    """
    rng = np.random.default_rng(cfg.data_seed if seed is None else seed)
    n_features = cfg.d - 1
    values = np.asarray(eigenvalues, dtype=float)
    Sigma = np.zeros((n_features, n_features))
    for start in range(0, n_features - n_features % 3, 3):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        Sigma[start:start + 3, start:start + 3] = (
            rotation @ np.diag(values) @ rotation.T)
    remainder = n_features % 3
    if remainder:
        rotation, _ = np.linalg.qr(rng.normal(size=(remainder, remainder)))
        Sigma[-remainder:, -remainder:] = (
            rotation @ np.diag(values[:remainder]) @ rotation.T)

    Z = rng.multivariate_normal(np.zeros(n_features), Sigma, size=cfg.n_total)
    X = np.hstack([np.ones((cfg.n_total, 1)), Z])
    beta_true = cfg.beta_true()
    y = (rng.uniform(size=cfg.n_total) <= expit(X @ beta_true)).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_fraction, random_state=cfg.split_seed,
        stratify=y, shuffle=True)
    return Dataset(np.ascontiguousarray(X_train), y_train,
                   np.ascontiguousarray(X_test), y_test, beta_true)
