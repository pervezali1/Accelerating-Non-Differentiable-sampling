"""Non-reversible anchored Langevin with block state-dependent skew-symmetric J.

Reusable library behind ``nonreversible_anchored_langevin.ipynb``.

The sampled object is the regression coefficient ``beta`` (written ``b`` in
code where it is a projection argument).  It is *unknown* and is what the
sampler explores; ``beta_true`` is the fixed vector used once to generate the
labels and is never used by the sampler.

Target
------
Uniform prior on a constraint set ``K`` times the logistic likelihood:

    pi_K(beta) ∝ exp(-U(beta)) 1_K(beta),
    U(beta) = sum_{j in train} [ log(1 + exp(X_j.beta)) - y_j X_j.beta ].

``U`` is a **sum** over the 1600 training rows, never a mean.  The mini-batch
estimator rescales accordingly,

    Ghat_k = (n_train / m) X_{B_k}^T [ sigmoid(X_{B_k} beta_k) - y_{B_k} ],

with ``B_k`` drawn uniformly without replacement within the iteration.

Anchor
------
    U0(beta) = U(beta) + rho H_K(beta),
    a(beta)  = exp(U(beta) - U0(beta)) = exp(-rho H_K(beta)).

``H_K`` is a purely geometric function of the constraint (``||beta||^2`` on the
ball, the normalised constraint value on the quartic set), so ``a`` is computed
**exactly from the geometry** — a noisy likelihood difference is never
exponentiated.  With ``rho = log 2`` and ``H_K in [0, 1]`` on ``K`` this gives
``1/2 <= a <= 1``.

This anchor is a *proposed non-trivial anchor for an already smooth target*,
used to make ``a`` state-dependent so the anchored machinery is exercised.  It
is **not** an extra Bayesian penalty: the target ``pi_K`` is unchanged, because
``a`` enters both the drift and the diffusion coefficient in the combination
that leaves ``exp(-U)`` invariant (see :func:`invariant_measure_note`).

Update
------
    beta_{k+1} = Pi_K[ beta_k - h a_k (v_k + alpha J(beta_k) v_k)
                       + sqrt(2 h a_k) xi_k ],
    v_k = Ghat_k + rho grad_H_K(beta_k),
    a_k = exp(-rho H_K(beta_k)),  xi_k ~ N(0, I_d).

No ``grad a`` correction is added, and ``J`` is never rescaled beyond the
constant block strengths ``s``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.model_selection import train_test_split

RHO_ANCHORED: float = math.log(2.0)

BETA_TRUE_9 = np.array(
    [0.35, -0.25, 0.15, 0.30, -0.20, 0.10, 0.25, -0.15, 0.20]
)
BETA_TRUE_3 = np.array([0.60, -0.30, 0.20])


# ==========================================================================
# 1. Configuration
# ==========================================================================
@dataclass(frozen=True)
class ExperimentConfig:
    """Every knob of the experiment.  Nothing is hard-coded elsewhere."""

    d: int = 9
    n_total: int = 2000
    test_fraction: float = 0.2
    batch_size: int = 50                       # m
    n_iterations: int = 1000
    step_size: float = 1e-4                    # h (the paper's candidate value)
    n_repeats: int = 100                       # R
    block_scales: tuple[float, ...] = (10.0, 10.0, 10.0)
    epsilon: float = 0.2                       # quartic smoothing
    Lambda: float = 1.0                        # quartic threshold
    checkpoint_every: int = 10
    data_seed: int = 2026
    split_seed: int = 2027
    sampler_seed: int = 3000
    # Step-size sensitivity
    sensitivity_divisors: tuple[float, ...] = (1.0, 2.0, 4.0)
    sensitivity_repeats: int = 20
    # Reporting
    target_accuracy: float = 0.64

    @property
    def n_train(self) -> int:
        return self.n_total - int(round(self.n_total * self.test_fraction))

    @property
    def n_test(self) -> int:
        return int(round(self.n_total * self.test_fraction))

    @property
    def n_blocks(self) -> int:
        if self.d % 3 != 0:
            raise ValueError("d must be a multiple of 3 for the block construction")
        return self.d // 3

    @property
    def scales(self) -> np.ndarray:
        """Block strengths as an array of length ``n_blocks``."""
        s = np.asarray(self.block_scales, dtype=float)
        if s.size != self.n_blocks:
            raise ValueError(
                f"block_scales has {s.size} entries but d = {self.d} needs "
                f"{self.n_blocks}"
            )
        return s

    def beta_true(self) -> np.ndarray:
        if self.d == 9:
            return BETA_TRUE_9.copy()
        if self.d == 3:
            return BETA_TRUE_3.copy()
        raise ValueError("beta_true is specified only for d = 3 and d = 9")


CONFIG_D3 = ExperimentConfig(d=3, block_scales=(10.0,))


#: The four compared methods: (name, rho, alpha).
METHODS: tuple[tuple[str, float, float], ...] = (
    ("Projected SGLD", 0.0, 0.0),
    ("Non-reversible SGLD", 0.0, 1.0),
    ("Reversible anchored Langevin", RHO_ANCHORED, 0.0),
    ("Non-reversible anchored Langevin", RHO_ANCHORED, 1.0),
)

#: Colourblind-safe, validated (worst adjacent CVD deltaE 9.2 protan / 22.9
#: normal).  Line style is a deliberate second encoding channel.
METHOD_STYLE: dict[str, dict[str, object]] = {
    "Projected SGLD": {"color": "#0173B2", "ls": "-", "lw": 1.7},
    "Non-reversible SGLD": {"color": "#DE8F05", "ls": "--", "lw": 1.7},
    "Reversible anchored Langevin": {"color": "#029E73", "ls": "-.", "lw": 1.7},
    "Non-reversible anchored Langevin": {"color": "#CC3311", "ls": "-", "lw": 2.8},
}


def invariant_measure_note() -> str:
    """The continuous-time identity, and what it does *not* claim."""
    return (
        "Continuous process:  d.beta = -a(beta) [I + alpha J(beta)] grad_U0(beta) dt "
        "+ sqrt(2 a(beta)) dW.\n"
        "  * Reversible part.  For pi ∝ exp(-U0)/a the Fokker-Planck flux is\n"
        "    -b pi + grad(a pi) = a grad_U0 pi - grad_U0 (a pi) = 0, since a pi ∝ "
        "exp(-U0).\n"
        "    With a = exp(U - U0) this gives pi ∝ exp(-U0)/a = exp(-U): the anchor "
        "cancels exactly.\n"
        "  * Non-reversible part.  Its flux is -alpha a J grad_U0 pi = "
        "alpha J grad(exp(-U0)) x const,\n"
        "    whose divergence is (div J).grad f + trace(J Hess f) = 0 + 0, because "
        "div J = 0 and a\n"
        "    skew matrix has zero Frobenius inner product with a symmetric one. So "
        "pi is unchanged.\n"
        "\n"
        "WHAT THIS DOES NOT SAY.  The implemented algorithm is a *projected "
        "stochastic-gradient*\n"
        "Euler-Maruyama discretisation. Three separate sources of bias remain, none "
        "of which the\n"
        "identity above controls:\n"
        "  (i)   finite step size h (Euler-Maruyama discretisation bias, O(h) in "
        "general);\n"
        "  (ii)  mini-batch gradient noise, which injects extra variance not matched "
        "by the\n"
        "        sqrt(2 h a) term and inflates the effective temperature;\n"
        "  (iii) the projection Pi_K, which puts mass on the boundary and is not a "
        "discretisation\n"
        "        of any reflected process that preserves pi_K exactly.\n"
        "Accuracy curves therefore say nothing directly about posterior fidelity."
    )


# ==========================================================================
# 2. Synthetic data
# ==========================================================================
@dataclass
class Dataset:
    """Frozen data set and stratified split, shared by every method/geometry."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    beta_true: np.ndarray
    data_seed: int
    split_seed: int

    @property
    def n_train(self) -> int:
        return self.X_train.shape[0]

    @property
    def n_test(self) -> int:
        return self.X_test.shape[0]

    @property
    def d(self) -> int:
        return self.X_train.shape[1]


def make_dataset(cfg: ExperimentConfig) -> Dataset:
    """Generate ``X_j ~ N(0, 2 I_d)`` and Bernoulli labels, then split 80/20.

    Coordinate standard deviation is ``sqrt(2)``.  There is no intercept and no
    feature standardisation.  Labels use the inverse-CDF form
    ``y_j = 1{u_j <= sigmoid(X_j.beta_true)}`` with ``u_j ~ Uniform(0,1)``.
    """
    rng = np.random.default_rng(cfg.data_seed)
    beta_true = cfg.beta_true()
    X = rng.normal(loc=0.0, scale=np.sqrt(2.0), size=(cfg.n_total, cfg.d))
    u = rng.uniform(0.0, 1.0, size=cfg.n_total)
    y = (u <= expit(X @ beta_true)).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_fraction,
        random_state=cfg.split_seed,
        stratify=y,
        shuffle=True,
    )
    return Dataset(
        X_train=np.ascontiguousarray(X_train),
        y_train=np.ascontiguousarray(y_train),
        X_test=np.ascontiguousarray(X_test),
        y_test=np.ascontiguousarray(y_test),
        beta_true=beta_true,
        data_seed=cfg.data_seed,
        split_seed=cfg.split_seed,
    )


# ==========================================================================
# 3. Posterior and gradients  (beta may be (d,) or (R, d))
# ==========================================================================
def _as_2d(beta: np.ndarray) -> tuple[np.ndarray, bool]:
    beta = np.asarray(beta, dtype=float)
    if beta.ndim == 1:
        return beta[None, :], True
    return beta, False


def potential_U(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``U(beta) = sum_j [softplus(X_j.beta) - y_j X_j.beta]`` (a SUM).

    ``np.logaddexp(0, z)`` is the numerically stable softplus; the
    label-dependent term ``- y_j X_j.beta`` is retained.
    """
    beta2, squeeze = _as_2d(beta)
    eta = beta2 @ X.T
    value = (np.logaddexp(0.0, eta) - y[None, :] * eta).sum(axis=1)
    return value[0] if squeeze else value


def full_gradient(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``grad U = X^T (sigmoid(X beta) - y)`` over all rows of ``X``."""
    beta2, squeeze = _as_2d(beta)
    residual = expit(beta2 @ X.T) - y[None, :]
    gradient = residual @ X
    return gradient[0] if squeeze else gradient


def minibatch_gradient(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    batch_index: np.ndarray,
) -> np.ndarray:
    """``Ghat = (n_train/m) X_B^T [sigmoid(X_B beta) - y_B]``, vectorised over replicates.

    Parameters
    ----------
    beta
        ``(R, d)`` current states.
    X, y
        Full training arrays; ``n_train`` is taken from ``X``.
    batch_index
        ``(R, m)`` integer indices, drawn uniformly **without replacement**
        within each row.

    The ``n_train / m`` factor makes this an unbiased estimator of the *summed*
    gradient.  It is never replaced by an unscaled batch average, and never uses
    ``n_total``.
    """
    beta2, squeeze = _as_2d(beta)
    n_train, m = X.shape[0], batch_index.shape[-1]
    Xb = X[batch_index]                       # (R, m, d)
    yb = y[batch_index]                       # (R, m)
    eta = np.einsum("rmd,rd->rm", Xb, beta2)
    residual = expit(eta) - yb
    gradient = (n_train / m) * np.einsum("rmd,rm->rd", Xb, residual)
    return gradient[0] if squeeze else gradient


def accuracy(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Plug-in accuracy of ``1{sigmoid(X.beta) >= 0.5}``, vectorised over replicates."""
    beta2, squeeze = _as_2d(beta)
    prediction = (beta2 @ X.T) >= 0.0        # sigmoid(z) >= 0.5  <=>  z >= 0
    value = (prediction == (y[None, :] >= 0.5)).mean(axis=1)
    return value[0] if squeeze else value


# ==========================================================================
# 4. Geometries: constraint, anchor, projection, initialisation, J vectors
# ==========================================================================
class Geometry:
    """Interface shared by the ball and the quartic constraint set."""

    name: str
    d: int

    # --- constraint ---
    def constraint_value(self, beta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def threshold(self) -> float:
        raise NotImplementedError

    def feasible(self, beta: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        return self.constraint_value(beta) <= self.threshold + tol

    # --- anchor ---
    def H(self, beta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def grad_H(self, beta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # --- geometry of the boundary ---
    def normal(self, beta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def boundary_point(self, direction: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # --- block vectors defining J (before the block strengths) ---
    def j_block_vectors(self, beta: np.ndarray) -> np.ndarray:
        """``(R, n_blocks, 3)`` vectors ``w_l`` with block ``[s_l w_l]_x``."""
        raise NotImplementedError

    # --- projection and initialisation ---
    def project(self, z: np.ndarray) -> "ProjectionOutcome":
        raise NotImplementedError

    def sample_uniform(self, rng: np.random.Generator, n: int) -> np.ndarray:
        raise NotImplementedError


@dataclass
class ProjectionOutcome:
    """Result of projecting a batch of proposals."""

    beta: np.ndarray                 # (R, d) projected states
    projected: np.ndarray            # (R,) bool, was the row infeasible?
    max_kkt_residual: float          # worst stationarity residual over projected rows
    max_feasibility_excess: float    # worst g(b) - threshold after projection


class BallGeometry(Geometry):
    """``K = {beta : ||beta||_2^2 <= 1}``, with ``H_K(beta) = ||beta||^2``."""

    name = "unit ball"

    def __init__(self, d: int) -> None:
        self.d = d

    def constraint_value(self, beta: np.ndarray) -> np.ndarray:
        beta2, squeeze = _as_2d(beta)
        value = np.sum(beta2 * beta2, axis=1)
        return value[0] if squeeze else value

    @property
    def threshold(self) -> float:
        return 1.0

    def H(self, beta: np.ndarray) -> np.ndarray:
        return self.constraint_value(beta)

    def grad_H(self, beta: np.ndarray) -> np.ndarray:
        return 2.0 * np.asarray(beta, dtype=float)

    def normal(self, beta: np.ndarray) -> np.ndarray:
        beta2, squeeze = _as_2d(beta)
        norm = np.linalg.norm(beta2, axis=1, keepdims=True)
        out = np.divide(beta2, norm, out=np.zeros_like(beta2), where=norm > 0)
        return out[0] if squeeze else out

    def boundary_point(self, direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        return direction / np.linalg.norm(direction)

    def j_block_vectors(self, beta: np.ndarray) -> np.ndarray:
        """``w_l = beta_{I_l}`` — the ball's normal is proportional to ``beta``."""
        beta2, _ = _as_2d(beta)
        return beta2.reshape(beta2.shape[0], -1, 3)

    def project(self, z: np.ndarray) -> ProjectionOutcome:
        """Euclidean projection: ``z`` if ``||z|| <= 1``, else ``z/||z||``."""
        z2, squeeze = _as_2d(z)
        norm = np.linalg.norm(z2, axis=1)
        outside = norm > 1.0
        beta = z2.copy()
        if np.any(outside):
            beta[outside] = z2[outside] / norm[outside, None]
        # KKT: b(1 + 2 mu) = z with mu = (||z|| - 1)/2; residual is exactly 0.
        residual = 0.0
        if np.any(outside):
            mu = (norm[outside] - 1.0) / 2.0
            residual = float(
                np.abs(beta[outside] * (1.0 + 2.0 * mu[:, None]) - z2[outside]).max()
            )
        excess = float((np.sum(beta * beta, axis=1) - 1.0).max())
        out = beta[0] if squeeze else beta
        return ProjectionOutcome(out, outside, residual, excess)

    def sample_uniform(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Uniform on the ball: ``Z V^{1/d} / ||Z||``."""
        Z = rng.standard_normal((n, self.d))
        V = rng.random(n)
        radius = V ** (1.0 / self.d)
        return Z * (radius / np.linalg.norm(Z, axis=1))[:, None]


class QuarticGeometry(Geometry):
    """``K = {beta : g(beta) = sum_i (beta_i^2 + eps^2)^2 <= Lambda}``.

    ``g_min = d eps^4`` is attained at the origin, ``D = Lambda - d eps^4``, and

        H_K(beta) = (g(beta) - d eps^4) / D  in [0, 1] on K,
        grad_g[i] = 4 beta_i (beta_i^2 + eps^2),  grad_H_K = grad_g / D.
    """

    name = "quartic set"

    def __init__(self, d: int, epsilon: float = 0.2, Lambda: float = 1.0) -> None:
        self.d = d
        self.epsilon = epsilon
        self.Lambda = Lambda
        self.g_min = d * epsilon ** 4
        self.D = Lambda - self.g_min
        if self.D <= 0:
            raise ValueError("Lambda must exceed d * epsilon**4")

    def constraint_value(self, beta: np.ndarray) -> np.ndarray:
        beta2, squeeze = _as_2d(beta)
        value = np.sum((beta2 * beta2 + self.epsilon ** 2) ** 2, axis=1)
        return value[0] if squeeze else value

    @property
    def threshold(self) -> float:
        return self.Lambda

    def grad_g(self, beta: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta, dtype=float)
        return 4.0 * beta * (beta * beta + self.epsilon ** 2)

    def H(self, beta: np.ndarray) -> np.ndarray:
        return (self.constraint_value(beta) - self.g_min) / self.D

    def grad_H(self, beta: np.ndarray) -> np.ndarray:
        return self.grad_g(beta) / self.D

    def normal(self, beta: np.ndarray) -> np.ndarray:
        gradient, squeeze = _as_2d(self.grad_g(beta))
        norm = np.linalg.norm(gradient, axis=1, keepdims=True)
        out = np.divide(gradient, norm, out=np.zeros_like(gradient), where=norm > 0)
        return out[0] if squeeze else out

    def boundary_point(self, direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        objective = lambda t: float(self.constraint_value(t * direction)) - self.Lambda
        upper = 1.0
        while objective(upper) < 0.0:
            upper *= 2.0
        t = brentq(objective, 0.0, upper, xtol=1e-14, rtol=8.9e-16, maxiter=200)
        return t * direction

    def j_block_vectors(self, beta: np.ndarray) -> np.ndarray:
        """``w_l = -grad_{I_l} g(beta)``.

        The quartic normal is proportional to ``grad g``, not to ``beta``, so
        the ball's ``[s beta]_x`` block would generally violate ``J n = 0`` here.
        """
        gradient, _ = _as_2d(self.grad_g(beta))
        return (-gradient).reshape(gradient.shape[0], -1, 3)

    # ---- Euclidean projection via the KKT system ----
    def _solve_coordinates(self, z_abs: np.ndarray, mu) -> np.ndarray:
        """Solve ``b + 4 mu b (b^2 + eps^2) = |z|`` for ``b >= 0``, elementwise.

        The equation is the depressed cubic ``b^3 + p b + q = 0`` with
        ``p = (1 + 4 mu eps^2)/(4 mu) > 0`` and ``q = -|z|/(4 mu)``.  With
        ``p > 0`` there is exactly one real root, given stably by the hyperbolic
        form ``b = -2 sqrt(p/3) sinh( arcsinh( q / (2 (p/3)^{3/2}) ) / 3 )``.
        The map is strictly increasing in ``b`` (derivative
        ``1 + 12 mu b^2 + 4 mu eps^2 > 0``), so the root is unique.
        """
        mu_array = np.asarray(mu, dtype=float)
        scalar_mu = mu_array.ndim == 0
        if scalar_mu and mu_array <= 0.0:
            return z_abs.copy()
        safe = np.where(mu_array > 0.0, mu_array, 1.0)
        p = (1.0 + 4.0 * safe * self.epsilon ** 2) / (4.0 * safe)
        q = -z_abs / (4.0 * safe)
        scale = (p / 3.0) ** 1.5
        theta = np.arcsinh(q / (2.0 * scale)) / 3.0
        root = -2.0 * np.sqrt(p / 3.0) * np.sinh(theta)
        # mu = 0 leaves the point unchanged (the map is the identity there).
        return np.where(np.broadcast_to(mu_array > 0.0, root.shape), root, z_abs)

    def _project_one(self, z: np.ndarray) -> tuple[np.ndarray, float]:
        """Project a single infeasible point; returns ``(b, mu)``."""
        sign = np.sign(z)
        z_abs = np.abs(z)

        def gap(mu: float) -> float:
            b = self._solve_coordinates(z_abs, mu)
            return float(np.sum((b * b + self.epsilon ** 2) ** 2)) - self.Lambda

        # gap(0) = g(z) - Lambda > 0; gap is decreasing and tends to
        # d eps^4 - Lambda < 0, so a bracket always exists.
        mu_high = 1.0
        for _ in range(200):
            if gap(mu_high) <= 0.0:
                break
            mu_high *= 2.0
        else:  # pragma: no cover
            raise RuntimeError("failed to bracket the projection multiplier mu")
        mu = brentq(gap, 0.0, mu_high, xtol=1e-14, rtol=8.9e-16, maxiter=300)
        return sign * self._solve_coordinates(z_abs, mu), float(mu)

    def project(self, z: np.ndarray, n_bisect: int = 100) -> ProjectionOutcome:
        """Exact Euclidean projection onto ``{g <= Lambda}`` (never radial scaling).

        The outer multiplier ``mu`` is found by a bracketed bisection run on all
        infeasible rows simultaneously; ``mu -> g(b(mu))`` is monotonically
        decreasing, so bisection is unconditionally reliable, and ``n_bisect``
        halvings of the bracket reach machine precision.  :meth:`_project_one`
        keeps the scalar ``brentq`` version, which the checks use as an
        independent reference.
        """
        z2, squeeze = _as_2d(z)
        beta = z2.copy()
        outside = self.constraint_value(z2) > self.Lambda
        worst_kkt = 0.0

        rows = np.nonzero(outside)[0]
        if rows.size:
            z_rows = z2[rows]
            sign, z_abs = np.sign(z_rows), np.abs(z_rows)

            def gap(mu_column: np.ndarray) -> np.ndarray:
                b = self._solve_coordinates(z_abs, mu_column)
                return np.sum((b * b + self.epsilon ** 2) ** 2, axis=1) - self.Lambda

            # gap(0) > 0 by construction; grow the upper end until gap <= 0.
            low = np.zeros(rows.size)
            high = np.ones(rows.size)
            for _ in range(200):
                need = gap(high[:, None]) > 0.0
                if not need.any():
                    break
                high[need] *= 2.0
            else:  # pragma: no cover
                raise RuntimeError("failed to bracket the projection multiplier mu")

            for _ in range(n_bisect):
                mid = 0.5 * (low + high)
                positive = gap(mid[:, None]) > 0.0
                low = np.where(positive, mid, low)
                high = np.where(positive, high, mid)
            mu = 0.5 * (low + high)

            b = sign * self._solve_coordinates(z_abs, mu[:, None])
            beta[rows] = b
            worst_kkt = float(
                np.abs(b + mu[:, None] * self.grad_g(b) - z_rows).max()
            )
        excess = float((self.constraint_value(beta) - self.Lambda).max())
        out = beta[0] if squeeze else beta
        return ProjectionOutcome(out, outside, worst_kkt, excess)

    @property
    def box_half_width(self) -> float:
        """``b = sqrt( sqrt(Lambda - (d-1) eps^4) - eps^2 )``."""
        inner = math.sqrt(self.Lambda - (self.d - 1) * self.epsilon ** 4)
        return math.sqrt(inner - self.epsilon ** 2)

    def sample_uniform(
        self, rng: np.random.Generator, n: int, max_rounds: int = 10_000
    ) -> np.ndarray:
        """Uniform on ``K`` by rejection sampling from ``[-b, b]^d``."""
        half = self.box_half_width
        accepted: list[np.ndarray] = []
        total = 0
        kept = 0
        for _ in range(max_rounds):
            proposals = rng.uniform(-half, half, size=(max(n, 256), self.d))
            total += proposals.shape[0]
            good = proposals[self.constraint_value(proposals) <= self.Lambda]
            kept += good.shape[0]
            if good.size:
                accepted.append(good)
            if sum(a.shape[0] for a in accepted) >= n:
                break
        else:  # pragma: no cover
            raise RuntimeError("rejection sampler failed to fill the requested draws")
        out = np.vstack(accepted)[:n]
        self.last_acceptance_rate = kept / total
        return out


def make_geometry(name: str, cfg: ExperimentConfig) -> Geometry:
    if name == "ball":
        return BallGeometry(cfg.d)
    if name == "quartic":
        return QuarticGeometry(cfg.d, cfg.epsilon, cfg.Lambda)
    raise ValueError(f"unknown geometry {name!r}")


# ==========================================================================
# 5. Block state-dependent skew-symmetric matrix
# ==========================================================================
def apply_J(
    beta: np.ndarray,
    v: np.ndarray,
    geometry: Geometry,
    scales: np.ndarray,
) -> np.ndarray:
    """Matrix-free ``J(beta) v`` via block cross products.

    Each block acts as ``[s_l w_l]_x v_{I_l} = (s_l w_l) x v_{I_l}``, so no
    ``d x d`` matrix is ever formed.  The block strength multiplies ``w_l``
    exactly once.
    """
    beta2, squeeze = _as_2d(beta)
    v2, _ = _as_2d(v)
    w = geometry.j_block_vectors(beta2) * np.asarray(scales)[None, :, None]
    blocks = np.cross(w, v2.reshape(v2.shape[0], -1, 3))
    out = blocks.reshape(v2.shape[0], -1)
    return out[0] if squeeze else out


def hat(w: np.ndarray) -> np.ndarray:
    """``[w]_x``: the 3x3 cross-product matrix with ``[w]_x v = w x v``."""
    w1, w2, w3 = float(w[0]), float(w[1]), float(w[2])
    return np.array(
        [
            [0.0, -w3, w2],
            [w3, 0.0, -w1],
            [-w2, w1, 0.0],
        ]
    )


def build_J(beta: np.ndarray, geometry: Geometry, scales: np.ndarray) -> np.ndarray:
    """Explicit ``d x d`` matrix ``J(beta)`` — for verification, not for sampling."""
    beta = np.asarray(beta, dtype=float).ravel()
    d = beta.size
    w = geometry.j_block_vectors(beta[None, :])[0] * np.asarray(scales)[:, None]
    J = np.zeros((d, d))
    for block, vector in enumerate(w):
        lo = 3 * block
        J[lo : lo + 3, lo : lo + 3] = hat(vector)
    return J


def divergence_J(
    beta: np.ndarray, geometry: Geometry, scales: np.ndarray, step: float = 1e-5
) -> np.ndarray:
    """Centred finite-difference ``div(J)_i = sum_j dJ_ij/dbeta_j``."""
    beta = np.asarray(beta, dtype=float).ravel()
    divergence = np.zeros(beta.size)
    for j in range(beta.size):
        plus, minus = beta.copy(), beta.copy()
        plus[j] += step
        minus[j] -= step
        divergence += (
            build_J(plus, geometry, scales)[:, j]
            - build_J(minus, geometry, scales)[:, j]
        ) / (2.0 * step)
    return divergence


# ==========================================================================
# 6. Random streams (shared across the four methods)
# ==========================================================================
GEOMETRY_ID: dict[str, int] = {"unit ball": 11, "quartic set": 22}


@dataclass
class Streams:
    """Pre-generated randomness, identical for all four methods.

    Within one replicate and geometry every method sees the same starting
    coefficients, the same mini-batch index sequence and the same Gaussian
    increments; replicates use independent sub-streams.  Batch indices and
    Gaussian increments come from **separate** spawned streams, so the noise is
    independent of the current mini-batch.
    """

    beta_init: np.ndarray            # (R, d)
    batch_index: np.ndarray          # (R, n_iterations, m) int32
    noise: np.ndarray                # (R, n_iterations, d)
    seed_key: tuple[int, ...]

    @property
    def n_repeats(self) -> int:
        return self.beta_init.shape[0]

    @property
    def n_iterations(self) -> int:
        return self.noise.shape[1]


def make_streams(
    cfg: ExperimentConfig,
    geometry: Geometry,
    n_iterations: int,
    n_repeats: int,
    n_train: int,
) -> Streams:
    """Build the shared randomness for one geometry."""
    key = (cfg.sampler_seed, GEOMETRY_ID[geometry.name])
    base = np.random.SeedSequence(list(key))
    init_ss, batch_ss, noise_ss = base.spawn(3)

    beta_init = geometry.sample_uniform(np.random.default_rng(init_ss), n_repeats)

    batch_index = np.empty((n_repeats, n_iterations, cfg.batch_size), dtype=np.int32)
    for r, seed in enumerate(batch_ss.spawn(n_repeats)):
        rng = np.random.default_rng(seed)
        for k in range(n_iterations):
            batch_index[r, k] = rng.choice(n_train, size=cfg.batch_size, replace=False)

    noise = np.empty((n_repeats, n_iterations, cfg.d))
    for r, seed in enumerate(noise_ss.spawn(n_repeats)):
        noise[r] = np.random.default_rng(seed).standard_normal((n_iterations, cfg.d))

    return Streams(beta_init, batch_index, noise, key)


# ==========================================================================
# 7. The common sampler
# ==========================================================================
@dataclass
class RunResult:
    """Checkpointed output of one (method, geometry) run over ``R`` replicates."""

    method: str
    geometry: str
    rho: float
    alpha: float
    step_size: float
    n_iterations: int
    n_repeats: int
    block_scales: tuple[float, ...]
    checkpoints: np.ndarray          # (n_ckpt,) iteration indices
    train_accuracy: np.ndarray       # (n_ckpt, R)
    test_accuracy: np.ndarray        # (n_ckpt, R)
    beta: np.ndarray                 # (n_ckpt, R, d)
    train_loss: np.ndarray           # (n_ckpt, R)  full U on the training set
    constraint: np.ndarray           # (n_ckpt, R)  ||beta||^2 or g(beta)
    anchor: np.ndarray               # (n_ckpt, R)  a(beta)
    projection_rate: float
    max_kkt_residual: float
    max_feasibility_excess: float
    n_nonfinite: int
    runtime: float
    full_gradient: bool = False
    #: Mean over iterations and replicates of ||alpha J v|| / ||v||: how much
    #: larger the added non-reversible term is than the reversible drift.
    drift_ratio: float = 0.0
    #: Mean per-step displacement ||proposal - beta|| before projection.
    step_displacement: float = 0.0

    @property
    def simulated_time(self) -> np.ndarray:
        """``t = k h`` — the axis on which different step sizes are comparable."""
        return self.checkpoints * self.step_size

    def mean_std(self, which: str = "test_accuracy") -> tuple[np.ndarray, np.ndarray]:
        """Across-replicate mean and sample standard deviation (``ddof=1``)."""
        values = getattr(self, which)
        return values.mean(axis=1), values.std(axis=1, ddof=1)


def run_sampler(
    dataset: Dataset,
    geometry: Geometry,
    streams: Streams,
    *,
    method: str,
    rho: float,
    alpha: float,
    scales: np.ndarray,
    step_size: float,
    n_iterations: int,
    checkpoint_every: int,
    use_full_gradient: bool = False,
) -> RunResult:
    """One implementation; the four methods differ only in ``rho`` and ``alpha``.

        beta_{k+1} = Pi_K[ beta_k - h a_k (v_k + alpha J(beta_k) v_k)
                           + sqrt(2 h a_k) xi_k ]

    No ``grad a`` correction is added and ``J`` is never rescaled.
    """
    X_train, y_train = dataset.X_train, dataset.y_train
    X_test, y_test = dataset.X_test, dataset.y_test
    h = step_size
    beta = streams.beta_init.copy()
    n_repeats = beta.shape[0]

    checkpoints = [0]
    train_accuracy = [accuracy(beta, X_train, y_train)]
    test_accuracy = [accuracy(beta, X_test, y_test)]
    beta_history = [beta.copy()]
    train_loss = [potential_U(beta, X_train, y_train)]
    constraint = [geometry.constraint_value(beta)]
    anchor = [np.exp(-rho * geometry.H(beta))]

    n_projected = 0
    n_nonfinite = 0
    worst_kkt = 0.0
    worst_excess = -np.inf
    ratio_sum = 0.0
    displacement_sum = 0.0

    start = time.perf_counter()
    for k in range(n_iterations):
        if use_full_gradient:
            gradient = full_gradient(beta, X_train, y_train)
        else:
            gradient = minibatch_gradient(
                beta, X_train, y_train, streams.batch_index[:, k, :]
            )

        # v = Ghat + rho grad_H: the anchor derivative is added ONCE and is not
        # multiplied by n_train/m.
        v = gradient if rho == 0.0 else gradient + rho * geometry.grad_H(beta)
        a = np.ones(n_repeats) if rho == 0.0 else np.exp(-rho * geometry.H(beta))

        if alpha == 0.0:
            drift = v
        else:
            non_reversible = alpha * apply_J(beta, v, geometry, scales)
            drift = v + non_reversible
            # Diagnostic: how big is the added term relative to the reversible one?
            ratio_sum += float(
                np.mean(
                    np.linalg.norm(non_reversible, axis=1)
                    / np.maximum(np.linalg.norm(v, axis=1), 1e-300)
                )
            )
        proposal = (
            beta
            - h * a[:, None] * drift
            + np.sqrt(2.0 * h * a)[:, None] * streams.noise[:, k, :]
        )

        displacement_sum += float(np.mean(np.linalg.norm(proposal - beta, axis=1)))

        bad = ~np.isfinite(proposal).all(axis=1)
        if np.any(bad):
            n_nonfinite += int(bad.sum())
            proposal[bad] = beta[bad]          # freeze; counted and reported

        outcome = geometry.project(proposal)
        beta = outcome.beta
        n_projected += int(outcome.projected.sum())
        worst_kkt = max(worst_kkt, outcome.max_kkt_residual)
        worst_excess = max(worst_excess, outcome.max_feasibility_excess)

        if (k + 1) % checkpoint_every == 0:
            checkpoints.append(k + 1)
            train_accuracy.append(accuracy(beta, X_train, y_train))
            test_accuracy.append(accuracy(beta, X_test, y_test))
            beta_history.append(beta.copy())
            train_loss.append(potential_U(beta, X_train, y_train))
            constraint.append(geometry.constraint_value(beta))
            anchor.append(np.exp(-rho * geometry.H(beta)))
    runtime = time.perf_counter() - start

    return RunResult(
        method=method,
        geometry=geometry.name,
        rho=rho,
        alpha=alpha,
        step_size=h,
        n_iterations=n_iterations,
        n_repeats=n_repeats,
        block_scales=tuple(float(x) for x in np.atleast_1d(scales)),
        checkpoints=np.asarray(checkpoints),
        train_accuracy=np.asarray(train_accuracy),
        test_accuracy=np.asarray(test_accuracy),
        beta=np.asarray(beta_history),
        train_loss=np.asarray(train_loss),
        constraint=np.asarray(constraint),
        anchor=np.asarray(anchor),
        projection_rate=n_projected / (n_iterations * n_repeats),
        max_kkt_residual=worst_kkt,
        max_feasibility_excess=float(worst_excess),
        n_nonfinite=n_nonfinite,
        runtime=runtime,
        full_gradient=use_full_gradient,
        drift_ratio=ratio_sum / n_iterations,
        step_displacement=displacement_sum / n_iterations,
    )


def run_all_methods(
    dataset: Dataset,
    geometry: Geometry,
    cfg: ExperimentConfig,
    *,
    n_iterations: int | None = None,
    n_repeats: int | None = None,
    step_size: float | None = None,
    checkpoint_every: int | None = None,
    use_full_gradient: bool = False,
    verbose: bool = True,
) -> dict[str, RunResult]:
    """Run all four methods on one geometry with shared randomness."""
    n_iterations = n_iterations or cfg.n_iterations
    n_repeats = n_repeats or cfg.n_repeats
    step_size = cfg.step_size if step_size is None else step_size
    checkpoint_every = checkpoint_every or cfg.checkpoint_every

    streams = make_streams(cfg, geometry, n_iterations, n_repeats, dataset.n_train)
    results: dict[str, RunResult] = {}
    for name, rho, alpha in METHODS:
        result = run_sampler(
            dataset, geometry, streams,
            method=name, rho=rho, alpha=alpha, scales=cfg.scales,
            step_size=step_size, n_iterations=n_iterations,
            checkpoint_every=checkpoint_every, use_full_gradient=use_full_gradient,
        )
        results[name] = result
        if verbose:
            mean, sd = result.mean_std("test_accuracy")
            print(
                f"    {name:<34} final test acc {mean[-1]:.4f} +/- {sd[-1]:.4f}  "
                f"proj rate {result.projection_rate:.4f}  "
                f"nonfinite {result.n_nonfinite}  {result.runtime:5.1f}s",
                flush=True,
            )
    return results


# ==========================================================================
# 8. Mathematical implementation checks
# ==========================================================================
def check_gradient_finite_difference(
    dataset: Dataset, seed: int = 7, step: float = 1e-6
) -> dict[str, float]:
    """Compare ``full_gradient`` with centred finite differences of ``U``."""
    rng = np.random.default_rng(seed)
    beta = rng.normal(scale=0.3, size=dataset.d)
    analytic = full_gradient(beta, dataset.X_train, dataset.y_train)
    numerical = np.zeros_like(beta)
    for i in range(beta.size):
        plus, minus = beta.copy(), beta.copy()
        plus[i] += step
        minus[i] -= step
        numerical[i] = (
            potential_U(plus, dataset.X_train, dataset.y_train)
            - potential_U(minus, dataset.X_train, dataset.y_train)
        ) / (2.0 * step)
    absolute = float(np.abs(analytic - numerical).max())
    return {
        "max_absolute_error": absolute,
        "max_relative_error": absolute / float(np.abs(analytic).max()),
    }


def check_minibatch_scaling_exhaustive(
    n_toy: int = 8, m_toy: int = 3, d_toy: int = 3, seed: int = 11
) -> dict[str, float]:
    """Enumerate **every** batch of a toy data set and check unbiasedness.

    Averaging ``(n/m) X_B^T (sigmoid - y)`` over all ``C(n, m)`` subsets must
    reproduce the full summed gradient exactly.  This is what pins down the
    ``n_train / m`` factor: a plain batch average would be off by that factor.
    """
    from itertools import combinations

    rng = np.random.default_rng(seed)
    X = rng.normal(scale=np.sqrt(2.0), size=(n_toy, d_toy))
    y = (rng.random(n_toy) < 0.5).astype(float)
    beta = rng.normal(scale=0.4, size=d_toy)

    batches = np.array(list(combinations(range(n_toy), m_toy)), dtype=np.int32)
    estimates = minibatch_gradient(
        np.repeat(beta[None, :], batches.shape[0], axis=0), X, y, batches
    )
    exact = full_gradient(beta, X, y)
    averaged = estimates.mean(axis=0)

    # What an unscaled batch average would give, for contrast.
    unscaled = averaged * (m_toy / n_toy)
    return {
        "n_batches": float(batches.shape[0]),
        "max_abs_error_scaled": float(np.abs(averaged - exact).max()),
        "relative_error_scaled": float(
            np.abs(averaged - exact).max() / np.abs(exact).max()
        ),
        "max_abs_error_if_unscaled": float(np.abs(unscaled - exact).max()),
    }


def check_matrix_identities(
    geometry: Geometry, scales: np.ndarray, n_points: int = 12, seed: int = 13
) -> dict[str, float]:
    """Verify ``J^T = -J``, ``div J = 0``, ``J n = 0`` on the boundary, ``J grad_H = 0``."""
    rng = np.random.default_rng(seed)
    d = geometry.d
    interior = [np.zeros(d)] + [
        rng.normal(scale=0.35, size=d) for _ in range(n_points)
    ]
    boundary = [
        geometry.boundary_point(rng.normal(size=d)) for _ in range(n_points)
    ]

    skew = 0.0
    divergence = 0.0
    grad_h = 0.0
    matrix_free = 0.0
    for beta in interior + boundary:
        J = build_J(beta, geometry, scales)
        skew = max(skew, float(np.abs(J + J.T).max()))
        divergence = max(
            divergence, float(np.abs(divergence_J(beta, geometry, scales)).max())
        )
        grad_h = max(
            grad_h, float(np.abs(J @ geometry.grad_H(beta).ravel()).max())
        )
        v = rng.normal(size=d)
        matrix_free = max(
            matrix_free,
            float(np.abs(apply_J(beta, v, geometry, scales) - J @ v).max()),
        )

    tangency = 0.0
    boundary_error = 0.0
    for beta in boundary:
        J = build_J(beta, geometry, scales)
        normal = geometry.normal(beta).ravel()
        tangency = max(tangency, float(np.abs(J @ normal).max()))
        boundary_error = max(
            boundary_error,
            abs(float(geometry.constraint_value(beta)) - geometry.threshold),
        )
    return {
        "max_skew_error": skew,
        "max_divergence": divergence,
        "max_tangency_on_boundary": tangency,
        "max_J_gradH": grad_h,
        "max_matrix_free_vs_explicit": matrix_free,
        "max_boundary_residual": boundary_error,
    }


def check_ball_J_fails_on_quartic(
    cfg: ExperimentConfig, n_points: int = 8, seed: int = 17
) -> dict[str, float]:
    """Show that the **ball** matrix violates ``J n = 0`` on the quartic boundary.

    The ball's normal is parallel to ``beta``; the quartic normal is parallel to
    ``grad g``, whose coordinates are ``beta_i`` reweighted by
    ``4(beta_i^2 + eps^2)``.  Those directions differ unless every ``|beta_i|``
    is equal, so the unmodified block ``[s beta_I]_x`` is not tangential there.
    """
    rng = np.random.default_rng(seed)
    quartic = QuarticGeometry(cfg.d, cfg.epsilon, cfg.Lambda)
    ball = BallGeometry(cfg.d)
    correct = 0.0
    wrong = 0.0
    for _ in range(n_points):
        beta = quartic.boundary_point(rng.normal(size=cfg.d))
        normal = quartic.normal(beta).ravel()
        correct = max(
            correct,
            float(np.abs(build_J(beta, quartic, cfg.scales) @ normal).max()),
        )
        wrong = max(
            wrong, float(np.abs(build_J(beta, ball, cfg.scales) @ normal).max())
        )
    return {"quartic_J_tangency": correct, "ball_J_on_quartic_boundary": wrong}


def check_anchor_bounds(
    geometry: Geometry, rho: float = RHO_ANCHORED, n_points: int = 4000, seed: int = 19
) -> dict[str, float]:
    """On ``K``: ``H_K in [0, 1]`` and therefore ``1/2 <= a <= 1``."""
    rng = np.random.default_rng(seed)
    beta = geometry.sample_uniform(rng, n_points)
    H = geometry.H(beta)
    a = np.exp(-rho * H)
    return {
        "min_H": float(H.min()),
        "max_H": float(H.max()),
        "min_a": float(a.min()),
        "max_a": float(a.max()),
        "bounds_hold": float(
            bool(H.min() >= -1e-12 and H.max() <= 1 + 1e-12
                 and a.min() >= 0.5 - 1e-12 and a.max() <= 1 + 1e-12)
        ),
    }


def check_projection(
    geometry: Geometry, n_points: int = 40, scale: float = 1.6, seed: int = 23
) -> dict[str, float]:
    """Feasibility and KKT residuals, plus a radial-scaling comparison."""
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=scale, size=(n_points, geometry.d))
    outcome = geometry.project(z)
    n_projected = int(outcome.projected.sum())

    # Radial scaling: t z with g(t z) = threshold. Feasible, but not the
    # Euclidean projection unless the set is a Euclidean ball.
    radial_worse = 0
    radial_gap = 0.0
    for row in np.nonzero(outcome.projected)[0]:
        objective = lambda t: (
            float(geometry.constraint_value(t * z[row])) - geometry.threshold
        )
        t = brentq(objective, 0.0, 1.0, xtol=1e-14, maxiter=200)
        d_radial = float(np.sum((t * z[row] - z[row]) ** 2))
        d_exact = float(np.sum((outcome.beta[row] - z[row]) ** 2))
        if d_radial > d_exact + 1e-12:
            radial_worse += 1
        radial_gap = max(radial_gap, d_radial - d_exact)
    return {
        "n_projected": float(n_projected),
        "max_kkt_residual": outcome.max_kkt_residual,
        "max_feasibility_excess": outcome.max_feasibility_excess,
        "n_radial_strictly_worse": float(radial_worse),
        "max_radial_excess_distance": radial_gap,
    }


def check_J_gradU0_nonzero(
    dataset: Dataset,
    geometry: Geometry,
    scales: np.ndarray,
    rho: float = RHO_ANCHORED,
    n_points: int = 8,
    seed: int = 29,
) -> dict[str, float]:
    """``J grad_H = 0`` exactly, but ``J grad_U0 = J grad_U`` must be nonzero.

    Otherwise the non-reversible term would do nothing at all.
    """
    rng = np.random.default_rng(seed)
    beta = geometry.sample_uniform(rng, n_points)
    grad_U0 = full_gradient(beta, dataset.X_train, dataset.y_train) + rho * geometry.grad_H(beta)
    J_grad = apply_J(beta, grad_U0, geometry, scales)
    norms = np.linalg.norm(J_grad, axis=1)
    return {
        "min_norm_J_gradU0": float(norms.min()),
        "median_norm_J_gradU0": float(np.median(norms)),
        "median_norm_gradU0": float(np.median(np.linalg.norm(grad_U0, axis=1))),
    }


def gradient_noise_report(
    dataset: Dataset,
    geometry: Geometry,
    cfg: ExperimentConfig,
    n_draws: int = 400,
    seed: int = 31,
) -> dict[str, float]:
    """Quantify how much ``J`` amplifies mini-batch gradient noise.

    The continuous-time invariance argument assumes the *exact* gradient.  With
    a stochastic gradient, ``J`` multiplies the gradient **error** as well as
    the gradient, so the added term injects extra variance that the
    ``sqrt(2 h a)`` term does not compensate.
    """
    rng = np.random.default_rng(seed)
    beta = geometry.sample_uniform(rng, 1)
    exact = full_gradient(beta, dataset.X_train, dataset.y_train)
    batches = np.stack(
        [rng.choice(dataset.n_train, cfg.batch_size, replace=False) for _ in range(n_draws)]
    )
    estimates = minibatch_gradient(
        np.repeat(beta, n_draws, axis=0), dataset.X_train, dataset.y_train, batches
    )
    error = np.linalg.norm(estimates - exact, axis=1).mean()

    J_exact = apply_J(beta, exact, geometry, cfg.scales)
    J_estimates = apply_J(
        np.repeat(beta, n_draws, axis=0), estimates, geometry, cfg.scales
    )
    J_error = np.linalg.norm(J_estimates - J_exact, axis=1).mean()

    h, a_typical = cfg.step_size, 0.75
    return {
        "norm_exact_gradient": float(np.linalg.norm(exact)),
        "mean_gradient_error": float(error),
        "mean_J_gradient_error": float(J_error),
        "amplification_factor": float(J_error / error),
        "step_from_gradient_noise": float(h * error),
        "step_from_J_gradient_noise": float(h * J_error),
        "injected_langevin_step": float(np.sqrt(2 * h * a_typical) * np.sqrt(cfg.d)),
    }


# ==========================================================================
# 9. Summaries
# ==========================================================================
def iterations_to_target(result: RunResult, target: float) -> np.ndarray:
    """First checkpoint iteration at which each replicate's TEST accuracy >= target.

    ``nan`` where a replicate never reaches the target.  Per replicate, never
    on the across-replicate mean.
    """
    reached = result.test_accuracy >= target              # (n_ckpt, R)
    out = np.full(result.n_repeats, np.nan)
    for r in range(result.n_repeats):
        hits = np.nonzero(reached[:, r])[0]
        if hits.size:
            out[r] = result.checkpoints[hits[0]]
    return out


def summarise(
    results: dict[str, RunResult], target: float
) -> "pd.DataFrame":  # noqa: F821
    """One row per method: final accuracy, time-to-target, projection, cost."""
    import pandas as pd

    rows = []
    for name, result in results.items():
        train_mean, train_sd = result.mean_std("train_accuracy")
        test_mean, test_sd = result.mean_std("test_accuracy")
        hits = iterations_to_target(result, target)
        reached = np.isfinite(hits)
        seconds_per_iteration_per_replicate = (
            result.runtime / result.n_iterations / result.n_repeats
        )
        rows.append(
            {
                "method": name,
                "rho": result.rho,
                "alpha": result.alpha,
                "final_train_acc": train_mean[-1],
                "final_train_sd": train_sd[-1],
                "final_test_acc": test_mean[-1],
                "final_test_sd": test_sd[-1],
                "frac_reaching_target": float(reached.mean()),
                "median_iters_to_target": (
                    float(np.median(hits[reached])) if reached.any() else np.nan
                ),
                "median_secs_to_target": (
                    float(np.median(hits[reached])) * seconds_per_iteration_per_replicate
                    if reached.any()
                    else np.nan
                ),
                "projection_rate": result.projection_rate,
                "drift_ratio_alphaJv_over_v": result.drift_ratio,
                "mean_step_displacement": result.step_displacement,
                "max_kkt_residual": result.max_kkt_residual,
                "n_nonfinite": result.n_nonfinite,
                "runtime_s": result.runtime,
                "secs_per_iter_per_replicate": seconds_per_iteration_per_replicate,
            }
        )
    return pd.DataFrame(rows)


def coefficient_moments(result: RunResult) -> dict[str, np.ndarray]:
    """Across-replicate mean and sd of each coordinate at every checkpoint."""
    return {
        "mean": result.beta.mean(axis=1),                  # (n_ckpt, d)
        "sd": result.beta.std(axis=1, ddof=1),             # (n_ckpt, d)
        "norm_mean": np.linalg.norm(result.beta, axis=2).mean(axis=1),
    }


# ==========================================================================
# 10. Figures
# ==========================================================================
def _stagger(values: Sequence[float], min_gap: float) -> list[float]:
    """Push overlapping label positions apart while preserving their order."""
    order = np.argsort(values)
    placed = np.asarray(values, dtype=float).copy()
    for rank in range(1, len(order)):
        lower, upper = order[rank - 1], order[rank]
        if placed[upper] - placed[lower] < min_gap:
            placed[upper] = placed[lower] + min_gap
    return list(placed)


def _draw_panel(ax, results: dict[str, RunResult], which: str, ylim, label_gap: float):
    finals = []
    for name, result in results.items():
        style = METHOD_STYLE[name]
        mean, sd = result.mean_std(which)
        lower = np.clip(mean - sd, 0.0, 1.0)      # clip the DISPLAYED band only
        upper = np.clip(mean + sd, 0.0, 1.0)
        ax.fill_between(
            result.checkpoints, lower, upper,
            color=style["color"], alpha=0.16, lw=0,
            zorder=2 if name.startswith("Non-reversible anchored") else 1,
        )
        ax.plot(
            result.checkpoints, mean,
            color=style["color"], ls=style["ls"], lw=style["lw"], label=name,
            zorder=4 if name.startswith("Non-reversible anchored") else 3,
        )
        finals.append((name, float(mean[-1]), result.checkpoints[-1]))

    positions = _stagger([value for _, value, _ in finals], label_gap)
    for (name, _, last), y in zip(finals, positions):
        ax.annotate(
            name.replace("Non-reversible", "Non-rev.").replace(" Langevin", ""),
            xy=(last, y), xytext=(6, 0), textcoords="offset points",
            color=METHOD_STYLE[name]["color"], fontsize=7.5, va="center",
            fontweight="bold" if name.startswith("Non-reversible anchored") else "normal",
        )
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)


def make_caption(cfg: ExperimentConfig, geometry_label: str, result: RunResult) -> str:
    """Caption carrying every setting the figure depends on."""
    return (
        f"d = {cfg.d};  constraint: {geometry_label};  step size h = {result.step_size:.3g};  "
        f"mini-batch m = {cfg.batch_size};  repeats R = {result.n_repeats};  "
        f"iterations = {result.n_iterations};  anchor rho = log 2 "
        f"({RHO_ANCHORED:.4f}) for anchored methods, 0 otherwise;  "
        f"block strengths s = {tuple(float(x) for x in cfg.scales)}.  "
        "Lines are the across-replicate mean; bands are mean +/- 1 sample sd "
        "(ddof = 1) of single-iterate accuracy, i.e. repeat-run variability -- "
        "NOT confidence or posterior credible intervals."
    )


def plot_accuracy_figure(
    results: dict[str, RunResult],
    cfg: ExperimentConfig,
    geometry_label: str,
    output_dir: str,
    tag: str,
    ylim: tuple[float, float] = (0.0, 1.0),
    title_extra: str = "",
) -> list[str]:
    """Training accuracy (left) and test accuracy (right) versus iteration."""
    import os

    import matplotlib.pyplot as plt

    any_result = next(iter(results.values()))
    zoomed = ylim != (0.0, 1.0)
    gap = (ylim[1] - ylim[0]) * 0.035

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    _draw_panel(axes[0], results, "train_accuracy", ylim, gap)
    _draw_panel(axes[1], results, "test_accuracy", ylim, gap)
    axes[0].set_title(f"Training accuracy  (n = {cfg.n_train})", fontsize=11)
    axes[1].set_title(f"Test accuracy  (n = {cfg.n_test})", fontsize=11)
    axes[0].legend(fontsize=7.5, loc="lower right", framealpha=0.92)

    fig.suptitle(
        f"Non-reversible anchored Langevin — {geometry_label}, d = {cfg.d}{title_extra}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    fig.text(
        0.5, 0.015, make_caption(cfg, geometry_label, any_result),
        ha="center", va="bottom", fontsize=7.4, wrap=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir, f"{tag}{'_zoom' if zoomed else ''}")
    paths = []
    for extension, kwargs in ((".png", {"dpi": 300}), (".pdf", {})):
        path = stem + extension
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_sensitivity_figure(
    sensitivity: dict[float, dict[str, RunResult]],
    cfg: ExperimentConfig,
    geometry_label: str,
    output_dir: str,
    tag: str,
) -> list[str]:
    """Step-size sensitivity against **simulated time** ``t = k h``."""
    import os

    import matplotlib.pyplot as plt

    divisors = sorted(sensitivity)
    fig, axes = plt.subplots(2, len(divisors), figsize=(5.0 * len(divisors), 8.2),
                             squeeze=False)
    for column, divisor in enumerate(divisors):
        runs = sensitivity[divisor]
        for name, result in runs.items():
            style = METHOD_STYLE[name]
            mean, sd = result.mean_std("test_accuracy")
            t = result.simulated_time
            axes[0][column].fill_between(
                t, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                color=style["color"], alpha=0.15, lw=0,
            )
            axes[0][column].plot(
                t, mean, color=style["color"], ls=style["ls"], lw=style["lw"], label=name
            )
            axes[1][column].plot(
                result.simulated_time, result.constraint.mean(axis=1),
                color=style["color"], ls=style["ls"], lw=style["lw"], label=name,
            )
        step = next(iter(runs.values())).step_size
        iterations = next(iter(runs.values())).n_iterations
        axes[0][column].set_title(
            f"h/{int(divisor)} = {step:.3g},  {iterations} iterations", fontsize=10
        )
        axes[0][column].set_ylim(0.0, 1.0)
        axes[0][column].set_ylabel("Accuracy (test)")
        axes[1][column].axhline(
            next(iter(runs.values())).constraint.max() * 0 + _threshold_of(geometry_label, cfg),
            color="k", ls=":", lw=1.0, label="constraint threshold",
        )
        axes[1][column].set_ylabel("mean constraint value")
        for row in (0, 1):
            axes[row][column].set_xlabel("Simulated time  t = k h")
            axes[row][column].grid(alpha=0.3)
    axes[0][0].legend(fontsize=7, loc="lower right")
    axes[1][0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"Step-size sensitivity at constant simulated time — {geometry_label}, d = {cfg.d}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.text(
        0.5, 0.012,
        f"R = {next(iter(sensitivity[divisors[0]].values())).n_repeats} repeats per "
        f"step size (sensitivity setting);  m = {cfg.batch_size};  "
        f"s = {tuple(float(x) for x in cfg.scales)};  total simulated time held fixed by scaling the "
        "iteration count with 1/h.",
        ha="center", fontsize=7.4,
    )
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir, tag)
    paths = []
    for extension, kwargs in ((".png", {"dpi": 300}), (".pdf", {})):
        path = stem + extension
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def _threshold_of(geometry_label: str, cfg: ExperimentConfig) -> float:
    return 1.0 if "ball" in geometry_label else cfg.Lambda


# ==========================================================================
# 11. Persistence
# ==========================================================================
def save_results(
    path: str,
    results_by_geometry: dict[str, dict[str, RunResult]],
    cfg: ExperimentConfig,
    dataset: Dataset,
    extra: dict | None = None,
) -> str:
    """Save every checkpoint array plus metadata so figures are regenerable."""
    import json
    import os
    import platform

    import matplotlib
    import pandas as pd
    import scipy
    import sklearn

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    meta_runs = []
    for geometry_label, results in results_by_geometry.items():
        for name, result in results.items():
            key = f"{geometry_label}|{name}".replace(" ", "_")
            arrays[f"{key}|checkpoints"] = result.checkpoints
            arrays[f"{key}|train_accuracy"] = result.train_accuracy
            arrays[f"{key}|test_accuracy"] = result.test_accuracy
            arrays[f"{key}|beta"] = result.beta
            arrays[f"{key}|train_loss"] = result.train_loss
            arrays[f"{key}|constraint"] = result.constraint
            arrays[f"{key}|anchor"] = result.anchor
            meta_runs.append(
                {
                    "geometry": geometry_label,
                    "method": name,
                    "rho": result.rho,
                    "alpha": result.alpha,
                    "step_size": result.step_size,
                    "n_iterations": result.n_iterations,
                    "n_repeats": result.n_repeats,
                    "block_scales": list(result.block_scales),
                    "projection_rate": result.projection_rate,
                    "max_kkt_residual": result.max_kkt_residual,
                    "max_feasibility_excess": result.max_feasibility_excess,
                    "n_nonfinite": result.n_nonfinite,
                    "runtime_s": result.runtime,
                    "drift_ratio": result.drift_ratio,
                    "step_displacement": result.step_displacement,
                    "full_gradient": result.full_gradient,
                }
            )
    np.savez_compressed(path, **arrays)

    metadata = {
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in cfg.__dict__.items()},
        "derived": {
            "n_train": cfg.n_train, "n_test": cfg.n_test,
            "rho_anchored": RHO_ANCHORED, "n_blocks": cfg.n_blocks,
        },
        "seeds": {
            "data_seed": cfg.data_seed,
            "split_seed": cfg.split_seed,
            "sampler_seed": cfg.sampler_seed,
        },
        "beta_true": dataset.beta_true.tolist(),
        "runs": meta_runs,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    if extra:
        metadata.update(extra)
    meta_path = os.path.splitext(path)[0] + "_metadata.json"
    with open(meta_path, "w") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    return meta_path


def check_projection_solvers_agree(
    geometry: "QuarticGeometry", n_points: int = 25, scale: float = 1.6, seed: int = 37
) -> dict[str, float]:
    """Vectorised bisection (used in the sampler) vs scalar ``brentq`` reference."""
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=scale, size=(n_points, geometry.d))
    fast = geometry.project(z).beta
    worst = 0.0
    for row in range(n_points):
        if geometry.constraint_value(z[row]) <= geometry.Lambda:
            reference = z[row]
        else:
            reference, _ = geometry._project_one(z[row])
        worst = max(worst, float(np.abs(fast[row] - reference).max()))
    return {"max_abs_difference": worst}
