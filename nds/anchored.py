r"""Anchored Langevin for a non-differentiable regularizer.

Swapping the Gaussian prior for a Laplace one makes the potential
non-differentiable, which is the setting the rest of this repository only
approximates: a central-difference surrogate of ``|w|`` smooths the kink by an
amount set by the difference step, and a subgradient scheme has an ``O(h)`` bias
that does not vanish with the discretisation alone.

**The target.**  With ``lambda`` the inverse Laplace scale,

.. code-block::

    U(w)   = sum_i [ log(1 + exp(x_i . w)) - y_i (x_i . w) ]  +  lambda sum_{j>=1} |w_j|
           = U_nll(w) + lambda ||w||_1     (the intercept is left unpenalised)

**The anchor.**  ``U_0`` replaces the kink with a smooth majorant of it,

.. code-block::

    U_0(w) = U_nll(w) + lambda sum_{j>=1} sqrt(w_j^2 + delta^2)

so ``U_0`` is smooth, ``U <= U_0``, and the gap is bounded by
``lambda (d - 1) delta``.

**The dynamics.**  Anchored Langevin (Gurbuzbalaban, Hu, Yuan and Zhu,
*Anchored Langevin Algorithms*, arXiv:2509.19455) follows the gradient of the
anchor and scales the diffusion by the exponential of the gap,

.. code-block::

    a(w) = exp(U(w) - U_0(w)) = exp(-lambda sum_j [sqrt(w_j^2 + delta^2) - |w_j|])  in (0, 1]
    dW_t = -a(W_t) grad U_0(W_t) dt + sqrt(2 a(W_t)) dB_t

and ``exp(-U)`` is exactly invariant.  The cleanest way to see it: the anchored
process is the ``U_0``-diffusion run under a random time change of rate ``a``,
and time-changing a diffusion with invariant density ``p`` at rate ``a`` leaves
invariant density proportional to ``p / a``, here
``exp(-U_0) / exp(U - U_0) = exp(-U)``.  Note what the gradient never touches:
the ``ell_1`` term enters only through the scalar clock ``a``, which is an
evaluation of ``U`` and ``U_0``, and in fact only of their penalties, since the
likelihood cancels in the difference.

**With an irreversible drift.**  The same time-change argument composes with the
skew term of this repository.  The irreversible ``U_0``-diffusion
``dY = -(I + J(Y)) grad U_0(Y) dt + Gamma_0(Y) dt + sqrt(2) dB`` has invariant
density ``exp(-U_0)``, so time-changing it at rate ``a`` gives

.. code-block::

    dW_t = a(W_t) [ -(I + J(W_t)) ghat_0(W_t) + (div J)(W_t) ] dt + sqrt(2 a(W_t)) dB_t

with invariant density ``exp(-U)`` for any skew ``J``, and still no derivative of
the non-differentiable term anywhere.  ``ghat_0`` is the central-difference
surrogate of ``grad U_0``, which is well behaved because the anchor is smooth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sampler import ChainResult, Geometry, _log_loss, _score
from .skew import SkewField, ZeroSkew
from .target import LogisticPosterior, _sigmoid, softplus


class LassoLogistic:
    """Logistic likelihood with a Laplace prior, plus its smooth anchor."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        penalty: float = 0.5,
        delta: float = 0.1,
        penalize_intercept: bool = False,
    ) -> None:
        self.nll = LogisticPosterior(X, y, prior_scale=None)
        self.X, self.y = self.nll.X, self.nll.y
        self.n, self.d = self.nll.n, self.nll.d
        self.penalty = float(penalty)
        self.delta = float(delta)
        self.weights = np.full(self.d, self.penalty)
        if not penalize_intercept:
            self.weights[0] = 0.0
        self.weights = self.weights[:, None]  # (d, 1)

    # ----------------------------------------------------------------- pieces
    def linear(self, W: np.ndarray) -> np.ndarray:
        return self.nll.linear(W)

    def negative_log_likelihood(self, W: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        Z = self.linear(W) if Z is None else Z
        return softplus(Z).sum(axis=0) - self.y @ Z

    def l1(self, W: np.ndarray) -> np.ndarray:
        return (self.weights * np.abs(W)).sum(axis=0)

    def smoothed_l1(self, W: np.ndarray) -> np.ndarray:
        return (self.weights * np.sqrt(W**2 + self.delta**2)).sum(axis=0)

    # -------------------------------------------------------------- potentials
    def potential(self, W: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        """The non-differentiable target potential ``U``."""
        return self.negative_log_likelihood(W, Z=Z) + self.l1(W)

    def anchor_potential(self, W: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        """The smooth anchor ``U_0``."""
        return self.negative_log_likelihood(W, Z=Z) + self.smoothed_l1(W)

    def log_weight(self, W: np.ndarray) -> np.ndarray:
        """``U - U_0`` per walker, shape (m,).

        The likelihood cancels, so this costs ``O(d)`` and never looks at the
        data.  It is at most zero and at least ``-lambda (d - 1) delta``, which
        bounds the clock ``a = exp(U - U_0)`` inside ``(0, 1]``.
        """
        return self.l1(W) - self.smoothed_l1(W)

    def clock(self, W: np.ndarray) -> np.ndarray:
        """``a(w) = exp(U - U_0)`` per walker, shape (1, m)."""
        return np.exp(self.log_weight(W))[None, :]

    @property
    def clock_floor(self) -> float:
        return float(np.exp(-self.weights.sum() * self.delta))

    # ------------------------------------------------------- derivative free
    def anchor_fd_grad(
        self, W: np.ndarray, eps: float = 1e-2, Z: np.ndarray | None = None
    ) -> np.ndarray:
        """Central-difference surrogate of ``grad U_0``, shape (d, m).

        Coordinate ``j`` differences both pieces of the anchor at ``w +/- eps
        e_j``; the likelihood shift enters only through the linear predictor, so
        no matrix product is repeated.  The anchor is smooth, which is what
        makes the difference quotient meaningful -- the same construction
        applied to ``U`` would be differencing across a kink.
        """
        Z = self.linear(W) if Z is None else Z
        out = np.empty((self.d, W.shape[1]))
        inv = 0.5 / eps
        for j in range(self.d):
            col = self.X[:, j : j + 1]
            shift = eps * col
            plus = softplus(Z + shift).sum(axis=0)
            minus = softplus(Z - shift).sum(axis=0)
            nll = (plus - minus) * inv - self.nll._Xty[j]
            if self.weights[j, 0]:
                wj = W[j]
                pen = self.weights[j, 0] * inv * (
                    np.sqrt((wj + eps) ** 2 + self.delta**2)
                    - np.sqrt((wj - eps) ** 2 + self.delta**2)
                )
            else:
                pen = 0.0
            out[j] = nll + pen
        return out

    def fd_grad(
        self, W: np.ndarray, eps: float = 1e-2, Z: np.ndarray | None = None
    ) -> np.ndarray:
        """Central-difference surrogate of ``grad U``, shape (d, m).

        Differencing ``|w|`` with step ``eps`` returns ``clip(w / eps, -1, 1)``,
        so this is the gradient of an ``eps``-Huber smoothing of the penalty
        rather than of the penalty itself: the kink is what makes the surrogate
        target the wrong law, which is the reason for the anchor.
        """
        Z = self.linear(W) if Z is None else Z
        out = np.empty((self.d, W.shape[1]))
        inv = 0.5 / eps
        for j in range(self.d):
            col = self.X[:, j : j + 1]
            shift = eps * col
            plus = softplus(Z + shift).sum(axis=0)
            minus = softplus(Z - shift).sum(axis=0)
            nll = (plus - minus) * inv - self.nll._Xty[j]
            pen = self.weights[j, 0] * inv * (np.abs(W[j] + eps) - np.abs(W[j] - eps))
            out[j] = nll + pen
        return out

    def predict_proba(self, W: np.ndarray, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ W)


def run_anchored_chain(
    target: LassoLogistic,
    skew: SkewField | None,
    step_size: float,
    n_iter: int,
    n_walkers: int = 24,
    seed: int = 0,
    X_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    W0: np.ndarray | None = None,
    geometry: Geometry | None = None,
    fd_eps: float = 1e-2,
    ref_mean: np.ndarray | None = None,
    ref_metric: np.ndarray | None = None,
    scheme: str = "anchored",
) -> ChainResult:
    """Anchored Langevin, optionally with an irreversible drift.

    One step, with ``a = exp(U - U_0)`` evaluated at the current state:

    .. code-block::

        w' = w + h a(w) [ -(D + J(w)) ghat_0(w) + (div J)(w) ] + sqrt(2 h a(w)) L xi

    ``scheme`` selects what the two baselines do instead, and both are biased:

    * ``"anchor"`` drops the clock and samples ``exp(-U_0)``, i.e. accepts the
      smoothing as the model.  Its bias is set by ``delta``, not by ``h``.
    * ``"fd"`` keeps the clock off and differences the *non-differentiable*
      potential, which is what a derivative-free method does when handed
      ``|w|``.  Central differences of ``|w|`` with step ``eps`` return
      ``clip(w / eps, -1, 1)``, the gradient of the ``eps``-Huber smoothing, so
      this chain targets an ``eps``-smoothed posterior however small ``h`` gets.
    """
    if scheme not in ("anchored", "anchor", "fd"):
        raise ValueError("scheme must be 'anchored', 'anchor' or 'fd'")
    skew = ZeroSkew() if skew is None else skew
    geo = Geometry.identity(target.d) if geometry is None else geometry
    rng = np.random.default_rng(seed)
    h = float(step_size)

    W = np.zeros((target.d, n_walkers)) if W0 is None else np.array(W0, float)
    acc_trace = np.empty((n_iter + 1, n_walkers))
    loss_trace = np.empty((n_iter + 1, n_walkers)) if X_eval is not None else None
    pot_trace = np.empty((n_iter + 1, n_walkers))
    track_error = ref_mean is not None and ref_metric is not None
    err_trace = np.empty((n_iter + 1, n_walkers)) if track_error else None
    W_sum = W.copy()
    clocks = np.empty(n_iter + 1)

    def error(mean_W: np.ndarray) -> np.ndarray:
        diff = mean_W - np.asarray(ref_mean).reshape(-1, 1)
        return np.sqrt(np.maximum((diff * (ref_metric @ diff)).sum(axis=0), 0.0))

    if X_eval is not None:
        P_sum = _sigmoid(X_eval @ W)
        acc_trace[0] = _score(P_sum, y_eval)
        loss_trace[0] = _log_loss(P_sum, y_eval)
    else:
        P_sum = None
        acc_trace[0] = np.nan
    pot_trace[0] = target.potential(W)
    clocks[0] = float(target.clock(W).mean())
    if track_error:
        err_trace[0] = error(W)

    for t in range(1, n_iter + 1):
        Z = target.linear(W)
        if scheme == "fd":
            scale = np.ones((1, n_walkers))
            G = target.fd_grad(W, eps=fd_eps, Z=Z)
        else:
            scale = target.clock(W) if scheme == "anchored" else np.ones((1, n_walkers))
            G = target.anchor_fd_grad(W, eps=fd_eps, Z=Z)
        drift = skew.divergence(W) - (geo.D @ G + skew.apply(W, G))
        noise = np.sqrt(2.0 * h * scale) * (geo.L @ rng.standard_normal((target.d, n_walkers)))
        W = W + h * scale * drift + noise

        W_sum += W
        pot_trace[t] = target.potential(W)
        clocks[t] = float(scale.mean())
        if track_error:
            err_trace[t] = error(W_sum / (t + 1))
        if P_sum is not None:
            P_sum += _sigmoid(X_eval @ W)
            P_mean = P_sum / (t + 1)
            acc_trace[t] = _score(P_mean, y_eval)
            loss_trace[t] = _log_loss(P_mean, y_eval)

    return ChainResult(
        accuracy=acc_trace,
        potential=pot_trace,
        acceptance=1.0,  # unadjusted
        step_size=h,
        posterior_mean=W_sum / (n_iter + 1),
        final_state=W,
        mean_error=err_trace,
        loss=loss_trace,
        meta={
            "label": skew.label,
            "n_walkers": n_walkers,
            "fd_eps": fd_eps,
            "scheme": scheme,
            "mean_clock": float(clocks.mean()),
            "clock_floor": target.clock_floor,
        },
    )


def reference_posterior_rwm(
    target: LassoLogistic,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    n_iter: int = 150000,
    n_warmup: int = 20000,
    n_chains: int = 8,
    thin: int = 10,
    seed: int = 0,
) -> dict:
    """Gold standard for the non-differentiable target: random-walk Metropolis.

    No gradients, no smoothing, no discretisation bias -- the accept/reject step
    makes ``exp(-U)`` exactly invariant however rough ``U`` is, which is what the
    anchored chain has to be measured against.  The proposal covariance is
    adapted during warm-up in the manner of Haario, Saksman and Tamminen, then
    frozen.
    """
    rng = np.random.default_rng(seed)
    d = target.d
    W = np.zeros((d, n_chains))
    U = target.potential(W)
    chol = np.eye(d) * 0.05
    log_scale = 0.0
    samples = []
    accepted = 0.0

    total = n_warmup + n_iter
    P_sum = np.zeros(len(y_eval))
    halves = [np.zeros(len(y_eval)), np.zeros(len(y_eval))]
    half_counts = [0, 0]
    w_sum = np.zeros(d)
    ww_sum = np.zeros((d, d))
    n_kept = 0

    for t in range(1, total + 1):
        step = np.exp(log_scale) * (chol @ rng.standard_normal((d, n_chains)))
        Wp = W + step
        Up = target.potential(Wp)
        take = np.log(rng.random(n_chains)) < -(Up - U)
        W = np.where(take, Wp, W)
        U = np.where(take, Up, U)
        rate = float(take.mean())

        if t <= n_warmup:
            log_scale += (rate - 0.234) / (10.0 + t) ** 0.6 * 5.0
            samples.append(W.copy())
            if t % 2000 == 0:  # re-estimate the proposal shape
                S = np.concatenate(samples[-min(len(samples), 4000) :], axis=1)
                cov = np.cov(S) + 1e-10 * np.eye(d)
                chol = np.linalg.cholesky(cov)
                log_scale = np.log(2.38 / np.sqrt(d))
        else:
            accepted += rate
            if (t - n_warmup) % thin == 0:
                P = _sigmoid(X_eval @ W)
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

    def accuracy(p):
        return float(
            ((p > 0.5) * y_eval + (p < 0.5) * (1 - y_eval) + (p == 0.5) * 0.5).mean()
        )

    def loss(p, eps=1e-12):
        q = np.clip(p, eps, 1 - eps)
        return float(-(y_eval * np.log(q) + (1 - y_eval) * np.log1p(-q)).mean())

    return {
        "accuracy": accuracy(p_mean),
        "loss": loss(p_mean),
        "accuracy_halves": [accuracy(halves[i] / half_counts[i]) for i in (0, 1)],
        "loss_halves": [loss(halves[i] / half_counts[i]) for i in (0, 1)],
        "predictive": p_mean,
        "posterior_mean": w_mean,
        "posterior_cov": cov,
        "acceptance": accepted / n_iter,
        "n_samples": n_kept,
        "step_scale": float(np.exp(log_scale)),
    }
