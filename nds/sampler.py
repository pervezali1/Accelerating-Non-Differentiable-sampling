"""Derivative-free irreversible sampler with an exact Metropolis correction.

One step, from walker state ``w`` with a fixed symmetric positive-definite
preconditioner ``D`` and step size ``h``:

    mu(w) = w + h [-(D + J(w)) ghat(w) + Gamma(w)],
    w'    = mu(w) + sqrt(2 h) L xi,        D = L L^T,  xi ~ Normal(0, I),

accepted with probability ``min(1, exp(-U(w') + U(w)) q(w|w') / q(w'|w))`` where
``q(.|w) = Normal(mu(w), 2 h D)``.

``ghat`` is the central-difference surrogate
:meth:`nds.target.LogisticPosterior.fd_grad`, so no derivative of ``U`` is ever
taken, and ``Gamma`` is the divergence correction of the
:class:`nds.skew.SkewField` in use.  Since ``mu`` is a deterministic function of
``w``, the accept/reject step makes ``exp(-U)`` the exact invariant law for
*every* choice of ``J``, with or without the divergence correction: dropping the
correction costs efficiency here, not correctness.  Pass ``metropolis=False`` to
sample the uncorrected diffusion, where dropping it does bias the invariant law.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky

from .skew import SkewField, ZeroSkew
from .target import LogisticPosterior, _sigmoid


@dataclass
class Geometry:
    """Fixed preconditioner ``D`` in the forms the sampler needs."""

    D: np.ndarray
    L: np.ndarray  # lower Cholesky factor, D = L L^T
    D_inv: np.ndarray

    @classmethod
    def from_matrix(cls, D: np.ndarray) -> "Geometry":
        D = np.ascontiguousarray(D, dtype=np.float64)
        L = cholesky(D, lower=True)
        D_inv = cho_solve(cho_factor(D, lower=True), np.eye(len(D)))
        return cls(D=D, L=L, D_inv=0.5 * (D_inv + D_inv.T))

    @classmethod
    def identity(cls, d: int) -> "Geometry":
        eye = np.eye(d)
        return cls(D=eye, L=eye.copy(), D_inv=eye.copy())


@dataclass
class ChainResult:
    """Per-iteration diagnostics of an ensemble run."""

    accuracy: np.ndarray  # (n_iter + 1, m) running posterior-predictive accuracy
    potential: np.ndarray  # (n_iter + 1, m) U(w_t)
    acceptance: float  # mean acceptance probability over walkers and iterations
    step_size: float
    posterior_mean: np.ndarray  # (d, m) running mean of w over the run
    final_state: np.ndarray  # (d, m)
    mean_error: np.ndarray | None = None  # (n_iter + 1, m) whitened error of the running mean
    meta: dict = field(default_factory=dict)


def _score(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Accuracy of a probability matrix (n, m); ties count as one half."""
    yc = y[:, None]
    correct = (P > 0.5) * yc + (P < 0.5) * (1.0 - yc) + (P == 0.5) * 0.5
    return correct.mean(axis=0)


class _Stepper:
    """One Metropolis-corrected irreversible step, vectorised over walkers."""

    def __init__(
        self,
        target: LogisticPosterior,
        skew: SkewField,
        geometry: Geometry,
        fd_eps: float,
        metropolis: bool,
    ) -> None:
        self.target = target
        self.skew = skew
        self.geo = geometry
        self.fd_eps = fd_eps
        self.metropolis = metropolis

    def drift(self, W: np.ndarray, G: np.ndarray, h: float) -> np.ndarray:
        pull = self.geo.D @ G + self.skew.apply(W, G)
        return W + h * (self.skew.divergence(W) - pull)

    def state(self, W: np.ndarray) -> tuple:
        Z = self.target.linear(W)
        return Z, self.target.potential(W, Z=Z), self.target.fd_grad(W, eps=self.fd_eps, Z=Z)

    def step(self, W, Z, U, G, h, rng):
        """Return the updated state plus the mean acceptance probability."""
        d, m = W.shape
        mu = self.drift(W, G, h)
        Wp = mu + np.sqrt(2.0 * h) * (self.geo.L @ rng.standard_normal((d, m)))
        Zp, Up, Gp = self.state(Wp)

        if self.metropolis:
            mu_back = self.drift(Wp, Gp, h)
            fwd = Wp - mu
            bwd = W - mu_back
            scale = 1.0 / (4.0 * h)
            log_ratio = (
                -(Up - U)
                - scale * (bwd * (self.geo.D_inv @ bwd)).sum(axis=0)
                + scale * (fwd * (self.geo.D_inv @ fwd)).sum(axis=0)
            )
            rate = float(np.exp(np.minimum(log_ratio, 0.0)).mean())
            take = np.log(rng.random(m)) < log_ratio
        else:
            rate, take = 1.0, np.ones(m, dtype=bool)

        W = np.where(take, Wp, W)
        Z = np.where(take, Zp, Z)
        U = np.where(take, Up, U)
        G = np.where(take, Gp, G)
        return W, Z, U, G, rate


def run_chain(
    target: LogisticPosterior,
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
    metropolis: bool = True,
    ref_mean: np.ndarray | None = None,
    ref_metric: np.ndarray | None = None,
) -> ChainResult:
    """Run ``n_walkers`` independent walkers for ``n_iter`` iterations.

    The accuracy trace is that of the *running* posterior predictive mean,
    ``mean_{s<=t} sigmoid(X_eval w_s)``, thresholded at one half with ties
    counting as one half -- so the ``w = 0`` start scores exactly 0.5.

    Given ``ref_mean`` and ``ref_metric`` (typically the inverse reference
    posterior covariance), the run also records the error of the running mean,
    ``sqrt((wbar_t - w_ref)^T M (wbar_t - w_ref))``, a far more sensitive probe
    of convergence than accuracy.
    """
    skew = ZeroSkew() if skew is None else skew
    geo = Geometry.identity(target.d) if geometry is None else geometry
    stepper = _Stepper(target, skew, geo, fd_eps, metropolis)
    rng = np.random.default_rng(seed)
    h = float(step_size)

    W = np.zeros((target.d, n_walkers)) if W0 is None else np.array(W0, float)
    Z, U, G = stepper.state(W)

    acc_trace = np.empty((n_iter + 1, n_walkers))
    pot_trace = np.empty((n_iter + 1, n_walkers))
    W_sum = W.copy()
    rate_sum = 0.0
    track_error = ref_mean is not None and ref_metric is not None
    err_trace = np.empty((n_iter + 1, n_walkers)) if track_error else None

    def _error(mean_W: np.ndarray) -> np.ndarray:
        diff = mean_W - np.asarray(ref_mean).reshape(-1, 1)
        return np.sqrt(np.maximum((diff * (ref_metric @ diff)).sum(axis=0), 0.0))

    if track_error:
        err_trace[0] = _error(W)

    if X_eval is not None:
        P_sum = _sigmoid(X_eval @ W)
        acc_trace[0] = _score(P_sum, y_eval)
    else:
        P_sum = None
        acc_trace[0] = np.nan
    pot_trace[0] = U

    for t in range(1, n_iter + 1):
        W, Z, U, G, rate = stepper.step(W, Z, U, G, h, rng)
        rate_sum += rate
        W_sum += W
        pot_trace[t] = U
        if track_error:
            err_trace[t] = _error(W_sum / (t + 1))
        if P_sum is not None:
            P_sum += _sigmoid(X_eval @ W)
            acc_trace[t] = _score(P_sum / (t + 1), y_eval)

    return ChainResult(
        accuracy=acc_trace,
        potential=pot_trace,
        acceptance=rate_sum / n_iter,
        step_size=h,
        posterior_mean=W_sum / (n_iter + 1),
        final_state=W,
        mean_error=err_trace,
        meta={"label": skew.label, "n_walkers": n_walkers, "fd_eps": fd_eps},
    )


def mean_acceptance(
    target: LogisticPosterior,
    skew: SkewField | None,
    step_size: float,
    n_iter: int,
    n_walkers: int = 8,
    seed: int = 0,
    geometry: Geometry | None = None,
    fd_eps: float = 1e-2,
) -> float:
    """Mean acceptance probability of a run started from ``w = 0``."""
    res = run_chain(
        target,
        skew,
        step_size,
        n_iter,
        n_walkers=n_walkers,
        seed=seed,
        geometry=geometry,
        fd_eps=fd_eps,
    )
    return res.acceptance


def acceptance_diagnostics(
    target: LogisticPosterior,
    skew: SkewField | None,
    step_size: float,
    W: np.ndarray,
    n_draws: int = 512,
    seed: int = 0,
    geometry: Geometry | None = None,
    fd_eps: float = 1e-2,
) -> dict:
    """Acceptance statistics for proposals made from the *fixed* point ``W``.

    Useful for seeing where the step-size ceiling comes from.  In the reversible
    case the ``O(sqrt(h))`` term of the log acceptance ratio cancels and the
    spread of ``log alpha`` decays like ``h^{3/2}``; with ``J != 0`` a term
    ``sqrt(2h) (J ghat) . xi`` survives, so the spread decays only like
    ``h^{1/2}`` and the step size has to shrink quadratically in ``|J ghat|``
    to keep the same acceptance rate.
    """
    skew = ZeroSkew() if skew is None else skew
    geo = Geometry.identity(target.d) if geometry is None else geometry
    stepper = _Stepper(target, skew, geo, fd_eps, True)
    rng = np.random.default_rng(seed)
    h = float(step_size)

    W0 = np.repeat(np.asarray(W, float).reshape(-1, 1), n_draws, axis=1)
    Z, U, G = stepper.state(W0)
    mu = stepper.drift(W0, G, h)
    Wp = mu + np.sqrt(2.0 * h) * (geo.L @ rng.standard_normal(W0.shape))
    Zp, Up, Gp = stepper.state(Wp)
    mu_back = stepper.drift(Wp, Gp, h)
    scale = 1.0 / (4.0 * h)
    fwd, bwd = Wp - mu, W0 - mu_back
    log_ratio = (
        -(Up - U)
        - scale * (bwd * (geo.D_inv @ bwd)).sum(axis=0)
        + scale * (fwd * (geo.D_inv @ fwd)).sum(axis=0)
    )
    return {
        "step_size": h,
        "acceptance": float(np.exp(np.minimum(log_ratio, 0.0)).mean()),
        "log_ratio_std": float(log_ratio.std()),
        "log_ratio_mean": float(log_ratio.mean()),
    }


def stable_step_size(
    target: LogisticPosterior,
    skew: SkewField | None,
    h0: float,
    n_iter: int = 150,
    n_walkers: int = 8,
    seed: int = 0,
    geometry: Geometry | None = None,
    fd_eps: float = 1e-2,
    shrink: float = 0.5,
    max_tries: int = 8,
    tolerance: float = 0.5,
) -> dict:
    """Largest step size at or below ``h0`` at which the *unadjusted* chain is stable.

    The design rule in :mod:`nds.design` picks ``h0`` from the warm-up estimate
    of the Hessian.  That estimate can be optimistic -- shrinkage inflates the
    smallest posterior variances, which understates the largest curvature -- and
    an explicit unadjusted step is unforgiving about it, so this guard runs a
    short pilot and halves ``h`` until the potential stays finite and does not
    climb back above its own minimum by more than ``tolerance``.  It fires rarely
    and the result records whether it did.
    """
    h = float(h0)
    for attempt in range(max_tries):
        res = run_chain(
            target, skew, h, n_iter, n_walkers=n_walkers, seed=seed,
            geometry=geometry, fd_eps=fd_eps, metropolis=False,
        )
        trace = res.potential.mean(axis=1)
        if np.isfinite(trace).all() and trace[-1] <= trace.min() * (1.0 + tolerance):
            return {"step_size": h, "reductions": attempt, "stable": True}
        h *= shrink
    return {"step_size": h, "reductions": max_tries, "stable": False}


def calibrate_step_size(
    target: LogisticPosterior,
    skew: SkewField | None,
    n_iter: int = 200,
    n_walkers: int = 6,
    seed: int = 0,
    geometry: Geometry | None = None,
    fd_eps: float = 1e-2,
    target_acceptance: float = 0.5,
    h_lo: float = 1e-7,
    h_hi: float = 10.0,
    n_bisect: int = 7,
) -> dict:
    """Bisect the step size so that a run from ``w = 0`` accepts ~half the time.

    Tuning on the whole transient, rather than at stationarity, is deliberate:
    the experiment measures convergence from ``w = 0``, and a step size that is
    only good near the mode leaves a Metropolis chain frozen at the start, where
    the gradient of ``U`` is largest.  Acceptance is monotone in ``h``, so plain
    bisection suffices, and every ``J`` variant is tuned the same way.
    """
    kw = dict(
        n_iter=n_iter, n_walkers=n_walkers, seed=seed, geometry=geometry, fd_eps=fd_eps
    )
    lo, hi = h_lo, h_hi
    trace = []
    for _ in range(n_bisect):
        mid = float(np.sqrt(lo * hi))
        rate = mean_acceptance(target, skew, mid, **kw)
        trace.append((mid, rate))
        if rate > target_acceptance:
            lo = mid
        else:
            hi = mid
    best = min(trace, key=lambda hr: abs(hr[1] - target_acceptance))
    return {"step_size": best[0], "pilot_acceptance": best[1], "trace": trace}


def warm_up(
    target: LogisticPosterior,
    n_iter: int = 500,
    n_walkers: int = 8,
    seed: int = 0,
    fd_eps: float = 1e-2,
    shrinkage: float = 0.1,
    n_rounds: int = 2,
) -> dict:
    """Estimate a preconditioner and a radial scale from ``J = 0`` pilot runs.

    Round one runs with ``D = I``, later rounds with the covariance estimated so
    far; the returned ``geometry`` is the shrunk sample covariance of the second
    half of the last pilot, and ``rho`` is half the typical whitened distance of
    the pilot ensemble from the ``w = 0`` start.  Both are ordinary adaptive-MCMC
    warm-up quantities: no gradients, no held-out data.
    """
    geo = Geometry.identity(target.d)
    info = {}
    for rnd in range(n_rounds):
        cal = calibrate_step_size(
            target, ZeroSkew(), n_iter=n_iter // 2, n_walkers=n_walkers,
            seed=seed + rnd, geometry=geo, fd_eps=fd_eps,
        )
        stepper = _Stepper(target, ZeroSkew(), geo, fd_eps, True)
        rng = np.random.default_rng(seed + 100 + rnd)
        W = np.zeros((target.d, n_walkers))
        Z, U, G = stepper.state(W)
        samples = []
        for t in range(1, n_iter + 1):
            W, Z, U, G, _ = stepper.step(W, Z, U, G, cal["step_size"], rng)
            if t > n_iter // 2:
                samples.append(W.copy())
        S = np.concatenate(samples, axis=1)  # (d, kept * walkers)
        cov = np.cov(S)
        cov = np.atleast_2d(cov)
        diag = np.diag(np.clip(np.diag(cov), 1e-12, None))
        cov = (1.0 - shrinkage) * cov + shrinkage * diag
        cov += 1e-10 * np.eye(target.d)
        geo = Geometry.from_matrix(cov)
        info = {
            "geometry": geo,
            "step_size": cal["step_size"],
            "pilot_acceptance": cal["pilot_acceptance"],
            "mean_state": S.mean(axis=1),
        }
    w_bar = info["mean_state"]
    radius = float(np.sqrt(max(w_bar @ (geo.D_inv @ w_bar), 1e-12)))
    info["rho"] = max(0.5 * radius, 1e-3)
    info["whitened_radius"] = radius
    return info
