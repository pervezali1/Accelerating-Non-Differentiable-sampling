"""Gold-standard posterior for each dataset, used only as a reference line.

The samplers under test are derivative free.  The reference here is not: it uses
the analytic gradient and Hessian of the logistic posterior to find the MAP, to
build a Laplace preconditioner, and to run a long preconditioned MALA chain.
That is deliberate -- the reference has to be more trustworthy than the methods
it is judging, and for these smooth benchmark posteriors the exact gradient is
the cheapest way to get there.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky

from .target import LogisticPosterior, _sigmoid


def map_estimate(target: LogisticPosterior, n_iter: int = 50, tol: float = 1e-10) -> np.ndarray:
    """Newton (IRLS) maximisation of the log posterior."""
    w = np.zeros(target.d)
    P = np.diag(target.prior_precision)
    for _ in range(n_iter):
        z = target.X @ w
        p = _sigmoid(z)
        grad = target.X.T @ (p - target.y) + target.prior_precision * w
        weights = np.clip(p * (1.0 - p), 1e-10, None)
        H = target.X.T @ (target.X * weights[:, None]) + P
        step = cho_solve(cho_factor(H, lower=True), grad)
        w = w - step
        if np.max(np.abs(step)) < tol:
            break
    return w


def laplace_covariance(target: LogisticPosterior, w: np.ndarray) -> np.ndarray:
    p = _sigmoid(target.X @ w)
    weights = np.clip(p * (1.0 - p), 1e-10, None)
    H = target.X.T @ (target.X * weights[:, None]) + np.diag(target.prior_precision)
    C = cho_solve(cho_factor(H, lower=True), np.eye(target.d))
    return 0.5 * (C + C.T)


def reference_posterior(
    target: LogisticPosterior,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    n_iter: int = 20000,
    n_chains: int = 8,
    n_warmup: int = 2000,
    thin: int = 5,
    seed: int = 0,
) -> dict:
    """Long preconditioned MALA run; returns the reference predictive summary."""
    rng = np.random.default_rng(seed)
    w_map = map_estimate(target)
    C = laplace_covariance(target, w_map)
    L = cholesky(C, lower=True)
    C_inv = cho_solve(cho_factor(C, lower=True), np.eye(target.d))

    W = w_map[:, None] + L @ rng.standard_normal((target.d, n_chains))
    Z = target.linear(W)
    U = target.potential(W, Z=Z)
    G = target.grad(W, Z=Z)
    log_h = np.log(0.3)

    def propose(W, G, h, rng):
        mu = W - h * (C @ G)
        Wp = mu + np.sqrt(2.0 * h) * (L @ rng.standard_normal(W.shape))
        return mu, Wp

    def log_q(diff, h):
        return -(diff * (C_inv @ diff)).sum(axis=0) / (4.0 * h)

    P_sum = np.zeros(len(y_eval))
    w_sum = np.zeros(target.d)
    ww_sum = np.zeros((target.d, target.d))
    n_kept = 0
    U_trace = []
    halves = [np.zeros(len(y_eval)), np.zeros(len(y_eval))]
    half_counts = [0, 0]
    accept = 0.0

    total = n_warmup + n_iter
    for t in range(1, total + 1):
        h = float(np.exp(log_h))
        mu, Wp = propose(W, G, h, rng)
        Zp = target.linear(Wp)
        Up = target.potential(Wp, Z=Zp)
        Gp = target.grad(Wp, Z=Zp)
        mu_back = Wp - h * (C @ Gp)
        log_ratio = -(Up - U) + log_q(W - mu_back, h) - log_q(Wp - mu, h)
        rate = float(np.exp(np.minimum(log_ratio, 0.0)).mean())
        take = np.log(rng.random(n_chains)) < log_ratio
        W = np.where(take, Wp, W)
        Z = np.where(take, Zp, Z)
        U = np.where(take, Up, U)
        G = np.where(take, Gp, G)

        if t <= n_warmup:  # dual-averaging-free Robbins-Monro adaptation
            log_h += (rate - 0.574) / (5.0 + t) ** 0.6 * 5.0
        else:
            accept += rate
            if (t - n_warmup) % thin == 0:
                P = _sigmoid(X_eval @ W)
                P_sum += P.sum(axis=1)
                which = 0 if (t - n_warmup) <= n_iter // 2 else 1
                halves[which] += P.sum(axis=1)
                half_counts[which] += n_chains
                w_sum += W.sum(axis=1)
                ww_sum += W @ W.T
                n_kept += n_chains
                U_trace.append(U.copy())

    p_mean = P_sum / n_kept
    w_mean = w_sum / n_kept
    cov = ww_sum / n_kept - np.outer(w_mean, w_mean)

    def score(p):
        return float(
            ((p > 0.5) * y_eval + (p < 0.5) * (1 - y_eval) + (p == 0.5) * 0.5).mean()
        )

    acc_halves = [score(halves[i] / half_counts[i]) for i in range(2)]
    return {
        "accuracy": score(p_mean),
        "accuracy_halves": acc_halves,
        "predictive": p_mean,
        "posterior_mean": w_mean,
        "posterior_cov": cov,
        "map": w_map,
        "laplace_cov": C,
        "step_size": float(np.exp(log_h)),
        "acceptance": accept / n_iter,
        "n_samples": n_kept,
        "potential_mean": float(np.mean(U_trace)),
    }
