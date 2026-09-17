#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
===============================================================================
Projected Non-Reversible Anchored Langevin (PNRAL) on the 3-D ball
===============================================================================

Target
------
    pi_C(x)  \propto  exp(-U(x)) * 1{ ||x||_2 <= R },     C = { x in R^3 : ||x|| <= R }

Anchored Langevin
-----------------
The potential U may be non-differentiable (here U(x) = |x1|+|x2|+|x3|).  We
therefore *anchor* it on a smooth surrogate U0 and carry the mismatch in a
scalar, state-dependent *anchor coefficient*

    a(x) = exp( U(x) - U0(x) )                     (never exp(U)/exp(U0)!)

The (unconstrained, reversible) anchored Langevin diffusion is

    dX_t = -a(X_t) grad U0(X_t) dt + sqrt(2 a(X_t)) dW_t .                  (1)

Stationarity check (why no div(a) correction appears): the Fokker-Planck
stationary condition for (1) is  div( a grad U0 p + grad(a p) ) = 0.  With
p = exp(-U)/Z we get  a p \propto exp(U-U0) exp(-U) = exp(-U0), hence

    grad(a p) = -grad U0 * exp(-U0)     and     a grad U0 p = +grad U0 * exp(-U0),

so the probability flux vanishes identically:  exp(-U) is invariant *exactly*,
with no gradient of U and no div(a) drift correction.  This is the whole point
of the anchored construction -- all non-smoothness sits inside the scalar a(x).

Non-reversible perturbation
---------------------------
Add a state-dependent skew-symmetric matrix J_s(x) to the drift:

    dX_t = -a(X_t) (I + alpha J_s(X_t)) grad U0(X_t) dt + sqrt(2 a(X_t)) dW_t .

With  J_s(x) = s * [x]_cross  (the cross-product matrix),

    J_s(x) = s * [[  0, -x3,  x2],
                  [ x3,   0, -x1],
                  [-x2,  x1,   0]],       J_s(x) g = s * (x cross g),

three identities make this free of correction terms:

  (i)   J_s(x)^T = -J_s(x)                  (skew-symmetry)
  (ii)  div J_s(x) = 0    row-wise:  d/dx2(-x3) + d/dx3(x2) = 0, etc.
  (iii) x^T J_s(x) = 0    since  J_s(x) x = s (x cross x) = 0.

(i)+(ii) imply the extra flux  -alpha J_s grad U0 * p = alpha J_s grad(exp(-U0))
is divergence-free:  div(alpha J grad q) = alpha (div J).grad q + alpha tr(J Hess q)
= 0 + 0, because the trace of (skew * symmetric) vanishes.  So the invariant law
is unchanged, but detailed balance is broken -> faster mixing, and the standard
div(J) drift correction is identically zero.
(iii) says the non-reversible drift is *tangent to every sphere* ||x|| = const,
in particular tangent to the boundary of C: it rotates probability mass around
the constraint set instead of pushing into it.

Projected update (the algorithm implemented here)
-------------------------------------------------
    g_k      = grad_U0(x_k)
    log a_k  = U(x_k) - U0(x_k)                    (log-domain, stable)
    a_k      = exp(log a_k)
    b_k      = -a_k (I + alpha J_s(x_k)) g_k       (reversible + non-reversible drift)
    y_{k+1}  = x_k + h b_k + sqrt(2 h a_k) * N(0, I_3)
    x_{k+1}  = Proj_C(y_{k+1}),      Proj_C(y) = y if ||y||<=R else R y/||y||

Test problem (constrained L1 / Laplace target)
----------------------------------------------
    U(x)      = |x1| + |x2| + |x3|                          (non-differentiable)
    U0(x)     = sum_j sqrt(x_j^2 + delta^2)                 (smooth anchor)
    grad U0_j = x_j / sqrt(x_j^2 + delta^2)                 (bounded by 1)
    log a(x)  = sum_j ( |x_j| - sqrt(x_j^2 + delta^2) )  in [-3 delta, 0]
  =>  exp(-3 delta) <= a(x) <= 1   (checked numerically below).

Usage
-----
    python pnr_anchored_langevin_ball3d.py              # full experiment
    python pnr_anchored_langevin_ball3d.py --quick      # fast smoke run
    python pnr_anchored_langevin_ball3d.py --outdir DIR

Outputs: PNG figures + CSV diagnostic tables in --outdir, plus printed tables.
===============================================================================
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

import arviz as az


# =============================================================================
# 1. CONFIGURATION SECTION  (all tunable parameters live here)
# =============================================================================
@dataclass
class Config:
    # --- geometry of the constraint set C = {||x|| <= R} ---
    R: float = 3.0                      # ball radius

    # --- anchor smoothing ---
    delta: float = 0.05                 # smoothing of |x| -> sqrt(x^2+delta^2)

    # --- non-reversible perturbation ---
    s: float = 1.0                      # scale inside J_s; only the product
                                        # alpha*s matters, so we FIX s = 1 and
                                        # tune alpha (requirement 8).
    alphas: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)

    # --- discretisation / MCMC control ---
    h: float = 0.02                     # base step size
    n_iter: int = 40_000                # iterations per chain
    burn_in: int = 8_000                # discarded warm-up iterations
    thin: int = 2                       # thinning factor for stored draws
    n_chains: int = 4                   # independent chains (requirement 6)

    # --- randomness (identical across alpha values, requirement 9) ---
    base_seed: int = 20_240_917         # master seed: chain inits + noise
    init_radius_frac: float = 0.6       # chains start at 0.6 R * random direction

    # --- step-size sensitivity study (requirement 13) ---
    step_size_factors: Tuple[float, ...] = (1.0, 0.5, 0.25)   # h, h/2, h/4
    step_size_alphas: Tuple[float, ...] = (0.0, 2.0)
    step_size_match_simulated_time: bool = True   # scale n_iter by 1/factor so
                                # that n_iter * h (the simulated time horizon)
                                # is constant: otherwise a smaller h simply
                                # explores less and its extra *accuracy* is
                                # masked by Monte-Carlo error.

    # --- validation thresholds (requirement 14) ---
    max_projection_fraction: float = 0.25   # warn above this projection rate
    ball_tol: float = 1e-9                  # tolerance on ||x|| <= R

    # --- reference sample (exact truncated Laplace, for bias assessment) ---
    n_reference: int = 200_000

    # --- I/O ---
    outdir: str = "pnral_outputs"
    max_acf_lag: int = 200
    dpi: int = 130

    def seed_for_chain(self, chain: int) -> int:
        """Noise seed of a chain -- deliberately independent of alpha and h so
        that different alpha/h configurations use *common random numbers*."""
        return self.base_seed + 1_000 * (chain + 1)

    def initial_states(self) -> np.ndarray:
        """Identical initial states for every configuration (requirement 9)."""
        rng = np.random.default_rng(self.base_seed)
        d = rng.normal(size=(self.n_chains, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)      # uniform directions
        return self.init_radius_frac * self.R * d          # strictly inside C


CFG = Config()

# Collected human-readable warnings (printed as a block at the end).
WARNINGS_LOG: List[str] = []


def warn(msg: str) -> None:
    """Record + emit a warning (requirement 14)."""
    WARNINGS_LOG.append(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


# =============================================================================
# 2. MODEL FUNCTIONS (reusable, requirement 2)
# =============================================================================
def target_potential(x: np.ndarray) -> np.ndarray:
    """U(x) = |x1| + |x2| + |x3|  -- the *non-differentiable* target potential.

    pi(x) \propto exp(-U(x)) on C: an isotropic Laplace law truncated to the ball.
    """
    x = np.asarray(x, dtype=float)
    return np.abs(x).sum(axis=-1)


def anchor_potential(x: np.ndarray, delta: float) -> np.ndarray:
    """U0(x) = sum_j sqrt(x_j^2 + delta^2) -- the smooth anchor (pseudo-Huber).

    U0 is C^infinity, has bounded gradient, and U0 -> U as delta -> 0, so the
    anchor coefficient a = exp(U-U0) stays in a narrow, well-conditioned range.
    """
    x = np.asarray(x, dtype=float)
    return np.sqrt(x * x + delta * delta).sum(axis=-1)


def grad_anchor(x: np.ndarray, delta: float) -> np.ndarray:
    """grad U0(x)_j = x_j / sqrt(x_j^2 + delta^2) -- smooth surrogate of sign(x_j).

    This is the ONLY gradient the sampler ever evaluates: the kink of |x| is
    never differentiated.  |grad U0_j| < 1, so the drift is globally bounded.
    """
    x = np.asarray(x, dtype=float)
    return x / np.sqrt(x * x + delta * delta)


def log_anchor_coefficient(x: np.ndarray, delta: float) -> np.ndarray:
    """log a(x) = U(x) - U0(x) = sum_j (|x_j| - sqrt(x_j^2 + delta^2)).

    Computed directly in the log domain (requirement 3): we never form
    exp(U)/exp(U0), which would overflow/underflow for large ||x||.
    Each term lies in [-delta, 0]  =>  log a in [-3 delta, 0].
    """
    x = np.asarray(x, dtype=float)
    return (np.abs(x) - np.sqrt(x * x + delta * delta)).sum(axis=-1)


def anchor_coefficient(x: np.ndarray, delta: float) -> np.ndarray:
    """a(x) = exp(U(x) - U0(x)) in (0, 1]: the state-dependent mobility."""
    return np.exp(log_anchor_coefficient(x, delta))


def J_s(x: np.ndarray, s: float) -> np.ndarray:
    """State-dependent skew-symmetric matrix J_s(x) = s * [x]_cross.

        J_s(x) = s [[  0, -x3,  x2],
                    [ x3,   0, -x1],
                    [-x2,  x1,   0]]

    Properties used by the algorithm (verified in verify_J_properties):
      J^T = -J          -> the perturbation adds no dissipation;
      div J = 0         -> no  div(J) drift correction is required;
      x^T J = 0 (J x=0) -> the induced drift is tangent to spheres, hence to
                           the boundary of the ball, so the non-reversible flow
                           never pushes the state through the constraint.
    Note J_s(x) g = s * (x cross g): a rotation of the gradient about the radius.
    """
    x = np.asarray(x, dtype=float)
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    zero = np.zeros_like(x1)
    J = np.stack(
        [
            np.stack([zero, -x3, x2], axis=-1),
            np.stack([x3, zero, -x1], axis=-1),
            np.stack([-x2, x1, zero], axis=-1),
        ],
        axis=-2,
    )
    return s * J


def project_to_ball(x: np.ndarray, R: float, return_flag: bool = False):
    """Euclidean projection onto C = {||x|| <= R}: the metric projection

        Proj_C(y) = y                if ||y|| <= R,
                    R * y / ||y||    otherwise.

    Projecting (rather than rejecting) keeps every iterate feasible and is the
    standard discretisation of reflected/constrained Langevin dynamics.
    """
    x = np.asarray(x, dtype=float)
    nrm = np.linalg.norm(x, axis=-1, keepdims=True)
    outside = nrm > R
    scale = np.where(outside, R / np.where(nrm > 0, nrm, 1.0), 1.0)
    proj = x * scale
    if return_flag:
        return proj, np.squeeze(outside, axis=-1)
    return proj


# =============================================================================
# 3. NUMERICAL VERIFICATION OF THE STRUCTURAL IDENTITIES (requirement 4)
# =============================================================================
def verify_J_properties(n_tests: int = 2_000, s: float = 1.0, seed: int = 0,
                        R: float = 3.0) -> pd.DataFrame:
    """Check J^T=-J, x^T J = 0, div J = 0 (finite differences) at random x."""
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=R, size=(n_tests, 3))

    J = J_s(X, s)                                        # (n,3,3)
    skew_err = np.abs(J + np.swapaxes(J, -1, -2)).max()  # ||J + J^T||_inf
    tangent_err = np.abs(np.einsum("ni,nij->nj", X, J)).max()   # ||x^T J||_inf

    # div J: (div J)_i = sum_j d J_ij / d x_j, by central differences.
    eps = 1e-5
    div = np.zeros((n_tests, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = eps
        dJ = (J_s(X + e, s) - J_s(X - e, s)) / (2 * eps)
        div += dJ[..., :, j]
    div_err = np.abs(div).max()

    # Also confirm J_s(x) g = s (x cross g) and that it is orthogonal to x.
    G = rng.normal(size=(n_tests, 3))
    cross_err = np.abs(np.einsum("nij,nj->ni", J, G) - s * np.cross(X, G)).max()
    orth_err = np.abs(np.einsum("ni,ni->n", X, np.einsum("nij,nj->ni", J, G))).max()

    out = pd.DataFrame(
        [
            ("max |J + J^T|", skew_err, 1e-12),
            ("max |x^T J|", tangent_err, 1e-10),
            ("max |div J| (central diff)", div_err, 1e-6),
            ("max |J g - s (x x g)|", cross_err, 1e-10),
            ("max |x . (J g)| (tangency)", orth_err, 1e-9),
        ],
        columns=["check", "max_error", "tolerance"],
    )
    out["passed"] = out["max_error"] <= out["tolerance"]
    for _, r in out.iterrows():
        if not r["passed"]:
            warn(f"Structural check FAILED: {r['check']} = {r['max_error']:.3e}")
    return out


def verify_anchor_bounds(n_tests: int = 5_000, delta: float = 0.05,
                         R: float = 3.0, seed: int = 1) -> pd.DataFrame:
    """Check the analytic bound exp(-3 delta) <= a(x) <= 1 on C."""
    rng = np.random.default_rng(seed)
    # uniform-ish points of the ball plus deliberate corner/axis cases
    d = rng.normal(size=(n_tests, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    X = d * (R * rng.random((n_tests, 1)) ** (1 / 3))
    X = np.vstack([X, np.zeros((1, 3)), np.array([[R, 0, 0]]),
                   np.full((1, 3), R / np.sqrt(3))])
    a = anchor_coefficient(X, delta)
    lo, hi = np.exp(-3 * delta), 1.0
    out = pd.DataFrame(
        [
            ("min a(x)", float(a.min()), f">= exp(-3 delta) = {lo:.8f}",
             bool(a.min() >= lo - 1e-12)),
            ("max a(x)", float(a.max()), f"<= 1 (= {hi:.1f})",
             bool(a.max() <= hi + 1e-12)),
        ],
        columns=["check", "value", "bound", "passed"],
    )
    for _, r in out.iterrows():
        if not r["passed"]:
            warn(f"Anchor-coefficient bound FAILED: {r['check']} = {r['value']:.6e}")
    return out


# =============================================================================
# 4. THE SAMPLER (requirement 2, 5, 16)
# =============================================================================
def run_sampler(
    n_iter: int,
    x0: Sequence[float],
    h: float,
    alpha: float,
    s: float,
    R: float,
    delta: float,
    seed: int,
    chain_id: int = 0,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, float]]:
    """Run ONE chain of the Projected Non-Reversible Anchored Langevin sampler.

    Returns
    -------
    samples : (n_iter, 3) array of states x_0, ..., x_{n_iter-1}
    diag    : per-iteration pandas DataFrame (requirement 5 & 16) with columns
              chain, iteration, x1, x2, x3, a, log_a, norm, projected,
              drift_norm, nr_drift_norm, U, U0
    meta    : scalar summary (runtime, projection fraction, extrema, ...)

    Row k of `diag` describes the state x_k *and* the step taken from it, i.e.
    `projected` is True when y_{k+1} fell outside C and had to be projected.
    """
    rng = np.random.default_rng(seed)
    I3 = np.eye(3)

    x = project_to_ball(np.asarray(x0, dtype=float), R)  # make sure x_0 in C
    if not np.isfinite(x).all():
        raise ValueError("Initial state is not finite.")

    samples = np.empty((n_iter, 3))
    a_rec = np.empty(n_iter)
    loga_rec = np.empty(n_iter)
    norm_rec = np.empty(n_iter)
    proj_rec = np.zeros(n_iter, dtype=bool)
    drift_rec = np.empty(n_iter)
    nrdrift_rec = np.empty(n_iter)

    n_nonfinite = 0
    t0 = time.perf_counter()
    for k in range(n_iter):
        # --- smooth anchor gradient: the only derivative ever evaluated -----
        g = grad_anchor(x, delta)

        # --- anchor coefficient in the log domain (never exp(U)/exp(U0)) ----
        log_a = log_anchor_coefficient(x, delta)
        a = np.exp(log_a)

        # --- drift: reversible part  -a g  plus non-reversible  -a alpha J g -
        J = J_s(x, s)                       # skew-symmetric, div-free, J x = 0
        b_rev = -a * g                      # gradient descent on U0, mobility a
        b_nr = -a * alpha * (J @ g)         # = -a*alpha*s*(x cross g): tangential
        b = -a * ((I3 + alpha * J) @ g)     # the update as specified

        # --- Euler-Maruyama proposal with state-dependent diffusion sqrt(2 a) -
        noise = rng.standard_normal(3)
        y = x + h * b + np.sqrt(2.0 * h * a) * noise

        if not np.isfinite(y).all():        # requirement 14: NaN / inf guard
            n_nonfinite += 1
            y = x.copy()                    # freeze rather than propagate NaN

        # --- metric projection back onto the ball ---------------------------
        x_new, projected = project_to_ball(y, R, return_flag=True)

        # --- record the iteration (requirement 5) ---------------------------
        samples[k] = x
        a_rec[k] = a
        loga_rec[k] = log_a
        norm_rec[k] = np.linalg.norm(x)
        proj_rec[k] = bool(projected)
        drift_rec[k] = np.linalg.norm(b)
        nrdrift_rec[k] = np.linalg.norm(b_nr)

        x = x_new
    runtime = time.perf_counter() - t0

    diag = pd.DataFrame(
        {
            "chain": chain_id,
            "iteration": np.arange(n_iter),
            "x1": samples[:, 0],
            "x2": samples[:, 1],
            "x3": samples[:, 2],
            "a": a_rec,
            "log_a": loga_rec,
            "norm": norm_rec,
            "projected": proj_rec,
            "drift_norm": drift_rec,
            "nr_drift_norm": nrdrift_rec,
        }
    )
    diag["U"] = target_potential(samples)
    diag["U0"] = anchor_potential(samples, delta)

    # ---- per-chain validation (requirement 4 & 14) -------------------------
    max_norm = float(norm_rec.max())
    a_min, a_max = float(a_rec.min()), float(a_rec.max())
    lo_bound = np.exp(-3 * delta)
    proj_frac = float(proj_rec.mean())

    if max_norm > R + CFG.ball_tol:
        warn(f"[alpha={alpha}, h={h}, chain {chain_id}] ball constraint violated: "
             f"max ||x|| = {max_norm:.12f} > R = {R}")
    if not np.isfinite(samples).all() or n_nonfinite:
        warn(f"[alpha={alpha}, h={h}, chain {chain_id}] {n_nonfinite} non-finite "
             "proposal(s) encountered and suppressed.")
    if a_min < lo_bound - 1e-12 or a_max > 1.0 + 1e-12:
        warn(f"[alpha={alpha}, h={h}, chain {chain_id}] anchor coefficient out of "
             f"[exp(-3 delta), 1] = [{lo_bound:.6f}, 1]: a in [{a_min:.6f}, {a_max:.6f}]")
    if proj_frac > CFG.max_projection_fraction:
        warn(f"[alpha={alpha}, h={h}, chain {chain_id}] excessive projection rate "
             f"{proj_frac:.1%} > {CFG.max_projection_fraction:.0%}: step size likely too large.")

    meta = dict(
        chain=chain_id, alpha=alpha, s=s, h=h, seed=seed, runtime_s=runtime,
        projection_fraction=proj_frac, max_norm=max_norm, a_min=a_min, a_max=a_max,
        n_nonfinite=n_nonfinite,
        mean_drift_norm=float(drift_rec.mean()),
        mean_nr_drift_norm=float(nrdrift_rec.mean()),
    )
    return samples, diag, meta


def run_chains(cfg: Config, alpha: float, h: float, n_iter: int | None = None,
               burn_in: int | None = None) -> Dict[str, object]:
    """Run `cfg.n_chains` independent chains with COMMON random numbers.

    Identical initial states and identical per-chain seeds are reused for every
    (alpha, h) configuration, so differences between configurations are due to
    the dynamics, not to Monte Carlo noise (requirement 9).
    """
    n_iter = cfg.n_iter if n_iter is None else n_iter
    burn_in = cfg.burn_in if burn_in is None else burn_in
    x0s = cfg.initial_states()
    all_samples, all_diags, metas = [], [], []
    for c in range(cfg.n_chains):
        smp, dg, mt = run_sampler(
            n_iter=n_iter, x0=x0s[c], h=h, alpha=alpha, s=cfg.s, R=cfg.R,
            delta=cfg.delta, seed=cfg.seed_for_chain(c), chain_id=c,
        )
        all_samples.append(smp)
        all_diags.append(dg)
        metas.append(mt)

    raw = np.stack(all_samples, axis=0)                 # (chain, iter, 3)
    diag = pd.concat(all_diags, ignore_index=True)
    diag["alpha"] = alpha
    diag["h"] = h
    meta = pd.DataFrame(metas)
    meta["alpha"] = alpha
    meta["h"] = h

    # post burn-in + thinning: the draws used for all posterior diagnostics
    draws = raw[:, burn_in::cfg.thin, :]
    return dict(raw=raw, draws=draws, diag=diag, meta=meta, alpha=alpha, h=h,
                n_iter=n_iter, burn_in=burn_in,
                runtime_s=float(meta["runtime_s"].sum()))


# =============================================================================
# 5. DIAGNOSTICS (requirement 10)
# =============================================================================
def to_inference_data(draws: np.ndarray) -> az.InferenceData:
    """(chain, draw, 3) -> ArviZ InferenceData with x1,x2,x3 and radius r."""
    return az.convert_to_inference_data(
        {
            "x1": draws[:, :, 0],
            "x2": draws[:, :, 1],
            "x3": draws[:, :, 2],
            "r": np.linalg.norm(draws, axis=-1),
        }
    )


def summarize_run(run: Dict[str, object], cfg: Config,
                  reference: np.ndarray | None = None) -> Dict[str, float]:
    """ESS, ESS/s, R-hat, integrated autocorrelation time, accuracy vs reference."""
    draws = run["draws"]
    idata = to_inference_data(draws)
    ess = az.ess(idata)
    ess_bulk = {v: float(ess[v].values) for v in ["x1", "x2", "x3", "r"]}
    ess_tail = {v: float(az.ess(idata, method="tail")[v].values)
                for v in ["x1", "x2", "x3"]}
    rhat = az.rhat(idata)
    rhat_v = {v: float(rhat[v].values) for v in ["x1", "x2", "x3", "r"]}

    n_draws = draws.shape[0] * draws.shape[1]
    runtime = run["runtime_s"]
    ess_min = min(ess_bulk[v] for v in ["x1", "x2", "x3"])
    ess_mean = float(np.mean([ess_bulk[v] for v in ["x1", "x2", "x3"]]))

    diag: pd.DataFrame = run["diag"]
    row = dict(
        alpha=run["alpha"], h=run["h"], s=cfg.s,
        n_chains=draws.shape[0], n_draws_per_chain=draws.shape[1],
        n_iter_per_chain=int(run["raw"].shape[1]),
        # simulated time horizon per chain: the physically meaningful budget
        # when comparing step sizes (n_iter * h).
        sim_time=float(run["raw"].shape[1] * run["h"]),
        runtime_s=runtime,
        ess_x1=ess_bulk["x1"], ess_x2=ess_bulk["x2"], ess_x3=ess_bulk["x3"],
        ess_r=ess_bulk["r"],
        ess_min=ess_min, ess_mean=ess_mean,
        ess_tail_min=min(ess_tail.values()),
        ess_per_sec_min=ess_min / runtime, ess_per_sec_mean=ess_mean / runtime,
        ess_frac=ess_mean / n_draws,
        iact_mean=n_draws / ess_mean,             # integrated autocorr. time
        rhat_max=max(rhat_v.values()),
        rhat_x1=rhat_v["x1"], rhat_x2=rhat_v["x2"], rhat_x3=rhat_v["x3"],
        projection_fraction=float(diag["projected"].mean()),
        max_norm=float(diag["norm"].max()),
        a_min=float(diag["a"].min()), a_max=float(diag["a"].max()),
        mean_drift_norm=float(diag["drift_norm"].mean()),
        mean_nr_drift_norm=float(diag["nr_drift_norm"].mean()),
        mean_radius=float(np.linalg.norm(draws, axis=-1).mean()),
        mean_abs_x1=float(np.abs(draws[:, :, 0]).mean()),
        sd_x1=float(draws[:, :, 0].std()),
    )
    if reference is not None:
        flat = draws.reshape(-1, 3)
        # KS distance of the x1 marginal against an exact (rejection) sample:
        # a scalar proxy for the discretisation + projection bias.
        row["ks_x1"] = float(sp_stats.ks_2samp(flat[:, 0], reference[:, 0]).statistic)
        row["ks_r"] = float(
            sp_stats.ks_2samp(np.linalg.norm(flat, axis=1),
                              np.linalg.norm(reference, axis=1)).statistic)
        row["bias_mean_abs_x1"] = row["mean_abs_x1"] - float(np.abs(reference[:, 0]).mean())
        row["bias_mean_radius"] = row["mean_radius"] - float(
            np.linalg.norm(reference, axis=1).mean())
    if row["rhat_max"] > 1.01:
        warn(f"[alpha={run['alpha']}, h={run['h']}] R-hat = {row['rhat_max']:.4f} > 1.01: "
             "chains may not have mixed.")
    return row


def reference_sample(cfg: Config, seed: int = 99) -> np.ndarray:
    """Exact i.i.d. draws from pi_C by rejection: Laplace(1)^3 restricted to C.

    Used only as ground truth for accuracy plots/statistics -- it plays no part
    in the sampler itself.
    """
    rng = np.random.default_rng(seed)
    out, need = [], cfg.n_reference
    while need > 0:
        prop = rng.laplace(loc=0.0, scale=1.0, size=(max(need * 2, 1000), 3))
        keep = prop[np.linalg.norm(prop, axis=1) <= cfg.R]
        out.append(keep)
        need -= len(keep)
    return np.vstack(out)[: cfg.n_reference]


def autocorr_curves(draws: np.ndarray, max_lag: int) -> Dict[str, np.ndarray]:
    """Chain-averaged ACF of each coordinate and of the radius."""
    series = {"x1": draws[:, :, 0], "x2": draws[:, :, 1], "x3": draws[:, :, 2],
              "r": np.linalg.norm(draws, axis=-1)}
    out = {}
    for name, arr in series.items():
        acf = np.mean([az.autocorr(arr[c])[: max_lag + 1] for c in range(arr.shape[0])],
                      axis=0)
        out[name] = acf
    return out


# =============================================================================
# 6. PLOTS (requirements 11, 12)
# =============================================================================
def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_traces(run, cfg, outdir: Path) -> None:
    """Trace plots of x1,x2,x3 and the radius, all chains overlaid."""
    draws = run["draws"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    names = ["x1", "x2", "x3", "||x||"]
    series = [draws[:, :, 0], draws[:, :, 1], draws[:, :, 2],
              np.linalg.norm(draws, axis=-1)]
    for ax, nm, arr in zip(axes, names, series):
        for c in range(arr.shape[0]):
            ax.plot(arr[c], lw=0.5, alpha=0.75, label=f"chain {c}")
        ax.set_ylabel(nm)
        if nm == "||x||":
            ax.axhline(cfg.R, color="k", ls="--", lw=1, label="R (boundary)")
    axes[0].legend(ncol=5, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("draw (after burn-in and thinning)")
    fig.suptitle(f"Trace plots — alpha = {run['alpha']}, s = {cfg.s}, h = {run['h']}")
    _save(fig, outdir / f"trace_alpha{run['alpha']:g}_h{run['h']:g}.png", cfg.dpi)


def plot_histograms(runs, cfg, reference, outdir: Path) -> None:
    """Coordinate histograms vs the exact truncated-Laplace marginal."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for j, ax in enumerate(axes):
        for run in runs:
            flat = run["draws"].reshape(-1, 3)
            sns.kdeplot(x=flat[:, j], ax=ax, lw=1.4, label=f"alpha={run['alpha']:g}")
        ax.hist(reference[:, j], bins=120, density=True, alpha=0.20, color="grey",
                label="exact (rejection)")
        ax.set_xlabel(f"x{j+1}")
        ax.set_title(f"marginal of x{j+1}")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Coordinate marginals — h = {runs[0]['h']}, delta = {cfg.delta}, R = {cfg.R}")
    _save(fig, outdir / "histograms_coordinates.png", cfg.dpi)

    # histogram + trace of the radius (requirement 11: radius plot)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for run in runs:
        r = np.linalg.norm(run["draws"], axis=-1).ravel()
        sns.kdeplot(x=r, ax=axes[0], lw=1.4, label=f"alpha={run['alpha']:g}")
    axes[0].hist(np.linalg.norm(reference, axis=1), bins=120, density=True,
                 alpha=0.2, color="grey", label="exact")
    axes[0].axvline(cfg.R, color="k", ls="--", lw=1)
    axes[0].set_xlabel("||x||"); axes[0].set_title("radius density")
    axes[0].legend(fontsize=8)
    r0 = np.linalg.norm(runs[0]["raw"], axis=-1)
    for c in range(r0.shape[0]):
        axes[1].plot(r0[c], lw=0.4, alpha=0.8)
    axes[1].axhline(cfg.R, color="k", ls="--", lw=1, label="R")
    axes[1].axvline(cfg.burn_in, color="r", ls=":", lw=1, label="end of burn-in")
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("||x_k||")
    axes[1].set_title(f"radius trace (alpha = {runs[0]['alpha']:g}), all iterations")
    axes[1].legend(fontsize=8)
    _save(fig, outdir / "radius.png", cfg.dpi)


def plot_acf(runs, cfg, outdir: Path) -> None:
    """Autocorrelation functions per coordinate, one panel per coordinate."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for run in runs:
        curves = autocorr_curves(run["draws"], cfg.max_acf_lag)
        for ax, nm in zip(axes, ["x1", "x2", "x3", "r"]):
            ax.plot(curves[nm], lw=1.3, label=f"alpha={run['alpha']:g}")
    for ax, nm in zip(axes, ["x1", "x2", "x3", "||x||"]):
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"ACF of {nm}")
        ax.set_xlabel("lag")
    axes[0].set_ylabel("autocorrelation")
    axes[0].legend(fontsize=8)
    fig.suptitle("Autocorrelation (chain-averaged, post burn-in, thinned)")
    _save(fig, outdir / "autocorrelation.png", cfg.dpi)


def plot_pairs(run, cfg, outdir: Path, n_max: int = 4000) -> None:
    """Pairwise scatter matrix (seaborn) of a thinned subsample."""
    flat = run["draws"].reshape(-1, 3)
    idx = np.linspace(0, len(flat) - 1, min(n_max, len(flat))).astype(int)
    df = pd.DataFrame(flat[idx], columns=["x1", "x2", "x3"])
    g = sns.pairplot(df, corner=True, diag_kind="kde",
                     plot_kws=dict(s=6, alpha=0.25, edgecolor="none"))
    g.figure.suptitle(f"Pairwise scatter — alpha = {run['alpha']:g}, h = {run['h']:g}",
                      y=1.02)
    path = outdir / f"pairs_alpha{run['alpha']:g}.png"
    g.figure.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(g.figure)
    print(f"  wrote {path}")


def plot_3d_with_sphere(runs, cfg, outdir: Path, n_max: int = 3000) -> None:
    """3-D scatter of the samples with the spherical boundary drawn (req. 12)."""
    n = len(runs)
    fig = plt.figure(figsize=(6 * n, 5.6))
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    sx = cfg.R * np.outer(np.cos(u), np.sin(v))
    sy = cfg.R * np.outer(np.sin(u), np.sin(v))
    sz = cfg.R * np.outer(np.ones_like(u), np.cos(v))
    for i, run in enumerate(runs):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        # wireframe of {||x|| = R}: the boundary the projection enforces
        ax.plot_wireframe(sx, sy, sz, color="grey", lw=0.3, alpha=0.5)
        flat = run["draws"].reshape(-1, 3)
        idx = np.linspace(0, len(flat) - 1, min(n_max, len(flat))).astype(int)
        p = flat[idx]
        onb = np.linalg.norm(p, axis=1) > cfg.R - 1e-9   # points sitting on dC
        ax.scatter(p[~onb, 0], p[~onb, 1], p[~onb, 2], s=3, alpha=0.3,
                   c="tab:blue", label="interior")
        if onb.any():
            ax.scatter(p[onb, 0], p[onb, 1], p[onb, 2], s=6, alpha=0.7,
                       c="tab:red", label="on boundary")
        ax.set_title(f"alpha = {run['alpha']:g}  (s = {cfg.s:g}, h = {run['h']:g})")
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
        ax.set_xlim(-cfg.R, cfg.R); ax.set_ylim(-cfg.R, cfg.R); ax.set_zlim(-cfg.R, cfg.R)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Samples inside C with the spherical boundary ||x|| = R")
    _save(fig, outdir / "samples_3d_sphere.png", cfg.dpi)


def plot_alpha_summaries(summary: pd.DataFrame, cfg: Config, outdir: Path) -> None:
    """ESS, ESS/s, projection fraction and R-hat as functions of alpha."""
    df = summary.sort_values("alpha")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    for col, lab in [("ess_x1", "x1"), ("ess_x2", "x2"), ("ess_x3", "x3"), ("ess_r", "||x||")]:
        ax.plot(df["alpha"], df[col], "o-", label=lab)
    ax.plot(df["alpha"], df["ess_min"], "k--", label="min over coords")
    ax.set_xlabel("alpha"); ax.set_ylabel("ESS"); ax.set_title("ESS vs alpha")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(df["alpha"], df["ess_per_sec_mean"], "o-", label="mean over coords")
    ax.plot(df["alpha"], df["ess_per_sec_min"], "s--", label="min over coords")
    ax.set_xlabel("alpha"); ax.set_ylabel("ESS / second")
    ax.set_title("ESS per second vs alpha"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(df["alpha"], 100 * df["projection_fraction"], "o-", color="tab:red")
    ax.axhline(100 * cfg.max_projection_fraction, ls="--", color="k", lw=1,
               label="warning threshold")
    ax.set_xlabel("alpha"); ax.set_ylabel("% of iterations projected")
    ax.set_title("Fraction of iterations requiring projection"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(df["alpha"], df["rhat_max"], "o-", color="tab:green", label="max R-hat")
    ax.axhline(1.01, ls="--", color="k", lw=1)
    ax2 = ax.twinx()
    if "ks_x1" in df:
        ax2.plot(df["alpha"], df["ks_x1"], "s--", color="tab:purple", label="KS(x1) vs exact")
        ax2.set_ylabel("KS distance", color="tab:purple")
    ax.set_xlabel("alpha"); ax.set_ylabel("max R-hat", color="tab:green")
    ax.set_title("Convergence and accuracy vs alpha")
    fig.suptitle(f"Non-reversible perturbation sweep (s = {cfg.s:g}, h = {cfg.h:g}, "
                 f"{cfg.n_chains} chains x {cfg.n_iter} iterations)")
    _save(fig, outdir / "ess_vs_alpha.png", cfg.dpi)


def plot_stepsize_study(step_summary: pd.DataFrame, cfg: Config, outdir: Path) -> None:
    """Step-size sensitivity: ESS, ESS/s, projection rate and bias vs h."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for alpha, grp in step_summary.groupby("alpha"):
        grp = grp.sort_values("h")
        axes[0].plot(grp["h"], grp["ess_mean"], "o-", label=f"alpha={alpha:g}")
        axes[1].plot(grp["h"], grp["ess_per_sec_mean"], "o-", label=f"alpha={alpha:g}")
        axes[2].plot(grp["h"], 100 * grp["projection_fraction"], "o-", label=f"alpha={alpha:g}")
        if "ks_x1" in grp:
            axes[3].plot(grp["h"], grp["ks_x1"], "o-", label=f"alpha={alpha:g}")
    for ax, ttl, yl in zip(
        axes,
        ["ESS vs step size", "ESS/s vs step size", "projection rate vs step size",
         "bias (KS on x1) vs step size"],
        ["ESS (mean over coords)", "ESS / second", "% projected", "KS distance to exact"],
    ):
        ax.set_xscale("log"); ax.set_xlabel("h"); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.legend(fontsize=8)
    fig.suptitle("Step-size sensitivity: h, h/2, h/4 (identical seeds and inits; "
                 "equal simulated time n_iter*h)")
    _save(fig, outdir / "step_size_sensitivity.png", cfg.dpi)


def plot_drift_decomposition(runs, cfg, outdir: Path) -> None:
    """Total drift norm vs non-reversible drift norm, and a(x) distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for run in runs:
        d = run["diag"]
        d = d[d["iteration"] >= cfg.burn_in]
        sns.kdeplot(x=d["drift_norm"], ax=axes[0], lw=1.3, label=f"alpha={run['alpha']:g}")
        if run["alpha"] > 0:
            sns.kdeplot(x=d["nr_drift_norm"], ax=axes[1], lw=1.3,
                        label=f"alpha={run['alpha']:g}")
        sns.kdeplot(x=d["a"], ax=axes[2], lw=1.3, label=f"alpha={run['alpha']:g}")
    axes[0].set_title("||b_k|| (total drift)"); axes[0].set_xlabel("drift norm")
    axes[1].set_title("||-a alpha J g|| (non-reversible part)")
    axes[1].set_xlabel("non-reversible drift norm")
    axes[2].set_title("anchor coefficient a(x)")
    axes[2].axvline(np.exp(-3 * cfg.delta), color="k", ls="--", lw=1,
                    label="exp(-3 delta)")
    axes[2].axvline(1.0, color="k", ls="--", lw=1)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Drift decomposition and anchor coefficient")
    _save(fig, outdir / "drift_and_anchor.png", cfg.dpi)


# =============================================================================
# 7. EXPERIMENT DRIVER
# =============================================================================
def main(argv: Sequence[str] | None = None) -> int:
    global CFG
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--quick", action="store_true", help="short smoke-test run")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--n-iter", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = Config()
    if args.quick:
        cfg.n_iter, cfg.burn_in, cfg.n_reference = 4_000, 800, 40_000
        cfg.alphas = (0.0, 1.0, 4.0)
        cfg.step_size_alphas = (0.0, 2.0)
    if args.n_iter:
        cfg.n_iter = args.n_iter
        cfg.burn_in = min(cfg.burn_in, cfg.n_iter // 5)
    if args.seed:
        cfg.base_seed = args.seed
    if args.outdir:
        cfg.outdir = args.outdir
    CFG = cfg

    sns.set_theme(style="whitegrid", context="notebook")
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 79)
    print("Projected Non-Reversible Anchored Langevin on the ball  ||x|| <= R")
    print("=" * 79)
    print("\n[Configuration]")
    for k, v in asdict(cfg).items():
        print(f"  {k:24s} = {v}")

    # ---------------------------------------------------------------- checks
    print("\n[1] Structural identities of J_s(x)  (skew, tangency, div J = 0)")
    jchecks = verify_J_properties(s=cfg.s, R=cfg.R, seed=cfg.base_seed)
    print(jchecks.to_string(index=False))
    print("\n[2] Anchor-coefficient bounds  exp(-3 delta) <= a(x) <= 1")
    abounds = verify_anchor_bounds(delta=cfg.delta, R=cfg.R, seed=cfg.base_seed)
    print(abounds.to_string(index=False))
    jchecks.to_csv(outdir / "checks_J.csv", index=False)
    abounds.to_csv(outdir / "checks_anchor.csv", index=False)

    # ------------------------------------------------------ reference sample
    print("\n[3] Exact reference sample (rejection from Laplace^3 truncated to C)")
    ref = reference_sample(cfg)
    print(f"  {len(ref)} exact draws; E|x1| = {np.abs(ref[:,0]).mean():.4f}, "
          f"E||x|| = {np.linalg.norm(ref,axis=1).mean():.4f}, "
          f"P(||x||>0.95R) = {np.mean(np.linalg.norm(ref,axis=1)>0.95*cfg.R):.4f}")

    # ------------------------------------------------------------ alpha sweep
    print(f"\n[4] alpha sweep at h = {cfg.h} (s = {cfg.s} fixed; only alpha*s matters)")
    runs, rows, diag_frames, meta_frames = [], [], [], []
    for alpha in cfg.alphas:
        run = run_chains(cfg, alpha=alpha, h=cfg.h)
        row = summarize_run(run, cfg, reference=ref)
        runs.append(run); rows.append(row)
        diag_frames.append(run["diag"]); meta_frames.append(run["meta"])
        print(f"  alpha={alpha:5g}  runtime={row['runtime_s']:6.2f}s  "
              f"ESS(min)={row['ess_min']:8.1f}  ESS/s={row['ess_per_sec_min']:7.2f}  "
              f"Rhat={row['rhat_max']:.4f}  proj={100*row['projection_fraction']:5.2f}%  "
              f"KS(x1)={row['ks_x1']:.4f}")
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "summary_alpha.csv", index=False)

    diagnostics = pd.concat(diag_frames, ignore_index=True)     # requirement 16
    chain_meta = pd.concat(meta_frames, ignore_index=True)
    chain_meta.to_csv(outdir / "per_chain_meta.csv", index=False)
    # per-iteration diagnostics are large: store the alpha=0 / best-alpha ones
    best_alpha = float(summary.loc[summary["ess_per_sec_min"].idxmax(), "alpha"])
    keep = diagnostics["alpha"].isin([0.0, best_alpha])
    diagnostics.loc[keep].to_csv(outdir / "diagnostics_per_iteration.csv.gz",
                                 index=False, compression="gzip")

    print("\n[5] alpha-sweep summary")
    cols = ["alpha", "h", "runtime_s", "ess_min", "ess_mean", "ess_per_sec_min",
            "ess_per_sec_mean", "iact_mean", "rhat_max", "projection_fraction",
            "max_norm", "a_min", "a_max", "ks_x1", "bias_mean_abs_x1"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    base = summary.loc[summary["alpha"] == 0.0, "ess_per_sec_min"].iloc[0]
    print(f"\n  Best alpha by ESS/s: {best_alpha:g} "
          f"(speed-up over alpha=0: "
          f"{summary['ess_per_sec_min'].max()/base:.2f}x)")

    # --------------------------------------------------- step-size experiment
    print(f"\n[6] Step-size sensitivity: h in "
          f"{[cfg.h*f for f in cfg.step_size_factors]}")
    if cfg.step_size_match_simulated_time:
        print("    (n_iter scaled by 1/factor so that the simulated time "
              "n_iter*h is identical across h)")
    step_rows = []
    for alpha in cfg.step_size_alphas:
        for f in cfg.step_size_factors:
            hh = cfg.h * f
            # Equal simulated time => the residual discrepancy vs the exact
            # sample isolates the O(h) Euler + projection bias.
            scale = int(round(1.0 / f)) if cfg.step_size_match_simulated_time else 1
            r = run_chains(cfg, alpha=alpha, h=hh,
                           n_iter=cfg.n_iter * scale, burn_in=cfg.burn_in * scale)
            row = summarize_run(r, cfg, reference=ref)
            step_rows.append(row)
            print(f"  alpha={alpha:4g} h={hh:7.5f} n_iter={row['n_iter_per_chain']:7d} "
                  f"T={row['sim_time']:6.1f}  ESS={row['ess_mean']:8.1f}  "
                  f"ESS/s={row['ess_per_sec_mean']:7.2f}  "
                  f"proj={100*row['projection_fraction']:5.2f}%  "
                  f"KS(x1)={row['ks_x1']:.4f}  bias E|x1|={row['bias_mean_abs_x1']:+.4f}")
    step_summary = pd.DataFrame(step_rows)
    step_summary.to_csv(outdir / "summary_step_size.csv", index=False)

    # ------------------------------------------------------------------ plots
    print("\n[7] Figures")
    plot_traces(runs[0], cfg, outdir)
    if len(runs) > 1:
        plot_traces(runs[-1], cfg, outdir)
    plot_histograms(runs, cfg, ref, outdir)
    plot_acf(runs, cfg, outdir)
    plot_pairs(runs[0], cfg, outdir)
    if len(runs) > 1:
        plot_pairs(runs[-1], cfg, outdir)
    show3d = [runs[0]] + ([runs[cfg.alphas.index(best_alpha)]] if best_alpha != 0.0 else [])
    plot_3d_with_sphere(show3d if len(show3d) > 1 else runs[:2], cfg, outdir)
    plot_alpha_summaries(summary, cfg, outdir)
    plot_stepsize_study(step_summary, cfg, outdir)
    plot_drift_decomposition(runs, cfg, outdir)

    # --------------------------------------------------------------- warnings
    print("\n[8] Validation warnings")
    if WARNINGS_LOG:
        for w in WARNINGS_LOG:
            print("  WARNING:", w)
    else:
        print("  none: all iterates finite, ||x_k|| <= R, a(x) in [exp(-3 delta), 1],")
        print("        projection rate below threshold, R-hat <= 1.01.")

    print(f"\nAll outputs written to: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
