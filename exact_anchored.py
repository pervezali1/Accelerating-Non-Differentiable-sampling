"""Anchored Langevin with the EXACT gradient and an l1 (LASSO) non-differentiable potential.

Target (per the new specification), with w = (w_0, w_1, ..., w_9) in R^10:

    U(w)   = sum_j [softplus(x_j.w) - y_j x_j.w] + w_0^2/(2 sigma^2)
             + lambda_lasso * sum_{j>=1} |w_j|                               <- NOT differentiable
    U_0(w) = sum_j [softplus(x_j.w) - y_j x_j.w] + w_0^2/(2 sigma^2)
             + lambda_lasso * sum_{j>=1} sqrt(w_j^2 + delta^2)               <- smooth anchor

    a(w) = exp(U(w) - U_0(w))
         = exp(-lambda_lasso * sum_{j>=1} [sqrt(w_j^2 + delta^2) - |w_j|])

Update actually implemented (exactly as specified, including the sign of the J term, which is
now PLUS -- the previous runs used minus; J is skew so both are legitimate, they just rotate
the opposite way):

    w_{k+1} = Pi_K[ w_k - eta a(w_k) grad U_0(w_k)
                        + eta alpha a(w_k) J_s(w_k) grad U_0(w_k)
                        + sqrt(2 eta a(w_k)) xi_{k+1} ]

Why this anchor is the right one here. 0 <= sqrt(t^2+delta^2) - |t| <= delta, so
exp(-lambda d delta) <= a <= 1 everywhere: a is bounded without any reference to the constraint.
The invariance argument is unchanged -- a*pi is proportional to exp(-U_0), J is skew with
div J = 0, and J n = 0 on the boundary -- so the continuous reflected process still targets
pi_K proportional to exp(-U) 1_K, with U the true non-differentiable potential.

DECLARED CHOICES (the specification fixes the form, not these numbers):
  * d = 10: an intercept w_0 plus the nine features used before. The Gaussian term acts on w_0
    only and the LASSO on w_1..w_9, which is exactly the 3 x 3 block structure J needs.
  * K constrains the FULL vector (ball ||w||^2 <= 2, or g(w) = sum_{i=0}^{9}(w_i^2+eps^2)^{p/2}
    <= Lambda). Leaving w_0 free would make K non-compact and break the reflection argument.
    Note g_min is now 10 eps^p, so D = Lambda - 10 eps^p.
  * sigma = 10 (weakly informative on the intercept).
  * lambda_lasso = LASSO_FRAC * n_train, so the penalty keeps a fixed relative weight against a
    SUMMED log-likelihood on datasets of very different size.
  * delta = log(2) / (9 * lambda_lasso), chosen so that a is in [1/2, 1] everywhere -- the same
    bound the earlier rho = log 2 anchor had, for continuity.
  * J acts on w_1..w_9 only; its intercept row and column are zero, which preserves J' = -J,
    div J = 0 and J n = 0 on the boundary of both geometries.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import Dict, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

from nral import (BallGeometry, SmoothedLpGeometry, Dataset, build_magic_dataset,
                  build_titanic_dataset, softplus, mean_sd, _wrap)

D10 = 10
BLOCKS10 = ((1, 2, 3), (4, 5, 6), (7, 8, 9))     # J never touches the intercept, index 0
SIGMA = 10.0
LASSO_FRAC = 0.01                                 # lambda_lasso = LASSO_FRAC * n_train
A_LOWER = 0.5                                     # target lower bound for a; fixes delta
STYLE = {"rev":  dict(color="#1f5fbf", linestyle="-", linewidth=2.0,
                      label="Reversible anchored Langevin (exact grad)"),
         "nrev": dict(color="#1a9850", linestyle="-", linewidth=2.0,
                      label="Non-reversible anchored Langevin (exact grad)")}


# --------------------------------------------------------------------------- potential
@dataclasses.dataclass(frozen=True)
class Potential:
    X: np.ndarray            # (n, 10), column 0 is the intercept
    y: np.ndarray            # (n,)
    sigma: float
    lam: float               # lambda_lasso
    delta: float

    def loglik_part(self, W):
        z = np.atleast_2d(W) @ self.X.T
        return np.sum(softplus(z) - self.y[None, :] * z, axis=1)

    def U(self, W):
        """The true, non-differentiable potential."""
        W2 = np.atleast_2d(W)
        out = (self.loglik_part(W2) + W2[:, 0] ** 2 / (2 * self.sigma ** 2)
               + self.lam * np.sum(np.abs(W2[:, 1:]), axis=1))
        return out[0] if np.ndim(W) == 1 else out

    def U0(self, W):
        """The smooth anchor: |w_j| replaced by sqrt(w_j^2 + delta^2)."""
        W2 = np.atleast_2d(W)
        out = (self.loglik_part(W2) + W2[:, 0] ** 2 / (2 * self.sigma ** 2)
               + self.lam * np.sum(np.sqrt(W2[:, 1:] ** 2 + self.delta ** 2), axis=1))
        return out[0] if np.ndim(W) == 1 else out

    def grad_U0(self, W):
        """EXACT gradient of the smooth anchor -- no mini-batching anywhere."""
        W2 = np.atleast_2d(W)
        resid = expit(W2 @ self.X.T) - self.y[None, :]       # (R, n)
        g = resid @ self.X                                    # (R, 10)
        g[:, 0] += W2[:, 0] / self.sigma ** 2
        g[:, 1:] += self.lam * W2[:, 1:] / np.sqrt(W2[:, 1:] ** 2 + self.delta ** 2)
        return g[0] if np.ndim(W) == 1 else g

    def anchor_a(self, W):
        """a = exp(U - U_0) = exp(-lambda sum_j [sqrt(w_j^2+delta^2) - |w_j|]).  Exact, never
        estimated: it depends only on w, not on the data."""
        W2 = np.atleast_2d(W)
        gap = np.sum(np.sqrt(W2[:, 1:] ** 2 + self.delta ** 2) - np.abs(W2[:, 1:]), axis=1)
        out = np.exp(-self.lam * gap)
        return out[0] if np.ndim(W) == 1 else out

    @property
    def a_lower_bound(self) -> float:
        """exp(-lambda * 9 * delta): the worst case, attained at w_1..w_9 = 0."""
        return float(np.exp(-self.lam * 9 * self.delta))


def make_potential(ds: Dataset, lasso_frac: float = LASSO_FRAC, sigma: float = SIGMA,
                   a_lower: float = A_LOWER, split: str = "train") -> Potential:
    X = ds.X_train if split == "train" else ds.X_test
    y = ds.y_train if split == "train" else ds.y_test
    Xi = np.column_stack([np.ones(len(X)), X])            # prepend the intercept column
    lam = lasso_frac * ds.n_train
    delta = -np.log(a_lower) / (9.0 * lam)                # so that a >= a_lower everywhere
    return Potential(np.ascontiguousarray(Xi), y.astype(float), sigma, lam, delta)


# --------------------------------------------------------------------------- geometry in R^10
def expand_scales10(scales: Sequence[float]) -> np.ndarray:
    v = np.zeros(D10)
    v[1:] = np.repeat(np.asarray(scales, dtype=float), 3)
    return v


def block_cross10(w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """blockdiag(0, [w_I1]_x, [w_I2]_x, [w_I3]_x) @ v -- the intercept row/col is zero."""
    out = np.zeros_like(v)
    for blk in BLOCKS10:
        sl = slice(blk[0], blk[-1] + 1)
        out[..., sl] = np.cross(w[..., sl], v[..., sl])
    return out


def J_axes10(W: np.ndarray, geom, scales: Sequence[float]) -> np.ndarray:
    """Ball: s * w.  Smoothed l_p: -s * grad g(w).  Both zeroed on the intercept."""
    e = expand_scales10(scales)
    return e * W if isinstance(geom, BallGeometry) else -e * geom.grad_g(W)


def apply_J10(W, V, geom, scales):
    return block_cross10(J_axes10(W, geom, scales), V)


def explicit_J10(w, geom, scales) -> np.ndarray:
    ax = J_axes10(np.asarray(w, dtype=float), geom, scales)
    J = np.zeros((D10, D10))
    for blk in BLOCKS10:
        sl = slice(blk[0], blk[-1] + 1)
        a1, a2, a3 = ax[blk[0]], ax[blk[1]], ax[blk[2]]
        J[sl, sl] = np.array([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])
    return J


def make_geom10(kind: str, *, r2: float = 2.0, p: float = 2.4, eps: float = 0.2,
                Lam: float = 4.0):
    if kind == "ball":
        return BallGeometry(r2=r2)
    return SmoothedLpGeometry(p=p, eps=eps, Lam=Lam, d=D10)      # g_min = 10 eps^p


def init_unit_ball10(rng: np.random.Generator, n_rows: int) -> np.ndarray:
    Z = rng.standard_normal((n_rows, D10))
    V = rng.random(n_rows)
    return (V ** (1.0 / D10))[:, None] * Z / np.linalg.norm(Z, axis=1, keepdims=True)


# --------------------------------------------------------------------------- sampler
def run_exact(pot: Potential, geom, *, alpha: int, scales, eta: float, n_iter: int,
              W0: np.ndarray, seed: int, checkpoint_every: int = 10) -> Dict[str, object]:
    """w <- Pi_K[ w - eta a grad U_0 + eta alpha a J grad U_0 + sqrt(2 eta a) xi ]."""
    W = np.array(W0, dtype=float, copy=True)
    R = W.shape[0]
    rng = np.random.default_rng(seed)
    assert np.all(geom.feasible(W)), "initial states must lie in K"
    ck = list(range(0, n_iter + 1, checkpoint_every))
    if ck[-1] != n_iter:
        ck.append(n_iter)
    idx = {k: i for i, k in enumerate(ck)}
    Ws = np.empty((len(ck), R, D10))
    proj = np.zeros(max(n_iter, 1))
    n_bad = 0
    t0 = time.perf_counter()
    for k in range(n_iter):
        i = idx.get(k)
        if i is not None:
            Ws[i] = W
        G = pot.grad_U0(W)                       # EXACT gradient
        a = pot.anchor_a(W)                      # exact, from the geometry of the penalty
        step = -eta * a[:, None] * G
        if alpha:
            step = step + eta * alpha * a[:, None] * apply_J10(W, G, geom, scales)
        prop = W + step + np.sqrt(2.0 * eta * a)[:, None] * rng.standard_normal((R, D10))
        bad = ~np.isfinite(prop).all(axis=1)
        if bad.any():
            n_bad += int(bad.sum()); prop[bad] = W[bad]
        W, was = geom.project(prop)
        proj[k] = np.count_nonzero(was)
    Ws[-1] = W
    return dict(checkpoints=np.asarray(ck), betas=Ws, projection_rate=float(proj.sum() / max(1, n_iter * R)),
                n_nonfinite=n_bad, runtime=time.perf_counter() - t0)


def accuracy10(X: np.ndarray, y: np.ndarray, Ws: np.ndarray) -> np.ndarray:
    Xi = np.column_stack([np.ones(len(X)), X])
    yb = y.astype(bool)
    out = np.empty(Ws.shape[:2])
    for i in range(Ws.shape[0]):
        out[i] = np.mean(((Ws[i] @ Xi.T) >= 0.0) == yb[None, :], axis=1)
    return out


# --------------------------------------------------------------------------- experiment driver
@dataclasses.dataclass(frozen=True)
class Exp10:
    key: str
    dataset: str
    geometry: str
    n_iter: int
    eps: float = 0.20
    Lam: float = 4.0
    p: float = 2.4
    r2: float = 2.0


EXPS = (
    Exp10("magic_ball",   "magic",   "ball", 1000),
    Exp10("magic_lp",     "magic",   "lp",   1000, eps=0.20),
    Exp10("titanic_ball", "titanic", "ball", 1500),
    Exp10("titanic_lp",   "titanic", "lp",   2000, eps=0.18),
)
ETA_GRID = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
S_GRID = [0.0, 0.25, 1.0, 2.0, 5.0]
SEARCH_SEED, CONFIRM_SEED = 3000, 4100


def geom_for(e: Exp10):
    return make_geom10(e.geometry, r2=e.r2, p=e.p, eps=e.eps, Lam=e.Lam)


def run_cell(ds: Dataset, pot: Potential, e: Exp10, eta: float, scales, n_reps: int, seed: int):
    geom = geom_for(e)
    W0 = init_unit_ball10(np.random.default_rng(seed + 1), n_reps)
    assert np.all(geom.feasible(W0))
    arms = {}
    for tag, alpha in (("rev", 0), ("nrev", 1)):
        r = run_exact(pot, geom, alpha=alpha, scales=scales, eta=eta, n_iter=e.n_iter,
                      W0=W0, seed=seed)                      # same noise stream for both arms
        Ws = r["betas"]
        arms[tag] = dict(checkpoints=r["checkpoints"], betas=Ws,
                         acc_train=accuracy10(ds.X_train, ds.y_train, Ws),
                         acc_test=accuracy10(ds.X_test, ds.y_test, Ws),
                         U=pot.U(Ws[-1]), a=pot.anchor_a(Ws[-1]),
                         projection_rate=r["projection_rate"], n_nonfinite=r["n_nonfinite"])
    return arms


def paired(arms) -> dict:
    d_tr = arms["nrev"]["acc_train"][-1] - arms["rev"]["acc_train"][-1]
    d_te = arms["nrev"]["acc_test"][-1] - arms["rev"]["acc_test"][-1]
    n = len(d_tr)
    se_tr = d_tr.std(ddof=1) / np.sqrt(n) + 1e-300
    se_te = d_te.std(ddof=1) / np.sqrt(n) + 1e-300
    return dict(d_train=float(d_tr.mean()), se_train=float(se_tr),
                t_train=float(d_tr.mean() / se_tr), win_train=float((d_tr > 0).mean()),
                d_test=float(d_te.mean()), se_test=float(se_te),
                t_test=float(d_te.mean() / se_te), win_test=float((d_te > 0).mean()))


def figure10(e: Exp10, ds: Dataset, arms, eta: float, scales, n_reps: int, outdir: str, note: str):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, split in zip(axes, ("train", "test")):
        for tag in ("rev", "nrev"):
            x = np.asarray(arms[tag]["checkpoints"], dtype=float)
            mu, sd = mean_sd(arms[tag][f"acc_{split}"], axis=1)
            st = STYLE[tag]
            ax.fill_between(x, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                            color=st["color"], alpha=0.16, linewidth=0)
            ax.plot(x, mu, color=st["color"], linestyle=st["linestyle"],
                    linewidth=st["linewidth"], label=st["label"])
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        gname = "ball" if e.geometry == "ball" else f"smoothed $\\ell_p$ ($p={e.p:g}$)"
        ax.set_title(f"{e.dataset.upper()}, {gname} -- exact gradient, LASSO anchor -- "
                     f"{'training' if split == 'train' else 'test'} accuracy", fontsize=9.5)
        i0 = int(0.4 * len(x))
        ins = ax.inset_axes([0.50, 0.12, 0.46, 0.40])
        lo, hi = [], []
        for tag in ("rev", "nrev"):
            mu, sd = mean_sd(arms[tag][f"acc_{split}"], axis=1)
            se = sd / np.sqrt(arms[tag][f"acc_{split}"].shape[1])
            ins.fill_between(x[i0:], mu[i0:] - se[i0:], mu[i0:] + se[i0:],
                             color=STYLE[tag]["color"], alpha=0.22, linewidth=0)
            ins.plot(x[i0:], mu[i0:], color=STYLE[tag]["color"], linewidth=1.5)
            lo.append((mu[i0:] - se[i0:]).min()); hi.append((mu[i0:] + se[i0:]).max())
        pad = 0.12 * (max(hi) - min(lo) + 1e-9)
        ins.set_ylim(min(lo) - pad, max(hi) + pad); ins.set_xlim(x[i0], x[-1])
        ins.tick_params(labelsize=6.5); ins.grid(True, alpha=0.25, linewidth=0.5)
        ins.set_title("zoom (mean $\\pm$ 1 s.e.)", fontsize=6.5, pad=2)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.92)
    pot = arms["_pot"]
    gtxt = ("ball ||w||^2 <= 2 (radius sqrt(2))" if e.geometry == "ball" else
            f"smoothed l_p  g(w) = sum_i (w_i^2+eps^2)^(p/2) <= Lambda, p = {e.p:g}, "
            f"eps = {e.eps:g}, Lambda = {e.Lam:g}")
    cap = (f"{e.dataset.upper()} -- {gtxt}.  d = 10 (intercept w_0 + 9 features).  "
           f"U = sum softplus - y x.w + w_0^2/(2 sigma^2) + lambda sum_{{j>=1}} |w_j|;  "
           f"U_0 replaces |w_j| by sqrt(w_j^2 + delta^2);  a = exp(U - U_0).  "
           f"EXACT gradient (no mini-batching).  sigma = {pot.sigma:g}, "
           f"lambda_lasso = {pot.lam:.3g} = {LASSO_FRAC:g} n_train, delta = {pot.delta:.3g} "
           f"(so a in [{pot.a_lower_bound:.2f}, 1]).  Block strengths s = "
           f"({', '.join(f'{v:g}' for v in scales)}), acting on w_1..w_9 only.  "
           f"Step eta = {eta:g} (actually used).  R = {n_reps} independent replicates on one "
           f"fixed stratified 80/20 split (n_train = {ds.n_train}, n_test = {ds.n_test}).  "
           f"Bands: mean +- 1 SD (ddof = 1) across replicates -- repeat-run variability at a "
           f"fixed split, not a confidence or credible interval.  {note}")
    fig.suptitle(f"{e.key}: exact-gradient anchored Langevin, s = "
                 f"({', '.join(f'{v:g}' for v in scales)})", fontsize=11, y=0.995)
    fig.text(0.5, -0.09, _wrap(cap), ha="center", va="top", fontsize=7.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = os.path.join(outdir, f"{e.key}_exact_anchored.png")
    pdf = os.path.join(outdir, f"{e.key}_exact_anchored.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-reps", type=int, default=40)
    ap.add_argument("--final-reps", type=int, default=150)
    ap.add_argument("--outdir", default="figures/exact")
    ap.add_argument("--resdir", default="results/exact")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.sweep_reps, args.final_reps = 4, 6
    os.makedirs(args.outdir, exist_ok=True); os.makedirs(args.resdir, exist_ok=True)

    datasets = {"magic": build_magic_dataset(), "titanic": build_titanic_dataset()}
    pots = {k: make_potential(v) for k, v in datasets.items()}
    for k, p in pots.items():
        print(f"[{k}] n_train = {datasets[k].n_train}  lambda_lasso = {p.lam:.4g}  "
              f"delta = {p.delta:.4g}  sigma = {p.sigma:g}  a in [{p.a_lower_bound:.4f}, 1]")
    exps = [dataclasses.replace(e, n_iter=120) for e in EXPS] if args.quick else list(EXPS)

    rows, chosen, t0 = [], {}, time.perf_counter()
    for e in exps:
        ds, pot = datasets[e.dataset], pots[e.dataset]
        for eta in ETA_GRID:
            for s in S_GRID:
                arms = run_cell(ds, pot, e, eta, (s, s, s), args.sweep_reps, SEARCH_SEED)
                st = paired(arms)
                rows.append(dict(experiment=e.key, eta=eta, s=s,
                                 rev_train=float(arms["rev"]["acc_train"][-1].mean()),
                                 nrev_train=float(arms["nrev"]["acc_train"][-1].mean()),
                                 rev_test=float(arms["rev"]["acc_test"][-1].mean()),
                                 nrev_test=float(arms["nrev"]["acc_test"][-1].mean()),
                                 rev_U=float(arms["rev"]["U"].mean()),
                                 nrev_U=float(arms["nrev"]["U"].mean()),
                                 proj_nrev=arms["nrev"]["projection_rate"], **st))
            print(f"  [{e.key}] eta = {eta:g} done ({time.perf_counter()-t0:.0f}s)")
        sub = pd.DataFrame([r for r in rows if r["experiment"] == e.key and r["s"] > 0])
        best = sub.loc[sub.d_train.idxmax()]
        chosen[e.key] = dict(eta=float(best.eta), s=float(best.s),
                             d_train=float(best.d_train), t_train=float(best.t_train))
        print(f"\n=== {e.key}: top cells by PAIRED TRAINING gain (R = {args.sweep_reps}) ===")
        print(sub.sort_values("d_train", ascending=False).head(6)
              [["eta", "s", "rev_train", "nrev_train", "d_train", "se_train", "t_train",
                "rev_test", "nrev_test", "proj_nrev"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        print(f"  -> selected (training only): eta = {best.eta:g}, s = {best.s:g}")
    pd.DataFrame(rows).to_csv(os.path.join(args.resdir, "exact_sweep.csv"), index=False)

    print("\n" + "=" * 94)
    print(f"CONFIRMATION on independent replicates (seed {CONFIRM_SEED}, R = {args.final_reps})")
    print("=" * 94)
    out = {}
    for e in exps:
        c = chosen[e.key]
        ds, pot = datasets[e.dataset], pots[e.dataset]
        arms = run_cell(ds, pot, e, c["eta"], (c["s"],) * 3, args.final_reps, CONFIRM_SEED)
        st = paired(arms)
        verdict = ("non-reversible WINS" if st["t_train"] > 2 and st["d_train"] > 0 else
                   "non-reversible LOSES" if st["t_train"] < -2 else "no significant difference")
        arms["_pot"] = pot
        note = (f"eta and s selected by a TRAINING-ONLY sweep over {len(ETA_GRID)} step sizes x "
                f"{len(S_GRID)-1} strengths; these curves are a CONFIRMATION run on independent "
                f"replicates (seed {CONFIRM_SEED}) not used for that selection. "
                f"Verdict on the paired training difference: {verdict}.")
        png, pdf = figure10(e, ds, arms, c["eta"], (c["s"],) * 3, args.final_reps, args.outdir, note)
        np.savez_compressed(os.path.join(args.resdir, f"{e.key}_exact.npz"),
                            checkpoints=arms["rev"]["checkpoints"],
                            rev_acc_train=arms["rev"]["acc_train"], rev_acc_test=arms["rev"]["acc_test"],
                            nrev_acc_train=arms["nrev"]["acc_train"], nrev_acc_test=arms["nrev"]["acc_test"],
                            rev_w=arms["rev"]["betas"], nrev_w=arms["nrev"]["betas"],
                            eta=c["eta"], s=c["s"], lam=pot.lam, delta=pot.delta)
        out[e.key] = dict(eta=c["eta"], s=c["s"], verdict=verdict, paired=st, png=png, pdf=pdf,
                          lam=pot.lam, delta=pot.delta, sigma=pot.sigma,
                          rev=dict(train=float(arms["rev"]["acc_train"][-1].mean()),
                                   train_sd=float(arms["rev"]["acc_train"][-1].std(ddof=1)),
                                   test=float(arms["rev"]["acc_test"][-1].mean()),
                                   test_sd=float(arms["rev"]["acc_test"][-1].std(ddof=1)),
                                   U=float(arms["rev"]["U"].mean()), a=float(arms["rev"]["a"].mean()),
                                   proj=arms["rev"]["projection_rate"]),
                          nrev=dict(train=float(arms["nrev"]["acc_train"][-1].mean()),
                                    train_sd=float(arms["nrev"]["acc_train"][-1].std(ddof=1)),
                                    test=float(arms["nrev"]["acc_test"][-1].mean()),
                                    test_sd=float(arms["nrev"]["acc_test"][-1].std(ddof=1)),
                                    U=float(arms["nrev"]["U"].mean()), a=float(arms["nrev"]["a"].mean()),
                                    proj=arms["nrev"]["projection_rate"]))
        r, nr = out[e.key]["rev"], out[e.key]["nrev"]
        print(f"\n[{e.key}]  eta = {c['eta']:g}, s = {c['s']:g}, R = {args.final_reps}")
        print(f"  Reversible      train {r['train']:.4f}+-{r['train_sd']:.4f}  "
              f"test {r['test']:.4f}+-{r['test_sd']:.4f}  U={r['U']:.1f}  a={r['a']:.4f}  proj={r['proj']:.3f}")
        print(f"  Non-reversible  train {nr['train']:.4f}+-{nr['train_sd']:.4f}  "
              f"test {nr['test']:.4f}+-{nr['test_sd']:.4f}  U={nr['U']:.1f}  a={nr['a']:.4f}  proj={nr['proj']:.3f}")
        print(f"  paired nrev - rev: train {st['d_train']:+.4f} +- {st['se_train']:.4f} "
              f"(t = {st['t_train']:+.2f}, wins {st['win_train']:.0%})   "
              f"test {st['d_test']:+.4f} +- {st['se_test']:.4f} (t = {st['t_test']:+.2f})")
        print(f"  VERDICT: {verdict}")
    with open(os.path.join(args.resdir, "exact_confirmation.json"), "w") as fh:
        json.dump(dict(eta_grid=ETA_GRID, s_grid=S_GRID, sigma=SIGMA, lasso_frac=LASSO_FRAC,
                       a_lower=A_LOWER, search_seed=SEARCH_SEED, confirm_seed=CONFIRM_SEED,
                       sweep_reps=args.sweep_reps, final_reps=args.final_reps,
                       selection="paired TRAINING gain only; test labels unused",
                       chosen=chosen, confirmation=out), fh, indent=2, default=float)
    print(f"\ntotal {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
