"""Does the non-reversible arm gain a lead as the LASSO weight lambda grows from 5 to 20?

Setup is exactly the one at commit 091b02b (exact_anchored.py): exact gradient, d = 10
(intercept + 9 features), K = ball or smoothed l_p, and

    U   = sum_j [softplus(x_j.w) - y_j x_j.w] + w_0^2/(2 sigma^2) + lambda sum_{j>=1} |w_j|
    U_0 = same with |w_j| -> sqrt(w_j^2 + delta^2)
    a   = exp(U - U_0)
    w <- Pi_K[ w - eta a grad U_0 + eta alpha a J_s(w) grad U_0 + sqrt(2 eta a) xi ]

The ONLY thing varied here is lambda, over {5, 7.5, 10, 15, 20} (absolute, not a fraction of
n_train: at 091b02b the rule lambda = 0.01 n_train gave 152.2 on MAGIC and 7.12 on Titanic, so
this range puts the two datasets on the same footing).

COUPLING TO DECLARE: delta = log(2) / (9 lambda) is kept, so that a stays in [1/2, 1] for every
lambda. Raising lambda therefore also sharpens the smoothing (delta falls from 0.0154 at
lambda = 5 to 0.00385 at lambda = 20). The two cannot both be held fixed while keeping the anchor
bound constant; delta is reported in every row so the coupling is visible.

Two step sizes per dataset -- one where the chain is mid-transient, one where it has converged --
because the (eta, s) grid showed the answer depends on which regime you are in. Both s = 1 and
s = 5 are run. Everything is reported; nothing is selected.
"""
from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exact_anchored import (D10, Exp10, CONFIRM_SEED, Potential, accuracy10, build_magic_dataset,
                            build_titanic_dataset, geom_for, init_unit_ball10, run_exact)
from nral import mean_sd, _wrap

LAMBDAS = [5.0, 7.5, 10.0, 15.0, 20.0]
S_VALS = [1.0, 5.0]
SIGMA = 10.0
A_LOWER = 0.5
R = 100
EXPS = (Exp10("magic_ball",   "magic",   "ball", 1000),
        Exp10("magic_lp",     "magic",   "lp",   1000, eps=0.20),
        Exp10("titanic_ball", "titanic", "ball", 1500),
        Exp10("titanic_lp",   "titanic", "lp",   2000, eps=0.18))
ETAS = {"magic": [1e-6, 1e-5], "titanic": [1e-5, 3e-5]}
OUT, RES = "figures/lambda", "results/lambda"


def potential_for(ds, lam: float) -> Potential:
    """Same construction as exact_anchored.make_potential, with lambda given absolutely."""
    Xi = np.column_stack([np.ones(len(ds.X_train)), ds.X_train])
    delta = -np.log(A_LOWER) / (9.0 * lam)          # keeps a in [A_LOWER, 1]
    return Potential(np.ascontiguousarray(Xi), ds.y_train.astype(float), SIGMA, lam, delta)


def arm(ds, pot, e, eta, alpha, s, W0, seed):
    r = run_exact(pot, geom_for(e), alpha=alpha, scales=(s, s, s), eta=eta, n_iter=e.n_iter,
                  W0=W0, seed=seed)
    W = r["betas"]
    wl = W[-1][:, 1:]
    return dict(ck=r["checkpoints"], tr=accuracy10(ds.X_train, ds.y_train, W),
                te=accuracy10(ds.X_test, ds.y_test, W), U=pot.U(W[-1]), a=pot.anchor_a(W[-1]),
                pen=pot.lam * np.abs(wl).sum(axis=1),          # the l1 term itself
                near_kink=(np.abs(wl) < 0.05).sum(axis=1).astype(float),
                proj=r["projection_rate"], nfin=r["n_nonfinite"])


def figure_lambda(e, ds, pot, eta, rev, arms, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    cols = {1.0: "#a6d96a", 5.0: "#00602f"}
    for ax, split in zip(axes, ("tr", "te")):
        x = np.asarray(rev["ck"], dtype=float)
        mu0, sd0 = mean_sd(rev[split], axis=1)
        ax.fill_between(x, np.clip(mu0 - sd0, 0, 1), np.clip(mu0 + sd0, 0, 1),
                        color="#1f5fbf", alpha=0.14, linewidth=0)
        ax.plot(x, mu0, color="#1f5fbf", linewidth=2.0, label="Reversible anchored Langevin")
        for s in S_VALS:
            mu, sd = mean_sd(arms[s][split], axis=1)
            ax.fill_between(x, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                            color=cols[s], alpha=0.12, linewidth=0)
            ax.plot(x, mu, color=cols[s], linewidth=1.8, label=f"Non-reversible, s = {s:g}")
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        gname = "ball" if e.geometry == "ball" else f"smoothed $\\ell_p$ ($p={e.p:g}$)"
        ax.set_title(f"{e.dataset.upper()}, {gname}, $\\lambda$ = {pot.lam:g}, $\\eta$ = {eta:g} "
                     f"-- {'training' if split == 'tr' else 'test'} accuracy", fontsize=9.5)
        i0 = int(0.4 * len(x))
        ins = ax.inset_axes([0.50, 0.12, 0.46, 0.40])
        lo, hi = [], []
        for nm, a, c in [("rev", rev, "#1f5fbf")] + [(s, arms[s], cols[s]) for s in S_VALS]:
            mu, sd = mean_sd(a[split], axis=1)
            se = sd / np.sqrt(a[split].shape[1])
            ins.fill_between(x[i0:], mu[i0:] - se[i0:], mu[i0:] + se[i0:], color=c, alpha=0.20,
                             linewidth=0)
            ins.plot(x[i0:], mu[i0:], color=c, linewidth=1.4)
            lo.append((mu[i0:] - se[i0:]).min()); hi.append((mu[i0:] + se[i0:]).max())
        pad = 0.12 * (max(hi) - min(lo) + 1e-9)
        ins.set_ylim(min(lo) - pad, max(hi) + pad); ins.set_xlim(x[i0], x[-1])
        ins.tick_params(labelsize=6.5); ins.grid(True, alpha=0.25, linewidth=0.5)
        ins.set_title("zoom (mean $\\pm$ 1 s.e.)", fontsize=6.5, pad=2)
    axes[0].legend(loc="upper left", fontsize=7.4, framealpha=0.92)
    gtxt = ("ball ||w||^2 <= 2 (radius sqrt(2))" if e.geometry == "ball" else
            f"smoothed l_p  g(w) = sum_i (w_i^2+eps^2)^(p/2) <= Lambda, p = {e.p:g}, "
            f"eps = {e.eps:g}, Lambda = {e.Lam:g}")
    cap = (f"{e.dataset.upper()} -- {gtxt}.  d = 10 (intercept w_0 + 9 features).  "
           f"U = sum softplus - y x.w + w_0^2/(2 sigma^2) + lambda sum_{{j>=1}} |w_j|;  U_0 "
           f"replaces |w_j| by sqrt(w_j^2 + delta^2);  a = exp(U - U_0).  EXACT gradient.  "
           f"sigma = {pot.sigma:g}.  lambda_lasso = {pot.lam:g} with delta = log2/(9 lambda) = "
           f"{pot.delta:.4g}, so a stays in [{pot.a_lower_bound:.2f}, 1] at every lambda "
           f"(raising lambda also sharpens the smoothing -- the two are coupled by that choice).  "
           f"Step eta = {eta:g}.  R = {R} independent replicates on one fixed stratified 80/20 "
           f"split (n_train = {ds.n_train}, n_test = {ds.n_test}).  Bands: mean +- 1 SD "
           f"(ddof = 1) -- repeat-run variability at a fixed split, not a confidence or credible "
           f"interval.  All lambda are reported in results/lambda/lambda_lasso.csv.")
    fig.suptitle(f"{e.key}: $\\lambda$ = {pot.lam:g}, $\\eta$ = {eta:g}", fontsize=11, y=0.995)
    fig.text(0.5, -0.09, _wrap(cap), ha="center", va="top", fontsize=7.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    nm = f"{e.key}_lam{pot.lam:g}_eta{eta:g}".replace("-", "m").replace(".", "p")
    png = os.path.join(outdir, f"{nm}.png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{nm}.pdf"), bbox_inches="tight")
    plt.close(fig)
    return png


def main() -> int:
    os.makedirs(OUT, exist_ok=True); os.makedirs(RES, exist_ok=True)
    datasets = {"magic": build_magic_dataset(), "titanic": build_titanic_dataset()}
    rows, t0 = [], time.perf_counter()
    for e in EXPS:
        ds = datasets[e.dataset]
        for eta in ETAS[e.dataset]:
            for lam in LAMBDAS:
                pot = potential_for(ds, lam)
                W0 = init_unit_ball10(np.random.default_rng(CONFIRM_SEED + 1), R)
                rev = arm(ds, pot, e, eta, 0, 0.0, W0, CONFIRM_SEED)
                arms = {s: arm(ds, pot, e, eta, 1, s, W0, CONFIRM_SEED) for s in S_VALS}
                base = dict(experiment=e.key, eta=eta, lam=lam, delta=pot.delta)
                rows.append(dict(base, s="reversible",
                                 train=float(rev["tr"][-1].mean()), test=float(rev["te"][-1].mean()),
                                 d_train=0.0, t_train=0.0, d_test=0.0, t_test=0.0,
                                 U=float(rev["U"].mean()), a=float(rev["a"].mean()),
                                 penalty=float(rev["pen"].mean()),
                                 near_kink=float(rev["near_kink"].mean()), proj=rev["proj"]))
                for s in S_VALS:
                    a_ = arms[s]
                    dtr = a_["tr"][-1] - rev["tr"][-1]; dte = a_["te"][-1] - rev["te"][-1]
                    se1 = dtr.std(ddof=1) / np.sqrt(R) + 1e-300
                    se2 = dte.std(ddof=1) / np.sqrt(R) + 1e-300
                    rows.append(dict(base, s=f"{s:g}",
                                     train=float(a_["tr"][-1].mean()), test=float(a_["te"][-1].mean()),
                                     d_train=float(dtr.mean()), t_train=float(dtr.mean() / se1),
                                     d_test=float(dte.mean()), t_test=float(dte.mean() / se2),
                                     U=float(a_["U"].mean()), a=float(a_["a"].mean()),
                                     penalty=float(a_["pen"].mean()),
                                     near_kink=float(a_["near_kink"].mean()), proj=a_["proj"]))
                figure_lambda(e, ds, pot, eta, rev, arms, OUT)
            print(f"  [{e.key}] eta = {eta:g} done ({time.perf_counter()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "lambda_lasso.csv"), index=False)
    pd.set_option("display.width", 230)
    for e in EXPS:
        for eta in ETAS[e.dataset]:
            sub = df[(df.experiment == e.key) & (df.eta == eta)]
            print(f"\n=== {e.key}, eta = {eta:g}, R = {R} ===")
            print(sub[["lam", "delta", "s", "train", "test", "d_train", "t_train",
                       "d_test", "t_test", "a", "near_kink", "proj"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\ntotal {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
