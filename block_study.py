"""Different block strengths s = (s_1, s_2, s_3) for the three J blocks.

J is blockdiag over the coordinate triples I_1 = w_1..w_3, I_2 = w_4..w_6, I_3 = w_7..w_9, and
each block carries its own strength. The blocks are NOT interchangeable:

    MAGIC   I_1 = PC1..PC3,  I_2 = PC4..PC6,  I_3 = PC7..PC9      (decreasing variance)
    Titanic I_1 = Age, SibSp, Parch;  I_2 = Fare, is_female, Pclass_2;
            I_3 = Pclass_3, Embarked_Q, Embarked_S

and their gradients differ by up to 8x (MAGIC: 5762 / 2292 / 732). The absolute rotational drift
contributed by block l is eta * s_l * r_l * ||grad_{I_l} U_0||, where r_l is the per-unit-s
rotation ratio measured at the initialisation. That gives a principled anisotropic rule rather
than a blind search:

    BALANCED:  s_l proportional to 1 / (r_l * ||grad_{I_l} U_0||)

so every block contributes the same rotational drift. It is tested against isotropic strengths,
the three single-block strengths (which decompose the effect block by block), the REVERSED
balanced triple as a control, and the hand-picked (2, 7, 2) used earlier.

Two step sizes per dataset: one where the chain is still mid-transient (where s mattered in the
eta x s grid) and one where it has converged. The whole set is reported; nothing is selected.
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

from exact_anchored import (BLOCKS10, Exp10, LASSO_FRAC, CONFIRM_SEED, accuracy10,
                            build_magic_dataset, build_titanic_dataset, geom_for, init_unit_ball10,
                            J_axes10, make_potential, run_exact)
from nral import mean_sd, _wrap


def fmt3(t) -> str:
    return "(" + ", ".join(f"{v:g}" for v in t) + ")"

R = 100
EXPS = (Exp10("magic_ball",   "magic",   "ball", 1000),
        Exp10("magic_lp",     "magic",   "lp",   1000, eps=0.20),
        Exp10("titanic_ball", "titanic", "ball", 1500),
        Exp10("titanic_lp",   "titanic", "lp",   2000, eps=0.18))
ETAS = {"magic": [1e-6, 1e-5], "titanic": [1e-5, 3e-5]}   # transient, then converged
OUT, RES = "figures/blocks", "results/blocks"


def block_scales_balanced(ds, pot, e, target_mean: float, n_pts: int = 400) -> tuple:
    """s_l proportional to 1/(r_l ||grad_{I_l} U_0||), rescaled to the requested mean."""
    rng = np.random.default_rng(12345)
    W = init_unit_ball10(rng, n_pts)
    G = pot.grad_U0(W)
    ax = J_axes10(W, geom_for(e), (1.0, 1.0, 1.0))
    w = []
    for blk in BLOCKS10:
        sl = slice(blk[0], blk[-1] + 1)
        gn = np.linalg.norm(G[:, sl], axis=1)
        rot = np.linalg.norm(np.cross(ax[:, sl], G[:, sl]), axis=1)
        w.append(float(np.median(rot)))          # = r_l * ||grad_I|| already, in absolute units
    inv = 1.0 / np.asarray(w)
    return tuple(float(round(v, 3)) for v in inv / inv.mean() * target_mean)


def arm(ds, pot, e, eta, alpha, s, W0, seed):
    r = run_exact(pot, geom_for(e), alpha=alpha, scales=s, eta=eta, n_iter=e.n_iter,
                  W0=W0, seed=seed)
    W = r["betas"]
    return dict(ck=r["checkpoints"], tr=accuracy10(ds.X_train, ds.y_train, W),
                te=accuracy10(ds.X_test, ds.y_test, W), U=pot.U(W[-1]),
                proj=r["projection_rate"], nfin=r["n_nonfinite"])


def figure_blocks(e, ds, pot, eta, rev, arms: dict, order, colors, styles, tag, outdir, subtitle):
    """Unchanged layout: training left, test right, [0,1] axis, mean +- 1 SD, zoom inset."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, split in zip(axes, ("tr", "te")):
        x = np.asarray(rev["ck"], dtype=float)
        mu0, sd0 = mean_sd(rev[split], axis=1)
        ax.fill_between(x, np.clip(mu0 - sd0, 0, 1), np.clip(mu0 + sd0, 0, 1),
                        color="#1f5fbf", alpha=0.14, linewidth=0)
        ax.plot(x, mu0, color="#1f5fbf", linewidth=2.0, label="Reversible anchored Langevin")
        for k in order:
            mu, _ = mean_sd(arms[k][split], axis=1)
            ax.plot(x, mu, color=colors[k], linestyle=styles[k], linewidth=1.7,
                    label=k if k.startswith(("balanced","reversed")) else f"s = {k}")
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        gname = "ball" if e.geometry == "ball" else f"smoothed $\\ell_p$ ($p={e.p:g}$)"
        ax.set_title(f"{e.dataset.upper()}, {gname}, $\\eta$ = {eta:g} -- "
                     f"{'training' if split == 'tr' else 'test'} accuracy", fontsize=9.5)
        i0 = int(0.4 * len(x))
        ins = ax.inset_axes([0.50, 0.12, 0.46, 0.40])
        lo, hi = [], []
        for nm, a, c, ls in [("rev", rev, "#1f5fbf", "-")] + \
                            [(k, arms[k], colors[k], styles[k]) for k in order]:
            mu, sd = mean_sd(a[split], axis=1)
            se = sd / np.sqrt(a[split].shape[1])
            ins.fill_between(x[i0:], mu[i0:] - se[i0:], mu[i0:] + se[i0:], color=c, alpha=0.18,
                             linewidth=0)
            ins.plot(x[i0:], mu[i0:], color=c, linestyle=ls, linewidth=1.3)
            lo.append((mu[i0:] - se[i0:]).min()); hi.append((mu[i0:] + se[i0:]).max())
        pad = 0.12 * (max(hi) - min(lo) + 1e-9)
        ins.set_ylim(min(lo) - pad, max(hi) + pad); ins.set_xlim(x[i0], x[-1])
        ins.tick_params(labelsize=6.5); ins.grid(True, alpha=0.25, linewidth=0.5)
        ins.set_title("zoom (mean $\\pm$ 1 s.e.)", fontsize=6.5, pad=2)
    axes[0].legend(loc="upper left", fontsize=6.8, framealpha=0.92)
    gtxt = ("ball ||w||^2 <= 2 (radius sqrt(2))" if e.geometry == "ball" else
            f"smoothed l_p  g(w) = sum_i (w_i^2+eps^2)^(p/2) <= Lambda, p = {e.p:g}, "
            f"eps = {e.eps:g}, Lambda = {e.Lam:g}")
    blocks = ("I_1 = PC1..PC3, I_2 = PC4..PC6, I_3 = PC7..PC9" if e.dataset == "magic" else
              "I_1 = Age/SibSp/Parch, I_2 = Fare/is_female/Pclass_2, "
              "I_3 = Pclass_3/Embarked_Q/Embarked_S")
    cap = (f"{e.dataset.upper()} -- {gtxt}.  d = 10 (intercept w_0 + 9 features).  "
           f"U = sum softplus - y x.w + w_0^2/(2 sigma^2) + lambda sum_{{j>=1}} |w_j|;  U_0 "
           f"replaces |w_j| by sqrt(w_j^2 + delta^2);  a = exp(U - U_0).  EXACT gradient.  "
           f"sigma = {pot.sigma:g}, lambda_lasso = {pot.lam:.3g} = {LASSO_FRAC:g} n_train, "
           f"delta = {pot.delta:.3g} (a in [{pot.a_lower_bound:.2f}, 1]).  Step eta = {eta:g}.  "
           f"Per-block strengths s = (s_1, s_2, s_3) on {blocks}.  {subtitle}  "
           f"R = {R} independent replicates on one fixed stratified 80/20 split "
           f"(n_train = {ds.n_train}, n_test = {ds.n_test}).  Band: mean +- 1 SD (ddof = 1) for "
           f"the reversible arm -- repeat-run variability at a fixed split, not a confidence or "
           f"credible interval.  All triples are reported in results/blocks/blocks.csv; nothing "
           f"here was selected.")
    fig.suptitle(f"{e.key}: per-block strengths at $\\eta$ = {eta:g} ({tag})", fontsize=11, y=0.995)
    fig.text(0.5, -0.09, _wrap(cap), ha="center", va="top", fontsize=7.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    nm = f"{e.key}_eta{eta:g}_{tag}".replace("-", "m").replace(".", "p")
    png = os.path.join(outdir, f"{nm}.png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{nm}.pdf"), bbox_inches="tight")
    plt.close(fig)
    return png


def main() -> int:
    os.makedirs(OUT, exist_ok=True); os.makedirs(RES, exist_ok=True)
    datasets = {"magic": build_magic_dataset(), "titanic": build_titanic_dataset()}
    pots = {k: make_potential(v) for k, v in datasets.items()}
    rows, meta, t0 = [], {}, time.perf_counter()

    for e in EXPS:
        ds, pot = datasets[e.dataset], pots[e.dataset]
        bal1 = block_scales_balanced(ds, pot, e, 1.0)
        bal5 = block_scales_balanced(ds, pot, e, 5.0)
        rev5 = tuple(reversed(bal5))
        meta[e.key] = dict(balanced_mean1=list(bal1), balanced_mean5=list(bal5),
                           reversed_mean5=list(rev5))
        TRIPLES = {"(1,1,1)": (1., 1., 1.), "(5,5,5)": (5., 5., 5.),
                   "(5,0,0)": (5., 0., 0.), "(0,5,0)": (0., 5., 0.), "(0,0,5)": (0., 0., 5.),
                   f"balanced {fmt3(bal1)}": bal1, f"balanced {fmt3(bal5)}": bal5,
                   f"reversed {fmt3(rev5)}": rev5,
                   "(2,7,2)": (2., 7., 2.)}
        print(f"\n[{e.key}] balanced(mean 1) = {fmt3(bal1)}   balanced(mean 5) = {fmt3(bal5)}"
              f"   reversed = {fmt3(rev5)}")
        for eta in ETAS[e.dataset]:
            W0 = init_unit_ball10(np.random.default_rng(CONFIRM_SEED + 1), R)
            rev = arm(ds, pot, e, eta, 0, (0., 0., 0.), W0, CONFIRM_SEED)
            arms = {k: arm(ds, pot, e, eta, 1, v, W0, CONFIRM_SEED) for k, v in TRIPLES.items()}
            rows.append(dict(experiment=e.key, eta=eta, s="reversible",
                             train=float(rev["tr"][-1].mean()), test=float(rev["te"][-1].mean()),
                             d_train=0.0, t_train=0.0, d_test=0.0, t_test=0.0,
                             U=float(rev["U"].mean()), proj=rev["proj"]))
            for k, a in arms.items():
                dtr = a["tr"][-1] - rev["tr"][-1]; dte = a["te"][-1] - rev["te"][-1]
                se1 = dtr.std(ddof=1) / np.sqrt(R) + 1e-300
                se2 = dte.std(ddof=1) / np.sqrt(R) + 1e-300
                rows.append(dict(experiment=e.key, eta=eta, s=k,
                                 train=float(a["tr"][-1].mean()), test=float(a["te"][-1].mean()),
                                 d_train=float(dtr.mean()), t_train=float(dtr.mean() / se1),
                                 d_test=float(dte.mean()), t_test=float(dte.mean() / se2),
                                 U=float(a["U"].mean()), proj=a["proj"]))
            mix = ["(1,1,1)", "(5,5,5)", f"balanced {fmt3(bal5)}",
                   f"reversed {fmt3(rev5)}", "(2,7,2)"]
            figure_blocks(e, ds, pot, eta, rev, arms, mix,
                          dict(zip(mix, ["#a6d96a", "#00602f", "#d95f02", "#7570b3", "#e7298a"])),
                          dict(zip(mix, ["-", "-", "-", "--", ":"])), "mix", OUT,
                          "Isotropic vs gradient-balanced vs reversed-balanced vs hand-picked.")
            blk = ["(5,0,0)", "(0,5,0)", "(0,0,5)", "(5,5,5)"]
            figure_blocks(e, ds, pot, eta, rev, arms, blk,
                          dict(zip(blk, ["#e6ab02", "#1b9e77", "#7570b3", "#00602f"])),
                          dict(zip(blk, ["-", "-", "-", "--"])), "perblock", OUT,
                          "One block rotated at a time, decomposing the effect.")
            print(f"  [{e.key}] eta = {eta:g} done ({time.perf_counter()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "blocks.csv"), index=False)
    with open(os.path.join(RES, "blocks_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    pd.set_option("display.width", 220)
    for e in EXPS:
        for eta in ETAS[e.dataset]:
            sub = df[(df.experiment == e.key) & (df.eta == eta)]
            print(f"\n=== {e.key}, eta = {eta:g}, R = {R} ===")
            print(sub[["s", "train", "test", "d_train", "t_train", "d_test", "t_test", "U", "proj"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\ntotal {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
