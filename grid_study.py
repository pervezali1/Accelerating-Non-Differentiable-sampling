"""How does accuracy change with the step size eta and the block strength s?

Exact gradient, LASSO anchor (see exact_anchored.py). The FULL grid is run and reported --
nothing is selected, so there is no selection bias to correct for.

  eta in {1e-4, 3e-5, 1e-5, 3e-6, 1e-6}
  s   in {0, 0.25, 1, 2, 5}          (s = 0 is a control: it must reproduce the reversible arm)

For each (experiment, eta) a figure is written in the SAME layout as before -- training accuracy
left, test accuracy right, common [0, 1] axis, mean +- 1 SD bands, zoom inset -- with the
reversible arm in blue and the non-reversible arm at each s in a green ramp.
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

from exact_anchored import (D10, Exp10, LASSO_FRAC, CONFIRM_SEED, accuracy10, build_magic_dataset,
                            build_titanic_dataset, geom_for, init_unit_ball10, make_potential,
                            run_exact)
from nral import mean_sd, _wrap

ETAS = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
SS = [0.0, 0.25, 1.0, 2.0, 5.0]
R = 100
REV_COLOR = "#1f5fbf"
NREV_COLORS = {0.0: "#bdbdbd", 0.25: "#a6d96a", 1.0: "#66bd63", 2.0: "#1a9850", 5.0: "#00602f"}
EXPS = (Exp10("magic_ball",   "magic",   "ball", 1000),
        Exp10("magic_lp",     "magic",   "lp",   1000, eps=0.20),
        Exp10("titanic_ball", "titanic", "ball", 1500),
        Exp10("titanic_lp",   "titanic", "lp",   2000, eps=0.18))
OUT, RES = "figures/grid", "results/grid"


def arm(ds, pot, e, eta, alpha, s, W0, seed):
    r = run_exact(pot, geom_for(e), alpha=alpha, scales=(s, s, s), eta=eta,
                  n_iter=e.n_iter, W0=W0, seed=seed)
    W = r["betas"]
    return dict(ck=r["checkpoints"],
                tr=accuracy10(ds.X_train, ds.y_train, W),
                te=accuracy10(ds.X_test, ds.y_test, W),
                U=pot.U(W[-1]), proj=r["projection_rate"], nfin=r["n_nonfinite"])


def figure_eta(e, ds, pot, eta, rev, nrev: dict, outdir: str):
    """Same layout as the previous figures; the only change is the extra green curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, split in zip(axes, ("tr", "te")):
        x = np.asarray(rev["ck"], dtype=float)
        series = [("Reversible anchored Langevin", rev[split], REV_COLOR, "-", 2.0)]
        for s in sorted(nrev):
            series.append((f"Non-reversible, s = {s:g}", nrev[s][split], NREV_COLORS[s], "-", 1.7))
        mu0, sd0 = mean_sd(series[0][1], axis=1)
        ax.fill_between(x, np.clip(mu0 - sd0, 0, 1), np.clip(mu0 + sd0, 0, 1),
                        color=REV_COLOR, alpha=0.14, linewidth=0)
        muN, sdN = mean_sd(series[-1][1], axis=1)
        ax.fill_between(x, np.clip(muN - sdN, 0, 1), np.clip(muN + sdN, 0, 1),
                        color=series[-1][2], alpha=0.14, linewidth=0)
        for lbl, a, c, ls, lw in series:
            ax.plot(x, mean_sd(a, axis=1)[0], color=c, linestyle=ls, linewidth=lw, label=lbl)
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        gname = "ball" if e.geometry == "ball" else f"smoothed $\\ell_p$ ($p={e.p:g}$)"
        ax.set_title(f"{e.dataset.upper()}, {gname}, $\\eta$ = {eta:g} -- "
                     f"{'training' if split == 'tr' else 'test'} accuracy", fontsize=9.5)
        i0 = int(0.4 * len(x))
        ins = ax.inset_axes([0.50, 0.12, 0.46, 0.40])
        lo, hi = [], []
        for lbl, a, c, ls, lw in series:
            mu, sd = mean_sd(a, axis=1)
            se = sd / np.sqrt(a.shape[1])
            ins.fill_between(x[i0:], mu[i0:] - se[i0:], mu[i0:] + se[i0:], color=c, alpha=0.20,
                             linewidth=0)
            ins.plot(x[i0:], mu[i0:], color=c, linewidth=1.3)
            lo.append((mu[i0:] - se[i0:]).min()); hi.append((mu[i0:] + se[i0:]).max())
        pad = 0.12 * (max(hi) - min(lo) + 1e-9)
        ins.set_ylim(min(lo) - pad, max(hi) + pad); ins.set_xlim(x[i0], x[-1])
        ins.tick_params(labelsize=6.5); ins.grid(True, alpha=0.25, linewidth=0.5)
        ins.set_title("zoom (mean $\\pm$ 1 s.e.)", fontsize=6.5, pad=2)
    axes[0].legend(loc="upper left", fontsize=7.2, framealpha=0.92)
    gtxt = ("ball ||w||^2 <= 2 (radius sqrt(2))" if e.geometry == "ball" else
            f"smoothed l_p  g(w) = sum_i (w_i^2+eps^2)^(p/2) <= Lambda, p = {e.p:g}, "
            f"eps = {e.eps:g}, Lambda = {e.Lam:g}")
    cap = (f"{e.dataset.upper()} -- {gtxt}.  d = 10 (intercept w_0 + 9 features).  "
           f"U = sum softplus - y x.w + w_0^2/(2 sigma^2) + lambda sum_{{j>=1}} |w_j|;  U_0 "
           f"replaces |w_j| by sqrt(w_j^2 + delta^2);  a = exp(U - U_0).  EXACT gradient.  "
           f"sigma = {pot.sigma:g}, lambda_lasso = {pot.lam:.3g} = {LASSO_FRAC:g} n_train, "
           f"delta = {pot.delta:.3g} (a in [{pot.a_lower_bound:.2f}, 1]).  Step eta = {eta:g}.  "
           f"R = {R} independent replicates on one fixed stratified 80/20 split "
           f"(n_train = {ds.n_train}, n_test = {ds.n_test}).  Bands: mean +- 1 SD (ddof = 1) "
           f"for the reversible arm and for s = 5 -- repeat-run variability at a fixed split, "
           f"not a confidence or credible interval.  The whole eta x s grid is reported in "
           f"results/grid/grid.csv; nothing here was selected.")
    fig.suptitle(f"{e.key}: accuracy vs block strength at $\\eta$ = {eta:g}", fontsize=11, y=0.995)
    fig.text(0.5, -0.09, _wrap(cap), ha="center", va="top", fontsize=7.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    tag = f"{e.key}_eta{eta:g}".replace("-", "m").replace(".", "p")
    png = os.path.join(outdir, f"{tag}.png"); pdf = os.path.join(outdir, f"{tag}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def main() -> int:
    os.makedirs(OUT, exist_ok=True); os.makedirs(RES, exist_ok=True)
    datasets = {"magic": build_magic_dataset(), "titanic": build_titanic_dataset()}
    pots = {k: make_potential(v) for k, v in datasets.items()}
    rows, t0 = [], time.perf_counter()
    for e in EXPS:
        ds, pot = datasets[e.dataset], pots[e.dataset]
        for eta in ETAS:
            W0 = init_unit_ball10(np.random.default_rng(CONFIRM_SEED + 1), R)
            rev = arm(ds, pot, e, eta, 0, 0.0, W0, CONFIRM_SEED)
            nrev = {s: arm(ds, pot, e, eta, 1, s, W0, CONFIRM_SEED) for s in SS}
            rows.append(dict(experiment=e.key, eta=eta, s="reversible",
                             train=float(rev["tr"][-1].mean()), train_sd=float(rev["tr"][-1].std(ddof=1)),
                             test=float(rev["te"][-1].mean()), test_sd=float(rev["te"][-1].std(ddof=1)),
                             d_train=0.0, t_train=0.0, d_test=0.0, t_test=0.0,
                             U=float(rev["U"].mean()), proj=rev["proj"], nonfinite=rev["nfin"]))
            for s in SS:
                a = nrev[s]
                dtr = a["tr"][-1] - rev["tr"][-1]; dte = a["te"][-1] - rev["te"][-1]
                se_tr = dtr.std(ddof=1) / np.sqrt(R) + 1e-300
                se_te = dte.std(ddof=1) / np.sqrt(R) + 1e-300
                rows.append(dict(experiment=e.key, eta=eta, s=f"{s:g}",
                                 train=float(a["tr"][-1].mean()), train_sd=float(a["tr"][-1].std(ddof=1)),
                                 test=float(a["te"][-1].mean()), test_sd=float(a["te"][-1].std(ddof=1)),
                                 d_train=float(dtr.mean()), t_train=float(dtr.mean() / se_tr),
                                 d_test=float(dte.mean()), t_test=float(dte.mean() / se_te),
                                 U=float(a["U"].mean()), proj=a["proj"], nonfinite=a["nfin"]))
            figure_eta(e, ds, pot, eta, rev, {s: nrev[s] for s in SS if s > 0}, OUT)
            curves = dict(checkpoints=rev["ck"], R=R, eta=eta)
            for nm, a in [("rev", rev)] + [(f"s{s:g}", nrev[s]) for s in SS]:
                for sp in ("tr", "te"):
                    mu, sd = mean_sd(a[sp], axis=1)
                    curves[f"{nm}_{sp}_mean"] = mu
                    curves[f"{nm}_{sp}_sd"] = sd
            np.savez_compressed(
                os.path.join(RES, f"curves_{e.key}_eta{eta:g}".replace("-", "m").replace(".", "p")
                             + ".npz"), **curves)
            print(f"  [{e.key}] eta = {eta:g} done ({time.perf_counter()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "grid.csv"), index=False)

    pd.set_option("display.width", 220)
    for e in EXPS:
        sub = df[df.experiment == e.key]
        print(f"\n================ {e.key} : final TEST accuracy (rows = eta, cols = s) ============")
        print(sub.pivot(index="eta", columns="s", values="test")
              .to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"---------------- {e.key} : paired t on TRAINING (non-rev - reversible) --------")
        print(sub[sub.s != "reversible"].pivot(index="eta", columns="s", values="t_train")
              .to_string(float_format=lambda v: f"{v:+.2f}"))
    print(f"\ntotal {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
