"""Every figure of exact_anchored_langevin.ipynb, with all runs stopped at iteration 600.

Nothing else changes: same potentials, same geometries, same block strengths, same seeds, same
replicate counts. Because run_chain draws its Gaussian increments sequentially and W0 depends only
on the seed, a 600-step run is bit-for-bit the first 600 steps of the notebook's longer run -- this
is a TRUNCATION of the published experiment, not a re-tuned one.

The burn-in for the time-average estimator stays at the declared 50%, which is now iterations
300-600 rather than 1000-2000.
"""
from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import exact_nb as X
from nral import _wrap, build_magic_dataset, build_titanic_dataset, mean_sd

N_ITER = 600
Y_LO = 0.40                 # accuracy axis floor, requested
OUT = os.path.join(os.getcwd(), "figures", "exact_nb_600")
RES = os.path.join(os.getcwd(), "results", "exact_nb_600")
TRUNC = (f"ALL RUNS STOPPED AT ITERATION {N_ITER} -- a truncation of the notebook's longer runs "
         f"(identical W0 and Gaussian stream), not a separate experiment.")


def accuracy_fig(spec, arms, pot, ds, scales, eta, R, tag=""):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, key, name in zip(axes, ("tr", "te"), ("training", "test")):
        for t, col, lab in (("rev", X.REV, "Reversible anchored Langevin"),
                            ("nrev", X.NREV, "Non-reversible anchored Langevin")):
            x, A = arms[t]["x"], arms[t][key]
            mu, sd = mean_sd(A, axis=1)
            ax.fill_between(x, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=col,
                            alpha=0.15, linewidth=0)      # may run below the axis floor
            ax.plot(x, mu, color=col, linewidth=2.0, label=lab)
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(Y_LO, 1.0)
        ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.9); ax.set_axisbelow(True)
        gname = "ball" if spec["geometry"] == "ball" else "smoothed $\\ell_p$ ($p=2.4$)"
        ax.set_title(f"{spec['dataset'].upper()}, {gname} — {name} accuracy", color=X.INK,
                     loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        i0 = int(0.4 * len(x))
        # On a [0.4, 1] axis the band fills the lower half, so park the inset ABOVE it: take the
        # highest band edge under the inset's own x-span and start just clear of that.
        j0 = int(0.46 * len(x))
        ceil = max(float(sum(mean_sd(arms[t][key], axis=1))[j0:].max()) for t in ("rev", "nrev"))
        bot = min(0.70, (ceil - Y_LO) / (1.0 - Y_LO) + 0.04)
        ins = ax.inset_axes([0.46, bot, 0.50, 0.95 - bot])
        lo, hi = [], []
        for t, col in (("rev", X.REV), ("nrev", X.NREV)):
            mu, sd = mean_sd(arms[t][key], axis=1)
            se = sd / np.sqrt(arms[t][key].shape[1])
            ins.fill_between(x[i0:], (mu - se)[i0:], (mu + se)[i0:], color=col, alpha=0.22,
                             linewidth=0)
            ins.plot(x[i0:], mu[i0:], color=col, linewidth=1.5)
            lo.append((mu - se)[i0:].min()); hi.append((mu + se)[i0:].max())
        pad = 0.18 * (max(hi) - min(lo) + 1e-9)
        ins.set_ylim(min(lo) - pad, max(hi) + pad); ins.set_xlim(x[i0], x[-1])
        ins.tick_params(labelsize=6.5); ins.grid(True, alpha=0.9)
        ins.set_title("zoom (mean $\\pm$ 1 s.e.)", fontsize=6.8, pad=2, color=X.INK2)
        for sp in ("top", "right"):
            ins.spines[sp].set_visible(False)
    # one shared legend beneath the panels: on a [0.4, 1] axis there is no in-panel corner
    # that is free of both the band and the zoom inset
    _h, _l = axes[0].get_legend_handles_labels()
    fig.legend(_h, _l, loc="lower center", bbox_to_anchor=(0.5, -0.055), ncol=2, fontsize=9.0)
    d_tr = X.paired(arms["nrev"]["tr"][-1], arms["rev"]["tr"][-1])
    d_te = X.paired(arms["nrev"]["te"][-1], arms["rev"]["te"][-1])
    gtxt = ("ball ||w||^2 <= 2 (radius sqrt(2))" if spec["geometry"] == "ball" else
            f"smoothed l_p, g(w) = sum_i (w_i^2 + eps^2)^(p/2) <= Lambda, p = 2.4, "
            f"eps = {spec['eps']:g}, Lambda = 4")
    fig.suptitle(f"{spec['key']}{tag}: exact gradient, LASSO anchor "
                 f"(first {N_ITER} iterations)", fontsize=12.0, y=1.005, x=0.02, ha="left",
                 color=X.INK)
    fig.text(0.5, -0.105, _wrap(
        f"{TRUNC} {spec['dataset'].upper()} -- {gtxt}. d = 10 (intercept w_0 + nine features), "
        f"no mini-batching: the EXACT gradient is used at every step. sigma = {pot.sigma:g}, "
        f"lambda = {pot.lam:.4g}, delta = {pot.delta:.4g}, so a in [{pot.a_lower:.2f}, 1]. Block "
        f"strengths s = ({', '.join(f'{v:g}' for v in scales)}) on w_1..w_9 only. Step "
        f"eta = {eta:g}. R = {R} coupled replicates (same W0, same Gaussian stream) on one fixed "
        f"stratified 80/20 split (n_train = {ds.n_train}, n_test = {ds.n_test}). Paired "
        f"difference AT ITERATION {N_ITER}, non-reversible minus reversible: training "
        f"{d_tr[0]:+.4f} (t = {d_tr[2]:+.2f}), test {d_te[0]:+.4f} (t = {d_te[2]:+.2f}). Bands: "
        f"mean +- 1 SD (ddof = 1) across replicates -- repeat-run variability at a fixed split, "
        f"not a confidence or credible interval. The ACCURACY AXIS IS [{Y_LO:g}, 1], so the lower "
        f"band edge runs off the bottom of the panel during the first few hundred iterations: "
        f"about a quarter of replicates start below {Y_LO:g} and the lowest state visited is near "
        f"0.27 (see figures/axis_explainer.png).", 126),
        ha="center", va="top", fontsize=7.0, color=X.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    name = f"{spec['key']}{tag}_600"
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    return d_tr, d_te


def sweep_fig(table, key):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharex=True)
    sub = table[table["experiment"] == key].sort_values("s")
    for ax, which, name in zip(axes, ("train", "test"), ("training", "test")):
        d = sub[f"d_{which}"].to_numpy() * 100.0
        se = sub[f"se_{which}"].to_numpy() * 100.0
        s = sub["s"].to_numpy(); xs = np.arange(len(s))
        ax.axhline(0.0, color=X.REV, lw=1.4, zorder=2,
                   label="Reversible anchored Langevin (reference)")
        ax.errorbar(xs, d, yerr=2.0 * se, fmt="s", ms=6.5, color=X.NREV, ecolor=X.NREV,
                    elinewidth=1.4, capsize=4, zorder=3, mfc=X.NREV, mec="white", mew=0.9,
                    label="Non-reversible, $\\pm$ 2 s.e. (paired)")
        for xi, di, sei in zip(xs, d, se):
            tip = di + (2.0 * sei if di >= 0 else -2.0 * sei)
            ax.annotate(f"{di:+.3f}", (xi, tip), textcoords="offset points",
                        xytext=(0, 7 if di >= 0 else -14), ha="center", fontsize=7.2,
                        color=X.INK2)
        lo = float(np.min(d - 2.0 * se)); hi = float(np.max(d + 2.0 * se))
        pad = 0.30 * (hi - lo + 1e-12)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(xs); ax.set_xticklabels([f"{v:g}" for v in s])
        ax.set_xlabel("block strength $s$ (all three blocks)")
        ax.set_ylabel(f"{name} accuracy difference (percentage points)")
        ax.set_title(f"{name} accuracy", fontsize=10.5, color=X.INK, pad=6, loc="left")
        ax.grid(True, axis="y"); ax.set_axisbelow(True); ax.margins(x=0.14)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(loc="best", fontsize=8.2)
    fig.suptitle(f"{key}: block-strength sweep at $\\eta = 3\\times10^{{-5}}$, "
                 f"iteration {N_ITER}", fontsize=12.0, y=1.02, x=0.02, ha="left", color=X.INK)
    fig.text(0.5, -0.09, _wrap(
        f"{TRUNC} Paired difference AT ITERATION {N_ITER}, non-reversible minus reversible, in "
        f"percentage points, over R = {X.R_ETA} coupled replicates sharing one W0 and one "
        f"Gaussian stream. eta = 3e-5 was SPECIFIED, not selected; s was swept, so all five "
        f"strengths are shown. The s = 0 point is the control -- with J = 0 the update is the "
        f"reversible update, so the difference is exactly 0.000 and the error bar has zero width. "
        f"Bars are +- 2 paired standard errors; with four live strengths per panel, |t| > 2.5 is "
        f"the honest bar. Seed {X.SEED_CONF} throughout.", 124),
        ha="center", va="top", fontsize=7.0, color=X.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, f"{key}_eta3e-5_sweep_600.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{key}_eta3e-5_sweep_600.pdf"), bbox_inches="tight")
    plt.close(fig)


def timeavg_fig(arms, R):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    rows = []
    for ax, (which, col), name in zip(axes, (("train", "tr"), ("test", "te")),
                                      ("training", "test")):
        rev_a, nrev_a = X.time_average(arms, col)
        lo = min(rev_a.min(), nrev_a.min()); hi = max(rev_a.max(), nrev_a.max())
        bins = np.linspace(lo, hi, 34) * 100
        ax.hist(rev_a * 100, bins=bins, color=X.REV, alpha=0.55, label="Reversible")
        ax.hist(nrev_a * 100, bins=bins, color=X.NREV, alpha=0.55, label="Non-reversible")
        for v, c, ls in ((rev_a.mean(), X.REV, "-"), (nrev_a.mean(), X.NREV, "--")):
            ax.axvline(v * 100, color=c, lw=1.6, ls=ls, zorder=4)
        da = X.paired(nrev_a, rev_a)
        df = X.paired(arms["nrev"][col][-1], arms["rev"][col][-1])
        rows.append(dict(split=which, R=R, avg_rev=float(rev_a.mean()),
                         avg_nrev=float(nrev_a.mean()), avg_d_pts=da[0] * 100,
                         avg_se_pts=da[1] * 100, avg_t=da[2], avg_win=da[3],
                         final_d_pts=df[0] * 100, final_t=df[2]))
        ax.set_xlabel(f"mean {name} accuracy (%) over iterations {N_ITER // 2}-{N_ITER}")
        ax.set_ylabel("replicates")
        ax.set_title(f"{name}: paired difference {da[0]*100:+.3f} pts "
                     f"(s.e. {da[1]*100:.3f}, t = {da[2]:+.2f})",
                     fontsize=9.8, color=X.INK, pad=6, loc="left")
        ax.grid(True, axis="y"); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_ylim(0.0, ax.get_ylim()[1] * 1.45)
        ins = ax.inset_axes((0.60, 0.56, 0.38, 0.38), zorder=6)
        ins.set_facecolor("white"); ins.patch.set_alpha(1.0)
        dd = (nrev_a - rev_a) * 100.0
        ins.hist(dd, bins=30, color=X.INK3)
        ins.axvline(0.0, color=X.REV, lw=1.6)
        ins.axvline(dd.mean(), color=X.NREV, lw=1.6, ls="--")
        ins.set_title("paired difference (pts)", fontsize=6.6, pad=2.5, color=X.INK2)
        ins.tick_params(labelsize=6.2); ins.set_yticks([]); ins.grid(False)
        for sp in ("top", "right", "left"):
            ins.spines[sp].set_visible(False)
        ins.spines["bottom"].set_color(X.GRIDC)
    axes[0].legend(loc="upper left", fontsize=8.4, bbox_to_anchor=(0.0, 1.0))
    fig.suptitle(f"titanic_lp, $\\eta = 3\\times10^{{-5}}$, $s = (2,2,2)$: the post-burn-in "
                 f"estimator over {N_ITER} iterations", fontsize=12.0, y=1.02, x=0.02, ha="left",
                 color=X.INK)
    fig.text(0.5, -0.07, _wrap(
        f"{TRUNC} R = {R} coupled replicates, seed {X.SEED_HIGH}. First half of each "
        f"{N_ITER}-step run discarded as burn-in (the declared 50%, so iterations "
        f"{N_ITER // 2}-{N_ITER}). Each replicate contributes one number per panel: its accuracy "
        f"averaged over the checkpointed post-burn-in states. The two arms share W0 and the "
        f"Gaussian stream, so the difference is paired and the histograms overlap far more than "
        f"the paired s.e. suggests -- the spread here is across replicates, the s.e. is across "
        f"PAIRS. The vertical lines are the arm means. At {N_ITER} iterations the chains are much "
        f"further from stationarity than in the notebook's 2000-step run, so this measures the "
        f"transient, not the target.", 124),
        ha="center", va="top", fontsize=7.0, color=X.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "titanic_lp_eta3e-5_timeaverage_600.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "titanic_lp_eta3e-5_timeaverage_600.pdf"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> int:
    os.makedirs(OUT, exist_ok=True); os.makedirs(RES, exist_ok=True)
    ds = {"magic": build_magic_dataset(), "titanic": build_titanic_dataset()}
    pots = {k: X.make_potential(v) for k, v in ds.items()}
    t0 = time.perf_counter()
    print(f"all runs truncated at {N_ITER} iterations\n")

    rows = []
    for spec in X.EXPERIMENTS:
        d, p = ds[spec["dataset"]], pots[spec["dataset"]]
        geom = X.make_geom(spec["geometry"], eps=spec["eps"])
        arms = X.run_pair(d, p, geom, eta=spec["eta"], scales=(5., 5., 5.), n_iter=N_ITER,
                          R=X.R_MAIN, seed=X.SEED_MAIN)
        dtr, dte = accuracy_fig(spec, arms, p, d, (5., 5., 5.), spec["eta"], X.R_MAIN)
        rows.append(dict(experiment=spec["key"], eta=spec["eta"], s=5.0, R=X.R_MAIN,
                         n_iter=N_ITER,
                         rev_train=float(arms["rev"]["tr"][-1].mean()),
                         nrev_train=float(arms["nrev"]["tr"][-1].mean()),
                         rev_test=float(arms["rev"]["te"][-1].mean()),
                         nrev_test=float(arms["nrev"]["te"][-1].mean()),
                         d_train=dtr[0], t_train=dtr[2], d_test=dte[0], t_test=dte[2],
                         proj_nrev=arms["nrev"]["proj"]))
        print(f"[{spec['key']}] eta={spec['eta']:g}  train {dtr[0]:+.4f} (t={dtr[2]:+.2f})   "
              f"test {dte[0]:+.4f} (t={dte[2]:+.2f})   [{time.perf_counter()-t0:.0f}s]")
    main_tbl = pd.DataFrame(rows)
    main_tbl.to_csv(os.path.join(RES, "main_experiments_600.csv"), index=False)
    print()
    print(main_tbl[["experiment", "eta", "rev_test", "nrev_test", "d_test", "t_test",
                    "proj_nrev"]].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print()

    rows3, keep = [], {}
    for spec in X.ETA3_EXPS:
        d, p = ds[spec["dataset"]], pots[spec["dataset"]]
        geom = X.make_geom(spec["geometry"], eps=spec["eps"])
        for s in X.S_GRID3:
            arms = X.run_pair(d, p, geom, eta=X.ETA3, scales=(s, s, s), n_iter=N_ITER,
                              R=X.R_ETA, seed=X.SEED_CONF)
            dtr = X.paired(arms["nrev"]["tr"][-1], arms["rev"]["tr"][-1])
            dte = X.paired(arms["nrev"]["te"][-1], arms["rev"]["te"][-1])
            rows3.append(dict(experiment=spec["key"], eta=X.ETA3, s=s, R=X.R_ETA, n_iter=N_ITER,
                              rev_train=float(arms["rev"]["tr"][-1].mean()),
                              nrev_train=float(arms["nrev"]["tr"][-1].mean()),
                              rev_test=float(arms["rev"]["te"][-1].mean()),
                              nrev_test=float(arms["nrev"]["te"][-1].mean()),
                              d_train=dtr[0], se_train=dtr[1], t_train=dtr[2],
                              d_test=dte[0], se_test=dte[1], t_test=dte[2],
                              proj_nrev=arms["nrev"]["proj"]))
            if s == 5.0:
                keep[spec["key"]] = (spec, arms, p, d)
        print(f"[{spec['key']}] sweep done  [{time.perf_counter()-t0:.0f}s]")
    eta3 = pd.DataFrame(rows3)
    eta3.to_csv(os.path.join(RES, "titanic_eta3e-5_600.csv"), index=False)
    z = eta3[eta3["s"] == 0.0]
    assert float(np.abs(z[["d_train", "d_test"]].to_numpy()).max()) == 0.0, "s = 0 control broken"
    print("\ns = 0 control: paired difference exactly 0 on both geometries and both splits.\n")
    for k in ("titanic_ball", "titanic_lp"):
        sweep_fig(eta3, k)
        spec, arms, p, d = keep[k]
        accuracy_fig(spec, arms, p, d, (5., 5., 5.), X.ETA3, X.R_ETA, tag="_eta3e-5")
        print(f"=== {k}, eta = {X.ETA3:g}, R = {X.R_ETA}, iteration {N_ITER} ===")
        print(eta3[eta3["experiment"] == k][["s", "rev_train", "nrev_train", "d_train", "t_train",
                                             "rev_test", "nrev_test", "d_test", "t_test",
                                             "proj_nrev"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        print()

    arms = X.run_pair(ds["titanic"], pots["titanic"],
                      X.make_geom(X.CONF_SPEC["geometry"], eps=X.CONF_SPEC["eps"]),
                      eta=X.ETA3, scales=X.CONF_S, n_iter=N_ITER, R=X.R_CONF, seed=X.SEED_HIGH)
    conf = timeavg_fig(arms, X.R_CONF)
    conf.to_csv(os.path.join(RES, "eta3e-5_confirmation_600.csv"), index=False)
    print(f"titanic_lp, eta = {X.ETA3:g}, s = {X.CONF_S}, R = {X.R_CONF}, seed {X.SEED_HIGH}, "
          f"iterations {N_ITER // 2}-{N_ITER}")
    print(conf.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print(f"\ntotal {time.perf_counter()-t0:.0f}s -> {OUT}")
    for f in sorted(os.listdir(OUT)):
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
