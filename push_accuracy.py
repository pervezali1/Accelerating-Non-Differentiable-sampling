"""Can the non-reversible arm beat the reversible one on ACCURACY by 0.1-0.2 percentage points?

Base: the 9d43048 setup (exact gradient, LASSO anchor, d = 10). Nothing about the sampler is
changed; only (eta, s) and the number of replicates.

Why this might work where earlier attempts did not. The grid and per-block studies already showed
gains of the RIGHT SIZE at converged step sizes -- gradient-balanced s gave +0.0014 (Titanic l_p)
and +0.0016 (Titanic ball) in test accuracy -- but at R = 100 those sit at t ~ 1.2, indistinguishable
from noise. The effect is not absent, it is unresolved. Raising R by 15x shrinks the paired SE by
~4x, which is the difference between t = 1.2 and t = 5.

Protocol:
  * SELECT on TRAINING accuracy only (mean over post-burn-in checkpoints, less noisy than a single
    final iterate), seed 3000, R = R_SEL. Candidate set is small and fixed in advance.
  * CONFIRM on an INDEPENDENT seed with a much larger R, reporting the paired test gain with its SE.
  * Report BOTH the matched-eta paired comparison (clean, common random numbers) AND the
    best-vs-best comparison against the reversible arm's own best eta.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results", "push_accuracy")
os.makedirs(OUT, exist_ok=True)

from exact_anchored import (BLOCKS10, Exp10, accuracy10, build_magic_dataset,  # noqa: E402
                            build_titanic_dataset, geom_for, init_unit_ball10, J_axes10,
                            make_potential, run_exact)

SEL_SEED, CONF_SEED = 3000, 4100
BURN = 0.4

EXPS = [
    (Exp10("titanic_ball", "titanic", "ball", 1500), [3e-5, 1e-4]),
    (Exp10("titanic_lp",   "titanic", "lp",   2000, eps=0.18), [3e-5, 1e-4]),
    (Exp10("magic_ball",   "magic",   "ball", 1000), [3e-6, 1e-5]),
    (Exp10("magic_lp",     "magic",   "lp",   1000, eps=0.20), [3e-6, 1e-5]),
]
R_SEL = {"titanic": 200, "magic": 150}
R_CONF = {"titanic": 1500, "magic": 600}
BVB_ETAS = {"titanic": [3e-5, 1e-4, 3e-4], "magic": [3e-6, 1e-5, 3e-5]}


def balanced(ds, pot, e, target_mean, n_pts=400):
    rng = np.random.default_rng(12345)
    W = init_unit_ball10(rng, n_pts)
    G = pot.grad_U0(W)
    ax = J_axes10(W, geom_for(e), (1.0, 1.0, 1.0))
    w = [float(np.median(np.linalg.norm(np.cross(ax[:, b[0]:b[-1] + 1], G[:, b[0]:b[-1] + 1]),
                                        axis=1))) for b in BLOCKS10]
    inv = 1.0 / np.asarray(w)
    return tuple(float(round(v, 3)) for v in inv / inv.mean() * target_mean)


def run(ds, pot, e, eta, alpha, s, W0, seed):
    r = run_exact(pot, geom_for(e), alpha=alpha, scales=s, eta=eta, n_iter=e.n_iter,
                  W0=W0, seed=seed, checkpoint_every=10)
    W = r["betas"]
    tr = accuracy10(ds.X_train, ds.y_train, W)
    te = accuracy10(ds.X_test, ds.y_test, W)
    b0 = int(BURN * tr.shape[0])
    return dict(tr_last=tr[-1], te_last=te[-1], tr_avg=tr[b0:].mean(axis=0),
                te_avg=te[b0:].mean(axis=0), proj=r["projection_rate"])


def paired(a, b, key):
    d = a[key] - b[key]
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) + 1e-300
    return float(d.mean()), float(se), float(d.mean() / se), float((d > 0).mean())


def main() -> int:
    datasets = {"magic": build_magic_dataset(), "titanic": build_titanic_dataset()}
    pots = {k: make_potential(v) for k, v in datasets.items()}
    rows, chosen, t0 = [], {}, time.perf_counter()

    for e, etas in EXPS:
        ds, pot = datasets[e.dataset], pots[e.dataset]
        cands = {"(0.25,0.25,0.25)": (.25, .25, .25), "(1,1,1)": (1., 1., 1.),
                 "(2,2,2)": (2., 2., 2.)}
        for m in (1.0, 2.0):
            b = balanced(ds, pot, e, m)
            cands[f"balanced({', '.join(f'{v:g}' for v in b)})"] = b
        Rs = R_SEL[e.dataset]
        W0 = init_unit_ball10(np.random.default_rng(SEL_SEED + 1), Rs)
        best = None
        for eta in etas:
            rev = run(ds, pot, e, eta, 0, (0., 0., 0.), W0, SEL_SEED)
            for lab, s in cands.items():
                nr = run(ds, pot, e, eta, 1, s, W0, SEL_SEED)
                d, se, t, w = paired(nr, rev, "tr_avg")          # TRAINING only
                rows.append(dict(stage="select", experiment=e.key, eta=eta, s=lab, R=Rs,
                                 d_train_avg=d, se=se, t=t, win=w))
                if best is None or d > best[0]:
                    best = (d, eta, lab, s)
        chosen[e.key] = dict(eta=best[1], s_label=best[2], s=list(best[3]), sel_gain=best[0])
        print(f"[{e.key}] selected on TRAINING only: eta={best[1]:g}, s={best[2]} "
              f"(train gain {best[0]:+.5f})   [{time.perf_counter()-t0:.0f}s]")

    print("\n" + "=" * 100)
    print(f"CONFIRMATION on independent seed {CONF_SEED}")
    print("=" * 100)
    conf = {}
    for e, _ in EXPS:
        ds, pot = datasets[e.dataset], pots[e.dataset]
        c = chosen[e.key]
        Rc = R_CONF[e.dataset]
        W0 = init_unit_ball10(np.random.default_rng(CONF_SEED + 1), Rc)
        rev = run(ds, pot, e, c["eta"], 0, (0., 0., 0.), W0, CONF_SEED)
        nrv = run(ds, pot, e, c["eta"], 1, tuple(c["s"]), W0, CONF_SEED)
        rec = dict(eta=c["eta"], s=c["s"], s_label=c["s_label"], R=Rc)
        for key in ("te_avg", "te_last", "tr_avg", "tr_last"):
            d, se, t, w = paired(nrv, rev, key)
            rec[key] = dict(rev=float(rev[key].mean()), nrev=float(nrv[key].mean()),
                            d=d, se=se, t=t, win=w)
        # best-vs-best: give the reversible arm its own best eta, chosen on TRAINING accuracy
        bb, bb_eta = None, None
        for eta in BVB_ETAS[e.dataset]:
            r2 = run(ds, pot, e, eta, 0, (0., 0., 0.), W0, CONF_SEED)
            v = float(r2["tr_avg"].mean())
            if bb is None or v > bb[0]:
                bb, bb_eta = (v, r2), eta
        rec["rev_best_eta"] = bb_eta
        rec["rev_best_te_avg"] = float(bb[1]["te_avg"].mean())
        rec["bvb_gap_te_avg"] = rec["te_avg"]["nrev"] - rec["rev_best_te_avg"]
        conf[e.key] = rec
        m = rec["te_avg"]
        print(f"\n[{e.key}]  eta={c['eta']:g}  s={c['s_label']}  R={Rc}")
        print(f"   matched-eta TEST (post-burn-in avg): rev={m['rev']:.5f}  nrev={m['nrev']:.5f}  "
              f"gain={m['d']*100:+.3f} pts  se={m['se']*100:.3f}  t={m['t']:+.2f}  win={m['win']:.0%}")
        l = rec["te_last"]
        print(f"   matched-eta TEST (final iterate):    rev={l['rev']:.5f}  nrev={l['nrev']:.5f}  "
              f"gain={l['d']*100:+.3f} pts  t={l['t']:+.2f}")
        print(f"   best-vs-best: reversible's own best eta={bb_eta:g} -> test {rec['rev_best_te_avg']:.5f}"
              f"   gap = {rec['bvb_gap_te_avg']*100:+.3f} pts")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "selection.csv"), index=False)
    with open(os.path.join(OUT, "confirmation.json"), "w") as fh:
        json.dump(dict(chosen=chosen, confirmation=conf, sel_seed=SEL_SEED,
                       conf_seed=CONF_SEED, burn=BURN), fh, indent=2, default=float)
    print(f"\ntotal {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------- figure
def figure(out_dir=None):
    """Accuracy curves for the one cell that clears best-vs-best: Titanic, smoothed l_p.

    Three curves, two entities. The reversible arm appears twice -- at the matched step size and
    at its OWN best step size -- distinguished by line style, not by colour, so colour still
    follows the entity.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nral import mean_sd, _wrap

    REV, NREV, INK, INK2, INK3, GRID = "#1f5fbf", "#1a9850", "#1f2328", "#57606a", "#8c959f", "#e4e6ea"
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.linewidth": 0.8, "grid.color": GRID,
                         "grid.linewidth": 0.7, "grid.linestyle": "-", "legend.frameon": False,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "font.size": 9})
    out_dir = out_dir or os.path.join(HERE, "figures", "push")
    os.makedirs(out_dir, exist_ok=True)

    e = Exp10("titanic_lp", "titanic", "lp", 2000, eps=0.18)
    ds = build_titanic_dataset(); pot = make_potential(ds)
    R, eta, s, eta_rev_best = 1500, 3e-5, (2., 2., 2.), 1e-4
    W0 = init_unit_ball10(np.random.default_rng(CONF_SEED + 1), R)

    series = []
    for lab, alpha, sc, et, col, ls in (
            ("Reversible, $\\eta$ = 3e-5 (matched)", 0, (0., 0., 0.), eta, REV, "-"),
            ("Reversible, $\\eta$ = 1e-4 (its own best)", 0, (0., 0., 0.), eta_rev_best, REV, "--"),
            ("Non-reversible, $\\eta$ = 3e-5, s = (2,2,2)", 1, s, eta, NREV, "-")):
        r = run_exact(pot, geom_for(e), alpha=alpha, scales=sc, eta=et, n_iter=e.n_iter,
                      W0=W0, seed=CONF_SEED, checkpoint_every=10)
        W = r["betas"]
        series.append(dict(lab=lab, col=col, ls=ls, x=np.asarray(r["checkpoints"], dtype=float),
                           tr=accuracy10(ds.X_train, ds.y_train, W),
                           te=accuracy10(ds.X_test, ds.y_test, W)))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, key, name in zip(axes, ("tr", "te"), ("training", "test")):
        for sd_ in series:
            mu, sd = mean_sd(sd_[key], axis=1)
            if sd_["ls"] == "-":
                ax.fill_between(sd_["x"], np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                                color=sd_["col"], alpha=0.13, linewidth=0)
            ax.plot(sd_["x"], mu, color=sd_["col"], linestyle=sd_["ls"], linewidth=1.9,
                    label=sd_["lab"])
        ax.set_xlim(series[0]["x"].min(), series[0]["x"].max()); ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.9); ax.set_axisbelow(True)
        ax.set_title(f"TITANIC, smoothed $\\ell_p$ — {name} accuracy", color=INK, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        i0 = int(0.4 * len(series[0]["x"]))
        ins = ax.inset_axes([0.46, 0.13, 0.50, 0.40])
        lo, hi = [], []
        for sd_ in series:
            mu, sd = mean_sd(sd_[key], axis=1)
            se = sd / np.sqrt(sd_[key].shape[1])
            ins.fill_between(sd_["x"][i0:], (mu - se)[i0:], (mu + se)[i0:], color=sd_["col"],
                             alpha=0.20, linewidth=0)
            ins.plot(sd_["x"][i0:], mu[i0:], color=sd_["col"], linestyle=sd_["ls"], linewidth=1.4)
            lo.append((mu - se)[i0:].min()); hi.append((mu + se)[i0:].max())
        pad = 0.18 * (max(hi) - min(lo) + 1e-9)
        ins.set_ylim(min(lo) - pad, max(hi) + pad); ins.set_xlim(series[0]["x"][i0], series[0]["x"][-1])
        ins.tick_params(labelsize=6.5); ins.grid(True, alpha=0.9)
        ins.set_title("zoom (mean $\\pm$ 1 s.e.)", fontsize=6.8, pad=2, color=INK2)
        for sp in ("top", "right"):
            ins.spines[sp].set_visible(False)
    axes[0].legend(loc="center right", fontsize=8.0, bbox_to_anchor=(1.02, 0.70))

    b0 = int(0.4 * len(series[0]["x"]))
    avg = [s_["te"][b0:].mean(axis=0) for s_ in series]
    d = avg[2] - avg[0]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(R))
    fig.suptitle("Non-reversible, ahead of the reversible arm's own best configuration",
                 fontsize=12.5, y=1.005, x=0.02, ha="left", color=INK)
    import textwrap
    fig.text(0.5, -0.05, "\n".join(textwrap.wrap(
        f"Titanic, smoothed l_p (p = 2.4, eps = 0.18, Lambda = 4), d = 10, exact gradient, LASSO "
        f"anchor (sigma = 10, lambda = {pot.lam:.3g}, delta = {pot.delta:.3g}). eta and s chosen "
        f"on TRAINING accuracy only over 10 candidates at seed 3000; these curves are a "
        f"confirmation run on independent seed {CONF_SEED} with R = {R} coupled chains. Metric is "
        f"accuracy averaged over the post-burn-in checkpoints, i.e. the height of the curve over "
        f"the last 60% of the run. Matched-eta paired gain {d.mean()*100:+.3f} percentage points "
        f"(t = {t:+.1f}). Against the reversible arm at its OWN best step size (eta = 1e-4, dashed): "
        f"{(avg[2].mean()-avg[1].mean())*100:+.3f} points. On the FINAL ITERATE alone the gain is "
        f"only +0.041 points (t = 1.3) -- the two arms share an invariant law, so a single iterate "
        f"cannot separate them; what moves is the average over the run.", 128)),
        ha="center", va="top", fontsize=7.0, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(out_dir, "push_accuracy_titanic_lp.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return p
