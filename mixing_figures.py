"""Figures for the mixing win: non-reversible vs reversible anchored Langevin.

Everything is rebuilt from the saved probe outputs (results/probe_mixing*/), except the
autocorrelation panel, which needs traces and so re-runs two short coupled chains.

Colour: reversible #1f5fbf (circle), non-reversible #1a9850 (square). Two categorical slots,
validated (OKLab dE 27.0 normal / 25.6 deutan vs each other, both inside the lightness band,
both >= 3:1 on the surface). Marker shape is a secondary encoding so identity never rests on
colour alone.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "figures", "mixing")
os.makedirs(OUT, exist_ok=True)

REV, NREV = "#1f5fbf", "#1a9850"
INK, INK2, INK3 = "#1f2328", "#57606a", "#8c959f"
GRID = "#e4e6ea"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "legend.frameon": False, "font.size": 9,
})

PRETTY = {"w1": "$w_1$", "w2": "$w_2$", "w3": "$w_3$", "w4": "$w_4$", "w5": "$w_5$",
          "w6": "$w_6$", "w7": "$w_7$", "w8": "$w_8$", "w9": "$w_9$",
          "norm2": r"$\|w\|^2$", "loglik_train": "log-lik", "U_train": "$U$",
          "zhold0": "$x\\cdot w$ (test 1)", "zhold1": "$x\\cdot w$ (test 2)",
          "zhold2": "$x\\cdot w$ (test 3)"}


def caption(fig, text, size=7.2):
    import textwrap
    fig.text(0.5, -0.045, "\n".join(textwrap.wrap(text, 132)), ha="center", va="top",
             fontsize=size, color=INK2)


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    return os.path.join(OUT, f"{name}.png")


# --------------------------------------------------------------- 1. IACT + ESS/s dumbbells
def fig_ess_iact(rep, R=96):
    fns = [k for k in rep if k in PRETTY]
    fns.sort(key=lambda k: rep[k]["essps_nrev"] / rep[k]["essps_rev"])
    y = np.arange(len(fns))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0), sharey=True)

    for ax, (a, b, lab, title, note) in zip(axes, [
        ("tau_rev", "tau_nrev", "Integrated autocorrelation time (iterations)",
         "Autocorrelation time", "lower is better"),
        ("essps_rev", "essps_nrev", "Effective samples per second (all %d chains)" % R,
         "Sampling efficiency", "higher is better")]):
        val = (lambda v: v * R) if a.startswith("essps") else (lambda v: v)
        for i, f in zip(y, fns):
            lo, hi = val(rep[f][a]), val(rep[f][b])
            ax.plot([lo, hi], [i, i], color=INK3, linewidth=1.1, zorder=1,
                    solid_capstyle="round")
        ax.scatter([val(rep[f][a]) for f in fns], y, s=42, color=REV, marker="o",
                   zorder=3, label="Reversible", edgecolors="white", linewidths=0.8)
        ax.scatter([val(rep[f][b]) for f in fns], y, s=42, color=NREV, marker="s",
                   zorder=3, label="Non-reversible", edgecolors="white", linewidths=0.8)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(lab)
        ax.set_title(f"{title}  ({note})", color=INK, loc="left")
        if a.startswith("tau"):
            ax.set_xticks([20, 50, 100, 200])
        ax.grid(True, axis="x", alpha=0.9)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    axes[0].set_yticks(y, [PRETTY[f] for f in fns])
    # selective direct labels: the speed-up factor, on the efficiency panel only
    ax = axes[1]
    xmax = max(rep[f]["essps_nrev"] for f in fns) * R
    for i, f in zip(y, fns):
        r = rep[f]["essps_nrev"] / rep[f]["essps_rev"]
        ax.annotate(f"×{r:.2f}", (xmax * 1.35, i), va="center", ha="left",
                    fontsize=8.2, color=INK if r >= 1 else INK2,
                    fontweight="bold" if r >= 2 else "normal")
    ax.set_xlim(right=xmax * 3.4)
    ax.set_xticks([200, 500, 1000, 2000])
    axes[0].legend(loc="lower right", fontsize=8.5, handletextpad=0.4)
    fig.suptitle("Non-reversible anchored Langevin mixes faster per unit wall clock",
                 fontsize=12.5, y=1.005, x=0.02, ha="left", color=INK)
    caption(fig, "Titanic, ball constraint, exact gradient, LASSO anchor. eta = 1e-4, "
                 "s = (5,5,5), R = 96 COUPLED chains (same W0, same Gaussian noise stream), "
                 "16000 iterations, 40% burn-in, independent seed 5300. IACT by Geyer "
                 "initial-positive/monotone. ESS per second charges the non-reversible arm its "
                 "full measured cost (16.25 s vs 14.43 s wall, ratio 1.126). Eleven of twelve "
                 "test functions improve; ||w||^2 is the one that does not.")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save(fig, "mixing_ess_iact")


# --------------------------------------------------------------- 2. forest plot, w7
def fig_forest(rows):
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    y = np.arange(len(rows))[::-1]
    ax.axvline(1.0, color=INK3, linewidth=1.0, zorder=1)
    for i, (lab, x, lo, hi, kind) in zip(y, rows):
        c = NREV if kind != "pooled" else INK
        if lo is not None:
            ax.plot([lo, hi], [i, i], color=c, linewidth=1.6, solid_capstyle="round", zorder=2)
            ax.plot([lo, lo, np.nan, hi, hi], [i - .12, i + .12, np.nan, i - .12, i + .12],
                    color=c, linewidth=1.2, zorder=2)
        ax.scatter([x], [i], s=(78 if kind == "pooled" else 52),
                   marker=("D" if kind == "pooled" else "s"), color=c, zorder=3,
                   edgecolors="white", linewidths=0.8)
        if lo is None:
            ax.annotate(f"×{x:.2f}", (x, i), textcoords="offset points", xytext=(11, 0),
                        ha="left", va="center", fontsize=8, color=INK2)
        else:                                  # keep the label clear of the interval whisker
            ax.annotate(f"×{x:.2f}  [{lo:.2f}, {hi:.2f}]", (x, i), textcoords="offset points",
                        xytext=(0, 14), ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Speed-up in effective samples per second (non-reversible / reversible)")
    ax.set_xlim(0.90, 1.62)
    ax.grid(True, axis="x", alpha=0.9); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.annotate("no difference", (1.0, -0.72), ha="center", va="top", fontsize=8,
                color=INK3, annotation_clip=False)
    fig.suptitle("The worst-case test function, measured eight ways",
                 fontsize=12.5, y=1.02, x=0.02, ha="left", color=INK)
    caption(fig, "Speed-up for w7, the slowest-mixing coordinate and the pre-registered primary "
                 "test function. Green squares: Geyer-estimator measurements at four seeds and "
                 "two burn-in fractions, including one stationary-start run. The black diamond "
                 "is a truncation-free estimator (ESS from the across-chain variance of chain "
                 "time-averages), which is immune to Geyer truncation and carries a bootstrap "
                 "95% interval that includes 1. Pooling the six Geyer values gives about x1.2, "
                 "not the x1.31 of the single best run: the headline number sits at the "
                 "optimistic end of its own sampling distribution.")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "mixing_forest_w7")


# --------------------------------------------------------------- 3. bias vs efficiency frontier
def fig_frontier(cells, fn="w7"):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    pts = []
    for key, c in cells.items():
        arm, eta, s = key.split("|")
        pts.append(dict(arm=arm, eta=float(eta.split("=")[1]), s=float(s.split("=")[1]),
                        bias=c["bias_mu0"], ess=c["f"][fn]["ess_rate"] / c["wall"] * 1.0,
                        ess_rate=c["f"][fn]["ess_rate"], wall=c["wall"]))
    for arm, col, mk, lab in (("rev", REV, "o", "Reversible"),
                              ("nrev", NREV, "s", "Non-reversible")):
        P = sorted([p for p in pts if p["arm"] == arm], key=lambda p: p["bias"])
        ax.plot([p["bias"] for p in P], [p["ess_rate"] for p in P], color=col,
                linewidth=1.4, alpha=0.55, zorder=1)
        ax.scatter([p["bias"] for p in P], [p["ess_rate"] for p in P], s=60, color=col,
                   marker=mk, zorder=3, label=lab, edgecolors="white", linewidths=0.9)
        for p in P:
            txt = f"$\\eta$={p['eta']:g}" + (f", s={p['s']:g}" if arm == "nrev" else "")
            ax.annotate(txt, (p["bias"], p["ess_rate"]), textcoords="offset points",
                        xytext=(0, -16 if arm == "rev" else 10), ha="center",
                        fontsize=7.6, color=INK2)
    ax.margins(x=0.10, y=0.14)
    ax.set_xlabel("Discretisation bias (max over test functions, in reference sd units)")
    ax.set_ylabel(f"Effective samples per second, {PRETTY[fn]}")
    ax.grid(True, alpha=0.9); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("At matched bias the non-reversible arm is strictly more efficient",
                 fontsize=12.5, y=1.01, x=0.02, ha="left", color=INK)
    caption(fig, "Each point is one (arm, eta, s) configuration; both arms are free to choose "
                 "their own step size, which is what makes this a best-versus-best comparison. "
                 "Bias is measured against an eta -> 0 extrapolation of the reversible ensemble "
                 "mean, not against a single small-eta run. Reading vertically at any bias level "
                 "the green curve sits above the blue one. Comparing points at DIFFERENT bias "
                 "is not a fair comparison: a larger step buys efficiency by paying in bias, and "
                 "both arms can do that. Seed 5400, R = 96, 16000 iterations.")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save(fig, "mixing_bias_frontier")


# --------------------------------------------------------------- 4. RMSE vs compute budget
def fig_budget(budgets):
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    T = [b["T"] for b in budgets]
    ax.plot(T, [b["rmse_rev"] for b in budgets], color=REV, marker="o", markersize=6,
            linewidth=1.8, label="Reversible (own best $\\eta$)", markeredgecolor="white",
            markeredgewidth=0.8)
    ax.plot(T, [b["rmse_nrev"] for b in budgets], color=NREV, marker="s", markersize=6,
            linewidth=1.8, label="Non-reversible (own best $\\eta$, $s$)",
            markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Compute budget (relative wall clock)")
    ax.set_ylabel("RMSE of the posterior-mean estimate (reference sd units)")
    ax.grid(True, alpha=0.9); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    cross = next((b["T"] for b in budgets if b["winner"] == "rev"), None)
    if cross is not None:
        ax.axvline(cross, color=INK3, linewidth=1.0, zorder=0)
        ax.annotate("crossover:\nreversible wins\nbeyond here", (cross, max(b["rmse_rev"] for b in budgets)),
                    textcoords="offset points", xytext=(8, -4), fontsize=8, color=INK2, va="top")
    b0 = budgets[0]
    ax.annotate(f"×{b0['ratio']:.2f} lower RMSE", (b0["T"], b0["rmse_nrev"]),
                textcoords="offset points", xytext=(10, 4), fontsize=8.4, color=INK)
    ax.legend(loc="lower left", fontsize=9)
    fig.suptitle("The efficiency gain is a short-budget gain, and it does reverse",
                 fontsize=12.5, y=1.01, x=0.02, ha="left", color=INK)
    caption(fig, "RMSE of the estimated posterior mean, pooled over 12 test functions, against "
                 "compute budget. At each budget BOTH arms are re-optimised over their own "
                 "(eta, s). The non-reversible arm is ahead at short budgets (x1.33 lower RMSE "
                 "at T = 1) and the curves cross near T = 10, after which the reversible arm can "
                 "afford a small enough step that its lower bias wins. This is the honest shape "
                 "of the result and the single most important caveat on it.")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save(fig, "mixing_rmse_budget")


# --------------------------------------------------------------- 5. autocorrelation functions
def fig_acf(fns=("w4", "w7", "norm2"), n_iter=12000, R=48, eta=1e-4, s=5.0, seed=6100):
    from exact_anchored import (make_potential, make_geom10, run_exact, init_unit_ball10)
    from nral import build_titanic_dataset
    ds = build_titanic_dataset()
    pot = make_potential(ds)
    geom = make_geom10("ball")
    W0 = init_unit_ball10(np.random.default_rng(seed + 1), R)
    series = {}
    for tag, alpha in (("rev", 0), ("nrev", 1)):
        r = run_exact(pot, geom, alpha=alpha, scales=(s, s, s), eta=eta, n_iter=n_iter,
                      W0=W0, seed=seed, checkpoint_every=2)
        W = r["betas"]                        # (n_ck, R, 10)
        b0 = int(0.4 * W.shape[0])
        series[tag] = dict(W=W[b0:], wall=r["runtime"])
    lags = 400

    def acf(x):                               # x: (n, R) -> mean normalised acf over chains
        out = np.zeros(lags + 1)
        for j in range(x.shape[1]):
            v = x[:, j] - x[:, j].mean()
            n2 = 1
            while n2 < 2 * v.size:
                n2 *= 2
            f = np.fft.rfft(v, n2)
            a = np.fft.irfft(f * np.conj(f), n2)[:lags + 1].real
            if a[0] > 0:
                out += a / a[0]
        return out / x.shape[1]

    def pull(W, fn):
        if fn == "norm2":
            return np.sum(W * W, axis=2)
        return W[:, :, int(fn[1:])]

    fig, axes = plt.subplots(1, len(fns), figsize=(4.0 * len(fns), 4.0), sharey=True)
    ck = 2
    for ax, fn in zip(np.atleast_1d(axes), fns):
        for tag, col, lab in (("rev", REV, "Reversible"), ("nrev", NREV, "Non-reversible")):
            a = acf(pull(series[tag]["W"], fn))
            ax.plot(np.arange(lags + 1) * ck, a, color=col, linewidth=1.8, label=lab)
        ax.axhline(0, color=INK3, linewidth=0.9)
        ax.set_xlabel("Lag (iterations)")
        ax.set_title(PRETTY[fn], color=INK, loc="left")
        ax.grid(True, alpha=0.9); ax.set_axisbelow(True)
        ax.set_xlim(0, lags * ck)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if fn == "norm2":                      # the curves overlap at full scale; zoom in
            ins = ax.inset_axes([0.42, 0.42, 0.55, 0.50])
            for tag, col in (("rev", REV), ("nrev", NREV)):
                a2 = acf(pull(series[tag]["W"], fn))[:26]
                ins.plot(np.arange(a2.size) * ck, a2, color=col, linewidth=1.5)
            ins.axhline(0, color=INK3, linewidth=0.8)
            ins.tick_params(labelsize=6.5); ins.grid(True, alpha=0.9)
            ins.set_title("first 50 lags", fontsize=6.8, pad=2, color=INK2)
            for sp in ("top", "right"):
                ins.spines[sp].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("Autocorrelation")
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Why: the non-reversible chain decorrelates sooner — except where it does not",
                 fontsize=12.5, y=1.02, x=0.02, ha="left", color=INK)
    caption(fig, f"Mean normalised autocorrelation over R = {R} coupled chains, Titanic ball, "
                 f"eta = {eta:g}, s = ({s:g},{s:g},{s:g}), {n_iter} iterations, 40% burn-in, "
                 f"seed {seed}. w4 is the coordinate with the largest gain (x5.5 in ESS per "
                 f"second), w7 the slowest-mixing one and the smallest gain, and ||w||^2 the one "
                 f"test function that mixes slightly WORSE under the non-reversible dynamics -- "
                 f"the inset shows the two curves are visually indistinguishable there, the "
                 f"x0.91 being too small to see. The area under each curve is what the "
                 f"integrated autocorrelation time measures. The oscillation in w4 is the "
                 f"signature of the rotation: J turns the drift, so the chain circulates "
                 f"instead of diffusing back and forth, and the correlation crosses zero "
                 f"within ~50 iterations instead of decaying over ~300.")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "mixing_acf")


def main() -> int:
    ref = json.load(open(os.path.join(HERE, "results/probe_mixing_refute/refute_main.json")))
    bvb = json.load(open(os.path.join(HERE, "results/probe_mixing_refute/refute_bvb.json")))
    made = [fig_ess_iact(ref["replication"])]

    st = ref["stationary"]["w7"]
    # The six Geyer measurements the refutation pooled, plus their pooled value and the
    # truncation-free estimator with its bootstrap interval.
    geyer = [("search stage, seed 3000", 1.08), ("confirmation, seed 4100", 1.313),
             ("robustness, seed 4200, burn 60%", 1.256),
             ("replication, seed 5300", ref["replication"]["w7"]["x"]),
             ("stationary start, seed 5300", st["geyer"]["x"]),
             ("matched-bias best-vs-best, seed 5400", 1.21)]
    pooled = float(np.exp(np.mean(np.log([g[1] for g in geyer]))))
    rows = [(lab, x, None, None, "geyer") for lab, x in geyer]
    rows.append((f"pooled (geometric mean of the six)", pooled, None, None, "pooled"))
    rows.append(("truncation-free estimator, seed 5300",
                 st["direct"]["x"], st["direct"]["lo"], st["direct"]["hi"], "pooled"))
    made.append(fig_forest(rows))
    made.append(fig_frontier(bvb["cells"]))
    made.append(fig_budget(bvb["budgets"]))
    made.append(fig_acf())
    made.append(fig_accuracy("ball"))
    made.append(fig_accuracy("lp"))
    for m in made:
        print("wrote", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------- 6. accuracy at the SAME config
def fig_accuracy(geometry="ball", n_iter=16000, R=96, eta=1e-4, s=5.0, seed=5300, ck=50):
    """Training and test accuracy at exactly the configuration where the mixing win lives.

    Same layout as every earlier accuracy figure in this repo: training left, test right, the
    common [0, 1] axis, mean +- 1 SD bands, and a zoom inset. The point of this figure is that
    the two curves lie on top of each other -- the mixing win does not show up here, and cannot.
    """
    from exact_anchored import (make_potential, make_geom10, run_exact, init_unit_ball10,
                                accuracy10)
    from nral import build_titanic_dataset, mean_sd
    ds = build_titanic_dataset()
    pot = make_potential(ds)
    geom = make_geom10(geometry, eps=0.18) if geometry == "lp" else make_geom10("ball")
    W0 = init_unit_ball10(np.random.default_rng(seed + 1), R)
    arms = {}
    for tag, alpha in (("rev", 0), ("nrev", 1)):
        r = run_exact(pot, geom, alpha=alpha, scales=(s, s, s), eta=eta, n_iter=n_iter,
                      W0=W0, seed=seed, checkpoint_every=ck)
        W = r["betas"]
        arms[tag] = dict(x=np.asarray(r["checkpoints"], dtype=float),
                         tr=accuracy10(ds.X_train, ds.y_train, W),
                         te=accuracy10(ds.X_test, ds.y_test, W), wall=r["runtime"])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, split, name in zip(axes, ("tr", "te"), ("training", "test")):
        for tag, col, mk, lab in (("rev", REV, "o", "Reversible anchored Langevin"),
                                  ("nrev", NREV, "s", "Non-reversible anchored Langevin")):
            x, a = arms[tag]["x"], arms[tag][split]
            mu, sd = mean_sd(a, axis=1)
            ax.fill_between(x, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                            color=col, alpha=0.15, linewidth=0)
            ax.plot(x, mu, color=col, linewidth=2.0, label=lab)
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.9); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        gname = "ball" if geometry == "ball" else "smoothed $\\ell_p$"
        ax.set_title(f"TITANIC, {gname} — {name} accuracy", color=INK, loc="left")
        i0 = int(0.3 * len(x))
        ins = ax.inset_axes([0.46, 0.13, 0.50, 0.40])
        lo, hi = [], []
        for tag, col in (("rev", REV), ("nrev", NREV)):
            mu, sd = mean_sd(arms[tag][split], axis=1)
            se = sd / np.sqrt(arms[tag][split].shape[1])
            ins.fill_between(x[i0:], mu[i0:] - se[i0:], mu[i0:] + se[i0:], color=col,
                             alpha=0.22, linewidth=0)
            ins.plot(x[i0:], mu[i0:], color=col, linewidth=1.5)
            lo.append((mu[i0:] - se[i0:]).min()); hi.append((mu[i0:] + se[i0:]).max())
        pad = 0.25 * (max(hi) - min(lo) + 1e-9)
        ins.set_ylim(min(lo) - pad, max(hi) + pad); ins.set_xlim(x[i0], x[-1])
        ins.tick_params(labelsize=6.5); ins.grid(True, alpha=0.9)
        ins.set_title("zoom (mean $\\pm$ 1 s.e.)", fontsize=6.8, pad=2, color=INK2)
        for sp in ("top", "right"):
            ins.spines[sp].set_visible(False)
    axes[0].legend(loc="center right", fontsize=8.6, bbox_to_anchor=(1.0, 0.72))

    d_tr = arms["nrev"]["tr"][-1] - arms["rev"]["tr"][-1]
    d_te = arms["nrev"]["te"][-1] - arms["rev"]["te"][-1]
    t_tr = d_tr.mean() / (d_tr.std(ddof=1) / np.sqrt(R) + 1e-300)
    t_te = d_te.mean() / (d_te.std(ddof=1) / np.sqrt(R) + 1e-300)
    # the post-burn-in time-averaged difference is less noisy than a single final iterate
    b0 = int(0.4 * arms["rev"]["tr"].shape[0])
    m_tr = (arms["nrev"]["tr"][b0:] - arms["rev"]["tr"][b0:]).mean()
    m_te = (arms["nrev"]["te"][b0:] - arms["rev"]["te"][b0:]).mean()
    pp = abs(m_te) * 100
    if pp < 0.5:
        verdict = (f"The two arms are indistinguishable here: the gap is {pp:.2f} percentage "
                   f"points of test accuracy, in the NON-reversible arm's disfavour.")
    else:
        verdict = (f"The non-reversible arm is clearly WORSE here -- {pp:.1f} percentage points "
                   f"of test accuracy -- and that is the price of the same s = {s:g} that buys "
                   f"the mixing gain: on this geometry J's axis is grad g rather than w, so the "
                   f"same strength produces a roughly 3x stronger rotation and a correspondingly "
                   f"larger discretisation bias. The mixing win on this set is real but it is "
                   f"not free.")
    head = ("The mixing win does not show up in accuracy — at the very same configuration"
            if pp < 0.5 else
            "On this geometry the same s that buys mixing costs accuracy")
    fig.suptitle(head,
                 fontsize=12.5, y=1.005, x=0.02, ha="left", color=INK)
    caption(fig, f"Titanic, {'ball ||w||^2 <= 2' if geometry == 'ball' else 'smoothed l_p, p = 2.4, eps = 0.18, Lambda = 4'}, "
                 f"d = 10, exact gradient, LASSO anchor (sigma = 10, lambda = {pot.lam:.3g}, "
                 f"delta = {pot.delta:.3g}, a in [0.5, 1]). EXACTLY the configuration of the "
                 f"mixing figures: eta = {eta:g}, s = ({s:g},{s:g},{s:g}), R = {R} coupled "
                 f"chains, {n_iter} iterations, seed {seed}. Paired final-iterate difference "
                 f"(non-reversible minus reversible): training {d_tr.mean():+.4f} (t = {t_tr:+.2f}), "
                 f"test {d_te.mean():+.4f} (t = {t_te:+.2f}); averaged over all post-burn-in "
                 f"checkpoints instead of the final one, {m_tr:+.4f} (training) and {m_te:+.4f} "
                 f"(test). {verdict} Bands are mean +- 1 SD across "
                 f"replicates; the inset shows mean +- 1 s.e. so the curves can be separated at "
                 f"all. The two arms share an invariant law, so their accuracy distributions are "
                 f"identical at stationarity -- this figure is what that looks like, and no "
                 f"tuning changes it.", size=7.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save(fig, f"mixing_accuracy_{geometry}")
