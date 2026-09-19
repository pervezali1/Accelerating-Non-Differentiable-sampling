"""Figure for regularizer_study.py -- reads the CSV, so layout can be iterated without re-running.

Panels A and B carry the mechanism: J rotates only INSIDE the blocks (1,2,3), (4,5,6), (7,8,9),
so anisotropy placed inside a block is exploitable and anisotropy placed across blocks is not.
Panel C carries the negative result: no regularizer buys an accuracy win, and the differences
that do reach significance are far smaller than one test row and unstable in sign.
"""
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "regularizer")
OUT = os.path.join(HERE, "figures")

# validated 2-slot categorical palette (dataviz reference instance, all six checks PASS:
# normal dE 33.6, worst CVD dE 24.7 protan)
C_IN, C_ACROSS = "#2a78d6", "#eb6834"
INK, INK2, INK3, GRIDC = "#1f2328", "#57606a", "#8c959f", "#e4e6ea"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIDC,
    "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "grid.color": GRIDC, "grid.linewidth": 0.7, "grid.linestyle": "-", "font.size": 9,
})

N_TEST = 179                       # Titanic test rows; one row = 100/179 percentage points


def style(v):
    """(colour, marker, filled, label-offset) per variant."""
    if v["kind"] == "between":
        return C_ACROSS, "s", True
    if v["lam2"] == 0.0:
        return C_IN, "D", False           # lasso: no ridge at all, a reference not a dose point
    return C_IN, "o", True


def scatter_panel(ax, tbl, xcol, xlabel, title, label_dx, logx=True, only=None):
    for _, r in tbl.iterrows():
        col, mk, filled = style(r)
        lo, hi = (np.exp(np.log(r["ess_ratio_geo"]) + z * 2 * r["ess_ratio_se"])
                  for z in (-1, 1))
        ax.errorbar(r[xcol], r["ess_ratio_geo"], yerr=[[r["ess_ratio_geo"] - lo],
                                                       [hi - r["ess_ratio_geo"]]],
                    fmt="none", ecolor=col, elinewidth=1.3, capsize=3, zorder=2)
        ax.plot(r[xcol], r["ess_ratio_geo"], mk, ms=8.5, color=col,
                mfc=col if filled else "white", mec=col if filled else col, mew=1.6, zorder=3)
        if only is None or r["variant"] in only:
            ax.annotate(r["variant"], (r[xcol], r["ess_ratio_geo"]),
                        textcoords="offset points", xytext=label_dx.get(r["variant"], (10, 5)),
                        fontsize=7.6, color=INK2)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("non-reversible ESS / reversible ESS")
    ax.set_title(title, color=INK, loc="left", fontsize=10.3, pad=6)
    ax.grid(True); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main() -> int:
    tbl = pd.read_csv(os.path.join(RES, "regularizer_summary.csv"))
    main_t = tbl[tbl["eta"] == tbl["eta"].max()].copy()      # the eta/3 check is not a variant
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.7))

    scatter_panel(axes[0], main_t, "aniso_within",
                  "posterior anisotropy INSIDE the blocks $J$ rotates\n"
                  "(geometric mean of the three 3$\\times$3 condition numbers)",
                  "A. Anisotropy $J$ can reach — the gain rises with it",
                  {"iso": (10, -4), "within k=30": (-18, -20), "lasso": (-14, 11),
                   "within k=3": (10, -6), "within k=10": (10, 2), "between k=10": (10, -4)})
    scatter_panel(axes[1], main_t, "aniso_between",
                  "posterior anisotropy ACROSS the blocks\n(ratio of block scales)",
                  "B. Anisotropy $J$ cannot reach — the gain falls",
                  {"between k=10": (-12, 12), "iso": (-30, -4), "within k=10": (10, 2),
                   "within k=3": (10, -4), "lasso": (10, 2), "within k=30": (10, -4)},
                  logx=False, only={"iso", "between k=10"})
    # the matched pair: same within-block and same total anisotropy, only the PLACEMENT differs
    pair = main_t.set_index("variant").loc[["iso", "between k=10"]]
    axes[1].plot(pair["aniso_between"], pair["ess_ratio_geo"], color=INK3, lw=1.2, ls="--",
                 zorder=1)
    axes[1].annotate(
        "matched pair: within-block anisotropy 1.69 vs 1.73,\n"
        "total anisotropy 5.48 vs 5.42 — only the PLACEMENT\n"
        "of the penalty differs, and the gain drops 15%",
        xy=(2.95, 2.03), fontsize=7.4, color=INK2, ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRIDC, lw=0.8))
    axes[1].annotate("the four INSIDE-block variants\nall sit at low across-block anisotropy",
                     xy=(2.40, 2.60), xytext=(2.62, 2.79), fontsize=7.4, color=INK2,
                     arrowprops=dict(arrowstyle="->", color=INK3, lw=1.0))
    axes[1].set_xlim(2.15, 3.95)
    for ax in axes[:2]:
        ax.set_ylim(1.55, 2.95)
    axes[0].set_xticks([2, 3, 5, 7, 10, 20])
    axes[0].set_xticklabels(["2", "3", "5", "7", "10", "20"])
    axes[0].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].plot([], [], "o", ms=8, color=C_IN, label="ridge anisotropy INSIDE each block")
    axes[0].plot([], [], "s", ms=8, color=C_ACROSS, label="ridge anisotropy ACROSS blocks")
    axes[0].plot([], [], "D", ms=8, color=C_IN, mfc="white", mew=1.6,
                 label="lasso only, no ridge (reference)")
    axes[0].legend(loc="lower right", fontsize=7.6)

    # --- panel C: the accuracy answer
    ax = axes[2]
    sub = tbl.copy().iloc[::-1]
    ypos = np.arange(len(sub))
    one_row = 100.0 / N_TEST
    ax.axvline(0.0, color=INK3, lw=1.2, zorder=1)
    for yi, (_, r) in zip(ypos, sub.iterrows()):
        se = abs(r["d_acc_pts"] / r["t_acc"]) if r["t_acc"] != 0 else 0.0
        col, mk, filled = style(r)
        if "eta" in str(r["variant"]):
            col, mk, filled = INK2, "o", True
        ax.errorbar(r["d_acc_pts"], yi, xerr=2 * se, fmt="none", ecolor=col, elinewidth=1.3,
                    capsize=3, zorder=2)
        ax.plot(r["d_acc_pts"], yi, mk, ms=7.5, color=col, mfc=col if filled else "white",
                mec=col, mew=1.5, zorder=3)
        ax.annotate(f"t = {r['t_acc']:+.1f}", (r["d_acc_pts"], yi), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.0, color=INK2)
    ax.set_yticks(ypos); ax.set_yticklabels(sub["variant"], fontsize=8.2)
    ax.set_ylim(-0.7, len(sub) + 0.15)
    half = 1.18 * max(abs(sub["d_acc_pts"]).max(),
                      (abs(sub["d_acc_pts"]) + 2 * abs(sub["d_acc_pts"] / sub["t_acc"])).max())
    ax.set_xlim(-half, half)
    ax.set_xlabel("test accuracy difference, non-reversible $-$ reversible\n(percentage points, "
                  "$\\pm$ 2 paired s.e.)")
    ax.set_title("C. And none of it buys accuracy", color=INK, loc="left", fontsize=10.3, pad=6)
    ax.annotate(f"ONE test row = {one_row:.2f} pts — {one_row / half:.0f}$\\times$ the "
                f"half-width of this axis", xy=(0, len(sub) - 0.10), ha="center", fontsize=7.6,
                color=INK2)
    ax.grid(True, axis="x"); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("Changing the regularizer moves the non-reversible speed-up from "
                 "1.88$\\times$ to 2.61$\\times$, and the accuracy by nothing", fontsize=12.5, y=1.03, x=0.012, ha="left", color=INK)
    fig.text(0.5, -0.17,
             "TITANIC, ball constraint, exact gradient, elastic net U = loglik + w_0^2/(2 sigma^2) "
             "+ lam1 sum|w_j| + (lam2/2) sum c_j w_j^2, with U_0 smoothing ONLY the l_1 term -- so "
             "a = exp(U - U_0) depends on (lam1, delta) alone and is IDENTICAL across every "
             "variant shown; the only thing that moves is c.  lam1 = 0.01 n_train = 7.12, "
             "lam2 = 0.05 n_train = 35.6, a in [0.5, 1].  eta = 1e-4, s = (5,5,5), 96 chains "
             "started from a common stationary state and sharing one Gaussian stream, 24000 "
             "measured iterations stored every 2, seed 7100 (used nowhere else in the "
             "repository).  Split-Rhat <= 1.031 on every coordinate of every run.  ESS from "
             "Geyer initial-positive/monotone IACT per chain per coordinate w_1..w_9; the ratio "
             "is the geometric mean over the 9 x 96 paired chain-coordinates and the bar is "
             "+- 2 paired s.e. on the log ratio.  `between k=10` and `within k=10` use the SAME "
             "multiset of c_j -- the same penalty anisotropy -- placed differently relative to "
             "the blocks; their posterior anisotropies are measured, not assumed.  Panel C's "
             "accuracy is averaged over the whole measured (post-burn-in) run; the axis there "
             "spans less than a third of ONE test row out of 179, so several differences are "
             "statistically resolvable and none is numerically meaningful.  The bottom row "
             "re-runs one variant at eta/3 with 3x the iterations, so the simulated time "
             "matches: the difference is finite-step bias, not a property of the target, which "
             "the two arms provably share.",
             ha="center", va="top", fontsize=7.0, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"regularizer_study.{ext}"),
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    print(f"-> {os.path.join(OUT, 'regularizer_study.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
