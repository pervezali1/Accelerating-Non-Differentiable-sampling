"""Per-block rotation strengths for non-reversible anchored Langevin, selected under a discretisation-bias constraint.

AUTO-GENERATED from blocks_notebook_source.py by make_notebook.py -- edit that file, not this one.
Contains exactly the library cells of per_block_strengths.ipynb, so the notebook and this module
can never drift apart.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from typing import Dict, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit

import exact_nb as X                                                          # noqa: E402
from exact_nb import D10, design_with_intercept, softplus                     # noqa: E402
from nral import _wrap, build_titanic_dataset                                 # noqa: E402
from probe_mixing import iact_geyer, split_rhat                               # noqa: E402

# AFTER the repository imports, not before: probe_mixing pulls in exact_anchored, which forces the
# Agg backend at module level, and that would leave every figure below invisible in the notebook.
try:
    get_ipython().run_line_magic("matplotlib", "inline")              # type: ignore[name-defined]
except Exception:
    matplotlib.use("Agg")

BLOCKS = X.BLOCKS10                     # ((1,2,3), (4,5,6), (7,8,9)); the intercept is never rotated
ETA = 1e-4
SEED_SEARCH, SEED_CONF, SEED_GOLD = 8100, 9200, 8107   # disjoint; used nowhere else in the repo
LAM1_FRAC, LAM2_FRAC = 0.01, 0.05
LEVELS = (1.0, 2.0, 5.0, 10.0, 20.0)
GRADED_K = (1.0, 5.0, 25.0)             # per-block within-block anisotropy, by construction
CK = 2
THIN_BIAS = 8
GOLD_DIV = 10.0

# DECLARED IN ADVANCE.  The reversible arm ignores s entirely, so its discretisation bias at this
# eta is one fixed number per potential.  A cell is admissible if the non-reversible arm's bias is
# at most CAP_MULT times it.  An exact match is the wrong rule: at small s the true bias sits below
# the Monte Carlo floor of the gold reference, so an exact cap rejects cells on rounding noise.
CAP_MULT = 2.0
BSTARS = (0.10, 0.25, 0.50)             # fixed thresholds, reported alongside, never used to select

QUICK = os.environ.get("BNB_MODE", "full").lower() == "quick"
R_G, NM_G = (8, 3000) if QUICK else (32, 160_000)          # gold
R_S, NB_S, NM_S = (12, 400, 1200) if QUICK else (32, 3000, 10_000)      # search
R_C, NB_C, NM_C = (16, 400, 1500) if QUICK else (96, 4000, 16_000)      # confirmation

# validated 3-slot categorical palette (dataviz reference instance; normal dE 27.6, worst CVD
# dE 9.2 deutan; the aqua slot carries a contrast WARN, relieved by direct labels + printed tables)
BLK = ("#2a78d6", "#eb6834", "#1baf7a")
INK, INK2, INK3, GRIDC = "#1f2328", "#57606a", "#8c959f", "#e4e6ea"
REV, NREV = "#1f5fbf", "#1a9850"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIDC,
    "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "grid.color": GRIDC, "grid.linewidth": 0.7, "grid.linestyle": "-", "font.size": 9,
})
RESULTS_DIR = os.environ.get("BNB_RESULTS", os.path.join(os.getcwd(), "results", "blocks_nb"))
FIGURE_DIR = os.environ.get("BNB_FIGURES", os.path.join(os.getcwd(), "figures", "blocks_nb"))
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
CHECKS: Dict[str, object] = {}

class ElasticPotential:
    """Elastic net on w_1..w_9 with per-coordinate ridge weights c; Gaussian prior on w_0."""

    def __init__(self, Xd, y, sigma, lam1, lam2, c, delta):
        self.X, self.y = Xd, y.astype(float)
        self.sigma, self.lam1, self.lam2, self.delta = sigma, lam1, lam2, delta
        self.c = np.asarray(c, dtype=float)
        assert self.c.shape == (D10 - 1,)

    def _loglik(self, W):
        z = np.atleast_2d(W) @ self.X.T
        return np.sum(softplus(z) - self.y[None, :] * z, axis=1)

    def _ridge(self, W2):
        return 0.5 * self.lam2 * np.sum(self.c[None, :] * W2[:, 1:] ** 2, axis=1)

    def U(self, W):
        W2 = np.atleast_2d(W)
        out = (self._loglik(W2) + W2[:, 0] ** 2 / (2 * self.sigma ** 2)
               + self.lam1 * np.sum(np.abs(W2[:, 1:]), axis=1) + self._ridge(W2))
        return out[0] if np.ndim(W) == 1 else out

    def U0(self, W):
        W2 = np.atleast_2d(W)
        out = (self._loglik(W2) + W2[:, 0] ** 2 / (2 * self.sigma ** 2)
               + self.lam1 * np.sum(np.sqrt(W2[:, 1:] ** 2 + self.delta ** 2), axis=1)
               + self._ridge(W2))
        return out[0] if np.ndim(W) == 1 else out

    def grad_U0(self, W):
        W2 = np.atleast_2d(W)
        g = (expit(W2 @ self.X.T) - self.y[None, :]) @ self.X
        g[:, 0] += W2[:, 0] / self.sigma ** 2
        g[:, 1:] += (self.lam1 * W2[:, 1:] / np.sqrt(W2[:, 1:] ** 2 + self.delta ** 2)
                     + self.lam2 * self.c[None, :] * W2[:, 1:])
        return g[0] if np.ndim(W) == 1 else g

    def a(self, W):
        W2 = np.atleast_2d(W)
        gap = np.sum(np.sqrt(W2[:, 1:] ** 2 + self.delta ** 2) - np.abs(W2[:, 1:]), axis=1)
        out = np.exp(-self.lam1 * gap)
        return out[0] if np.ndim(W) == 1 else out


def graded_c(ks=GRADED_K) -> np.ndarray:
    """Block l gets the spread (1/k_l, 1, k_l)."""
    return np.concatenate([[1.0 / k, 1.0, k] for k in ks])

def test_functions(Ws, pot):
    """(T, R, 10) -> (13, n): the ten coordinates, |w|^2, U_0, and TRAINING accuracy."""
    F = Ws[::THIN_BIAS].reshape(-1, D10)
    acc = np.mean(((F @ pot.X.T) >= 0.0) == pot.y.astype(bool)[None, :], axis=1)
    return np.vstack([F.T, np.sum(F ** 2, axis=1)[None, :], pot.U0(F)[None, :], acc[None, :]])


def gold_reference(pot, geom, *, seed):
    eta_g = ETA / GOLD_DIV
    W0 = X.init_unit_ball(np.random.default_rng(seed + 1), R_G)
    nb = int(NB_S * GOLD_DIV)
    burn = X.run_chain(pot, geom, alpha=0, scales=(0., 0., 0.), eta=eta_g, n_iter=nb,
                       W0=W0, seed=seed, checkpoint_every=max(1, nb // 2))
    r = X.run_chain(pot, geom, alpha=0, scales=(0., 0., 0.), eta=eta_g, n_iter=NM_G,
                    W0=burn["W"][-1], seed=seed + 99, checkpoint_every=int(CK * GOLD_DIV))
    T = test_functions(r["W"], pot)
    return T.mean(axis=1), T.std(axis=1, ddof=1)


def ess_per_chain(S: np.ndarray, ck: int) -> np.ndarray:
    """S is (T, R). Geyer initial-positive/monotone IACT per chain -> ESS in iterations."""
    T, R = S.shape
    out = np.empty(R)
    for r in range(R):
        tau = iact_geyer(S[:, r])
        out[r] = np.nan if not np.isfinite(tau) else T / (2.0 * max(tau, 0.5))
    return out

def measure(pot, geom, scales, *, R, n_burn, n_meas, seed, gold):
    W0 = X.init_unit_ball(np.random.default_rng(seed + 1), R)
    burn = X.run_chain(pot, geom, alpha=0, scales=(0., 0., 0.), eta=ETA, n_iter=n_burn,
                       W0=W0, seed=seed, checkpoint_every=max(1, n_burn // 2))
    Wstat = burn["W"][-1]
    res = {}
    for tag, alpha in (("rev", 0), ("nrev", 1)):
        res[tag] = X.run_chain(pot, geom, alpha=alpha, scales=scales, eta=ETA, n_iter=n_meas,
                               W0=Wstat, seed=seed + 99, checkpoint_every=CK)
        assert res[tag]["n_nonfinite"] == 0, f"non-finite states at s={scales}"

    per_block, logr_all, rhat = [], [], 0.0
    for blk in BLOCKS:
        lr = []
        for j in blk:
            e = {}
            for tag in ("rev", "nrev"):
                S = res[tag]["W"][:, :, j]
                e[tag] = ess_per_chain(S, CK)
                rhat = max(rhat, split_rhat(S))
            ok = np.isfinite(e["rev"]) & np.isfinite(e["nrev"]) & (e["rev"] > 0)
            lr.append(np.log(e["nrev"][ok] / e["rev"][ok]))
        L = np.concatenate(lr)
        logr_all.append(L)
        se = L.std(ddof=1) / np.sqrt(L.size)
        per_block.append((float(np.exp(L.mean())), float(se), float(L.mean() / se)))
    LA = np.concatenate(logr_all)
    seA = LA.std(ddof=1) / np.sqrt(LA.size)

    gm, gs = gold
    bias = {t: float(np.max(np.abs(test_functions(res[t]["W"], pot).mean(axis=1) - gm)
                            / np.maximum(gs, 1e-12))) for t in ("rev", "nrev")}
    acc = {t: X.accuracy(DS.X_test, DS.y_test, res[t]["W"]).mean(axis=0) for t in ("rev", "nrev")}
    d_acc = X.paired(acc["nrev"], acc["rev"])
    return dict(per_block=per_block, overall=float(np.exp(LA.mean())), overall_se=float(seA),
                overall_t=float(LA.mean() / seA), rhat=float(rhat),
                bias_rev=bias["rev"], bias_nrev=bias["nrev"],
                acc_rev=float(acc["rev"].mean()), acc_nrev=float(acc["nrev"].mean()),
                d_acc_pts=d_acc[0] * 100.0, t_acc=d_acc[2],
                proj_rev=res["rev"]["projection_rate"], proj_nrev=res["nrev"]["projection_rate"])

def response_figure(search, pname, outdir):
    """Left: the gain, which rises monotonically in s and is therefore NOT a selection criterion
    on its own. Right: the bias that decides which of those gains are real."""
    sub = search[(search.potential == pname) & (search["mode"] != "uniform")]
    uni = search[(search.potential == pname) & (search["mode"] == "uniform")].sort_values("s")
    cap = float(sub["bias_rev"].median()) * CAP_MULT
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for b in range(3):
        d = sub[sub.target_block == b + 1].sort_values("s")
        lab = f"block {b + 1} = $w_{{{','.join(str(j) for j in BLOCKS[b])}}}$"
        if pname == "graded":
            lab += f"  ($k={GRADED_K[b]:g}$)"
        for ax, col in zip(axes, ("own_ratio", "bias_nrev")):
            ax.plot(d["s"], d[col], "-o", color=BLK[b], ms=6.5, lw=1.9, mfc=BLK[b],
                    mec="white", mew=1.0, label=lab, zorder=3)
        # direct label at the right end (the aqua slot carries a contrast WARN)
        axes[0].annotate(f"block {b + 1}", (d["s"].iloc[-1], d["own_ratio"].iloc[-1]),
                         textcoords="offset points", xytext=(7, -2), fontsize=7.6, color=BLK[b])
    axes[0].plot(uni["s"], uni["overall"], "--s", color=INK3, ms=5.5, lw=1.6,
                 label="uniform $s=(v,v,v)$, all nine coordinates", zorder=2)
    axes[0].axhline(1.0, color=INK3, lw=1.0, ls=":", zorder=1)
    axes[0].set_yscale("log"); axes[0].set_ylabel("non-reversible ESS / reversible ESS")
    axes[0].set_title("gain — rises without limit, so it cannot select on its own",
                      color=INK, loc="left", fontsize=10.2, pad=6)
    axes[0].legend(loc="upper left", fontsize=7.8)

    axes[1].axhline(cap, color="#b2182b", lw=1.5, ls="--", zorder=4)
    axes[1].annotate(f"admissibility cap = {CAP_MULT:g} $\\times$ the reversible arm's own bias "
                     f"({cap:.3f})", (LEVELS[0], cap), textcoords="offset points",
                     xytext=(2, 6), fontsize=7.6, color="#b2182b")
    axes[1].axhline(float(sub["bias_rev"].median()), color=REV, lw=1.4, zorder=2)
    axes[1].annotate("reversible arm", (LEVELS[-1], float(sub["bias_rev"].median())),
                     textcoords="offset points", xytext=(-52, -12), fontsize=7.6, color=REV)
    axes[1].set_yscale("log"); axes[1].set_ylabel("bias (gold sd units, max over 13 functions)")
    axes[1].set_title("bias — what actually decides", color=INK, loc="left", fontsize=10.2, pad=6)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xticks(LEVELS)
        ax.set_xticklabels([f"{v:g}" for v in LEVELS])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("block strength $s_\\ell$ (other two blocks at zero)")
        ax.grid(True); ax.set_axisbelow(True)
        ax.margins(x=0.12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"Stage A — per-block response, `{pname}` potential", fontsize=12.0, y=1.02,
                 x=0.015, ha="left", color=INK)
    fig.text(0.5, -0.08, _wrap(
        f"TITANIC, ball constraint, eta = {ETA:g}, R = {R_S} coupled chains from a common "
        f"stationary state sharing one Gaussian stream, {NM_S} measured iterations stored every "
        f"{CK}, search seed {SEED_SEARCH}. Each curve scans ONE block's strength with the other "
        f"two held at zero, so it is that block's own response. ESS from Geyer "
        f"initial-positive/monotone IACT per chain per coordinate; the block ratio is the "
        f"geometric mean over its three coordinates x all chains. Bias is measured against a "
        f"reversible gold run at eta/{GOLD_DIV:g} over thirteen test functions (ten coordinates, "
        f"|w|^2, U_0, TRAINING accuracy -- no test labels). The left panel is exactly the trap: "
        f"the gain never stops rising, so `argmax ESS` would pick s = {LEVELS[-1]:g} every time. "
        f"The right panel shows what that costs.", 128),
        ha="center", va="top", fontsize=7.0, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"stageA_{pname}.{ext}"),
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    return fig, cap

def select(sub, cap, value_col="own_ratio"):
    ok = sub[sub["bias_nrev"] <= cap]
    if len(ok) == 0:
        return float(sub.sort_values("s")["s"].iloc[0]), False
    return float(ok.loc[ok[value_col].idxmax(), "s"]), True

def confirm_figure(confirm, pname, outdir):
    """Left: speed-up, admissible cells filled and inadmissible ones hollow with a red edge.
    Right: the accuracy difference, on an axis scaled to ONE test row."""
    sub = confirm[confirm.potential == pname].iloc[::-1].reset_index(drop=True)
    ypos = np.arange(len(sub))
    one_row = 100.0 / DS.n_test
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 0.52 * len(sub) + 2.6),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    ax.axvline(1.0, color=INK3, lw=1.2, zorder=1)
    for yi, r in sub.iterrows():
        lo, hi = (math.exp(math.log(r["overall"]) + z * 2 * r["overall_se"]) for z in (-1, 1))
        col = NREV if r["admissible"] else "#b2182b"
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.4, zorder=2)
        ax.plot(r["overall"], yi, "o" if r["admissible"] else "X", ms=8.5, color=col,
                mfc=col if r["admissible"] else "white", mec=col, mew=1.6, zorder=3)
        if not r["admissible"]:
            ax.annotate("bias inadmissible", (hi, yi), textcoords="offset points",
                        xytext=(8, -2.5), fontsize=7.2, color="#b2182b")
    ax.set_yticks(ypos); ax.set_yticklabels(
        [r["label"] if "(" in r["label"]
         else f"{r['label']}  ({r['s1']:g},{r['s2']:g},{r['s3']:g})"
         for _, r in sub.iterrows()], fontsize=8.0)
    ax.set_xscale("log"); ax.set_xlabel("non-reversible ESS / reversible ESS ($\\pm$ 2 s.e.)")
    ax.set_title("speed-up", color=INK, loc="left", fontsize=10.3, pad=6)
    ax.grid(True, axis="x"); ax.set_axisbelow(True)
    ax.margins(x=0.22)

    # Scale this axis to the ADMISSIBLE rows: one inadmissible cell can sit ten points out and
    # would otherwise squash every row that matters into the zero line. Off-scale rows are pinned
    # at the edge and labelled with their true value, so nothing is hidden.
    ax = axes[1]
    adm = sub[sub["admissible"]]
    se_all = np.where(sub["t_acc"] != 0, np.abs(sub["d_acc_pts"] / sub["t_acc"]), 0.0)
    se_adm = se_all[sub["admissible"].to_numpy()]
    span = float(np.max(np.abs(adm["d_acc_pts"].to_numpy()) + 2 * se_adm)) if len(adm) else 0.05
    half = max(span * 1.35, 0.02)
    ax.axvline(0.0, color=INK3, lw=1.2, zorder=1)
    for yi, r in sub.iterrows():
        se = se_all[yi]
        col = NREV if r["admissible"] else "#b2182b"
        if abs(r["d_acc_pts"]) > half:                       # off scale: pin and label
            xp = math.copysign(half * 0.97, r["d_acc_pts"])
            ax.plot(xp, yi, "X", ms=8.0, color=col, mfc="white", mec=col, mew=1.6, zorder=3,
                    clip_on=False)
            ax.annotate(f"{r['d_acc_pts']:+.2f} pts, off scale", (xp, yi),
                        textcoords="offset points",       # always inward, to stay on the axes
                        xytext=(9 if xp < 0 else -9, -2.5), fontsize=7.2, color=col,
                        ha="left" if xp < 0 else "right")
            continue
        ax.plot([r["d_acc_pts"] - 2 * se, r["d_acc_pts"] + 2 * se], [yi, yi], color=col, lw=1.4)
        ax.plot(r["d_acc_pts"], yi, "o" if r["admissible"] else "X", ms=7.5, color=col,
                mfc=col if r["admissible"] else "white", mec=col, mew=1.5, zorder=3)
    ax.set_xlim(-half, half)
    ax.set_yticks(ypos); ax.set_yticklabels([])
    ax.set_xlabel("test accuracy difference, non-reversible $-$ reversible (pts, $\\pm$ 2 s.e.)")
    ax.set_title("and what it does to accuracy", color=INK, loc="left", fontsize=10.3, pad=6)
    ax.grid(True, axis="x"); ax.set_axisbelow(True)
    ax.annotate(f"the whole axis spans {2 * half / one_row:.2f} of ONE test row "
                f"({one_row:.2f} pts out of {DS.n_test})",
                xy=(0, len(sub) - 0.30), ha="center", fontsize=7.6, color=INK2)
    for a_ in axes:
        a_.set_ylim(-0.7, len(sub) + 0.05)
        for sp in ("top", "right", "left"):
            a_.spines[sp].set_visible(False)

    fig.suptitle(f"Stage C — confirmation on seed {SEED_CONF}, `{pname}` potential",
                 fontsize=12.0, y=1.005, x=0.015, ha="left", color=INK)
    fig.text(0.5, -0.02 - 0.62 / len(sub), _wrap(
        f"R = {R_C} coupled chains, {NM_C} measured iterations, seed {SEED_CONF}, which took no "
        f"part in the Stage A search. Filled green: the non-reversible arm's discretisation bias "
        f"is within {CAP_MULT:g}x the reversible arm's own, the cap declared before the search. "
        f"Hollow red cross: it is not, and the speed-up on that row is bought with error rather "
        f"than earned -- those rows are shown, not hidden, because they are what an unconstrained "
        f"search would have returned. The REVERSED and ROTATED rows carry the identical multiset "
        f"of strengths as `composed s*` in a different order, so any difference between them is "
        f"placement, not total strength. Right panel: the axis spans a fraction of ONE test row "
        f"out of {DS.n_test}, so differences there are resolvable but not numerically meaningful; "
        f"both arms provably share the invariant law, so what is plotted is finite-step bias.",
        126), ha="center", va="top", fontsize=7.0, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"stageC_{pname}.{ext}"),
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    return fig
