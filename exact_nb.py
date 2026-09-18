"""Exact-gradient anchored Langevin with LASSO-smoothed anchor (the commit-29bbf2f formulation).

AUTO-GENERATED from exact_notebook_source.py by make_notebook.py -- edit that file, not this one.
Contains exactly the library cells of exact_anchored_langevin.ipynb, so the notebook and this module
can never drift apart.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from typing import Dict, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit

try:                       # inline figures in Jupyter; a headless backend as a script
    get_ipython().run_line_magic("matplotlib", "inline")          # type: ignore[name-defined]
except Exception:
    matplotlib.use("Agg")

from nral import (BallGeometry, SmoothedLpGeometry, build_magic_dataset,   # noqa: E402
                  build_titanic_dataset, mean_sd, softplus, _wrap)

D10 = 10                                     # intercept w_0 plus nine features
BLOCKS10 = ((1, 2, 3), (4, 5, 6), (7, 8, 9))  # J acts on w_1..w_9; the intercept is never rotated
SIGMA = 10.0                                 # Gaussian prior sd on the intercept
LASSO_FRAC = 0.01                            # lambda = LASSO_FRAC * n_train
A_LOWER = 0.5                                # target lower bound on a; this fixes delta
SEED_MAIN, SEED_CONF, SEED_HIGH = 3000, 4100, 5200   # three disjoint streams

REV, NREV = "#1f5fbf", "#1a9850"             # validated two-slot categorical palette
INK, INK2, INK3, GRIDC = "#1f2328", "#57606a", "#8c959f", "#e4e6ea"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIDC,
    "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "grid.color": GRIDC, "grid.linewidth": 0.7, "grid.linestyle": "-", "font.size": 9,
})

RESULTS_DIR = os.environ.get("XNB_RESULTS", os.path.join(os.getcwd(), "results", "exact_nb"))
FIGURE_DIR = os.environ.get("XNB_FIGURES", os.path.join(os.getcwd(), "figures", "exact_nb"))
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

QUICK = os.environ.get("XNB_MODE", "full").lower() == "quick"
R_MAIN = 20 if QUICK else 100                # replicates for the four accuracy figures
R_ETA = 20 if QUICK else 150                 # replicates for the eta = 3e-5 study
R_CONF = 60 if QUICK else 1500               # replicates for the section 10 confirmation

PKG_VERSIONS = {"python": sys.version.split()[0], "platform": platform.platform(),
                "numpy": np.__version__, "scipy": scipy.__version__,
                "sklearn": sklearn.__version__, "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__}
print("versions:", PKG_VERSIONS)
print("mode:", "QUICK (a smoke test, NOT the experiment)" if QUICK else "full",
      f"| R_MAIN={R_MAIN}  R_ETA={R_ETA}  R_CONF={R_CONF}")

def design_with_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend the intercept column. Column 0 is w_0; columns 1..9 carry the LASSO."""
    return np.ascontiguousarray(np.column_stack([np.ones(len(X)), X]))

class Potential:
    """U (non-smooth), U_0 (smooth anchor), the exact grad U_0, and a = exp(U - U_0)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, sigma: float, lam: float, delta: float):
        self.X, self.y, self.sigma, self.lam, self.delta = X, y.astype(float), sigma, lam, delta

    def _loglik(self, W):
        z = np.atleast_2d(W) @ self.X.T
        return np.sum(softplus(z) - self.y[None, :] * z, axis=1)

    def U(self, W):
        """The true, NON-differentiable potential."""
        W2 = np.atleast_2d(W)
        out = (self._loglik(W2) + W2[:, 0] ** 2 / (2 * self.sigma ** 2)
               + self.lam * np.sum(np.abs(W2[:, 1:]), axis=1))
        return out[0] if np.ndim(W) == 1 else out

    def U0(self, W):
        """The smooth anchor: |w_j| -> sqrt(w_j^2 + delta^2)."""
        W2 = np.atleast_2d(W)
        out = (self._loglik(W2) + W2[:, 0] ** 2 / (2 * self.sigma ** 2)
               + self.lam * np.sum(np.sqrt(W2[:, 1:] ** 2 + self.delta ** 2), axis=1))
        return out[0] if np.ndim(W) == 1 else out

    def grad_U0(self, W):
        """EXACT gradient of the anchor. No mini-batching anywhere in this notebook."""
        W2 = np.atleast_2d(W)
        g = (expit(W2 @ self.X.T) - self.y[None, :]) @ self.X
        g[:, 0] += W2[:, 0] / self.sigma ** 2
        g[:, 1:] += self.lam * W2[:, 1:] / np.sqrt(W2[:, 1:] ** 2 + self.delta ** 2)
        return g[0] if np.ndim(W) == 1 else g

    def a(self, W):
        """a = exp(U - U_0), exact: it depends only on w, never on the data."""
        W2 = np.atleast_2d(W)
        gap = np.sum(np.sqrt(W2[:, 1:] ** 2 + self.delta ** 2) - np.abs(W2[:, 1:]), axis=1)
        out = np.exp(-self.lam * gap)
        return out[0] if np.ndim(W) == 1 else out

    @property
    def a_lower(self) -> float:
        """exp(-9 lambda delta): the worst case, attained at w_1 = ... = w_9 = 0."""
        return float(np.exp(-self.lam * 9 * self.delta))


def make_potential(ds, lasso_frac: float = LASSO_FRAC, sigma: float = SIGMA,
                   a_lower: float = A_LOWER, lam: Optional[float] = None) -> Potential:
    lam = lasso_frac * ds.n_train if lam is None else lam
    delta = -math.log(a_lower) / (9.0 * lam)          # so that a >= a_lower everywhere
    return Potential(design_with_intercept(ds.X_train), ds.y_train, sigma, lam, delta)

def make_geom(kind: str, *, r2: float = 2.0, p: float = 2.4, eps: float = 0.2, Lam: float = 4.0):
    if kind == "ball":
        return BallGeometry(r2=r2)
    return SmoothedLpGeometry(p=p, eps=eps, Lam=Lam, d=D10)        # g_min = 10 eps^p


def init_unit_ball(rng: np.random.Generator, n_rows: int) -> np.ndarray:
    Z = rng.standard_normal((n_rows, D10))
    V = rng.random(n_rows)
    return (V ** (1.0 / D10))[:, None] * Z / np.linalg.norm(Z, axis=1, keepdims=True)

def expand_scales(scales: Sequence[float]) -> np.ndarray:
    """(s1, s2, s3) -> a length-10 vector, zero on the intercept, s_l repeated over its triple."""
    s = np.asarray(scales, dtype=float)
    assert s.shape == (3,), f"need three block strengths, got {s.shape}"
    v = np.zeros(D10)
    v[1:] = np.repeat(s, 3)
    return v


def J_axes(W: np.ndarray, geom, scales: Sequence[float]) -> np.ndarray:
    e = expand_scales(scales)
    return e * W if isinstance(geom, BallGeometry) else -e * geom.grad_g(W)


def apply_J(W: np.ndarray, V: np.ndarray, geom, scales: Sequence[float]) -> np.ndarray:
    """Matrix-free blockdiag(0, [u_1]_x, [u_2]_x, [u_3]_x) @ V."""
    u = J_axes(W, geom, scales)
    out = np.zeros_like(V)
    for blk in BLOCKS10:
        sl = slice(blk[0], blk[-1] + 1)
        out[..., sl] = np.cross(u[..., sl], V[..., sl])
    return out


def explicit_J(w: np.ndarray, geom, scales: Sequence[float]) -> np.ndarray:
    """Dense 10x10 J(w) -- verification helper only."""
    u = J_axes(np.asarray(w, dtype=float), geom, scales)
    J = np.zeros((D10, D10))
    for blk in BLOCKS10:
        sl = slice(blk[0], blk[-1] + 1)
        a1, a2, a3 = u[blk[0]], u[blk[1]], u[blk[2]]
        J[sl, sl] = np.array([[0.0, -a3, a2], [a3, 0.0, -a1], [-a2, a1, 0.0]])
    return J


def divergence_J(w, geom, scales, h: float = 1e-5) -> np.ndarray:
    """(div J)_i = sum_j d/dw_j J_ij, by central differences on the explicit matrix."""
    w = np.asarray(w, dtype=float)
    div = np.zeros(D10)
    for j in range(D10):
        e = np.zeros(D10); e[j] = h
        div += (explicit_J(w + e, geom, scales) - explicit_J(w - e, geom, scales))[:, j] / (2 * h)
    return div

def run_chain(pot: Potential, geom, *, alpha: int, scales: Sequence[float], eta: float,
              n_iter: int, W0: np.ndarray, seed: int, checkpoint_every: int = 10) -> dict:
    """Vectorised over replicates: W has shape (R, 10). Re-seeding reproduces the identical
    Gaussian stream, which is what makes the two arms a PAIRED comparison."""
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
        G = pot.grad_U0(W)
        a = pot.a(W)
        step = -eta * a[:, None] * G
        if alpha:
            step = step + eta * alpha * a[:, None] * apply_J(W, G, geom, scales)
        prop = W + step + np.sqrt(2.0 * eta * a)[:, None] * rng.standard_normal((R, D10))
        bad = ~np.isfinite(prop).all(axis=1)
        if bad.any():
            n_bad += int(bad.sum()); prop[bad] = W[bad]
        W, was = geom.project(prop)
        proj[k] = np.count_nonzero(was)
    Ws[-1] = W
    assert np.all(geom.feasible(W)), "final states left K"
    return dict(checkpoints=np.asarray(ck), W=Ws, runtime=time.perf_counter() - t0,
                projection_rate=float(proj.sum() / max(1, n_iter * R)), n_nonfinite=n_bad)


def accuracy(X: np.ndarray, y: np.ndarray, Ws: np.ndarray) -> np.ndarray:
    """(n_ck, R, 10) -> (n_ck, R). Predictions use the CURRENT w, on ALL rows of the split."""
    Xi = design_with_intercept(X)
    yb = y.astype(bool)
    out = np.empty(Ws.shape[:2])
    for i in range(Ws.shape[0]):
        out[i] = np.mean(((Ws[i] @ Xi.T) >= 0.0) == yb[None, :], axis=1)
    return out


def paired(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float, float]:
    """Mean, standard error, t and win-rate of the paired difference a - b."""
    d = a - b
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) + 1e-300
    return float(d.mean()), float(se), float(d.mean() / se), float((d > 0).mean())


def run_pair(ds, pot, geom, *, eta: float, scales: Sequence[float], n_iter: int, R: int,
             seed: int) -> dict:
    """Both arms from the same W0 and the same noise stream."""
    W0 = init_unit_ball(np.random.default_rng(seed + 1), R)
    out = {}
    for tag, alpha in (("rev", 0), ("nrev", 1)):
        r = run_chain(pot, geom, alpha=alpha, scales=scales, eta=eta, n_iter=n_iter,
                      W0=W0, seed=seed)
        out[tag] = dict(x=np.asarray(r["checkpoints"], dtype=float),
                        tr=accuracy(ds.X_train, ds.y_train, r["W"]),
                        te=accuracy(ds.X_test, ds.y_test, r["W"]),
                        U=pot.U(r["W"][-1]), a=pot.a(r["W"][-1]),
                        proj=r["projection_rate"], wall=r["runtime"],
                        nonfinite=r["n_nonfinite"])
    return out

EXPERIMENTS = (
    dict(key="magic_ball",   dataset="magic",   geometry="ball", n_iter=1000, eta=1e-6, eps=0.20),
    dict(key="magic_lp",     dataset="magic",   geometry="lp",   n_iter=1000, eta=1e-6, eps=0.20),
    dict(key="titanic_ball", dataset="titanic", geometry="ball", n_iter=1500, eta=1e-5, eps=0.18),
    dict(key="titanic_lp",   dataset="titanic", geometry="lp",   n_iter=2000, eta=1e-5, eps=0.18),
)


def accuracy_figure(spec, arms, pot, ds, scales, eta, R, outdir, extra_note="", tag=""):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, key, name in zip(axes, ("tr", "te"), ("training", "test")):
        for t, col, lab in (("rev", REV, "Reversible anchored Langevin"),
                            ("nrev", NREV, "Non-reversible anchored Langevin")):
            x, A = arms[t]["x"], arms[t][key]
            mu, sd = mean_sd(A, axis=1)
            ax.fill_between(x, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=col,
                            alpha=0.15, linewidth=0)
            ax.plot(x, mu, color=col, linewidth=2.0, label=lab)
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.9); ax.set_axisbelow(True)
        gname = "ball" if spec["geometry"] == "ball" else "smoothed $\\ell_p$ ($p=2.4$)"
        ax.set_title(f"{spec['dataset'].upper()}, {gname} — {name} accuracy", color=INK, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        i0 = int(0.4 * len(x))
        ins = ax.inset_axes([0.46, 0.13, 0.50, 0.40])
        lo, hi = [], []
        for t, col in (("rev", REV), ("nrev", NREV)):
            mu, sd = mean_sd(arms[t][key], axis=1)
            se = sd / np.sqrt(arms[t][key].shape[1])
            ins.fill_between(x[i0:], (mu - se)[i0:], (mu + se)[i0:], color=col, alpha=0.22,
                             linewidth=0)
            ins.plot(x[i0:], mu[i0:], color=col, linewidth=1.5)
            lo.append((mu - se)[i0:].min()); hi.append((mu + se)[i0:].max())
        pad = 0.18 * (max(hi) - min(lo) + 1e-9)
        ins.set_ylim(min(lo) - pad, max(hi) + pad); ins.set_xlim(x[i0], x[-1])
        ins.tick_params(labelsize=6.5); ins.grid(True, alpha=0.9)
        ins.set_title("zoom (mean $\\pm$ 1 s.e.)", fontsize=6.8, pad=2, color=INK2)
        for sp in ("top", "right"):
            ins.spines[sp].set_visible(False)
    axes[0].legend(loc="center right", fontsize=8.4, bbox_to_anchor=(1.02, 0.70))
    d_tr = paired(arms["nrev"]["tr"][-1], arms["rev"]["tr"][-1])
    d_te = paired(arms["nrev"]["te"][-1], arms["rev"]["te"][-1])
    gtxt = ("ball ||w||^2 <= 2 (radius sqrt(2))" if spec["geometry"] == "ball" else
            f"smoothed l_p, g(w) = sum_i (w_i^2 + eps^2)^(p/2) <= Lambda, p = 2.4, "
            f"eps = {spec['eps']:g}, Lambda = 4")
    fig.suptitle(f"{spec['key']}{tag}: exact gradient, LASSO anchor", fontsize=12.0, y=1.005,
                 x=0.02, ha="left", color=INK)
    fig.text(0.5, -0.05, _wrap(
        f"{spec['dataset'].upper()} -- {gtxt}. d = 10 (intercept w_0 + nine features), no "
        f"mini-batching: the EXACT gradient is used at every step. U = sum softplus - y x.w + "
        f"w_0^2/(2 sigma^2) + lambda sum_{{j>=1}} |w_j|; U_0 replaces |w_j| by "
        f"sqrt(w_j^2 + delta^2); a = exp(U - U_0). sigma = {pot.sigma:g}, "
        f"lambda = {pot.lam:.4g} = {LASSO_FRAC:g} n_train, delta = {pot.delta:.4g}, so "
        f"a in [{pot.a_lower:.2f}, 1]. Block strengths s = ({', '.join(f'{v:g}' for v in scales)}) "
        f"on w_1..w_9 only. Step eta = {eta:g}. R = {R} coupled replicates (same W0, same Gaussian "
        f"stream) on one fixed stratified 80/20 split (n_train = {ds.n_train}, "
        f"n_test = {ds.n_test}). Paired final-iterate difference, non-reversible minus reversible: "
        f"training {d_tr[0]:+.4f} (t = {d_tr[2]:+.2f}), test {d_te[0]:+.4f} (t = {d_te[2]:+.2f}). "
        f"Bands: mean +- 1 SD (ddof = 1) across replicates -- repeat-run variability at a fixed "
        f"split, not a confidence or credible interval. {extra_note}", 126),
        ha="center", va="top", fontsize=7.0, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    name = f"{spec['key']}{tag}_exact_nb"
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    return fig, d_tr, d_te

ETA3 = 3e-5
S_GRID3 = (0.0, 0.25, 1.0, 2.0, 5.0)
ETA3_EXPS = (EXPERIMENTS[2], EXPERIMENTS[3])          # titanic_ball, titanic_lp


def sweep_figure(table: pd.DataFrame, key: str, outdir: str):
    """Paired difference (non-reversible minus reversible) against block strength, with 2 s.e.
    bars. The s = 0 point is the control: it is exactly zero by construction."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharex=True)
    for ax, which, name in zip(axes, ("train", "test"), ("training", "test")):
        sub = table[table["experiment"] == key].sort_values("s")
        d = sub[f"d_{which}"].to_numpy() * 100.0
        se = sub[f"se_{which}"].to_numpy() * 100.0
        s = sub["s"].to_numpy()
        xs = np.arange(len(s))
        ax.axhline(0.0, color=REV, lw=1.4, zorder=2,
                   label="Reversible anchored Langevin (reference)")
        ax.errorbar(xs, d, yerr=2.0 * se, fmt="s", ms=6.5, color=NREV, ecolor=NREV,
                    elinewidth=1.4, capsize=4, zorder=3, mfc=NREV, mec="white", mew=0.9,
                    label="Non-reversible, $\\pm$ 2 s.e. (paired)")
        for xi, di, sei in zip(xs, d, se):
            tip = di + (2.0 * sei if di >= 0 else -2.0 * sei)
            ax.annotate(f"{di:+.3f}", (xi, tip), textcoords="offset points",
                        xytext=(0, 7 if di >= 0 else -14), ha="center", fontsize=7.2,
                        color=INK2)
        lo = float(np.min(d - 2.0 * se)); hi = float(np.max(d + 2.0 * se))
        pad = 0.30 * (hi - lo + 1e-12)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(xs); ax.set_xticklabels([f"{v:g}" for v in s])
        ax.set_xlabel("block strength $s$ (all three blocks)")
        ax.set_ylabel(f"{name} accuracy difference (percentage points)")
        ax.set_title(f"{name} accuracy", fontsize=10.5, color=INK, pad=6, loc="left")
        ax.grid(True, axis="y"); ax.set_axisbelow(True)
        ax.margins(x=0.14)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(loc="best", fontsize=8.2)
    fig.suptitle(f"{key}: block-strength sweep at the specified $\\eta = 3\\times10^{{-5}}$",
                 fontsize=12.0, y=1.02, x=0.02, ha="left", color=INK)
    fig.text(0.5, -0.09, _wrap(
        f"Paired final-iterate difference, non-reversible minus reversible, in percentage points, "
        f"over R = {int(table[table['experiment'] == key]['R'].iloc[0])} coupled replicates sharing "
        f"one W0 and one Gaussian stream. eta = 3e-5 was SPECIFIED, not selected; s was swept, so "
        f"all five strengths are shown. The s = 0 point is the control -- with J = 0 the update is "
        f"the reversible update, so the difference is exactly 0.000 and the error bar has zero "
        f"width. Bars are +- 2 paired standard errors; with four live strengths per panel, |t| > "
        f"2.5 is the honest bar. Seed {SEED_CONF} throughout.", 124),
        ha="center", va="top", fontsize=7.0, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(outdir, f"{key}_eta3e-5_sweep.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{key}_eta3e-5_sweep.pdf"), bbox_inches="tight")
    return fig

def time_average(arms: dict, key: str, frac: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Per-replicate accuracy averaged over checkpoints in the last `1 - frac` of the run."""
    out = []
    for t in ("rev", "nrev"):
        x, A = arms[t]["x"], arms[t][key]
        m = x >= frac * x[-1]
        out.append(A[m].mean(axis=0))
    return out[0], out[1]

CONF_SPEC = EXPERIMENTS[3]                 # titanic_lp
CONF_S = (2.0, 2.0, 2.0)
