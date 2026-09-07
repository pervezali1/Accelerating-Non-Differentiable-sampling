#!/usr/bin/env python3
"""Constrained anchored Langevin on Titanic / MAGIC, and its accuracy learning curve.

Self-contained: NumPy + pandas + matplotlib, no torch and nothing from the ``ands``
package.  Run it and it downloads the data, samples the posterior, and writes
``learning_curve_standalone.png``.

    python examples/learning_curve_standalone.py                # both datasets, ~3 min
    python examples/learning_curve_standalone.py --dataset titanic --walkers 500

Most of that time is the exact reference chain, which exists only to establish the
ceiling; drop --ref-steps if you only want the curve.

WHAT IS BEING SAMPLED

For labels y_i in {-1,+1} and standardised features phi_i, write psi_i = y_i phi_i.  The
Gibbs posterior over classifier weights, restricted to a Euclidean ball, is

    U(w) = tau * mean_i max(0, 1 - w.psi_i) + lam * ||w||_1
    pi(w) ~ exp(-U(w)) * 1{||w||_2 <= R}

Both terms are non-differentiable, and U is only ever *evaluated*.  The sampler works
from a smoothed anchor U0 >= U and the time change exp(Delta), Delta = U - U0:

    x <- P_K( x - eta * exp(Delta) [ (I + J(x)) grad U0(x) - div J(x) ]
              + sqrt(2 eta) exp(Delta/2) xi )

The hinge is an AVERAGE, not a sum.  With a sum the anchor gap grows like n*tau*delta and
exp(Delta) underflows on any real dataset -- the sampler silently freezes.

THE THREE SKEW FIELDS

Projection onto K imposes two conditions on J: the flux must be divergence-free
(invariance), and the skew drift must be tangential at the wall, J(x) nu(x) = 0.

  none    J = 0.
  const   J = s*A, constant.  div J = 0, but J nu != 0 -- violates the wall condition.
  axial   J(x) = (s/R^2)[ ||x||^2 A + x (Ax)^T - (Ax) x^T ].
          Annihilates x identically, so J nu = 0 by construction, but
          div J(x) = -(s/R^2)(d-2) A x, which is NOT zero for d > 2 and must be carried.

THE CURVE

Every chain starts from w = 0, where every margin is exactly zero.  Counting a tie as a
coin flip puts that start at chance, 0.5, and the trace then climbs to whatever the
posterior supports -- about 0.79 here.  That ceiling is not 1: these are linear
classifiers on real, noisy data, and a sampler that climbed past its own target's
accuracy would be reporting a bug rather than a success.
"""

import argparse
import io
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TITANIC_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
MAGIC_URL = ("https://raw.githubusercontent.com/mikeizbicki/datasets/master/"
             "csv/uci/magic04.data")
MAGIC_COLS = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym",
              "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------
def _fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return io.BytesIO(r.read())


def _standardise(X):
    """Zero mean, unit variance, constant columns dropped, intercept prepended."""
    X = np.asarray(X, float)
    sd = X.std(axis=0)
    X = X[:, sd > 1e-12]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return np.column_stack([np.ones(len(X)), X])


def load_titanic():
    df = pd.read_csv(_fetch(TITANIC_URL))
    df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    feats = np.column_stack([
        (df["pclass"] == 1), (df["pclass"] == 2), (df["sex"] == "female"),
        df["age"], df["sibsp"], df["parch"], np.log1p(df["fare"]),
        (df["embarked"] == "C"), (df["embarked"] == "Q")]).astype(float)
    y = np.where(df["survived"].values == 1, 1.0, -1.0)
    return _standardise(feats), y


def load_magic(n_max=2000, seed=0):
    df = pd.read_csv(_fetch(MAGIC_URL), header=None, names=MAGIC_COLS)
    y = np.where(df["class"].values == "g", 1.0, -1.0)
    X = df[MAGIC_COLS[:-1]].values.astype(float)
    for j, nm in enumerate(MAGIC_COLS[:-1]):           # heavy-tailed columns
        if nm in ("fLength", "fWidth", "fSize", "fDist"):
            X[:, j] = np.log(X[:, j] + 1e-6)
    if n_max is not None and n_max < len(df):          # stratified subsample
        rng = np.random.default_rng(seed)
        idx = np.concatenate([
            rng.choice(np.flatnonzero(y == c), int(round(n_max * np.mean(y == c))), replace=False)
            for c in (1.0, -1.0)])
        rng.shuffle(idx)
        X, y = X[idx], y[idx]
    return _standardise(X), y


# ----------------------------------------------------------------------------------
# the target
# ----------------------------------------------------------------------------------
class BallGibbsSVM:
    def __init__(self, Phi, y, tau=100.0, lam=1.0, delta=0.02, R=1.4):
        self.Psi = y[:, None] * Phi                    # margin of w on row i is w.psi_i
        self.n, self.d = self.Psi.shape
        self.tau, self.lam, self.delta, self.R = tau, lam, delta, R

    def anchored_step(self, w):
        """(Delta, grad U0) from a single pass over the margins.

        U, U0 and grad U0 all read the same (N, n) margin matrix, and forming it is the
        whole cost of a step, so they are computed together.
        """
        z = 1.0 - w @ self.Psi.T                       # (N, n)
        root = np.sqrt(z * z + self.delta ** 2)
        rw = np.sqrt(w * w + self.delta ** 2)
        # Delta = U - U0 <= 0, because both smoothings dominate what they replace
        Delta = (self.tau * (np.maximum(z, 0.0).mean(1) - (0.5 * (z + root)).mean(1))
                 + self.lam * (np.abs(w) - rw).sum(1))
        grad = -(self.tau / self.n) * ((0.5 * (1.0 + z / root)) @ self.Psi) + self.lam * w / rw
        return Delta, grad

    def project(self, w):
        nrm = np.linalg.norm(w, axis=1, keepdims=True)
        return w * np.minimum(1.0, self.R / np.maximum(nrm, 1e-300))

    def accuracy_per_draw(self, w):
        """Accuracy of each draw on its own; an exact zero margin counts as a coin flip.

        The tie matters in exactly one place and it is where the curve starts: at w = 0
        every margin is exactly zero.  For any w != 0 on continuous features ties have
        probability zero.
        """
        m = np.atleast_2d(w) @ self.Psi.T
        return ((m > 0) + 0.5 * (m == 0)).mean(axis=1)


# ----------------------------------------------------------------------------------
# skew fields
# ----------------------------------------------------------------------------------
def unit_skew(d):
    """Tridiagonal skew (+1 above the diagonal, -1 below), scaled to unit spectral norm,
    so that ``s`` means the same push for every field and every dimension."""
    A = np.diag(np.ones(d - 1), 1) - np.diag(np.ones(d - 1), -1)
    return A / np.linalg.norm(A, 2)


def skew_apply(kind, x, g, A, s, R):
    """J(x) g, row-wise."""
    if kind == "none":
        return np.zeros_like(g)
    if kind == "const":
        return s * (g @ A.T)
    c = s / R ** 2                                     # axial
    Ax, Ag = x @ A.T, g @ A.T
    r2 = (x * x).sum(1, keepdims=True)
    xg = (x * g).sum(1, keepdims=True)
    Axg = (Ax * g).sum(1, keepdims=True)
    return c * (r2 * Ag + Axg * x - xg * Ax)


def skew_div(kind, x, A, s, R, drop_correction=False):
    """div J, in closed form.  Zero for `none` and `const`; for the axial field
    -(s/R^2)(d-2) A x, which vanishes only at d = 2."""
    if kind != "axial" or drop_correction:
        return np.zeros_like(x)
    return -(s / R ** 2) * (x.shape[1] - 2) * (x @ A.T)


# ----------------------------------------------------------------------------------
# the sampler
# ----------------------------------------------------------------------------------
def run_chain(target, kind="none", s=0.0, eta=3e-4, n_steps=600, N=1000, seed=106,
              drop_correction=False, track_every=10):
    """Projected anchored Langevin from w = 0.  Returns (iterations, accuracy trace).

    The accuracy trace has columns (mean, 5th percentile, 95th percentile) across walkers.
    """
    A = unit_skew(target.d)
    rng = np.random.default_rng(seed)
    x = np.zeros((N, target.d))
    sqrt2eta = np.sqrt(2.0 * eta)

    iters, acc = [], []

    def record(it):
        a = target.accuracy_per_draw(x)
        iters.append(it)
        acc.append([a.mean(), np.quantile(a, 0.05), np.quantile(a, 0.95)])

    record(0)                                          # the w = 0 start, at chance
    for it in range(n_steps):
        Delta, gU0 = target.anchored_step(x)
        drift = np.exp(Delta)[:, None] * (
            gU0 + skew_apply(kind, x, gU0, A, s, target.R)
            - skew_div(kind, x, A, s, target.R, drop_correction))
        noise = np.exp(0.5 * Delta)[:, None] * rng.standard_normal((N, target.d))
        x = target.project(x - eta * drift + sqrt2eta * noise)
        if (it + 1) % track_every == 0:
            record(it + 1)
    return np.array(iters), np.array(acc), x


# ----------------------------------------------------------------------------------
# reference: exact random-walk Metropolis on pi restricted to K
# ----------------------------------------------------------------------------------
def exact_potential(target, w):
    z = 1.0 - w @ target.Psi.T
    return target.tau * np.maximum(z, 0.0).mean(1) + target.lam * np.abs(w).sum(1)


def run_rwm(target, n_chains=800, n_steps=4000, step=None, seed=1):
    """Metropolis against the EXACT non-smooth U, proposals outside K rejected.

    This is reversible for pi*1_K with no projection bias, so it gives the ceiling the
    learning curve is aiming at.  A two-stage run adapts the proposal scale.
    """
    rng = np.random.default_rng(seed)
    d = target.d
    scale = (2.38 / np.sqrt(d)) * 0.15 if step is None else step
    x = np.zeros((n_chains, d))
    Ux = exact_potential(target, x)
    for _ in range(n_steps):
        prop = x + scale * rng.standard_normal((n_chains, d))
        inside = np.linalg.norm(prop, axis=1) <= target.R
        Up = np.where(inside, exact_potential(target, prop), np.inf)
        take = (np.log(rng.random(n_chains)) < (Ux - Up)) & inside
        x = np.where(take[:, None], prop, x)
        Ux = np.where(take, Up, Ux)
    return x


# ----------------------------------------------------------------------------------
def study(name, Phi, y, R, args):
    target = BallGibbsSVM(Phi, y, tau=args.tau, lam=args.lam, delta=args.delta, R=R)
    print("\n=== {}  (n={}, d={}, R={}) ===".format(name, target.n, target.d, R))

    ref = run_rwm(target, n_chains=args.ref_chains, n_steps=args.ref_steps)
    ceiling = target.accuracy_per_draw(ref).mean()
    print("exact posterior accuracy (the ceiling): {:.4f}".format(ceiling))

    methods = [("none", 0.0, False, "J = 0"),
               ("const", args.s, False, "constant $J_a$"),
               ("axial", args.s, False, "state-dependent $J_s$"),
               ("axial", args.s, True, "$J_s$, correction dropped")]
    out = {}
    for kind, s, drop, label in methods:
        it, acc, _ = run_chain(target, kind, s, eta=args.eta, n_steps=args.steps,
                               N=args.walkers, drop_correction=drop)
        out[label] = (it, acc)
        hit = it[np.argmax(acc[:, 0] >= 0.99 * ceiling)] if (acc[:, 0] >= 0.99 * ceiling).any() else None
        print("  {:28s} start {:.3f} -> final {:.4f}   99% of ceiling at iter {}".format(
            label, acc[0, 0], acc[-1, 0], hit))
    return out, ceiling


def plot(results, args):
    fig, axes = plt.subplots(1, len(results), figsize=(7.0 * len(results), 4.8),
                             squeeze=False)
    colors = {"J = 0": "#C44E52", "constant $J_a$": "#4C72B0",
              "state-dependent $J_s$": "#55A868", "$J_s$, correction dropped": "#8172B2"}
    styles = ["-", "-", (0, (5, 2)), (0, (1, 1.6))]
    widths = [3.4, 2.0, 2.0, 2.0]
    for ax, (name, (curves, ceiling)) in zip(axes[0], results.items()):
        for (label, (it, acc)), ls, lw in zip(curves.items(), styles, widths):
            ax.plot(it, acc[:, 0], color=colors[label], linewidth=lw, linestyle=ls,
                    label=label)
            if label == "J = 0":
                ax.fill_between(it, acc[:, 1], acc[:, 2], color=colors[label],
                                alpha=0.15, linewidth=0, label="5-95% across walkers")
        ax.axhline(ceiling, color="k", linestyle="--", linewidth=1.4,
                   label="exact posterior ({:.3f})".format(ceiling))
        ax.axhline(0.5, color="0.45", linestyle=":", linewidth=1.4, label="chance (0.5)")
        ax.set_xlabel("Iterations", fontsize=13)
        ax.set_ylabel("classification accuracy", fontsize=13)
        ax.set_title(name, fontsize=13)
        ax.set_ylim(0.47, ceiling + 0.035)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("Accuracy from the $w = 0$ start", fontsize=15)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print("\nwrote {}".format(os.path.abspath(args.out)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["titanic", "magic", "both"], default="both")
    p.add_argument("--walkers", type=int, default=1000)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--eta", type=float, default=3e-4)
    p.add_argument("--s", type=float, default=4.0, help="skew strength")
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=0.02, help="anchor smoothing scale")
    p.add_argument("--magic-rows", type=int, default=2000)
    p.add_argument("--ref-chains", type=int, default=800)
    p.add_argument("--ref-steps", type=int, default=4000)
    p.add_argument("--out", default="learning_curve_standalone.png")
    args = p.parse_args()

    results = {}
    if args.dataset in ("titanic", "both"):
        Phi, y = load_titanic()
        results["Titanic (survival)"] = study("Titanic", Phi, y, 1.4, args)
    if args.dataset in ("magic", "both"):
        Phi, y = load_magic(args.magic_rows)
        results["MAGIC Gamma Telescope"] = study("MAGIC", Phi, y, 1.9, args)
    plot(results, args)


if __name__ == "__main__":
    main()
