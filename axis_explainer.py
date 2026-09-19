"""Why the accuracy figures look like they start at 0.4, and what the [0, 1] axis is for.

Three things are separate and get conflated:

1. The y-axis is [0, 1], NOT [0.4, 1]. Accuracy is a proportion and the full axis is drawn on
   purpose, so that differences of a tenth of a percentage point are not made to look large. Every
   accuracy panel carries a zoom inset for the actual resolution.

2. The shaded band is mean +- 1 SD ACROSS REPLICATES. At iteration 0 the mean is 0.500 and the SD
   is ~0.11, so the band bottom is ~0.39. That is the "0.4" -- it is the bottom of a one-sigma
   band, not a floor on the data. Roughly a fifth to a quarter of replicates START below 0.40 and
   the band simply does not draw them.

3. Nothing forbids low accuracy. Both constraint sets are symmetric (w in K <=> -w in K) and
   acc(w) + acc(-w) = 1 exactly, so every high-accuracy state the sampler reaches has an equally
   feasible mirror with accuracy 1 - acc. The sampler never goes there because U does: at the
   final states U(-w) - U(w) ~ 400, a relative density of e^-400.

Writes figures/axis_explainer.png / .pdf.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

import exact_nb as X
from nral import _wrap, build_magic_dataset, build_titanic_dataset, mean_sd

OUT = os.path.join(os.getcwd(), "figures")
N_RANDOM = 200_000
SPEC = X.EXPERIMENTS[3]            # titanic_lp, eta = 1e-5, 2000 iterations
R = 100


def chunked_accuracy(Xd, y, W, chunk=2000):
    return np.concatenate([X.accuracy(Xd, y, W[i:i + chunk][None])[0]
                           for i in range(0, len(W), chunk)])


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    ds = {"magic": build_magic_dataset(), "titanic": build_titanic_dataset()}
    pot = X.make_potential(ds["titanic"])

    # (1) accuracy of a random initial state, and the exact antisymmetry
    init, anti = {}, {}
    for name, d in ds.items():
        W = X.init_unit_ball(np.random.default_rng(11), N_RANDOM)
        a = chunked_accuracy(d.X_train, d.y_train, W)
        am = chunked_accuracy(d.X_train, d.y_train, -W)
        init[name] = a
        anti[name] = float(np.abs(a + am - 1.0).max())
        print(f"[{name}] random w: mean={a.mean():.4f} sd={a.std():.4f} "
              f"min={a.min():.4f}  below 0.40: {(a < 0.40).mean():.1%}  "
              f"max|acc(w)+acc(-w)-1| = {anti[name]:.1e}")

    # (2) one full experiment, keeping the per-replicate envelope
    d = ds["titanic"]
    geom = X.make_geom(SPEC["geometry"], eps=SPEC["eps"])
    W0 = X.init_unit_ball(np.random.default_rng(X.SEED_MAIN + 1), R)
    r = X.run_chain(pot, geom, alpha=0, scales=(5., 5., 5.), eta=SPEC["eta"],
                    n_iter=SPEC["n_iter"], W0=W0, seed=X.SEED_MAIN)
    x = np.asarray(r["checkpoints"], dtype=float)
    A = X.accuracy(d.X_test, d.y_test, r["W"])                # (n_ck, R)
    Wf = r["W"][-1]
    accf, accm = (X.accuracy(d.X_test, d.y_test, Wf[None])[0],
                  X.accuracy(d.X_test, d.y_test, (-Wf)[None])[0])
    Uf, Um = pot.U(Wf), pot.U(-Wf)
    feas = bool(np.all(geom.feasible(Wf)) and np.all(geom.feasible(-Wf)))
    dU = float((Um - Uf).mean())
    print(f"\nfinal states: acc(w) {accf.mean():.4f} / acc(-w) {accm.mean():.4f}; "
          f"both feasible: {feas}; mean U(-w)-U(w) = {dU:.1f}")
    print(f"observed test accuracy over all replicates and checkpoints: "
          f"[{A.min():.4f}, {A.max():.4f}]")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

    # --- panel A: the one-sigma band is not a floor ---
    ax = axes[0]
    a = init["titanic"]
    mu, sd = a.mean(), a.std(ddof=1)
    ax.hist(a, bins=90, color=X.INK3)
    ax.axvspan(mu - sd, mu + sd, color=X.REV, alpha=0.16, zorder=0)
    ax.axvline(mu, color=X.REV, lw=1.8, zorder=4)
    ax.axvline(0.40, color="#b2182b", lw=1.6, ls="--", zorder=4)
    frac = (a < 0.40).mean()
    top = ax.get_ylim()[1]
    ax.set_ylim(0.0, top * 2.05)                      # headroom, so nothing overlaps the bars
    ax.text(0.5, top * 1.82, f"shaded strip = mean $\\pm$ 1 SD, lower edge {mu - sd:.3f}\n"
            f"— this is the apparent \"0.4\" in every accuracy figure",
            ha="center", va="top", fontsize=8.4, color=X.REV)
    ax.annotate(f"{frac:.0%} of replicates start below 0.40\n— drawn by no figure in the study",
                xy=(0.40, top * 0.99), xytext=(0.02, top * 1.42),
                fontsize=8.2, color="#b2182b",
                arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.2))
    ax.annotate(f"mean = {mu:.4f}\n(exactly 1/2, panel C)", xy=(mu, top * 1.03),
                xytext=(0.60, top * 1.42), fontsize=8.2, color=X.REV,
                arrowprops=dict(arrowstyle="->", color=X.REV, lw=1.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(f"TITANIC training accuracy at a random $w_0$ ({N_RANDOM:,} draws)")
    ax.set_ylabel("random draws")
    ax.set_title("A. The band bottom is $\\mu-\\sigma$, not a floor", color=X.INK, loc="left",
                 fontsize=10.5, pad=6)
    ax.grid(True, axis="y"); ax.set_axisbelow(True)

    # --- panel B: the same run drawn both ways ---
    ax = axes[1]
    mu_t, sd_t = mean_sd(A, axis=1)
    ax.fill_between(x, A.min(axis=1), A.max(axis=1), color=X.NREV, alpha=0.16, linewidth=0,
                    label="min–max across the 100 replicates")
    ax.fill_between(x, np.clip(mu_t - sd_t, 0, 1), np.clip(mu_t + sd_t, 0, 1), color=X.REV,
                    alpha=0.22, linewidth=0, label="mean $\\pm$ 1 SD — what the figures draw")
    ax.plot(x, mu_t, color=X.REV, lw=2.0, label="mean")
    ax.axhline(0.40, color="#b2182b", lw=1.4, ls="--", zorder=4)
    ax.annotate(f"lowest state visited anywhere: {A.min():.3f}",
                xy=(x[np.argmin(A.min(axis=1))], A.min()), xytext=(430, 0.13), fontsize=8.2,
                color=X.INK2, arrowprops=dict(arrowstyle="->", color=X.INK2, lw=1.2))
    ax.set_ylim(0.0, 1.0); ax.set_xlim(x.min(), x.max())
    ax.set_xlabel("Iterations"); ax.set_ylabel("test accuracy")
    ax.set_title("B. titanic_lp, the run behind the band", color=X.INK, loc="left",
                 fontsize=10.5, pad=6)
    ax.legend(loc="upper left", fontsize=7.6)
    ax.grid(True); ax.set_axisbelow(True)

    # --- panel C: the mirror state is feasible; U is what forbids it ---
    ax = axes[2]
    ax.scatter(accf, Uf, s=26, color=X.NREV, marker="s", edgecolor="white", linewidth=0.6,
               zorder=3, label="final states $w$ (visited)")
    ax.scatter(accm, Um, s=26, color="#b2182b", marker="o", edgecolor="white", linewidth=0.6,
               zorder=3, label="mirror states $-w$ (feasible, never visited)")
    for i in range(0, R, 4):
        ax.plot([accf[i], accm[i]], [Uf[i], Um[i]], color=X.GRIDC, lw=0.7, zorder=1)
    ax.axvline(0.5, color=X.INK3, lw=1.2, ls=":", zorder=2)
    ax.annotate(f"$\\Delta U \\approx {dU:.0f}$\nrelative density $e^{{-\\Delta U}} \\approx 10^{{-177}}$",
                xy=(0.5, 0.5 * (Uf.mean() + Um.mean())), xytext=(0.5, 0.5 * (Uf.mean() + Um.mean())),
                fontsize=8.6, color=X.INK, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=X.GRIDC, lw=0.8))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(None, Um.max() + 0.42 * (Um.max() - Uf.min()))
    ax.set_xlabel("test accuracy"); ax.set_ylabel("potential $U(w)$")
    ax.set_title("C. The constraint permits it; the potential forbids it", color=X.INK,
                 loc="left", fontsize=10.5, pad=6)
    ax.legend(loc="upper right", fontsize=7.6)
    ax.grid(True); ax.set_axisbelow(True)

    for a_ in axes:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)

    fig.suptitle("Why the accuracy curves appear to live above 0.4", fontsize=12.5, y=1.02,
                 x=0.015, ha="left", color=X.INK)
    fig.text(0.5, -0.10, _wrap(
        f"The y-axis in every accuracy panel of the study is [0, 1], not [0.4, 1] -- accuracy is a "
        f"proportion and the full axis is drawn deliberately, so that differences of a tenth of a "
        f"percentage point are not visually inflated; each panel carries a zoom inset for the real "
        f"resolution. (A) Accuracy of a random initial state on TITANIC over {N_RANDOM:,} draws "
        f"uniform in the unit ball: mean {mu:.4f}, SD {sd:.4f}, minimum {a.min():.4f}. The shaded "
        f"strip is mean +- 1 SD, whose lower edge is {mu - sd:.3f} -- that is the apparent 0.4. "
        f"{frac:.0%} of draws fall below 0.40 and no figure in the study draws them. (B) The same "
        f"reversible run whose band appears in the main figures, with the full min-max envelope "
        f"added: the lowest state any replicate visits is {A.min():.3f}, far below the band. "
        f"(C) Both constraint sets are symmetric in w, and acc(w) + acc(-w) = 1 EXACTLY -- checked "
        f"here to {max(anti.values()):.0e} over {N_RANDOM:,} draws on each dataset, because a "
        f"linear rule and its negation disagree on every point with x.w != 0. So each final state "
        f"has an equally feasible mirror at accuracy {accm.mean():.3f}. The geometry does not rule "
        f"the low-accuracy half out; the potential does, by {dU:.0f} units of U. The mean at "
        f"iteration 0 is exactly 1/2 for the same reason: the initial law is symmetric under "
        f"w -> -w, so accuracy pairs up around 1/2 whatever the data look like.", 150),
        ha="center", va="top", fontsize=7.2, color=X.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "axis_explainer.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "axis_explainer.pdf"), bbox_inches="tight")
    print(f"\n-> {os.path.join(OUT, 'axis_explainer.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
