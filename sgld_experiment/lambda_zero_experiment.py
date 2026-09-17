"""lambda_lasso = 0: the anchor is vacuous, so this isolates the effect of J.

At ``lambda_lasso = 0`` the non-differentiable part ``g = lambda * sum|w_j|``
vanishes, ``U = f`` is smooth, ``a(w) = exp(U - U0) == 1`` identically, and the
update collapses **exactly** to plain projected Langevin,

    x_{k+1} = Pi_K[ x_k - eta grad_f(x_k) + eta alpha J_s(x_k) grad_f(x_k)
                    + sqrt(2 eta) xi_{k+1} ].

That makes this a control: any difference between the reversible and
non-reversible runs here is attributable to ``J`` alone, with no anchoring
involved.  Comparing the result against the lambda = 10 run answers "is the
benefit coming from J or from the anchor?".

    python lambda_zero_experiment.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anchored_lasso as lasso
import anchored_sgld as nral

OUTPUT_DIR = "results_lambda0"
BURN_CHECKPOINTS = 20
STYLE = {"Reversible anchored Langevin":     {"color": "#0173B2", "ls": "--", "lw": 2.0},
         "Non-reversible anchored Langevin": {"color": "#CC3311", "ls": "-",  "lw": 2.6}}


def ergodic_variance(run) -> float:
    return float(run.w[BURN_CHECKPOINTS:].mean(axis=0).var(axis=0, ddof=1).sum())


def figure(runs, cfg, geometry, tag, quick):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for ax, which, title in (
        (axes[0], "train_accuracy", f"Training accuracy  (n = {cfg.n_train})"),
        (axes[1], "test_accuracy",  f"Test accuracy  (n = {cfg.n_test})"),
    ):
        plateau = []
        for name, run in runs.items():
            st = STYLE[name]; mean, sd = run.mean_std(which)
            ax.fill_between(run.checkpoints, np.clip(mean - sd, 0, 1),
                            np.clip(mean + sd, 0, 1), color=st["color"], alpha=0.15, lw=0)
            ax.plot(run.checkpoints, mean, color=st["color"], ls=st["ls"], lw=st["lw"],
                    label=name)
            plateau.append(mean[len(run.checkpoints) // 4:])
        plateau = np.concatenate(plateau); span = plateau.max() - plateau.min()
        ax.set_ylim(plateau.min() - 4 * span - 0.004, plateau.max() + 2 * span + 0.004)
        ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy")
        ax.set_title(title, fontsize=11); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle(f"lambda_lasso = 0 (anchor vacuous, a == 1) — {geometry.name}"
                 + ("  [QUICK MODE]" if quick else ""), fontsize=13)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.text(0.5, 0.015,
        f"d = {cfg.d} (intercept + 8 slopes);  constraint: {geometry.name}, threshold "
        f"{geometry.threshold:.4g};  lambda_lasso = 0 so U = f is SMOOTH and a(w) = 1 "
        f"identically -- the update is plain projected Langevin;\n"
        f"EXACT gradient;  eta = {cfg.eta:g};  R = {cfg.n_repeats};  iterations = "
        f"{cfg.n_iterations};  block strengths s = "
        f"{tuple(float(x) for x in cfg.scales)};  alpha = 0 vs 1.\n"
        "Lines are the across-replicate mean; bands are mean +/- 1 sample sd (ddof = 1) of "
        "single-iterate accuracy -- repeat-run variability, NOT confidence or credible "
        "intervals.  Axis windowed on the plateau, not [0, 1].",
        ha="center", fontsize=7.3)
    paths = []
    for ext, kw in ((".png", {"dpi": 300}), (".pdf", {})):
        path = os.path.join(OUTPUT_DIR, f"accuracy_{tag}{ext}")
        fig.savefig(path, bbox_inches="tight", **kw); paths.append(path)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base = lasso.LassoConfig(
        lambda_lasso=0.0,
        n_repeats=20 if args.quick else 100,
        n_iterations=300 if args.quick else 1000,
        block_scales=(3.0,) * 3,
    )
    dataset = lasso.make_dataset(base)
    target = lasso.LassoTarget(dataset.X_train, dataset.y_train, 0.0,
                               base.sigma_intercept, base.delta_anchor)
    ball = nral.BallGeometry(base.d)
    l1 = nral.L1SmoothBallGeometry(base.d, base.epsilon, base.l1_radius)

    # --- the degenerate-limit identity, asserted rather than assumed ---
    streams = lasso.make_streams(base, l1)
    anchored = lasso.run_chain(dataset, target, l1, streams, base,
                               method="anchored", alpha=0.0)
    w = streams.w_init.copy()
    for k in range(base.n_iterations):
        w = l1.project(w - base.eta * target.grad_f(w)
                       + np.sqrt(2 * base.eta) * streams.noise[:, k, :]).beta
    identity_gap = float(np.abs(anchored.w[-1] - w).max())
    print(f"anchor vacuous: max |anchored chain - plain projected Langevin| = "
          f"{identity_gap:.3e}")
    print(f"a(w) identically 1: {bool((target.a(anchored.w[-1]) == 1.0).all())}")
    assert identity_gap == 0.0
    assert float(np.abs(target.U(anchored.w[-1]) - target.U0(anchored.w[-1])).max()) == 0.0

    s_grid = (1.0, 3.0, 10.0) if args.quick else (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
    rows, figure_paths = [], []
    print(f"\n{'s':>5} {'geometry':<18} {'var NR/REV':>11} {'acc diff':>11} {'t':>6} {'proj':>6}")
    for s in s_grid:
        for geometry in (ball, l1):
            runs = lasso.run_both(dataset, target, geometry,
                                  replace(base, block_scales=(s,) * 3), verbose=False)
            rev = runs["Reversible anchored Langevin"]
            nr = runs["Non-reversible anchored Langevin"]
            diff, se, t = lasso.paired_difference(nr.test_accuracy[-1], rev.test_accuracy[-1])
            rows.append({"geometry": geometry.name, "s": s,
                         "ergodic_variance_ratio": ergodic_variance(nr) / ergodic_variance(rev),
                         "rev_test_acc": rev.test_accuracy[-1].mean(),
                         "nr_test_acc": nr.test_accuracy[-1].mean(),
                         "paired_accuracy_diff": diff, "accuracy_t": t,
                         "projection_rate": nr.projection_rate})
            print(f"{s:5.1f} {geometry.name:<18} {rows[-1]['ergodic_variance_ratio']:11.4f} "
                  f"{diff:+11.5f} {t:6.2f} {nr.projection_rate:6.3f}", flush=True)
            if s == base.scales[0]:
                figure_paths += figure(runs, base, geometry,
                                       "ball" if geometry is ball else "l1", args.quick)
    sweep = pd.DataFrame(rows)

    summary = {"lambda_lasso": 0.0, "anchor_identity_gap": identity_gap}
    held_rows = []
    for geometry in (l1, ball):
        sub = sweep[sweep.geometry == geometry.name]
        best = float(sub.loc[sub.ergodic_variance_ratio.idxmin(), "s"])
        print(f"\n{geometry.name}: best s = {best}; held-out confirmation")
        ratios = []
        for offset in ((101,) if args.quick else (101, 202, 303, 404)):
            runs = lasso.run_both(dataset, target, geometry,
                                  replace(base, block_scales=(best,) * 3),
                                  seed_offset=offset, verbose=False)
            ratio = (ergodic_variance(runs["Non-reversible anchored Langevin"])
                     / ergodic_variance(runs["Reversible anchored Langevin"]))
            ratios.append(ratio)
            held_rows.append({"geometry": geometry.name, "seed_offset": offset,
                              "ergodic_variance_ratio": ratio})
            print(f"   offset {offset}: ratio {ratio:.4f}", flush=True)
        summary[geometry.name] = {
            "selected_s": best, "held_out_mean_ratio": float(np.mean(ratios)),
            "held_out_all_below_one": bool(all(x < 1 for x in ratios)),
            "variance_reduction_percent": 100.0 * (1.0 - float(np.mean(ratios))),
        }
        print(f"   mean {np.mean(ratios):.4f} "
              f"({100*(1-np.mean(ratios)):.1f}% reduction); "
              f"all below 1: {all(x < 1 for x in ratios)}")

    sweep.to_csv(os.path.join(OUTPUT_DIR, "sweep.csv"), index=False)
    pd.DataFrame(held_rows).to_csv(os.path.join(OUTPUT_DIR, "held_out.csv"), index=False)
    summary["figures"] = figure_paths
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print("\nwrote:", *sorted(os.listdir(OUTPUT_DIR)), sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
