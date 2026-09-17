"""Anchored vs non-reversible anchored Langevin at a chosen ``lambda_lasso``.

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
    fig.suptitle(f"lambda_lasso = {cfg.lambda_lasso:g} — {geometry.name}"
                 + ("  [QUICK MODE]" if quick else ""), fontsize=13)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.text(0.5, 0.015,
        f"d = {cfg.d} (intercept + 8 slopes);  constraint: {geometry.name}, threshold "
        f"{geometry.threshold:.4g};  lambda_lasso = {cfg.lambda_lasso:g}"
        + (" so U = f is SMOOTH and a(w) = 1 identically -- the update is plain projected "
           "Langevin;\n" if cfg.lambda_lasso == 0 else
           f" with anchor delta = {cfg.delta_anchor}, so a(w) in "
           f"[{np.exp(-(cfg.d-1)*cfg.lambda_lasso*cfg.delta_anchor):.4f}, 1];\n") +
        f"EXACT gradient;  eta = {cfg.eta:g};  R = {cfg.n_repeats};  iterations = "
        f"{cfg.n_iterations};  block strengths s = "
        f"{tuple(float(x) for x in cfg.scales)};  alpha = 0 vs 1.\n"
        "Lines are the across-replicate mean; bands are mean +/- 1 sample sd (ddof = 1) of "
        "single-iterate accuracy -- repeat-run variability, NOT confidence or credible "
        "intervals.  Axis windowed on the plateau, not [0, 1].",
        ha="center", fontsize=7.3)
    paths = []
    for ext, kw in ((".png", {"dpi": 300}), (".pdf", {})):
        path = os.path.join(globals()["OUTPUT_DIR"], f"accuracy_{tag}{ext}")
        fig.savefig(path, bbox_inches="tight", **kw); paths.append(path)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--lambda-lasso", dest="lambda_lasso", type=float, default=0.0,
                        help="0 makes the anchor vacuous; larger values make it bite")
    args = parser.parse_args()
    output_dir = f"results_lambda{args.lambda_lasso:g}".replace(".", "p")
    globals()["OUTPUT_DIR"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    base = lasso.LassoConfig(
        lambda_lasso=args.lambda_lasso,
        n_repeats=20 if args.quick else 100,
        n_iterations=300 if args.quick else 1000,
        block_scales=(3.0,) * 3,
    )
    dataset = lasso.make_dataset(base)
    target = lasso.LassoTarget(dataset.X_train, dataset.y_train, args.lambda_lasso,
                               base.sigma_intercept, base.delta_anchor)
    ball = nral.BallGeometry(base.d)
    l1 = nral.L1SmoothBallGeometry(base.d, base.epsilon, base.l1_radius)

    # --- anchor diagnostics: how much work is the anchor actually doing? ---
    streams = lasso.make_streams(base, l1)
    anchored = lasso.run_chain(dataset, target, l1, streams, base,
                               method="anchored", alpha=0.0)
    w = streams.w_init.copy()
    for k in range(base.n_iterations):
        w = l1.project(w - base.eta * target.grad_f(w)
                       + np.sqrt(2 * base.eta) * streams.noise[:, k, :]).beta
    identity_gap = float(np.abs(anchored.w[-1] - w).max())
    states = anchored.w[BURN_CHECKPOINTS:].reshape(-1, base.d)[::37]
    a_values = target.a(states)
    print(f"lambda_lasso = {args.lambda_lasso:g}")
    print(f"  a bounds [exp(-8*lambda*delta), 1]     = [{target.a_lower_bound:.6f}, 1]")
    print(f"  a observed along the chain             = "
          f"[{a_values.min():.6f}, {a_values.max():.6f}]")
    print(f"  max |U - U0| along the chain           = "
          f"{float(np.abs(target.U(states) - target.U0(states)).max()):.6f}")
    print(f"  Lipschitz bound L                      = {target.lipschitz_constant():.1f}"
          f"   (eta*L = {base.eta*target.lipschitz_constant():.4f})")
    print(f"  max |chain - plain projected Langevin| = {identity_gap:.3e}   "
          f"{'(anchor vacuous)' if args.lambda_lasso == 0 else '(anchor active)'}")
    if args.lambda_lasso == 0.0:
        assert identity_gap == 0.0, "at lambda = 0 the anchor must disappear exactly"
        assert float(np.abs(target.U(states) - target.U0(states)).max()) == 0.0
    else:
        assert identity_gap > 0.0, "a non-zero lambda must change the chain"

    # --- convergence check -------------------------------------------------
    # A strong anchor makes a(w) small, which shrinks the effective step eta*a
    # and can leave the chain far from stationarity within the iteration budget.
    # The ergodic-average statistic below is then measuring the transient, not
    # an asymptotic variance, so this must be checked rather than assumed.
    mean_accuracy, _ = anchored.mean_std("test_accuracy")
    half = len(anchored.checkpoints) // 2
    quarter = half + (len(anchored.checkpoints) - half) // 2
    drift = float(np.linalg.norm(anchored.w[half:quarter].mean(axis=0)
                                 - anchored.w[quarter:].mean(axis=0), axis=1).mean())
    still_rising = bool(mean_accuracy[-1] > mean_accuracy[half] + 1e-4)
    converged = (not still_rising) and drift < 0.08
    print(f"  accuracy mid -> final                  = {mean_accuracy[half]:.4f} -> "
          f"{mean_accuracy[-1]:.4f}   still rising: {still_rising}")
    print(f"  ergodic-average drift across the window= {drift:.5f}")
    if not converged:
        print("  *** WARNING: the chain has NOT converged in this iteration budget. ***")
        print("  *** The ergodic-variance ratios below reflect the transient and are ***")
        print("  *** NOT comparable to converged runs. Increase n_iterations.        ***")
    summary_convergence = {"still_rising": still_rising, "ergodic_drift": drift,
                           "converged": bool(converged)}

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

    summary = {"lambda_lasso": args.lambda_lasso,
               "anchor_identity_gap": identity_gap,
               "a_lower_bound": target.a_lower_bound,
               "a_observed_min": float(a_values.min()),
               "lipschitz": target.lipschitz_constant(),
               "convergence": summary_convergence}
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

    sweep.to_csv(os.path.join(output_dir, "sweep.csv"), index=False)
    pd.DataFrame(held_rows).to_csv(os.path.join(output_dir, "held_out.csv"), index=False)
    summary["figures"] = figure_paths
    with open(os.path.join(output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print("\nwrote:", *sorted(os.listdir(output_dir)), sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
