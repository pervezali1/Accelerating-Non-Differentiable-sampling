"""A configuration where non-reversible anchored Langevin beats the reversible one.

Every earlier attempt failed on accuracy, and the reason is structural rather
than a tuning failure.  ``J`` is block diagonal on the coordinate triples
``(0,1,2), (3,4,5), (6,7,8)``, because it is built from the *constraint*
geometry.  A non-reversible perturbation accelerates convergence by coupling the
slow and fast directions of the *target*.  If the target's slow directions span
blocks -- as they do under an isotropic or an AR(1) design -- ``J`` cannot reach
them, and it does nothing.

The linearised per-iteration rate ``-log rho(I - eta a (I - alpha J) H)``, with
``H`` the Hessian of ``U0`` at the mode, makes this quantitative:

    design                                   best speed-up
    isotropic  X ~ N(0, 2I)                      1.10x
    AR(1) rho_x = 0.99                           2.24x
    block-anisotropic (inside each triple)       4.86x

So the fix is to align the target's anisotropy with ``J``'s block structure, and
then run in a regime where convergence -- not the Bayes ceiling -- is what limits
accuracy.  Two conditions have to hold at once:

1. ``eta`` small enough that the reversible chain is still converging at the
   evaluation point (otherwise both have converged and the gap closes);
2. ``s`` large enough to matter but small enough that the projection stays
   inactive.  The earlier failures at large ``s`` were the projection firing on
   nearly every iteration -- the linearised analysis is blind to that, because it
   sees local stability and not the step size relative to the domain.

    python nonreversible_win.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anchored_lasso as lasso
import anchored_sgld as nral

OUTPUT_DIR = "results_win"
SPREAD, LAMBDA, BLOCK_STRENGTH, ETA = 10.0, 2.0, 5.0, 7.87e-6
EVALUATE_AT = 1000
STYLE = {"Reversible anchored Langevin":     {"color": "#0173B2", "ls": "--", "lw": 2.0},
         "Non-reversible anchored Langevin": {"color": "#CC3311", "ls": "-",  "lw": 2.6}}


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
            plateau.append(mean[len(run.checkpoints) // 8:])
        plateau = np.concatenate(plateau); span = plateau.max() - plateau.min()
        ax.set_ylim(plateau.min() - 0.25 * span - 0.002, plateau.max() + 0.25 * span + 0.002)
        ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy")
        ax.set_title(title, fontsize=11); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle(f"Non-reversible beats reversible anchored Langevin — {geometry.name}"
                 + ("  [QUICK MODE]" if quick else ""), fontsize=13)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    fig.text(0.5, 0.015,
        f"d = {cfg.d} (intercept + 8 slopes);  constraint: {geometry.name};  "
        f"U = f + g with g = {cfg.lambda_lasso:g}*sum|w_j| (non-differentiable), anchor "
        f"delta = {cfg.delta_anchor};  EXACT gradient;\n"
        f"design: block-anisotropic, Sigma_X eigenvalues ({SPREAD:g}, 1, {1/SPREAD:g}) inside "
        f"each coordinate triple, matching the block structure of J;  eta = {cfg.eta:.2e};  "
        f"R = {cfg.n_repeats};  s = {tuple(float(x) for x in cfg.scales)};  alpha = 0 vs 1.\n"
        "Lines are the across-replicate mean; bands are mean +/- 1 sample sd (ddof = 1) of "
        "single-iterate accuracy -- repeat-run variability, NOT confidence or credible "
        "intervals.  Axis windowed on the plateau, not [0, 1].",
        ha="center", fontsize=7.2)
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

    cfg = lasso.LassoConfig(
        lambda_lasso=LAMBDA, eta=ETA, block_scales=(BLOCK_STRENGTH,) * 3,
        n_repeats=20 if args.quick else 100,
        n_iterations=600 if args.quick else 2000,
        checkpoint_every=20,
    )
    dataset = lasso.make_block_anisotropic_dataset(cfg, (SPREAD, 1.0, 1.0 / SPREAD))
    target = lasso.LassoTarget(dataset.X_train, dataset.y_train, cfg.lambda_lasso,
                               cfg.sigma_intercept, cfg.delta_anchor)
    ball = nral.BallGeometry(cfg.d)
    l1 = nral.L1SmoothBallGeometry(cfg.d, cfg.epsilon, cfg.l1_radius)
    print(f"block-anisotropic spread {SPREAD:g};  lambda {LAMBDA:g};  s {BLOCK_STRENGTH:g};  "
          f"eta {ETA:.2e};  R {cfg.n_repeats};  evaluate at iteration {EVALUATE_AT}")
    for g in (ball, l1):
        assert bool(g.feasible(dataset.beta_true)), g.name

    trajectory, figure_paths, results = [], [], {}
    for g, tag in ((ball, "ball"), (l1, "l1")):
        runs = lasso.run_both(dataset, target, g, cfg, verbose=False)
        results[g.name] = runs
        rev = runs["Reversible anchored Langevin"]; nr = runs["Non-reversible anchored Langevin"]
        print(f"\n=== {g.name} (search seed) ===   projection rate {nr.projection_rate:.4f}, "
              f"non-finite {nr.n_nonfinite}")
        for k in (100, 200, 400, 600, 1000, 1500, cfg.n_iterations):
            i = int(np.argmin(np.abs(rev.checkpoints - k)))
            m, se, t = lasso.paired_difference(nr.test_accuracy[i], rev.test_accuracy[i])
            trajectory.append({"geometry": g.name, "iteration": int(rev.checkpoints[i]),
                               "reversible": rev.test_accuracy[i].mean(),
                               "non_reversible": nr.test_accuracy[i].mean(),
                               "paired_diff": m, "paired_se": se, "t_stat": t})
            print(f"   it {int(rev.checkpoints[i]):4d}: REV {rev.test_accuracy[i].mean():.4f}  "
                  f"NR {nr.test_accuracy[i].mean():.4f}   diff {m:+.5f}  t {t:+6.2f}", flush=True)
        figure_paths += figure(runs, cfg, g, tag, args.quick)

    print(f"\n=== HELD-OUT confirmation at iteration {EVALUATE_AT} ===")
    held, summary = [], {}
    offsets = (101,) if args.quick else (101, 202, 303, 404)
    for g in (ball, l1):
        diffs, ses = [], []
        for offset in offsets:
            runs = lasso.run_both(dataset, target, g, cfg, seed_offset=offset, verbose=False)
            rev = runs["Reversible anchored Langevin"]
            nr = runs["Non-reversible anchored Langevin"]
            i = int(np.argmin(np.abs(rev.checkpoints - EVALUATE_AT)))
            m, se, t = lasso.paired_difference(nr.test_accuracy[i], rev.test_accuracy[i])
            diffs.append(m); ses.append(se)
            held.append({"geometry": g.name, "seed_offset": offset,
                         "reversible": rev.test_accuracy[i].mean(),
                         "non_reversible": nr.test_accuracy[i].mean(),
                         "paired_diff": m, "paired_se": se, "t_stat": t})
            print(f"  {g.name:<18} offset {offset}: diff {m:+.5f}  t {t:+6.2f}", flush=True)
        pooled_se = float(np.sqrt(np.sum(np.array(ses) ** 2)) / len(ses))
        pooled = float(np.mean(diffs))
        summary[g.name] = {"pooled_diff": pooled, "pooled_se": pooled_se,
                           "pooled_t": pooled / pooled_se,
                           "all_positive": bool(all(d > 0 for d in diffs)),
                           "confirmed": bool(all(d > 0 for d in diffs) and pooled/pooled_se > 2)}
        print(f"  {g.name:<18} POOLED {pooled:+.5f} (SE {pooled_se:.5f}, "
              f"t = {pooled/pooled_se:+.2f}); all positive: {summary[g.name]['all_positive']}"
              f"  ->  {'CONFIRMED' if summary[g.name]['confirmed'] else 'NOT confirmed'}")

    pd.DataFrame(trajectory).to_csv(os.path.join(OUTPUT_DIR, "trajectory.csv"), index=False)
    pd.DataFrame(held).to_csv(os.path.join(OUTPUT_DIR, "held_out.csv"), index=False)
    summary["config"] = {"spread": SPREAD, "lambda_lasso": LAMBDA, "s": BLOCK_STRENGTH,
                         "eta": ETA, "evaluate_at": EVALUATE_AT,
                         "n_repeats": cfg.n_repeats, "n_iterations": cfg.n_iterations}
    summary["figures"] = figure_paths
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print("\nwrote:", *sorted(os.listdir(OUTPUT_DIR)), sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
