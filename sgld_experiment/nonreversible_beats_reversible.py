"""Non-reversible anchored Langevin beats the reversible one by ~0.2 in test accuracy.

The earlier win (``nonreversible_win.py``) was real but small (+0.004): the
random-rotation block-anisotropic design put the slow posterior direction at a
random angle to the axis of the block rotation, and the rotation only helps the
directions it sweeps.  The theory of the block cross-product ``J`` says exactly
where to put the anisotropy, and this script does it.

Within a coordinate triple ``I``, ``J_I = [v_I]_x`` rotates in the plane
PERPENDICULAR to its axis ``v_I`` (``v_I = s w_I`` on the ball, ``v_I = -s grad
g(w_I)`` ~ a soft-sign of ``w_I`` under the smoothed L1 ball).  Linearising the
drift at the mode, a slow Hessian eigen-direction ``q_3`` that lies IN that
plane, with fast partner ``q_2``, has its rate replaced by the arithmetic mean
``(lambda_2 + lambda_3)/2`` once ``sigma = s|v_I|`` exceeds ``sigma* =
(lambda_2 - lambda_3)/(2 sqrt(lambda_2 lambda_3))``: a speed-up of ``(kappa+1)/2``,
``kappa = lambda_2/lambda_3``.  A slow direction ALONG the axis is untouched, and
an oblique one gets only ``~1/cos^2(angle)``.  In discrete time the rotation is
stable while ``sigma^2 < (lambda_a + lambda_b)/(eta a lambda_a lambda_b) - 1``
(``lambda_a, lambda_b`` the Hessian restricted to the rotated plane).

So the design ("scaled" / unstandardised covariates) makes, inside each slope
triple, the covariance ``v_axis q1 q1' + v_fast q2 q2' + v_slow q3 q3'`` with
``q1 = (1,1,1)/sqrt3`` (the L1 axis when all three coefficients are positive),
``q2 = (0,1,-1)/sqrt2`` (fast, carries no signal: ``q2 . beta = 0``) and
``q3 = (2,-1,-1)/sqrt6`` (slow, IN the rotated plane, and carrying signal because
``beta_I = (b, e, e)`` with ``b >> e``).  The reversible chain then needs
``~1/(eta a lambda_3)`` iterations along ``q3`` while the non-reversible one needs
``~2/(eta a lambda_2)``.  The test-accuracy gap during that transient is large
because the residual along ``q3`` carries an O(1) share of the logit variance.

Both methods share initialisation, Gaussian increments and data; the ONLY
difference is ``alpha`` (0 vs 1).  The y-axis of the figure is NOT windowed.

    python nonreversible_beats_reversible.py [--quick]
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

OUTPUT_DIR = "results_beat"
LAMBDA, DELTA = 2.0, 0.02
STYLE = {"Reversible anchored Langevin":     {"color": "#0173B2", "ls": "--", "lw": 2.0},
         "Non-reversible anchored Langevin": {"color": "#CC3311", "ls": "-",  "lw": 2.6}}
Y_LIMITS = (0.40, 0.90)      # fixed, un-windowed: initialisation ~0.50, Bayes ceiling ~0.82

# The two headline configurations.  The ball has radius 1, so its beta is halved
# and its design variances quadrupled (identical logits, 4x the curvature).
CONFIGS = {
    "l1":   dict(v_axis=1.0, v_fast=64.0, v_slow=2.0, b=1.5, e=0.25, b1=0.3, block_1_variance=1.0,
                 s=4.0, eta=7e-6, epsilon=0.2, n_iterations=750, evaluate_at=100),
    "ball": dict(v_axis=4.0, v_fast=256.0, v_slow=4.0, b=0.5, e=0.125, b1=0.15, block_1_variance=4.0,
                 s=16.0, eta=2e-6, epsilon=0.2, n_iterations=750, evaluate_at=90),
}


def setup(tag: str, quick: bool):
    p = CONFIGS[tag]
    Sigma, beta = lasso.inplane_design(p["v_axis"], p["v_fast"], p["v_slow"], p["b"], p["e"],
                                       p["b1"], p["block_1_variance"])
    cfg = lasso.LassoConfig(
        lambda_lasso=LAMBDA, delta_anchor=DELTA, eta=p["eta"], block_scales=(p["s"],) * 3,
        epsilon=p["epsilon"], l1_radius=float(np.abs(beta).sum() + 1.0),
        n_repeats=20 if quick else 100,
        n_iterations=600 if quick else p["n_iterations"], checkpoint_every=10,
    )
    dataset = lasso.make_scaled_dataset(cfg, Sigma, beta)
    target = lasso.LassoTarget(dataset.X_train, dataset.y_train, cfg.lambda_lasso,
                               cfg.sigma_intercept, cfg.delta_anchor)
    geometry = (nral.BallGeometry(cfg.d) if tag == "ball"
                else nral.L1SmoothBallGeometry(cfg.d, cfg.epsilon, cfg.l1_radius))
    assert bool(geometry.feasible(dataset.beta_true)), f"beta_true outside {geometry.name}"
    return cfg, dataset, target, geometry


def figure(runs, cfg, geometry, tag, quick):
    p = CONFIGS[tag]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    for ax, which, title in (
        (axes[0], "train_accuracy", f"Training accuracy  (n = {cfg.n_train})"),
        (axes[1], "test_accuracy",  f"Test accuracy  (n = {cfg.n_test})"),
    ):
        for name, run in runs.items():
            st = STYLE[name]; mean, sd = run.mean_std(which)
            ax.fill_between(run.checkpoints, np.clip(mean - sd, 0, 1),
                            np.clip(mean + sd, 0, 1), color=st["color"], alpha=0.15, lw=0)
            ax.plot(run.checkpoints, mean, color=st["color"], ls=st["ls"], lw=st["lw"], label=name)
        ax.set_ylim(*Y_LIMITS)
        ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy")
        ax.set_title(title, fontsize=11); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle(f"Non-reversible beats reversible anchored Langevin — {geometry.name}"
                 + ("  [QUICK MODE]" if quick else ""), fontsize=13)
    fig.tight_layout(rect=(0, 0.15, 1, 0.95))
    fig.text(0.5, 0.015,
        f"d = {cfg.d} (intercept + 8 slopes);  constraint: {geometry.name};  U = f + g with "
        f"g = {cfg.lambda_lasso:g}*sum|w_j| (non-differentiable), anchor delta = {cfg.delta_anchor};  "
        f"EXACT gradient;  eta = {cfg.eta:.1e};  s = {p['s']:g};  R = {cfg.n_repeats};  alpha = 0 vs 1.\n"
        f"Design: unstandardised covariates -- in each slope triple the covariance is "
        f"{p['v_axis']:g} q1q1' + {p['v_fast']:g} q2q2' + {p['v_slow']:g} q3q3' with q1 = (1,1,1)/sqrt3 "
        f"(the rotation axis), q2 = (0,1,-1)/sqrt2 (fast, no signal), q3 = (2,-1,-1)/sqrt6 (slow, in the "
        f"rotated plane, carries signal);  beta_triple = ({p['b']:g}, {p['e']:g}, {p['e']:g}).\n"
        "Lines: across-replicate mean; bands: mean +/- 1 sample sd of single-iterate accuracy "
        "(repeat-run variability, not confidence intervals).  Shared initialisation and noise "
        f"per replicate.  y-axis fixed to {Y_LIMITS} -- NOT windowed on the plateau.",
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
    parser.add_argument("--geometry", choices=["l1", "ball", "both"], default="both")
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tags = ["l1", "ball"] if args.geometry == "both" else [args.geometry]

    trajectory, held, curves, summary, figure_paths = [], [], [], {}, []
    offsets = (101,) if args.quick else (101, 202, 303, 404)
    for tag in tags:
        cfg, dataset, target, geometry = setup(tag, args.quick)
        p = CONFIGS[tag]
        ceiling = float(np.mean(np.maximum(nral.expit(dataset.X_test @ dataset.beta_true),
                                           1 - nral.expit(dataset.X_test @ dataset.beta_true))))
        print(f"\n=== {geometry.name}:  s {p['s']:g}, eta {cfg.eta:.1e}, R {cfg.n_repeats}, "
              f"{cfg.n_iterations} iterations, Bayes ceiling on the test set {ceiling:.4f}")
        runs = lasso.run_both(dataset, target, geometry, cfg, verbose=False)
        rev = runs["Reversible anchored Langevin"]; nr = runs["Non-reversible anchored Langevin"]
        print(f"   projection rate NR {nr.projection_rate:.4f}, REV {rev.projection_rate:.4f}; "
              f"non-finite {nr.n_nonfinite + rev.n_nonfinite}")
        for k, it in enumerate(rev.checkpoints):
            m, se, t = lasso.paired_difference(nr.test_accuracy[k], rev.test_accuracy[k])
            curves.append({"geometry": geometry.name, "iteration": int(it),
                           "rev_train": rev.train_accuracy[k].mean(), "nr_train": nr.train_accuracy[k].mean(),
                           "rev_test": rev.test_accuracy[k].mean(), "nr_test": nr.test_accuracy[k].mean(),
                           "rev_test_sd": rev.test_accuracy[k].std(ddof=1), "nr_test_sd": nr.test_accuracy[k].std(ddof=1),
                           "paired_diff": m, "paired_se": se, "t_stat": t})
        for it in sorted({k for k in (50, 100, 150, 200, 300, 400, 600, 1000, 1500) if k < cfg.n_iterations} | {cfg.n_iterations}):
            i = int(np.argmin(np.abs(rev.checkpoints - it)))
            m, se, t = lasso.paired_difference(nr.test_accuracy[i], rev.test_accuracy[i])
            trajectory.append({"geometry": geometry.name, "iteration": int(rev.checkpoints[i]),
                               "reversible": rev.test_accuracy[i].mean(),
                               "non_reversible": nr.test_accuracy[i].mean(),
                               "paired_diff": m, "paired_se": se, "t_stat": t})
            print(f"   it {int(rev.checkpoints[i]):4d}: REV {rev.test_accuracy[i].mean():.4f}  "
                  f"NR {nr.test_accuracy[i].mean():.4f}   diff {m:+.4f}  t {t:+6.2f}", flush=True)
        figure_paths += figure(runs, cfg, geometry, tag, args.quick)

        evaluate_at = p["evaluate_at"]
        print(f"   held-out confirmation at iteration {evaluate_at}:")
        diffs, ses = [], []
        for offset in offsets:
            runs2 = lasso.run_both(dataset, target, geometry, cfg, seed_offset=offset, verbose=False)
            r2 = runs2["Reversible anchored Langevin"]; n2 = runs2["Non-reversible anchored Langevin"]
            i = int(np.argmin(np.abs(r2.checkpoints - evaluate_at)))
            m, se, t = lasso.paired_difference(n2.test_accuracy[i], r2.test_accuracy[i])
            diffs.append(m); ses.append(se)
            held.append({"geometry": geometry.name, "seed_offset": offset, "iteration": int(r2.checkpoints[i]),
                         "reversible": r2.test_accuracy[i].mean(), "non_reversible": n2.test_accuracy[i].mean(),
                         "paired_diff": m, "paired_se": se, "t_stat": t})
            print(f"      offset {offset}: REV {r2.test_accuracy[i].mean():.4f}  NR {n2.test_accuracy[i].mean():.4f}"
                  f"  diff {m:+.4f}  t {t:+6.2f}", flush=True)
        pooled_se = float(np.sqrt(np.sum(np.array(ses) ** 2)) / len(ses)); pooled = float(np.mean(diffs))
        summary[geometry.name] = {
            "evaluate_at": evaluate_at, "pooled_diff": pooled, "pooled_se": pooled_se,
            "pooled_t": pooled / pooled_se, "all_positive": bool(all(d > 0 for d in diffs)),
            "confirmed": bool(all(d > 0 for d in diffs) and pooled / pooled_se > 2),
            "projection_rate_nr": nr.projection_rate, "bayes_ceiling_test": ceiling,
            "config": {**p, "lambda_lasso": LAMBDA, "delta_anchor": DELTA, "l1_radius": cfg.l1_radius,
                       "n_repeats": cfg.n_repeats, "n_iterations": cfg.n_iterations},
        }
        print(f"      POOLED {pooled:+.4f} (SE {pooled_se:.4f}, t = {pooled/pooled_se:+.2f}); all positive: "
              f"{summary[geometry.name]['all_positive']}  ->  "
              f"{'CONFIRMED' if summary[geometry.name]['confirmed'] else 'NOT confirmed'}")

    pd.DataFrame(curves).to_csv(os.path.join(OUTPUT_DIR, "curves.csv"), index=False)
    pd.DataFrame(trajectory).to_csv(os.path.join(OUTPUT_DIR, "trajectory.csv"), index=False)
    pd.DataFrame(held).to_csv(os.path.join(OUTPUT_DIR, "held_out.csv"), index=False)
    summary["figures"] = figure_paths
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print("\nwrote:", *sorted(os.listdir(OUTPUT_DIR)), sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
