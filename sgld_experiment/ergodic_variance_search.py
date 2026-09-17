"""Exact-gradient anchored Langevin: the observable non-reversibility actually targets.

Single-iterate accuracy is saturated on this problem (both methods sit at the
Bayes ceiling), so it cannot resolve a sampler improvement.  Non-reversible
perturbations are not designed to move the invariant measure at all — they are
designed to reduce the **asymptotic variance of ergodic (time) averages**.  That
is what is measured here, with exact gradients and the LASSO target.

    python ergodic_variance_search.py
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd

import anchored_lasso as lasso
import anchored_sgld as nral

OUTPUT_DIR = "results_lasso_exact"
BURN_CHECKPOINTS = 20            # = iteration 200


def ergodic_variance(run: lasso.RunResult) -> float:
    """Across-replicate variance of the time-averaged coefficient."""
    return float(run.w[BURN_CHECKPOINTS:].mean(axis=0).var(axis=0, ddof=1).sum())


def main() -> int:
    base = lasso.LassoConfig(n_repeats=100, n_iterations=1000)
    dataset = lasso.make_dataset(base)
    target = lasso.LassoTarget(dataset.X_train, dataset.y_train, base.lambda_lasso,
                               base.sigma_intercept, base.delta_anchor)
    ball = nral.BallGeometry(base.d)
    lp = lasso.SmoothLpBallGeometry(base.d, base.p_constraint, base.epsilon,
                                    Lambda=base.lp_Lambda)
    rows = []
    for s in (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0):
        for geometry in (ball, lp):
            runs = lasso.run_both(dataset, target, geometry,
                                  replace(base, block_scales=(s,) * 3), verbose=False)
            rev = runs["Reversible anchored Langevin"]
            nr = runs["Non-reversible anchored Langevin"]
            accuracy_diff, se, t = lasso.paired_difference(
                nr.test_accuracy[-1], rev.test_accuracy[-1])
            rows.append({
                "geometry": geometry.name, "s": s,
                "ergodic_variance_ratio": ergodic_variance(nr) / ergodic_variance(rev),
                "paired_accuracy_diff": accuracy_diff, "accuracy_t": t,
                "projection_rate": nr.projection_rate, "drift_ratio": nr.drift_ratio,
            })
            print(f"  {geometry.name:<18} s={s:5.1f}  var ratio "
                  f"{rows[-1]['ergodic_variance_ratio']:8.4f}   accuracy diff "
                  f"{accuracy_diff:+.5f} (t={t:+5.2f})", flush=True)
    sweep = pd.DataFrame(rows)

    ball_rows = sweep[sweep.geometry == ball.name]
    best_s = float(ball_rows.loc[ball_rows.ergodic_variance_ratio.idxmin(), "s"])
    print(f"\nbest s on the ball = {best_s}; confirming on held-out sampler seeds")
    held = []
    for offset in (101, 202, 303, 404):
        runs = lasso.run_both(dataset, target, ball,
                              replace(base, block_scales=(best_s,) * 3),
                              seed_offset=offset, verbose=False)
        ratio = (ergodic_variance(runs["Non-reversible anchored Langevin"])
                 / ergodic_variance(runs["Reversible anchored Langevin"]))
        held.append({"seed_offset": offset, "ergodic_variance_ratio": ratio})
        print(f"  offset {offset}: ratio {ratio:.4f}", flush=True)
    held = pd.DataFrame(held)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sweep.to_csv(os.path.join(OUTPUT_DIR, "ergodic_variance_sweep.csv"), index=False)
    held.to_csv(os.path.join(OUTPUT_DIR, "ergodic_variance_heldout.csv"), index=False)
    summary = {
        "selected_s": best_s,
        "search_seed_ratio": float(ball_rows.ergodic_variance_ratio.min()),
        "held_out_mean_ratio": float(held.ergodic_variance_ratio.mean()),
        "held_out_all_below_one": bool((held.ergodic_variance_ratio < 1).all()),
        "variance_reduction_percent": 100.0 * (1.0 - float(held.ergodic_variance_ratio.mean())),
        "note": ("Accuracy is saturated on this problem and cannot resolve a sampler "
                 "improvement; the ergodic-average variance is the quantity "
                 "non-reversible perturbations are designed to reduce."),
    }
    with open(os.path.join(OUTPUT_DIR, "ergodic_variance_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nheld-out mean ratio {summary['held_out_mean_ratio']:.4f} "
          f"({summary['variance_reduction_percent']:.1f}% variance reduction); "
          f"all below 1: {summary['held_out_all_below_one']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
