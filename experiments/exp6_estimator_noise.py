#!/usr/bin/env python3
"""How trustworthy is the 2-Wasserstein number on a heavy tail?

Every convergence curve in this repository is read against the estimator's own
noise, so that noise is measured directly: the sliced 2-Wasserstein statistic is
evaluated on *exact* i.i.d. draws from the target, many times, and its sampling
distribution is reported.

Two things come out, and both change how the paper's Figure 8 should be read.

1. The floor decays like ``n^{-(1/2 - 1/nu)}`` -- for ``nu = 3`` that is
   ``n^{-1/6}``, so quadrupling the sample size buys about 20 %.  At the paper's
   ``n = 5000`` nothing below roughly 0.23 is signal.
2. The sampling distribution of the estimator is itself heavy tailed: single
   repetitions land a factor of five above the median.  Averaging it over runs
   is dominated by outliers, so the **median** across repetitions is the summary
   to compare, not the mean.

The naive midpoint quantile estimator is measured alongside: it truncates the
two tail cells and so reports a systematically smaller number.
"""

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skewanchor import metrics, runner  # noqa: E402
from skewanchor.targets import anisotropic_student_t, isotropic_polynomial  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "data")


def profile(target, n, reps, rng):
    acc, mid = [], []
    for _ in range(reps):
        x = target.sample(n, rng)
        acc.append(metrics.axis_sliced_w2(x, target))
        mid.append(metrics.sliced_w2_midpoint(x, target))
    acc, mid = np.array(acc), np.array(mid)
    return {
        "n": n, "reps": reps,
        "mean": float(acc.mean()), "median": float(np.median(acc)),
        "std": float(acc.std()), "max": float(acc.max()),
        "max_over_median": float(acc.max() / np.median(acc)),
        "midpoint_mean": float(mid.mean()), "midpoint_median": float(np.median(mid)),
        "truncation_ratio": float(np.median(acc) / np.median(mid)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    targets = [
        ("paper Fig 8: d=1, iota=2 (nu=3)", isotropic_polynomial(1, 2.0)),
        ("isotropic d=3, iota=3 (nu=3)", isotropic_polynomial(3, 3.0)),
        ("anisotropic d=2, nu=3, kappa=100", anisotropic_student_t(2, 3.0, 100.0)),
        ("anisotropic d=2, nu=5, kappa=100", anisotropic_student_t(2, 5.0, 100.0)),
        ("anisotropic d=2, nu=8, kappa=100", anisotropic_student_t(2, 8.0, 100.0)),
        ("anisotropic d=5, nu=8, kappa=100", anisotropic_student_t(5, 8.0, 100.0)),
    ]

    results = {"reps": args.reps, "profiles": [], "decay": []}

    print(f"{'target':34s} {'mean':>8} {'median':>8} {'std':>8} {'max':>8} "
          f"{'max/med':>8} {'vs midpt':>9}")
    for name, t in targets:
        p = profile(t, 5000, args.reps, rng)
        p["target"] = name
        results["profiles"].append(p)
        print(f"{name:34s} {p['mean']:8.4f} {p['median']:8.4f} {p['std']:8.4f} "
              f"{p['max']:8.4f} {p['max_over_median']:8.2f} {p['truncation_ratio']:9.2f}x")

    print("\nhow the floor falls with the sample size (paper's target, nu=3)")
    t = isotropic_polynomial(1, 2.0)
    prev = None
    for n in (500, 1000, 2000, 5000, 10000, 20000, 50000):
        p = profile(t, n, max(40, args.reps // 4), rng)
        p["target"] = "paper Fig 8"
        results["decay"].append(p)
        ratio = "" if prev is None else f"  ({p['median'] / prev:.3f} of the previous n)"
        print(f"  n={n:6d}  median floor={p['median']:.4f}{ratio}")
        prev = p["median"]
    exps = np.polyfit(np.log([r["n"] for r in results["decay"]]),
                      np.log([r["median"] for r in results["decay"]]), 1)
    results["fitted_exponent"] = float(exps[0])
    print(f"  fitted decay exponent {exps[0]:.3f} against the predicted "
          f"-(1/2 - 1/nu) = {-(0.5 - 1 / 3):.3f} for nu = 3")

    print("\nwrote", runner.save_json(results, os.path.join(OUT, "exp6_estimator_noise.json")))


if __name__ == "__main__":
    main()
