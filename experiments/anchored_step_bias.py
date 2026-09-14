#!/usr/bin/env python3
"""How much of the gap to the exact posterior is the step size?

The anchored chains in ``run_anchored_srnsgld.py`` run unadjusted at the paper's
step size, so they are biased: their invariant law is ``e^{-U} 1_K`` only in the
limit.  This measures how much of the observed gap that accounts for, by running
the reversible chain (``J = 0``) at a ladder of step sizes with the total
simulated time held fixed -- ``eta k`` constant -- and reporting the whitened
distance from the time-averaged mean to the exact constrained lasso posterior's
mean, together with the accuracy of the ensemble it ends with.  Accuracy is
scored only at the end here, since the point is the bias and not the trajectory.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from nds.anchored_constrained import (  # noqa: E402
    Ball,
    LassoLogistic,
    SmoothedLpBall,
    ZeroField,
    per_walker_accuracy,
)
from nds.data import load_nine  # noqa: E402


def chain(target, domain, data, eta, n_iter, n_walkers, batch, seed, ref_mean, metric):
    """The ``J = 0`` anchored chain, scoring only the ensemble it ends with."""
    rng = np.random.default_rng(seed)
    d = target.d
    W = domain.uniform(d, n_walkers, rng, radius=1.0)
    zero = ZeroField(d)
    w_sum, n_kept = np.zeros(d), 0
    first = n_iter // 2
    for t in range(n_iter):
        G = target.anchor_grad(W, target.batch(batch, rng))
        a = target.clock(W)[None, :]
        Y = W - eta * a * G + np.sqrt(2.0 * eta * a) * rng.standard_normal((d, n_walkers))
        W, _, _ = domain.retract(Y, zero)
        if t >= first:
            w_sum += W.sum(axis=1)
            n_kept += n_walkers
    diff = w_sum / n_kept - ref_mean
    return {
        "mean_error": float(np.sqrt(diff @ (metric @ diff))),
        "accuracy_test": float(per_walker_accuracy(W, data["X_test"], data["y_test"]).mean()),
        "accuracy_train": float(per_walker_accuracy(W, data["X_train"], data["y_train"]).mean()),
    }


def main() -> None:
    from run_anchored_srnsgld import LAM_SCALE, SETTINGS  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=["titanic", "magic"])
    parser.add_argument("--domain", default="ball", choices=["ball", "lp"])
    parser.add_argument("--divisors", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--n-walkers", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    lines = [
        "# The step size and the gap to the exact posterior",
        "",
        "`J = 0`, the paper's batch size and walker count, total simulated time `eta k`",
        "held fixed as `eta` shrinks.  `error` is the whitened distance from the",
        "time-averaged mean (second half of the run) to the exact constrained lasso",
        "posterior's mean, in units of its own standard deviations.",
        "",
        "| problem | eta | iterations | error | accuracy, train | accuracy, test |",
        "|---|---|---|---|---|---|",
    ]
    for problem in args.problems:
        cfg = SETTINGS[problem]
        ds = load_nine(problem, seed=args.split_seed, test_fraction=0.2)
        data = {"X_train": ds.X_train, "y_train": ds.y_train,
                "X_test": ds.X_test, "y_test": ds.y_test}
        lam = LAM_SCALE[problem] * len(ds.y_train)
        target = LassoLogistic(ds.X_train, ds.y_train, lam=lam, delta=0.5 / lam)
        domain = (
            Ball(cfg["radius_sq"], squared=True) if args.domain == "ball"
            else SmoothedLpBall(cfg["lp"]["p"], cfg["lp"]["eps"], cfg["lp"]["level"])
        )
        ref = dict(np.load(os.path.join(
            RESULTS, f"reference_anchored_{problem}_{args.domain}_lam{lam:.4g}.npz")))
        ref_mean = np.asarray(ref["posterior_mean"], float)
        metric = np.linalg.pinv(np.asarray(ref["posterior_cov"], float))
        for div in args.divisors:
            started = time.time()
            out = chain(
                target, domain, data, cfg["step_size"] / div, cfg["n_iter"] * div,
                args.n_walkers, cfg["batch"], args.seed, ref_mean, metric,
            )
            lines.append(
                f"| {problem} | {cfg['step_size'] / div:.2e} | {cfg['n_iter'] * div} | "
                f"{out['mean_error']:.3f} | {out['accuracy_train']:.4f} | "
                f"{out['accuracy_test']:.4f} |"
            )
            print(lines[-1], f"({time.time() - started:.0f}s)", flush=True)
        lines.append(
            f"| {problem} **exact** | -- | -- | 0 | "
            f"{float(ref['accuracy_train']):.4f} | {float(ref['accuracy_test']):.4f} |"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(RESULTS, f"anchored_step_bias_{args.domain}.md"), "w") as h:
        h.write(text)


if __name__ == "__main__":
    main()
