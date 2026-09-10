#!/usr/bin/env python3
"""Where the step-size ceiling comes from.

For proposals made from a fixed point, the spread of the log acceptance ratio
decays like ``h^{3/2}`` when ``J = 0`` -- the ``O(sqrt(h))`` term cancels, which
is what makes MALA-style schemes tolerate large steps -- and only like
``h^{1/2}`` once a rotation is switched on, because the surviving term is
``sqrt(2h) (J ghat) . xi``.  The fitted slopes below are the reason a constant
``J`` of the same magnitude as the reversible part has to run at a much smaller
step size.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nds.data import load  # noqa: E402
from nds.sampler import acceptance_diagnostics, warm_up  # noqa: E402
from nds.skew import ConstantSkew, LocalizedSkew, ZeroSkew, random_skew, whiten  # noqa: E402
from nds.target import LogisticPosterior  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="titanic")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-draws", type=int, default=2048)
    parser.add_argument("--warmup-iter", type=int, default=400)
    args = parser.parse_args()

    ds = load(args.dataset)
    target = LogisticPosterior(ds.X_train, ds.y_train)
    pilot = warm_up(target, n_iter=args.warmup_iter, n_walkers=8, seed=0)
    geo, rho = pilot["geometry"], pilot["rho"]
    A = whiten(random_skew(target.d, 0), geo.L)
    ref = np.load(os.path.join(RESULTS, f"reference_{args.dataset}.npz"))

    fields = {
        "J = 0": ZeroSkew(),
        "constant J_a": ConstantSkew(A, args.alpha),
        "state-dependent J_s": LocalizedSkew(A, args.alpha, rho=rho, metric=geo.D_inv),
    }
    points = {"w = 0 (cold start)": np.zeros(target.d), "posterior mean": ref["posterior_mean"]}
    grid = np.logspace(-6, -2, 9)

    for point_name, w in points.items():
        print(f"\n{args.dataset}: proposals from {point_name}")
        print(f"{'field':22s} {'slope of std(log alpha) vs h':>30s}   acceptance at h=1e-3")
        for label, field in fields.items():
            stats = [acceptance_diagnostics(target, field, h, w, n_draws=args.n_draws,
                                            geometry=geo, seed=3) for h in grid]
            spread = np.array([s["log_ratio_std"] for s in stats])
            slope = np.polyfit(np.log(grid), np.log(spread), 1)[0]
            at_mid = acceptance_diagnostics(target, field, 1e-3, w, n_draws=args.n_draws,
                                            geometry=geo, seed=3)["acceptance"]
            print(f"{label:22s} {slope:30.2f}   {at_mid:.3f}")


if __name__ == "__main__":
    main()
