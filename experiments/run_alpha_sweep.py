#!/usr/bin/env python3
"""How the irreversibility strength ``alpha`` pays for itself, or does not.

For a ladder of strengths the script calibrates a step size for the constant
and the state-dependent field and records what the chain achieves in a fixed
iteration budget.  It is the diagnostic behind the headline result: a rotation
that is strong in the Metropolis proposal has to be paid for with a smaller
step size, because the ``O(sqrt(h))`` term that cancels in the reversible
acceptance ratio does not cancel once ``J`` is switched on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nds.data import PRETTY_NAMES, load  # noqa: E402
from nds.metrics import iterations_to_reach  # noqa: E402
from nds.sampler import calibrate_step_size, run_chain, warm_up  # noqa: E402
from nds.skew import ConstantSkew, LocalizedSkew, ZeroSkew, random_skew, whiten  # noqa: E402
from nds.target import LogisticPosterior  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def sweep_dataset(name: str, args) -> dict:
    ds = load(name, seed=args.split_seed)
    target = LogisticPosterior(ds.X_train, ds.y_train, prior_scale=args.prior_scale)
    ref = np.load(os.path.join(RESULTS, f"reference_{name}.npz"))
    ref_acc = float(ref["accuracy"])
    ref_mean = ref["posterior_mean"]
    ref_metric = np.linalg.pinv(ref["posterior_cov"])

    pilot = warm_up(target, n_iter=args.warmup_iter, n_walkers=8, seed=args.seed)
    geo, rho = pilot["geometry"], pilot["rho"]
    A = whiten(random_skew(target.d, seed=args.skew_seed), geo.L)

    rows = []
    for alpha in args.alphas:
        for kind in ("constant", "state"):
            if alpha == 0.0:
                skew = ZeroSkew()
            elif kind == "constant":
                skew = ConstantSkew(A, alpha)
            else:
                skew = LocalizedSkew(A, alpha, rho=rho, metric=geo.D_inv)
            cal = calibrate_step_size(
                target, skew, n_iter=args.calibrate_iter, n_walkers=6,
                seed=args.seed + 11, geometry=geo,
            )
            res = run_chain(
                target, skew, cal["step_size"], args.n_iter, n_walkers=args.n_walkers,
                seed=args.seed + 101, X_eval=ds.X_test, y_eval=ds.y_test, geometry=geo,
                ref_mean=ref_mean, ref_metric=ref_metric,
            )
            curve = res.accuracy.mean(axis=1)
            rows.append({
                "dataset": name,
                "kind": kind,
                "alpha": alpha,
                "step_size": cal["step_size"],
                "acceptance": res.acceptance,
                "accuracy_final": float(curve[-1]),
                "iters_to_ref_0.005": iterations_to_reach(curve, ref_acc, 0.005),
                "mean_error_final": float(res.mean_error[-1].mean()),
            })
            print(f"[{name}] alpha={alpha:<5g} {kind:9s} h={cal['step_size']:.4g} "
                  f"acc={res.acceptance:.3f} final={curve[-1]:.4f} "
                  f"err={res.mean_error[-1].mean():.3f}", flush=True)
    return {
        "dataset": name,
        "pretty_name": PRETTY_NAMES[name],
        "reference_accuracy": ref_acc,
        "rho": float(rho),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["titanic", "breast_cancer"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--n-iter", type=int, default=600)
    parser.add_argument("--n-walkers", type=int, default=16)
    parser.add_argument("--warmup-iter", type=int, default=400)
    parser.add_argument("--calibrate-iter", type=int, default=200)
    parser.add_argument("--prior-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skew-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    out = [sweep_dataset(name, args) for name in args.datasets]
    path = os.path.join(RESULTS, "alpha_sweep.json")
    with open(path, "w") as handle:
        json.dump({"datasets": out, "n_iter": args.n_iter}, handle, indent=2)
    print(f"wrote {path} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
