#!/usr/bin/env python3
"""Does the clock earn its keep?  A long run of the three schemes.

Over a short run the Monte Carlo error of the running mean dominates, and the
two baselines that ignore the clock even look slightly better, because they do
not pay its slowdown.  Run them long enough and the difference shows: the
anchored chain keeps converging towards the reference, while ``anchor`` and
``fd`` flatten out at the bias their smoothing introduces.

Only the small dataset by default -- this needs tens of thousands of iterations,
which is minutes on Titanic and an hour on MAGIC.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nds.anchored import LassoLogistic, run_anchored_chain  # noqa: E402
from nds.data import PRETTY_NAMES, load  # noqa: E402
from nds.design import explicit_step_size  # noqa: E402
from nds.sampler import Geometry  # noqa: E402
from nds.skew import ZeroSkew  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SCHEMES = ("anchored", "anchor", "fd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["titanic"])
    parser.add_argument("--penalty", type=float, default=5.0)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--n-iter", type=int, default=30000)
    parser.add_argument("--n-walkers", type=int, default=24)
    parser.add_argument("--safety", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = {}
    for name in args.datasets:
        ds = load(name, seed=0)
        target = LassoLogistic(ds.X_train, ds.y_train, penalty=args.penalty, delta=args.delta)
        ref = np.load(os.path.join(RESULTS, f"reference_lasso_{name}_p{args.penalty:g}.npz"))
        ref_mean, ref_metric = ref["posterior_mean"], np.linalg.pinv(ref["posterior_cov"])
        H = np.linalg.pinv(ref["posterior_cov"])
        h = explicit_step_size(H, None, safety=args.safety)
        arrays, rows = {}, []
        for scheme in SCHEMES:
            t0 = time.time()
            res = run_anchored_chain(
                target, ZeroSkew(), h, args.n_iter, n_walkers=args.n_walkers,
                seed=args.seed + 11, X_eval=ds.X_test, y_eval=ds.y_test,
                geometry=Geometry.identity(target.d), scheme=scheme,
                ref_mean=ref_mean, ref_metric=ref_metric,
            )
            arrays[f"error_{scheme}"] = res.mean_error.mean(axis=1)
            arrays[f"loss_{scheme}"] = res.loss.mean(axis=1)
            mean_hat = res.posterior_mean.mean(axis=1)
            rows.append({
                "dataset": name, "scheme": scheme, "step_size": h,
                "mean_clock": res.meta["mean_clock"],
                "error_tail": float(res.mean_error[-args.n_iter // 10:].mean()),
                "loss_final": float(res.loss[-1].mean()),
                "l1_of_mean": float(np.abs(mean_hat[1:]).sum()),
            })
            print(f"[{name}] {scheme:9s} clock={res.meta['mean_clock']:.3f} "
                  f"bias(tail)={rows[-1]['error_tail']:.3f} loss={rows[-1]['loss_final']:.4f} "
                  f"|w|_1={rows[-1]['l1_of_mean']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        meta = {
            "dataset": name, "pretty_name": PRETTY_NAMES[name], "n_iter": args.n_iter,
            "penalty": args.penalty, "delta": args.delta, "step_size": h,
            "clock_floor": target.clock_floor,
            "reference_loss": float(ref["loss"]), "reference_accuracy": float(ref["accuracy"]),
            "reference_l1_of_mean": float(np.abs(ref_mean[1:]).sum()),
            "rows": rows, "schemes": list(SCHEMES),
        }
        np.savez_compressed(os.path.join(RESULTS, f"bias_anchored_{name}_d{args.delta:g}.npz"),
                            meta=json.dumps(meta), **arrays)
        out[name] = meta
    with open(os.path.join(RESULTS, "anchored_bias.json"), "w") as handle:
        json.dump(out, handle, indent=2)
    print("wrote results/anchored_bias.json")


if __name__ == "__main__":
    main()
