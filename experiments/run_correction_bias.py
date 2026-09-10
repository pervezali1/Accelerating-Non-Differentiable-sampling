#!/usr/bin/env python3
"""What the divergence correction is for.

With the Metropolis correction switched on, ``exp(-U)`` is invariant whatever
``J`` does, so dropping ``Gamma = div J`` changes nothing but efficiency -- which
is why the two state-dependent curves in the accuracy figure lie on top of each
other.  The correction only earns its keep in the *uncorrected* diffusion.  This
script runs the same three fields with ``metropolis=False`` at one shared step
size and tracks how far the running mean settles from the reference posterior:

* ``J = 0``            -- pure Euler discretisation bias,
* ``J_s`` with Gamma   -- the same order of bias,
* ``J_s`` without Gamma -- a visibly larger, step-size-independent-looking bias.
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
from nds.sampler import calibrate_step_size, run_chain, warm_up  # noqa: E402
from nds.skew import LocalizedSkew, ZeroSkew, random_skew, whiten  # noqa: E402
from nds.target import LogisticPosterior  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run(name: str, args) -> dict:
    ds = load(name, seed=args.split_seed)
    target = LogisticPosterior(ds.X_train, ds.y_train, prior_scale=args.prior_scale)
    ref = np.load(os.path.join(RESULTS, f"reference_{name}.npz"))
    ref_mean, ref_acc = ref["posterior_mean"], float(ref["accuracy"])
    ref_metric = np.linalg.pinv(ref["posterior_cov"])

    pilot = warm_up(target, n_iter=args.warmup_iter, n_walkers=8, seed=args.seed)
    geo, rho = pilot["geometry"], pilot["rho"]
    A = whiten(random_skew(target.d, seed=args.skew_seed), geo.L)
    base = calibrate_step_size(
        target, ZeroSkew(), n_iter=args.calibrate_iter, n_walkers=6,
        seed=args.seed + 11, geometry=geo,
    )["step_size"]
    h = base * args.step_fraction

    variants = [
        ("zero", ZeroSkew()),
        ("state", LocalizedSkew(A, args.alpha, rho=rho, metric=geo.D_inv)),
        ("state_nocorr", LocalizedSkew(A, args.alpha, rho=rho, metric=geo.D_inv, drop_correction=True)),
    ]
    arrays, rows = {}, []
    for key, skew in variants:
        res = run_chain(
            target, skew, h, args.n_iter, n_walkers=args.n_walkers, seed=args.seed + 101,
            X_eval=ds.X_test, y_eval=ds.y_test, geometry=geo, metropolis=False,
            ref_mean=ref_mean, ref_metric=ref_metric,
        )
        arrays[f"error_{key}"] = res.mean_error
        arrays[f"accuracy_{key}"] = res.accuracy
        tail = res.mean_error[-args.n_iter // 10 :].mean()
        rows.append({
            "dataset": name,
            "variant": key,
            "step_size": h,
            "bias_whitened": float(tail),
            "accuracy_final": float(res.accuracy[-1].mean()),
            "potential_final": float(res.potential[-1].mean()),
        })
        print(f"[{name}] uncorrected {key:13s} h={h:.4g} bias={tail:.3f} "
              f"acc={res.accuracy[-1].mean():.4f}", flush=True)

    meta = {
        "dataset": name,
        "pretty_name": PRETTY_NAMES[name],
        "reference_accuracy": ref_acc,
        "step_size": h,
        "alpha": args.alpha,
        "rho": float(rho),
        "n_iter": args.n_iter,
        "n_walkers": args.n_walkers,
        "rows": rows,
        "variant_order": ["zero", "state", "state_nocorr"],
    }
    np.savez_compressed(
        os.path.join(RESULTS, f"bias_{name}.npz"), meta=json.dumps(meta), **arrays
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["titanic", "breast_cancer"])
    parser.add_argument("--n-iter", type=int, default=10000)
    parser.add_argument("--n-walkers", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--step-fraction", type=float, default=0.25,
                        help="fraction of the Metropolis-calibrated step size to use")
    parser.add_argument("--warmup-iter", type=int, default=400)
    parser.add_argument("--calibrate-iter", type=int, default=200)
    parser.add_argument("--prior-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skew-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    out = [run(name, args) for name in args.datasets]
    path = os.path.join(RESULTS, "correction_bias.json")
    with open(path, "w") as handle:
        json.dump({"datasets": out}, handle, indent=2)
    print(f"wrote {path} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
