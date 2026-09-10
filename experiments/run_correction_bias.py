#!/usr/bin/env python3
"""What the divergence correction is for, and when it is negligible.

With the Metropolis correction on, ``exp(-U)`` is invariant whatever ``J`` does,
so dropping ``Gamma = div J`` costs efficiency and nothing else -- which is why
the two state-dependent curves of the accuracy figure lie on top of each other.
``Gamma`` can only bias the *uncorrected* diffusion, and how much depends
entirely on how fast ``J`` varies where the chain actually is:

* ``radial`` -- the field used in the accuracy figure, ``J_s(w) = alpha s(w) A``
  with ``s`` a profile centred on ``w = 0``.  The posterior bulk of these
  problems sits many whitened standard deviations from the origin, where that
  profile has all but saturated, so ``Gamma`` is a few per cent of the drift and
  dropping it is invisible.
* ``directional`` -- ``J_c(w) = alpha tanh(c . (w - w_bulk)) A``, centred on the
  bulk, so ``J`` varies by order one across the posterior and ``Gamma`` is
  comparable to the reversible drift.  This is the regime where dropping the
  correction visibly biases the invariant law.

Each run is the uncorrected diffusion at one shared step size, so the ``J = 0``
curve is the pure Euler discretisation bias to compare against.
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
from nds.skew import (  # noqa: E402
    DirectionalSkew,
    LocalizedSkew,
    ZeroSkew,
    random_skew,
    whiten,
)
from nds.target import LogisticPosterior  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run(name: str, args) -> dict:
    ds = load(name, seed=args.split_seed)
    target = LogisticPosterior(ds.X_train, ds.y_train, prior_scale=args.prior_scale)
    ref = np.load(os.path.join(RESULTS, f"reference_{name}.npz"))
    ref_mean, ref_acc = ref["posterior_mean"], float(ref["accuracy"])
    ref_metric = np.linalg.pinv(ref["posterior_cov"])

    pilot = warm_up(target, n_iter=args.warmup_iter, n_walkers=8, seed=args.seed)
    geo, rho, w_bulk = pilot["geometry"], pilot["rho"], pilot["mean_state"]
    A = whiten(random_skew(target.d, seed=args.skew_seed), geo.L)
    direction = np.linalg.solve(geo.L.T, np.eye(target.d)[:, 0])
    h = args.step_fraction * calibrate_step_size(
        target, ZeroSkew(), n_iter=args.calibrate_iter, n_walkers=6,
        seed=args.seed + 11, geometry=geo,
    )["step_size"]

    def make(profile: str, drop: bool):
        if profile == "radial":
            return LocalizedSkew(
                A, args.alpha, rho=rho, metric=geo.D_inv, drop_correction=drop
            )
        return DirectionalSkew(
            A, args.alpha, direction=direction, length_scale=args.length_scale,
            center=w_bulk, drop_correction=drop,
        )

    # how large the correction is, relative to the reversible drift, in the bulk
    W_bulk = ref_mean.reshape(-1, 1)
    G_bulk = target.fd_grad(W_bulk)
    strength = {}
    for profile in ("radial", "directional"):
        field = make(profile, False)
        strength[profile] = {
            "gamma_norm": float(np.linalg.norm(field.divergence(W_bulk))),
            "reversible_drift_norm": float(np.linalg.norm(geo.D @ G_bulk)),
            "rotation_norm": float(np.linalg.norm(field.apply(W_bulk, G_bulk))),
        }

    arrays, rows = {}, []
    jobs = [("shared", "zero", ZeroSkew())]
    for profile in ("radial", "directional"):
        jobs.append((profile, "with_correction", make(profile, False)))
        jobs.append((profile, "no_correction", make(profile, True)))

    for profile, variant, field in jobs:
        res = run_chain(
            target, field, h, args.n_iter, n_walkers=args.n_walkers, seed=args.seed + 101,
            X_eval=ds.X_test, y_eval=ds.y_test, geometry=geo, metropolis=False,
            ref_mean=ref_mean, ref_metric=ref_metric,
        )
        key = f"{profile}_{variant}"
        arrays[f"error_{key}"] = res.mean_error
        arrays[f"accuracy_{key}"] = res.accuracy
        tail = float(res.mean_error[-args.n_iter // 10 :].mean())
        rows.append({
            "dataset": name, "profile": profile, "variant": variant, "step_size": h,
            "bias_whitened": tail, "accuracy_final": float(res.accuracy[-1].mean()),
            "potential_final": float(res.potential[-2000:].mean()),
        })
        print(f"[{name}] {profile:12s} {variant:15s} h={h:.4g} bias={tail:.3f} "
              f"acc={res.accuracy[-1].mean():.4f}", flush=True)

    meta = {
        "dataset": name, "pretty_name": PRETTY_NAMES[name], "reference_accuracy": ref_acc,
        "step_size": h, "alpha": args.alpha, "rho": float(rho),
        "length_scale": args.length_scale, "n_iter": args.n_iter,
        "n_walkers": args.n_walkers, "field_strength": strength, "rows": rows,
        "curves": ["shared_zero", "radial_with_correction", "radial_no_correction",
                   "directional_with_correction", "directional_no_correction"],
    }
    np.savez_compressed(
        os.path.join(RESULTS, f"bias_{name}.npz"), meta=json.dumps(meta), **arrays
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["titanic", "breast_cancer"])
    parser.add_argument("--n-iter", type=int, default=6000)
    parser.add_argument("--n-walkers", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--length-scale", type=float, default=1.0,
                        help="tanh scale of the directional profile, in posterior sd")
    parser.add_argument("--step-fraction", type=float, default=0.125,
                        help="fraction of the Metropolis-calibrated step size to use")
    parser.add_argument("--warmup-iter", type=int, default=400)
    parser.add_argument("--calibrate-iter", type=int, default=200)
    parser.add_argument("--prior-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skew-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    for name in args.datasets:
        meta = run(name, args)
        # one file per dataset, so that datasets can be run as parallel processes
        path = os.path.join(RESULTS, f"correction_bias_{name}.json")
        with open(path, "w") as handle:
            json.dump(meta, handle, indent=2)
        print(f"wrote {path}", flush=True)
    print(f"finished in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
