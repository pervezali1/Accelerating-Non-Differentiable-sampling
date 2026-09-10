#!/usr/bin/env python3
"""Convergence of classification accuracy from the ``w = 0`` start.

For each dataset the script

1. builds a gold-standard reference posterior (long gradient-based MALA),
2. warms up a preconditioner ``D`` and a radial scale ``rho`` with ``J = 0``,
3. calibrates a step size separately for each ``J`` variant, and
4. runs all variants from ``w = 0`` with common random seeds,

recording per-iteration accuracy of the running posterior predictive mean, the
whitened error of the running posterior mean, and the potential trace.

Usage::

    python experiments/run_accuracy.py --datasets titanic --n-iter 600
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
from nds.metrics import ess, iterations_to_reach  # noqa: E402
from nds.reference import reference_posterior  # noqa: E402
from nds.sampler import Geometry, calibrate_step_size, run_chain, warm_up  # noqa: E402
from nds.skew import ConstantSkew, LocalizedSkew, ZeroSkew, random_skew, whiten  # noqa: E402
from nds.target import LogisticPosterior  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def build_variants(A: np.ndarray, alpha: float, rho: float, metric: np.ndarray) -> list:
    """The four fields compared in the figures."""
    return [
        ("zero", ZeroSkew()),
        ("constant", ConstantSkew(A, alpha)),
        ("state", LocalizedSkew(A, alpha, rho=rho, metric=metric)),
        (
            "state_nocorr",
            LocalizedSkew(A, alpha, rho=rho, metric=metric, drop_correction=True),
        ),
    ]


def get_reference(name: str, ds, target: LogisticPosterior, args) -> dict:
    """Load the cached reference posterior for ``name`` or compute it."""
    path = os.path.join(RESULTS, f"reference_{name}.npz")
    if os.path.exists(path) and not args.refresh_reference:
        blob = np.load(path, allow_pickle=False)
        return {k: blob[k] for k in blob.files}
    ref = reference_posterior(
        target,
        ds.X_test,
        ds.y_test,
        n_iter=args.reference_iter,
        n_chains=8,
        seed=args.seed,
    )
    out = {
        "accuracy": np.array(ref["accuracy"]),
        "accuracy_halves": np.array(ref["accuracy_halves"]),
        "predictive": ref["predictive"],
        "posterior_mean": ref["posterior_mean"],
        "posterior_cov": ref["posterior_cov"],
        "map": ref["map"],
        "acceptance": np.array(ref["acceptance"]),
        "n_samples": np.array(ref["n_samples"]),
    }
    os.makedirs(RESULTS, exist_ok=True)
    np.savez(path, **out)
    return out


def run_dataset(name: str, args) -> dict:
    started = time.time()
    ds = load(name, seed=args.split_seed)
    target = LogisticPosterior(ds.X_train, ds.y_train, prior_scale=args.prior_scale)
    ref = get_reference(name, ds, target, args)
    ref_acc = float(ref["accuracy"])
    ref_mean = ref["posterior_mean"]
    ref_metric = np.linalg.pinv(ref["posterior_cov"])

    if args.geometry == "identity":
        geo = Geometry.identity(target.d)
        rho = args.rho if args.rho else float(np.sqrt(target.d))
        pilot = {"step_size": None, "whitened_radius": None}
    else:
        pilot = warm_up(
            target,
            n_iter=args.warmup_iter,
            n_walkers=args.warmup_walkers,
            seed=args.seed,
            n_rounds=args.warmup_rounds,
        )
        geo = pilot["geometry"]
        rho = args.rho if args.rho else pilot["rho"]

    A = whiten(random_skew(target.d, seed=args.skew_seed), geo.L)
    arrays, summary = {}, []

    for key, skew in build_variants(A, args.alpha, rho, geo.D_inv):
        cal = calibrate_step_size(
            target,
            skew,
            n_iter=args.calibrate_iter,
            n_walkers=args.calibrate_walkers,
            seed=args.seed + 11,
            geometry=geo,
        )
        res = run_chain(
            target,
            skew,
            cal["step_size"],
            args.n_iter,
            n_walkers=args.n_walkers,
            seed=args.seed + 101,
            X_eval=ds.X_test,
            y_eval=ds.y_test,
            geometry=geo,
            ref_mean=ref_mean,
            ref_metric=ref_metric,
        )
        arrays[f"accuracy_{key}"] = res.accuracy
        arrays[f"error_{key}"] = res.mean_error
        arrays[f"potential_{key}"] = res.potential
        curve = res.accuracy.mean(axis=1)
        half = args.n_iter // 2
        summary.append(
            {
                "dataset": name,
                "variant": key,
                "label": skew.label,
                "step_size": cal["step_size"],
                "pilot_acceptance": cal["pilot_acceptance"],
                "acceptance": res.acceptance,
                "accuracy_final": float(curve[-1]),
                "accuracy_at_100": float(curve[min(100, args.n_iter)]),
                "iters_to_ref_0.005": iterations_to_reach(curve, ref_acc, 0.005),
                "iters_to_ref_0.01": iterations_to_reach(curve, ref_acc, 0.01),
                "mean_error_final": float(res.mean_error[-1].mean()),
                "potential_final": float(res.potential[-1].mean()),
                "ess_potential_second_half": ess(res.potential[half:]),
            }
        )
        print(
            f"[{name}] {key:13s} h={cal['step_size']:.4g} acc={res.acceptance:.3f} "
            f"final={curve[-1]:.4f} err={res.mean_error[-1].mean():.3f} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

    meta = {
        "dataset": name,
        "pretty_name": PRETTY_NAMES[name],
        "dim": int(target.d),
        "n_train": int(len(ds.y_train)),
        "n_test": int(len(ds.y_test)),
        "reference_accuracy": ref_acc,
        "reference_accuracy_halves": [float(v) for v in np.atleast_1d(ref["accuracy_halves"])],
        "alpha": args.alpha,
        "rho": float(rho),
        "geometry": args.geometry,
        "n_iter": args.n_iter,
        "n_walkers": args.n_walkers,
        "prior_scale": args.prior_scale,
        "warmup_step_size": pilot.get("step_size"),
        "whitened_radius": pilot.get("whitened_radius"),
        "condition_number": float(np.linalg.cond(geo.D)),
        "summary": summary,
        "runtime_seconds": time.time() - started,
        "variant_order": ["zero", "constant", "state", "state_nocorr"],
    }
    tag = f"{name}_{args.geometry}_alpha{args.alpha:g}"
    np.savez_compressed(
        os.path.join(RESULTS, f"traces_{tag}.npz"), meta=json.dumps(meta), **arrays
    )
    with open(os.path.join(RESULTS, f"summary_{tag}.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[{name}] done in {meta['runtime_seconds']:.0f}s", flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["titanic", "magic", "breast_cancer", "spambase"])
    parser.add_argument("--n-iter", type=int, default=600)
    parser.add_argument("--n-walkers", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0, help="irreversibility strength")
    parser.add_argument("--rho", type=float, default=0.0, help="0 uses the warm-up radius")
    parser.add_argument("--geometry", choices=["warmup", "identity"], default="warmup")
    parser.add_argument("--prior-scale", type=float, default=2.0)
    parser.add_argument("--warmup-iter", type=int, default=400)
    parser.add_argument("--warmup-walkers", type=int, default=8)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--calibrate-iter", type=int, default=200)
    parser.add_argument("--calibrate-walkers", type=int, default=6)
    parser.add_argument("--reference-iter", type=int, default=20000)
    parser.add_argument("--refresh-reference", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skew-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    for name in args.datasets:
        run_dataset(name, args)


if __name__ == "__main__":
    main()
