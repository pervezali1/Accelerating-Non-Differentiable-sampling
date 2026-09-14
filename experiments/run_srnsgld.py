#!/usr/bin/env python3
"""Constrained Bayesian logistic regression in ``d = 9``: PSGLD against SRNSGLD.

The setting is the one used for the skew-reflected non-reversible dynamics:
Titanic (``n = 891``) and MAGIC Gamma Telescope (``n = 19020``), both reduced to
exactly nine standardised features, a uniform prior on the centred ball
``K_r``, walkers started from the uniform law on the centred *unit* ball, and
mini-batch (stochastic) gradients.

Three fields, one step size, one mini-batch stream:

* ``J = 0``      -- projected SGLD;
* constant ``J_a``   -- superdiagonal ``+a``, subdiagonal ``-a``;
* state-dependent ``J_s(x)`` -- block diagonal, three ``3 x 3`` hat maps of the
  coordinate triples ``(x1,x2,x3)``, ``(x4,x5,x6)``, ``(x7,x8,x9)``.

Both skew fields are divergence free, so no correction term is estimated, and
both are applied at a *matched* operator norm ``rho`` so that "constant" and
"state-dependent" mean the same amount of rotation.  The step size comes from
the curvature at the constrained MAP and is shared by all three fields, so a
difference between the curves cannot be a difference of step size -- which is
the mistake the gated experiment in this repository made on its two
high-dimensional datasets.

Every field is scored on the same test split against the same exact reference,
a random-walk Metropolis chain on the constrained posterior.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nds.constrained import (  # noqa: E402
    Ball,
    FramedField,
    MinibatchLogistic,
    curvature_step_size,
    eigen_frame,
    field_from_rho,
    reference_constrained_rwm,
    run_constrained_sgld,
)
from nds.data import PRETTY_NAMES, load_nine  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
BATCH = {"titanic": 256, "magic": 4096}  # chosen so the mini-batch noise is
                                         # about the size of the injected noise


def reference(name: str, model, ball, ds, args, start=None) -> dict:
    """Exact constrained posterior, cached on disk."""
    path = os.path.join(RESULTS, f"reference_ball_{name}_r{ball.radius:g}.npz")
    if os.path.exists(path) and not args.refresh_reference:
        return dict(np.load(path))
    started = time.time()
    ref = reference_constrained_rwm(
        model, ball, ds.X_test, ds.y_test,
        n_iter=args.ref_iter, n_warmup=args.ref_warmup, thin=args.ref_thin,
        seed=args.seed, start=start,
    )
    print(
        f"[{name}] reference: accuracy {ref['accuracy']:.4f} (halves "
        f"{ref['accuracy_halves'][0]:.4f}/{ref['accuracy_halves'][1]:.4f}), "
        f"loss {ref['loss']:.4f}, acceptance {ref['acceptance']:.2f}, "
        f"|mean| {ref['mean_radius']:.3f}, {time.time() - started:.0f}s",
        flush=True,
    )
    np.savez_compressed(path, **ref)
    return ref


def run_dataset(name: str, args) -> dict:
    started = time.time()
    ds = load_nine(name, seed=args.split_seed)
    model = MinibatchLogistic(ds.X_train, ds.y_train, prior_scale=args.prior_scale)
    ball = Ball(args.radius)
    h, geom = curvature_step_size(model, ball, safety=args.safety)
    ref = reference(name, model, ball, ds, args, start=np.asarray(geom["map"], float))
    ref_mean = np.asarray(ref["posterior_mean"], float)
    ref_metric = np.linalg.pinv(np.asarray(ref["posterior_cov"], float))

    if args.step_size is not None:
        h = float(args.step_size)
    batch = args.batch_size or BATCH.get(name, 1024)
    print(
        f"[{name}] d={model.d} n_train={model.n} r={ball.radius} h={h:.3e} "
        f"batch={batch} |MAP_r|={geom['map_radius']:.3f} "
        f"lambda_max={geom['hessian_max']:.1f} cond={geom['condition']:.1f}",
        flush=True,
    )

    frame = None
    if args.frame == "eigen":
        frame = eigen_frame(np.asarray(geom["hessian"], float))

    runs = []
    plans = [("zero", 0.0)]
    for key in ("constant", "state"):
        plans += [(key, rho) for rho in args.rho]
    for key, rho in plans:
        field = field_from_rho(key, model.d, rho, ball.radius)
        if frame is not None and key != "zero":
            field = FramedField(field, frame)
        out = run_constrained_sgld(
            model, field, ball, ds.X_test, ds.y_test,
            n_iter=args.n_iter, n_walkers=args.n_walkers, step_size=h,
            batch_size=batch, seed=args.seed, start_radius=args.start_radius,
            reference_mean=ref_mean, reference_metric=ref_metric,
            grad=args.grad, fd_eps=args.fd_eps,
        )
        out["rho"] = float(rho)
        runs.append(out)
        print(
            f"    {key:9s} rho={rho:4.2f}  accuracy {out['accuracy'][-1]:.4f}  "
            f"loss {out['loss'][-1]:.4f}  error {out['mean_error'][-1]:.4f}  "
            f"boundary {out['boundary_rate']:.3f}  missed {out['missed_reflections']}  "
            f"batch/Brownian noise {out['minibatch_noise_ratio']:.2f}",
            flush=True,
        )

    tag = f"{name}_srnsgld" + ("" if args.frame == "features" else f"_{args.frame}") + args.tag_suffix
    np.savez_compressed(
        os.path.join(RESULTS, f"traces_{tag}.npz"),
        **{
            f"{r['key']}_rho{r['rho']:g}_{k}": r[k]
            for r in runs
            for k in ("accuracy", "loss", "mean_error", "radius")
        },
        reference_accuracy=ref["accuracy"],
        reference_loss=ref["loss"],
    )
    meta = {
        "dataset": name,
        "pretty": PRETTY_NAMES.get(name, name),
        "d": int(model.d),
        "n_train": int(model.n),
        "n_test": int(len(ds.y_test)),
        "features": ds.feature_names,
        "radius": ball.radius,
        "start_radius": args.start_radius,
        "step_size": h,
        "safety": args.safety,
        "batch_size": batch,
        "n_iter": args.n_iter,
        "n_walkers": args.n_walkers,
        "gradient": args.grad,
        "frame": args.frame,
        "geometry": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in geom.items()},
        "reference": {
            "accuracy": float(ref["accuracy"]),
            "loss": float(ref["loss"]),
            "accuracy_halves": np.asarray(ref["accuracy_halves"]).tolist(),
            "loss_halves": np.asarray(ref["loss_halves"]).tolist(),
            "acceptance": float(ref["acceptance"]),
            "mean_radius": float(np.linalg.norm(ref_mean)),
            "posterior_mean": ref_mean.tolist(),
        },
        "runs": [
            {
                "key": r["key"],
                "label": r["label"],
                "rho": r["rho"],
                "amplitude": r["amplitude"],
                "accuracy_final": float(r["accuracy"][-1]),
                "loss_final": float(r["loss"][-1]),
                "mean_error_final": float(r["mean_error"][-1]),
                "mean_radius_final": float(r["radius"][-1]),
                "boundary_rate": r["boundary_rate"],
                "missed_reflections": r["missed_reflections"],
                "minibatch_noise_ratio": r["minibatch_noise_ratio"],
            }
            for r in runs
        ],
        "runtime_seconds": time.time() - started,
    }
    with open(os.path.join(RESULTS, f"summary_{tag}.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[{name}] done in {meta['runtime_seconds']:.0f}s", flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["titanic", "magic"])
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--n-walkers", type=int, default=32)
    parser.add_argument("--radius", type=float, default=2.0,
                        help="radius of the constraint ball K_r")
    parser.add_argument("--start-radius", type=float, default=1.0,
                        help="walkers start uniform on the ball of this radius")
    parser.add_argument("--rho", nargs="+", type=float,
                        default=[0.25, 0.5, 1.0, 2.0, 4.0],
                        help="rotation strengths, as the field's operator norm "
                             "relative to the gradient part of the drift")
    parser.add_argument("--safety", type=float, default=0.15)
    parser.add_argument("--step-size", type=float, default=None,
                        help="override the curvature step-size rule")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--frame", choices=["features", "eigen"], default="features",
                        help="orthonormal frame the skew field is written in: the raw "
                             "feature order, or the Hessian eigenbasis with every 3-block "
                             "pairing a slow direction with a fast one")
    parser.add_argument("--grad", choices=["exact", "fd"], default="exact",
                        help="mini-batch gradient, or central differences of the "
                             "mini-batch potential (derivative free, 2d evaluations)")
    parser.add_argument("--fd-eps", type=float, default=1e-2)
    parser.add_argument("--prior-scale", type=float, default=None,
                        help="Gaussian prior scale; the default is none at all, so "
                             "that the ball is the only prior")
    parser.add_argument("--ref-iter", type=int, default=200000)
    parser.add_argument("--ref-warmup", type=int, default=20000)
    parser.add_argument("--ref-thin", type=int, default=10)
    parser.add_argument("--refresh-reference", action="store_true")
    parser.add_argument("--tag-suffix", default="",
                        help="appended to the output tag, so variants do not overwrite")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    for name in args.datasets:
        run_dataset(name, args)


if __name__ == "__main__":
    main()
