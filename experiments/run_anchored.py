#!/usr/bin/env python3
"""Anchored Langevin on a non-differentiable target, with and without a rotation.

The prior becomes Laplace instead of Gaussian, so the potential has a kink at
every ``w_j = 0``:

    U(w)   = U_nll(w) + lambda ||w||_1                     (the target)
    U_0(w) = U_nll(w) + lambda sum_j sqrt(w_j^2 + delta^2)  (the smooth anchor)
    a(w)   = exp(U(w) - U_0(w)) in (0, 1]                   (the clock)

Anchored Langevin (arXiv:2509.19455) follows ``grad U_0`` and scales the
diffusion by ``a``; the composition with a skew term is derived in
:mod:`nds.anchored`.  Three fields are compared, plus two baselines that show
what the anchor is for:

* ``anchor`` -- drop the clock and sample ``exp(-U_0)``, i.e. treat the smoothing
  as the model;
* ``fd`` -- keep the clock off and difference the non-differentiable potential,
  which targets an ``eps``-Huber smoothing of it.

The gold standard is random-walk Metropolis on ``U`` itself: no gradients, no
smoothing, exact whatever the kink does.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nds.anchored import (  # noqa: E402
    LassoLogistic,
    reference_posterior_rwm,
    run_anchored_chain,
)
from nds.data import PRETTY_NAMES, load  # noqa: E402
from nds.design import calibrate_amplitude, explicit_step_size, paired_skew, spectral_report  # noqa: E402
from nds.metrics import iterations_to_reach  # noqa: E402
from nds.sampler import Geometry, warm_up  # noqa: E402
from nds.skew import ConstantSkew, GatedSkew, ZeroSkew  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def get_reference(name: str, ds, target: LassoLogistic, args) -> dict:
    tag = f"reference_lasso_{name}_p{args.penalty:g}"
    path = os.path.join(RESULTS, f"{tag}.npz")
    if os.path.exists(path) and not args.refresh_reference:
        blob = np.load(path)
        return {k: blob[k] for k in blob.files}
    ref = reference_posterior_rwm(
        target, ds.X_test, ds.y_test, n_iter=args.reference_iter,
        n_warmup=args.reference_iter // 5, n_chains=8, seed=args.seed,
    )
    out = {
        "accuracy": np.array(ref["accuracy"]),
        "loss": np.array(ref["loss"]),
        "accuracy_halves": np.array(ref["accuracy_halves"]),
        "loss_halves": np.array(ref["loss_halves"]),
        "predictive": ref["predictive"],
        "posterior_mean": ref["posterior_mean"],
        "posterior_cov": ref["posterior_cov"],
        "acceptance": np.array(ref["acceptance"]),
        "n_samples": np.array(ref["n_samples"]),
    }
    np.savez(path, **out)
    return out


def run_dataset(name: str, args) -> dict:
    started = time.time()
    ds = load(name, seed=args.split_seed)
    target = LassoLogistic(ds.X_train, ds.y_train, penalty=args.penalty, delta=args.delta)
    ref = get_reference(name, ds, target, args)
    ref_acc, ref_loss = float(ref["accuracy"]), float(ref["loss"])
    ref_mean = ref["posterior_mean"]
    ref_metric = np.linalg.pinv(ref["posterior_cov"])
    ref_l1 = float(np.abs(ref_mean[1:]).sum())
    print(f"[{name}] reference accuracy {ref_acc:.4f} loss {ref_loss:.4f} "
          f"halves {np.round(ref['loss_halves'], 4).tolist()}", flush=True)

    # warm-up runs Metropolis on the true, non-differentiable U, so it is exact
    # however rough the difference-quotient drift is
    pilot = warm_up(target, n_iter=args.warmup_iter, n_walkers=args.warmup_walkers,
                    seed=args.seed, n_rounds=args.warmup_rounds, shrinkage=args.shrinkage)
    H_hat, m_hat = pilot["geometry"].D_inv, pilot["mean_state"]
    identity = Geometry.identity(target.d)
    r0 = float(np.sqrt(max(m_hat @ (H_hat @ m_hat), 1e-12)))

    def make_constant(amplitude):
        J, _ = paired_skew(H_hat, amplitude=amplitude)
        return ConstantSkew(J, 1.0), J

    def make_gated(amplitude, drop=False):
        J, _ = paired_skew(H_hat, amplitude=amplitude)
        field = GatedSkew(J, 1.0, radius=args.gate_fraction * r0, width=args.gate_width * r0,
                          metric=H_hat, center=m_hat, drop_correction=drop)
        return field, J

    calibrations = {}
    for key, factory in (("constant", make_constant), ("state", make_gated)):
        calibrations[key] = calibrate_amplitude(
            target, H_hat, m_hat, make_field=factory, amplitudes=tuple(args.amplitude_ladder),
            safety=args.safety, n_iter=args.calibrate_iter, n_walkers=16, seed=args.seed,
            runner=run_anchored_chain,
        )
        print(f"[{name}] {key}: amplitude {calibrations[key]['amplitude']:g} from "
              + ", ".join(f"{r['amplitude']:g}:{r['score']:.2f}"
                          for r in calibrations[key]["ladder"]), flush=True)

    a_const = calibrations["constant"]["amplitude"]
    a_state = calibrations["state"]["amplitude"]
    J_const, design = paired_skew(H_hat, amplitude=a_const)
    J_state, design_state = paired_skew(H_hat, amplitude=a_state)
    h_zero = explicit_step_size(H_hat, None, safety=args.safety)
    h_const = explicit_step_size(H_hat, J_const, safety=args.safety)
    h_state = explicit_step_size(H_hat, J_state, safety=args.safety)

    runs = [
        ("zero", ZeroSkew(), h_zero, "anchored"),
        ("constant", make_constant(a_const)[0], h_const, "anchored"),
        ("state", make_gated(a_state)[0], h_state, "anchored"),
        ("state_nocorr", make_gated(a_state, drop=True)[0], h_state, "anchored"),
        # baselines: what happens without the clock
        ("anchor_only", ZeroSkew(), h_zero, "anchor"),
        ("fd_nonsmooth", ZeroSkew(), h_zero, "fd"),
    ]

    arrays, summary = {}, []
    for key, field, h, scheme in runs:
        res = run_anchored_chain(
            target, field, h, args.n_iter, n_walkers=args.n_walkers, seed=args.seed + 101,
            X_eval=ds.X_test, y_eval=ds.y_test, geometry=identity, scheme=scheme,
            ref_mean=ref_mean, ref_metric=ref_metric,
        )
        arrays[f"accuracy_{key}"] = res.accuracy
        arrays[f"loss_{key}"] = res.loss
        arrays[f"error_{key}"] = res.mean_error
        arrays[f"potential_{key}"] = res.potential
        acc, loss = res.accuracy.mean(axis=1), res.loss.mean(axis=1)
        mean_hat = res.posterior_mean.mean(axis=1)
        summary.append({
            "dataset": name,
            "variant": key,
            "scheme": scheme,
            "amplitude": {"constant": a_const, "state": a_state,
                          "state_nocorr": a_state}.get(key, 0.0),
            "step_size": h,
            "mean_clock": res.meta["mean_clock"],
            "accuracy_final": float(acc[-1]),
            "accuracy_gap": abs(float(acc[-1]) - ref_acc),
            "loss_final": float(loss[-1]),
            "loss_gap": float(loss[-1]) - ref_loss,
            "iters_to_acc_band": iterations_to_reach(acc, ref_acc, 0.005),
            "iters_to_loss_band": iterations_to_reach(loss, ref_loss, 0.01 * ref_loss),
            "mean_error_final": float(res.mean_error[-1].mean()),
            "l1_of_mean": float(np.abs(mean_hat[1:]).sum()),
            "l1_of_mean_vs_reference": float(np.abs(mean_hat[1:]).sum() - ref_l1),
        })
        print(f"[{name}] {key:13s} {scheme:8s} h={h:.4g} clock={res.meta['mean_clock']:.3f} "
              f"acc={acc[-1]:.4f} loss={loss[-1]:.4f} err={res.mean_error[-1].mean():.3f} "
              f"({time.time() - started:.0f}s)", flush=True)

    meta = {
        "dataset": name,
        "pretty_name": PRETTY_NAMES[name],
        "dim": int(target.d),
        "n_train": int(len(ds.y_train)),
        "n_test": int(len(ds.y_test)),
        "penalty": args.penalty,
        "delta": args.delta,
        "clock_floor": target.clock_floor,
        "reference_accuracy": ref_acc,
        "reference_loss": ref_loss,
        "reference_accuracy_halves": np.atleast_1d(ref["accuracy_halves"]).tolist(),
        "reference_loss_halves": np.atleast_1d(ref["loss_halves"]).tolist(),
        "reference_l1_of_mean": ref_l1,
        "amplitude": {"constant": a_const, "state": a_state},
        "amplitude_calibration": calibrations,
        "step_sizes": {"zero": h_zero, "constant": h_const, "state": h_state},
        "gate_radius": args.gate_fraction * r0,
        "whitened_start_radius": r0,
        "n_iter": args.n_iter,
        "n_walkers": args.n_walkers,
        "safety": args.safety,
        "design": {k: (float(v) if np.ndim(v) == 0 else np.asarray(v).tolist())
                   for k, v in design.items()},
        "design_state": {k: (float(v) if np.ndim(v) == 0 else np.asarray(v).tolist())
                         for k, v in design_state.items()},
        "summary": summary,
        "variant_order": ["zero", "constant", "state"],
        "runtime_seconds": time.time() - started,
    }
    np.savez_compressed(os.path.join(RESULTS, f"traces_{name}_anchored.npz"),
                        meta=json.dumps(meta), **arrays)
    with open(os.path.join(RESULTS, f"summary_{name}_anchored.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[{name}] done in {meta['runtime_seconds']:.0f}s", flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["titanic", "magic"])
    parser.add_argument("--penalty", type=float, default=5.0,
                        help="lambda of the L1 penalty (inverse Laplace scale)")
    parser.add_argument("--delta", type=float, default=0.02,
                        help="anchor smoothing; the clock floor is exp(-lambda (d-1) delta)")
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--n-walkers", type=int, default=24)
    parser.add_argument("--safety", type=float, default=0.15)
    parser.add_argument("--amplitude-ladder", nargs="+", type=float,
                        default=[0.0, 0.2, 0.5, 1.0])
    parser.add_argument("--calibrate-iter", type=int, default=200)
    parser.add_argument("--gate-fraction", type=float, default=0.75)
    parser.add_argument("--gate-width", type=float, default=0.1)
    parser.add_argument("--warmup-iter", type=int, default=300)
    parser.add_argument("--warmup-walkers", type=int, default=8)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    parser.add_argument("--reference-iter", type=int, default=100000)
    parser.add_argument("--refresh-reference", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    for name in args.datasets:
        run_dataset(name, args)


if __name__ == "__main__":
    main()
