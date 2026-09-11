#!/usr/bin/env python3
"""Accuracy from the ``w = 0`` start when ``J`` is designed from the geometry.

Two things differ from ``run_accuracy.py``, and both are the point.

**The dynamics are unadjusted.**  A Metropolis-Hastings chain satisfies detailed
balance by construction, so the accept/reject step destroys exactly the
irreversibility that ``J`` is there to supply; all that survives of ``J`` is a
worse proposal, which is why the Metropolis-corrected experiment finds no
benefit and a large cost.  The irreversible-Langevin literature works with the
unadjusted dynamics, and so does this run.  Discretisation bias is controlled by
the step-size rule below and reported with the results.

**``J`` is designed, not drawn at random.**  ``nds.design.paired_skew`` couples
the ``k``-th slowest eigendirection of the warm-up Hessian estimate to the
``k``-th fastest, at the amplitude where the pair's two relaxation rates collide
at their mean.  That removes the slowest direction as the bottleneck, which is
the only thing a skew term can do: it cannot change ``trace((D + J) H)``.

Every field runs with ``D = I``, from ``w = 0``, with common walker seeds, and at
its own step size ``h = 2 safety / max Re mu``.  That rule gives the reversible
baseline its own classical optimum and fixes the worst-case variance inflation
of the unadjusted chain at ``2 / (2 - 2 safety)``, so the fields are compared at
matched discretisation bias rather than at a matched step size.  A reversible
run that uses the same warm-up covariance as a *preconditioner* is recorded
alongside as ``precond``: it is the method to beat if you are willing to put the
covariance in the drift, and the figures show it.
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
from nds.design import (  # noqa: E402
    calibrate_amplitude,
    discrete_decay,
    explicit_step_size,
    paired_skew,
    spectral_report,
    variance_inflation,
)
from nds.metrics import ess, iterations_to_reach  # noqa: E402
from nds.sampler import Geometry, run_chain, stable_step_size, warm_up  # noqa: E402
from nds.skew import ConstantSkew, GatedSkew, ZeroSkew  # noqa: E402
from nds.target import LogisticPosterior  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_dataset(name: str, args) -> dict:
    started = time.time()
    ds = load(name, seed=args.split_seed)
    target = LogisticPosterior(ds.X_train, ds.y_train, prior_scale=args.prior_scale)
    ref = np.load(os.path.join(RESULTS, f"reference_{name}.npz"))
    ref_acc, ref_mean = float(ref["accuracy"]), ref["posterior_mean"]
    ref_metric = np.linalg.pinv(ref["posterior_cov"])

    pilot = warm_up(
        target, n_iter=args.warmup_iter, n_walkers=args.warmup_walkers,
        seed=args.seed, n_rounds=args.warmup_rounds, shrinkage=args.shrinkage,
    )
    precond, m_hat = pilot["geometry"], pilot["mean_state"]
    H_hat = precond.D_inv  # inverse of the warm-up covariance
    identity = Geometry.identity(target.d)

    identity_geo = identity
    r0 = float(np.sqrt(max(m_hat @ (H_hat @ m_hat), 1e-12)))  # whitened distance 0 -> bulk
    gate_radius = args.gate_fraction * r0

    def make_constant(amplitude: float):
        J_a, _ = paired_skew(H_hat, amplitude=amplitude)
        return ConstantSkew(J_a, 1.0), J_a

    def make_state(amplitude: float, drop: bool = False):
        """The designed rotation, gated off while the walker is still descending.

        The amplitude comes from a quadratic model of the bulk, so applying it in
        the far field -- where that model does not hold and the gradient is
        largest -- is what makes a constant rotation deflect the descent and
        overshoot, and it is why the calibration has to back the constant field
        off on a near-separable posterior.  A gate at a fraction of the whitened
        distance from ``w = 0`` to the bulk separates the two regimes: outside
        it the chain is the reversible one, but running at the *design's* step
        size, which is the larger of the two; inside it the full rotation
        applies.  So the gated field should not be behind the baseline at any
        iteration, which a gradual profile cannot promise.
        """
        J_a, _ = paired_skew(H_hat, amplitude=amplitude)
        field = GatedSkew(
            J_a, 1.0, radius=args.gate_fraction * r0, width=args.gate_width * r0,
            metric=H_hat, center=m_hat, drop_correction=drop,
        )
        return field, J_a

    calibrations = {}
    for key, factory in (("constant", make_constant), ("state", make_state)):
        if args.amplitude is not None:
            calibrations[key] = {"amplitude": float(args.amplitude), "ladder": []}
            continue
        calibrations[key] = calibrate_amplitude(
            target, H_hat, m_hat, make_field=factory,
            amplitudes=tuple(args.amplitude_ladder), safety=args.safety,
            n_iter=args.calibrate_iter, seed=args.seed, guard_iter=args.guard_iter,
        )
        print(f"[{name}] {key}: calibrated amplitude {calibrations[key]['amplitude']:g} from "
              + ", ".join(f"{r['amplitude']:g}:{r['score']:.2f}"
                          for r in calibrations[key]["ladder"]), flush=True)

    a_const = calibrations["constant"]["amplitude"]
    a_state = calibrations["state"]["amplitude"]
    J, design = paired_skew(H_hat, amplitude=a_const)
    J_state, design_state = paired_skew(H_hat, amplitude=a_state)
    h_zero = explicit_step_size(H_hat, None, safety=args.safety)
    h_const = explicit_step_size(H_hat, J, safety=args.safety)
    h_state = explicit_step_size(H_hat, J_state, safety=args.safety)

    fields = [
        ("zero", ZeroSkew(), h_zero, identity_geo),
        ("constant", make_constant(a_const)[0], h_const, identity_geo),
        ("state", make_state(a_state)[0], h_state, identity_geo),
        ("state_nocorr", make_state(a_state, drop=True)[0], h_state, identity_geo),
        # the stronger reversible method: same covariance, used as a preconditioner
        ("precond", ZeroSkew(), explicit_step_size(H_hat, None, args.safety, precond.D), precond),
    ]

    # size of the correction relative to the reversible drift, over the bulk
    rng = np.random.default_rng(args.seed + 7)
    W_bulk = m_hat[:, None] + precond.L @ rng.standard_normal((target.d, 64))
    G_bulk = target.fd_grad(W_bulk)
    field_probe = make_state(a_state)[0]
    gamma_ratio = float(
        np.median(
            np.linalg.norm(field_probe.divergence(W_bulk), axis=0)
            / np.maximum(np.linalg.norm(G_bulk, axis=0), 1e-30)
        )
    )

    arrays, summary = {}, []
    for key, field, h_design_rule, geo in fields:
        guard = stable_step_size(
            target, field, h_design_rule, n_iter=args.guard_iter, n_walkers=8,
            seed=args.seed + 5, geometry=geo,
        )
        h = guard["step_size"]
        res = run_chain(
            target, field, h, args.n_iter, n_walkers=args.n_walkers,
            seed=args.seed + 101, X_eval=ds.X_test, y_eval=ds.y_test, geometry=geo,
            metropolis=False, ref_mean=ref_mean, ref_metric=ref_metric,
        )
        arrays[f"accuracy_{key}"] = res.accuracy
        arrays[f"loss_{key}"] = res.loss
        arrays[f"error_{key}"] = res.mean_error
        arrays[f"potential_{key}"] = res.potential
        curve = res.accuracy.mean(axis=1)
        J_used = {"constant": J, "state": J_state, "state_nocorr": J_state}.get(key)
        summary.append({
            "dataset": name,
            "variant": key,
            "step_size": h,
            "accuracy_at_100": float(curve[min(100, args.n_iter)]),
            "accuracy_final": float(curve[-1]),
            "loss_final": float(res.loss[-1].mean()),
            "iters_to_ref_0.005": iterations_to_reach(curve, ref_acc, 0.005),
            "iters_to_ref_0.01": iterations_to_reach(curve, ref_acc, 0.01),
            "mean_error_final": float(res.mean_error[-1].mean()),
            "potential_final": float(res.potential[-1].mean()),
            "ess_potential_second_half": ess(res.potential[args.n_iter // 2 :]),
            "predicted_decay": discrete_decay(H_hat, J_used, h, geo.D),
            "variance_inflation": variance_inflation(H_hat, J_used, h, geo.D),
            "step_size_from_rule": h_design_rule,
            "step_reductions": guard["reductions"],
            "diverged": bool(~np.isfinite(res.potential[-1]).all()),
        })
        print(
            f"[{name}] {key:13s} h={h:.4g} final={curve[-1]:.4f} "
            f"reach={summary[-1]['iters_to_ref_0.005']:5d} err={res.mean_error[-1].mean():.3f} "
            f"decay={summary[-1]['predicted_decay']:.5f} ({time.time() - started:.0f}s)",
            flush=True,
        )

    meta = {
        "dataset": name,
        "pretty_name": PRETTY_NAMES[name],
        "dim": int(target.d),
        "n_train": int(len(ds.y_train)),
        "n_test": int(len(ds.y_test)),
        "reference_accuracy": ref_acc,
        "geometry": "designed",
        "unadjusted": True,
        "amplitude": {"constant": a_const, "state": a_state},
        "amplitude_calibration": calibrations,
        "safety": args.safety,
        "gate_radius": gate_radius,
        "gate_width": args.gate_width * r0,
        "gamma_over_drift_in_bulk": gamma_ratio,
        "n_iter": args.n_iter,
        "n_walkers": args.n_walkers,
        "prior_scale": args.prior_scale,
        "step_sizes": {"zero": h_zero, "constant": h_const, "state": h_state},
        "gate_fraction": args.gate_fraction,
        "whitened_start_radius": r0,
        "design": {
            k: (float(v) if np.isscalar(v) or np.ndim(v) == 0 else np.asarray(v).tolist())
            for k, v in design.items()
        },
        "design_state": {
            k: (float(v) if np.isscalar(v) or np.ndim(v) == 0 else np.asarray(v).tolist())
            for k, v in design_state.items()
        },
        "design_on_reference": {
            k: (float(v) if np.isscalar(v) or np.ndim(v) == 0 else np.asarray(v).tolist())
            for k, v in spectral_report(np.linalg.pinv(ref["posterior_cov"]), J_state).items()
        },
        "summary": summary,
        "variant_order": ["zero", "constant", "state", "state_nocorr"],
        "runtime_seconds": time.time() - started,
    }
    tag = f"{name}_designed" if args.amplitude is None else f"{name}_designed_fixed{a_const:g}"
    np.savez_compressed(
        os.path.join(RESULTS, f"traces_{tag}.npz"), meta=json.dumps(meta), **arrays
    )
    with open(os.path.join(RESULTS, f"summary_{tag}.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[{name}] done in {meta['runtime_seconds']:.0f}s", flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["titanic", "magic", "breast_cancer", "spambase"]
    )
    parser.add_argument("--n-iter", type=int, default=600)
    parser.add_argument("--n-walkers", type=int, default=32)
    parser.add_argument("--amplitude", type=float, default=None,
                        help="fixed block coupling, in units of the collision value; "
                             "omit to calibrate it from a short pilot")
    parser.add_argument("--amplitude-ladder", nargs="+", type=float,
                        default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0],
                        help="amplitudes the calibration tries")
    parser.add_argument("--calibrate-iter", type=int, default=300,
                        help="pilot length of the amplitude calibration")
    parser.add_argument("--safety", type=float, default=0.15,
                        help="step size is 2 * safety / max Re mu")
    parser.add_argument("--shrinkage", type=float, default=0.05,
                        help="warm-up covariance shrinkage; less of it estimates the "
                             "largest curvature better, which is what sets the step size")
    parser.add_argument("--guard-iter", type=int, default=150,
                        help="pilot length of the unadjusted stability guard")
    parser.add_argument("--gate-fraction", type=float, default=0.75,
                        help="gate radius of the state-dependent field, as a fraction of "
                             "the whitened distance from w = 0 to the bulk")
    parser.add_argument("--gate-width", type=float, default=0.1,
                        help="gate transition width, in the same units")
    parser.add_argument("--prior-scale", type=float, default=2.0)
    parser.add_argument("--warmup-iter", type=int, default=600)
    parser.add_argument("--warmup-walkers", type=int, default=8)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    for name in args.datasets:
        run_dataset(name, args)


if __name__ == "__main__":
    main()
