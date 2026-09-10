#!/usr/bin/env python3
"""Main simulation experiment: skew acceleration on anisotropic heavy tails.

Mirrors Section 6.4 of the paper -- an ensemble of particles started from a
prior, 2-Wasserstein distance to the exact target against iteration count --
but on an *anisotropic* Student-t, which is the regime where a skew-symmetric
perturbation can do anything at all.

Fairness.  Three protocols are run and all three are reported:

  ``common``   every method at one common stepsize;
  ``equalbias`` every anchored variant at the stepsize that puts its stationary
               covariance at the same relative error (from the exact analysis);
  ``tuned``    every method at its own best stepsize from a grid, chosen by the
               accuracy reached inside the iteration budget.

Each curve is accompanied by ``w2_floor``: the same statistic evaluated on
exact i.i.d. draws.  No sampler can go below it, so a curve touching the floor
has converged as far as the measurement can see.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skewanchor import analysis, metrics, runner, skew  # noqa: E402
from skewanchor.targets import anisotropic_student_t  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "data")

SETTINGS = {
    "d2_nu5_k100": dict(d=2, nu=5.0, kappa=100.0),
    "d2_nu3_k100": dict(d=2, nu=3.0, kappa=100.0),
    "d5_nu8_k100": dict(d=5, nu=8.0, kappa=100.0),
    "d10_nu12_k100": dict(d=10, nu=12.0, kappa=100.0),
    "d2_nu5_k1000": dict(d=2, nu=5.0, kappa=1000.0),
}


def iterations_to(iters, values, threshold):
    """First recorded iteration at which ``values`` drops below ``threshold``."""
    v = np.asarray(values, dtype=np.float64)
    hit = np.where(v <= threshold)[0]
    return int(np.asarray(iters)[hit[0]]) if hit.size else -1


def build_methods(target, bias, fracs):
    """The list of (name, label, J, eta) to compare under the equal-bias protocol."""
    J_opt = skew.lnp_optimal(target.Sigma_inv)
    norm_opt = float(np.linalg.norm(J_opt, 2))
    out = []
    eta0 = analysis.eta_for_bias(target, None, bias)
    out.append(("anchored", "anchored (J=0)", None, eta0))
    for f in fracs:
        J = J_opt * f
        eta = analysis.eta_for_bias(target, J, bias)
        out.append(("skew_anchored", f"skew-anchored |J|={f * norm_opt:.2f}", J, eta))
    return out, J_opt, norm_opt, eta0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="d2_nu5_k100", choices=sorted(SETTINGS))
    ap.add_argument("--n", type=int, default=5000, help="particles (paper uses 5000)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--reps", type=int, default=10, help="paper averages over 100")
    ap.add_argument("--pilot-reps", type=int, default=3)
    ap.add_argument("--bias", type=float, default=0.02)
    ap.add_argument("--prior", default="normal10", choices=["normal10", "uniform5"])
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.7])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = SETTINGS[args.setting]
    target = anisotropic_student_t(**cfg)
    print(target)
    th = target.theory()
    print(f"  Theorem 3 condition (beta > d/2): {th['theorem3_satisfied']}")
    print(f"  Corollary 13 condition (beta > d kappa/2 = {th['corollary13_beta_threshold']:.1f}): "
          f"{th['corollary13_satisfied']}  <- the paper's W2 theorem does not cover this target")

    rng = np.random.default_rng(0)
    floor_mean, floor_std = metrics.w2_reference_floor(target, args.n, rng, n_rep=25)
    print(f"  W2 estimator floor at n={args.n}: {floor_mean:.4f} +- {floor_std:.4f}")

    methods, J_opt, norm_opt, eta0 = build_methods(target, args.bias, args.fracs)
    print(f"  optimal-direction |J| = {norm_opt:.3f}; equal-bias stepsize at J=0: {eta0:.3e}")

    results = {
        "setting": args.setting, "config": cfg, "n": args.n, "steps": args.steps,
        "reps": args.reps, "bias": args.bias, "prior": args.prior,
        "w2_floor": floor_mean, "w2_floor_std": floor_std,
        "J_optimal_norm": norm_opt, "J_optimal": J_opt.tolist(),
        "theory": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                   for k, v in th.items()},
        "equalbias": [], "common": [], "tuned": [], "baseline_grid": [],
    }

    record_at = runner.log_schedule(args.steps, 45)

    # ---------------------------------------------------------- equal bias
    print("\n--- protocol: equal discretisation bias, equal cost per iteration ---")
    for name, label, J, eta in methods:
        t0 = time.time()
        agg = runner.run_repeated(target, name, eta, args.n, args.steps, args.reps,
                                  prior=args.prior, J=J, base_seed=11, record_at=record_at)
        agg["label"] = label
        agg["gap"] = analysis.ou_gap(target, J)
        agg["predicted_rate"] = analysis.ms_rate(target, J, eta)
        agg["predicted_bias"] = analysis.covariance_bias(target, J, eta)
        if agg["iters"]:
            agg["iters_to_2xfloor"] = iterations_to(agg["iters"], agg["w2"]["mean"], 2 * floor_mean)
            agg["iters_to_slow10"] = iterations_to(agg["iters"], agg["slow"]["mean"], 0.10)
            agg["final_w2"] = agg["w2"]["mean"][-1]
        results["equalbias"].append(agg)
        print(f"  {label:28s} eta={eta:.2e} predicted_rate={agg['predicted_rate']:.5f} "
              f"iters_to_2xfloor={agg.get('iters_to_2xfloor', -1):6d} "
              f"iters_to_slow<10%={agg.get('iters_to_slow10', -1):6d} "
              f"final_W2={agg.get('final_w2', float('nan')):.4f} "
              f"diverged={agg['n_diverged']}/{args.reps} ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------- common stepsize
    print("\n--- protocol: one common stepsize for every method ---")
    eta_common = eta0
    common = [("ula", "ULA", None), ("skew_ula", "skew-ULA", J_opt),
              ("mala", "MALA", None), ("underdamped", "underdamped Langevin", None),
              ("anchored", "anchored (J=0)", None),
              ("skew_anchored", f"skew-anchored |J|={norm_opt:.2f}", J_opt)]
    for name, label, J in common:
        agg = runner.run_repeated(target, name, eta_common, args.n, args.steps,
                                  args.pilot_reps, prior=args.prior, J=J, base_seed=21,
                                  record_at=record_at)
        agg["label"] = label
        if agg["iters"]:
            agg["iters_to_2xfloor"] = iterations_to(agg["iters"], agg["w2"]["mean"], 2 * floor_mean)
            agg["final_w2"] = agg["w2"]["mean"][-1]
        results["common"].append(agg)
        print(f"  {label:28s} eta={eta_common:.2e} iters_to_2xfloor="
              f"{agg.get('iters_to_2xfloor', -1):6d} final_W2={agg.get('final_w2', float('nan')):.4f} "
              f"diverged={agg['n_diverged']}/{args.pilot_reps}")

    # ------------------------------------------------- per-method tuned eta
    print("\n--- protocol: each method at its own best stepsize (grid reported in full) ---")
    grid = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0]
    for name, label, J in common:
        best = None
        for mult in grid:
            eta = eta0 * mult
            agg = runner.run_repeated(target, name, eta, args.n, args.steps,
                                      args.pilot_reps, prior=args.prior, J=J, base_seed=31,
                                      record_at=record_at)
            entry = {"method": name, "label": label, "eta": eta, "mult": mult,
                     "n_diverged": agg["n_diverged"]}
            if agg["iters"]:
                entry["final_w2"] = agg["w2"]["mean"][-1]
                entry["iters_to_2xfloor"] = iterations_to(agg["iters"], agg["w2"]["mean"],
                                                          2 * floor_mean)
                entry["curve_iters"] = agg["iters"]
                entry["curve_w2"] = agg["w2"]["mean"]
            else:
                entry["final_w2"] = float("inf")
                entry["iters_to_2xfloor"] = -1
            results["baseline_grid"].append(entry)
            key = (entry["iters_to_2xfloor"] if entry["iters_to_2xfloor"] > 0 else 10 ** 9,
                   entry["final_w2"])
            if best is None or key < best[0]:
                best = (key, entry)
        results["tuned"].append(best[1])
        e = best[1]
        print(f"  {label:28s} best eta={e['eta']:.2e} ({e['mult']}x) "
              f"iters_to_2xfloor={e['iters_to_2xfloor']:6d} final_W2={e['final_w2']:.4f}")

    out = args.out or os.path.join(OUT, f"exp2_{args.setting}_{args.prior}.json")
    print("\nwrote", runner.save_json(results, out))


if __name__ == "__main__":
    main()
