#!/usr/bin/env python3
"""Exact-theory sweeps: how much can a skew-symmetric J accelerate?

Everything here comes from the closed-form second-moment analysis in
``skewanchor.analysis`` -- no Monte Carlo error at all.  ``exp2`` then checks
these predictions by simulation.

The comparison is at **equal discretisation bias and equal cost per iteration**.
Adding J speeds the dynamics up but forces a smaller stepsize, so comparing at a
common stepsize would flatter or penalise J arbitrarily.  For each J we solve for
the stepsize that puts the chain's stationary covariance at a fixed relative
error, then compare the per-iteration rates ``-log rho(L)``.

Columns:
  gap        continuous-time rate ``min Re lambda(B)``
  ceiling    ``Tr(B)/d`` -- the best any skew J can reach, since ``Tr(J A) = 0``
  eta        equal-bias stepsize
  rate       per-iteration rate at that stepsize
  speedup    ``rate(J) / rate(0)``
"""

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skewanchor import analysis, runner, skew  # noqa: E402
from skewanchor.targets import anisotropic_student_t, isotropic_polynomial  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "data")

FRACS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 1.0, 1.3, 1.7, 2.5, 4.0)
FIXED_DELTAS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)


def describe(target):
    th = target.theory()
    return {"d": target.d, "nu": target.nu, "beta": target.beta, "iota": target.iota,
            "kappa": target.cond, "Sigma_evals": target.Sigma_evals.tolist(),
            "corollary13_satisfied": bool(th["corollary13_satisfied"]),
            "corollary13_beta_threshold": th["corollary13_beta_threshold"],
            "theorem3_satisfied": bool(th["theorem3_satisfied"])}


def _row(target, J, kind, delta, bias, base_rate):
    eta = analysis.eta_for_bias(target, J, bias)
    rate = analysis.ms_rate(target, J, eta) if eta > 0 else float("-inf")
    return {"kind": kind, "delta": float(delta),
            "gap": analysis.ou_gap(target, J), "eta": float(eta), "rate": float(rate),
            "speedup": float(rate / base_rate) if base_rate > 0 else float("nan"),
            "eta_max_ms": analysis.max_stable_stepsize(target, J),
            "eta_max_drift": analysis.deterministic_stepsize_limit(target, J)}


def sweep(target, bias):
    """Equal-bias rate along the optimal J direction and for the fixed families."""
    base_eta = analysis.eta_for_bias(target, None, bias)
    base_rate = analysis.ms_rate(target, None, base_eta)
    J_opt = skew.lnp_optimal(target.Sigma_inv)
    norm_opt = float(np.linalg.norm(J_opt, 2)) if target.d > 1 else 0.0

    rows = [_row(target, None, "none", 0.0, bias, base_rate)]
    for f in FRACS:
        if f == 0.0 or norm_opt == 0.0:
            continue
        rows.append(_row(target, J_opt * f, "optimal_direction", f * norm_opt, bias, base_rate))
    for kind in ("cyclic", "pairwise", "eigen"):
        for delta in FIXED_DELTAS:
            J = skew.build(kind, target, delta)
            if np.max(np.abs(J)) == 0.0:
                continue
            rows.append(_row(target, J, kind, delta, bias, base_rate))
    # negative control: a J that commutes with Sigma^{-1} cannot change the gap
    J_comm = skew.commuting(target.Sigma, 2.0)
    if np.max(np.abs(J_comm)) > 0:
        rows.append(_row(target, J_comm, "commuting", 2.0, bias, base_rate))

    return {"base_eta": float(base_eta), "base_rate": float(base_rate),
            "ceiling": analysis.ou_gap_ceiling(target),
            "gap_ceiling_ratio": analysis.ou_gap_ceiling(target) / analysis.ou_gap(target, None),
            "J_optimal_norm": norm_opt, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", type=float, default=0.02)
    ap.add_argument("--biases", type=float, nargs="*", default=[0.005, 0.02, 0.05])
    ap.add_argument("--dims", type=int, nargs="*", default=[2, 3, 5, 10],
                    help="dimensions for the scaling sweep; d=20 costs roughly 10x d=10")
    args = ap.parse_args()

    results = {"bias": args.bias, "sweeps": [], "scaling": [], "isotropic_control": [],
               "bias_sensitivity": []}

    print("=== equal-bias sweeps (bias = %.3g) ===" % args.bias)
    for d, nu, kappa in [(2, 5.0, 100.0), (2, 3.0, 100.0), (2, 5.0, 10.0),
                         (3, 6.0, 100.0), (5, 8.0, 100.0), (10, 12.0, 100.0),
                         (5, 8.0, 1000.0)]:
        t = anisotropic_student_t(d, nu, kappa)
        entry = {"target": describe(t)}
        entry.update(sweep(t, args.bias))
        best = max(entry["rows"], key=lambda r: r["rate"])
        entry["best"] = best
        results["sweeps"].append(entry)
        print(f"d={d:2d} nu={nu:4g} kappa={kappa:6g}: rate {entry['base_rate']:.5f} -> "
              f"{best['rate']:.5f} ({best['speedup']:5.2f}x) via {best['kind']} "
              f"||J||={best['delta']:.2f}; stepsize {entry['base_eta']:.2e} -> {best['eta']:.2e}; "
              f"continuous-gap ceiling {entry['gap_ceiling_ratio']:.1f}x", flush=True)

    print("\n=== scaling in dimension and condition number ===")
    out_path = os.path.join(OUT, "exp1_theory_sweeps.json")
    for d in args.dims:
        for kappa in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
            nu = max(5.0, 2.0 * d)
            t = anisotropic_student_t(d, nu, kappa)
            sw = sweep(t, args.bias)
            best = max(sw["rows"], key=lambda r: r["rate"])
            row = {"d": d, "kappa": kappa, "nu": nu, "base_rate": sw["base_rate"],
                   "best_rate": best["rate"], "speedup": best["speedup"],
                   "best_kind": best["kind"], "best_delta": best["delta"],
                   "base_eta": sw["base_eta"], "best_eta": best["eta"],
                   "gap_ceiling_ratio": sw["gap_ceiling_ratio"]}
            results["scaling"].append(row)
            print(f"d={d:2d} kappa={kappa:7g}: speedup {row['speedup']:6.2f}x "
                  f"(gap ceiling would allow {row['gap_ceiling_ratio']:6.1f}x)", flush=True)
            runner.save_json(results, out_path)   # checkpoint after every row

    print("\n=== negative control: isotropic targets (the paper's Section 6.4) ===")
    for d, iota in [(1, 2.0), (3, 3.0), (5, 4.0), (10, 7.0)]:
        t = isotropic_polynomial(d, iota)
        sw = sweep(t, args.bias)
        best = max(sw["rows"], key=lambda r: r["rate"])
        entry = {"target": describe(t), "base_rate": sw["base_rate"],
                 "best_rate": best["rate"], "speedup": best["speedup"],
                 "gap": analysis.ou_gap(t, None), "ceiling": sw["ceiling"]}
        results["isotropic_control"].append(entry)
        print(f"isotropic d={d:2d} iota={iota:g} (nu={t.nu:g}): gap={entry['gap']:.3f} "
              f"ceiling={entry['ceiling']:.3f} best speedup={entry['speedup']:.4f}x")

    print("\n=== sensitivity to the bias level held fixed ===")
    t = anisotropic_student_t(2, 5.0, 100.0)
    for b in args.biases:
        sw = sweep(t, b)
        best = max(sw["rows"], key=lambda r: r["rate"])
        results["bias_sensitivity"].append({"bias": b, "speedup": best["speedup"],
                                            "best_delta": best["delta"],
                                            "base_rate": sw["base_rate"],
                                            "best_rate": best["rate"]})
        print(f"bias={b:.3g}: speedup={best['speedup']:.2f}x at ||J||={best['delta']:.2f}")

    path = runner.save_json(results, out_path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
