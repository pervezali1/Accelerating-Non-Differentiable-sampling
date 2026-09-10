#!/usr/bin/env python3
"""Where the acceleration goes: the SDE speeds up much more than the algorithm.

Two exact rates are computed for each ``J``:

``continuous``  the rate at which the *SDE's* second moment reaches equilibrium.
                This is the acceleration the skew perturbation buys in the
                dynamics, with no discretisation in the picture.
``equal-bias``  the per-iteration rate of the Euler-Maruyama chain, run at the
                stepsize that holds its stationary covariance at a fixed
                relative error.  This is what a practitioner actually gets.

The gap between them is the price of the smaller stepsize a larger drift forces,
and it is the single most important caveat in this whole extension: the
continuous-time spectral gap, which is what the classical non-reversible
literature optimises, overstates the realisable gain by roughly an order of
magnitude here.
"""

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skewanchor import analysis, runner, skew  # noqa: E402
from skewanchor.targets import anisotropic_student_t, isotropic_polynomial  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "data")

FRACS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4, 1.7, 2.4, 3.0, 4.0, 6.0)


def decompose(target, bias, fracs=FRACS):
    J_opt = skew.lnp_optimal(target.Sigma_inv)
    norm_opt = float(np.linalg.norm(J_opt, 2))
    r0_c = analysis.continuous_ms_rate(target, None)
    e0 = analysis.eta_for_bias(target, None, bias)
    r0_d = analysis.ms_rate(target, None, e0)
    rows = []
    for f in fracs:
        J = None if f == 0.0 else J_opt * f
        eta = analysis.eta_for_bias(target, J, bias)
        rows.append({
            "frac": f, "J_norm": f * norm_opt,
            "continuous_rate": analysis.continuous_ms_rate(target, J),
            "continuous_speedup": analysis.continuous_ms_rate(target, J) / r0_c,
            "eta": eta,
            "discrete_rate": analysis.ms_rate(target, J, eta) if eta > 0 else float("-inf"),
            "discrete_speedup": (analysis.ms_rate(target, J, eta) / r0_d) if eta > 0 and r0_d > 0
            else float("nan"),
            "gap": analysis.ou_gap(target, J),
            "rho_J": analysis.rho_J(target, J),
            "assumption12": analysis.assumption12(target, J),
        })
    return {"J_optimal_norm": norm_opt, "base_continuous_rate": r0_c,
            "base_discrete_rate": r0_d, "base_eta": e0, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", type=float, default=0.02)
    args = ap.parse_args()

    results = {"bias": args.bias, "targets": []}
    settings = [(2, 5.0, 100.0), (2, 5.0, 1000.0), (3, 6.0, 100.0),
                (5, 8.0, 100.0), (10, 12.0, 100.0)]

    for d, nu, kappa in settings:
        t = anisotropic_student_t(d, nu, kappa)
        dec = decompose(t, args.bias)
        dec["target"] = {"d": d, "nu": nu, "kappa": kappa, "beta": t.beta}
        results["targets"].append(dec)
        best_c = max(dec["rows"], key=lambda r: r["continuous_rate"])
        best_d = max(dec["rows"], key=lambda r: r["discrete_rate"])
        print(f"d={d:2d} nu={nu:4g} kappa={kappa:6g}:")
        print(f"  {'|J|':>8} {'SDE rate':>10} {'x':>7} {'eta':>10} {'iter rate':>10} {'x':>7} "
              f"{'rho_J':>8} {'Asm12':>6}")
        for r in dec["rows"]:
            print(f"  {r['J_norm']:8.2f} {r['continuous_rate']:10.3f} "
                  f"{r['continuous_speedup']:7.2f} {r['eta']:10.2e} {r['discrete_rate']:10.5f} "
                  f"{r['discrete_speedup']:7.2f} {r['rho_J']:8.2f} "
                  f"{str(r['assumption12']['assumption12_holds']):>6}")
        print(f"  best SDE speed-up {best_c['continuous_speedup']:.1f}x at |J|={best_c['J_norm']:.2f}; "
              f"best per-iteration speed-up {best_d['discrete_speedup']:.2f}x at "
              f"|J|={best_d['J_norm']:.2f}\n")

    print("negative control (isotropic):")
    for d, iota in [(3, 3.0), (5, 4.0)]:
        t = isotropic_polynomial(d, iota)
        dec = decompose(t, args.bias, fracs=(0.0,))
        Jc = skew.cyclic(d, 4.0)
        rc = analysis.continuous_ms_rate(t, Jc)
        results["targets"].append({"target": {"d": d, "iota": iota, "isotropic": True},
                                   "base_continuous_rate": dec["base_continuous_rate"],
                                   "with_J_continuous_rate": rc})
        print(f"  d={d} iota={iota:g}: SDE rate {dec['base_continuous_rate']:.4f} (J=0) vs "
              f"{rc:.4f} (|J|=4) -- identical, as the radial-symmetry argument requires")

    print("\nwrote", runner.save_json(results, os.path.join(OUT, "exp5_rate_decomposition.json")))


if __name__ == "__main__":
    main()
