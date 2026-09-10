#!/usr/bin/env python3
"""Single-chain efficiency: does the skew term reduce the asymptotic variance?

The ensemble experiments measure how fast the *law* of the chain reaches the
target.  Classical non-reversible theory (Hwang, Hwang-Sheu and Sheu; Rey-Bellet
and Spiliopoulos) makes a second, different prediction: the asymptotic variance
of a time average is reduced.  That is what this experiment measures, on the
slow direction of ``Sigma`` -- the coordinate a reversible sampler mixes in
last.

Reported per method: the integrated autocorrelation time of the slow-direction
coordinate, and the effective sample size per iteration.  Chains are run at the
equal-bias stepsize so the comparison is again at matched accuracy.
"""

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skewanchor import analysis, metrics, runner, samplers, skew  # noqa: E402
from skewanchor.targets import anisotropic_student_t  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "data")


def run_chains(target, method, eta, J, n_chains, n_steps, burn, seed, v_slow):
    """``n_chains`` chains advanced together; returns the slow-direction traces."""
    rng = np.random.default_rng(seed)
    x = target.sample(n_chains, rng)          # start at stationarity: pure mixing test
    step, _ = samplers.make_step(method, target, eta, J=J, rng=rng, n=n_chains)
    trace = np.empty((n_steps, n_chains))
    rng_run = np.random.default_rng(seed + 7717)
    for k in range(n_steps):
        x = step(x, rng_run)
        if not np.all(np.isfinite(x)):
            return None
        trace[k] = x @ v_slow
    return trace[burn:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--nu", type=float, default=5.0)
    ap.add_argument("--kappa", type=float, default=100.0)
    ap.add_argument("--chains", type=int, default=64)
    ap.add_argument("--steps", type=int, default=120_000)
    ap.add_argument("--burn", type=int, default=20_000)
    ap.add_argument("--bias", type=float, default=0.02)
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.7])
    args = ap.parse_args()

    target = anisotropic_student_t(args.d, args.nu, args.kappa)
    v_slow = target.Sigma_evecs[:, int(np.argmax(target.Sigma_evals))]
    J_opt = skew.lnp_optimal(target.Sigma_inv)
    norm_opt = float(np.linalg.norm(J_opt, 2))
    print(target, f"\n  slow direction eigenvalue {target.Sigma_evals.max():.4g}, "
                  f"|J_opt| = {norm_opt:.3f}")

    results = {"config": vars(args), "J_optimal_norm": norm_opt, "rows": []}
    for frac in args.fracs:
        J = None if frac == 0.0 else J_opt * frac
        eta = analysis.eta_for_bias(target, J, args.bias)
        trace = run_chains(target, "skew_anchored", eta, J, args.chains,
                           args.steps, args.burn, seed=101, v_slow=v_slow)
        if trace is None:
            print(f"  |J|={frac * norm_opt:6.2f}  DIVERGED")
            continue
        taus = [metrics.iact_geyer(trace[:, c]) for c in range(trace.shape[1])]
        taus = np.array([t for t in taus if np.isfinite(t)])
        mean_tau, lo, hi = runner.bootstrap_ci(taus)
        row = {"frac": frac, "J_norm": frac * norm_opt, "eta": eta,
               "iact_mean": mean_tau, "iact_lo": lo, "iact_hi": hi,
               "ess_per_iter": 1.0 / mean_tau,
               "predicted_rate": analysis.ms_rate(target, J, eta),
               "gap": analysis.ou_gap(target, J),
               "trace_var": float(trace.var())}
        results["rows"].append(row)
        print(f"  |J|={row['J_norm']:6.2f} eta={eta:.2e} IACT={mean_tau:9.1f} "
              f"[{lo:.1f}, {hi:.1f}]  ESS/iter={row['ess_per_iter']:.3e}")

    if results["rows"]:
        base = results["rows"][0]["iact_mean"]
        for r in results["rows"]:
            r["iact_speedup"] = base / r["iact_mean"]
        best = min(results["rows"], key=lambda r: r["iact_mean"])
        print(f"\n  best: |J|={best['J_norm']:.2f} reduces the autocorrelation time "
              f"{best['iact_speedup']:.2f}x versus J=0")

    path = runner.save_json(results, os.path.join(
        OUT, f"exp3_single_chain_d{args.d}_k{int(args.kappa)}.json"))
    print("wrote", path)


if __name__ == "__main__":
    main()
