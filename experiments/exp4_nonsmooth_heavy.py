#!/usr/bin/env python3
"""Heavy tailed *and* non-differentiable, with skew acceleration.

The paper treats non-smooth potentials (Sections 6.1-6.3) and heavy tails
(Section 6.4) in separate experiments.  This one combines them:

    U(x)  = iota log q(x) + sum_i MCP_lambda(x_i)
    U0(x) = beta log q(x) + sum_i MCP^eps_lambda(x_i)

with ``q`` the anisotropic Student-t quadratic form and MCP the minimax concave
penalty from the paper's Section 6.2, smoothed exactly as in its Eq. (67).  The
penalty is bounded, so the polynomial tail survives, and the target admits an
exact rejection sampler -- which is what makes the Wasserstein reference honest.

Because the projected quantiles are not available in closed form here, the
metric is the two-sample sliced ``W_2`` against exact draws, and the floor is
measured between two independent exact samples.
"""

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skewanchor import analysis, metrics, nonsmooth, runner, samplers, skew  # noqa: E402
from skewanchor.targets import anisotropic_student_t  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--nu", type=float, default=5.0)
    ap.add_argument("--kappa", type=float, default=100.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--a", type=float, default=2.0)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--bias", type=float, default=0.02)
    ap.add_argument("--prior", default="normal10")
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.7])
    ap.add_argument("--ramp", type=float, default=0.0,
                    help="warm up the skew field over this many stiff relaxation times "
                         "(0 disables it). Removes the transient hump.")
    args = ap.parse_args()

    target = nonsmooth.make(args.d, args.nu, args.kappa, lam=args.lam, a=args.a, eps=args.eps)
    base = anisotropic_student_t(args.d, args.nu, args.kappa)
    rng = np.random.default_rng(0)
    print(target)
    print(f"  rejection acceptance rate: {target.acceptance_rate(rng, 200_000):.3f}")
    target.cov(rng, 1_000_000)

    directions = [np.asarray(v) for v in np.eye(args.d)]
    ref = [target.sample(args.n, rng) for _ in range(12)]
    floors = [metrics.sliced_w2_two_sample(ref[i], ref[i + 1], directions)
              for i in range(0, len(ref) - 1, 2)]
    floor = float(np.mean(floors))
    print(f"  two-sample sliced-W2 floor at n={args.n}: {floor:.4f} "
          f"(+- {np.std(floors):.4f})")

    J_opt = skew.lnp_optimal(target.Sigma_inv)
    norm_opt = float(np.linalg.norm(J_opt, 2))
    record_at = runner.log_schedule(args.steps, 40)
    reference = target.sample(args.n, rng)

    def recorder(x, k):
        return {"w2": metrics.sliced_w2_two_sample(x, reference, directions),
                "slow": metrics.slow_direction_error(x, target)}

    results = {"config": vars(args), "floor": floor, "J_optimal_norm": norm_opt, "rows": []}
    if args.ramp > 0:
        print(f"  warm-up ramp: {args.ramp:g} stiff relaxation times")

    print("\n--- equal-bias stepsize taken from the Student-t core ---")
    for frac in args.fracs:
        J = None if frac == 0.0 else J_opt * frac
        eta = analysis.eta_for_bias(base, J, args.bias)
        curves, n_div = [], 0
        t0 = time.time()
        for r in range(args.reps):
            rng_i = np.random.default_rng(500 + r)
            x0 = runner.make_prior(args.prior, args.d, args.n, rng_i)
            sch = (samplers.warmup_schedule(base, eta, args.ramp)
                   if args.ramp > 0 and J is not None else None)
            res = samplers.run("skew_anchored", target, x0, eta, args.steps,
                               np.random.default_rng(9000 + r), J=J,
                               record_at=record_at, recorder=recorder, schedule=sch)
            if res.diverged_at is not None:
                n_div += 1
                continue
            curves.append((res.iters, res.column("w2"), res.column("slow")))
        if not curves:
            print(f"  |J|={frac * norm_opt:6.2f}  all {args.reps} replications diverged")
            continue
        iters = np.asarray(curves[0][0])
        w2 = np.mean([c[1] for c in curves], axis=0)
        slow = np.mean([c[2] for c in curves], axis=0)
        hit = int(iters[np.argmax(w2 <= 2 * floor)]) if (w2 <= 2 * floor).any() else -1
        hit_slow = int(iters[np.argmax(slow <= 0.10)]) if (slow <= 0.10).any() else -1
        results["rows"].append({"frac": frac, "J_norm": frac * norm_opt, "eta": eta,
                                "iters": iters.tolist(), "w2": w2.tolist(),
                                "slow": slow.tolist(), "n_diverged": n_div,
                                "iters_to_2xfloor": hit, "iters_to_slow10": hit_slow,
                                "final_w2": float(w2[-1]),
                                "hump": metrics.transient_rise(w2, floor), "ramp": args.ramp})
        print(f"  |J|={frac * norm_opt:6.2f} eta={eta:.2e} hump={metrics.transient_rise(w2, floor):5.2f}x "
              f"iters_to_2xfloor={hit:6d} iters_to_slow<10%={hit_slow:6d} "
              f"final_W2={w2[-1]:.4f} diverged={n_div}/{args.reps} ({time.time() - t0:.0f}s)")

    # ULA on a non-differentiable potential: a subgradient step, for reference
    print("\n--- subgradient ULA baseline (grad U undefined at x_i = 0) ---")
    eta = analysis.eta_for_bias(base, None, args.bias)
    for mult in (1.0, 4.0, 16.0):
        curves, n_div = [], 0
        for r in range(max(3, args.reps // 2)):
            rng_i = np.random.default_rng(700 + r)
            x0 = runner.make_prior(args.prior, args.d, args.n, rng_i)
            res = samplers.run("ula", target, x0, eta * mult, args.steps,
                               np.random.default_rng(9500 + r), record_at=record_at,
                               recorder=recorder)
            if res.diverged_at is not None:
                n_div += 1
                continue
            curves.append((res.iters, res.column("w2"), res.column("slow")))
        if not curves:
            print(f"  ULA eta={eta * mult:.2e}: all replications diverged")
            continue
        iters = np.asarray(curves[0][0])
        w2 = np.mean([c[1] for c in curves], axis=0)
        hit = int(iters[np.argmax(w2 <= 2 * floor)]) if (w2 <= 2 * floor).any() else -1
        results.setdefault("ula", []).append(
            {"eta": eta * mult, "mult": mult, "iters": iters.tolist(), "w2": w2.tolist(),
             "iters_to_2xfloor": hit, "final_w2": float(w2[-1]), "n_diverged": n_div})
        print(f"  ULA eta={eta * mult:.2e} iters_to_2xfloor={hit:6d} final_W2={w2[-1]:.4f}")

    if results["rows"]:
        base_row = results["rows"][0]
        for r in results["rows"]:
            if base_row["iters_to_slow10"] > 0 and r["iters_to_slow10"] > 0:
                r["speedup_slow"] = base_row["iters_to_slow10"] / r["iters_to_slow10"]
        best = max((r for r in results["rows"] if "speedup_slow" in r),
                   key=lambda r: r["speedup_slow"], default=None)
        if best:
            print(f"\n  best: |J|={best['J_norm']:.2f} reaches 10% slow-direction error "
                  f"{best['speedup_slow']:.2f}x sooner than J=0")

    print("wrote", runner.save_json(results, os.path.join(
        OUT, f"exp4_nonsmooth_d{args.d}_k{int(args.kappa)}"
             f"{'_ramp' if args.ramp > 0 else ''}.json")))


if __name__ == "__main__":
    main()
