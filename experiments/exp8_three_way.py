#!/usr/bin/env python3
"""Three fields, one figure: J = 0, J constant, J state dependent.

Same target and protocol as ``exp4`` -- heavy tailed *and* non-differentiable,
a Student-t core plus the paper's MCP penalty -- reduced to the three cases that
matter, each at the stepsize that puts it at the same accuracy, and each with
the warm-up ramp that removes the transient overshoot.

The three fields are members of one family,

    c = delta e^{U-U0} J_0 [ (1 + a w1) grad U0 - a grad w1 ],

with ``delta = 0`` (reversible), ``a = 0`` (constant), and ``a < 0`` (tilted
towards the soft axis), so the comparison is like for like.
"""

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skewanchor import analysis, metrics, nonsmooth, runner, samplers, skewfield as sf  # noqa: E402
from skewanchor.targets import anisotropic_student_t  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "data")
PCTS = np.array([0.5, 0.7, 0.9])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="composite", choices=["composite", "student_t"])
    ap.add_argument("--kappa", type=float, default=100.0)
    ap.add_argument("--nu", type=float, default=5.0)
    ap.add_argument("--delta", type=float, default=4.95)
    ap.add_argument("--tilt", type=float, default=-3.0)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--bias", type=float, default=0.02)
    ap.add_argument("--ramp", type=float, default=5.0)
    args = ap.parse_args()

    core = anisotropic_student_t(2, args.nu, args.kappa)
    if args.target == "composite":
        T = nonsmooth.make(2, args.nu, args.kappa, lam=1.0, a=2.0, eps=0.1)
        geometry = core
    else:
        T = core
        geometry = None
    rng = np.random.default_rng(0)
    print(T)

    # exact reference draws and the measurement floor
    ref = [T.sample(args.n, rng) for _ in range(8)]
    dirs = [np.asarray(v) for v in np.eye(2)]
    floor = float(np.mean([metrics.sliced_w2_two_sample(ref[i], ref[i + 1], dirs)
                           for i in (0, 2, 4, 6)]))
    big = T.sample(400_000, rng)
    V = core.Sigma_evecs
    qtrue = np.quantile(np.abs(big @ V), PCTS, axis=0).T
    print(f"  two-sample sliced-W2 floor at n={args.n}: {floor:.4f}")

    def field(delta, a):
        if delta == 0.0:
            return None
        return sf.StreamField2D.quadrupole(T, delta, a, 0.0, geometry=geometry)

    def make(delta, a, eta, ramp=True):
        sch = samplers.warmup_schedule(core, eta, args.ramp) if (ramp and delta) else None
        return lambda: samplers.stream_euler_step(T, eta, field(delta, a), schedule=sch)

    def quantile_bias(mk, n=15000, burn=2500, avg=1500, seed=11):
        r0 = np.random.default_rng(seed)
        x = T.sample(n, r0)
        st = mk()
        rr = np.random.default_rng(seed + 1)
        for _ in range(burn):
            x = st(x, rr)
            if not np.all(np.isfinite(x)):
                return float("inf")
        acc = np.zeros_like(qtrue)
        for _ in range(avg):
            x = st(x, rr)
            if not np.all(np.isfinite(x)):
                return float("inf")
            acc += np.quantile(np.abs(x @ V), PCTS, axis=0).T
        return float(np.max(np.abs(acc / avg / qtrue - 1.0)))

    eta0 = analysis.eta_for_bias(core, None, args.bias, integrator="euler")
    level = quantile_bias(make(0.0, 0.0, eta0))
    print(f"  calibration: J=0 at eta={eta0:.3e} -> quantile bias {level:.4f}\n")

    CASES = [("$J = 0$ (reversible anchored)", 0.0, 0.0),
             (f"$J$ constant, $\\|J\\|$={args.delta:g}", args.delta, 0.0),
             (f"$J$ state dependent, $a$={args.tilt:g}", args.delta, args.tilt)]
    record_at = runner.log_schedule(args.steps, 45)
    results = {"config": vars(args), "floor": floor, "level": level, "rows": []}

    for label, delta, a in CASES:
        t0 = time.time()
        probe = eta0 if delta == 0 else 3.0e-4
        b = quantile_bias(make(delta, a, probe))
        eta = probe * (level / b)
        curves = []
        for r in range(args.reps):
            x = runner.make_prior("normal10", 2, args.n, np.random.default_rng(500 + r))
            st = make(delta, a, eta)()
            rr = np.random.default_rng(9000 + r)
            w, k = [], 0
            for tk in record_at:
                while k < tk:
                    x = st(x, rr)
                    k += 1
                    if not np.all(np.isfinite(x)):
                        break
                w.append(metrics.sliced_w2_two_sample(x, ref[0], dirs))
            curves.append(w)
        w = np.mean(curves, axis=0)
        it = np.asarray(record_at)
        hit = int(it[np.argmax(w <= 2 * floor)]) if (w <= 2 * floor).any() else -1
        row = {"label": label, "delta": delta, "tilt": a, "eta": eta,
               "iters": it.tolist(), "w2": w.tolist(), "iters_to_2xfloor": hit,
               "hump": float(np.max(w) / w[0]), "final_w2": float(w[-1])}
        results["rows"].append(row)
        print(f"  {label:38s} eta={eta:.2e} hump={row['hump']:5.2f}x "
              f"iters_to_2xfloor={hit:6d} final_W2={w[-1]:.4f}  ({time.time() - t0:.0f}s)")

    base = results["rows"][0]["iters_to_2xfloor"]
    for row in results["rows"]:
        row["speedup"] = base / row["iters_to_2xfloor"] if row["iters_to_2xfloor"] > 0 else float("nan")
    print()
    for row in results["rows"]:
        print(f"  {row['label']:38s} {row['speedup']:6.2f}x")
    print("\nwrote", runner.save_json(results, os.path.join(
        OUT, f"exp8_three_way_{args.target}.json")))


if __name__ == "__main__":
    main()
