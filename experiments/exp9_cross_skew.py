#!/usr/bin/env python3
"""The cross-product state-dependent field, in three dimensions.

    J_s(x) v = s (x x v),   J_s(x) = [[0, -s x3, s x2], [s x3, 0, -s x1], [-s x2, s x1, 0]]

``div J_s = 0`` identically, so it goes straight into the anchored drift with no
correction term.  With the canonical anchor the added drift reduces exactly to

    c(x) = s (2 beta / nu) (x x Sigma^{-1} x),

which is tangent to spheres and to the anchor's level sets, and vanishes when
Sigma is isotropic.

Compared against the reversible anchored dynamics and against the best constant
field, all at the stepsize that equalises accuracy and all with the warm-up
ramp.
"""

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skewanchor import analysis, metrics, runner, samplers, skew, skewfield as sf  # noqa: E402
from skewanchor.targets import anisotropic_student_t  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "data")
PCTS = np.array([0.5, 0.7, 0.9])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=6.0)
    ap.add_argument("--kappa", type=float, default=100.0)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--bias", type=float, default=0.02)
    ap.add_argument("--ramp", type=float, default=3.0)
    ap.add_argument("--svals", type=float, nargs="*",
                    default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--deltas", type=float, nargs="*", default=[3.0, 6.0, 12.0])
    args = ap.parse_args()

    T = anisotropic_student_t(3, args.nu, args.kappa)
    print(T)
    rng = np.random.default_rng(0)
    V = T.Sigma_evecs
    sc = np.sqrt([V[:, i] @ T.Sigma @ V[:, i] for i in range(3)])
    qtrue = np.array([[c * stats.t.ppf((1 + p) / 2, T.nu) for p in PCTS] for c in sc])
    floor, fstd = metrics.w2_reference_floor(T, args.n, rng, n_rep=20)
    print(f"  sliced-W2 floor at n={args.n}: {floor:.4f} +- {fstd:.4f}")

    Jopt = skew.lnp_optimal(T.Sigma_inv)
    nopt = float(np.linalg.norm(Jopt, 2))
    print(f"  optimal constant field: |J| = {nopt:.2f}\n")

    def make(kind, mag, eta):
        if kind == "zero":
            fld = None
        elif kind == "const":
            fld = sf.ConstantSkew(mag * Jopt / nopt)
        else:
            fld = sf.CrossProductSkew(mag)
        ramp = samplers.warmup_schedule(T, eta, args.ramp) if (fld is not None) else None
        return lambda: samplers.field_anchored_step(T, eta, fld, "euler", schedule=ramp)

    def qbias(mk, n=12000, burn=2500, avg=1500, seed=11):
        x = T.sample(n, np.random.default_rng(seed))
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

    eta0 = analysis.eta_for_bias(T, None, args.bias, integrator="euler")
    level = qbias(make("zero", 0.0, eta0))
    record_at = runner.log_schedule(args.steps, 55)
    print(f"  calibration: J=0 at eta={eta0:.3e} -> quantile bias {level:.4f}\n")

    results = {"config": vars(args), "floor": floor, "level": level,
               "J_optimal_norm": nopt, "rows": []}

    def evaluate(label, kind, mag, probe):
        t0 = time.time()
        b = qbias(make(kind, mag, probe))
        if not np.isfinite(b) or b <= 0:
            print(f"  {label:34s} diverged at the probe stepsize")
            return
        eta = probe * (level / b)
        curves = []
        for r in range(args.reps):
            x = runner.make_prior("normal10", 3, args.n, np.random.default_rng(500 + r))
            st = make(kind, mag, eta)()
            rr = np.random.default_rng(9000 + r)
            w, k = [], 0
            for tk in record_at:
                while k < tk:
                    x = st(x, rr)
                    k += 1
                    if not np.all(np.isfinite(x)):
                        break
                w.append(metrics.axis_sliced_w2(x, T))
            curves.append(w)
        w = np.mean(curves, axis=0)
        it = np.asarray(record_at)
        hit = int(it[np.argmax(w <= 2 * floor)]) if (w <= 2 * floor).any() else -1
        row = {"label": label, "kind": kind, "mag": mag, "eta": eta,
               "iters": it.tolist(), "w2": w.tolist(), "iters_to_2xfloor": hit,
               "hump": metrics.transient_rise(w, floor), "final_w2": float(w[-1])}
        results["rows"].append(row)
        print(f"  {label:34s} eta={eta:.2e} hump={row['hump']:5.2f}x "
              f"iters={hit:6d} final_W2={w[-1]:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    evaluate("J = 0 (reversible anchored)", "zero", 0.0, eta0)
    print()
    for dl in args.deltas:
        evaluate(f"J constant, |J|={dl:g}", "const", dl, 4e-4)
    print()
    for sv in args.svals:
        evaluate(f"J_s(x) = s (x cross .), s={sv:g}", "cross", sv, 4e-4)

    base = next((r["iters_to_2xfloor"] for r in results["rows"] if r["kind"] == "zero"), -1)
    if base > 0:
        for r in results["rows"]:
            r["speedup"] = base / r["iters_to_2xfloor"] if r["iters_to_2xfloor"] > 0 else float("nan")
        print("\n  speed-ups against the reversible anchored dynamics")
        for kind in ("const", "cross"):
            best = max((r for r in results["rows"] if r["kind"] == kind and r["iters_to_2xfloor"] > 0),
                       key=lambda r: r["speedup"], default=None)
            if best:
                print(f"    best {kind:6s}: {best['label']:34s} {best['speedup']:6.2f}x")
    print("\nwrote", runner.save_json(results, os.path.join(OUT, "exp9_cross_skew.json")))


if __name__ == "__main__":
    main()
