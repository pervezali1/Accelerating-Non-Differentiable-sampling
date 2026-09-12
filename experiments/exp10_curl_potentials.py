#!/usr/bin/env python3
r"""The divergence-free family in three dimensions, by choice of potential.

Produces ``results/data/exp10_curl_potentials.json`` and, through
``make_figures.py``, ``fig11_curl_potentials_*.png``.

The family
----------
In ``d >= 3`` the field

    J(x) v = grad f(x) x v

has ``div J = 0`` for **every** scalar ``f``.  It therefore substitutes straight
into the anchored drift with no correction term and preserves any target with
any anchor.  Three members are compared here:

    f linear                    ->  a CONSTANT J
    f = s |x|^2 / 2             ->  J_s(x) v = s (x x v), the cross-product field
    f = s x_stiff x_soft        ->  a hyperbolic field

Written out, the cross-product member is

    J_s(x) = [[  0,    -s x3,   s x2 ],
              [  s x3,   0,    -s x1 ],
              [ -s x2,   s x1,   0   ]]

Why the choice of ``f`` matters
-------------------------------
Every field of the form ``c = e^{U-U0} J grad U0`` conserves ``U0``, because
``<grad U0, J grad U0> = 0`` for skew ``J``.  The curl family conserves a
*second* function, ``f`` itself, since ``grad f x grad U0`` is orthogonal to
both.  Two conserved functions in three dimensions pin the flow to a curve.

For ``f = |x|^2/2`` that second function is ``|x|``.  That is fatal: the
transport which actually accelerates carries mass from the soft axis to the
stiff axis *at fixed q*, where the reversible drift is ``kappa`` times faster,
and those two points differ in ``|x|`` by a factor ``1/sqrt(kappa)``.  A field
conserving ``|x|`` cannot connect them.  ``f = s x_stiff x_soft`` vanishes on
both axes, so its level sets do contain both, and it recovers part of the gain.

Protocol
--------
Every field runs at the stepsize that puts it at the same accuracy, found by
matching the stationary error of the 50th, 70th and 90th percentiles of
``|<x, v>|`` along the principal axes, calibrated so that ``J = 0`` sits at the
2 % stationary covariance bias of the exact analysis.  The stepsize search is
adaptive, halving the probe until the measurement is finite, so a field is never
reported as divergent merely because the first stepsize tried was too large.
All fields carry the warm-up ramp.
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


class GradCurl(sf.SkewField):
    """``J(x) v = grad f(x) x v`` -- divergence free for every ``f``."""

    is_constant = False

    def __init__(self, grad_f):
        self.grad_f = grad_f
        self.d = 3

    def apply(self, x, v):
        return np.cross(self.grad_f(np.atleast_2d(x)), np.atleast_2d(v))

    def divergence(self, x, h=None):
        return np.zeros_like(np.atleast_2d(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=6.0)
    ap.add_argument("--kappa", type=float, default=100.0)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--bias", type=float, default=0.02)
    ap.add_argument("--ramp", type=float, default=10.0,
                    help="warm-up length in relaxations of the fastest direction; "
                         "10 leaves no hump in d=3 and costs no iterations")
    ap.add_argument("--jnorm", type=float, default=6.0)
    ap.add_argument("--sphere-s", type=float, nargs="*", default=[0.1, 0.3, 0.5, 1.0])
    ap.add_argument("--hyper-s", type=float, nargs="*",
                    default=[0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--smoke", action="store_true", help="tiny run, to check it works")
    args = ap.parse_args()

    if args.smoke:
        args.n, args.steps, args.reps = 800, 2000, 1
        args.sphere_s, args.hyper_s = [0.1], [0.3]

    T = anisotropic_student_t(3, args.nu, args.kappa)
    V, ev = T.Sigma_evecs, T.Sigma_evals
    e_soft, e_stiff = V[:, int(np.argmax(ev))], V[:, int(np.argmin(ev))]
    sc = np.sqrt([V[:, i] @ T.Sigma @ V[:, i] for i in range(3)])
    qtrue = np.array([[c * stats.t.ppf((1 + p) / 2, T.nu) for p in PCTS] for c in sc])

    rng = np.random.default_rng(0)
    floor, fstd = metrics.w2_reference_floor(T, args.n, rng, n_rep=20)
    Jopt = skew.lnp_optimal(T.Sigma_inv)
    nopt = float(np.linalg.norm(Jopt, 2))
    record_at = runner.log_schedule(args.steps, 60)
    print(T)
    print(f"  sliced-W2 floor at n={args.n}: {floor:.4f} +- {fstd:.4f}")
    print(f"  optimal constant field |J| = {nopt:.2f}\n")

    def field(kind, mag):
        if kind == "zero":
            return None
        if kind == "linear":                       # a constant J
            return sf.ConstantSkew(mag * Jopt / nopt)
        if kind == "sphere":                       # f = s |x|^2 / 2
            return GradCurl(lambda x, s=mag: s * x)
        if kind == "hyper":                        # f = s x_stiff x_soft
            return GradCurl(lambda x, s=mag: s * (np.outer(x @ e_stiff, e_soft)
                                                  + np.outer(x @ e_soft, e_stiff)))
        raise ValueError(kind)

    def make(fld, eta):
        sch = samplers.warmup_schedule(T, eta, args.ramp) if fld is not None else None
        return lambda: samplers.field_anchored_step(T, eta, fld, "euler", schedule=sch)

    def qbias(mk, n=12000, burn=2500, avg=1500, seed=11):
        if args.smoke:
            n, burn, avg = 3000, 400, 300
        x = T.sample(n, np.random.default_rng(seed))
        st = mk()
        rr = np.random.default_rng(seed + 1)
        for _ in range(burn + avg):
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
    level = qbias(make(None, eta0))
    print(f"  calibration: J=0 at eta={eta0:.3e} -> quantile bias {level:.4f}\n")

    rows = []

    def evaluate(label, kind, mag):
        t0 = time.time()
        probe = 4e-4
        for _ in range(14):                        # adaptive: never blame the stepsize
            b = qbias(make(field(kind, mag), probe))
            if np.isfinite(b) and b > 0:
                break
            probe *= 0.35
        else:
            print(f"  {label:40s} unstable at every stepsize tried")
            return
        eta = probe * (level / b)
        for _ in range(6):
            if np.isfinite(qbias(make(field(kind, mag), eta))):
                break
            eta *= 0.4
        curves = []
        for r in range(args.reps):
            x = runner.make_prior("normal10", 3, args.n, np.random.default_rng(500 + r))
            st = make(field(kind, mag), eta)()
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
        w = np.nanmean(curves, axis=0)
        it = np.asarray(record_at)
        hit = int(it[np.argmax(w <= 2 * floor)]) if (w <= 2 * floor).any() else -1
        rows.append({"label": label, "kind": kind, "mag": mag, "eta": eta,
                     "iters": hit, "final": float(w[-1]),
                     "w2": np.asarray(w).tolist(), "rec": record_at})
        print(f"  {label:40s} eta={eta:.2e} iters={hit:6d} final_W2={w[-1]:.4f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

    evaluate("J = 0", "zero", 0.0)
    evaluate(f"f linear  (= constant J), |J|={args.jnorm:g}", "linear", args.jnorm)
    for s in args.sphere_s:
        evaluate(f"f = |x|^2/2  (yours), s={s:g}", "sphere", s)
    for s in args.hyper_s:
        evaluate(f"f = s x_stiff x_soft, s={s:g}", "hyper", s)

    base = next((r["iters"] for r in rows if r["kind"] == "zero"), -1)
    if base > 0:
        print("\n  speed-ups vs J = 0")
        for r in rows:
            if r["iters"] > 0:
                print(f"    {r['label']:40s} {base / r['iters']:6.2f}x")
    path = os.path.join(OUT, "exp10_curl_potentials.json" if not args.smoke
                        else "exp10_smoke.json")
    # the floor and the ramp are needed to read the curves, so they travel with
    # them rather than being hardcoded in the figure
    out = {"meta": {"d": 3, "nu": args.nu, "kappa": args.kappa, "bias": args.bias,
                    "ramp": args.ramp, "jnorm": args.jnorm, "floor": float(floor),
                    "floor_std": float(fstd), "n": args.n, "steps": args.steps,
                    "reps": args.reps},
           "rows": rows}
    print("\nwrote", runner.save_json(out, path))


if __name__ == "__main__":
    main()
