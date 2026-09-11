#!/usr/bin/env python3
"""State-dependent skew fields and rotation-aware integrators.

Answers two questions on the anisotropic heavy-tailed target.

1. Does letting ``J`` depend on ``x`` help?  Under the naive admissibility
   condition the only two-dimensional freedom is a radial profile ``psi(q)``,
   which is blind to direction -- and it *hurts*.  Adding the correction term
   ``-e^{U-U0} div J`` removes the condition, and the resulting stream-function
   family can tilt the rotation towards the soft axis, which helps a great deal.
2. Does the integrator matter?  The binding constraint on the whole method is
   discretisation bias, not stability, and explicit Euler's error on a rotation
   is what produces it.  Advancing the linear part by its Cayley transform or
   matrix exponential removes that term.

Fairness.  Every method is run at the stepsize that puts it at the same
accuracy.  Two criteria are reported because they disagree, and the
disagreement is itself a finding: the skew perturbation's bias sits mostly in
the tail.

``covariance``  relative error of the stationary covariance.  Exact in closed
                form for a constant field (any integrator); the sample estimate
                is very noisy at ``nu = 5`` because the sample covariance needs
                fourth moments.
``quantile``    relative error of the 50th, 70th and 90th percentiles of
                ``|<x, v>|`` along the principal axes.  Finite variance for any
                ``nu``, so it is the one used for the simulated comparisons,
                calibrated to agree with the covariance criterion at ``J = 0``.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skewanchor import analysis, runner, samplers, skew, skewfield as sf  # noqa: E402
from skewanchor.targets import anisotropic_student_t  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "data")
PCTS = np.array([0.5, 0.7, 0.9])


class Protocol:
    """Equal-accuracy stepsize selection and convergence measurement."""

    def __init__(self, target, n_bias=20000, burn=3000, avg=2000, n_rate=5000,
                 steps=12000, reps=3, seed=11):
        self.t = target
        self.n_bias, self.burn, self.avg = n_bias, burn, avg
        self.n_rate, self.steps, self.reps, self.seed = n_rate, steps, reps, seed
        self.V = target.Sigma_evecs
        sc = np.sqrt([self.V[:, i] @ target.Sigma @ self.V[:, i] for i in range(target.d)])
        self.qtrue = np.array([[s * stats.t.ppf((1 + p) / 2, target.nu) for p in PCTS]
                               for s in sc])
        self.v_slow = self.V[:, int(np.argmax(target.Sigma_evals))]
        self.truth_slow = float(self.v_slow @ target.cov() @ self.v_slow)
        self.record = sorted(set(np.unique(
            np.round(np.geomspace(1, steps, 70)).astype(int)).tolist()))

    def quantile_bias(self, make_step, eta):
        rng = np.random.default_rng(self.seed)
        x = self.t.sample(self.n_bias, rng)
        step = make_step(eta)
        rr = np.random.default_rng(self.seed + 1)
        for _ in range(self.burn):
            x = step(x, rr)
            if not np.all(np.isfinite(x)):
                return float("inf")
        acc = np.zeros_like(self.qtrue)
        for _ in range(self.avg):
            x = step(x, rr)
            if not np.all(np.isfinite(x)):
                return float("inf")
            acc += np.quantile(np.abs(x @ self.V), PCTS, axis=0).T
        return float(np.max(np.abs(acc / self.avg / self.qtrue - 1.0)))

    def eta_for(self, make_step, level, probe):
        """Stepsize hitting ``level``; the bias is linear in eta in this range."""
        b = self.quantile_bias(make_step, probe)
        if not np.isfinite(b) or b <= 0:
            return None
        return probe * (level / b)

    def iterations(self, make_step, eta):
        hits = []
        for r in range(self.reps):
            rng = np.random.default_rng(100 + r)
            x = rng.standard_normal((self.n_rate, self.t.d)) * np.sqrt(10.0)
            step = make_step(eta)
            rr = np.random.default_rng(900 + r)
            hit, k = -1, 0
            for tk in self.record:
                while k < tk:
                    x = step(x, rr)
                    k += 1
                    if not np.all(np.isfinite(x)):
                        return self.steps * 3
                if abs((x @ self.v_slow).var() - self.truth_slow) / self.truth_slow <= 0.10:
                    hit = tk
                    break
            hits.append(hit if hit > 0 else self.steps * 3)
        return float(np.median(hits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=100.0)
    ap.add_argument("--nu", type=float, default=5.0)
    ap.add_argument("--bias", type=float, default=0.02)
    ap.add_argument("--deltas", type=float, nargs="*", default=[4.95, 9.9, 19.8])
    ap.add_argument("--tilts", type=float, nargs="*",
                    default=[-1.6, -1.3, -1.1, -0.9, -0.6, -0.3, 0.0, 0.3])
    args = ap.parse_args()

    T = anisotropic_student_t(2, args.nu, args.kappa)
    P = Protocol(T)
    print(T)

    # calibrate the quantile criterion against the exact covariance criterion at J = 0
    eta0 = analysis.eta_for_bias(T, None, args.bias, integrator="euler")
    base_step = lambda e: samplers.field_anchored_step(T, e, None, "euler")
    level = P.quantile_bias(base_step, eta0)
    base_iters = P.iterations(base_step, eta0)
    print(f"  calibration: Euler J=0 at eta={eta0:.3e} (covariance bias {args.bias:.1%}) "
          f"-> quantile bias {level:.4f}")
    print(f"  baseline: {base_iters:.0f} iterations\n")

    res = {"config": vars(args), "eta0": eta0, "level": level,
           "base_iters": base_iters, "exact": [], "measured": []}

    # ---- exact, constant field, every integrator (no Monte Carlo at all)
    Jo = skew.lnp_optimal(T.Sigma_inv)
    n0 = float(np.linalg.norm(Jo, 2))
    e0 = analysis.eta_for_bias(T, None, args.bias, integrator="euler")
    r0 = analysis.ms_rate(T, None, e0, "euler")
    print("--- exact (constant field, covariance criterion) ---")
    for integ in ("euler", "cayley", "expm"):
        best = (0.0, 0.0, 0.0)
        for f in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
            J = None if f == 0 else Jo * f
            e = analysis.eta_for_bias(T, J, args.bias, integrator=integ)
            rate = analysis.ms_rate(T, J, e, integ) if e > 0 else 0.0
            res["exact"].append({"integrator": integ, "J_norm": f * n0, "eta": e,
                                 "rate": rate, "speedup": rate / r0})
            if rate / r0 > best[0]:
                best = (rate / r0, f * n0, e)
        print(f"  {integ:7s} best {best[0]:6.2f}x at |J|={best[1]:7.2f} (eta={best[2]:.2e})")

    # ---- measured
    def record(label, make_step, probe, **meta):
        t0 = time.time()
        eta = P.eta_for(make_step, level, probe)
        if eta is None:
            print(f"  {label:40s} diverged")
            return
        it = P.iterations(make_step, eta)
        row = dict(label=label, eta=eta, iters=it, speedup=base_iters / it, **meta)
        res["measured"].append(row)
        print(f"  {label:40s} eta={eta:9.2e} iters={it:7.0f} "
              f"speedup={base_iters / it:7.2f}x  ({time.time() - t0:.0f}s)", flush=True)

    print("\n--- measured: constant field ---")
    for integ in ("euler", "cayley"):
        for dl in args.deltas:
            J = dl * np.array([[0.0, 1.0], [-1.0, 0.0]])
            record(f"constant |J|={dl:5.2f} {integ}",
                   lambda e, J=J, i=integ: samplers.field_anchored_step(T, e, sf.ConstantSkew(J), i),
                   5e-4, kind="constant", J_norm=dl, integrator=integ)

    print("\n--- measured: radial modulation (naive state dependence) ---")
    for kind in ("decay", "grow", "bulk"):
        prof = sf.make_profile(kind, 1.0, q0=1.0, p=1.0)
        mean = float(np.mean(prof(T.q(T.sample(200000, np.random.default_rng(3))))))
        fld = sf.RadialModulated(np.array([[0.0, 1.0], [-1.0, 0.0]]), T,
                                 sf.make_profile(kind, 4.95 / mean, q0=1.0, p=1.0))
        record(f"radial {kind} mean|J|=4.95 euler",
               lambda e, f=fld: samplers.field_anchored_step(T, e, f, "euler"),
               3e-4, kind=f"radial_{kind}", integrator="euler")

    print("\n--- measured: stream field (corrected state dependence) ---")
    for integ in ("euler", "cayley"):
        for dl in args.deltas[:2]:
            for a in args.tilts:
                fld = sf.StreamField2D.quadrupole(T, dl, a, 0.0)
                record(f"stream a={a:+.2f} |J|={dl:5.2f} {integ}",
                       lambda e, f=fld, i=integ: samplers.stream_anchored_step(T, e, f, i),
                       5e-4, kind="stream", J_norm=dl, tilt=a, integrator=integ)

    if res["measured"]:
        best = max(res["measured"], key=lambda r: r["speedup"])
        print(f"\nbest measured: {best['label']} at {best['speedup']:.2f}x")
        res["best"] = best
    print("wrote", runner.save_json(res, os.path.join(
        OUT, f"exp6_state_dependent_k{int(args.kappa)}.json")))


if __name__ == "__main__":
    main()
