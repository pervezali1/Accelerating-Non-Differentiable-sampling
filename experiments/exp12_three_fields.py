#!/usr/bin/env python3
"""d = 3: J = 0, a constant J_a at a = 4, and optionally J_s at s = 4.

    J_s(x) = [[  0,    -s x3,   s x2 ],
              [  s x3,   0,    -s x1 ],
              [ -s x2,   s x1,   0   ]]

All three on the explicit Euler scheme.  The exponential integrator needs the
drift's linear part to be a fixed matrix, which J_s is not, so putting the
constant field on it would compare integrators rather than fields.

Protocol as established: stepsizes equalised on the stationary error of the
50th/70th/90th percentiles along the principal axes, calibrated so J = 0 sits
at 2 % stationary covariance bias; five replications; a run counts only if all
five survive, and the stepsize is backed off until they do.

``--prior`` selects the starting ensemble, the paper's two:

    normal10   X_0 ~ N(0, 10 I_d)
    uniform5   X_0 ~ Uniform(-5, 5)^d

The equal-accuracy stepsize is a property of the stationary law and the target,
not of the prior, so it is the same for both; only the transient differs.  The
prior is recorded in the output and carried in its filename.
"""
import argparse
import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy import stats
from skewanchor.targets import anisotropic_student_t
from skewanchor import analysis, metrics, runner, samplers, skew, skewfield as sf

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "data")

_ap = argparse.ArgumentParser()
_ap.add_argument("--prior", default="normal10", choices=["normal10", "uniform5"],
                 help="starting ensemble: N(0, 10 I) or Uniform(-5, 5)^d")
_ap.add_argument("--fields", nargs="*", default=["zero", "const"],
                 choices=["zero", "const", "cross"],
                 help="J = 0, the constant J_a, and the cross-product J_s. "
                      "J_s at s = 4 needs --steps 40000 to show anything")
_ap.add_argument("--steps", type=int, default=20000)
_ap.add_argument("--n", type=int, default=5000)
_ap.add_argument("--reps", type=int, default=5)
ARGS = _ap.parse_args()

T = anisotropic_student_t(3, 6.0, 100.0)
V = T.Sigma_evecs
Jopt = skew.lnp_optimal(T.Sigma_inv); nopt = float(np.linalg.norm(Jopt, 2))
PCTS = (0.5, 0.7, 0.9)
scl = np.sqrt([V[:, i] @ T.Sigma @ V[:, i] for i in range(3)])
qtrue = np.array([[c * stats.t.ppf((1 + p) / 2, T.nu) for p in PCTS] for c in scl])
N, REPS, STEPS = ARGS.n, ARGS.reps, ARGS.steps
REC = runner.log_schedule(STEPS, 90)
floor, fstd = metrics.w2_reference_floor(T, N, np.random.default_rng(0), n_rep=20)

def mk(field, eta, ramp):
    sch = samplers.warmup_schedule(T, eta, ramp) if field is not None else None
    return lambda: samplers.field_anchored_step(T, eta, field, "euler", schedule=sch)

def qbias(make, n=12000, burn=2500, avg=1500, seed=11):
    x = T.sample(n, np.random.default_rng(seed)); st = make()
    rr = np.random.default_rng(seed + 1)
    for _ in range(burn + avg):
        x = st(x, rr)
        if not np.all(np.isfinite(x)): return float("inf")
    acc = np.zeros_like(qtrue)
    for _ in range(avg):
        x = st(x, rr)
        if not np.all(np.isfinite(x)): return float("inf")
        acc += np.quantile(np.abs(x @ V), PCTS, axis=0).T
    return float(np.max(np.abs(acc / avg / qtrue - 1.0)))

eta0 = analysis.eta_for_bias(T, None, 0.02, integrator="euler")
LEVEL = qbias(mk(None, eta0, 0))
print(f"target {T}")
print(f"floor {floor:.4f} +- {fstd:.4f}   J=0 at eta={eta0:.3e} -> quantile bias {LEVEL:.4f}\n")
print(f"{'field':>16} {'eta':>10} {'backoff':>8} {'div':>5} {'rise':>6} {'iters':>7} {'speedup':>8}")

rows, base = [], {}

def run(key, label, field, ramp):
    t0 = time.time()
    probe = 4e-4
    for _ in range(18):
        b = qbias(mk(field, probe, ramp))
        if np.isfinite(b) and b > 0: break
        probe *= 0.35
    else:
        print(f"{label:>16}   unstable at every stepsize tried"); return
    eta = probe * (LEVEL / b)
    for _ in range(8):
        if np.isfinite(qbias(mk(field, eta, ramp))): break
        eta *= 0.4
    eta_equal = eta

    def ensemble(eta):
        curves, div = [], 0
        for r in range(REPS):
            x = runner.make_prior(ARGS.prior, 3, N, np.random.default_rng(500 + r))
            st = mk(field, eta, ramp)()
            rr = np.random.default_rng(9000 + r)
            w, k, blew = [], 0, False
            for tk in REC:
                while k < tk:
                    x = st(x, rr); k += 1
                    if not np.all(np.isfinite(x)): blew = True; break
                w.append(metrics.axis_sliced_w2(x, T))
                if blew: break
            div += int(blew); curves.append(w + [np.nan] * (len(REC) - len(w)))
        return np.nanmean(curves, axis=0), div

    w, div = ensemble(eta)
    for _ in range(9):                       # J_s at s=4 needs a lot of room
        if div == 0 and np.all(np.isfinite(w)): break
        eta *= 0.4; w, div = ensemble(eta)
    it = np.asarray(REC); ok = np.isfinite(w) & (w <= 2 * floor)
    hit = int(it[np.argmax(ok)]) if ok.any() and div == 0 else -1
    rise = metrics.transient_rise(w, floor) if div == 0 and hit > 0 else float('nan')
    if key == "zero": base["it"] = hit
    sp = base["it"] / hit if hit > 0 else float('nan')
    print(f"{label:>16} {eta:10.2e} {eta_equal/eta:7.1f}x {div:>2}/{REPS} {rise:5.2f}x "
          f"{hit:7d} {sp:7.2f}x  ({time.time()-t0:.0f}s)", flush=True)
    rows.append(dict(key=key, label=label, eta=eta, eta_equal=eta_equal,
                     backoff=eta_equal/eta, ramp=ramp, diverged=div, rise=rise,
                     iters=hit, speedup=sp, w2=list(map(float, w)), rec=list(REC)))

PRIOR_LABEL = {"normal10": "X_0 ~ N(0, 10 I)",
               "uniform5": "X_0 ~ Uniform(-5, 5)^d"}
print(f"prior: {ARGS.prior}  ({PRIOR_LABEL[ARGS.prior]})\n")

SPEC = {"zero":  ("J = 0",       None,                                0.0),
        "const": ("J_a,  a = 4", sf.ConstantSkew(4.0 * Jopt / nopt),  10.0),
        "cross": ("J_s,  s = 4", sf.CrossProductSkew(4.0),            40.0)}
for key in ARGS.fields:
    run(key, *SPEC[key])

path = os.path.join(OUT, f"exp12_three_fields_{ARGS.prior}.json")
json.dump(dict(floor=float(floor), floor_std=float(fstd), d=3, nu=6.0, kappa=100.0,
               prior=ARGS.prior, prior_label=PRIOR_LABEL[ARGS.prior],
               n=N, reps=REPS, steps=STEPS, rows=rows),
          open(path, "w"), indent=1)
print("\nwrote", path)
