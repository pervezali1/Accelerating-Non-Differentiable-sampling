#!/usr/bin/env python3
"""Two independent improvements to J_s, measured.

J_s is the curl family with f = s|x|^2/2.  Both defects are in the choice of f,
not in the cross-product form, so both can be fixed without giving up the one
property it was chosen for -- div J = 0, hence no correction term.

  (1) TRANSPORT.  c = (2b/nu)(grad f x Sigma^-1 x) conserves f as well as the
      anchor.  For f = s|x|^2/2 that second invariant is |x|, and the transport
      that accelerates has to change |x| by 1/sqrt(kappa).  Generalising to
      f = (1/2) x' M x for symmetric M keeps div J = 0 and conserves x'Mx
      instead.  Taking M = Sigma^-1 with the MIDDLE eigenvalue changed puts the
      whole field in the stiff-soft plane (measured share 1.000 against 0.507).

  (2) STEPSIZE.  grad f = s M x is linear, so c is quadratic while the
      reversible drift is linear: |c|/|b| grows without bound and the stepsize
      is set by the worst particle in the tail (99.99th/median = 11.7 for the
      user's field, 1.4 for a constant J).  Saturating the potential,
      f = s sqrt(r^2 + x'Mx), bounds grad f and makes c linear like b.
"""
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
from experiments.exp10_curl_potentials import GradCurl

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "data")
T = anisotropic_student_t(3, 6.0, 100.0)
V, ev = T.Sigma_evecs, T.Sigma_evals
order = np.argsort(ev); Bs = V[:, order]; aa = 1.0 / ev[order]
M_MID = Bs @ np.diag([aa[0], 0.0, aa[2]]) @ Bs.T
M_MID /= np.linalg.norm(M_MID, 2)
Jopt = skew.lnp_optimal(T.Sigma_inv); nopt = float(np.linalg.norm(Jopt, 2))
PCTS = (0.5, 0.7, 0.9)
scl = np.sqrt([V[:, i] @ T.Sigma @ V[:, i] for i in range(3)])
qtrue = np.array([[c * stats.t.ppf((1 + p) / 2, T.nu) for p in PCTS] for c in scl])
N, REPS, STEPS = 5000, 5, 20000
REC = runner.log_schedule(STEPS, 90)
floor, _ = metrics.w2_reference_floor(T, N, np.random.default_rng(0), n_rep=20)

def grad_quad(s, M):
    return lambda x: s * (x @ M.T)

def grad_sat(s, M, r):
    """grad of f = s sqrt(r^2 + x'Mx).  Requires M positive semi-definite:
    with an indefinite M the radicand goes negative and this returns NaN, which
    the stepsize search then misreports as 'unstable at every stepsize'."""
    lo = np.linalg.eigvalsh(M).min()
    if lo < -1e-12:
        raise ValueError(f"M must be positive semi-definite; min eigenvalue {lo:.3g}")
    def g(x):
        q = np.sqrt(r * r + np.einsum('ni,ij,nj->n', x, M, x))
        return s * (x @ M.T) / q[:, None]
    return g

def grad_mixed(v, s, M, r):
    """f = <v, x> + s sqrt(r^2 + x'Mx).

    Every 3x3 skew matrix is [w]_x for some w, so a CONSTANT J is itself the
    curl family with a linear potential.  The linear and the quadratic pieces
    are therefore two terms of one potential, and there is no reason to use
    only one of them.
    """
    gq = grad_sat(s, M, r) if s else (lambda x: np.zeros_like(x))
    return lambda x: v[None, :] + gq(x)

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
print(f"floor {floor:.4f}   J=0 at eta={eta0:.3e} -> quantile bias {LEVEL:.4f}\n")
print(f"{'field':>40} {'eta':>10} {'off':>6} {'div':>5} {'rise':>6} {'iters':>7} {'sp':>7}")
rows, base = [], {}

def run(key, label, field, ramp):
    t0 = time.time(); probe = 4e-4
    for _ in range(18):
        b = qbias(mk(field, probe, ramp))
        if np.isfinite(b) and b > 0: break
        probe *= 0.35
    else:
        print(f"{label:>40}  unstable at every stepsize"); return
    eta = probe * (LEVEL / b)
    for _ in range(8):
        if np.isfinite(qbias(mk(field, eta, ramp))): break
        eta *= 0.4
    eta_eq = eta
    def ens(eta):
        curves, div = [], 0
        for r in range(REPS):
            x = runner.make_prior("normal10", 3, N, np.random.default_rng(500 + r))
            st = mk(field, eta, ramp)(); rr = np.random.default_rng(9000 + r)
            w, k, blew = [], 0, False
            for tk in REC:
                while k < tk:
                    x = st(x, rr); k += 1
                    if not np.all(np.isfinite(x)): blew = True; break
                w.append(metrics.axis_sliced_w2(x, T))
                if blew: break
            div += int(blew); curves.append(w + [np.nan]*(len(REC)-len(w)))
        return np.nanmean(curves, axis=0), div
    w, div = ens(eta)
    for _ in range(9):
        if div == 0 and np.all(np.isfinite(w)): break
        eta *= 0.4; w, div = ens(eta)
    it = np.asarray(REC); ok = np.isfinite(w) & (w <= 2*floor)
    hit = int(it[np.argmax(ok)]) if ok.any() and div == 0 else -1
    rise = metrics.transient_rise(w, floor) if div == 0 and hit > 0 else float('nan')
    if key == "zero": base["it"] = hit
    sp = base["it"]/hit if hit > 0 else float('nan')
    print(f"{label:>40} {eta:10.2e} {eta_eq/eta:5.1f}x {div:>2}/{REPS} {rise:5.2f}x "
          f"{hit:7d} {sp:6.2f}x ({time.time()-t0:.0f}s)", flush=True)
    rows.append(dict(key=key, label=label, eta=eta, backoff=eta_eq/eta, ramp=ramp,
                     diverged=div, rise=rise, iters=hit, speedup=sp,
                     w2=list(map(float, w)), rec=list(REC)))

Jc = 4.0 * Jopt / nopt
VOPT = np.array([Jc[2, 1], Jc[0, 2], Jc[1, 0]])       # J_a = [VOPT]_x, |VOPT| = 4

def off_M(mu):
    C = np.diag([aa[0], 0.0, aa[2]]).astype(float)
    C[0, 2] = C[2, 0] = mu * aa[0]
    Mo = Bs @ C @ Bs.T
    return Mo / np.linalg.norm(Mo, 2)

run("zero",  "J = 0", None, 0.0)
run("const", "J_a,  a = 4   (f = <v,x>)", sf.ConstantSkew(Jc), 10.0)
run("mid30", "M-middle saturating, s = 30", GradCurl(grad_sat(30.0, M_MID, 0.03)), 10.0)
for s in (8.0, 20.0, 40.0):
    run(f"mix{s:g}", f"f = <v,x> + {s:g} sqrt(r^2+x'Mx)   (both)",
        GradCurl(grad_mixed(VOPT, s, M_MID, 0.03)), 10.0)
run("off08", "M-middle + offdiag 0.08, s = 30",
    GradCurl(grad_sat(30.0, off_M(0.08), 0.03)), 10.0)
json.dump(dict(floor=float(floor), rows=rows, M_mid=M_MID.tolist()),
          open(os.path.join(OUT, "exp13_raw.json"), "w"), indent=1)
print("\nwrote", os.path.join(OUT, "exp13_raw.json"))
