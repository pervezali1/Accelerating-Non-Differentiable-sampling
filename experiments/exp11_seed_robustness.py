#!/usr/bin/env python3
r"""Is a reported speed-up an artefact of one set of random seeds?

Written after one was.  ``exp10`` reported the three-dimensional curl field
``f = s x_stiff x_soft`` converging in 301 iterations at ``s = 1``, a 7.50x
speed-up that beat the best constant field.  It does no such thing: at that
stepsize the field diverges in five replications out of five, in each of three
disjoint seed sets.  Two defects combined to hide it.

* **``nanmean`` over replications.**  A curve is the mean over ``reps`` chains,
  and ``np.nanmean`` stays finite as long as *one* chain survives.  Four chains
  out of five blowing up therefore reads as convergence, at whatever iteration
  the survivor happened to reach the floor.  Divergences have to be counted, and
  a field counted as stable only when every replication is.
* **One seed set.**  With four of five diverging, whether the run reports a
  number at all is a coin flip on the seed.  The fix is to ask the same question
  of several disjoint seed sets and look at the spread, which is what this does.

Each cell here is ``reps`` chains from the wide prior at a fixed stepsize, and
reports iterations to twice the measurement floor together with how many chains
blew up.  A cell with any divergence reports no iteration count at all.

    python experiments/exp11_seed_robustness.py
    python experiments/exp11_seed_robustness.py --smoke
"""

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skewanchor import metrics, runner, samplers, skew, skewfield as sf  # noqa: E402
from skewanchor.targets import anisotropic_student_t  # noqa: E402
from experiments.exp10_curl_potentials import GradCurl  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=6.0)
    ap.add_argument("--kappa", type=float, default=100.0)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ramp", type=float, default=10.0,
                    help="warm-up for a constant field, in fast relaxations")
    ap.add_argument("--ramp-growth", type=float, default=40.0,
                    help="warm-up for a field whose strength grows with |x|; "
                         "see exp10 for why it needs a longer hold")
    ap.add_argument("--blocks", type=int, nargs="*", default=[500, 20500, 40500],
                    help="disjoint seed blocks; each gives one independent answer")
    ap.add_argument("--hyper-s", type=float, nargs="*", default=[0.5, 0.7, 1.0, 1.5, 2.0])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.steps, args.reps = 800, 1500, 2
        args.blocks, args.hyper_s = [500, 20500], [0.5]

    T = anisotropic_student_t(3, args.nu, args.kappa)
    V, ev = T.Sigma_evecs, T.Sigma_evals
    e_soft, e_stiff = V[:, int(np.argmax(ev))], V[:, int(np.argmin(ev))]
    Jopt = skew.lnp_optimal(T.Sigma_inv)
    nopt = float(np.linalg.norm(Jopt, 2))
    floor, _ = metrics.w2_reference_floor(T, args.n, np.random.default_rng(0), n_rep=20)
    record_at = runner.log_schedule(args.steps, 60)

    def hyper(s):
        return GradCurl(lambda x, s=s: s * (np.outer(x @ e_stiff, e_soft)
                                            + np.outer(x @ e_soft, e_stiff)))

    # (label, field, stepsize, warm-up length).  A field whose strength grows
    # with |x| gets the longer hold, exactly as in exp10 -- the ramp is keyed
    # to the linear relaxation time and does not know that the wide prior makes
    # such a field ten times stronger through the transient.
    cases = [("J = 0", None, 1.17e-03, 0.0),
             ("constant |J|=6", sf.ConstantSkew(6.0 * Jopt / nopt), 8.52e-04, args.ramp)]
    cases += [(f"hyperbolic s={s:g}", hyper(s), 1.10e-03, args.ramp_growth)
              for s in args.hyper_s]

    def cell(fld, eta, block, n_relax):
        """One seed block: mean curve, iterations to 2x floor, divergences."""
        sch = samplers.warmup_schedule(T, eta, n_relax) if fld is not None else None
        curves, diverged = [], 0
        for r in range(args.reps):
            x = runner.make_prior("normal10", 3, args.n, np.random.default_rng(block + r))
            st = samplers.field_anchored_step(T, eta, fld, "euler", schedule=sch)
            rr = np.random.default_rng(block * 7 + r)
            w, k, blew = [], 0, False
            for tk in record_at:
                while k < tk:
                    x = st(x, rr)
                    k += 1
                    if not np.all(np.isfinite(x)):
                        blew = True
                        break
                w.append(metrics.axis_sliced_w2(x, T))
                if blew:
                    break
            diverged += int(blew)
            curves.append(w + [np.nan] * (len(record_at) - len(w)))
        w = np.nanmean(curves, axis=0)
        it = np.asarray(record_at)
        ok = np.isfinite(w) & (w <= 2 * floor)
        hit = int(it[np.argmax(ok)]) if ok.any() and diverged == 0 else -1
        return hit, diverged

    print(T)
    print(f"  sliced-W2 floor at n={args.n}: {floor:.4f}")
    print(f"  ramp {args.ramp:g} (constant) / {args.ramp_growth:g} (growth), "
          f"{args.reps} replications per cell, "
          f"{len(args.blocks)} disjoint seed blocks\n")
    head = " ".join(f"{'blk ' + str(b):>17}" for b in args.blocks)
    print(f"{'field':>18} {head}   speed-ups")

    rows, base = [], {}
    for label, fld, eta, n_relax in cases:
        t0 = time.time()
        cells, sp = [], []
        for b in args.blocks:
            hit, div = cell(fld, eta, b, n_relax)
            if fld is None:
                base[b] = hit
            cells.append({"block": b, "iters": hit, "diverged": div})
            sp.append(base[b] / hit if hit > 0 and base.get(b, -1) > 0 else float("nan"))
        shown = " ".join(f"{c['iters']:>7d} ({c['diverged']}/{args.reps} div)" for c in cells)
        print(f"{label:>18} {shown}   " + " ".join(f"{v:5.2f}x" for v in sp)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        rows.append({"label": label, "eta": eta, "ramp": n_relax,
                     "cells": cells, "speedups": sp})

    unstable = [r["label"] for r in rows
                if any(c["diverged"] for c in r["cells"])]
    if unstable:
        print("\n  divergent at this stepsize in at least one seed block: "
              + ", ".join(unstable))

    out = {"meta": {"nu": args.nu, "kappa": args.kappa, "n": args.n,
                    "steps": args.steps, "reps": args.reps, "ramp": args.ramp,
                    "ramp_growth": args.ramp_growth,
                    "blocks": args.blocks, "floor": float(floor)},
           "rows": rows}
    path = os.path.join(OUT, "exp11_seed_robustness.json" if not args.smoke
                        else "exp11_smoke.json")
    print("\nwrote", runner.save_json(out, path))


if __name__ == "__main__":
    main()
