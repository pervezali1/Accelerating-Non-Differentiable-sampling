r"""Playing with the stepsize and the MCP strength, with the two matrices fixed.

The matrices are exactly the ones in the paper and are not varied in form:

    J_a = [[0, a, 0], [-a, 0, a], [0, -a, 0]]                          constant
    J_s(x) = [[0,-s x3, s x2], [s x3, 0, -s x1], [-s x2, s x1, 0]]     = s (x cross .)

What is varied is the stepsize ``eta``, the MCP strength ``lambda``, and the
strengths ``a`` and ``s``.

Why the protocol changed.  Every earlier experiment on this target gave each
field its *equal-bias* stepsize -- the largest ``eta`` whose exact stationary
covariance bias is 2% -- so that all methods sit at the same asymptotic
accuracy before their rates are compared.  That is the right protocol for
comparing rates, but it answers a different question from "which chain reaches
a given accuracy first", because 2% covariance bias is far below the accuracy
the run can actually resolve: with ``n = 5000`` the sliced-W2 estimator's own
two-sample floor is about 0.045, and the measured plateau sits on that floor
until the covariance bias is several times 2%.

So here every field, ``J = 0`` included, is given its own best stepsize: the
one minimising the iterations needed to reach sliced W2 <= 2 x floor and
*stay* there.  A curve that dips under the threshold and climbs back out has
crossed a transient, not reached the accuracy, so the crossing only counts
when every later recorded point stays within 30% of the threshold.

How much this measurement can resolve.  Not much, and ``--seed`` is here to
show it: the crossing is read off a log-spaced grid whose points are 9% apart,
from a handful of chains against a single reference draw, and the estimator
floor that sets the threshold is itself a heavy-tailed average.  Re-running the
*identical* ``J = 0`` configuration on a disjoint seed block moves its crossing
by more than half.  Only speedups computed against ``J = 0`` *within the same
block* mean anything, and only when they survive a change of block and a change
of accuracy level -- which is what :func:`summarise` is for.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skewanchor import analysis, metrics, nonsmooth, runner, samplers, skewfield as sf
from skewanchor.targets import anisotropic_student_t

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "results", "data")
D, NU, KAPPA = 3, 6.0, 100.0
MCP_A, MCP_EPS = 2.0, 0.1
DIRS = [np.asarray(v) for v in np.eye(D)]


def J_a_matrix(a):
    J = np.zeros((3, 3))
    J[0, 1] = a
    J[1, 0] = -a
    J[1, 2] = a
    J[2, 1] = -a
    return J


class CrossSkew:
    """``J_s(x) v = s (x cross v)``.  ``div J = curl(s x) = 0``, so no correction."""

    is_constant = False

    def __init__(self, s):
        self.s = float(s)
        self.d = 3

    def apply(self, x, v):
        return self.s * np.cross(np.atleast_2d(x), np.atleast_2d(v))

    def divergence(self, x, h=None):
        return np.zeros_like(np.atleast_2d(x))


def ensemble(target, core, ref, field, eta, ramp, prior, rec, n, reps, seed=0):
    """Replications in lockstep, run to the end so the bias plateau is visible."""
    xs = [runner.make_prior(prior, D, n, np.random.default_rng(seed + 500 + r))
          for r in range(reps)]
    make = ((lambda: samplers.warmup_schedule(core, eta, ramp)) if field is not None
            else (lambda: None))
    steps = [samplers.field_anchored_step(target, eta, field, "euler", schedule=make())
             for _ in range(reps)]
    rngs = [np.random.default_rng(seed + 9000 + r) for r in range(reps)]
    done, w = 0, []
    for t in rec:
        for r in range(reps):
            for _ in range(t - done):
                xs[r] = steps[r](xs[r], rngs[r])
            if not np.all(np.isfinite(xs[r])):
                return w + [np.nan] * (len(rec) - len(w)), 1
        done = t
        w.append(float(np.mean([metrics.sliced_w2_two_sample(x, ref, DIRS) for x in xs])))
    return w, 0


def sustained(w, rec, floor, slack=1.3):
    """First recorded iteration at or below ``2 x floor`` that the curve stays near."""
    thr = 2.0 * floor
    w = np.asarray(w, dtype=float)
    if not np.all(np.isfinite(w)):
        return -1
    for i, v in enumerate(w):
        if v <= thr and np.all(w[i:] <= slack * thr):
            return int(rec[i])
    return -1


def exact_ceiling(target, family, bias=0.02, strengths=None):
    """Best equal-bias speedup a one-parameter family of skew matrices can reach.

    Computed on :func:`skewanchor.nonsmooth.gaussianised`, the log-quadratic
    stand-in with the same anchor Hessian at the origin, so it is exact
    arithmetic rather than a simulation.
    """
    g = nonsmooth.gaussianised(target)
    strengths = np.geomspace(0.05, 12.0, 60) if strengths is None else strengths
    r0 = analysis.ms_rate(g, None, analysis.eta_for_bias(g, None, bias))
    best = (0.0, 0.0)
    for a in strengths:
        eta = analysis.eta_for_bias(g, family(a), bias)
        if eta <= 0:
            continue
        r = analysis.ms_rate(g, family(a), eta) / r0
        if r > best[0]:
            best = (float(r), float(a))
    return best


LEVELS = (1.3, 1.5, 2.0, 3.0)


def summarise(rows, lams):
    """Iterations to several accuracy levels, read back off the stored curves.

    The ``2 x floor`` criterion saturates: past a certain speed every chain
    meets it at the same recorded iteration whatever its stepsize, because the
    threshold sits close to the estimator's own floor and the curve flattens
    onto it. Tightening the level separates the fields again -- and tightening
    it is what favours the *small* stepsize, so it is the comparison a skew
    field would most like to be judged on.
    """
    for lam in lams:
        sub = [r for r in rows if r["lam"] == lam]
        if not sub:
            continue
        floor = sub[0]["floor"]
        labels = list(dict.fromkeys(r["label"] for r in sub))
        best = {}
        print(f"\n=== lambda = {lam:g}: iterations to each accuracy level "
              f"(floor {floor:.4f}), with the stepsize that gets there")
        print(f"{'':>15}" + "".join(f"{(str(L) + 'x floor'):>14}" for L in LEVELS))
        for k in labels:
            rs = sorted((r for r in sub if r["label"] == k), key=lambda r: r["mult"])
            cells, picks = [], []
            for L in LEVELS:
                cand = [(sustained(r["w2"], r["rec"], floor * L / 2.0), r["mult"]) for r in rs]
                cand = [c for c in cand if c[0] > 0]
                if cand:
                    it, m = min(cand)
                    picks.append(it)
                    cells.append(f"{it:8d} @{m:4.2g}")
                else:
                    picks.append(None)
                    cells.append(f"{'--':>14}")
            best[k] = picks
            print(f"{k:>15}" + "".join(cells))
        print(f"{'speedup':>15}" + "".join(f"{(str(L) + 'x floor'):>14}" for L in LEVELS))
        for k in labels:
            if k == "J = 0":
                continue
            cells = []
            for i in range(len(LEVELS)):
                b0, bk = best["J = 0"][i], best[k][i]
                cells.append(f"{b0 / bk:13.2f}x" if b0 and bk else f"{'--':>14}")
            print(f"{k:>15}" + "".join(cells))


def aggregate():
    """Every field against ``J = 0`` *within its own block*, over all blocks and levels.

    Absolute crossings are not comparable across seed blocks -- the estimator
    floor that sets the threshold is itself a heavy-tailed average and moves by
    30% between blocks -- so the only meaningful number is the ratio to
    ``J = 0`` measured in the same block, and the only honest summary is that
    ratio's spread over blocks and accuracy levels.
    """
    files = sorted(glob.glob(os.path.join(DATA, "exp16_stepsize_and_lambda_*.json")))
    if not files:
        print("nothing to aggregate")
        return
    agg, table = {}, []
    for f in files:
        b = json.load(open(f))
        rows = b["rows"]
        floor = rows[0]["floor"]
        name = (os.path.basename(f)[len("exp16_stepsize_and_lambda_"):-len(".json")])

        def best(label, level):
            hits = [sustained(r["w2"], r["rec"], floor * level / 2.0)
                    for r in rows if r["label"] == label]
            hits = [h for h in hits if h > 0]
            return min(hits) if hits else None

        for label in dict.fromkeys(r["label"] for r in rows):
            if label == "J = 0":
                continue
            row = []
            for level in LEVELS:
                z, k = best("J = 0", level), best(label, level)
                v = (z / k) if (z and k) else None
                row.append(v)
                if v:
                    agg.setdefault(label, []).append(v)
            table.append((name, label, row))
    print("\nevery field against J = 0 in its own block, at each accuracy level")
    print(f"{'block':>26} {'field':>15}" + "".join(f"{(str(L) + 'x'):>10}" for L in LEVELS))
    for name, label, row in table:
        print(f"{name:>26} {label:>15}"
              + "".join(f"{v:9.2f}x" if v else f"{'--':>10}" for v in row))
    print(f"\n{'field':>15} {'cells':>6} {'geometric mean':>16} {'range':>16}")
    for label, vs in agg.items():
        print(f"{label:>15} {len(vs):6d} {np.exp(np.mean(np.log(vs))):15.2f}x "
              f"{min(vs):7.2f}-{max(vs):.2f}x")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--mults", type=float, nargs="+", default=[1, 1.41, 2, 2.83, 4, 5.66, 8])
    ap.add_argument("--avals", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--svals", type=float, nargs="+", default=[0.05, 0.1])
    ap.add_argument("--prior", default="normal10")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0,
                    help="offset for every seed -- the priors, the chains, the reference "
                         "sample and the floor. Re-running a cell on a disjoint block is "
                         "the only way to tell a real gap from the grid's resolution.")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--ceiling-lams", type=float, nargs="+",
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5])
    ap.add_argument("--out", default=None)
    ap.add_argument("--aggregate", action="store_true",
                    help="skip the chains; just summarise every exp16 json already written")
    args = ap.parse_args()

    if args.aggregate:
        aggregate()
        return

    core = anisotropic_student_t(D, NU, KAPPA)
    eta0 = analysis.eta_for_bias(core, None, 0.02)      # the protocol used up to now
    rec = runner.log_schedule(args.steps, 70)
    out = args.out or (f"exp16_stepsize_and_lambda_{args.prior}"
                       + (f"_seed{args.seed}" if args.seed else "") + ".json")

    fields = ([("J = 0", "zero", None, 0.0, 0.0)]
              + [(f"J_a,  a = {a:g}", "const", sf.ConstantSkew(J_a_matrix(a)), a, 20.0)
                 for a in args.avals]
              + [(f"J_s,  s = {s:g}", "cross", CrossSkew(s), s, 40.0) for s in args.svals])

    print(f"prior {args.prior}   n {args.n}   reps {args.reps}   steps {args.steps}   "
          f"seed block {args.seed}")
    print(f"eta0 (the old equal-bias stepsize) {eta0:.4e}\n", flush=True)

    # The exact equal-bias ceilings are pure arithmetic on the log-quadratic
    # stand-in, so they are swept over a finer grid of lambda than the chains.
    ceilings = []
    for lam in sorted(set(args.ceiling_lams) | set(args.lams)):
        t = nonsmooth.make(D, NU, KAPPA, lam=lam, a=MCP_A, eps=MCP_EPS)
        g = nonsmooth.gaussianised(t)
        i, j = int(np.argmin(np.diag(t.Sigma))), int(np.argmax(np.diag(t.Sigma)))

        def stiff_soft(a, i=i, j=j):
            J = np.zeros((3, 3))
            J[i, j] = a
            J[j, i] = -a
            return J

        c_tri = exact_ceiling(t, J_a_matrix)
        c_ss = exact_ceiling(t, stiff_soft)
        ceilings.append(dict(lam=lam, cond=float(np.linalg.cond(g.Sigma)),
                             kappa=nonsmooth.smoothed_curvature(lam, MCP_A, MCP_EPS),
                             tridiagonal=c_tri[0], tridiagonal_a=c_tri[1],
                             stiff_soft=c_ss[0], stiff_soft_a=c_ss[1]))
    print("exact equal-bias ceilings on the log-quadratic stand-in")
    print(f"{'lambda':>7} {'kappa':>8} {'cond':>7} {'tridiagonal J_a':>20} {'any skew J':>20}")
    for c in ceilings:
        print(f"{c['lam']:7.2f} {c['kappa']:8.3f} {c['cond']:7.1f} "
              f"{c['tridiagonal']:9.2f}x at a={c['tridiagonal_a']:5.2f} "
              f"{c['stiff_soft']:9.2f}x at a={c['stiff_soft_a']:5.2f}")
    print(flush=True)

    rows = []
    for lam in args.lams:
        target = nonsmooth.make(D, NU, KAPPA, lam=lam, a=MCP_A, eps=MCP_EPS)
        rng = np.random.default_rng(args.seed)
        target.cov(rng, 1_000_000)
        blocks = [target.sample(args.n, rng) for _ in range(12)]
        floor = float(np.mean([metrics.sliced_w2_two_sample(blocks[i], blocks[i + 1], DIRS)
                               for i in range(0, 12, 2)]))
        ref = target.sample(args.n, rng)

        c = next(c for c in ceilings if c["lam"] == lam)
        c["floor"] = floor
        print(f"=== lambda = {lam:g}   floor {floor:.4f}   threshold {2 * floor:.4f}   "
              f"acceptance {target.acceptance_rate(rng, 200_000):.3f}")
        print(f"    effective cond {c['cond']:.1f};  exact equal-bias ceiling: "
              f"tridiagonal {c['tridiagonal']:.2f}x, "
              f"ideal stiff<->soft {c['stiff_soft']:.2f}x")
        print(f"{'field':>14} {'eta/eta0':>9} {'div':>5} {'plateau':>8} {'iters':>7} {'sp':>7}",
              flush=True)

        base = None
        for label, kind, field, strength, ramp in fields:
            best = None
            for m in args.mults:
                eta = m * eta0
                w, div = ensemble(target, core, ref, field, eta, ramp,
                                  args.prior, rec, args.n, args.reps, args.seed)
                hit = sustained(w, rec, floor)
                tail = np.asarray(w[-6:], dtype=float)
                plateau = float(np.nanmean(tail)) if np.any(np.isfinite(tail)) else float("nan")
                if hit > 0 and (best is None or hit < best[0]):
                    best = (hit, eta, m, w)
                sp = (base / hit) if (base and hit > 0) else float("nan")
                print(f"{label:>14} {m:9.2f} {div:>3}/{args.reps} {plateau:8.4f} "
                      f"{hit:7d} {sp:6.2f}x", flush=True)
                rows.append(dict(lam=lam, label=label, kind=kind, strength=strength,
                                 ramp=ramp, mult=m, eta=eta, diverged=div, iters=hit,
                                 plateau=plateau, floor=floor, prior=args.prior,
                                 w2=[None if not np.isfinite(v) else float(v) for v in w],
                                 rec=list(rec)))
                json.dump(dict(prior=args.prior, n=args.n, reps=args.reps, steps=args.steps,
                               eta0=eta0, mcp_a=MCP_A, mcp_eps=MCP_EPS, seed=args.seed,
                               ceilings=ceilings, rows=rows),
                          open(os.path.join(DATA, out), "w"), indent=1)
            if kind == "zero":
                base = best[0] if best else None
            if best:
                print(f"{'-> best eta':>14} {best[2]:9.2f} {'':>5} {'':>8} {best[0]:7d} "
                      f"{(base / best[0]) if base else float('nan'):6.2f}x\n", flush=True)
            else:
                print(f"{'-> best eta':>14}        --  never reaches the accuracy\n", flush=True)

    summarise(rows, args.lams)
    json.dump(dict(prior=args.prior, n=args.n, reps=args.reps, steps=args.steps,
                   eta0=eta0, mcp_a=MCP_A, mcp_eps=MCP_EPS, seed=args.seed,
                   ceilings=ceilings, rows=rows, levels=list(LEVELS)),
              open(os.path.join(DATA, out), "w"), indent=1)
    print("wrote", os.path.join(DATA, out))


if __name__ == "__main__":
    main()
