#!/usr/bin/env python3
"""Can the two skew fields be made to beat ``J = 0``, and at what amplitude?

The headline runs in ``run_anchored_srnsgld.py`` set the rotation amplitude from
one conservative a-priori rule (a step's rotational displacement at most a tenth
of the radius).  That rule says nothing about whether the amplitude is any
*good*.  This sweeps it, one field at a time, over a geometric ladder of
multiples of the paper's own ``a`` and ``s``, with several walker seeds shared
across fields, and scores each run two ways:

* ``band`` -- the first scored iteration whose mean test accuracy is within
  0.005 of the exact constrained lasso posterior's, which is the paper's
  "fewer iterations to the same accuracy";
* ``error`` -- the whitened distance from the running mean to the exact
  posterior's mean, at the end of the run and a quarter of the way in, which is
  what accuracy stops resolving once it saturates.

``--step-divisors`` also sweeps the step size, because on the larger datasets
the paper's ``eta`` is so big that every field reaches the accuracy ceiling
within a dozen iterations and there is nothing left for a field to win.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
RESULTS = os.path.join(ROOT, "results")

from nds.anchored_constrained import (  # noqa: E402
    ConstantField,
    ZeroField,
    anchor_hessian,
    ball_axial_field,
    constrained_lasso_map,
    curvature_blocks,
    reference_path,
    run_anchored_srnsgld,
    outward_tilt_direction,
    sublevel_axial_field,
    tilted_axial_field,
)
from nds.constrained import FramedField, eigen_frame  # noqa: E402


def make_field(key: str, d: int, dcfg: dict, kappa: float, domain_key: str,
               profile: str, frame: np.ndarray | None = None, tilt: float = 0.0,
               domain=None, direction=None):
    """The paper's field at ``kappa`` times its own amplitude.

    ``frame`` is an optional orthogonal change of basis, which for the
    state-dependent field regroups which coordinates share a ``3``-block.  That
    keeps every assumption the construction needs -- the conjugated field is
    still skew, still divergence free, still annihilates ``x``, and a centred
    ball is invariant under it -- so it is the same construction read in a
    different frame, not a different one.
    """
    if key == "zero":
        return ZeroField(d)
    if key == "constant":
        return ConstantField(d, dcfg["a"] * kappa)
    s = np.atleast_1d(np.asarray(dcfg["s"], float))
    if profile == "flat":
        s = np.full_like(s, s.mean())
    s = s * kappa
    if tilt:
        field = tilted_axial_field(
            d, s.tolist(), domain, direction, tilt,
            p=2.0 if domain_key == "ball" else dcfg["p"],
            eps=0.0 if domain_key == "ball" else dcfg["eps"],
        )
    else:
        field = (
            ball_axial_field(d, s.tolist()) if domain_key == "ball"
            else sublevel_axial_field(d, s.tolist(), dcfg["p"], dcfg["eps"])
        )
    if frame is None:
        return field
    wrapped = FramedField(field, frame, label=field.label)
    wrapped.key = field.key
    return wrapped


def band(accuracy: np.ndarray, scored_at: np.ndarray, ref: float, tol: float = 0.005):
    mean = accuracy.mean(axis=1)
    hit = np.flatnonzero(np.abs(mean - ref) < tol)
    return int(scored_at[hit[0]]) if len(hit) else None


def main() -> None:
    from run_anchored_srnsgld import build  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="titanic")
    parser.add_argument("--domain", default="ball", choices=["ball", "lp"])
    parser.add_argument("--kappas", nargs="+", type=float,
                        default=[0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 2.0])
    parser.add_argument("--step-divisors", nargs="+", type=float, default=[1.0],
                        help="each divisor scales eta down and the iteration count up, "
                             "holding the simulated time fixed")
    parser.add_argument("--eta", type=float, default=None,
                        help="override the step size without touching the iteration "
                             "count, for a fixed-budget comparison")
    parser.add_argument("--tilts", nargs="+", type=float, default=[0.0],
                        help="non-radial h in the paper's recipe: h = 1 + tilt (u . x). "
                             "0 is the paper's own choice, h = 1")
    parser.add_argument("--tilt-direction", default="outward",
                        choices=["outward", "map", "slow", "e1"],
                        help="direction of the non-radial part of h; 'outward' is the "
                             "closed-form maximiser of the mean outward radial push")
    parser.add_argument("--fields", nargs="+", default=["zero", "constant", "state"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--profile", default="paper", choices=["paper", "flat"])
    parser.add_argument("--block-order", default="columns",
                        choices=["columns", "curvature", "eigen"],
                        help="which coordinates share a 3-block of the state-dependent "
                             "field: the paper's column order, a permutation pairing slow "
                             "with fast coordinates, or the anchor Hessian's eigenbasis")
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--n-walkers", type=int, default=None)
    parser.add_argument("--score-every", type=int, default=None)
    parser.add_argument("--start-radius", type=float, default=1.0)
    parser.add_argument("--regularizer", default="l1",
                        choices=["l1", "group", "tv", "linf"])
    parser.add_argument("--lam-scale", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--clock-budget", type=float, default=0.5)
    parser.add_argument("--rotation-budget", type=float, default=0.1)
    parser.add_argument("--tag", default="")
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()
    args.amp_scale = 1.0  # build() only needs this to construct its own fields
    args.amp_scale_constant = None
    args.amp_scale_state = None
    args.tilt = 0.0
    args.no_clock = False
    args.step_size = None
    args.seed = args.seeds[0]

    cfg, dcfg, data, domain, target, _, _ = build(args.problem, args.domain, args)
    d = target.d
    n_iter_base = args.n_iter or dcfg.get("n_iter", cfg["n_iter"])
    n_walkers = args.n_walkers or cfg["n_walkers"]
    eta_base = args.eta or cfg["step_size"]

    ref = dict(np.load(os.path.join(
        RESULTS,
        reference_path(args.problem, args.domain, target.lam, args.regularizer))))
    ref_mean = np.asarray(ref["posterior_mean"], float)
    metric = np.linalg.pinv(np.asarray(ref["posterior_cov"], float))
    ref_acc = float(ref["accuracy_test"])

    frame = None
    x_star = constrained_lasso_map(target, domain)
    H = anchor_hessian(target, x_star)
    if args.block_order != "columns":
        frame = (
            curvature_blocks(target, domain) if args.block_order == "curvature"
            else eigen_frame(H)
        )
    if args.tilt_direction == "outward":
        direction = outward_tilt_direction(
            target, domain, start_radius=args.start_radius, seed=args.seeds[0]
        )
    elif args.tilt_direction == "map":
        direction = x_star
    elif args.tilt_direction == "slow":
        direction = np.linalg.eigh(H)[1][:, 0]
    else:
        direction = np.eye(d)[0]

    rows = []
    for div in args.step_divisors:
        eta = eta_base / div
        n_iter = int(round(n_iter_base * div))
        score_every = args.score_every or max(1, n_iter // 200)
        quarter = n_iter // 4
        plans = [("zero", 0.0, 0.0)] if "zero" in args.fields else []
        if "constant" in args.fields:
            plans += [("constant", k, 0.0) for k in args.kappas]
        if "state" in args.fields:
            plans += [("state", k, t) for k in args.kappas for t in args.tilts]
        for key, kappa, tilt in plans:
            field = make_field(key, d, dcfg, kappa, args.domain, args.profile,
                               frame if key == "state" else None, tilt=tilt,
                               domain=domain, direction=direction)
            started = time.time()
            per_seed = []
            for seed in args.seeds:
                out = run_anchored_srnsgld(
                    target, field, domain, data, n_iter=n_iter, n_walkers=n_walkers,
                    step_size=eta, batch_size=cfg["batch"], seed=seed,
                    start_radius=args.start_radius, reference_mean=ref_mean,
                    reference_metric=metric, score_every=score_every,
                )
                b = band(out["accuracy_test"], out["scored_at"], ref_acc)
                q = int(np.searchsorted(out["scored_at"], quarter))
                # the tail of the budget: with a fixed, short horizon what matters
                # is where the chain has got to by the last iteration, not when it
                # first touched a band
                tail = out["accuracy_test"][int(0.8 * len(out["accuracy_test"])):]
                per_seed.append({
                    "band": b if b is not None else n_iter,
                    "reached": b is not None,
                    "error": out["mean_error"],
                    "error_running": float(out["running_error"][-1]),
                    "error_quarter": float(out["running_error"][min(q, len(out["running_error"]) - 1)]),
                    "gap_tail": float(abs(tail.mean() - ref_acc)),
                    "accuracy_tail": float(tail.mean()),
                    "accuracy_test": float(out["accuracy_test"][-1].mean()),
                    "boundary_rate": out["boundary_rate"],
                    "failed": out["failed_retractions"],
                    "clock": float(out["clock"].mean()),
                })
            agg = {
                "field": key,
                "kappa": kappa,
                "tilt": tilt,
                "eta": eta,
                "n_iter": n_iter,
                "reached_all": all(r["reached"] for r in per_seed),
            }
            for stat in ("band", "error", "error_running", "error_quarter",
                         "gap_tail", "accuracy_tail", "accuracy_test",
                         "boundary_rate", "failed", "clock"):
                vals = np.array([r[stat] for r in per_seed], float)
                agg[stat] = float(vals.mean())
                agg[stat + "_se"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            rows.append(agg)
            print(
                f"eta={eta:.2e} {key:9s} kappa={kappa:<5g} tilt={tilt:<4g} band {agg['band']:7.1f}"
                f" +/- {agg['band_se']:5.1f}{'' if agg['reached_all'] else ' (not all reached)'}"
                f"  error {agg['error']:8.3f} +/- {agg['error_se']:.3f}"
                f"  run {agg['error_running']:8.3f}"
                f"  gap_tail {agg['gap_tail']:.4f} +/- {agg['gap_tail_se']:.4f}"
                f"  acc {agg['accuracy_tail']:.4f}  bnd {agg['boundary_rate']:.3f}"
                f"  failed {agg['failed']:.0f}  ({time.time() - started:.0f}s)",
                flush=True,
            )

    tag = f"{args.problem}_{args.domain}"
    if args.regularizer != "l1":
        tag += f"_{args.regularizer}"
    if args.block_order != "columns":
        tag += f"_{args.block_order}"
    if any(args.tilts):
        tag += f"_tilt{args.tilt_direction}"
    tag += args.tag
    meta = {
        "problem": args.problem,
        "domain": args.domain,
        "profile": args.profile,
        "block_order": args.block_order,
        "regularizer": args.regularizer,
        "lam": target.lam,
        "delta": target.delta,
        "n_walkers": n_walkers,
        "batch_size": cfg["batch"],
        "seeds": args.seeds,
        "reference": {"accuracy_test": ref_acc, "posterior_mean": ref_mean.tolist()},
        "paper_amplitudes": {"a": dcfg["a"], "s": np.atleast_1d(dcfg["s"]).tolist()},
        "rows": rows,
    }
    with open(os.path.join(RESULTS, f"sweep_anchored_{tag}.json"), "w") as h:
        json.dump(meta, h, indent=2)

    lines = [
        f"# Amplitude sweep: {args.problem}, {args.domain}",
        "",
        f"`lam` = {target.lam:.4g}, `delta` = {target.delta:.4g}, "
        f"{n_walkers} walkers, batch {cfg['batch']}, seeds {args.seeds}, "
        f"profile `{args.profile}`, block order `{args.block_order}`.  "
        f"`kappa` multiplies the paper's own amplitudes "
        f"(`a` = {dcfg['a']:g}, `s` = {np.atleast_1d(dcfg['s']).tolist()}).",
        "",
        "| eta | iterations | field | kappa | tilt | band | gap at the tail | "
        "error | running error | accuracy | boundary | failed |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        star = "" if r["reached_all"] else " (not all)"
        lines.append(
            f"| {r['eta']:.2e} | {r['n_iter']} | `{r['field']}` | {r['kappa']:g} | "
            f"{r['tilt']:g} | "
            f"{r['band']:.0f} +/- {r['band_se']:.0f}{star} | "
            f"{r['gap_tail']:.4f} +/- {r['gap_tail_se']:.4f} | "
            f"{r['error']:.3f} +/- {r['error_se']:.3f} | {r['error_running']:.3f} | "
            f"{r['accuracy_tail']:.4f} | {r['boundary_rate']:.3f} | {r['failed']:.0f} |"
        )
    open(os.path.join(RESULTS, f"sweep_anchored_{tag}.md"), "w").write("\n".join(lines) + "\n")
    print("wrote", os.path.join(RESULTS, f"sweep_anchored_{tag}.md"))


if __name__ == "__main__":
    main()
