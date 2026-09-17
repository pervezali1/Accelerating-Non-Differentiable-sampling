"""What does tightening the smoothed-l_p level Lambda do?

Lambda only enters the smoothed set K = {g(beta) <= Lambda}, g(beta) = sum_i (beta_i^2+eps^2)^{p/2}.
Facts that fix the experiment (see the printed geometry table):

  * D = Lambda - 9 eps^p must stay > 0. At Lambda = 1, D = 0.811 (MAGIC) / 0.853 (Titanic),
    so the anchor construction and the bound 1/2 <= a <= 1 on K are untouched.
  * K shrinks: Euclidean radius 1.74-2.05 at Lambda = 4 down to 0.90-1.07 at Lambda = 1.
  * The specified unit-ball initialisation becomes INFEASIBLE: max g over the unit ball is
    1.216 (MAGIC) / 1.170 (Titanic), which exceeds Lambda = 1. Something has to give, so every
    run here starts from a common smaller ball of radius INIT_RADIUS, feasible for every Lambda
    compared. That is a declared deviation from the original initialisation.
  * ||grad g|| on the boundary falls from ~4.6 to ~1.8, and J's axis is -s grad g, so at fixed s
    the non-reversible drift is ~2.5x weaker at Lambda = 1 than at Lambda = 4.

Also worth stating before looking at any number: accuracy depends only on sign(X beta), so it is
INVARIANT under beta -> c beta for c > 0. Shrinking K by a near-uniform factor therefore cannot
change accuracy by itself. Any effect has to come from (i) the anisotropy of the l_p set
distorting the DIRECTION of beta, or (ii) the fixed diffusion sqrt(2 h a) being large relative to
a smaller set, which adds directional noise.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from nral import (BATCH_SIZE, CHECKPOINT_EVERY, RHO_ANCHORED, ExperimentSpec, SmoothedLpGeometry,
                  accuracy_curve, build_magic_dataset, build_titanic_dataset,
                  make_initial_states, potential_U, run_sampler, stream_seed_for)

S5 = (5.0, 5.0, 5.0)
LAMBDAS = [4.0, 2.0, 1.0, 0.5]
H_GRID = [1e-4, 1e-5]
INIT_RADIUS = 0.55            # feasible for every Lambda above, in both datasets
SEED = 4100

SPECS = [
    ExperimentSpec("magic_lp",   "magic",   "lp", 1000, S5, "magic_lp",   p=2.4, eps=0.20, Lam=4.0),
    ExperimentSpec("titanic_lp", "titanic", "lp", 2000, S5, "titanic_lp", p=2.4, eps=0.18, Lam=4.0),
]


def run_cell(ds, spec, Lam, h, n_reps, seed=SEED):
    geom = SmoothedLpGeometry(p=spec.p, eps=spec.eps, Lam=Lam)
    beta0 = make_initial_states(spec.dataset, n_reps, sampler_seed=seed) * INIT_RADIUS
    assert np.all(geom.feasible(beta0)), "initial states infeasible for this Lambda"
    out = {}
    for mk, alpha in (("ranch", 0), ("nranch", 1)):
        res = run_sampler(ds.X_train, ds.y_train, geom, rho=RHO_ANCHORED, alpha=alpha,
                          block_scales=S5, h=h, n_iter=spec.n_iter, beta0=beta0,
                          stream_seed=stream_seed_for(spec.dataset, "main", sampler_seed=seed),
                          m=BATCH_SIZE, checkpoint_every=CHECKPOINT_EVERY)
        b = res["betas"]
        out[mk] = dict(acc_train=accuracy_curve(ds.X_train, ds.y_train, b),
                       acc_test=accuracy_curve(ds.X_test, ds.y_test, b),
                       betas=b, projection_rate=res["projection_rate"],
                       n_nonfinite=res["n_nonfinite"],
                       train_loss=potential_U(b[-1], ds.X_train, ds.y_train))
    return out, geom


def confirm_spec_init(datasets, n_reps=100, h=1e-5, lams=(4.0, 2.0), seed=SEED):
    """Re-check the Lambda comparison with the ORIGINAL unit-ball initialisation.

    The sweep above had to shrink the initial ball to radius 0.55 so that every Lambda was
    feasible, and that shrink is itself a confound. Lambda = 1 cannot be included here at all:
    max g over the unit ball is 1.216 (MAGIC) / 1.170 (Titanic), so the specified initialisation
    is infeasible for it. That is the point.
    """
    rows = []
    for spec in SPECS:
        ds = datasets[spec.dataset]
        for Lam in lams:
            geom = SmoothedLpGeometry(p=spec.p, eps=spec.eps, Lam=Lam)
            b0 = make_initial_states(spec.dataset, n_reps, sampler_seed=seed)
            assert np.all(geom.feasible(b0)), f"unit-ball init infeasible at Lambda = {Lam}"
            arms = {}
            for mk, alpha in (("ranch", 0), ("nranch", 1)):
                r = run_sampler(ds.X_train, ds.y_train, geom, rho=RHO_ANCHORED, alpha=alpha,
                                block_scales=S5, h=h, n_iter=spec.n_iter, beta0=b0,
                                stream_seed=stream_seed_for(spec.dataset, "main", sampler_seed=seed),
                                m=BATCH_SIZE, checkpoint_every=CHECKPOINT_EVERY)
                b = r["betas"]
                arms[mk] = dict(tr=accuracy_curve(ds.X_train, ds.y_train, b)[-1],
                                te=accuracy_curve(ds.X_test, ds.y_test, b)[-1],
                                U=float(potential_U(b[-1], ds.X_train, ds.y_train).mean()),
                                proj=r["projection_rate"])
            dtr = arms["nranch"]["tr"] - arms["ranch"]["tr"]
            dte = arms["nranch"]["te"] - arms["ranch"]["te"]
            rows.append(dict(
                dataset=spec.dataset, Lam=Lam, h=h, init="unit ball (as specified)",
                rev_train=float(arms["ranch"]["tr"].mean()),
                nonrev_train=float(arms["nranch"]["tr"].mean()),
                d_train=float(dtr.mean()),
                t_train=float(dtr.mean() / (dtr.std(ddof=1) / np.sqrt(n_reps))),
                rev_test=float(arms["ranch"]["te"].mean()),
                nonrev_test=float(arms["nranch"]["te"].mean()),
                d_test=float(dte.mean()),
                t_test=float(dte.mean() / (dte.std(ddof=1) / np.sqrt(n_reps))),
                rev_U=arms["ranch"]["U"], nonrev_U=arms["nranch"]["U"],
                proj_rev=arms["ranch"]["proj"], proj_nonrev=arms["nranch"]["proj"]))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--resdir", default="results/strength")
    args = ap.parse_args()
    os.makedirs(args.resdir, exist_ok=True)
    datasets = {"magic": build_magic_dataset(), "titanic": build_titanic_dataset()}

    rows = []
    t0 = time.perf_counter()
    for spec in SPECS:
        ds = datasets[spec.dataset]
        for h in H_GRID:
            for Lam in LAMBDAS:
                arms, geom = run_cell(ds, spec, Lam, h, args.reps)
                d_tr = arms["nranch"]["acc_train"][-1] - arms["ranch"]["acc_train"][-1]
                d_te = arms["nranch"]["acc_test"][-1] - arms["ranch"]["acc_test"][-1]
                n = len(d_tr)
                se_tr = d_tr.std(ddof=1) / np.sqrt(n)
                se_te = d_te.std(ddof=1) / np.sqrt(n)
                nb = np.linalg.norm(arms["nranch"]["betas"][-1], axis=1)
                rb = np.linalg.norm(arms["ranch"]["betas"][-1], axis=1)
                rows.append(dict(
                    dataset=spec.dataset, h=h, Lam=Lam, D=geom.D,
                    rev_train=float(arms["ranch"]["acc_train"][-1].mean()),
                    rev_test=float(arms["ranch"]["acc_test"][-1].mean()),
                    rev_test_sd=float(arms["ranch"]["acc_test"][-1].std(ddof=1)),
                    nonrev_train=float(arms["nranch"]["acc_train"][-1].mean()),
                    nonrev_test=float(arms["nranch"]["acc_test"][-1].mean()),
                    nonrev_test_sd=float(arms["nranch"]["acc_test"][-1].std(ddof=1)),
                    d_train=float(d_tr.mean()), t_train=float(d_tr.mean() / (se_tr + 1e-300)),
                    d_test=float(d_te.mean()), t_test=float(d_te.mean() / (se_te + 1e-300)),
                    rev_norm=float(rb.mean()), nonrev_norm=float(nb.mean()),
                    rev_U=float(arms["ranch"]["train_loss"].mean()),
                    nonrev_U=float(arms["nranch"]["train_loss"].mean()),
                    proj_rev=arms["ranch"]["projection_rate"],
                    proj_nonrev=arms["nranch"]["projection_rate"]))
                print(f"  {spec.dataset:8s} h={h:g} Lambda={Lam:<4g} done "
                      f"({time.perf_counter()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.resdir, "lambda_study.csv"), index=False)

    pd.set_option("display.width", 200)
    for dsname in ("magic", "titanic"):
        for h in H_GRID:
            sub = df[(df.dataset == dsname) & (df.h == h)]
            print(f"\n=== {dsname}, smoothed l_p, s = (5,5,5), h = {h:g}, R = {args.reps} ===")
            print(sub[["Lam", "D", "rev_train", "nonrev_train", "d_train", "t_train",
                       "rev_test", "nonrev_test", "d_test", "t_test",
                       "rev_norm", "nonrev_norm", "proj_rev", "proj_nonrev"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    conf = confirm_spec_init(datasets)
    conf.to_csv(os.path.join(args.resdir, "lambda_spec_init.csv"), index=False)
    print("\n=== same comparison with the SPECIFIED unit-ball init (Lambda = 1 cannot appear: "
          "the unit ball is not inside it) ===")
    print(conf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\ntotal {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
