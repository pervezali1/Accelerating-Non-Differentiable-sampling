"""Joint (step size h, block strength s) search for the two ANCHORED methods.

Motivation. At the originally specified h = 1e-4 no block strength makes non-reversible
anchored Langevin beat the reversible arm on accuracy (see strength_search.py): on MAGIC the
extra J drift is one to two times the diameter of K per step, so it is pure discretisation
error, and on Titanic the two arms have both already converged by the end of the run, so the
accuracy metric has saturated and cannot express a difference. h = 1e-4 was already documented
as too large for MAGIC, so refining it is a correction, not a fudge.

This script therefore sweeps h and s together at the SPECIFIED iteration budgets, because the
figures' x-axis is iterations: a smaller h keeps the chain inside its transient for the whole
plotted range, which is exactly where a non-reversible drift can do work, and it also makes
larger s numerically affordable.

Protocol, unchanged from strength_search.py:
  * the reversible arm is independent of s and is run once per (experiment, h);
  * every comparison is paired -- same beta0, same mini-batches, same Gaussian increments;
  * (h, s) is SELECTED ON TRAINING DATA ONLY (mean training accuracy over the last 25% of
    checkpoints). Test labels choose nothing; the test column is a held-out read afterwards;
  * the entire grid is written out, so the size of the search is visible;
  * s = 0 is a control: with J = 0 the arms must be bit-identical.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

import nral
from nral import H_REPORTED, ExperimentSpec, build_magic_dataset, build_titanic_dataset
from strength_search import SWEEP, sweep_one, final_figure

H_GRID = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6]

SPECS = [
    ExperimentSpec("magic_ball",   "magic",   "ball", 1000, (5., 5., 5.), "magic_ball"),
    ExperimentSpec("magic_lp",     "magic",   "lp",   1000, (5., 5., 5.), "magic_lp",
                   p=2.4, eps=0.20, Lam=4.0),
    ExperimentSpec("titanic_ball", "titanic", "ball", 1500, (2., 7., 2.), "titanic_ball"),
    ExperimentSpec("titanic_lp",   "titanic", "lp",   2000, (2., 7., 2.), "titanic_lp",
                   p=2.4, eps=0.18, Lam=4.0),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-reps", type=int, default=25)
    ap.add_argument("--final-reps", type=int, default=100)
    ap.add_argument("--outdir", default="figures/strength")
    ap.add_argument("--resdir", default="results/strength")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.sweep_reps, args.final_reps = 4, 6
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.resdir, exist_ok=True)

    datasets = {"magic": build_magic_dataset(), "titanic": build_titanic_dataset()}
    specs = [dataclasses.replace(s, n_iter=150) for s in SPECS] if args.quick else SPECS

    t0 = time.perf_counter()
    all_rows, chosen = [], {}
    for spec in specs:
        ds = datasets[spec.dataset]
        frames = []
        for h in H_GRID:
            df = sweep_one(ds, spec, args.sweep_reps, spec.n_iter, h=h)
            frames.append(df)
            print(f"  [{spec.key}] h = {h:g} done ({time.perf_counter()-t0:.0f}s)")
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(os.path.join(args.resdir, f"hs_sweep_{spec.key}.csv"), index=False)
        all_rows.append(df)

        nr = df[df.method == "non-reversible"].copy()
        best = nr.loc[nr.sel_train_acc.idxmax()]
        chosen[spec.key] = dict(h=float(best.h), s=list(best.s_tuple), s_label=best.s,
                                sel_train_acc=float(best.sel_train_acc),
                                paired_dtrain=float(best.paired_dtrain),
                                paired_dtrain_se=float(best.paired_dtrain_se))
        print(f"\n=== {spec.key}: best non-reversible cell by TRAINING accuracy ===")
        top = nr.sort_values("sel_train_acc", ascending=False).head(8)
        print(top[["h", "s", "sel_train_acc", "final_train_acc", "final_test_acc",
                   "paired_dtrain", "paired_dtrain_se", "train_loss_U", "projection_rate"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        gain = nr.sort_values("paired_dtrain", ascending=False).head(8)
        print(f"  largest PAIRED TRAINING gains over the reversible arm at the same h:")
        print(gain[["h", "s", "paired_dtrain", "paired_dtrain_se", "final_train_acc",
                    "final_test_acc", "projection_rate"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # Final R = 100 runs at the training-selected (h, s) for each experiment.
    finals = {}
    for spec in specs:
        c = chosen[spec.key]
        note = (f"Step size h and block strengths s selected by a TRAINING-ONLY sweep over "
                f"{len(H_GRID)} step sizes x {len(SWEEP)} strength triples (criterion: mean "
                f"training accuracy over the last 25% of checkpoints). Test labels were not used "
                f"to select anything.")
        print(f"\n=== final {spec.key} at h = {c['h']:g}, s = {tuple(c['s'])} "
              f"(R = {args.final_reps}) ===")
        summ, png, pdf, npz = final_figure(
            datasets[spec.dataset], spec, tuple(c["s"]), args.final_reps, spec.n_iter,
            args.outdir, f"{spec.key}_anchored_tuned", note, h=c["h"])
        finals[spec.key] = dict(h=c["h"], s=c["s"], summary=summ, png=png, pdf=pdf, npz=npz)
        for mk in ("ranch", "nranch"):
            v = summ[mk]
            print(f"  {v['label']:36s} train {v['train_mean']:.4f}+-{v['train_sd']:.4f}  "
                  f"test {v['test_mean']:.4f}+-{v['test_sd']:.4f}  U={v['train_loss_U']:.1f} "
                  f"proj={v['projection_rate']:.3f}")
        p = summ["paired"]
        print(f"  paired (non-rev - rev): train {p['dtrain']:+.4f} +- {p['dtrain_se']:.4f} "
              f"(wins {p['win_rate_train']:.0%}),  test {p['dtest']:+.4f} +- {p['dtest_se']:.4f} "
              f"(wins {p['win_rate_test']:.0%})")

    with open(os.path.join(args.resdir, "hs_search.json"), "w") as fh:
        json.dump(dict(h_grid=H_GRID, s_grid=[list(s) for s in SWEEP],
                       sweep_reps=args.sweep_reps, final_reps=args.final_reps,
                       selection="training accuracy, last 25% of checkpoints; test unused",
                       chosen=chosen, finals=finals), fh, indent=2, default=float)
    print(f"\ntotal {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
