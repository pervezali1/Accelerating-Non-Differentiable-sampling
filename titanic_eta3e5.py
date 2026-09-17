"""Titanic at the specified eta = 3e-5, exact gradient + LASSO anchor.

eta is FIXED by the request, not selected, so there is no step-size selection bias here. The
block strength s is still swept, and the whole sweep is reported (4 strengths plus the s = 0
control, which must reproduce the reversible arm bit for bit) so the size of the search is
visible: with 4 comparisons, treat |t| > ~2.5 rather than 2 as the bar.

All numbers come from R = 150 replicates on the confirmation seed (4100).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from exact_anchored import (Exp10, CONFIRM_SEED, make_potential, run_cell, paired, figure10,
                            build_titanic_dataset)

ETA = 3e-5
S_GRID = [0.0, 0.25, 1.0, 2.0, 5.0]
R = 150
EXPS = (Exp10("titanic_ball", "titanic", "ball", 1500),
        Exp10("titanic_lp",   "titanic", "lp",   2000, eps=0.18))
OUT, RES = "figures/exact", "results/exact"


def main() -> int:
    os.makedirs(OUT, exist_ok=True); os.makedirs(RES, exist_ok=True)
    ds = build_titanic_dataset()
    pot = make_potential(ds)
    print(f"titanic: n_train = {ds.n_train}, lambda_lasso = {pot.lam:.4g}, "
          f"delta = {pot.delta:.4g}, sigma = {pot.sigma:g}, a in [{pot.a_lower_bound:.4f}, 1]")
    print(f"eta = {ETA:g} (specified, not selected), R = {R}, seed {CONFIRM_SEED}\n")

    rows, keep = [], {}
    t0 = time.perf_counter()
    for e in EXPS:
        for s in S_GRID:
            arms = run_cell(ds, pot, e, ETA, (s, s, s), R, CONFIRM_SEED)
            st = paired(arms)
            rows.append(dict(experiment=e.key, eta=ETA, s=s,
                             rev_train=float(arms["rev"]["acc_train"][-1].mean()),
                             rev_train_sd=float(arms["rev"]["acc_train"][-1].std(ddof=1)),
                             nrev_train=float(arms["nrev"]["acc_train"][-1].mean()),
                             nrev_train_sd=float(arms["nrev"]["acc_train"][-1].std(ddof=1)),
                             rev_test=float(arms["rev"]["acc_test"][-1].mean()),
                             rev_test_sd=float(arms["rev"]["acc_test"][-1].std(ddof=1)),
                             nrev_test=float(arms["nrev"]["acc_test"][-1].mean()),
                             nrev_test_sd=float(arms["nrev"]["acc_test"][-1].std(ddof=1)),
                             rev_U=float(arms["rev"]["U"].mean()),
                             nrev_U=float(arms["nrev"]["U"].mean()),
                             proj_rev=arms["rev"]["projection_rate"],
                             proj_nrev=arms["nrev"]["projection_rate"], **st))
            if s == 5.0:
                arms["_pot"] = pot
                keep[e.key] = (e, arms, st)
        sub = pd.DataFrame([r for r in rows if r["experiment"] == e.key])
        print(f"=== {e.key}, eta = {ETA:g}, R = {R} (all strengths reported) ===")
        print(sub[["s", "rev_train", "nrev_train", "d_train", "se_train", "t_train",
                   "rev_test", "nrev_test", "d_test", "se_test", "t_test",
                   "rev_U", "nrev_U", "proj_nrev"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        print()
    pd.DataFrame(rows).to_csv(os.path.join(RES, "titanic_eta3e-5.csv"), index=False)

    for key, (e, arms, st) in keep.items():
        verdict = ("non-reversible WINS" if st["t_train"] > 2 and st["d_train"] > 0 else
                   "non-reversible LOSES" if st["t_train"] < -2 else "no significant difference")
        note = (f"eta = {ETA:g} was SPECIFIED, not selected. s = 5 shown here; the full "
                f"{len(S_GRID)-1}-strength sweep at this eta is in results/exact/"
                f"titanic_eta3e-5.csv. Verdict on the paired training difference: {verdict}.")
        png, pdf = figure10(e, ds, arms, ETA, (5.0, 5.0, 5.0), R, OUT, note)
        # distinct file names so the eta = 1e-5 figures are not overwritten
        for src, dst in ((png, png.replace("_exact_anchored", "_exact_anchored_eta3e-5")),
                         (pdf, pdf.replace("_exact_anchored", "_exact_anchored_eta3e-5"))):
            os.replace(src, dst)
        print(f"{key}: s = 5 -> {verdict};  figure {png.replace('_exact_anchored', '_exact_anchored_eta3e-5')}")
    print(f"\ntotal {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
