"""Confirmation runs for the requested block strength s = (5, 5, 5).

Two anchored methods only (rho = log 2):
    reversible      alpha = 0      (does not depend on s at all)
    non-reversible  alpha = 1

Findings that motivate this script (see results/strength/hs_sweep_*.csv):
  * at the originally specified h = 1e-4, s = 5 LOSES everywhere -- badly on MAGIC, where one
    non-reversible drift step is larger than the constraint set;
  * on Titanic it WINS at h = 1e-5 on both geometries (paired training gain ~3.5 standard
    errors), because at that step the chains are still inside their transient over the plotted
    1500 / 2000 iterations and the rotation genuinely accelerates it;
  * on MAGIC the gain is still negative at h = 1e-5 but rises towards 0 as h shrinks, so this
    script scans two smaller steps before concluding.

Protocol. h is chosen per experiment by the PAIRED TRAINING gain only -- test labels select
nothing. Because that choice is a maximum over a grid, and the maximum of noisy estimates is
biased upwards, every reported number comes from a CONFIRMATION run on an INDEPENDENT set of
replicates (sampler seed 4100 instead of the 3000 used for the search). The confirmation is what
the figures and the headline numbers show.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nral import (BATCH_SIZE, CHECKPOINT_EVERY, DIM, METHOD_LABEL, RHO_ANCHORED, Dataset,
                  ExperimentSpec, accuracy_curve, build_magic_dataset, build_titanic_dataset,
                  make_geometry, make_initial_states, mean_sd, potential_U, run_sampler,
                  sample_unit_ball, stream_seed_for, _wrap)

S5 = (5.0, 5.0, 5.0)
SEARCH_SEED = 3000            # the seed the (h, s) search used
CONFIRM_SEED = 4100           # independent replicates for every reported number
STYLE = {"ranch":  dict(color="#1f5fbf", linestyle="-", linewidth=2.0),
         "nranch": dict(color="#1a9850", linestyle="-", linewidth=2.0)}

SPECS = [
    ExperimentSpec("magic_ball",   "magic",   "ball", 1000, S5, "magic_ball"),
    ExperimentSpec("magic_lp",     "magic",   "lp",   1000, S5, "magic_lp",
                   p=2.4, eps=0.20, Lam=4.0),
    ExperimentSpec("titanic_ball", "titanic", "ball", 1500, S5, "titanic_ball"),
    ExperimentSpec("titanic_lp",   "titanic", "lp",   2000, S5, "titanic_lp",
                   p=2.4, eps=0.18, Lam=4.0),
]
# Step sizes scanned per dataset for s = 5 (MAGIC gets the extra small steps).
H_SCAN = {"magic":   [1e-5, 3e-6, 1e-6, 3e-7, 1e-7],
          "titanic": [3e-5, 1e-5, 3e-6]}


def run_pair(ds: Dataset, spec: ExperimentSpec, h: float, n_reps: int, seed: int) -> dict:
    """Both anchored arms at s = (5,5,5), fully paired (same beta0, batches, noise)."""
    geom = make_geometry(spec)
    beta0 = make_initial_states(spec.dataset, n_reps, sampler_seed=seed)
    out = {}
    for mk, alpha in (("ranch", 0), ("nranch", 1)):
        res = run_sampler(ds.X_train, ds.y_train, geom, rho=RHO_ANCHORED, alpha=alpha,
                          block_scales=S5, h=h, n_iter=spec.n_iter, beta0=beta0,
                          stream_seed=stream_seed_for(spec.dataset, "main", sampler_seed=seed),
                          m=BATCH_SIZE, checkpoint_every=CHECKPOINT_EVERY)
        b = res["betas"]
        out[mk] = dict(checkpoints=res["checkpoints"], betas=b,
                       acc_train=accuracy_curve(ds.X_train, ds.y_train, b),
                       acc_test=accuracy_curve(ds.X_test, ds.y_test, b),
                       projection_rate=res["projection_rate"], n_nonfinite=res["n_nonfinite"],
                       train_loss=np.array([potential_U(b[i], ds.X_train, ds.y_train)
                                            for i in range(b.shape[0])]))
    return out


def paired_stats(arms: dict) -> dict:
    d_tr = arms["nranch"]["acc_train"][-1] - arms["ranch"]["acc_train"][-1]
    d_te = arms["nranch"]["acc_test"][-1] - arms["ranch"]["acc_test"][-1]
    n = len(d_tr)
    return dict(dtrain=float(d_tr.mean()), dtrain_se=float(d_tr.std(ddof=1) / np.sqrt(n)),
                dtest=float(d_te.mean()), dtest_se=float(d_te.std(ddof=1) / np.sqrt(n)),
                t_train=float(d_tr.mean() / (d_tr.std(ddof=1) / np.sqrt(n) + 1e-300)),
                t_test=float(d_te.mean() / (d_te.std(ddof=1) / np.sqrt(n) + 1e-300)),
                win_train=float((d_tr > 0).mean()), win_test=float((d_te > 0).mean()))


def make_figure(spec: ExperimentSpec, ds: Dataset, arms: dict, h: float, n_reps: int,
                outdir: str, note: str):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, split in zip(axes, ("train", "test")):
        for mk in ("ranch", "nranch"):
            x = np.asarray(arms[mk]["checkpoints"], dtype=float)
            mu, sd = mean_sd(arms[mk][f"acc_{split}"], axis=1)
            st = STYLE[mk]
            ax.fill_between(x, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                            color=st["color"], alpha=0.16, linewidth=0)
            ax.plot(x, mu, color=st["color"], linestyle=st["linestyle"],
                    linewidth=st["linewidth"], label=METHOD_LABEL[mk])
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        gname = "ball" if spec.geometry_kind == "ball" else f"smoothed $\\ell_p$ ($p={spec.p:g}$)"
        ax.set_title(f"{spec.dataset.upper()}, {gname} -- "
                     f"{'training' if split == 'train' else 'test'} accuracy", fontsize=10)

        # The required axis is the common [0, 1] range, on which a one-point difference is
        # invisible; the inset is a zoom of the SAME curves over the last 60% of the run, so the
        # separation can be read off without rescaling the primary axis.
        i0 = int(0.4 * len(x))
        ins = ax.inset_axes([0.50, 0.12, 0.46, 0.40])
        lo_v, hi_v = [], []
        for mk in ("ranch", "nranch"):
            mu, sd = mean_sd(arms[mk][f"acc_{split}"], axis=1)
            se = sd / np.sqrt(arms[mk][f"acc_{split}"].shape[1])
            st = STYLE[mk]
            ins.fill_between(x[i0:], mu[i0:] - se[i0:], mu[i0:] + se[i0:],
                             color=st["color"], alpha=0.22, linewidth=0)
            ins.plot(x[i0:], mu[i0:], color=st["color"], linewidth=1.5)
            lo_v.append((mu[i0:] - se[i0:]).min()); hi_v.append((mu[i0:] + se[i0:]).max())
        pad = 0.12 * (max(hi_v) - min(lo_v) + 1e-9)
        ins.set_ylim(min(lo_v) - pad, max(hi_v) + pad)
        ins.set_xlim(x[i0], x[-1])
        ins.tick_params(labelsize=6.5)
        ins.grid(True, alpha=0.25, linewidth=0.5)
        ins.set_title("zoom (mean $\\pm$ 1 s.e.)", fontsize=6.5, pad=2)
    axes[0].legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    geom_txt = ("ball ||beta||^2 <= 2 (radius sqrt(2))" if spec.geometry_kind == "ball" else
                f"smoothed l_p  g(beta) = sum_i (beta_i^2+eps^2)^(p/2) <= Lambda, p = {spec.p:g}, "
                f"eps = {spec.eps:g}, Lambda = {spec.Lam:g}")
    cap = (f"{spec.dataset.upper()} -- {geom_txt}.  d = {DIM}, no intercept.  "
           f"Block strengths s = (5, 5, 5).  Step size h = {h:g} (actually used).  "
           f"Mini-batch m = {BATCH_SIZE}.  R = {n_reps} independent replicates on one fixed "
           f"stratified 80/20 split (n_train = {ds.n_train}, n_test = {ds.n_test}).  "
           f"Anchor rho = log 2 = {RHO_ANCHORED:.6f} for both arms.  Bands: mean +- 1 SD "
           f"(ddof = 1) across replicates -- repeat-run variability at a fixed split, not a "
           f"confidence or credible interval.  {note}")
    fig.suptitle(f"{spec.key}: anchored Langevin, s = 5", fontsize=11, y=0.995)
    fig.text(0.5, -0.09, _wrap(cap), ha="center", va="top", fontsize=7.2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = os.path.join(outdir, f"{spec.key}_s5_anchored.png")
    pdf = os.path.join(outdir, f"{spec.key}_s5_anchored.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(
        os.path.join(outdir.replace("figures", "results"), f"{spec.key}_s5_anchored.npz"),
        checkpoints=arms["ranch"]["checkpoints"],
        ranch_acc_train=arms["ranch"]["acc_train"], ranch_acc_test=arms["ranch"]["acc_test"],
        nranch_acc_train=arms["nranch"]["acc_train"], nranch_acc_test=arms["nranch"]["acc_test"],
        ranch_betas=arms["ranch"]["betas"], nranch_betas=arms["nranch"]["betas"],
        h=h, block_scales=np.asarray(S5))
    return png, pdf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-reps", type=int, default=40)
    ap.add_argument("--final-reps", type=int, default=100)
    ap.add_argument("--outdir", default="figures/strength")
    ap.add_argument("--resdir", default="results/strength")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True); os.makedirs(args.resdir, exist_ok=True)
    datasets = {"magic": build_magic_dataset(), "titanic": build_titanic_dataset()}

    t0 = time.perf_counter()
    scan_rows, chosen = [], {}
    for spec in SPECS:
        ds = datasets[spec.dataset]
        print(f"\n=== step-size scan at s = (5,5,5): {spec.key} "
              f"(R = {args.scan_reps}, search seed {SEARCH_SEED}) ===")
        rows = []
        for h in H_SCAN[spec.dataset]:
            arms = run_pair(ds, spec, h, args.scan_reps, SEARCH_SEED)
            st = paired_stats(arms)
            rows.append(dict(experiment=spec.key, h=h,
                             rev_train=float(arms["ranch"]["acc_train"][-1].mean()),
                             nonrev_train=float(arms["nranch"]["acc_train"][-1].mean()),
                             paired_dtrain=st["dtrain"], paired_dtrain_se=st["dtrain_se"],
                             t_train=st["t_train"], win_train=st["win_train"],
                             rev_U=float(arms["ranch"]["train_loss"][-1].mean()),
                             nonrev_U=float(arms["nranch"]["train_loss"][-1].mean()),
                             proj_nonrev=arms["nranch"]["projection_rate"]))
        df = pd.DataFrame(rows)
        scan_rows.append(df)
        print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        best = df.loc[df.paired_dtrain.idxmax()]
        chosen[spec.key] = float(best.h)
        print(f"  -> h chosen on the PAIRED TRAINING gain only: h = {best.h:g} "
              f"({best.paired_dtrain:+.4f} +- {best.paired_dtrain_se:.4f}, t = {best.t_train:.2f})")
    pd.concat(scan_rows, ignore_index=True).to_csv(
        os.path.join(args.resdir, "s5_step_scan.csv"), index=False)

    print("\n" + "=" * 92)
    print(f"CONFIRMATION on independent replicates (sampler seed {CONFIRM_SEED}, "
          f"R = {args.final_reps})")
    print("=" * 92)
    confirm = {}
    for spec in SPECS:
        h = chosen[spec.key]
        ds = datasets[spec.dataset]
        arms = run_pair(ds, spec, h, args.final_reps, CONFIRM_SEED)
        st = paired_stats(arms)
        verdict = ("non-reversible WINS" if st["t_train"] > 2 and st["dtrain"] > 0 else
                   "non-reversible LOSES" if st["t_train"] < -2 else "no significant difference")
        note = (f"h selected per experiment by the paired TRAINING gain over {len(H_SCAN[spec.dataset])} "
                f"candidates; these curves are a CONFIRMATION run on independent replicates "
                f"(seed {CONFIRM_SEED}) not used for that selection. Verdict on the paired "
                f"training difference: {verdict}.")
        png, pdf = make_figure(spec, ds, arms, h, args.final_reps, args.outdir, note)
        rec = dict(h=h, s=list(S5), verdict=verdict, paired=st,
                   ranch=dict(train=float(arms["ranch"]["acc_train"][-1].mean()),
                              train_sd=float(arms["ranch"]["acc_train"][-1].std(ddof=1)),
                              test=float(arms["ranch"]["acc_test"][-1].mean()),
                              test_sd=float(arms["ranch"]["acc_test"][-1].std(ddof=1)),
                              U=float(arms["ranch"]["train_loss"][-1].mean()),
                              proj=arms["ranch"]["projection_rate"]),
                   nranch=dict(train=float(arms["nranch"]["acc_train"][-1].mean()),
                               train_sd=float(arms["nranch"]["acc_train"][-1].std(ddof=1)),
                               test=float(arms["nranch"]["acc_test"][-1].mean()),
                               test_sd=float(arms["nranch"]["acc_test"][-1].std(ddof=1)),
                               U=float(arms["nranch"]["train_loss"][-1].mean()),
                               proj=arms["nranch"]["projection_rate"]),
                   png=png, pdf=pdf)
        confirm[spec.key] = rec
        print(f"\n[{spec.key}]  h = {h:g}, s = (5,5,5), R = {args.final_reps}")
        print(f"  Reversible      train {rec['ranch']['train']:.4f}+-{rec['ranch']['train_sd']:.4f}"
              f"   test {rec['ranch']['test']:.4f}+-{rec['ranch']['test_sd']:.4f}"
              f"   U={rec['ranch']['U']:.1f}  proj={rec['ranch']['proj']:.3f}")
        print(f"  Non-reversible  train {rec['nranch']['train']:.4f}+-{rec['nranch']['train_sd']:.4f}"
              f"   test {rec['nranch']['test']:.4f}+-{rec['nranch']['test_sd']:.4f}"
              f"   U={rec['nranch']['U']:.1f}  proj={rec['nranch']['proj']:.3f}")
        print(f"  paired non-rev - rev: train {st['dtrain']:+.4f} +- {st['dtrain_se']:.4f} "
              f"(t = {st['t_train']:+.2f}, wins {st['win_train']:.0%})   "
              f"test {st['dtest']:+.4f} +- {st['dtest_se']:.4f} "
              f"(t = {st['t_test']:+.2f}, wins {st['win_test']:.0%})")
        print(f"  VERDICT: {verdict}")

    with open(os.path.join(args.resdir, "s5_confirmation.json"), "w") as fh:
        json.dump(dict(s=list(S5), h_scan=H_SCAN, chosen_h=chosen,
                       search_seed=SEARCH_SEED, confirm_seed=CONFIRM_SEED,
                       scan_reps=args.scan_reps, final_reps=args.final_reps,
                       selection="paired TRAINING gain only; test labels unused",
                       confirmation=confirm), fh, indent=2, default=float)
    print(f"\ntotal {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
