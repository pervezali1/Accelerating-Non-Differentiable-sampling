"""Search the block strength s for the two ANCHORED methods only.

Question asked: is there a block strength s at which
    non-reversible anchored Langevin  (rho = log 2, alpha = 1)
beats
    reversible anchored Langevin      (rho = log 2, alpha = 0)
on the ball and on the smoothed l_p ("smoothed ball") constraint?

Protocol (the honest version of "search until it wins"):
  * The reversible arm does not depend on s at all, so it is run ONCE per
    (dataset, geometry) and reused as the baseline for every s.
  * Every s uses the SAME initial states, the SAME mini-batches and the SAME Gaussian
    increments as the baseline, so each comparison is exactly paired and the paired
    difference has a standard error that is much smaller than the run-to-run SD.
  * s is SELECTED ON TRAINING DATA ONLY -- the mean training accuracy over the last 25% of
    checkpoints. Test labels are never used to choose anything. The test column is then an
    honest held-out read of the configuration that training selected.
  * The full sweep is reported, so the size of the search is visible. s = 0 is included as a
    sanity check: with J = 0 the two arms must be bit-identical (paired difference exactly 0).

Usage:  python strength_search.py [--sweep-reps 30] [--final-reps 100] [--quick]
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

import nral
from nral import (DIM, BATCH_SIZE, H_REPORTED, RHO_ANCHORED, CHECKPOINT_EVERY, METHOD_LABEL,
                  ExperimentSpec, Dataset, build_magic_dataset, build_titanic_dataset,
                  make_geometry, make_initial_states, stream_seed_for, run_sampler,
                  accuracy_curve, mean_sd, potential_U, caption_for, _wrap)

ANCHORED = (
    ("ranch",  "Reversible anchored Langevin",     RHO_ANCHORED, 0),
    ("nranch", "Non-reversible anchored Langevin", RHO_ANCHORED, 1),
)
STYLE = {
    "ranch":  dict(color="#1f5fbf", linestyle="-", linewidth=1.9),
    "nranch": dict(color="#1a9850", linestyle="-", linewidth=1.9),
}

# Isotropic grid plus the two anisotropic triples used in the original experiments.
SWEEP = [(0.0, 0.0, 0.0), (0.02, 0.02, 0.02), (0.05, 0.05, 0.05), (0.1, 0.1, 0.1),
         (0.25, 0.25, 0.25), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0),
         (3.0, 3.0, 3.0), (5.0, 5.0, 5.0), (7.0, 7.0, 7.0), (10.0, 10.0, 10.0),
         (2.0, 7.0, 2.0), (1.0, 0.25, 1.0)]

SELECT_TAIL = 0.25          # training accuracy is averaged over the last 25% of checkpoints


def _tail_mean(acc: np.ndarray) -> np.ndarray:
    """acc is (n_ck, R) -> (R,) mean over the last SELECT_TAIL of the checkpoints."""
    k = max(1, int(round(SELECT_TAIL * acc.shape[0])))
    return acc[-k:].mean(axis=0)


def run_arm(ds: Dataset, spec: ExperimentSpec, alpha: int, scales, beta0, n_iter: int,
            h: float = H_REPORTED) -> dict:
    geom = make_geometry(spec)
    res = run_sampler(ds.X_train, ds.y_train, geom, rho=RHO_ANCHORED, alpha=alpha,
                      block_scales=scales, h=h, n_iter=n_iter, beta0=beta0,
                      stream_seed=stream_seed_for(spec.dataset, "main"),
                      m=BATCH_SIZE, checkpoint_every=CHECKPOINT_EVERY)
    betas = res["betas"]
    return dict(
        checkpoints=res["checkpoints"], betas=betas,
        acc_train=accuracy_curve(ds.X_train, ds.y_train, betas),
        acc_test=accuracy_curve(ds.X_test, ds.y_test, betas),
        projection_rate=res["projection_rate"], n_nonfinite=res["n_nonfinite"],
        train_loss=np.array([potential_U(betas[i], ds.X_train, ds.y_train)
                             for i in range(betas.shape[0])]),
        runtime_sec=res["runtime_sec"],
    )


def sweep_one(ds: Dataset, spec: ExperimentSpec, n_reps: int, n_iter: int,
              h: float = H_REPORTED) -> pd.DataFrame:
    beta0 = make_initial_states(spec.dataset, n_reps)
    base = run_arm(ds, spec, 0, (1.0, 1.0, 1.0), beta0, n_iter, h=h)  # scales unused when alpha=0
    b_tr_sel = _tail_mean(base["acc_train"])
    rows = [dict(experiment=spec.key, dataset=spec.dataset, geometry=spec.geometry_kind, h=h,
                 s="reversible (no J)", s_tuple=None, method="reversible",
                 sel_train_acc=float(b_tr_sel.mean()),
                 final_train_acc=float(base["acc_train"][-1].mean()),
                 final_train_sd=float(base["acc_train"][-1].std(ddof=1)),
                 final_test_acc=float(base["acc_test"][-1].mean()),
                 final_test_sd=float(base["acc_test"][-1].std(ddof=1)),
                 paired_dtrain=0.0, paired_dtrain_se=0.0,
                 paired_dtest=0.0, paired_dtest_se=0.0,
                 train_loss_U=float(base["train_loss"][-1].mean()),
                 projection_rate=base["projection_rate"], n_nonfinite=base["n_nonfinite"])]
    for scales in SWEEP:
        arm = run_arm(ds, spec, 1, scales, beta0, n_iter, h=h)
        sel = _tail_mean(arm["acc_train"])
        d_tr = sel - b_tr_sel                                      # paired, TRAIN only
        d_te = arm["acc_test"][-1] - base["acc_test"][-1]
        rows.append(dict(
            experiment=spec.key, dataset=spec.dataset, geometry=spec.geometry_kind, h=h,
            s=f"({','.join(f'{v:g}' for v in scales)})", s_tuple=list(scales),
            method="non-reversible",
            sel_train_acc=float(sel.mean()),
            final_train_acc=float(arm["acc_train"][-1].mean()),
            final_train_sd=float(arm["acc_train"][-1].std(ddof=1)),
            final_test_acc=float(arm["acc_test"][-1].mean()),
            final_test_sd=float(arm["acc_test"][-1].std(ddof=1)),
            paired_dtrain=float(d_tr.mean()),
            paired_dtrain_se=float(d_tr.std(ddof=1) / np.sqrt(len(d_tr))),
            paired_dtest=float(d_te.mean()),
            paired_dtest_se=float(d_te.std(ddof=1) / np.sqrt(len(d_te))),
            train_loss_U=float(arm["train_loss"][-1].mean()),
            projection_rate=arm["projection_rate"], n_nonfinite=arm["n_nonfinite"]))
    return pd.DataFrame(rows)


def final_figure(ds: Dataset, spec: ExperimentSpec, scales, n_reps: int, n_iter: int,
                 outdir: str, tag: str, selection_note: str, h: float = H_REPORTED):
    beta0 = make_initial_states(spec.dataset, n_reps)
    arms = {"ranch": run_arm(ds, spec, 0, scales, beta0, n_iter, h=h),
            "nranch": run_arm(ds, spec, 1, scales, beta0, n_iter, h=h)}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, split in zip(axes, ("train", "test")):
        for mk, _lbl, _rho, _a in ANCHORED:
            a = arms[mk][f"acc_{split}"]
            x = np.asarray(arms[mk]["checkpoints"], dtype=float)
            mu, sd = mean_sd(a, axis=1)
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
    axes[0].legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    meta = dict(spec=dataclasses.asdict(spec), dataset=spec.dataset, h=h, m=BATCH_SIZE,
                n_replicates=n_reps, n_train=ds.n_train, n_test=ds.n_test,
                rho_anchored=RHO_ANCHORED, block_scales=list(map(float, scales)))
    cap = caption_for(meta) + "  " + selection_note
    fig.suptitle(tag, fontsize=11, y=0.995)
    fig.text(0.5, -0.09, _wrap(cap), ha="center", va="top", fontsize=7.2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = os.path.join(outdir, f"{tag}.png"); pdf = os.path.join(outdir, f"{tag}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    summary = {}
    for mk, lbl, _r, _a in ANCHORED:
        tr, te = arms[mk]["acc_train"][-1], arms[mk]["acc_test"][-1]
        summary[mk] = dict(label=lbl, train_mean=float(tr.mean()), train_sd=float(tr.std(ddof=1)),
                           test_mean=float(te.mean()), test_sd=float(te.std(ddof=1)),
                           projection_rate=arms[mk]["projection_rate"],
                           n_nonfinite=arms[mk]["n_nonfinite"],
                           train_loss_U=float(arms[mk]["train_loss"][-1].mean()))
    d_tr = arms["nranch"]["acc_train"][-1] - arms["ranch"]["acc_train"][-1]
    d_te = arms["nranch"]["acc_test"][-1] - arms["ranch"]["acc_test"][-1]
    summary["paired"] = dict(
        dtrain=float(d_tr.mean()), dtrain_se=float(d_tr.std(ddof=1) / np.sqrt(len(d_tr))),
        dtest=float(d_te.mean()), dtest_se=float(d_te.std(ddof=1) / np.sqrt(len(d_te))),
        win_rate_train=float((d_tr > 0).mean()), win_rate_test=float((d_te > 0).mean()))
    npz = os.path.join(outdir, f"{tag}.npz")
    np.savez_compressed(
        npz, checkpoints=arms["ranch"]["checkpoints"],
        ranch_acc_train=arms["ranch"]["acc_train"], ranch_acc_test=arms["ranch"]["acc_test"],
        nranch_acc_train=arms["nranch"]["acc_train"], nranch_acc_test=arms["nranch"]["acc_test"],
        ranch_betas=arms["ranch"]["betas"], nranch_betas=arms["nranch"]["betas"],
        block_scales=np.asarray(scales, dtype=float))
    return summary, png, pdf, npz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-reps", type=int, default=30)
    ap.add_argument("--final-reps", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default="figures/strength")
    ap.add_argument("--resdir", default="results/strength")
    args = ap.parse_args()
    if args.quick:
        args.sweep_reps, args.final_reps = 4, 6
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.resdir, exist_ok=True)

    datasets = {"magic": build_magic_dataset(), "titanic": build_titanic_dataset()}
    specs = [
        ExperimentSpec("magic_ball",   "magic",   "ball", 1000, (5., 5., 5.), "magic_ball"),
        ExperimentSpec("magic_lp",     "magic",   "lp",   1000, (5., 5., 5.), "magic_lp",
                       p=2.4, eps=0.20, Lam=4.0),
        ExperimentSpec("titanic_ball", "titanic", "ball", 1500, (2., 7., 2.), "titanic_ball"),
        ExperimentSpec("titanic_lp",   "titanic", "lp",   2000, (2., 7., 2.), "titanic_lp",
                       p=2.4, eps=0.18, Lam=4.0),
    ]
    if args.quick:
        specs = [dataclasses.replace(s, n_iter=150) for s in specs]

    sweeps, chosen, finals = {}, {}, {}
    t0 = time.perf_counter()
    for spec in specs:
        ds = datasets[spec.dataset]
        print(f"\n=== sweep {spec.key} (R = {args.sweep_reps}, {spec.n_iter} iters, "
              f"h = {H_REPORTED:g}) ===")
        df = sweep_one(ds, spec, args.sweep_reps, spec.n_iter)
        sweeps[spec.key] = df
        print(df[["s", "sel_train_acc", "final_train_acc", "final_test_acc", "paired_dtrain",
                  "paired_dtrain_se", "train_loss_U", "projection_rate"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        nr = df[df.method == "non-reversible"]
        best = nr.loc[nr.sel_train_acc.idxmax()]
        chosen[spec.key] = dict(s=list(best.s_tuple), s_label=best.s,
                                sel_train_acc=float(best.sel_train_acc),
                                paired_dtrain=float(best.paired_dtrain),
                                paired_dtrain_se=float(best.paired_dtrain_se))
        print(f"  -> selected on TRAINING accuracy only: s = {best.s}  "
              f"(paired train gain {best.paired_dtrain:+.4f} +- {best.paired_dtrain_se:.4f})")
        df.to_csv(os.path.join(args.resdir, f"sweep_{spec.key}.csv"), index=False)

    for spec in specs:
        s = tuple(chosen[spec.key]["s"])
        note = (f"Block strengths selected by a training-only sweep over {len(SWEEP)} candidates "
                f"(criterion: mean TRAINING accuracy over the last {int(SELECT_TAIL*100)}% of "
                f"checkpoints); test labels were not used to choose s.")
        print(f"\n=== final {spec.key} at s = {s} (R = {args.final_reps}) ===")
        summ, png, pdf, npz = final_figure(datasets[spec.dataset], spec, s, args.final_reps,
                                           spec.n_iter, args.outdir,
                                           f"{spec.key}_anchored_accuracy", note)
        finals[spec.key] = dict(s=list(s), summary=summ, png=png, pdf=pdf, npz=npz)
        for mk in ("ranch", "nranch"):
            v = summ[mk]
            print(f"  {v['label']:36s} train {v['train_mean']:.4f}+-{v['train_sd']:.4f}  "
                  f"test {v['test_mean']:.4f}+-{v['test_sd']:.4f}  "
                  f"U={v['train_loss_U']:.1f} proj={v['projection_rate']:.3f}")
        p = summ["paired"]
        print(f"  paired (non-rev - rev): train {p['dtrain']:+.4f} +- {p['dtrain_se']:.4f} "
              f"(wins {p['win_rate_train']:.0%}),  test {p['dtest']:+.4f} +- {p['dtest_se']:.4f} "
              f"(wins {p['win_rate_test']:.0%})")

    # The user asked specifically about s = 5, so it is reported explicitly whether or not the
    # training-only selection picked it.
    s5 = (5.0, 5.0, 5.0)
    s5_out = {}
    for spec in specs:
        if tuple(chosen[spec.key]["s"]) == s5:
            s5_out[spec.key] = finals[spec.key]
            continue
        print(f"\n=== requested s = (5,5,5) for {spec.key} (R = {args.final_reps}) ===")
        summ, png, pdf, npz = final_figure(
            datasets[spec.dataset], spec, s5, args.final_reps, spec.n_iter, args.outdir,
            f"{spec.key}_anchored_accuracy_s5",
            "Block strengths FIXED at s = (5,5,5) as requested -- not selected by the sweep.")
        s5_out[spec.key] = dict(s=list(s5), summary=summ, png=png, pdf=pdf, npz=npz)
        for mk in ("ranch", "nranch"):
            v = summ[mk]
            print(f"  {v['label']:36s} train {v['train_mean']:.4f}+-{v['train_sd']:.4f}  "
                  f"test {v['test_mean']:.4f}+-{v['test_sd']:.4f}  "
                  f"U={v['train_loss_U']:.1f} proj={v['projection_rate']:.3f}")
        p = summ["paired"]
        print(f"  paired (non-rev - rev): train {p['dtrain']:+.4f} +- {p['dtrain_se']:.4f} "
              f"(wins {p['win_rate_train']:.0%}),  test {p['dtest']:+.4f} +- {p['dtest_se']:.4f} "
              f"(wins {p['win_rate_test']:.0%})")

    out = dict(protocol=dict(
        selection="mean TRAINING accuracy over the last 25% of checkpoints; test labels unused",
        sweep_candidates=[list(s) for s in SWEEP], sweep_reps=args.sweep_reps,
        final_reps=args.final_reps, h=H_REPORTED, m=BATCH_SIZE, rho=RHO_ANCHORED,
        paired="shared beta0, mini-batches and Gaussian increments across the two arms"),
        chosen=chosen, finals=finals, s5=s5_out)
    with open(os.path.join(args.resdir, "strength_search.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\ntotal {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
