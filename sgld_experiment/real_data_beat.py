"""Accuracy figures for reversible vs non-reversible anchored Langevin on REAL data.

    python real_data_beat.py '<json config for real_hunt.py>' [--quick] [--tag NAME]

Runs both samplers with shared randomness (R replicates), draws training/test
accuracy against iteration with a fixed, un-windowed y-axis, prints the paired
differences and confirms the configuration on four held-out sampler seeds.
Outputs go to results_real/<tag>.{png,pdf,json}.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace

import anchored_lasso as lasso
import real_hunt

OUTPUT_DIR = "results_real"
STYLE = {"Reversible anchored Langevin":     {"color": "#0173B2", "ls": "--", "lw": 2.0},
         "Non-reversible anchored Langevin": {"color": "#CC3311", "ls": "-",  "lw": 2.6}}


def figure(runs, cfg, info, cfgd, tag, y_limits):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    for ax, which, title in ((axes[0], "train_accuracy", f"Training accuracy  (n = {info['n_train']})"),
                             (axes[1], "test_accuracy",  f"Test accuracy  (n = {info['n_test']})")):
        for name, run in runs.items():
            st = STYLE[name]; mean, sd = run.mean_std(which)
            ax.fill_between(run.checkpoints, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=st["color"], alpha=0.15, lw=0)
            ax.plot(run.checkpoints, mean, color=st["color"], ls=st["ls"], lw=st["lw"], label=name)
        ax.set_ylim(*y_limits); ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy"); ax.set_title(title, fontsize=11); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="lower right")
    pretty = {"titanic": "Titanic", "magic": "MAGIC gamma telescope"}[info["dataset"]]
    fig.suptitle(f"{pretty} — non-reversible vs reversible anchored Langevin ({cfgd.get('geometry', 'l1')} constraint)", fontsize=13)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    fig.text(0.5, 0.012,
             f"d = {info['d']} (intercept + {len(info['features'])} columns: {', '.join(info['features'])});  n_train = {info['n_train']}, n_test = {info['n_test']};  "
             f"U = f + g, g = {cfg.lambda_lasso:g}*sum|w_j|, anchor delta = {cfg.delta_anchor};  EXACT gradient;  eta = {cfg.eta:.1e};  s = {cfgd.get('s')};  R = {cfg.n_repeats}.\n"
             f"Preprocessing: scaling = '{cfgd.get('scaling', 'standard')}', order = '{cfgd.get('order', 'natural')}', signs aligned = {cfgd.get('align', True)}"
             + (f", block reparametrisation (v_axis, v_fast, v_slow) = ({cfgd.get('v_axis', 1)}, {cfgd.get('v_fast', 64)}, {cfgd.get('v_slow', 2)}), (b, e) = ({cfgd.get('b', 1.5)}, {cfgd.get('e', 0.25)})" if cfgd.get("scaling") == "design" else "")
             + f".  Reference (scikit-learn logistic regression) test accuracy {info['reference_test_accuracy']:.3f}.\n"
             "Lines: across-replicate mean; bands: mean +/- 1 sd of single-iterate accuracy (repeat-run variability).  Shared initialisation and noise per replicate.  "
             f"y-axis fixed to {y_limits}, not windowed.",
             ha="center", fontsize=7.0)
    paths = []
    for ext, kw in ((".png", {"dpi": 300}), (".pdf", {})):
        path = os.path.join(OUTPUT_DIR, f"{tag}{ext}"); fig.savefig(path, bbox_inches="tight", **kw); paths.append(path)
    plt.close(fig)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config"); ap.add_argument("--quick", action="store_true"); ap.add_argument("--tag", default=None)
    ap.add_argument("--ylim", default="0.4,0.9")
    args = ap.parse_args()
    cfgd = json.loads(args.config)
    cfgd.setdefault("R", 20 if args.quick else 100)
    tag = args.tag or cfgd.get("tag") or f"{cfgd.get('dataset', 'titanic')}_{cfgd.get('scaling', 'standard')}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    y_limits = tuple(float(v) for v in args.ylim.split(","))

    cfg, ds, tg, geom, info = real_hunt.build(cfgd)
    print(f"=== {info['dataset']}: d {info['d']}, n_train {info['n_train']}, reference accuracy {info['reference_test_accuracy']:.4f}, "
          f"R {cfg.n_repeats}, {cfg.n_iterations} iterations, eta {cfg.eta:.1e}, s {cfgd.get('s')}")
    runs = lasso.run_both(ds, tg, geom, cfg, verbose=False)
    rev, nr = runs["Reversible anchored Langevin"], runs["Non-reversible anchored Langevin"]
    print(f"   projection rate NR {nr.projection_rate:.4f}, REV {rev.projection_rate:.4f}; non-finite {nr.n_nonfinite + rev.n_nonfinite}")
    trajectory = []
    for it in sorted({k for k in (25, 50, 100, 150, 200, 300, 400, 600, 800, 1500, 2000) if k < cfg.n_iterations} | {cfg.n_iterations}):
        i = int(np.argmin(np.abs(rev.checkpoints - it)))
        m, se, t = lasso.paired_difference(nr.test_accuracy[i], rev.test_accuracy[i])
        trajectory.append({"iteration": int(rev.checkpoints[i]), "reversible": float(rev.test_accuracy[i].mean()),
                           "non_reversible": float(nr.test_accuracy[i].mean()), "paired_diff": float(m), "paired_se": float(se), "t_stat": float(t)})
        print(f"   it {int(rev.checkpoints[i]):4d}: REV {rev.test_accuracy[i].mean():.4f}  NR {nr.test_accuracy[i].mean():.4f}   diff {m:+.4f}  t {t:+6.2f}", flush=True)
    gap = nr.test_accuracy.mean(axis=1) - rev.test_accuracy.mean(axis=1)
    evaluate_at = int(cfgd.get("eval_at", rev.checkpoints[int(np.argmax(gap))]))
    paths = figure(runs, cfg, info, cfgd, tag, y_limits)

    print(f"   held-out confirmation at iteration {evaluate_at}:")
    held, diffs, ses = [], [], []
    for offset in ((101,) if args.quick else (101, 202, 303, 404)):
        r2 = lasso.run_both(ds, tg, geom, cfg, seed_offset=offset, verbose=False)
        a, b = r2["Reversible anchored Langevin"], r2["Non-reversible anchored Langevin"]
        i = int(np.argmin(np.abs(a.checkpoints - evaluate_at)))
        m, se, t = lasso.paired_difference(b.test_accuracy[i], a.test_accuracy[i]); diffs.append(m); ses.append(se)
        held.append({"seed_offset": offset, "iteration": int(a.checkpoints[i]), "reversible": float(a.test_accuracy[i].mean()),
                     "non_reversible": float(b.test_accuracy[i].mean()), "paired_diff": float(m), "paired_se": float(se), "t_stat": float(t)})
        print(f"      offset {offset}: REV {a.test_accuracy[i].mean():.4f}  NR {b.test_accuracy[i].mean():.4f}  diff {m:+.4f}  t {t:+6.2f}", flush=True)
    pooled_se = float(np.sqrt(np.sum(np.square(ses))) / len(ses)); pooled = float(np.mean(diffs))
    confirmed = bool(all(d > 0 for d in diffs) and pooled / pooled_se > 2)
    print(f"      POOLED {pooled:+.4f} (SE {pooled_se:.4f}, t = {pooled / pooled_se:+.2f}); all positive: {all(d > 0 for d in diffs)}  ->  "
          f"{'CONFIRMED' if confirmed else 'NOT confirmed'}")
    summary = {"config": cfgd, "info": info, "evaluate_at": evaluate_at, "trajectory": trajectory, "held_out": held,
               "pooled_diff": pooled, "pooled_se": pooled_se, "pooled_t": pooled / pooled_se, "confirmed": confirmed,
               "projection_rate_nr": nr.projection_rate, "figures": paths,
               "curves": {"iteration": rev.checkpoints.tolist(), "rev_test": rev.test_accuracy.mean(axis=1).tolist(),
                          "nr_test": nr.test_accuracy.mean(axis=1).tolist(), "rev_train": rev.train_accuracy.mean(axis=1).tolist(),
                          "nr_train": nr.train_accuracy.mean(axis=1).tolist()}}
    with open(os.path.join(OUTPUT_DIR, f"{tag}.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    print("wrote", *paths, os.path.join(OUTPUT_DIR, f"{tag}.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
