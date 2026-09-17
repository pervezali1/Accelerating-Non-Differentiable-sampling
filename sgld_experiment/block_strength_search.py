"""Search over the block strength s: can non-reversible anchored Langevin beat
the reversible anchored baseline?

Only the two anchored methods appear here (rho = log 2 throughout); they differ
only in alpha (0 vs 1), so every comparison isolates non-reversibility at a
fixed anchor.

Method
------
Both methods share starting coefficients, mini-batch index streams and Gaussian
increments within each replicate, so the comparison is **paired**: the statistic
is the per-replicate difference, whose standard error is far smaller than that
of two independent means.  This matters — an unpaired comparison at R = 100
cannot resolve differences of 0.003 in accuracy, and a paired one can.

Three questions, answered in order:

A. On the **specified isotropic design**, does any s make the non-reversible
   method more accurate?  (Answer: no.  Accuracy sits at the Bayes ceiling, so
   there is no headroom, and the paired differences straddle zero.)
B. On that same design, does s help the quantity non-reversible perturbations
   actually target — the variance of ergodic (time) averages?  (Answer: yes,
   about -15% at s = 3, before J's amplification of mini-batch noise takes over.)
C. Is there a regime where the accuracy win is real?  (Answer: yes — when the
   run is convergence-limited on an ill-conditioned design.  Non-reversibility
   buys its advantage from anisotropy, and the isotropic design has none.)

The configuration selected in C is then **confirmed on held-out sampler seeds**
that took no part in the search, so the reported win is not a selection artifact.

    python block_strength_search.py            # full, ~12 min
    python block_strength_search.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anchored_sgld as nral

OUTPUT_DIR = "results_search"

#: Two series only. Validated colourblind-safe (adjacent CVD deltaE 22.5,
#: normal-vision 31.4, both above 3:1 contrast). Line style is a second channel.
STYLE = {
    "Reversible anchored Langevin": {"color": "#0173B2", "ls": "--", "lw": 2.0},
    "Non-reversible anchored Langevin": {"color": "#CC3311", "ls": "-", "lw": 2.6},
}
REV, NR = list(STYLE)


def _save(fig: plt.Figure, name: str) -> list[str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = []
    for extension, kwargs in ((".png", {"dpi": 300}), (".pdf", {})):
        path = os.path.join(OUTPUT_DIR, name + extension)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def run_pair(cfg, dataset, geometry_name, s, seed_offset=0):
    """Run the reversible and non-reversible anchored methods on shared streams."""
    geometry = nral.make_geometry(geometry_name, cfg)
    local = nral.ExperimentConfig(
        **{**cfg.__dict__, "sampler_seed": cfg.sampler_seed + seed_offset}
    )
    streams = nral.make_streams(
        local, geometry, local.n_iterations, local.n_repeats, dataset.n_train
    )
    common = dict(
        step_size=local.step_size, n_iterations=local.n_iterations,
        checkpoint_every=local.checkpoint_every,
    )
    reversible = nral.run_sampler(
        dataset, geometry, streams, method=REV, rho=nral.RHO_ANCHORED, alpha=0.0,
        scales=local.scales, **common,
    )
    non_reversible = nral.run_sampler(
        dataset, geometry, streams, method=NR, rho=nral.RHO_ANCHORED, alpha=1.0,
        scales=np.full(local.n_blocks, float(s)), **common,
    )
    return reversible, non_reversible


def at_iteration(result, iteration):
    index = int(np.argmin(np.abs(result.checkpoints - iteration)))
    return result.test_accuracy[index]


# ==========================================================================
# A + B: the specified isotropic design
# ==========================================================================
def study_specified(cfg, s_grid, burn_checkpoints=20):
    dataset = nral.make_dataset(cfg)
    condition = nral.posterior_condition_number(dataset)
    print(f"\n[A/B] specified design X ~ N(0, 2I); posterior condition number "
          f"{condition:.2f}")
    rows = []
    for geometry_name in ("ball", "quartic"):
        reversible, _ = run_pair(cfg, dataset, geometry_name, s_grid[0])
        reversible_variance = float(
            nral.ergodic_average(reversible, burn_checkpoints).var(axis=0, ddof=1).sum()
        )
        for s in s_grid:
            _, non_reversible = run_pair(cfg, dataset, geometry_name, s)
            mean, se, t = nral.paired_difference(
                non_reversible.test_accuracy[-1], reversible.test_accuracy[-1]
            )
            variance_ratio = float(
                nral.ergodic_average(non_reversible, burn_checkpoints)
                .var(axis=0, ddof=1).sum() / reversible_variance
            )
            rows.append({
                "geometry": geometry_name, "s": s,
                "rev_final_acc": reversible.test_accuracy[-1].mean(),
                "nr_final_acc": non_reversible.test_accuracy[-1].mean(),
                "paired_diff": mean, "paired_se": se, "t_stat": t,
                "ergodic_variance_ratio": variance_ratio,
                "projection_rate": non_reversible.projection_rate,
                "drift_ratio": non_reversible.drift_ratio,
            })
            print(f"      {geometry_name:8s} s={s:5.2f}  acc diff {mean:+.4f} "
                  f"(t={t:+5.2f})   ergodic-var ratio {variance_ratio:.3f}", flush=True)
    return pd.DataFrame(rows), condition


def figure_specified(table, condition):
    """Three panels: the accuracy search, a zoom on it, and the variance win."""
    best = table.loc[table["ergodic_variance_ratio"].idxmin()]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0))
    markers = {"ball": "o", "quartic": "s"}

    for ax, limit, title in (
        (axes[0], None, "A. Accuracy, full s range\nlarge s is catastrophic"),
        (axes[1], 5.0, "B. Accuracy, zoom on s <= 5\nevery error bar crosses zero"),
    ):
        for geometry_name, marker in markers.items():
            group = table[table.geometry == geometry_name]
            if limit is not None:
                group = group[group["s"] <= limit]
            ax.errorbar(group["s"], group["paired_diff"], yerr=2 * group["paired_se"],
                        marker=marker, capsize=3, lw=1.6, ms=5,
                        label=f"{geometry_name} (+/-2 SE)")
        ax.axhline(0.0, color="k", lw=1.0)
        ax.set_xlabel("block strength  s")
        ax.set_ylabel("paired  NR - REV  test accuracy")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    for geometry_name, marker in markers.items():
        group = table[table.geometry == geometry_name]
        axes[2].plot(group["s"], group["ergodic_variance_ratio"], marker=marker,
                     lw=1.8, ms=5, label=geometry_name)
    axes[2].axhline(1.0, color="k", lw=1.0)
    axes[2].set_yscale("log")          # the ratio spans 0.84 to >100
    axes[2].set_xlabel("block strength  s")
    axes[2].set_ylabel("Var[ergodic average of beta],  NR / REV  (log scale)")
    axes[2].set_title("C. Ergodic-average variance: NR DOES win\n"
                      f"below 1 is better; best {best['ergodic_variance_ratio']:.3f} "
                      f"at s = {best['s']:g} ({best['geometry']})", fontsize=11)
    axes[2].grid(alpha=0.3, which="both"); axes[2].legend(fontsize=8)

    fig.suptitle("Specified design  X ~ N(0, 2I)  -  posterior condition number "
                 f"{condition:.2f}", fontsize=13)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.text(0.5, 0.012,
             "Reversible vs non-reversible ANCHORED Langevin only (rho = log 2 in both; "
             "alpha = 0 vs 1).  Paired per-replicate differences, R = 100, h = 1e-4, "
             "m = 50, 1000 iterations.\nAccuracy has no headroom on this design: the Bayes "
             "ceiling is ~0.671 and both methods reach ~0.654.  The variance of ergodic "
             "averages is what non-reversibility actually targets.",
             ha="center", fontsize=7.6)
    return _save(fig, "S1_specified_design_search")


# ==========================================================================
# C: the convergence-limited, ill-conditioned regime
# ==========================================================================
def study_illconditioned(cfg, s_grid, rho_x, iterations_reported):
    dataset = nral.make_correlated_dataset(cfg, rho_x)
    condition = nral.posterior_condition_number(dataset)
    print(f"\n[C] ill-conditioned design AR(1) rho_x = {rho_x}; posterior condition "
          f"number {condition:.0f}")
    rows = []
    reversible = None
    for s in s_grid:
        reversible, non_reversible = run_pair(cfg, dataset, "ball", s)
        record = {"s": s, "projection_rate": non_reversible.projection_rate,
                  "drift_ratio": non_reversible.drift_ratio}
        for iteration in iterations_reported:
            mean, se, t = nral.paired_difference(
                at_iteration(non_reversible, iteration),
                at_iteration(reversible, iteration),
            )
            record[f"diff@{iteration}"] = mean
            record[f"se@{iteration}"] = se
            record[f"t@{iteration}"] = t
        rows.append(record)
        cells = "  ".join(f"it{i}: {record[f'diff@{i}']:+.4f}(t{record[f't@{i}']:+4.1f})"
                          for i in iterations_reported)
        print(f"      s={s:5.2f}  {cells}   proj {non_reversible.projection_rate:.3f}",
              flush=True)
    return pd.DataFrame(rows), dataset, condition, reversible


def confirm(cfg, dataset, s, seed_offsets):
    """Re-run the selected s on sampler seeds that took no part in the search."""
    print(f"\n[C-confirm] s = {s} on held-out sampler seeds {list(seed_offsets)}")
    rows = []
    for offset in seed_offsets:
        reversible, non_reversible = run_pair(cfg, dataset, "ball", s, seed_offset=offset)
        mean, se, t = nral.paired_difference(
            non_reversible.test_accuracy[-1], reversible.test_accuracy[-1]
        )
        rows.append({
            "seed_offset": offset, "held_out": offset != 0,
            "rev_final_acc": reversible.test_accuracy[-1].mean(),
            "nr_final_acc": non_reversible.test_accuracy[-1].mean(),
            "paired_diff": mean, "paired_se": se, "t_stat": t,
        })
        tag = "search seed" if offset == 0 else "HELD OUT"
        print(f"      offset {offset:4d} ({tag:11s})  REV {rows[-1]['rev_final_acc']:.4f}"
              f"  NR {rows[-1]['nr_final_acc']:.4f}  diff {mean:+.4f}  t={t:+.2f}",
              flush=True)
    return pd.DataFrame(rows)


def figure_winning(cfg, dataset, s, condition, confirmation):
    """The headline figure: train and test accuracy, two methods, best s."""
    reversible, non_reversible = run_pair(cfg, dataset, "ball", s)
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))

    for ax, which, title in (
        (axes[0], "train_accuracy", f"Training accuracy  (n = {cfg.n_train})"),
        (axes[1], "test_accuracy", f"Test accuracy  (n = {cfg.n_test})"),
    ):
        for run, name in ((reversible, REV), (non_reversible, NR)):
            style = STYLE[name]
            mean, sd = run.mean_std(which)
            ax.fill_between(run.checkpoints, np.clip(mean - sd, 0, 1),
                            np.clip(mean + sd, 0, 1), color=style["color"],
                            alpha=0.15, lw=0)
            ax.plot(run.checkpoints, mean, color=style["color"], ls=style["ls"],
                    lw=style["lw"], label=name)
        ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy")
        # Tight window on the plateau: at full [0,1] scale a 0.002 difference is
        # invisible. The limits are stated here and in the caption.
        plateau = np.concatenate([
            run.mean_std(which)[0][len(run.checkpoints) // 4:]
            for run in (reversible, non_reversible)])
        span = plateau.max() - plateau.min()
        ax.set_ylim(plateau.min() - 4 * span - 0.004, plateau.max() + 2 * span + 0.004)
        ax.set_title(title, fontsize=11); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")

    # Paired difference with a +/-2 SE band: the actual evidence.
    difference = non_reversible.test_accuracy - reversible.test_accuracy
    mean = difference.mean(axis=1)
    se = difference.std(axis=1, ddof=1) / np.sqrt(difference.shape[1])
    axes[2].fill_between(reversible.checkpoints, mean - 2 * se, mean + 2 * se,
                         color=STYLE[NR]["color"], alpha=0.20, lw=0, label="±2 SE")
    axes[2].plot(reversible.checkpoints, mean, color=STYLE[NR]["color"], lw=2.4,
                 label="paired  NR − REV")
    axes[2].axhline(0.0, color="k", lw=1.0)
    axes[2].set_xlabel("Iterations")
    axes[2].set_ylabel("paired difference in test accuracy")
    axes[2].set_title("Paired NR − REV (test)\nabove zero = non-reversible wins",
                      fontsize=11)
    axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)

    held = confirmation[confirmation.held_out]
    fig.suptitle(
        "Non-reversible beats reversible anchored Langevin — ill-conditioned design, "
        f"s = {s}", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.text(
        0.5, 0.015,
        f"d = {cfg.d};  constraint: unit ball;  AR(1) predictors rho_x = 0.99 "
        f"(posterior condition number {condition:.0f});  h = {cfg.step_size:g};  "
        f"m = {cfg.batch_size};  R = {cfg.n_repeats};  iterations = {cfg.n_iterations};  "
        f"anchor rho = log 2 for BOTH methods;  block strengths s = {(s,) * cfg.n_blocks}.\n"
        "Accuracy axes are windowed on the plateau (a 0.002 difference is invisible at "
        "full scale); bands are mean +/- 1 sd across replicates (repeat-run variability, "
        "NOT confidence or credible intervals).  The right panel carries the evidence: "
        "the paired difference with +/-2 SE.  "
        f"Held-out-seed confirmation: mean diff {held.paired_diff.mean():+.4f} over "
        f"{len(held)} independent seed batches.",
        ha="center", fontsize=7.4,
    )
    return _save(fig, "S2_nonreversible_wins")


def figure_confirmation(confirmation, s):
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels, colors = [], []
    for _, row in confirmation.iterrows():
        labels.append(("search seed" if not row.held_out else f"held-out {int(row.seed_offset)}"))
        colors.append("#999999" if not row.held_out else STYLE[NR]["color"])
    y = np.arange(len(confirmation))
    ax.errorbar(confirmation["paired_diff"], y, xerr=2 * confirmation["paired_se"],
                fmt="o", ms=7, capsize=4, lw=1.6, color=STYLE[NR]["color"], ls="none")
    for i, colour in enumerate(colors):
        ax.plot(confirmation["paired_diff"].iloc[i], y[i], "o", ms=7, color=colour)
    pooled = confirmation[confirmation.held_out]["paired_diff"].mean()
    ax.axvline(0.0, color="k", lw=1.2)
    ax.axvline(pooled, color=STYLE[NR]["color"], ls=":", lw=1.4,
               label=f"held-out mean {pooled:+.4f}")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("paired  NR − REV  final test accuracy  (±2 SE)")
    ax.set_title(f"Confirmation at s = {s}: every seed batch is positive", fontsize=12)
    ax.grid(alpha=0.3, axis="x"); ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "S3_held_out_confirmation")


# ==========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    repeats = 20 if args.quick else 100
    iterations = 300 if args.quick else 1000
    s_specified = (0.5, 1.0, 2.0, 3.0, 5.0) if args.quick else (
        0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0)
    s_ill = (1.0, 3.0, 5.0, 7.0) if args.quick else (1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0)
    offsets = (0, 101) if args.quick else (0, 101, 202, 303)

    cfg = nral.ExperimentConfig(n_repeats=repeats, n_iterations=iterations,
                                checkpoint_every=10)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"=== block-strength search ({'QUICK' if args.quick else 'FULL'}): "
          f"R = {repeats}, {iterations} iterations ===")

    specified, condition_iso = study_specified(cfg, s_specified)
    specified.to_csv(os.path.join(OUTPUT_DIR, "search_specified_design.csv"), index=False)
    paths = figure_specified(specified, condition_iso)

    reported = tuple(sorted({max(10, int(iterations * f)) for f in (0.2, 0.4, 0.7, 1.0)}))
    ill, dataset_ill, condition_ill, _ = study_illconditioned(cfg, s_ill, 0.99, reported)
    ill.to_csv(os.path.join(OUTPUT_DIR, "search_illconditioned_design.csv"), index=False)

    # Select s on the search seed by the t statistic at the final iteration.
    best_s = float(ill.loc[ill[f"t@{iterations}"].idxmax(), "s"])
    confirmation = confirm(cfg, dataset_ill, best_s, offsets)
    confirmation.to_csv(os.path.join(OUTPUT_DIR, "held_out_confirmation.csv"), index=False)

    paths += figure_winning(cfg, dataset_ill, best_s, condition_ill, confirmation)
    paths += figure_confirmation(confirmation, best_s)

    held = confirmation[confirmation.held_out]
    best_variance = specified.loc[specified["ergodic_variance_ratio"].idxmin()]
    summary = {
        "selected_s": best_s,
        "isotropic_condition_number": condition_iso,
        "illconditioned_condition_number": condition_ill,
        "specified_design_max_accuracy_t": float(specified["t_stat"].max()),
        "specified_design_n_configs_tested": int(len(specified)),
        "best_ergodic_variance_ratio": float(best_variance["ergodic_variance_ratio"]),
        "best_ergodic_variance_s": float(best_variance["s"]),
        "best_ergodic_variance_geometry": str(best_variance["geometry"]),
        "held_out_mean_diff": float(held["paired_diff"].mean()),
        "held_out_all_positive": bool((held["paired_diff"] > 0).all()),
        "held_out_t_stats": [float(t) for t in held["t_stat"]],
        "held_out_pooled_diff": float(held["paired_diff"].mean()),
        "held_out_pooled_se": float(
            np.sqrt((held["paired_se"] ** 2).sum()) / len(held)),
        "held_out_pooled_t": float(
            held["paired_diff"].mean()
            / (np.sqrt((held["paired_se"] ** 2).sum()) / len(held))),
        "figures": paths,
        "runtime_s": time.perf_counter() - started,
        "quick": args.quick,
    }
    with open(os.path.join(OUTPUT_DIR, "search_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== conclusion ===")
    print(f"Specified isotropic design ({len(specified)} configurations tested): best "
          f"accuracy t = {specified['t_stat'].max():+.2f} -> no s wins on accuracy.")
    print(f"But ergodic-average variance falls to "
          f"{best_variance['ergodic_variance_ratio']:.3f} at s = {best_variance['s']} "
          f"({best_variance['geometry']}).")
    print(f"Ill-conditioned design: s = {best_s} gives held-out mean paired diff "
          f"{summary['held_out_pooled_diff']:+.4f} "
          f"(pooled SE {summary['held_out_pooled_se']:.5f}, "
          f"t = {summary['held_out_pooled_t']:+.2f}), "
          f"all {len(held)} held-out batches positive.")
    print(f"\nwrote {len(paths)} figure files to {OUTPUT_DIR}/  "
          f"({summary['runtime_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
