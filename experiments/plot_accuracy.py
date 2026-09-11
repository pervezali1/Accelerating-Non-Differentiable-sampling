#!/usr/bin/env python3
"""Figures and summary tables for the four-dataset accuracy experiment.

Reads the trace files written by ``run_accuracy.py`` and writes

* ``figures/accuracy_four_datasets.png`` -- accuracy against iterations,
* ``figures/accuracy_four_datasets_logx.png`` -- the same on a log iteration axis,
* ``figures/posterior_mean_error.png`` -- error of the running posterior mean,
* ``results/summary.csv`` and ``results/summary.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

# Categorical palette, validated for colour-vision deficiency; dash patterns
# repeat the same information so identity never rests on colour alone.
STYLE = {
    "zero": dict(color="#C0334D", lw=2.6, ls="-", zorder=4),
    "constant": dict(color="#2B5FD9", lw=2.0, ls="-", zorder=3),
    "state": dict(color="#0E9F5C", lw=2.0, ls=(0, (6, 3)), zorder=5),
    "state_nocorr": dict(color="#7C4DBE", lw=2.0, ls=(0, (1.5, 2.5)), zorder=6),
}
LABELS = {
    "zero": r"$J = 0$",
    "constant": r"constant $J_a$",
    "state": r"state-dependent $J_s$",
    "state_nocorr": r"$J_s$, correction dropped",
}
# optional extra series: the reversible method that spends the warm-up
# covariance on a preconditioner instead of on designing J
PRECOND_STYLE = dict(color="#6b7280", lw=1.4, ls=(0, (4, 2, 1, 2)), zorder=8)
PRECOND_LABEL = "reversible, preconditioned"
INK = "#1f2328"
MUTED = "#6b7280"


def load_traces(dataset: str, geometry: str, alpha: float, tag: str | None = None) -> dict:
    stem = tag if tag else f"{geometry}_alpha{alpha:g}"
    path = os.path.join(RESULTS, f"traces_{dataset}_{stem}.npz")
    blob = np.load(path, allow_pickle=False)
    meta = json.loads(str(blob["meta"]))
    return {"meta": meta, "arrays": {k: blob[k] for k in blob.files if k != "meta"}}


def _panel_frame(ax) -> None:
    ax.grid(True, axis="y", color="#d8dade", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c7cad0")
    ax.tick_params(colors=MUTED, labelsize=9)


def accuracy_figure(
    runs: list, out_path: str, log_x: bool = False,
    title: str = "Accuracy from the $w = 0$ start",
) -> None:
    n = len(runs)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.6 * ncols, 3.9 * nrows), squeeze=False, layout="constrained"
    )

    for ax, run in zip(axes.ravel(), runs):
        meta, arrays = run["meta"], run["arrays"]
        ref = meta["reference_accuracy"]
        iters = np.arange(meta["n_iter"] + 1)
        x = iters + 1 if log_x else iters
        _panel_frame(ax)

        band = np.percentile(arrays["accuracy_zero"], [5, 95], axis=1)
        ax.fill_between(x, band[0], band[1], color=STYLE["zero"]["color"], alpha=0.15, lw=0,
                        label="5-95% across walkers", zorder=2)
        for key in meta["variant_order"]:
            ax.plot(x, arrays[f"accuracy_{key}"].mean(axis=1), label=LABELS[key], **STYLE[key])
        if "accuracy_precond" in arrays:
            ax.plot(x, arrays["accuracy_precond"].mean(axis=1), label=PRECOND_LABEL,
                    **PRECOND_STYLE)

        ax.axhline(ref, color=INK, ls=(0, (5, 3)), lw=1.2, zorder=7)
        ax.axhline(0.5, color=MUTED, ls=(0, (1, 3)), lw=1.0, zorder=1)
        ax.annotate(
            f"exact posterior {ref:.3f}",
            xy=(x[-1], ref), xytext=(-4, 5), textcoords="offset points",
            ha="right", va="bottom", fontsize=8.5, color=INK,
        )
        ax.annotate(
            "chance", xy=(x[-1], 0.5), xytext=(-4, 4), textcoords="offset points",
            ha="right", va="bottom", fontsize=8.5, color=MUTED,
        )

        lo = min(0.48, band[0].min() - 0.01)
        ax.set_ylim(lo, max(ref, band[1].max()) + 0.035)
        if log_x:
            ax.set_xscale("log")
            ax.set_xlabel("iteration (log scale)", color=INK, fontsize=10)
        else:
            ax.set_xlim(0, meta["n_iter"])
            ax.set_xlabel("iterations", color=INK, fontsize=10)
        ax.set_ylabel("classification accuracy", color=INK, fontsize=10)
        ax.set_title(
            f"{meta['pretty_name']}  (d = {meta['dim']}, n = {meta['n_train']})",
            color=INK, fontsize=11.5, pad=8,
        )

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    order = [labels.index(LABELS[k]) for k in runs[0]["meta"]["variant_order"]]
    if PRECOND_LABEL in labels:
        order.append(labels.index(PRECOND_LABEL))
    order.append(labels.index("5-95% across walkers"))
    fig.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="outside lower center", ncols=min(len(order), 5), frameon=False, fontsize=9.5,
    )
    fig.suptitle(title, fontsize=14, color=INK)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("wrote", out_path)


def error_figure(runs: list, out_path: str) -> None:
    n = len(runs)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.6 * ncols, 3.9 * nrows), squeeze=False, layout="constrained"
    )
    for ax, run in zip(axes.ravel(), runs):
        meta, arrays = run["meta"], run["arrays"]
        iters = np.arange(meta["n_iter"] + 1) + 1
        _panel_frame(ax)
        for key in meta["variant_order"]:
            ax.plot(iters, arrays[f"error_{key}"].mean(axis=1), label=LABELS[key], **STYLE[key])
        if "error_precond" in arrays:
            ax.plot(iters, arrays["error_precond"].mean(axis=1), label=PRECOND_LABEL,
                    **PRECOND_STYLE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("iteration (log scale)", color=INK, fontsize=10)
        ax.set_ylabel("whitened error of running mean", color=INK, fontsize=10)
        ax.set_title(
            f"{meta['pretty_name']}  (d = {meta['dim']})", color=INK, fontsize=11.5, pad=8
        )
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=min(len(labels), 5),
               frameon=False, fontsize=9.5)
    fig.suptitle(
        "Distance of the running posterior mean from the reference posterior",
        fontsize=13.5, color=INK,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("wrote", out_path)


# written in this order when a run records them; the Metropolis-corrected runs
# report an acceptance rate and the unadjusted ones report a predicted decay
PREFERRED_FIELDS = [
    "dataset", "variant", "step_size", "acceptance", "predicted_decay",
    "accuracy_at_100", "accuracy_final", "iters_to_ref_0.005", "mean_error_final",
    "ess_potential_second_half", "variance_inflation", "step_reductions",
]


def write_summary(runs: list, suffix: str = "") -> None:
    entries = [entry for run in runs for entry in run["meta"]["summary"]]
    fields = [f for f in PREFERRED_FIELDS if all(f in entry for entry in entries)]
    rows = []
    for run in runs:
        for entry in run["meta"]["summary"]:
            row = {k: entry[k] for k in fields}
            row["reference_accuracy"] = run["meta"]["reference_accuracy"]
            rows.append(row)
    csv_path = os.path.join(RESULTS, f"summary{suffix}.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["reference_accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", csv_path)

    def cell(value) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    md = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        if row.get("iters_to_ref_0.005", 0) == -1:
            row["iters_to_ref_0.005"] = "not reached"
        md.append("| " + " | ".join(cell(row[f]) for f in fields) + " |")
    md_path = os.path.join(RESULTS, f"summary{suffix}.md")
    with open(md_path, "w") as handle:
        handle.write("\n".join(md) + "\n")
    print("wrote", md_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["titanic", "magic", "breast_cancer", "spambase"]
    )
    parser.add_argument("--geometry", default="warmup")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--title", default="Accuracy from the $w = 0$ start")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="which fields to draw; defaults to every one the run recorded")
    parser.add_argument("--tag", default=None,
                        help="trace-file stem after the dataset name; defaults to "
                             "'<geometry>_alpha<alpha>'")
    parser.add_argument(
        "--suffix", default="", help="appended to every output name; set it when "
        "plotting a non-default geometry so the default outputs are not overwritten"
    )
    args = parser.parse_args()

    os.makedirs(FIGURES, exist_ok=True)
    runs = []
    for name in args.datasets:
        try:
            runs.append(load_traces(name, args.geometry, args.alpha, tag=args.tag))
        except FileNotFoundError:
            print(f"missing traces for {name}; skipping", file=sys.stderr)
    if not runs:
        raise SystemExit("no traces found")

    if args.variants:
        for run in runs:
            run["meta"]["variant_order"] = [
                v for v in args.variants if f"accuracy_{v}" in run["arrays"]
            ]
            if "precond" not in args.variants:
                run["arrays"].pop("accuracy_precond", None)
                run["arrays"].pop("error_precond", None)

    sfx = args.suffix
    accuracy_figure(runs, os.path.join(FIGURES, f"accuracy_four_datasets{sfx}.png"),
                    title=args.title)
    accuracy_figure(runs, os.path.join(FIGURES, f"accuracy_four_datasets_logx{sfx}.png"),
                    log_x=True, title=args.title)
    error_figure(runs, os.path.join(FIGURES, f"posterior_mean_error{sfx}.png"))
    write_summary(runs, suffix=sfx)


if __name__ == "__main__":
    main()
