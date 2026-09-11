#!/usr/bin/env python3
"""Three fields, one line each: ``J = 0``, constant ``J_a``, state-dependent ``J_s``.

Two figures, both from the traces written by ``run_designed.py``:

* ``figures/three_lines_accuracy.png`` -- test accuracy of the running posterior
  predictive, zoomed to the range the curves actually occupy so that a few
  thousandths are visible;
* ``figures/three_lines_gap.png`` -- the absolute gap to the reference posterior's
  accuracy on a log scale, which is the unambiguous version: on the two
  high-dimensional posteriors the baseline sits *above* the reference rather
  than at it, and a curve above the line is not a curve that has converged.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

SERIES = [
    ("zero", r"$J = 0$", dict(color="#C0334D", lw=2.6, ls="-", zorder=4)),
    ("constant", r"constant $J_a$", dict(color="#2B5FD9", lw=2.0, ls="-", zorder=5)),
    ("state", r"state-dependent $J_s$", dict(color="#0E9F5C", lw=2.2, ls=(0, (6, 3)), zorder=6)),
]
INK, MUTED = "#1f2328", "#6b7280"


def frame(ax) -> None:
    ax.grid(True, axis="y", color="#d8dade", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c7cad0")
    ax.tick_params(colors=MUTED, labelsize=9)


def load(datasets: list, tag: str) -> list:
    runs = []
    for name in datasets:
        path = os.path.join(RESULTS, f"traces_{name}_{tag}.npz")
        if not os.path.exists(path):
            print(f"missing {path}; skipping")
            continue
        blob = np.load(path, allow_pickle=False)
        runs.append({
            "meta": json.loads(str(blob["meta"])),
            "curves": {k: blob[f"accuracy_{k}"].mean(axis=1) for k, _, _ in SERIES},
        })
    return runs


def panel_grid(n: int):
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.7 * ncols, 3.9 * nrows), squeeze=False, layout="constrained"
    )
    return fig, axes


def accuracy_figure(runs: list, out: str, skip: int = 2) -> None:
    fig, axes = panel_grid(len(runs))
    for ax, run in zip(axes.ravel(), runs):
        meta, curves = run["meta"], run["curves"]
        ref = meta["reference_accuracy"]
        iters = np.arange(meta["n_iter"] + 1)
        frame(ax)
        for key, label, style in SERIES:
            ax.plot(iters, curves[key], label=label, **style)
        band = 0.005
        ax.axhspan(ref - band, ref + band, color="#d8dade", alpha=0.45, lw=0, zorder=1)
        ax.axhline(ref, color=INK, ls=(0, (5, 3)), lw=1.2, zorder=7)
        # the dashed line and the band are self-evident; label them out of the way
        ax.annotate(
            f"dashed: exact posterior {ref:.3f}, band $\\pm$ 0.005",
            xy=(0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=8.5, color=MUTED, zorder=8,
        )
        # zoom past the first couple of iterations, where every curve leaves 0.5
        tail = np.concatenate([c[skip:] for c in curves.values()] + [[ref]])
        lo, hi = tail.min(), tail.max()
        pad = 0.12 * (hi - lo) + 0.003
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0, meta["n_iter"])
        amp = meta["amplitude"]
        ax.set_title(
            f"{meta['pretty_name']}  (d = {meta['dim']}, n = {meta['n_train']})\n"
            f"amplitude {amp['constant']:g} constant, {amp['state']:g} gated",
            color=INK, fontsize=10.5, pad=8,
        )
        ax.set_xlabel("iterations", color=INK, fontsize=10)
        ax.set_ylabel("classification accuracy", color=INK, fontsize=10)
    for ax in axes.ravel()[len(runs):]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=10)
    fig.suptitle("Accuracy from the $w = 0$ start", fontsize=14, color=INK)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def gap_figure(runs: list, out: str) -> None:
    fig, axes = panel_grid(len(runs))
    for ax, run in zip(axes.ravel(), runs):
        meta, curves = run["meta"], run["curves"]
        ref = meta["reference_accuracy"]
        iters = np.arange(meta["n_iter"] + 1) + 1
        frame(ax)
        for key, label, style in SERIES:
            ax.plot(iters, np.maximum(np.abs(curves[key] - ref), 1e-4), label=label, **style)
        ax.axhline(0.005, color=MUTED, ls=(0, (1, 3)), lw=1.0, zorder=2)
        ax.annotate(
            "0.005 band", xy=(0.02, 0.005), xycoords=("axes fraction", "data"),
            xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=MUTED,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("iteration (log scale)", color=INK, fontsize=10)
        ax.set_ylabel("|accuracy - exact posterior|", color=INK, fontsize=10)
        ax.set_title(
            f"{meta['pretty_name']}  (d = {meta['dim']})", color=INK, fontsize=11, pad=8
        )
    for ax in axes.ravel()[len(runs):]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=10)
    fig.suptitle(
        "Distance from the exact posterior's accuracy, lower is better",
        fontsize=13.5, color=INK,
    )
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["titanic", "magic", "breast_cancer", "spambase"]
    )
    parser.add_argument("--tag", default="designed")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    os.makedirs(FIGURES, exist_ok=True)
    runs = load(args.datasets, args.tag)
    if not runs:
        raise SystemExit("no traces found")
    accuracy_figure(runs, os.path.join(FIGURES, f"three_lines_accuracy{args.suffix}.png"))
    gap_figure(runs, os.path.join(FIGURES, f"three_lines_gap{args.suffix}.png"))


if __name__ == "__main__":
    main()
