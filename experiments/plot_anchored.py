#!/usr/bin/env python3
"""Accuracy and loss for the anchored runs written by ``run_anchored.py``.

``figures/anchored_accuracy_loss.png`` -- three fields, both metrics, one row per
dataset, against the random-walk Metropolis reference for the non-differentiable
target.
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
RESULTS, FIGURES = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")

SERIES = [
    ("zero", r"$J = 0$", dict(color="#C0334D", lw=2.6, ls="-", zorder=4)),
    ("constant", r"constant $J_a$", dict(color="#2B5FD9", lw=2.0, ls="-", zorder=5)),
    ("state", r"state-dependent $J_s$", dict(color="#0E9F5C", lw=2.2, ls=(0, (6, 3)), zorder=6)),
]
INK, MUTED = "#1f2328", "#6b7280"


def style(ax) -> None:
    ax.grid(True, axis="y", color="#d8dade", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c7cad0")
    ax.tick_params(colors=MUTED, labelsize=9)


BIAS_STYLE = {
    "anchored": dict(color="#0E9F5C", lw=2.4, ls="-", zorder=5),
    "anchor": dict(color="#2B5FD9", lw=2.0, ls=(0, (6, 3)), zorder=4),
    "fd": dict(color="#7C4DBE", lw=2.0, ls=(0, (1.5, 2.5)), zorder=4),
}
BIAS_LABEL = {
    "anchored": r"anchored: $-a\,\nabla U_0$, $a = e^{U - U_0}$",
    "anchor": r"anchor only: $-\nabla U_0$ (samples $e^{-U_0}$)",
    "fd": r"differences of $U$ (samples an $\epsilon$-smoothed law)",
}


def bias_figure(datasets: list, deltas: list) -> None:
    """Long runs of the three schemes: where the clock starts to matter."""
    panels = []
    for name in datasets:
        for delta in deltas:
            path = os.path.join(RESULTS, f"bias_anchored_{name}_d{delta:g}.npz")
            if os.path.exists(path):
                blob = np.load(path, allow_pickle=False)
                panels.append((json.loads(str(blob["meta"])), blob))
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 2, figsize=(11.4, 4.0 * len(panels)),
                             squeeze=False, layout="constrained")
    for row, (meta, blob) in enumerate(panels):
        iters = np.arange(meta["n_iter"] + 1) + 1
        for col, metric in enumerate(["error", "loss"]):
            ax = axes[row][col]
            style(ax)
            for scheme in meta["schemes"]:
                ax.plot(iters, blob[f"{metric}_{scheme}"], label=BIAS_LABEL[scheme],
                        **BIAS_STYLE[scheme])
            ax.set_xscale("log")
            if metric == "error":
                ax.set_yscale("log")
                ax.set_ylabel("whitened error of the running mean", color=INK, fontsize=10)
            else:
                ax.axhline(meta["reference_loss"], color=INK, ls=(0, (5, 3)), lw=1.2)
                lo = min(meta["reference_loss"], min(blob[f"loss_{s}"][-1] for s in meta["schemes"]))
                hi = max(blob[f"loss_{s}"][meta["n_iter"] // 20] for s in meta["schemes"])
                ax.set_ylim(lo - 0.002, hi + 0.002)
                ax.set_ylabel("predictive log-loss (nats)", color=INK, fontsize=10)
            ax.set_xlabel("iteration (log scale)", color=INK, fontsize=10)
            ax.set_title(f"{meta['pretty_name']}: {meta['n_iter']} iterations, "
                         f"$\\lambda$ = {meta['penalty']:g}, $\\delta$ = {meta['delta']:g}",
                         color=INK, fontsize=10.5, pad=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=1, frameon=False, fontsize=9.5)
    fig.suptitle("The clock costs speed and buys exactness, and how much of each "
                 "depends on $\\delta$", fontsize=13, color=INK)
    out = os.path.join(FIGURES, "anchored_clock_bias.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["titanic", "magic"])
    parser.add_argument("--skip", type=int, default=2, help="iterations to ignore when zooming")
    parser.add_argument("--bias-deltas", nargs="+", type=float, default=[0.02, 0.1],
                        help="anchor smoothings to look for in the long-run bias check")
    args = parser.parse_args()

    runs = []
    for name in args.datasets:
        path = os.path.join(RESULTS, f"traces_{name}_anchored.npz")
        if not os.path.exists(path):
            print(f"missing {path}; skipping")
            continue
        blob = np.load(path, allow_pickle=False)
        meta = json.loads(str(blob["meta"]))
        runs.append((meta, {k: {m: blob[f"{m}_{k}"].mean(axis=1) for m in ("accuracy", "loss")}
                            for k, _, _ in SERIES}))
    if not runs:
        raise SystemExit("no anchored traces found")

    fig, axes = plt.subplots(len(runs), 2, figsize=(11.4, 4.0 * len(runs)),
                             squeeze=False, layout="constrained")
    for row, (meta, curves) in enumerate(runs):
        iters = np.arange(meta["n_iter"] + 1)
        for col, (metric, label) in enumerate([("accuracy", "classification accuracy"),
                                               ("loss", "predictive log-loss (nats)")]):
            ax = axes[row][col]
            style(ax)
            ref = meta["reference_accuracy"] if metric == "accuracy" else meta["reference_loss"]
            band = 0.005 if metric == "accuracy" else 0.01 * ref
            ax.axhspan(ref - band, ref + band, color="#d8dade", alpha=0.45, lw=0, zorder=1)
            ax.axhline(ref, color=INK, ls=(0, (5, 3)), lw=1.2, zorder=7)
            flat = []
            for key, legend, spec in SERIES:
                curve = curves[key][metric]
                flat.append(curve[args.skip:])
                ax.plot(iters, curve, label=legend, **spec)
            flat = np.concatenate(flat + [[ref]])
            pad = 0.12 * (flat.max() - flat.min()) + 1e-3
            ax.set_ylim(flat.min() - pad, flat.max() + pad)
            ax.set_xlim(0, meta["n_iter"])
            ax.set_xlabel("iterations", color=INK, fontsize=10)
            ax.set_ylabel(label, color=INK, fontsize=10)
            amp = meta["amplitude"]
            ax.set_title(
                f"{meta['pretty_name']}  (d = {meta['dim']}, n = {meta['n_train']})\n"
                f"$\\lambda$ = {meta['penalty']:g}, $\\delta$ = {meta['delta']:g}, "
                f"amplitude {amp['constant']:g} constant, {amp['state']:g} gated",
                color=INK, fontsize=10.5, pad=8,
            )
            note = ("dashed: exact posterior, band $\\pm$0.005" if metric == "accuracy"
                    else "dashed: exact posterior, band $\\pm$1%")
            y, va = (0.06, "bottom") if metric == "accuracy" else (0.96, "top")
            ax.annotate(note, xy=(0.98, y), xycoords="axes fraction", ha="right", va=va,
                        fontsize=8.5, color=MUTED)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=10)
    fig.suptitle("Anchored Langevin on the $\\ell_1$ target, from the $w = 0$ start",
                 fontsize=14, color=INK)
    os.makedirs(FIGURES, exist_ok=True)
    out = os.path.join(FIGURES, "anchored_accuracy_loss.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)
    plt.close(fig)
    bias_figure(args.datasets, args.bias_deltas)


if __name__ == "__main__":
    main()
