#!/usr/bin/env python3
"""Plot the uncorrected-diffusion bias written by ``run_correction_bias.py``."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLE = {
    "zero": dict(color="#C0334D", lw=2.6, ls="-"),
    "with_correction": dict(color="#0E9F5C", lw=2.0, ls=(0, (6, 3))),
    "no_correction": dict(color="#7C4DBE", lw=2.0, ls=(0, (1.5, 2.5))),
}
LABELS = {
    "zero": r"$J = 0$ (discretisation bias only)",
    "with_correction": r"$J \neq 0$ with $\Gamma = \nabla\cdot J$",
    "no_correction": r"$J \neq 0$, correction dropped",
}
TITLES = {
    "radial": "radial profile, centred on $w = 0$ (as in the accuracy figure)",
    "directional": "directional profile, centred on the posterior bulk",
}
INK, MUTED = "#1f2328", "#6b7280"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["titanic", "breast_cancer"])
    args = parser.parse_args()

    sets = []
    for name in args.datasets:
        blob = np.load(os.path.join(ROOT, "results", f"bias_{name}.npz"))
        sets.append(json.loads(str(blob["meta"])))
    profiles = ["radial", "directional"]
    fig, axes = plt.subplots(
        len(sets), 2, figsize=(5.8 * 2, 4.0 * len(sets)), squeeze=False, layout="constrained"
    )
    for row, meta in enumerate(sets):
        blob = np.load(os.path.join(ROOT, "results", f"bias_{meta['dataset']}.npz"))
        iters = np.arange(meta["n_iter"] + 1) + 1
        for col, profile in enumerate(profiles):
            ax = axes[row][col]
            ax.plot(iters, blob["error_shared_zero"].mean(axis=1), label=LABELS["zero"], **STYLE["zero"])
            for variant in ("with_correction", "no_correction"):
                ax.plot(
                    iters, blob[f"error_{profile}_{variant}"].mean(axis=1),
                    label=LABELS[variant], **STYLE[variant],
                )
            strength = meta["field_strength"][profile]
            ratio = strength["gamma_norm"] / max(strength["reversible_drift_norm"], 1e-30)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, axis="y", color="#d8dade", lw=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.set_xlabel("iteration (log scale)", color=INK, fontsize=10)
            ax.set_ylabel("whitened error of running mean", color=INK, fontsize=10)
            ax.set_title(
                f"{meta['pretty_name']}\n{TITLES[profile]}", color=INK, fontsize=10.5, pad=8
            )
            ax.annotate(
                rf"$\|\Gamma\| / \|D\hat g\| = {ratio:.2f}$ in the bulk",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=9, color=MUTED,
            )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=9.5)
    fig.suptitle(
        "Uncorrected diffusion: the plateau is the bias in the invariant law",
        fontsize=13.5, color=INK,
    )
    out = os.path.join(ROOT, "figures", "correction_bias.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
