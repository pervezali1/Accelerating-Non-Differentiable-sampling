#!/usr/bin/env python3
"""Plot the uncorrected-diffusion bias written by ``run_correction_bias.py``."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLE = {
    "zero": dict(color="#C0334D", lw=2.6, ls="-"),
    "state": dict(color="#0E9F5C", lw=2.0, ls=(0, (6, 3))),
    "state_nocorr": dict(color="#7C4DBE", lw=2.0, ls=(0, (1.5, 2.5))),
}
LABELS = {
    "zero": r"$J = 0$",
    "state": r"$J_s$ with $\Gamma = \nabla\cdot J_s$",
    "state_nocorr": r"$J_s$, correction dropped",
}
INK, MUTED = "#1f2328", "#6b7280"


def main() -> None:
    with open(os.path.join(ROOT, "results", "correction_bias.json")) as handle:
        sets = json.load(handle)["datasets"]
    fig, axes = plt.subplots(
        1, len(sets), figsize=(5.6 * len(sets), 4.1), squeeze=False, layout="constrained"
    )
    for ax, meta in zip(axes.ravel(), sets):
        blob = np.load(os.path.join(ROOT, "results", f"bias_{meta['dataset']}.npz"))
        iters = np.arange(meta["n_iter"] + 1) + 1
        for key in meta["variant_order"]:
            ax.plot(iters, blob[f"error_{key}"].mean(axis=1), label=LABELS[key], **STYLE[key])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, axis="y", color="#d8dade", lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel("iteration (log scale)", color=INK, fontsize=10)
        ax.set_ylabel("whitened error of running mean", color=INK, fontsize=10)
        ax.set_title(meta["pretty_name"], color=INK, fontsize=11.5, pad=8)
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
