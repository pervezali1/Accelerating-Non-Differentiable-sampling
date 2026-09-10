#!/usr/bin/env python3
"""Plot the irreversibility-strength sweep written by ``run_alpha_sweep.py``."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLE = {
    "constant": dict(color="#2B5FD9", marker="o", lw=2.0, ls="-", label=r"constant $J_a$"),
    "state": dict(color="#0E9F5C", marker="s", lw=2.0, ls=(0, (6, 3)), label=r"state-dependent $J_s$"),
}
INK, MUTED = "#1f2328", "#6b7280"


def main() -> None:
    with open(os.path.join(ROOT, "results", "alpha_sweep.json")) as handle:
        blob = json.load(handle)
    sets = blob["datasets"]
    fig, axes = plt.subplots(
        len(sets), 2, figsize=(10.5, 3.9 * len(sets)), squeeze=False, layout="constrained"
    )
    for row, entry in enumerate(sets):
        rows = entry["rows"]
        for col, (field, ylabel, logy) in enumerate(
            [
                ("step_size", "calibrated step size $h$", True),
                ("mean_error_final", "whitened error of the mean at 600 iterations", True),
            ]
        ):
            ax = axes[row][col]
            for kind, style in STYLE.items():
                # alpha = 0 is the reversible baseline and is stored under both kinds
                pts = sorted((r["alpha"], r[field]) for r in rows if r["kind"] == kind)
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, markersize=5, **style)
            if logy:
                ax.set_yscale("log")
            ax.grid(True, axis="y", color="#d8dade", lw=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.set_xlabel(r"irreversibility strength $\alpha$", color=INK, fontsize=10)
            ax.set_ylabel(ylabel, color=INK, fontsize=10)
            ax.set_title(entry["pretty_name"], color=INK, fontsize=11.5, pad=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False, fontsize=9.5)
    fig.suptitle(
        r"Cost of the rotation: $\alpha = 0$ is the reversible baseline",
        fontsize=13.5, color=INK,
    )
    out = os.path.join(ROOT, "figures", "alpha_sweep.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
