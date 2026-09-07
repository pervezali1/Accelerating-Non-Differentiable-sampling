"""Figures, in the style of the ball notebook: W1 traces per coordinate, 2-d contours
with the constraint circle, and the strength sweep."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle

from .experiments import COLOR, ROOT

FIGDIR = os.path.join(ROOT, "figures")


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, "{}.{}".format(name, ext)), dpi=160,
                    bbox_inches="tight")


def w1_traces(runs, floor, feature_names, title, name, coords=(1, 2, 3)):
    """One panel per coordinate: W1 to the reference against iteration, log scale,
    with the reference-resolution floor drawn in."""
    fig, axes = plt.subplots(1, len(coords), figsize=(5.3 * len(coords), 4.4))
    for ax, i in zip(np.atleast_1d(axes), coords):
        for lab, r in runs.items():
            ax.plot(r["iters"], r["W1"][:, i], linewidth=2, color=COLOR[lab], label=lab)
        ax.axhline(floor[i], color="k", linestyle="--", linewidth=1.3,
                   label="reference floor")
        ax.set_yscale("log")
        ax.set_xlabel("Iterations", fontsize=13)
        ax.set_ylabel(r"$W_1$", fontsize=13)
        ax.set_title("{}  ({})".format(feature_names[i], i), fontsize=12)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    _save(fig, name)
    return fig


def density_panels(ref, runs, R, title, name, dims=(1, 2), feature_names=None):
    """2-d marginal contours with the constraint circle drawn, target first."""
    panels = [("reference (exact target)", ref)] + [(lab, r["x"]) for lab, r in runs.items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.9 * len(panels), 4.7),
                             sharex=True, sharey=True)
    i, j = dims
    for ax, (lab, X) in zip(axes, panels):
        sns.kdeplot(x=X[:, i], y=X[:, j], fill=True, levels=8, ax=ax, thresh=0.02)
        ax.add_patch(Circle((0, 0), R, edgecolor="b", facecolor="none", linewidth=1.6))
        ax.set_title(lab, fontsize=12)
        ax.set_xlim(-1.05 * R, 1.05 * R)
        ax.set_ylim(-1.05 * R, 1.05 * R)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(feature_names[i] if feature_names else "$x_{}$".format(i), fontsize=12)
    axes[0].set_ylabel(feature_names[j] if feature_names else "$x_{}$".format(j), fontsize=12)
    fig.suptitle(title, fontsize=15, y=1.02)
    fig.tight_layout()
    _save(fig, name)
    return fig


def sweep_panels(sweep, floor_mean, ref_boundary, title, name):
    """Mean W1 and boundary mass against skew strength, one panel each."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for lab, series in sweep.items():
        s, w1, bd = series["s"], series["W1_mean"], series["boundary"]
        axes[0].plot(s, w1, "o-", color=COLOR[lab], linewidth=2, label=lab)
        axes[1].plot(s, 100 * np.array(bd), "o-", color=COLOR[lab], linewidth=2, label=lab)
    axes[0].axhline(floor_mean, color="k", linestyle="--", linewidth=1.3, label="reference floor")
    axes[1].axhline(100 * ref_boundary, color="k", linestyle="--", linewidth=1.3,
                    label="target near-boundary mass")
    axes[0].set_ylabel(r"mean $W_1$ over coordinates", fontsize=13)
    axes[1].set_ylabel(r"mass within 0.1% of $\partial K$  (%)", fontsize=13)
    for ax in axes:
        ax.set_xlabel("skew strength $s$", fontsize=13)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=10)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    _save(fig, name)
    return fig


def ablation_panel(ablation, floor_mean, title, name):
    """Mean W1 against strength, with and without the div-J correction."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for lab, series in ablation.items():
        ax.plot(series["s"], series["W1_mean"], "o-", color=COLOR[lab], linewidth=2, label=lab)
    ax.axhline(floor_mean, color="k", linestyle="--", linewidth=1.3, label="reference floor")
    ax.set_xlabel("skew strength $s$", fontsize=13)
    ax.set_ylabel(r"mean $W_1$ over coordinates", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, name)
    return fig
