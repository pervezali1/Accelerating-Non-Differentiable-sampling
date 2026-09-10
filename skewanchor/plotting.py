r"""Shared plotting style.

Colours come from a validated categorical palette: the six-slot light-mode order
clears the lightness band, chroma floor, adjacent-pair colour-vision-deficiency
separation (worst 9.1) and the normal-vision floor (worst 19.6).  Three of the
slots sit below 3:1 contrast against the surface, so every multi-series figure
carries direct labels as well as a legend -- identity is never colour alone.

Ordered quantities (the magnitude of ``J``) use a single-hue sequential ramp
rather than categorical hues, starting no lighter than the step that clears 2:1
against the surface.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["CATEGORICAL", "SEQUENTIAL", "PALETTES", "use_style", "finish", "label_line"]

PALETTES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text_secondary": "#52514e",
        "grid": "#e3e2de",
        "categorical": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                        "#008300", "#4a3aa7", "#e34948"],
        # ordinal ramp: start at step 250, which clears 2:1 on the light surface
        "sequential": ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
                       "#1c5cab", "#184f95", "#104281", "#0d366b"],
        "reference": "#8a8a86",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "text_secondary": "#c3c2b7",
        "grid": "#333331",
        "categorical": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
                        "#008300", "#9085e9", "#e66767"],
        "sequential": ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                       "#5598e7", "#3987e5", "#2a78d6", "#184f95"],
        "reference": "#7a7a76",
    },
}

CATEGORICAL = PALETTES["light"]["categorical"]
SEQUENTIAL = PALETTES["light"]["sequential"]


def use_style(mode="light"):
    p = PALETTES[mode]
    plt.rcParams.update({
        "figure.facecolor": p["surface"],
        "axes.facecolor": p["surface"],
        "savefig.facecolor": p["surface"],
        "axes.edgecolor": p["grid"],
        "axes.labelcolor": p["text_secondary"],
        "axes.titlecolor": p["text"],
        "text.color": p["text"],
        "xtick.color": p["text_secondary"],
        "ytick.color": p["text_secondary"],
        "grid.color": p["grid"],
        "grid.linewidth": 0.7,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.dpi": 200,
    })
    return p


def ramp(n, mode="light"):
    """``n`` evenly spread steps of the sequential ramp, dark end last."""
    seq = PALETTES[mode]["sequential"]
    if n == 1:
        return [seq[len(seq) // 2]]
    idx = [round(i * (len(seq) - 1) / (n - 1)) for i in range(n)]
    return [seq[i] for i in idx]


def label_line(ax, x, y, text, color, dx=1.02, va="center", fontsize=8):
    """Direct label at the right end of a line -- the relief for low-contrast hues."""
    ax.annotate(text, xy=(x, y), xytext=(4, 0), textcoords="offset points",
                color=color, fontsize=fontsize, va=va, ha="left", clip_on=False)


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
