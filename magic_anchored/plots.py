r"""Figures for the ``alpha`` comparison.

``alpha`` is an *ordered* quantity, so every figure encodes it with a single-hue
sequential ramp (light = reversible baseline, dark = strongest rotation) rather
than with categorical hues: an ordered variable drawn in unrelated colours makes
the reader recover the order from a legend instead of reading it off the ink,
and a one-hue ramp is colour-vision-safe by construction.  The reversible
baseline ``alpha = 0`` is additionally dashed, so it is identifiable without
colour at all.

Every figure is written twice, as a high-resolution PNG and as a vector PDF.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from diagnostics import ChainDiagnostics, PredictiveScores  # noqa: E402
from sampler import SamplerResult  # noqa: E402

__all__ = ["alpha_colours", "save", "make_all_figures"]

INK, MUTED, GRID = "#1f2328", "#6b7280", "#d8dade"
DPI = 300


def alpha_colours(alphas: Sequence[float]) -> dict[float, str]:
    """One sequential-ramp colour per ``alpha``, light (small) to dark (large)."""
    ramp = plt.get_cmap("Blues")
    order = sorted(set(float(a) for a in alphas))
    if len(order) == 1:
        return {order[0]: matplotlib.colors.to_hex(ramp(0.75))}
    positions = np.linspace(0.40, 0.95, len(order))
    return {
        a: matplotlib.colors.to_hex(ramp(p)) for a, p in zip(order, positions)
    }


def _style(ax, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c7cad0")
    ax.tick_params(colors=MUTED, labelsize=8)


def _line_kwargs(alpha: float, colours: Mapping[float, str]) -> dict:
    """Colour from the ramp; the baseline is dashed so colour is never the only cue."""
    return {
        "color": colours[float(alpha)],
        "linestyle": (0, (5, 2)) if alpha == 0.0 else "-",
        "label": rf"$\alpha = {alpha:g}$" + (" (reversible)" if alpha == 0 else ""),
    }


def save(fig, out_dir: str, name: str) -> list[str]:
    """Write ``name`` as both PNG and PDF, returning the paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI)
        paths.append(path)
    plt.close(fig)
    return paths


def _autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation of a 1-D series, lags ``0 .. max_lag``."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    n_fft = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, n=n_fft)
    ac = np.fft.irfft(f * np.conjugate(f), n=n_fft)[: max_lag + 1].real
    return ac / ac[0] if ac[0] > 0 else np.zeros(max_lag + 1)


def make_all_figures(
    chains: Mapping[float, Sequence[SamplerResult]],
    diags: Mapping[float, ChainDiagnostics],
    scores: Mapping[float, PredictiveScores],
    param_names: Sequence[str],
    center: np.ndarray,
    radius: float,
    out_dir: str,
    pair_coefficients: tuple[int, int, int] = (1, 2, 9),
    max_lag: int = 200,
    alpha_star: float | None = None,
    max_scatter: int = 4_000,
) -> list[str]:
    """Produce the ten requested figures and return every path written."""
    alphas = sorted(chains)
    colours = alpha_colours(alphas)
    names = list(param_names)
    D = len(names)
    written: list[str] = []
    center = np.asarray(center, dtype=np.float64).ravel()

    # 1. trace plots -- first chain of each alpha, a grid over coefficients
    n_col = 3
    n_row = int(np.ceil(D / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.2 * n_col, 1.9 * n_row),
                             layout="constrained", squeeze=False)
    for j in range(D):
        ax = axes[j // n_col][j % n_col]
        _style(ax)
        for a in alphas:
            r = chains[a][0]
            ax.plot(r.samples[:, j], lw=0.55, alpha=0.9, **_line_kwargs(a, colours))
        ax.set_title(names[j], fontsize=9, color=INK)
        ax.set_xlabel("retained draw", fontsize=7.5, color=MUTED)
    for j in range(D, n_row * n_col):
        axes[j // n_col][j % n_col].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(alphas),
               frameon=False, fontsize=9)
    fig.suptitle("Trace plots, first chain of each $\\alpha$", fontsize=12, color=INK)
    written += save(fig, out_dir, "01_traces")

    # 2. autocorrelation
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.2 * n_col, 1.9 * n_row),
                             layout="constrained", squeeze=False)
    for j in range(D):
        ax = axes[j // n_col][j % n_col]
        _style(ax)
        ax.axhline(0.0, color=INK, lw=0.8)
        for a in alphas:
            r = chains[a][0]
            lag = min(max_lag, max(r.samples.shape[0] // 4, 1))
            ac = _autocorrelation(r.samples[:, j], lag)
            ax.plot(np.arange(ac.size), ac, lw=1.0, **_line_kwargs(a, colours))
        ax.set_title(names[j], fontsize=9, color=INK)
        ax.set_xlabel("lag (retained draws)", fontsize=7.5, color=MUTED)
    for j in range(D, n_row * n_col):
        axes[j // n_col][j % n_col].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(alphas),
               frameon=False, fontsize=9)
    fig.suptitle("Autocorrelation of the retained draws", fontsize=12, color=INK)
    written += save(fig, out_dir, "02_autocorrelation")

    # 3. coefficient posteriors -- all chains pooled per alpha
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.2 * n_col, 1.9 * n_row),
                             layout="constrained", squeeze=False)
    for j in range(D):
        ax = axes[j // n_col][j % n_col]
        _style(ax)
        for a in alphas:
            pooled = np.concatenate([r.samples[:, j] for r in chains[a]])
            kw = _line_kwargs(a, colours)
            ax.hist(pooled, bins=60, density=True, histtype="step", lw=1.1,
                    color=kw["color"], linestyle=kw["linestyle"], label=kw["label"])
        ax.set_title(names[j], fontsize=9, color=INK)
    for j in range(D, n_row * n_col):
        axes[j // n_col][j % n_col].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(alphas),
               frameon=False, fontsize=9)
    fig.suptitle("Marginal posteriors, all chains pooled\n"
                 "(these should agree across $\\alpha$ -- they target the same law)",
                 fontsize=12, color=INK)
    written += save(fig, out_dir, "03_coefficient_posteriors")

    # 4. ESS by coefficient  &  6. R-hat by coefficient
    for name, attr, title, ref in (
        ("04_ess_by_coefficient", "ess", "Effective sample size by coefficient", None),
        ("06_rhat_by_coefficient", "r_hat", "Split $\\hat{R}$ by coefficient", 1.01),
    ):
        fig, ax = plt.subplots(figsize=(10.5, 4.6), layout="constrained")
        _style(ax)
        x = np.arange(D)
        width = 0.8 / len(alphas)
        for k, a in enumerate(alphas):
            ax.bar(x + (k - (len(alphas) - 1) / 2) * width,
                   getattr(diags[a], attr), width * 0.92,
                   color=colours[a], label=rf"$\alpha = {a:g}$",
                   hatch="//" if a == 0.0 else None, edgecolor="white", linewidth=0.0)
        if ref is not None:
            ax.axhline(ref, color="#C2410C", lw=1.2, ls=(0, (5, 3)),
                       label=f"threshold {ref}")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylabel(attr.upper() if attr == "ess" else "split $\\hat{R}$",
                      color=INK, fontsize=10)
        ax.set_title(title, fontsize=12, color=INK, pad=8)
        ax.legend(frameon=False, fontsize=8.5, ncols=3)
        written += save(fig, out_dir, name)

    # 5. ESS per second versus alpha
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), layout="constrained")
    for ax, key, label in (
        (axes[0], "ess_per_second", "total ESS per second"),
        (axes[1], "min_ess", "minimum ESS over coefficients"),
        (axes[2], "mean_squared_jump", "mean squared jumping distance"),
    ):
        _style(ax, grid_axis="both")
        values = [getattr(diags[a], key) for a in alphas]
        ax.plot(alphas, values, marker="o", color="#2B5FD9", markersize=7,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        # alpha spans three decades, so a linear axis crushes every value the
        # specification named into the left edge; symlog keeps alpha = 0 on it
        ax.set_xscale("symlog", linthresh=0.25, linscale=0.4)
        ax.set_xticks(alphas)
        ax.set_xticklabels([f"{a:g}" for a in alphas], fontsize=8)
        ax.minorticks_off()
        base = values[0]
        for i, (a, v) in enumerate(zip(alphas, values)):
            ax.annotate(f"{v / base:.2f}x" if base else "--", xy=(a, v),
                        xytext=(0, 11 if i % 2 == 0 else -15),
                        textcoords="offset points",
                        ha="center", fontsize=8, color=MUTED)
        if alpha_star is not None:
            ax.axvline(alpha_star, color="#C2410C", lw=1.1, ls=(0, (4, 3)),
                       zorder=2)
        ax.set_xlabel(r"$\alpha$   (symlog)", color=INK, fontsize=10)
        ax.set_ylabel(label, color=INK, fontsize=9.5)
    axes[0].set_title("Throughput against the reversible baseline",
                      fontsize=11.5, color=INK)
    note = ("labels are ratios to $\\alpha = 0$"
            + (rf"; the dashed line is $\alpha^* = {alpha_star:.1f}$, where the "
               r"rotation matches the gradient" if alpha_star else ""))
    fig.suptitle("Efficiency versus the non-reversible strength $\\alpha$\n" + note,
                 fontsize=12, color=INK)
    written += save(fig, out_dir, "05_efficiency_vs_alpha")

    # 7. radius against iteration, and 8. its histogram
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.0, 4.4), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.6, 1.0]})
    _style(ax)
    for a in alphas:
        r = chains[a][0]
        step = max(1, r.radius_trace.size // 4000)
        kw = _line_kwargs(a, colours)
        ax.plot(np.arange(0, r.radius_trace.size, step),
                r.radius_trace[::step], lw=0.5, alpha=0.75, **kw)
    ax.axhline(radius, color="#C2410C", lw=1.3, ls=(0, (5, 3)),
               label=f"constraint $R = {radius:.4f}$")
    ax.axvline(chains[alphas[0]][0].burn_in, color=MUTED, lw=0.9, ls=":",
               label="end of burn-in")
    ax.set_xlabel("iteration", color=INK, fontsize=10)
    ax.set_ylabel(r"$\|w_k - w_{\mathrm{center}}\|_2$", color=INK, fontsize=10)
    ax.set_title("Distance from the constraint centre", fontsize=11.5, color=INK)
    ax.legend(frameon=False, fontsize=7.5, ncols=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.14))
    _style(bx)
    for a in alphas:
        pooled = np.concatenate([r.radius_trace[r.burn_in :] for r in chains[a]])
        kw = _line_kwargs(a, colours)
        bx.hist(pooled, bins=70, density=True, histtype="step", lw=1.1,
                color=kw["color"], linestyle=kw["linestyle"], label=kw["label"])
    bx.axvline(radius, color="#C2410C", lw=1.3, ls=(0, (5, 3)))
    bx.set_xlabel(r"$\|w - w_{\mathrm{center}}\|_2$", color=INK, fontsize=10)
    bx.set_title("Retained distances, all chains", fontsize=11.5, color=INK)
    written += save(fig, out_dir, "07_08_radius_trace_and_histogram")

    # 9. projection frequency, and the clock, by alpha
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 4.2), layout="constrained")
    _style(ax)
    ax.bar([str(a) for a in alphas],
           [diags[a].projection_frequency for a in alphas],
           color=[colours[a] for a in alphas], width=0.62)
    ax.axhline(0.10, color="#C2410C", lw=1.2, ls=(0, (5, 3)),
               label="10% warning threshold")
    for k, a in enumerate(alphas):
        ax.annotate(f"{diags[a].projection_frequency:.2%}", xy=(k, diags[a].projection_frequency),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=8.5, color=MUTED)
    ax.set_xlabel(r"$\alpha$", color=INK, fontsize=10)
    ax.set_ylabel("fraction of proposals projected", color=INK, fontsize=10)
    ax.set_title("How often the constraint binds", fontsize=11.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5)
    _style(bx)
    for a in alphas:
        pooled = np.concatenate([r.a for r in chains[a]])
        kw = _line_kwargs(a, colours)
        bx.hist(pooled, bins=70, density=True, histtype="step", lw=1.1,
                color=kw["color"], linestyle=kw["linestyle"], label=kw["label"])
    bx.set_xlabel(r"anchoring clock $a(w)$", color=INK, fontsize=10)
    bx.set_title(r"The clock $a(w) = e^{U - U_0} \in (0, 1]$",
                 fontsize=11.5, color=INK)
    bx.legend(frameon=False, fontsize=8)
    written += save(fig, out_dir, "09_projection_and_clock")

    # 10. pair plots for selected coefficients
    picks = [j for j in pair_coefficients if j < D][:3]
    if len(picks) >= 2:
        pairs = [(picks[i], picks[j]) for i in range(len(picks))
                 for j in range(i + 1, len(picks))]
        # the joint shape is a property of the target, so it should be the same
        # for every alpha; drawing all of them is eight copies of one picture.
        # The baseline and the strongest rotation are what a reader needs to
        # compare, and any disagreement between them is the thing to see.
        shown = [alphas[0], alphas[-1]] if len(alphas) > 1 else list(alphas)
        # shared axes per row, because the claim being checked is that the two
        # columns coincide -- with independent limits they always look similar
        fig, axes = plt.subplots(len(pairs), len(shown),
                                 figsize=(3.4 * len(shown), 3.2 * len(pairs)),
                                 layout="constrained", squeeze=False,
                                 sharex="row", sharey="row")
        for row, (j1, j2) in enumerate(pairs):
            for col, a in enumerate(shown):
                ax = axes[row][col]
                _style(ax, grid_axis="both")
                pooled = np.concatenate([r.samples for r in chains[a]], axis=0)
                # a scatter of tens of thousands of draws is saturated ink and a
                # very large vector file; a fixed-seed thinning of the pooled
                # draws shows the same cloud
                if pooled.shape[0] > max_scatter:
                    take = np.random.default_rng(0).choice(
                        pooled.shape[0], max_scatter, replace=False
                    )
                    pooled = pooled[np.sort(take)]
                ax.scatter(pooled[:, j1], pooled[:, j2], s=2.5, alpha=0.3,
                           color=colours[a], linewidths=0, rasterized=True)
                if row == 0:
                    ax.set_title(rf"$\alpha = {a:g}$", fontsize=10, color=INK)
                if col == 0:
                    ax.set_ylabel(names[j2], fontsize=9, color=INK)
                ax.set_xlabel(names[j1], fontsize=9, color=MUTED)
        fig.suptitle("Joint posteriors for selected coefficients:\n"
                     "the reversible baseline against the strongest rotation "
                     "(these should coincide)", fontsize=12, color=INK)
        written += save(fig, out_dir, "10_pair_plots")

    # predictive calibration, since the scores are computed anyway
    if scores:
        fig, ax = plt.subplots(figsize=(6.2, 5.6), layout="constrained")
        _style(ax, grid_axis="both")
        ax.plot([0, 1], [0, 1], color=INK, lw=1.1, ls=(0, (5, 3)),
                label="perfect calibration")
        for a in alphas:
            if a not in scores:
                continue
            sc = scores[a]
            kw = _line_kwargs(a, colours)
            ax.plot(sc.calibration_predicted, sc.calibration_observed,
                    marker="o", markersize=5, lw=1.2, markeredgecolor="white",
                    markeredgewidth=0.8, **kw)
        ax.set_xlabel("mean predicted probability", color=INK, fontsize=10)
        ax.set_ylabel("observed frequency", color=INK, fontsize=10)
        ax.set_title("Test-set calibration of the posterior predictive",
                     fontsize=11.5, color=INK)
        ax.legend(frameon=False, fontsize=8.5)
        written += save(fig, out_dir, "11_calibration")

    return written
