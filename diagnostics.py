"""Convergence / efficiency diagnostics and every figure requested in §10.

ESS and R-hat are taken from ArviZ.  The integrated autocorrelation time is
recovered from the effective sample size via the standard identity

    ESS = (n_chains * n_draws) / tau,      tau = 1 + 2 * sum_{l>=1} rho(l),

so ``tau`` is reported both in units of *stored draws* and, after multiplying
by the thinning interval, in units of *sampler iterations*.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sampler import ChainOutput, stack_samples

sns.set_theme(context="notebook", style="whitegrid", palette="colorblind")

#: Colour map shared by every alpha-indexed figure.
ALPHA_PALETTE = "viridis"


def coefficient_names(d: int) -> list[str]:
    """``['intercept', 'x1', ..., 'x8']``."""
    return ["intercept"] + [f"x{j}" for j in range(1, d)]


def to_inference_data(
    samples: np.ndarray,
    names: Sequence[str] | None = None,
) -> az.InferenceData:
    """Wrap a ``(n_chains, n_draws, d)`` array as ArviZ ``InferenceData``."""
    samples = np.asarray(samples, dtype=float)
    if names is None:
        names = coefficient_names(samples.shape[-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return az.from_dict(
            posterior={"w": samples},
            coords={"coef": list(names)},
            dims={"w": ["coef"]},
        )


def mean_squared_jumping_distance(chains: Sequence[ChainOutput], burn_in: int) -> float:
    """``mean_k ||w_{k+1} - w_k||^2`` over all post-burn-in, *unthinned* steps."""
    values = []
    for chain in chains:
        post = chain.trace[burn_in:]
        if post.shape[0] < 2:
            continue
        steps = np.diff(post, axis=0)
        values.append(np.sum(steps ** 2, axis=1))
    if not values:
        return float("nan")
    return float(np.concatenate(values).mean())


@dataclass
class AlphaSummary:
    """Everything reported for one value of ``alpha``."""

    alpha: float
    step_size: float
    swirl: float
    n_chains: int
    n_draws: int
    runtime: float
    ess: np.ndarray               # (d,) bulk ESS per coordinate
    ess_tail: np.ndarray          # (d,)
    rhat_split: np.ndarray        # (d,)
    rhat_rank: np.ndarray         # (d,)
    iat_draws: np.ndarray         # (d,) integrated autocorrelation time
    iat_iterations: np.ndarray    # (d,) same, in sampler iterations
    posterior_mean: np.ndarray    # (d,)
    posterior_sd: np.ndarray      # (d,)
    msjd: float
    projection_frequency: float
    mean_projection_distance: float
    mean_operator_norm_J: float
    max_operator_norm_J: float
    mean_a: float
    min_a: float
    mean_slack: float
    min_slack: float
    names: Sequence[str]

    @property
    def min_ess(self) -> float:
        return float(np.min(self.ess))

    @property
    def median_ess(self) -> float:
        return float(np.median(self.ess))

    @property
    def ess_per_second(self) -> float:
        """Minimum ESS per second of total sampling wall-clock time."""
        return self.min_ess / self.runtime if self.runtime > 0 else float("nan")

    @property
    def median_ess_per_second(self) -> float:
        return self.median_ess / self.runtime if self.runtime > 0 else float("nan")

    @property
    def max_rhat(self) -> float:
        return float(np.max(self.rhat_split))

    def to_row(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "step_size": self.step_size,
            "swirl": self.swirl,
            "n_chains": self.n_chains,
            "n_draws": self.n_draws,
            "runtime_s": self.runtime,
            "min_ess": self.min_ess,
            "median_ess": self.median_ess,
            "min_ess_per_s": self.ess_per_second,
            "median_ess_per_s": self.median_ess_per_second,
            "max_split_rhat": self.max_rhat,
            "median_iat_iterations": float(np.median(self.iat_iterations)),
            "msjd": self.msjd,
            "projection_frequency": self.projection_frequency,
            "mean_projection_distance": self.mean_projection_distance,
            "mean_operator_norm_J": self.mean_operator_norm_J,
            "max_operator_norm_J": self.max_operator_norm_J,
            "mean_a": self.mean_a,
            "min_a": self.min_a,
            "mean_slack": self.mean_slack,
            "min_slack": self.min_slack,
        }


def summarise_alpha(
    chains: Sequence[ChainOutput],
    burn_in: int,
    thin: int,
    names: Sequence[str] | None = None,
) -> AlphaSummary:
    """Compute every §10 per-alpha diagnostic from a list of chains."""
    samples = stack_samples(list(chains))
    d = samples.shape[-1]
    if names is None:
        names = coefficient_names(d)

    idata = to_inference_data(samples, names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ess = np.asarray(az.ess(idata, method="bulk")["w"].values, dtype=float)
        ess_tail = np.asarray(az.ess(idata, method="tail")["w"].values, dtype=float)
        rhat_split = np.asarray(
            az.rhat(idata, method="split")["w"].values, dtype=float
        )
        rhat_rank = np.asarray(az.rhat(idata, method="rank")["w"].values, dtype=float)

    n_chains, n_draws = samples.shape[0], samples.shape[1]
    total_draws = n_chains * n_draws
    with np.errstate(divide="ignore", invalid="ignore"):
        iat_draws = total_draws / ess
    iat_iterations = iat_draws * thin

    # Pool the post-burn-in per-iteration diagnostics across chains.
    def pooled(key: str) -> np.ndarray:
        parts = [c.diagnostics[key][burn_in:] for c in chains if key in c.diagnostics]
        return np.concatenate(parts) if parts else np.array([np.nan])

    projected = pooled("projected")
    projection_distance = pooled("projection_distance")
    operator_norm = pooled("operator_norm_J")
    anchor = pooled("a")
    slack = pooled("slack")

    projection_frequency = float(np.mean(projected))
    if np.any(projected):
        mean_projection_distance = float(np.mean(projection_distance[projected.astype(bool)]))
    else:
        mean_projection_distance = 0.0

    flat = samples.reshape(-1, d)
    return AlphaSummary(
        alpha=float(chains[0].alpha),
        step_size=float(chains[0].step_size),
        swirl=float(chains[0].swirl),
        n_chains=n_chains,
        n_draws=n_draws,
        runtime=float(sum(c.runtime for c in chains)),
        ess=ess,
        ess_tail=ess_tail,
        rhat_split=rhat_split,
        rhat_rank=rhat_rank,
        iat_draws=iat_draws,
        iat_iterations=iat_iterations,
        posterior_mean=flat.mean(axis=0),
        posterior_sd=flat.std(axis=0, ddof=1),
        msjd=mean_squared_jumping_distance(chains, burn_in),
        projection_frequency=projection_frequency,
        mean_projection_distance=mean_projection_distance,
        mean_operator_norm_J=float(np.mean(operator_norm)),
        max_operator_norm_J=float(np.max(operator_norm)),
        mean_a=float(np.mean(anchor)),
        min_a=float(np.min(anchor)),
        mean_slack=float(np.mean(slack)),
        min_slack=float(np.min(slack)),
        names=list(names),
    )


def summary_table(summaries: Sequence[AlphaSummary]) -> pd.DataFrame:
    """One row per alpha with the headline efficiency metrics."""
    return pd.DataFrame([s.to_row() for s in summaries])


def per_coefficient_table(summaries: Sequence[AlphaSummary]) -> pd.DataFrame:
    """Long-format table: ESS, R-hat, IAT and posterior moments per coefficient."""
    rows = []
    for summary in summaries:
        for j, name in enumerate(summary.names):
            rows.append(
                {
                    "alpha": summary.alpha,
                    "coefficient": name,
                    "ess_bulk": summary.ess[j],
                    "ess_tail": summary.ess_tail[j],
                    "ess_per_s": summary.ess[j] / summary.runtime,
                    "split_rhat": summary.rhat_split[j],
                    "rank_rhat": summary.rhat_rank[j],
                    "iat_draws": summary.iat_draws[j],
                    "iat_iterations": summary.iat_iterations[j],
                    "posterior_mean": summary.posterior_mean[j],
                    "posterior_sd": summary.posterior_sd[j],
                }
            )
    return pd.DataFrame(rows)


# ==========================================================================
# Figures
# ==========================================================================
def _save(fig: plt.Figure, output_dir: str, filename: str) -> str:
    """Tidy the layout and write the figure to ``output_dir/filename``."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with warnings.catch_warnings():
        # tight_layout is noisy about supxlabel/supylabel on some backends.
        warnings.simplefilter("ignore")
        fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _alpha_colors(alphas: Sequence[float]) -> list:
    cmap = plt.get_cmap(ALPHA_PALETTE)
    if len(alphas) == 1:
        return [cmap(0.5)]
    return [cmap(i / (len(alphas) - 1) * 0.9) for i in range(len(alphas))]


def plot_traces(
    chains_by_alpha: Mapping[float, Sequence[ChainOutput]],
    burn_in: int,
    output_dir: str,
    names: Sequence[str] | None = None,
    max_points: int = 4000,
) -> list[str]:
    """(1) Trace plots: every coefficient, all chains overlaid, per alpha."""
    paths = []
    for alpha, chains in chains_by_alpha.items():
        d = chains[0].d
        labels = list(names or coefficient_names(d))
        fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
        for j, ax in enumerate(axes.ravel()[:d]):
            for c, chain in enumerate(chains):
                series = chain.trace[:, j]
                stride = max(1, series.size // max_points)
                ax.plot(
                    np.arange(0, series.size, stride),
                    series[::stride],
                    lw=0.5,
                    alpha=0.8,
                    label=f"chain {c}" if j == 0 else None,
                )
            ax.axvline(burn_in, color="k", ls="--", lw=0.8)
            ax.set_title(labels[j], fontsize=10)
        axes.ravel()[0].legend(fontsize=7, loc="upper right")
        fig.suptitle(f"Trace plots — alpha = {alpha} (dashed line: end of burn-in)")
        fig.supxlabel("iteration")
        paths.append(_save(fig, output_dir, f"01_trace_alpha_{alpha}.png"))
    return paths


def plot_autocorrelation(
    summaries_samples: Mapping[float, np.ndarray],
    output_dir: str,
    names: Sequence[str] | None = None,
    max_lag: int = 100,
) -> str:
    """(2) Autocorrelation of every coefficient, one panel per coefficient."""
    alphas = list(summaries_samples.keys())
    any_samples = next(iter(summaries_samples.values()))
    d = any_samples.shape[-1]
    labels = list(names or coefficient_names(d))
    colors = _alpha_colors(alphas)

    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True, sharey=True)
    for j, ax in enumerate(axes.ravel()[:d]):
        for alpha, color in zip(alphas, colors):
            samples = summaries_samples[alpha][..., j]
            acf = np.mean(
                [_autocorrelation(chain, max_lag) for chain in samples], axis=0
            )
            ax.plot(acf, color=color, lw=1.4, label=f"alpha={alpha}")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_title(labels[j], fontsize=10)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("Autocorrelation of the retained draws (averaged over chains)")
    fig.supxlabel("lag (in stored draws)")
    fig.supylabel("autocorrelation")
    return _save(fig, output_dir, "02_autocorrelation.png")


def _autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation function via FFT."""
    series = np.asarray(series, dtype=float)
    series = series - series.mean()
    n = series.size
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(series, size)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[:n].real
    if acf[0] == 0.0:
        return np.zeros(min(max_lag + 1, n))
    acf /= acf[0]
    return acf[: min(max_lag + 1, n)]


def plot_ess_by_coefficient(
    summaries: Sequence[AlphaSummary],
    output_dir: str,
) -> str:
    """(3) ESS per coefficient, grouped bars over alpha."""
    table = per_coefficient_table(summaries)
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(
        data=table, x="coefficient", y="ess_bulk", hue="alpha", ax=ax, palette=ALPHA_PALETTE
    )
    ax.set_ylabel("bulk ESS")
    ax.set_title("Effective sample size by coefficient")
    ax.legend(title="alpha", fontsize=8)
    return _save(fig, output_dir, "03_ess_by_coefficient.png")


def plot_ess_per_second(summaries: Sequence[AlphaSummary], output_dir: str) -> str:
    """(4) ESS per second versus alpha (minimum and median over coordinates)."""
    alphas = [s.alpha for s in summaries]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas, [s.ess_per_second for s in summaries], "o-", label="min ESS / s")
    ax.plot(
        alphas,
        [s.median_ess_per_second for s in summaries],
        "s--",
        label="median ESS / s",
    )
    ax.set_xlabel("alpha (non-reversibility strength)")
    ax.set_ylabel("ESS per second")
    ax.set_title("Sampling efficiency versus alpha")
    ax.legend()
    return _save(fig, output_dir, "04_ess_per_second.png")


def plot_rhat(summaries: Sequence[AlphaSummary], output_dir: str) -> str:
    """(5) Split R-hat per coefficient and alpha."""
    table = per_coefficient_table(summaries)
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(
        data=table, x="coefficient", y="split_rhat", hue="alpha", ax=ax, palette=ALPHA_PALETTE
    )
    ax.axhline(1.01, color="crimson", ls="--", lw=1.0, label="1.01 threshold")
    ax.set_ylim(0.99, max(1.02, float(table["split_rhat"].max()) * 1.01))
    ax.set_ylabel("split R-hat")
    ax.set_title("Split R-hat by coefficient")
    ax.legend(title="alpha", fontsize=8)
    return _save(fig, output_dir, "05_rhat.png")


def plot_posterior_intervals(
    summaries_samples: Mapping[float, np.ndarray],
    beta_true: np.ndarray,
    output_dir: str,
    names: Sequence[str] | None = None,
    credible_mass: float = 0.95,
) -> str:
    """(6) Posterior coefficient intervals with the ground truth overlaid."""
    alphas = list(summaries_samples.keys())
    d = beta_true.size
    labels = list(names or coefficient_names(d))
    colors = _alpha_colors(alphas)
    offsets = np.linspace(-0.3, 0.3, len(alphas))

    fig, ax = plt.subplots(figsize=(12, 6))
    lower_q = (1.0 - credible_mass) / 2.0
    for (alpha, color, offset) in zip(alphas, colors, offsets):
        flat = summaries_samples[alpha].reshape(-1, d)
        lo = np.quantile(flat, lower_q, axis=0)
        hi = np.quantile(flat, 1.0 - lower_q, axis=0)
        mean = flat.mean(axis=0)
        positions = np.arange(d) + offset
        ax.errorbar(
            positions,
            mean,
            yerr=np.vstack([mean - lo, hi - mean]),
            fmt="o",
            color=color,
            capsize=3,
            ms=4,
            lw=1.2,
            label=f"alpha={alpha}",
        )
    ax.plot(np.arange(d), beta_true, "kx", ms=11, mew=2, label="beta_true")
    ax.axhline(0.0, color="grey", lw=0.7)
    ax.set_xticks(np.arange(d))
    ax.set_xticklabels(labels)
    ax.set_ylabel("coefficient")
    ax.set_title(f"Posterior {int(credible_mass * 100)}% intervals")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, output_dir, "06_posterior_intervals.png")


def plot_g_histogram(
    chains_by_alpha: Mapping[float, Sequence[ChainOutput]],
    burn_in: int,
    Lambda_constraint: float,
    g_at_origin: float,
    output_dir: str,
) -> str:
    """(7) Histogram of ``g(w)`` along the retained portion of every chain."""
    alphas = list(chains_by_alpha.keys())
    colors = _alpha_colors(alphas)
    fig, ax = plt.subplots(figsize=(9, 5))
    for alpha, color in zip(alphas, colors):
        values = np.concatenate(
            [c.diagnostics["g"][burn_in:] for c in chains_by_alpha[alpha]]
        )
        ax.hist(values, bins=80, histtype="step", lw=1.5, color=color, label=f"alpha={alpha}")
    ax.axvline(Lambda_constraint, color="crimson", ls="--", label="Lambda_constraint")
    ax.axvline(g_at_origin, color="grey", ls=":", label="g(0) = d*eps^p")
    ax.set_xlabel("g(w)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of the constraint function along the chains")
    ax.legend(fontsize=8)
    return _save(fig, output_dir, "07_g_histogram.png")


def plot_distance_to_boundary(
    chains_by_alpha: Mapping[float, Sequence[ChainOutput]],
    burn_in: int,
    output_dir: str,
    max_points: int = 4000,
) -> str:
    """(8) Slack ``Lambda_constraint - g(w)`` over time and its distribution."""
    alphas = list(chains_by_alpha.keys())
    colors = _alpha_colors(alphas)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for alpha, color in zip(alphas, colors):
        series = chains_by_alpha[alpha][0].diagnostics["slack"]
        stride = max(1, series.size // max_points)
        axes[0].plot(
            np.arange(0, series.size, stride),
            series[::stride],
            lw=0.7,
            color=color,
            label=f"alpha={alpha}",
        )
        pooled = np.concatenate(
            [c.diagnostics["slack"][burn_in:] for c in chains_by_alpha[alpha]]
        )
        axes[1].hist(pooled, bins=60, histtype="step", lw=1.5, color=color)
    axes[0].axvline(burn_in, color="k", ls="--", lw=0.8)
    axes[0].axhline(0.0, color="crimson", ls="--", lw=1.0)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("Lambda_constraint - g(w)")
    axes[0].set_title("Distance to the boundary (chain 0)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Lambda_constraint - g(w)")
    axes[1].set_title("Distance to the boundary (post burn-in, all chains)")
    return _save(fig, output_dir, "08_distance_to_boundary.png")


def plot_projection_frequency(
    summaries: Sequence[AlphaSummary],
    chains_by_alpha: Mapping[float, Sequence[ChainOutput]],
    burn_in: int,
    output_dir: str,
    window: int = 500,
) -> str:
    """(9) Projection frequency by alpha and its running average over time."""
    alphas = [s.alpha for s in summaries]
    colors = _alpha_colors(alphas)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(
        [str(a) for a in alphas],
        [s.projection_frequency for s in summaries],
        color=colors,
    )
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("fraction of iterations projected")
    axes[0].set_title("Projection frequency (post burn-in)")

    for alpha, color in zip(alphas, colors):
        flags = np.concatenate(
            [c.diagnostics["projected"].astype(float) for c in chains_by_alpha[alpha][:1]]
        )
        if flags.size >= window:
            kernel = np.ones(window) / window
            running = np.convolve(flags, kernel, mode="valid")
            axes[1].plot(running, lw=1.0, color=color, label=f"alpha={alpha}")
    axes[1].axvline(burn_in, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(f"projection rate ({window}-iteration window)")
    axes[1].set_title("Running projection rate (chain 0)")
    axes[1].legend(fontsize=8)
    return _save(fig, output_dir, "09_projection_frequency.png")


def plot_operator_norm(
    chains_by_alpha: Mapping[float, Sequence[ChainOutput]],
    burn_in: int,
    output_dir: str,
    max_points: int = 4000,
) -> str:
    """(10) ``||J(w)||_2`` over the course of the simulation, plus drift norms."""
    alphas = list(chains_by_alpha.keys())
    colors = _alpha_colors(alphas)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for alpha, color in zip(alphas, colors):
        series = chains_by_alpha[alpha][0].diagnostics["operator_norm_J"]
        stride = max(1, series.size // max_points)
        axes[0].plot(
            np.arange(0, series.size, stride),
            series[::stride],
            lw=0.7,
            color=color,
            label=f"alpha={alpha}",
        )
        rev = np.concatenate(
            [c.diagnostics["reversible_drift_norm"][burn_in:] for c in chains_by_alpha[alpha]]
        )
        nonrev = np.concatenate(
            [
                c.diagnostics["nonreversible_drift_norm"][burn_in:]
                for c in chains_by_alpha[alpha]
            ]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(rev > 0, nonrev / rev, 0.0)
        axes[1].plot(alpha, float(np.mean(ratio)), "o", color=color, ms=8)
    axes[0].axvline(burn_in, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("||J(w)||_2")
    axes[0].set_title("Operator norm of J along the chain (chain 0)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("mean ||non-reversible drift|| / ||reversible drift||")
    axes[1].set_title("Relative size of the non-reversible term")
    return _save(fig, output_dir, "10_operator_norm_J.png")


def plot_pairs(
    summaries_samples: Mapping[float, np.ndarray],
    output_dir: str,
    coefficient_pairs: Sequence[tuple[int, int]] = ((1, 2), (2, 3), (1, 3)),
    names: Sequence[str] | None = None,
) -> str:
    """(11) Two-dimensional pair plots for selected coefficients."""
    alphas = list(summaries_samples.keys())
    d = next(iter(summaries_samples.values())).shape[-1]
    labels = list(names or coefficient_names(d))
    n_rows, n_cols = len(coefficient_pairs), len(alphas)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.4 * n_cols, 3.2 * n_rows), squeeze=False
    )
    for r, (i, j) in enumerate(coefficient_pairs):
        for c, alpha in enumerate(alphas):
            flat = summaries_samples[alpha].reshape(-1, d)
            ax = axes[r][c]
            ax.hexbin(flat[:, i], flat[:, j], gridsize=40, cmap="magma", mincnt=1)
            if r == 0:
                ax.set_title(f"alpha = {alpha}", fontsize=10)
            if c == 0:
                ax.set_ylabel(labels[j])
            ax.set_xlabel(labels[i])
    fig.suptitle("Posterior pair plots")
    return _save(fig, output_dir, "11_pair_plots.png")


def plot_step_size_sensitivity(table: pd.DataFrame, output_dir: str) -> str:
    """Step-size sensitivity: efficiency and accuracy at h, h/2, h/4."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for divisor, group in table.groupby("step_divisor"):
        label = f"h/{int(divisor)}" if divisor != 1 else "h"
        axes[0].plot(group["alpha"], group["min_ess_per_s"], "o-", label=label)
        axes[1].plot(group["alpha"], group["msjd"], "o-", label=label)
        axes[2].plot(group["alpha"], group["max_split_rhat"], "o-", label=label)
    axes[0].set_ylabel("min ESS / s")
    axes[1].set_ylabel("mean squared jumping distance")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("max split R-hat")
    for ax in axes:
        ax.set_xlabel("alpha")
        ax.legend(fontsize=8)
    fig.suptitle("Step-size sensitivity")
    return _save(fig, output_dir, "12_step_size_sensitivity.png")


# ==========================================================================
# Section 11: posterior prediction
# ==========================================================================
@dataclass
class PredictiveSummary:
    """Test-set performance of the posterior predictive mean probability."""

    alpha: float
    accuracy: float
    balanced_accuracy: float
    roc_auc: float
    log_loss: float
    brier_score: float
    sensitivity: float
    specificity: float
    confusion_matrix: np.ndarray
    probability_bar: np.ndarray
    posterior_mean: np.ndarray
    l2_error_vs_beta_true: float

    def to_row(self) -> dict[str, float]:
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        return {
            "alpha": self.alpha,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "roc_auc": self.roc_auc,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "l2_error_vs_beta_true": self.l2_error_vs_beta_true,
        }


def posterior_predictive(
    samples: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    beta_true: np.ndarray,
    alpha: float,
    threshold: float = 0.5,
) -> PredictiveSummary:
    """Posterior-predictive evaluation on the test set.

    For every retained draw ``w^(m)``,
    ``probability_i^(m) = sigmoid(X_test[i] @ w^(m))``, and the reported
    predictive probability is the posterior mean
    ``probability_bar_i = mean_m probability_i^(m)``.
    """
    from scipy.special import expit
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        log_loss,
        roc_auc_score,
    )

    flat = np.asarray(samples, dtype=float).reshape(-1, X_test.shape[1])
    # (n_draws, n_test) -> mean over draws.  Chunked to bound memory.
    chunk = max(1, int(2e7 // max(X_test.shape[0], 1)))
    accumulator = np.zeros(X_test.shape[0])
    for start in range(0, flat.shape[0], chunk):
        block = flat[start : start + chunk]
        accumulator += expit(block @ X_test.T).sum(axis=0)
    probability_bar = accumulator / flat.shape[0]

    predictions = (probability_bar >= threshold).astype(int)
    cm = confusion_matrix(y_test.astype(int), predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    posterior_mean = flat.mean(axis=0)
    return PredictiveSummary(
        alpha=alpha,
        accuracy=float(accuracy_score(y_test, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, predictions)),
        roc_auc=float(roc_auc_score(y_test, probability_bar)),
        log_loss=float(log_loss(y_test, probability_bar, labels=[0, 1])),
        brier_score=float(brier_score_loss(y_test, probability_bar)),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        confusion_matrix=cm,
        probability_bar=probability_bar,
        posterior_mean=posterior_mean,
        l2_error_vs_beta_true=float(np.linalg.norm(posterior_mean - beta_true)),
    )


def plot_calibration(
    predictives: Sequence[PredictiveSummary],
    y_test: np.ndarray,
    output_dir: str,
    n_bins: int = 10,
) -> str:
    """(12) Posterior-predictive calibration curves plus the probability histogram."""
    from sklearn.calibration import calibration_curve

    alphas = [p.alpha for p in predictives]
    colors = _alpha_colors(alphas)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.0, label="perfect calibration")
    for predictive, color in zip(predictives, colors):
        fraction, mean_predicted = calibration_curve(
            y_test, predictive.probability_bar, n_bins=n_bins, strategy="quantile"
        )
        axes[0].plot(
            mean_predicted, fraction, "o-", color=color, label=f"alpha={predictive.alpha}"
        )
        axes[1].hist(
            predictive.probability_bar,
            bins=40,
            histtype="step",
            lw=1.4,
            color=color,
            label=f"alpha={predictive.alpha}",
        )
    axes[0].set_xlabel("mean posterior predictive probability")
    axes[0].set_ylabel("observed frequency")
    axes[0].set_title("Calibration curve (test set)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("probability_bar")
    axes[1].set_ylabel("count")
    axes[1].set_title("Posterior mean predictive probabilities")
    axes[1].legend(fontsize=8)
    return _save(fig, output_dir, "13_calibration.png")


def plot_confusion(predictive: PredictiveSummary, output_dir: str) -> str:
    """Confusion-matrix heat map for one alpha."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        predictive.confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["pred 0", "pred 1"],
        yticklabels=["true 0", "true 1"],
        ax=ax,
    )
    ax.set_title(f"Confusion matrix (alpha = {predictive.alpha})")
    return _save(fig, output_dir, f"14_confusion_alpha_{predictive.alpha}.png")


# ==========================================================================
# Accuracy and loss curves along the chain
# ==========================================================================
# There is no epoch loop in MCMC, so "training curves" mean something specific
# here.  The loss is the potential itself: U(w) is the negative log posterior
# (up to a constant), and its likelihood part divided by n is the mean
# cross-entropy — the quantity an optimiser would call the training loss.  The
# accuracy is the plug-in accuracy of the state w_k.  Both are recorded at
# every iteration, so the first panels below are genuine per-iteration curves.
#
# A Bayesian run also has a second, more meaningful curve: the test performance
# of the *posterior-averaged* predictive probability as more draws are
# accumulated.  That is the curve that actually converges to the reported
# number, and it is what the right-hand panels show.

#: Two-colour split for train vs test. Validated colourblind-safe
#: (worst adjacent CVD deltaE 26.2, normal-vision 33.5).  The orange falls below
#: 3:1 contrast on a light surface, so both series also carry a direct label
#: and a distinct line style rather than relying on hue alone.
TRAIN_COLOR = "#0173B2"
TEST_COLOR = "#DE8F05"


def predictive_trace(
    trace: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    """Plug-in accuracy and mean cross-entropy at each (strided) state ``w_k``.

    Returns ``iteration``, ``accuracy`` and ``log_loss`` (nats per observation,
    i.e. the likelihood part of ``U`` divided by ``n``).
    """
    from scipy.special import expit

    states = np.asarray(trace, dtype=float)[::stride]
    linear = states @ X.T                                   # (n_states, n_obs)
    probability = expit(linear)
    accuracy = ((probability >= 0.5) == (y[None, :] >= 0.5)).mean(axis=1)
    log_loss = (np.logaddexp(0.0, linear) - y[None, :] * linear).mean(axis=1)
    return {
        "iteration": np.arange(0, states.shape[0] * stride, stride),
        "accuracy": accuracy,
        "log_loss": log_loss,
    }


def posterior_averaging_curve(
    samples: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_checkpoints: int = 60,
) -> dict[str, np.ndarray]:
    """Test accuracy / log loss of the posterior mean predictive vs draw count.

    Draws are interleaved across chains so that the curve reflects the pooled
    posterior at every checkpoint rather than one chain at a time.
    """
    from scipy.special import expit

    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 3:                                   # (chain, draw, d)
        samples = samples.transpose(1, 0, 2).reshape(-1, samples.shape[-1])
    n_draws = samples.shape[0]
    # Geometrically spaced: these curves are read on a log x-axis, and linear
    # spacing would leave the first two decades covered by a single segment.
    checkpoints = np.unique(
        np.geomspace(1, n_draws, n_checkpoints).round().astype(int)
    )

    accumulator = np.zeros(X_test.shape[0])
    accuracy = np.empty(checkpoints.size)
    log_loss = np.empty(checkpoints.size)
    previous = 0
    for index, checkpoint in enumerate(checkpoints):
        block = samples[previous:checkpoint]
        accumulator += expit(block @ X_test.T).sum(axis=0)
        previous = checkpoint
        probability_bar = np.clip(accumulator / checkpoint, 1e-12, 1 - 1e-12)
        accuracy[index] = ((probability_bar >= 0.5) == (y_test >= 0.5)).mean()
        log_loss[index] = -(
            y_test * np.log(probability_bar)
            + (1 - y_test) * np.log1p(-probability_bar)
        ).mean()
    return {"n_draws": checkpoints, "accuracy": accuracy, "log_loss": log_loss}


def _label_series(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    text: str,
    color: str,
    fraction: float,
) -> None:
    """Direct-label a line at a given fraction along it.

    Train and test curves often sit on top of each other here, so the two
    labels are placed at different x positions rather than both at the right
    edge, where they would collide.
    """
    index = min(int(fraction * (len(x) - 1)), len(x) - 1)
    ax.annotate(
        text,
        xy=(x[index], y[index]),
        xytext=(0, 7),
        textcoords="offset points",
        color=color,
        fontsize=9,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
    )


def plot_loss_curves(
    chains_by_alpha: Mapping[float, Sequence[ChainOutput]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    burn_in: int,
    output_dir: str,
    stride: int = 10,
) -> str:
    """Loss curves: the potential U / U0 and the train-vs-test cross-entropy."""
    alphas = list(chains_by_alpha.keys())
    colors = _alpha_colors(alphas)
    baseline = chains_by_alpha[alphas[0]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) U(w_k) for every chain at the baseline alpha: burn-in convergence.
    for c, chain in enumerate(baseline):
        series = chain.diagnostics["U"][::stride]
        axes[0][0].plot(
            np.arange(series.size) * stride, series, lw=0.8, label=f"chain {c}"
        )
    axes[0][0].axvline(burn_in, color="k", ls="--", lw=0.9)
    # Log x, not log y: the descent finishes within the first ~100 iterations
    # and is invisible on a linear iteration axis.
    axes[0][0].set_xscale("log")
    axes[0][0].set_xlabel("iteration (log scale)")
    axes[0][0].set_ylabel("U(w)  [negative log posterior]")
    axes[0][0].set_title(
        f"Loss along the chain, alpha = {alphas[0]}\n"
        "Burn-in descent from overdispersed starts (dashed: end of burn-in)",
        fontsize=10,
    )
    axes[0][0].legend(fontsize=8)

    # (b) Post burn-in U by alpha: the stationary level must be identical.
    for alpha, color in zip(alphas, colors):
        pooled = np.concatenate(
            [c.diagnostics["U"][burn_in:] for c in chains_by_alpha[alpha]]
        )
        axes[0][1].hist(
            pooled, bins=70, histtype="step", lw=1.6, color=color,
            density=True, label=f"alpha={alpha}",
        )
    axes[0][1].set_xlabel("U(w), post burn-in")
    axes[0][1].set_ylabel("density")
    axes[0][1].set_title(
        "Stationary distribution of the loss.\n"
        "Every alpha targets the same posterior, so these must coincide.",
        fontsize=10,
    )
    axes[0][1].legend(fontsize=8)

    # (c) Train vs test cross-entropy along the chain (same units, one axis).
    train = predictive_trace(baseline[0].trace, X_train, y_train, stride)
    test = predictive_trace(baseline[0].trace, X_test, y_test, stride)
    axes[1][0].plot(train["iteration"], train["log_loss"], lw=1.0,
                    color=TRAIN_COLOR, ls="-", label="train")
    axes[1][0].plot(test["iteration"], test["log_loss"], lw=1.0,
                    color=TEST_COLOR, ls="--", label="test")
    axes[1][0].axvline(burn_in, color="k", ls="--", lw=0.9)
    axes[1][0].set_xlabel("iteration")
    axes[1][0].set_ylabel("cross-entropy  [nats / observation]")
    axes[1][0].set_title(
        f"Train vs test loss at the state w_k (chain 0, alpha = {alphas[0]})",
        fontsize=10,
    )
    _label_series(axes[1][0], train["iteration"], train["log_loss"],
                  "train", TRAIN_COLOR, 0.35)
    _label_series(axes[1][0], test["iteration"], test["log_loss"],
                  "test", TEST_COLOR, 0.70)
    axes[1][0].legend(fontsize=8, loc="upper right")

    # (d) Test loss of the posterior-averaged predictive vs number of draws.
    for alpha, color in zip(alphas, colors):
        samples = stack_samples(list(chains_by_alpha[alpha]))
        curve = posterior_averaging_curve(samples, X_test, y_test)
        axes[1][1].plot(
            curve["n_draws"], curve["log_loss"], lw=1.6, color=color,
            label=f"alpha={alpha}",
        )
    single_draw_loss = float(
        np.mean(
            predictive_trace(
                baseline[0].trace[burn_in:], X_test, y_test, stride
            )["log_loss"]
        )
    )
    axes[1][1].axhline(
        single_draw_loss, color="grey", ls=":", lw=1.4,
        label="mean single-draw loss",
    )
    axes[1][1].set_xscale("log")
    axes[1][1].set_xlabel("retained posterior draws (pooled over chains)")
    axes[1][1].set_ylabel("test cross-entropy  [nats / observation]")
    axes[1][1].set_title(
        "Posterior-averaged test loss.\n"
        "Averaging gains over a single draw; alpha only affects how fast\n"
        "the Monte-Carlo error shrinks, not the limit.", fontsize=10
    )
    axes[1][1].legend(fontsize=8)
    return _save(fig, output_dir, "15_loss_curves.png")


def plot_accuracy_curves(
    chains_by_alpha: Mapping[float, Sequence[ChainOutput]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    burn_in: int,
    output_dir: str,
    stride: int = 10,
) -> str:
    """Accuracy curves: plug-in accuracy at ``w_k`` and the posterior average."""
    alphas = list(chains_by_alpha.keys())
    colors = _alpha_colors(alphas)
    baseline = chains_by_alpha[alphas[0]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) Train and test accuracy at w_k (same units -> one axis).
    train = predictive_trace(baseline[0].trace, X_train, y_train, stride)
    test = predictive_trace(baseline[0].trace, X_test, y_test, stride)
    axes[0][0].plot(train["iteration"], train["accuracy"], lw=0.9,
                    color=TRAIN_COLOR, ls="-", label="train")
    axes[0][0].plot(test["iteration"], test["accuracy"], lw=0.9,
                    color=TEST_COLOR, ls="--", label="test")
    axes[0][0].axvline(burn_in, color="k", ls="--", lw=0.9)
    axes[0][0].set_xlabel("iteration")
    axes[0][0].set_ylabel("plug-in accuracy")
    axes[0][0].set_title(
        f"Train vs test accuracy at the state w_k (chain 0, alpha = {alphas[0]})",
        fontsize=10,
    )
    _label_series(axes[0][0], train["iteration"], train["accuracy"],
                  "train", TRAIN_COLOR, 0.35)
    _label_series(axes[0][0], test["iteration"], test["accuracy"],
                  "test", TEST_COLOR, 0.70)
    axes[0][0].legend(fontsize=8, loc="lower right")
    axes[0][0].set_title(
        f"Train vs test accuracy at the state w_k (chain 0, alpha = {alphas[0]}).\n"
        "The two track each other: with n = 2000 and d = 9 there is no overfitting.",
        fontsize=10,
    )

    # (b) Post burn-in spread of the plug-in test accuracy, by alpha.
    spread = []
    for alpha in alphas:
        values = np.concatenate(
            [
                predictive_trace(c.trace[burn_in:], X_test, y_test, stride)["accuracy"]
                for c in chains_by_alpha[alpha]
            ]
        )
        spread.append(values)
    parts = axes[0][1].violinplot(spread, showmedians=True, widths=0.8)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.75)
    for key in ("cmins", "cmaxes", "cbars", "cmedians"):
        if key in parts:
            parts[key].set_color("0.35")
            parts[key].set_linewidth(1.0)
    axes[0][1].set_xticks(range(1, len(alphas) + 1))
    axes[0][1].set_xticklabels([str(a) for a in alphas])
    axes[0][1].set_xlabel("alpha")
    axes[0][1].set_ylabel("plug-in test accuracy at w_k")
    axes[0][1].set_title(
        "Posterior spread of the single-draw test accuracy.\n"
        "Identical across alpha, as the invariant measure requires.", fontsize=10
    )

    # (c) Posterior-averaged test accuracy vs number of draws.
    for alpha, color in zip(alphas, colors):
        samples = stack_samples(list(chains_by_alpha[alpha]))
        curve = posterior_averaging_curve(samples, X_test, y_test)
        axes[1][0].plot(
            curve["n_draws"], curve["accuracy"], lw=1.6, color=color,
            label=f"alpha={alpha}",
        )
    single_draw_accuracy = float(np.mean(spread[0]))
    axes[1][0].axhline(
        single_draw_accuracy, color="grey", ls=":", lw=1.4,
        label="mean single-draw accuracy",
    )
    axes[1][0].set_xscale("log")
    axes[1][0].set_xlabel("retained posterior draws (pooled over chains)")
    axes[1][0].set_ylabel("test accuracy of the posterior mean predictive")
    axes[1][0].set_title(
        "Posterior-averaged test accuracy.\n"
        "Accuracy is a blunt metric: averaging buys little here, unlike the\n"
        "cross-entropy. Spread between alphas is Monte-Carlo noise.", fontsize=10
    )
    axes[1][0].legend(fontsize=8)

    # (d) Same, on the training set, for the train/test comparison.
    for alpha, color in zip(alphas, colors):
        samples = stack_samples(list(chains_by_alpha[alpha]))
        curve = posterior_averaging_curve(samples, X_train, y_train)
        axes[1][1].plot(
            curve["n_draws"], curve["accuracy"], lw=1.6, color=color,
            label=f"alpha={alpha}",
        )
    axes[1][1].set_xscale("log")
    axes[1][1].set_xlabel("retained posterior draws (pooled over chains)")
    axes[1][1].set_ylabel("train accuracy of the posterior mean predictive")
    axes[1][1].set_title("Posterior-averaged train accuracy", fontsize=10)
    axes[1][1].legend(fontsize=8)
    return _save(fig, output_dir, "16_accuracy_curves.png")
