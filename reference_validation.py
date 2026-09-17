"""Section 13: exact-reference validation on a weighted-L1 target.

Target
------
    U(x)  = sum_i omega_i |x_i|        on  K = {g(x) <= Lambda_constraint},
    U0(x) = sum_i omega_i sqrt(x_i^2 + delta_anchor^2).

Unconstrained, ``exp(-U)`` is a product of Laplace densities with scales
``1 / omega_i``.  Restricted to ``K`` it is that product *truncated* to ``K``,
so exact independent draws are obtained by rejection sampling:

    x_i ~ Laplace(0, 1 / omega_i),   keep the draw iff g(x) <= Lambda_constraint.

This gives a ground truth against which the projected anchored Langevin output
is compared (means, variances, empirical CDFs, 1-D Wasserstein distances, a
multivariate energy distance, pair plots and the boundary-distance
distribution).

The acceptance probability of the rejection sampler is reported.  If it falls
below ``ReferenceConfig.min_acceptance_warn`` a warning is issued and the user
is asked to change the constraint configuration -- the target is **never**
silently modified.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import energy_distance, wasserstein_distance

from config import (
    CHAIN_SEED_BASE,
    ConstraintConfig,
    ProjectionConfig,
    ReferenceConfig,
    SamplerConfig,
)
from constraint import g_value
from diagnostics import _save, summarise_alpha
from projection import project_onto_K
from sampler import ChainOutput, run_chains, stack_samples
from target import WeightedL1Target


@dataclass
class ReferenceSample:
    """Independent truncated-Laplace draws plus the acceptance statistics."""

    draws: np.ndarray                 # (n_accepted, d)
    n_proposed: int
    acceptance_probability: float
    warning: str | None


def sample_reference(
    omega: np.ndarray,
    constraint: ConstraintConfig,
    n_proposed: int,
    seed: int,
    min_acceptance_warn: float = 0.01,
) -> ReferenceSample:
    """Rejection-sample ``exp(-sum omega_i |x_i|)`` restricted to ``K``."""
    rng = np.random.default_rng(seed)
    d = omega.size
    proposals = rng.laplace(loc=0.0, scale=1.0 / omega, size=(n_proposed, d))
    values = g_value(proposals, constraint.p_constraint, constraint.epsilon_constraint)
    accepted = proposals[values <= constraint.Lambda_constraint]
    probability = accepted.shape[0] / n_proposed

    message = None
    if probability < min_acceptance_warn:
        message = (
            f"Rejection-sampling acceptance probability is {probability:.5f}, below "
            f"the {min_acceptance_warn:.3f} threshold. The reference sample is too "
            "small to be trustworthy. The target has NOT been changed; please "
            "widen the constraint (increase radius_budget / Lambda_constraint) or "
            "increase the weights omega_i, then re-run."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return ReferenceSample(
        draws=accepted,
        n_proposed=n_proposed,
        acceptance_probability=probability,
        warning=message,
    )


def multivariate_energy_distance(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    max_points: int = 2500,
    seed: int = 0,
) -> float:
    """Energy distance ``2 E||X-Y|| - E||X-X'|| - E||Y-Y'||``.

    Both samples are subsampled to at most ``max_points`` rows to keep the
    pairwise distance matrices small; the subsampling is seeded.
    """
    rng = np.random.default_rng(seed)

    def subsample(sample: np.ndarray) -> np.ndarray:
        if sample.shape[0] <= max_points:
            return sample
        index = rng.choice(sample.shape[0], size=max_points, replace=False)
        return sample[index]

    A, B = subsample(np.asarray(sample_a)), subsample(np.asarray(sample_b))

    def mean_distance(P: np.ndarray, Q: np.ndarray) -> float:
        diff = P[:, None, :] - Q[None, :, :]
        return float(np.sqrt((diff ** 2).sum(axis=-1)).mean())

    return float(
        2.0 * mean_distance(A, B) - mean_distance(A, A) - mean_distance(B, B)
    )


def compare_to_reference(
    langevin: np.ndarray,
    reference: np.ndarray,
    label: str,
    seed: int = 0,
    ess: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-coordinate mean, variance, Wasserstein and 1-D energy distances.

    When ``ess`` (the per-coordinate effective sample size of the Langevin
    draws) is supplied, the Monte-Carlo standard error of each coordinate mean,
    ``sd / sqrt(ESS)``, is reported alongside, together with the mean error
    expressed in units of that standard error.  Without it a discrepancy cannot
    be attributed to discretisation bias rather than sampling noise.
    """
    langevin = np.asarray(langevin, dtype=float).reshape(-1, reference.shape[1])
    rows = []
    for j in range(reference.shape[1]):
        effective = float(ess[j]) if ess is not None else float(langevin.shape[0])
        mc_se = float(langevin[:, j].std(ddof=1) / np.sqrt(max(effective, 1.0)))
        rows.append(
            {
                "mc_se_mean": mc_se,
                "ess": effective,
                "run": label,
                "coordinate": j,
                "mean_langevin": float(langevin[:, j].mean()),
                "mean_reference": float(reference[:, j].mean()),
                "mean_abs_error": float(
                    abs(langevin[:, j].mean() - reference[:, j].mean())
                ),
                "var_langevin": float(langevin[:, j].var(ddof=1)),
                "var_reference": float(reference[:, j].var(ddof=1)),
                "var_rel_error": float(
                    abs(langevin[:, j].var(ddof=1) - reference[:, j].var(ddof=1))
                    / reference[:, j].var(ddof=1)
                ),
                "wasserstein_1d": float(
                    wasserstein_distance(langevin[:, j], reference[:, j])
                ),
                "energy_1d": float(energy_distance(langevin[:, j], reference[:, j])),
            }
        )
    frame = pd.DataFrame(rows)
    # How many Monte-Carlo standard errors separate the two means?
    frame["mean_error_in_se"] = frame["mean_abs_error"] / frame["mc_se_mean"]
    # Relative standard error of a variance estimate is ~sqrt(2/ESS).
    frame["var_rel_se"] = np.sqrt(2.0 / frame["ess"].clip(lower=1.0))
    frame["energy_multivariate"] = multivariate_energy_distance(
        langevin, reference, seed=seed
    )
    return frame


def run_reference_experiment(
    reference_cfg: ReferenceConfig,
    constraint: ConstraintConfig,
    projection_cfg: ProjectionConfig | None = None,
    step_divisors: Sequence[float] = (1.0, 4.0),
    verbose: bool = True,
) -> dict[str, object]:
    """Full §13 validation: reference draws, Langevin runs and the comparison."""
    if projection_cfg is None:
        projection_cfg = ProjectionConfig()

    omega = reference_cfg.omega
    target = WeightedL1Target(omega=omega, delta_anchor=reference_cfg.delta_anchor)

    reference = sample_reference(
        omega,
        constraint,
        reference_cfg.n_reference,
        reference_cfg.seed,
        reference_cfg.min_acceptance_warn,
    )
    if verbose:
        print(
            f"  rejection sampler: {reference.draws.shape[0]} / "
            f"{reference.n_proposed} accepted "
            f"(acceptance probability = {reference.acceptance_probability:.4f})",
            flush=True,
        )

    base_step = reference_cfg.step_scale / target.lipschitz_constant()

    # Overdispersed, feasible starting points shared by every alpha.
    init_rng = np.random.default_rng(reference_cfg.seed + 11)
    initial_points = np.asarray(
        [
            project_onto_K(
                init_rng.laplace(0.0, 1.0 / omega), constraint, projection_cfg
            ).z
            for _ in range(reference_cfg.n_chains)
        ]
    )

    chains_by_key: dict[str, list[ChainOutput]] = {}
    samples_by_key: dict[str, np.ndarray] = {}
    comparison_frames = []
    summary_rows = []

    for divisor in step_divisors:
        step_size = base_step / divisor
        # Time-matched runs: halving the step doubles the iteration count (and
        # the burn-in and the thinning interval), so every configuration
        # simulates the same amount of diffusion time and retains the same
        # number of draws.  Otherwise a smaller step would simply look worse
        # because it explored less, confounding bias with Monte-Carlo error.
        n_iterations = int(round(reference_cfg.n_iterations * divisor))
        burn_in = int(round(reference_cfg.burn_in * divisor))
        thin = max(1, int(round(reference_cfg.thin * divisor)))
        for alpha in reference_cfg.alphas:
            key = f"alpha={alpha}, h/{int(divisor)}"
            sampler_cfg = SamplerConfig(
                n_iterations=n_iterations,
                burn_in=burn_in,
                thin=thin,
                step_size=step_size,
                alpha=alpha,
                swirl=1.0,
                seed=CHAIN_SEED_BASE + 500,
                record_diagnostics=True,
            )
            chains = run_chains(
                target, initial_points, constraint, sampler_cfg, projection_cfg
            )
            samples = stack_samples(chains)
            chains_by_key[key] = chains
            samples_by_key[key] = samples

            summary = summarise_alpha(
                chains,
                burn_in,
                thin,
                [f"x{j}" for j in range(constraint.d)],
            )
            row = summary.to_row()
            row["key"] = key
            row["step_divisor"] = divisor
            summary_rows.append(row)

            frame = compare_to_reference(
                samples, reference.draws, key, seed=reference_cfg.seed, ess=summary.ess
            )
            frame["step_divisor"] = divisor
            frame["alpha"] = alpha
            frame["step_size"] = step_size
            frame["n_iterations"] = n_iterations
            comparison_frames.append(frame)
            if verbose:
                print(
                    f"  {key:<20} max |mean err| = "
                    f"{frame['mean_abs_error'].max():.4f} "
                    f"({frame['mean_error_in_se'].max():.1f} MC SE)  "
                    f"max W1 = {frame['wasserstein_1d'].max():.4f}  "
                    f"energy = {frame['energy_multivariate'].iloc[0]:.5f}  "
                    f"min ESS = {summary.min_ess:.0f}  "
                    f"proj = {summary.projection_frequency:.3f}",
                    flush=True,
                )

    return {
        "reference": reference,
        "target": target,
        "base_step_size": base_step,
        "chains_by_key": chains_by_key,
        "samples_by_key": samples_by_key,
        "comparison": pd.concat(comparison_frames, ignore_index=True),
        "summary": pd.DataFrame(summary_rows),
    }


# ==========================================================================
# Figures for the validation target
# ==========================================================================
def plot_reference_cdfs(
    samples_by_key: Mapping[str, np.ndarray],
    reference: np.ndarray,
    output_dir: str,
) -> str:
    """Empirical CDFs of the Langevin draws against the exact reference."""
    d = reference.shape[1]
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    for j, ax in enumerate(axes.ravel()[:d]):
        grid = np.sort(reference[:, j])
        ax.plot(
            grid,
            np.arange(1, grid.size + 1) / grid.size,
            "k-",
            lw=2.0,
            label="reference" if j == 0 else None,
        )
        for key, samples in samples_by_key.items():
            flat = np.sort(samples.reshape(-1, d)[:, j])
            ax.plot(
                flat,
                np.arange(1, flat.size + 1) / flat.size,
                lw=1.0,
                alpha=0.85,
                label=key if j == 0 else None,
            )
        ax.set_title(f"x{j}", fontsize=10)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("Empirical CDFs: projected anchored Langevin vs exact reference")
    return _save(fig, output_dir, "R1_reference_cdfs.png")


def plot_reference_moments(comparison: pd.DataFrame, output_dir: str) -> str:
    """Coordinate means / variances and the Wasserstein distances."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    first = comparison[comparison["run"] == comparison["run"].iloc[0]]
    axes[0].plot(first["coordinate"], first["mean_reference"], "k-o", label="reference")
    axes[1].plot(first["coordinate"], first["var_reference"], "k-o", label="reference")
    for key, group in comparison.groupby("run"):
        axes[0].plot(group["coordinate"], group["mean_langevin"], "--s", ms=4, label=key)
        axes[1].plot(group["coordinate"], group["var_langevin"], "--s", ms=4, label=key)
        axes[2].plot(group["coordinate"], group["wasserstein_1d"], "-o", ms=4, label=key)
    axes[0].set_ylabel("coordinate mean")
    axes[1].set_ylabel("coordinate variance")
    axes[2].set_ylabel("1-D Wasserstein distance to reference")
    for ax in axes:
        ax.set_xlabel("coordinate")
        ax.legend(fontsize=7)
    fig.suptitle("Validation target: moments and distances")
    return _save(fig, output_dir, "R2_reference_moments.png")


def plot_reference_pairs(
    samples_by_key: Mapping[str, np.ndarray],
    reference: np.ndarray,
    output_dir: str,
    coordinate_pairs: Sequence[tuple[int, int]] = ((0, 1), (0, 8), (3, 4)),
) -> str:
    """Pair plots of the Langevin draws next to the reference draws."""
    d = reference.shape[1]
    keys = list(samples_by_key.keys())
    n_cols = 1 + len(keys)
    fig, axes = plt.subplots(
        len(coordinate_pairs), n_cols, figsize=(3.3 * n_cols, 3.1 * len(coordinate_pairs)),
        squeeze=False,
    )
    for r, (i, j) in enumerate(coordinate_pairs):
        axes[r][0].hexbin(reference[:, i], reference[:, j], gridsize=40, cmap="magma", mincnt=1)
        axes[r][0].set_ylabel(f"x{j}")
        axes[r][0].set_xlabel(f"x{i}")
        if r == 0:
            axes[r][0].set_title("reference", fontsize=10)
        for c, key in enumerate(keys, start=1):
            flat = samples_by_key[key].reshape(-1, d)
            axes[r][c].hexbin(flat[:, i], flat[:, j], gridsize=40, cmap="magma", mincnt=1)
            axes[r][c].set_xlabel(f"x{i}")
            if r == 0:
                axes[r][c].set_title(key, fontsize=9)
    fig.suptitle("Validation target: pair plots")
    return _save(fig, output_dir, "R3_reference_pairs.png")


def plot_reference_boundary(
    samples_by_key: Mapping[str, np.ndarray],
    reference: np.ndarray,
    constraint: ConstraintConfig,
    output_dir: str,
) -> str:
    """Boundary-distance distribution ``Lambda_constraint - g(x)``."""
    p, eps, Lambda = (
        constraint.p_constraint,
        constraint.epsilon_constraint,
        constraint.Lambda_constraint,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        Lambda - g_value(reference, p, eps),
        bins=70,
        histtype="step",
        lw=2.2,
        color="k",
        density=True,
        label="reference",
    )
    for key, samples in samples_by_key.items():
        flat = samples.reshape(-1, reference.shape[1])
        ax.hist(
            Lambda - g_value(flat, p, eps),
            bins=70,
            histtype="step",
            lw=1.3,
            density=True,
            label=key,
        )
    ax.set_xlabel("Lambda_constraint - g(x)")
    ax.set_ylabel("density")
    ax.set_title("Validation target: distance-to-boundary distribution")
    ax.legend(fontsize=8)
    return _save(fig, output_dir, "R4_reference_boundary.png")
