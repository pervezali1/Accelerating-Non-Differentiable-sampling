"""Orchestration of the alpha comparison and the step-size sensitivity study.

Everything except ``alpha`` is held fixed across the comparison: the target,
the constraint, the step size, the starting-point policy (and the actual
starting points), the number of iterations, the burn-in, the thinning interval
and every random seed.  Chain ``c`` always uses seed ``CHAIN_SEED_BASE + c``,
so the ``alpha`` runs are exactly paired.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from config import (
    CHAIN_SEED_BASE,
    INIT_SEED,
    ConstraintConfig,
    ExperimentConfig,
    ProjectionConfig,
    SamplerConfig,
    describe_seeds,
)
from constraint import g_value
from diagnostics import (
    AlphaSummary,
    PredictiveSummary,
    coefficient_names,
    per_coefficient_table,
    posterior_predictive,
    summarise_alpha,
    summary_table,
)
from projection import project_onto_K
from sampler import ChainOutput, run_chains, stack_samples
from synthetic_data import LogisticDataset
from target import LogisticTarget


def derive_step_size(target: LogisticTarget, step_scale: float) -> float:
    """``step_size = step_scale / L`` with ``L`` an upper bound on ``||Hess U0||``.

    The explicit Euler-Maruyama discretisation of the anchored dynamics is
    stable for ``step_size * a * L < 2``; since ``a <= 1`` a scale well below 2
    is used.  The same value is shared by every ``alpha`` so the comparison is
    not confounded by the discretisation.
    """
    return step_scale / target.lipschitz_constant()


def make_initial_points(
    target: LogisticTarget,
    constraint: ConstraintConfig,
    cfg: ExperimentConfig,
    projection_cfg: ProjectionConfig | None = None,
) -> np.ndarray:
    """Overdispersed starting points, identical for every ``alpha``.

    ``start_policy = "map_jitter"`` centres the chains on the smooth MAP (the
    minimiser of ``U0``); ``"zero_jitter"`` centres them at the origin.  Every
    point is projected onto ``K`` before it is returned, so the chains always
    start feasible.
    """
    if projection_cfg is None:
        projection_cfg = ProjectionConfig()
    rng = np.random.default_rng(INIT_SEED)
    centre = (
        target.smooth_map()
        if cfg.start_policy == "map_jitter"
        else np.zeros(constraint.d)
    )
    points = []
    for _ in range(cfg.n_chains):
        candidate = centre + cfg.start_jitter_sd * rng.standard_normal(constraint.d)
        points.append(project_onto_K(candidate, constraint, projection_cfg).z)
    return np.asarray(points)


@dataclass
class ExperimentResults:
    """Everything produced by :func:`run_alpha_comparison`."""

    chains_by_alpha: dict[float, list[ChainOutput]]
    samples_by_alpha: dict[float, np.ndarray]
    summaries: list[AlphaSummary]
    predictives: list[PredictiveSummary]
    step_size: float
    burn_in: int
    thin: int
    constraint: ConstraintConfig
    names: list[str] = field(default_factory=list)

    @property
    def alphas(self) -> list[float]:
        return [s.alpha for s in self.summaries]

    def summary_frame(self) -> pd.DataFrame:
        return summary_table(self.summaries)

    def coefficient_frame(self) -> pd.DataFrame:
        return per_coefficient_table(self.summaries)

    def predictive_frame(self) -> pd.DataFrame:
        return pd.DataFrame([p.to_row() for p in self.predictives])


def run_alpha_comparison(
    dataset: LogisticDataset,
    target: LogisticTarget,
    cfg: ExperimentConfig,
    constraint: ConstraintConfig | None = None,
    step_size: float | None = None,
    verbose: bool = True,
) -> ExperimentResults:
    """Run ``n_chains`` chains for every ``alpha`` and collect the diagnostics."""
    if constraint is None:
        constraint = cfg.constraint
    if step_size is None:
        step_size = (
            cfg.sampler.step_size
            if cfg.sampler.step_size is not None
            else derive_step_size(target, cfg.sampler.step_scale)
        )

    initial_points = make_initial_points(target, constraint, cfg, cfg.projection)
    names = coefficient_names(constraint.d)

    chains_by_alpha: dict[float, list[ChainOutput]] = {}
    samples_by_alpha: dict[float, np.ndarray] = {}
    summaries: list[AlphaSummary] = []
    predictives: list[PredictiveSummary] = []

    for alpha in cfg.alphas:
        sampler_cfg = SamplerConfig(
            n_iterations=cfg.sampler.n_iterations,
            burn_in=cfg.sampler.burn_in,
            thin=cfg.sampler.thin,
            step_size=step_size,
            step_scale=cfg.sampler.step_scale,
            alpha=alpha,
            swirl=cfg.sampler.swirl,
            seed=CHAIN_SEED_BASE,          # identical seeds for every alpha
            record_diagnostics=True,
        )
        chains = run_chains(
            target, initial_points, constraint, sampler_cfg, cfg.projection
        )
        summary = summarise_alpha(chains, cfg.sampler.burn_in, cfg.sampler.thin, names)
        samples = stack_samples(chains)

        chains_by_alpha[alpha] = chains
        samples_by_alpha[alpha] = samples
        summaries.append(summary)
        predictives.append(
            posterior_predictive(
                samples, dataset.X_test, dataset.y_test, dataset.beta_true, alpha
            )
        )
        if verbose:
            print(
                f"  alpha={alpha:<5} runtime={summary.runtime:7.2f}s  "
                f"min ESS={summary.min_ess:8.1f}  median ESS={summary.median_ess:8.1f}  "
                f"min ESS/s={summary.ess_per_second:7.2f}  "
                f"max split R-hat={summary.max_rhat:.4f}  "
                f"MSJD={summary.msjd:.3e}  proj={summary.projection_frequency:.4f}",
                flush=True,
            )

    return ExperimentResults(
        chains_by_alpha=chains_by_alpha,
        samples_by_alpha=samples_by_alpha,
        summaries=summaries,
        predictives=predictives,
        step_size=step_size,
        burn_in=cfg.sampler.burn_in,
        thin=cfg.sampler.thin,
        constraint=constraint,
        names=names,
    )


def run_step_size_sensitivity(
    target: LogisticTarget,
    cfg: ExperimentConfig,
    base_step_size: float,
    constraint: ConstraintConfig | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Repeat the alpha comparison at ``h``, ``h/2`` and ``h/4``.

    Shorter chains are used (``cfg.sensitivity_iterations``) because the point
    is the *relative* behaviour across step sizes, not a final posterior.
    """
    if constraint is None:
        constraint = cfg.constraint
    initial_points = make_initial_points(target, constraint, cfg, cfg.projection)
    names = coefficient_names(constraint.d)

    rows = []
    for divisor in cfg.step_size_divisors:
        step_size = base_step_size / divisor
        for alpha in cfg.alphas:
            sampler_cfg = SamplerConfig(
                n_iterations=cfg.sensitivity_iterations,
                burn_in=cfg.sensitivity_burn_in,
                thin=cfg.sampler.thin,
                step_size=step_size,
                alpha=alpha,
                swirl=cfg.sampler.swirl,
                seed=CHAIN_SEED_BASE,
                record_diagnostics=True,
            )
            chains = run_chains(
                target, initial_points, constraint, sampler_cfg, cfg.projection
            )
            summary = summarise_alpha(
                chains, cfg.sensitivity_burn_in, cfg.sampler.thin, names
            )
            row = summary.to_row()
            row["step_divisor"] = divisor
            rows.append(row)
            if verbose:
                print(
                    f"  h/{int(divisor)}  alpha={alpha:<5} "
                    f"min ESS/s={summary.ess_per_second:7.2f}  "
                    f"MSJD={summary.msjd:.3e}  "
                    f"max split R-hat={summary.max_rhat:.4f}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def feasibility_audit(
    results: ExperimentResults,
    tol: float = 1e-8,
) -> pd.DataFrame:
    """Confirm that **every** sampled state lies in ``K`` (§12.3)."""
    constraint = results.constraint
    p, eps = constraint.p_constraint, constraint.epsilon_constraint
    rows = []
    for alpha, chains in results.chains_by_alpha.items():
        worst = -np.inf
        n_states = 0
        for chain in chains:
            values = g_value(chain.trace, p, eps)
            worst = max(worst, float(values.max()))
            n_states += values.size
        rows.append(
            {
                "alpha": alpha,
                "n_states": n_states,
                "max_g": worst,
                "Lambda_constraint": constraint.Lambda_constraint,
                "max_violation": worst - constraint.Lambda_constraint,
                "all_feasible": bool(worst <= constraint.Lambda_constraint + tol),
            }
        )
    return pd.DataFrame(rows)


def save_artifacts(
    results: ExperimentResults,
    dataset: LogisticDataset,
    cfg: ExperimentConfig,
    output_dir: str,
    extra: Mapping[str, object] | None = None,
) -> None:
    """Persist the tables and the run metadata as CSV / JSON."""
    os.makedirs(output_dir, exist_ok=True)
    results.summary_frame().to_csv(
        os.path.join(output_dir, "summary_by_alpha.csv"), index=False
    )
    results.coefficient_frame().to_csv(
        os.path.join(output_dir, "per_coefficient.csv"), index=False
    )
    results.predictive_frame().to_csv(
        os.path.join(output_dir, "posterior_predictive.csv"), index=False
    )
    feasibility_audit(results).to_csv(
        os.path.join(output_dir, "feasibility_audit.csv"), index=False
    )

    metadata: dict[str, object] = {
        "seeds": describe_seeds(),
        "alphas": list(cfg.alphas),
        "n_chains": cfg.n_chains,
        "n_iterations": cfg.sampler.n_iterations,
        "burn_in": cfg.sampler.burn_in,
        "thin": cfg.sampler.thin,
        "step_size": results.step_size,
        "step_scale": cfg.sampler.step_scale,
        "swirl": cfg.sampler.swirl,
        "start_policy": cfg.start_policy,
        "start_jitter_sd": cfg.start_jitter_sd,
        "lambda_lasso": cfg.target.lambda_lasso,
        "delta_anchor": cfg.target.delta_anchor,
        "sigma_intercept": cfg.target.sigma_intercept,
        "p_constraint": results.constraint.p_constraint,
        "epsilon_constraint": results.constraint.epsilon_constraint,
        "radius_budget": results.constraint.radius_budget,
        "Lambda_constraint": results.constraint.Lambda_constraint,
        "g_at_origin": results.constraint.g_at_origin,
        "beta_true": dataset.beta_true.tolist(),
        "shrink_factor": dataset.shrink_factor,
    }
    if extra:
        metadata.update(dict(extra))
    with open(os.path.join(output_dir, "metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2, default=str)
