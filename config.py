"""Configuration objects for the constrained Non-Reversible Anchored Langevin experiment.

All tunable quantities live here so that every module reads from a single,
immutable source of truth.  Naming follows the specification exactly:

======================  ====================================================
``p_constraint``        exponent of the smoothed l^p constraint
``epsilon_constraint``  smoothing parameter of the constraint
``Lambda_constraint``   constraint threshold
``lambda_lasso``        L1-prior strength
``delta_anchor``        smoothing parameter of the L1 anchor
``alpha``               non-reversibility strength
``step_size``           Euler step size
======================  ====================================================

``Lambda_constraint`` (constraint threshold) and ``lambda_lasso`` (LASSO
penalty) are deliberately distinct names and are never aliased.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Global seeds.  Every stochastic object in the project derives its seed from
# these constants, so the whole study is reproducible bit-for-bit.
# --------------------------------------------------------------------------
MASTER_SEED: int = 20240917
DATA_SEED: int = MASTER_SEED + 1          # design matrix + Bernoulli responses
INIT_SEED: int = MASTER_SEED + 2          # chain starting points
CHAIN_SEED_BASE: int = MASTER_SEED + 1000  # per-chain Gaussian innovations
REFERENCE_SEED: int = MASTER_SEED + 3     # exact-reference validation target
FD_SEED: int = MASTER_SEED + 4            # finite-difference / unit-test probes


@dataclass(frozen=True)
class DataConfig:
    """Synthetic Bayesian logistic-regression design."""

    n_train: int = 2000
    n_test: int = 1000
    n_features: int = 8          # number of predictors (excluding intercept)
    rho: float = 0.5             # AR(1) correlation of the predictors
    seed: int = DATA_SEED
    #: ``beta_raw[0]`` is the intercept; the remaining 8 entries are slopes.
    beta_raw: Tuple[float, ...] = (
        -0.25, 1.20, -1.00, 0.80, 0.00, 0.60, -0.50, 0.30, 0.00,
    )

    @property
    def d(self) -> int:
        """Total parameter dimension (intercept + predictors)."""
        return self.n_features + 1

    def __post_init__(self) -> None:
        if len(self.beta_raw) != self.d:
            raise ValueError(
                f"beta_raw has length {len(self.beta_raw)}, expected d = {self.d}"
            )


@dataclass(frozen=True)
class TargetConfig:
    """Potential ``U`` and its smooth anchor ``U0``."""

    #: L1-prior strength (LASSO).  NOT the constraint threshold.
    lambda_lasso: float = 10.0
    #: Weak Gaussian prior on the (unpenalised) intercept.
    sigma_intercept: float = 5.0
    #: Smoothing parameter of the L1 anchor, sqrt(w^2 + delta_anchor^2).
    delta_anchor: float = 0.02

    def __post_init__(self) -> None:
        if self.lambda_lasso < 0.0:
            raise ValueError("lambda_lasso must be non-negative")
        if self.delta_anchor <= 0.0:
            raise ValueError("delta_anchor must be strictly positive")
        if self.sigma_intercept <= 0.0:
            raise ValueError("sigma_intercept must be strictly positive")


@dataclass(frozen=True)
class ConstraintConfig:
    """Smoothed l^p constraint ``K = {w : g(w) <= Lambda_constraint}``.

    ``g(w) = sum_i (w_i^2 + epsilon_constraint^2)^(p_constraint/2)``
    ``Lambda_constraint = d * epsilon_constraint**p_constraint
                          + radius_budget**p_constraint``
    """

    d: int = 9
    p_constraint: float = 1.5
    epsilon_constraint: float = 0.05
    radius_budget: float = 4.5

    @property
    def g_at_origin(self) -> float:
        """``g(0) = d * epsilon_constraint ** p_constraint``."""
        return self.d * self.epsilon_constraint ** self.p_constraint

    @property
    def Lambda_constraint(self) -> float:
        """Constraint threshold (never called ``lambda``)."""
        return self.g_at_origin + self.radius_budget ** self.p_constraint

    @property
    def interior_level(self) -> float:
        """Mid-point level set used to place ``beta_true`` safely inside ``K``.

        ``g(0) + 0.5 * (Lambda_constraint - g(0))``
        """
        return self.g_at_origin + 0.5 * (self.Lambda_constraint - self.g_at_origin)

    def __post_init__(self) -> None:
        if self.p_constraint < 1.0:
            # For p < 1 the scalar KKT map z -> z + eta p z (z^2+eps^2)^(p/2-1)
            # can fail to be monotone, which breaks the projection solver.
            raise ValueError("p_constraint < 1 is not supported by the projection")
        if self.epsilon_constraint <= 0.0:
            raise ValueError("epsilon_constraint must be strictly positive")
        if self.radius_budget <= 0.0:
            raise ValueError("radius_budget must be strictly positive")
        if not self.Lambda_constraint > self.g_at_origin:
            raise ValueError("Lambda_constraint must exceed g(0) = d*eps^p")


@dataclass(frozen=True)
class SamplerConfig:
    """Projected Non-Reversible Anchored Langevin sampler settings."""

    n_iterations: int = 20_000
    burn_in: int = 5_000
    thin: int = 5
    #: Euler step size.  ``None`` means "derive from the curvature of U0"
    #: via ``step_scale / L``; see :func:`experiment.derive_step_size`.
    step_size: float | None = None
    step_scale: float = 0.20
    #: Non-reversibility strength.
    alpha: float = 0.0
    #: Constant swirl multiplier ``s``; J is replaced by ``s * J``.  Only the
    #: product ``alpha * s`` matters, so the primary experiment fixes s = 1.
    swirl: float = 1.0
    seed: int = CHAIN_SEED_BASE
    #: Store the full per-iteration diagnostic trace.
    record_diagnostics: bool = True

    @property
    def n_stored(self) -> int:
        """Number of retained draws after burn-in and thinning."""
        return len(range(self.burn_in, self.n_iterations, self.thin))

    def __post_init__(self) -> None:
        if self.burn_in >= self.n_iterations:
            raise ValueError("burn_in must be smaller than n_iterations")
        if self.thin < 1:
            raise ValueError("thin must be >= 1")
        if self.swirl <= 0.0:
            raise ValueError("swirl multiplier s must be positive")


@dataclass(frozen=True)
class ProjectionConfig:
    """Tolerances for the exact Euclidean projection onto ``K``."""

    xtol: float = 1e-13          # brentq tolerance for the inner scalar solves
    eta_xtol: float = 1e-12      # brentq tolerance for the outer eta search
    feasibility_tol: float = 1e-8
    max_eta_expansions: int = 200
    eta_high_init: float = 1.0
    eta_growth: float = 2.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level comparison across the non-reversibility strength ``alpha``."""

    alphas: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)
    n_chains: int = 4
    #: Step-size sensitivity study uses h, h/2, h/4.
    step_size_divisors: Tuple[float, ...] = (1.0, 2.0, 4.0)
    #: Shorter runs for the sensitivity sweep to keep total runtime sane.
    sensitivity_iterations: int = 8_000
    sensitivity_burn_in: int = 2_000
    #: Starting-point policy: "map_jitter" or "zero_jitter".  Identical across
    #: every alpha because the same INIT_SEED is used.
    start_policy: str = "map_jitter"
    start_jitter_sd: float = 0.25
    data: DataConfig = field(default_factory=DataConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    output_dir: str = "results"

    def __post_init__(self) -> None:
        if self.constraint.d != self.data.d:
            raise ValueError("constraint.d must equal data.d")
        if self.start_policy not in {"map_jitter", "zero_jitter"}:
            raise ValueError("start_policy must be 'map_jitter' or 'zero_jitter'")


@dataclass(frozen=True)
class ReferenceConfig:
    """Section 13: exact-reference validation on the weighted-L1 target.

    ``U(x) = sum_i omega_i |x_i|`` restricted to the same constraint ``K``.
    """

    d: int = 9
    # A larger delta than the logistic run: U0's curvature is lambda/delta, so a
    # very small delta would force an impractically small step on this target.
    delta_anchor: float = 0.05
    n_iterations: int = 100_000
    burn_in: int = 20_000
    thin: int = 20
    step_scale: float = 0.80
    n_chains: int = 4
    alphas: Tuple[float, ...] = (0.0, 0.5)
    n_reference: int = 200_000      # proposals for the rejection sampler
    min_acceptance_warn: float = 0.01
    seed: int = REFERENCE_SEED

    @property
    def omega(self) -> np.ndarray:
        """Unequal positive weights ``omega_i``."""
        return np.linspace(0.8, 2.4, self.d)


#: The step-size sensitivity study and the tight-constraint variant reuse
#: the default constraint but shrink the radius so that ``K`` actually binds.
TIGHT_RADIUS_BUDGET: float = 1.45


def tight_constraint(base: ConstraintConfig) -> ConstraintConfig:
    """Return a copy of ``base`` whose constraint is active at the posterior.

    With ``radius_budget = 4.5`` the logistic posterior (which concentrates
    within O(n^{-1/2}) of ``beta_true``) never reaches the boundary, so the
    projection statistics are identically zero.  The tight variant keeps every
    other setting fixed and only shrinks the radius budget.
    """
    return ConstraintConfig(
        d=base.d,
        p_constraint=base.p_constraint,
        epsilon_constraint=base.epsilon_constraint,
        radius_budget=TIGHT_RADIUS_BUDGET,
    )


def describe_seeds() -> dict[str, int]:
    """Report every fixed seed used in the study."""
    return {
        "MASTER_SEED": MASTER_SEED,
        "DATA_SEED": DATA_SEED,
        "INIT_SEED": INIT_SEED,
        "CHAIN_SEED_BASE": CHAIN_SEED_BASE,
        "REFERENCE_SEED": REFERENCE_SEED,
        "FD_SEED": FD_SEED,
    }
