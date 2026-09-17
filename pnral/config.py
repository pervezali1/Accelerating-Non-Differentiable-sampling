"""Configuration objects for the Projected Non-Reversible Anchored Langevin
(PNRAL) experiment on the MAGIC Gamma Telescope data set.

Every quantity that the algorithm depends on is declared here exactly once so
that a run is fully reproducible from the serialised JSON configuration.

Parameter-name discipline (deliberately verbose, see the specification):

``p_constraint``        exponent of the smoothed l_p constraint functional g
``epsilon_constraint``  smoothing parameter of g
``Lambda_constraint``   threshold defining K = {w : g(w) <= Lambda_constraint}
``lambda_lasso``        coefficient of the L1 (LASSO) penalty in the potential
``delta_anchor``        smoothing parameter of the anchor potential U0
``sigma_intercept``     prior standard deviation of the (unpenalised) intercept
``alpha``               strength of the non-reversible drift
``step_size``           Euler--Maruyama step size h
``swirl_scale``         constant multiplying the skew-symmetric matrix J

``lambda`` is never used on its own: the constraint threshold is always
``Lambda_constraint`` and the LASSO penalty is always ``lambda_lasso``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

__all__ = [
    "DataConfig",
    "TargetConfig",
    "ConstraintConfig",
    "SamplerConfig",
    "ExperimentConfig",
]


@dataclass
class DataConfig:
    """Location and preprocessing of the MAGIC Gamma Telescope data."""

    #: Path to the raw UCI file ``magic04.data`` (comma separated, no header).
    data_path: str = "data/magic04.data"
    #: Fraction of rows held out for the final (test-set only) evaluation.
    test_size: float = 0.20
    #: Seed of the *stratified* train/test split.
    split_seed: int = 20240917
    #: Never silently substitute data: the synthetic surrogate is opt-in only.
    use_synthetic: bool = False
    #: Number of rows of the synthetic surrogate (real file has 19020 rows).
    synthetic_n_rows: int = 19020
    #: Seed of the synthetic surrogate generator.
    synthetic_seed: int = 7


@dataclass
class TargetConfig:
    """Parameters of the (non-smooth) target potential U and its anchor U0."""

    #: Coefficient of the L1 penalty applied to the slopes only.
    #:
    #: The likelihood is a *sum* over ~15k training rows, so a penalty of
    #: order 10^2 is required to shift the posterior by a noticeable
    #: fraction of a posterior standard deviation.  With lambda_lasso = 100
    #: and delta_anchor = 0.01 the anchoring coefficient a(w) stays in a
    #: numerically comfortable range (roughly [0.1, 1] at the posterior mode)
    #: while remaining genuinely state dependent.
    lambda_lasso: float = 100.0
    #: Standard deviation of the weak Gaussian prior on the intercept.
    sigma_intercept: float = 10.0
    #: Smoothing parameter of the anchor: |w| -> sqrt(w^2 + delta_anchor^2).
    delta_anchor: float = 0.01


@dataclass
class ConstraintConfig:
    """Smoothed l_p constraint g(w) <= Lambda_constraint."""

    p_constraint: float = 1.5
    epsilon_constraint: float = 0.05
    #: ``None`` triggers the pilot-based selection procedure of section 6.
    #: A float bypasses the pilot procedure entirely (user supplied value).
    Lambda_constraint: Optional[float] = None

    # ---- pilot-based threshold selection ---------------------------------
    #: Empirical quantile of g(w_k) over the unconstrained pilot samples.
    pilot_quantile: float = 0.999
    #: Lambda = g_min + pilot_inflation * (q999 - g_min).
    pilot_inflation: float = 1.10
    pilot_iterations: int = 6000
    pilot_burn_in: int = 2000
    pilot_thinning: int = 1
    #: The pilot chain uses a deliberately conservative step size.
    pilot_step_size_factor: float = 0.25
    pilot_seed: int = 4242

    # ---- constraint-sensitivity experiment (section 6.9) ------------------
    lambda_small_factor: float = 0.90
    lambda_large_factor: float = 1.10

    # ---- projection ------------------------------------------------------
    #: Absolute tolerance used when asserting g(w) <= Lambda_constraint.
    projection_tolerance: float = 1.0e-8
    #: Tolerance of the inner/outer root finders of the projection.
    projection_root_tolerance: float = 1.0e-12
    #: ``True`` uses the literal ``scipy.optimize.brentq`` implementation of
    #: section 11; ``False`` uses the vectorised safeguarded-Newton solver,
    #: which is validated against the brentq path in ``tests/``.
    projection_use_brentq: bool = False


@dataclass
class SamplerConfig:
    """Projected Non-Reversible Anchored Langevin sampler settings."""

    #: alpha = 0 is the constrained *reversible* anchored baseline.
    alphas: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)
    swirl_scale: float = 1.0
    #: "disjoint" is the primary manuscript-faithful construction;
    #: "overlapping" is the labelled sensitivity extension of section 9.
    triples_mode: str = "disjoint"

    number_of_chains: int = 4
    n_iterations: int = 20000
    burn_in: int = 5000
    thinning: int = 1

    #: ``None`` calibrates h from the smoothness guide L <= 0.25*||X||_2^2
    #: and then halves it until the chain at max(alphas) is stable.
    step_size: Optional[float] = None
    #: h0 = step_size_safety / L_guide before stability halving.
    step_size_safety: float = 0.5
    #: Maximum number of halvings during automatic step-size calibration.
    step_size_max_halvings: int = 12
    #: Length of each trial chain used for step-size calibration.
    calibration_iterations: int = 1500
    #: A trial step size is rejected if the projection frequency exceeds this.
    calibration_max_projection_frequency: float = 0.25

    #: Chain c of every alpha uses seed ``chain_seed_base + c`` -- identical
    #: seeds, identical initial points and identical data across all alphas.
    chain_seed_base: int = 1000
    #: Standard deviation of the Gaussian perturbation around the
    #: constrained MAP used to initialise each chain.
    init_perturbation_scale: float = 0.02
    init_seed: int = 31337

    #: Store every ``diagnostic_trace_thin``-th per-iteration diagnostic row
    #: in the CSV traces (the full traces always go into the NPZ archive).
    diagnostic_trace_thin: int = 25


@dataclass
class ExperimentConfig:
    """Top level configuration: everything needed to reproduce a run."""

    data: DataConfig = field(default_factory=DataConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)

    output_dir: str = "results"
    #: Master seed; every sub-seed above is reported explicitly as well.
    master_seed: int = 20240917

    #: Alphas used for the step-size and constraint-radius sensitivity runs.
    sensitivity_alphas: Tuple[float, ...] = (0.0, 0.5)
    #: Sensitivity runs are shorter than the primary comparison.
    sensitivity_iterations: int = 8000
    sensitivity_burn_in: int = 2000
    run_step_size_sensitivity: bool = True
    run_constraint_sensitivity: bool = True
    run_overlapping_extension: bool = True

    #: Number of posterior samples processed per chunk when forming the
    #: posterior-averaged predictive probabilities.
    prediction_chunk_size: int = 500

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-``dict`` view suitable for ``json.dump``."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Write the configuration to ``path`` as indented JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=False)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentConfig":
        """Rebuild an :class:`ExperimentConfig` from :meth:`to_dict` output."""
        payload = dict(payload)
        data = DataConfig(**payload.pop("data"))
        target = TargetConfig(**payload.pop("target"))
        constraint = ConstraintConfig(**payload.pop("constraint"))
        sampler_payload = dict(payload.pop("sampler"))
        sampler_payload["alphas"] = tuple(sampler_payload["alphas"])
        sampler = SamplerConfig(**sampler_payload)
        payload["sensitivity_alphas"] = tuple(payload.get("sensitivity_alphas", ()))
        return cls(data=data, target=target, constraint=constraint,
                   sampler=sampler, **payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Read a configuration written by :meth:`to_json`."""
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    # -- convenience -------------------------------------------------------
    def smoke_test(self) -> "ExperimentConfig":
        """Return a small-but-complete configuration used by the test-suite."""
        import copy

        cfg = copy.deepcopy(self)
        cfg.sampler.n_iterations = 1200
        cfg.sampler.burn_in = 400
        cfg.sampler.number_of_chains = 2
        cfg.sampler.calibration_iterations = 300
        cfg.constraint.pilot_iterations = 800
        cfg.constraint.pilot_burn_in = 200
        cfg.sensitivity_iterations = 600
        cfg.sensitivity_burn_in = 200
        return cfg
