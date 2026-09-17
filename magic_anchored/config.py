"""Configuration objects for the constrained non-reversible anchored sampler.

Every number that changes the experiment lives in one of these dataclasses, so a
run is reproducible from the JSON they serialise to.  The dataclasses are frozen:
the whole point of the comparison across ``alpha`` is that nothing else moves,
and a frozen config makes an accidental mutation an error rather than a silent
difference between chains.

Mathematical notation used throughout the package
------------------------------------------------
``w = (beta_0, beta_1, ..., beta_p)`` is the parameter vector, ``beta_0`` the
intercept, so ``D = p + 1``.  ``X`` is the design matrix whose first column is
the intercept column of ones.  ``U`` is the (kinked) target potential, ``U0``
its smooth anchor, ``a(w) = exp(U(w) - U0(w))`` the anchoring clock,
``J_s(w)`` the state-dependent skew-symmetric field, ``C`` the constraint set,
``alpha`` the non-reversible strength and ``h`` the step size.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Sequence

__all__ = [
    "DataConfig",
    "TargetConfig",
    "FieldConfig",
    "SamplerConfig",
    "PilotConfig",
    "ExperimentConfig",
    "ConstraintSpec",
    "to_json",
]


@dataclass(frozen=True)
class DataConfig:
    """Where the data is and how it is split."""

    path: str = "data/magic04.data"
    test_size: float = 0.2
    split_seed: int = 0
    #: labels in the raw file; ``positive_label`` is mapped to 1 and the other to 0
    positive_label: str = "g"
    negative_label: str = "h"


@dataclass(frozen=True)
class TargetConfig:
    r"""The posterior: logistic likelihood, weak Gaussian intercept prior, LASSO.

    ``U(w) = sum_i [softplus(x_i'w) - y_i x_i'w] + beta_0^2 / (2 sigma0^2)
              + lambda sum_{j>=1} |beta_j|``

    ``lambda_`` is left as ``None`` by default and resolved to
    ``lambda_scale * n_train`` once the training set size is known.  That scaling
    is deliberate: the likelihood term is a *sum* over observations, so a penalty
    fixed independently of ``n`` becomes negligible as ``n`` grows and the
    constraint and the kink both stop mattering.

    ``delta`` smooths the kink in the anchor.  It appears in ``U0`` and never in
    ``U``, so it changes the algorithm and not the target, and it trades off two
    ways: the clock is bounded below by ``exp(-lambda p delta)``, so small
    ``delta`` keeps the clock near one, while the anchor's curvature at a
    coordinate sitting on the kink is ``lambda / delta``, so small ``delta``
    forces a small step size.  Left as ``None`` it is calibrated to maximise the
    product ``h(delta) a(delta)``.

    ``tempered`` divides the log-likelihood by ``n_train``.  It is off by
    default and exists only so that the tempered posterior can be requested
    explicitly; it is *not* the target this experiment reports.
    """

    lambda_: float | None = None
    lambda_scale: float = 0.01
    sigma0: float = 10.0
    #: ``None`` means "calibrate with :func:`sampler.calibrate_delta`"
    delta: float | None = None
    tempered: bool = False

    def resolve(self, n_train: int) -> "TargetConfig":
        """Fill in ``lambda_`` from ``lambda_scale`` if it was left unset."""
        if self.lambda_ is not None:
            return self
        return replace(self, lambda_=self.lambda_scale * n_train)


@dataclass(frozen=True)
class FieldConfig:
    r"""The state-dependent skew field ``J_s(w)``.

    Only the product ``alpha * s`` controls the strength of the non-reversible
    perturbation, so ``s`` is fixed at 1 and ``alpha`` is the knob.  ``triples``
    is left ``None`` to mean "the cyclic cover from
    :func:`geometry.generate_cyclic_triples`".
    """

    s: float = 1.0
    triples: tuple[tuple[int, int, int], ...] | None = None


@dataclass(frozen=True)
class SamplerConfig:
    r"""One chain of the projected anchored Euler scheme.

    ``w_{k+1} = P_C[ w_k - h a_k (I + alpha J_s(w_k)) grad U0(w_k)
                     + sqrt(2 h a_k) xi_k ]``
    """

    alpha: float = 0.0
    h: float = 1e-6
    n_iter: int = 20_000
    burn_in: int = 5_000
    thin: int = 5
    seed: int = 0
    #: ``w_0 = P_C(center + init_scale * xi)``, with ``xi`` standard normal
    init_scale: float = 1e-2
    #: if False the proposal is not projected (used only for the pilot chain)
    project: bool = True
    #: store the full trajectory, not just the retained samples
    store_trajectory: bool = True

    @property
    def n_retained(self) -> int:
        """How many samples survive burn-in and thinning."""
        return len(range(self.burn_in, self.n_iter, self.thin))


@dataclass(frozen=True)
class PilotConfig:
    """The unconstrained reversible run that fixes the constraint radius."""

    n_iter: int = 20_000
    burn_in: int = 5_000
    thin: int = 1
    seed: int = 12_345
    quantile: float = 0.999
    inflation: float = 1.10
    init_scale: float = 1e-2


@dataclass(frozen=True)
class ExperimentConfig:
    """The whole comparison."""

    #: the first five are the values named in the specification; the last three
    #: reach the scale at which the rotation is actually comparable to the
    #: gradient (see :func:`geometry.natural_alpha_scale`, about 13 here), and
    #: without them the sweep only shows that a 3% perturbation does nothing
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 10.0, 50.0, 200.0)
    n_chains: int = 4
    #: chain ``c`` of every alpha uses seed ``base_seed + c``, so the noise
    #: streams and the initial points are shared across alpha values
    base_seed: int = 1_000
    #: ``h`` is calibrated from the anchor Hessian unless given explicitly
    h: float | None = None
    #: ``h = step_safety / lambda_max(Hess U0(w_center))``
    step_safety: float = 0.1
    step_size_factors: tuple[float, ...] = (1.0, 0.5, 0.25)
    radius_factors: tuple[float, ...] = (0.9, 1.0, 1.1)
    #: a fixed radius, bypassing the pilot procedure
    fixed_radius: float | None = None
    n_iter: int = 20_000
    burn_in: int = 5_000
    thin: int = 5
    out_dir: str = "outputs"
    #: chains are independent, so they can be run in a process pool; results do
    #: not depend on this (each chain's noise is a function of its seed alone).
    #: -1 means "one worker per CPU"
    n_jobs: int = 1
    #: how many trajectory points to write per chain.  The full trajectory is
    #: kept in memory -- the validation checks and the radius figure use it --
    #: but writing all of it is what turns a 20-minute run into a gigabyte of
    #: NPZ.  ``None`` writes every step.
    trajectory_points: int | None = 5_000
    #: sensitivity studies are cheaper: fewer chains and, optionally, shorter.
    #: They ask whether the *answer* moves with h or R, which needs far less
    #: effective sample size than the efficiency comparison between alphas.
    n_chains_sensitivity: int = 2
    n_iter_sensitivity: int | None = None
    burn_in_sensitivity: int | None = None
    make_figures: bool = True


@dataclass(frozen=True)
class ConstraintSpec:
    r"""The frozen constraint ``C = {w : ||w - center||_2 <= radius}``.

    This object is created once, written to JSON, and reused for every ``alpha``
    and every chain.  Validation check 9 is that it is never recomputed.
    """

    center: tuple[float, ...]
    radius: float
    #: provenance, so a saved spec explains itself
    source: str = "pilot"
    pilot_quantile: float | None = None
    pilot_inflation: float | None = None
    pilot_exceedance_fraction: float | None = None

    @property
    def dim(self) -> int:
        return len(self.center)


def to_json(obj: Any, path: str) -> None:
    """Write a dataclass (or a dict of them) to ``path`` as readable JSON."""

    def encode(value: Any) -> Any:
        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        return value

    with open(path, "w") as handle:
        json.dump(encode(obj), handle, indent=2, sort_keys=True, default=float)
