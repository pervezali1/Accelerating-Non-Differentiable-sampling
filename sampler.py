"""Projected Non-Reversible Anchored Langevin sampler.

One iteration
-------------
    grad_k  = grad_U0(w_k)
    log_a_k = log_a(w_k),      a_k = exp(log_a_k)
    J_k     = construct_J(w_k, p_constraint, epsilon_constraint, swirl)

    drift_k = -a_k * (I_d + alpha * J_k) @ grad_k

    xi_k ~ N(0, I_d)
    proposal = w_k + step_size * drift_k + sqrt(2 * step_size * a_k) * xi_k
    w_{k+1}  = projection_K(proposal)

Correctness of the drift
------------------------
The unconstrained diffusion ``dw = -a (I + alpha J) grad_U0 dt + sqrt(2a) dW``
leaves ``pi ∝ exp(-U0)/a = exp(-U)`` invariant.  The reversible part is the
anchored Langevin flux (see :mod:`target`); the extra term contributes the
flux ``-alpha a J grad_U0 pi = alpha J grad(exp(-U0)) * const``, whose
divergence is ``(div J) . grad f + trace(J Hess f) = 0 + 0`` -- the first term
vanishes because ``div J = 0`` and the second because a skew matrix has zero
Frobenius inner product with a symmetric one.  Consequently:

* **no ``div(J)`` correction is added** to the drift, and
* ``alpha = 0`` removes *only* the non-reversible term, leaving the reversible
  anchored baseline ``-a grad_U0`` completely untouched.

``J`` is never normalised by a state-dependent quantity.  The only rescaling
available is the constant swirl ``s`` (``J -> s J``), and only the product
``alpha * s`` matters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from config import ConstraintConfig, ProjectionConfig, SamplerConfig
from constraint import g_value
from nonreversible_matrix import construct_J, coordinate_triples, operator_norm_J
from projection import project_onto_K


class AnchoredTarget(Protocol):
    """Minimal interface the sampler needs from a target."""

    def U(self, w: np.ndarray) -> float: ...
    def U0(self, w: np.ndarray) -> float: ...
    def grad_U0(self, w: np.ndarray) -> np.ndarray: ...
    def log_a(self, w: np.ndarray) -> float: ...


@dataclass
class ChainOutput:
    """Full output of one chain, including the per-iteration diagnostics."""

    samples: np.ndarray                 # (n_stored, d) post burn-in, thinned
    trace: np.ndarray                   # (n_iterations, d) every state
    alpha: float
    step_size: float
    swirl: float
    seed: int
    runtime: float                      # wall-clock seconds for the chain
    initial_point: np.ndarray
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_iterations(self) -> int:
        return self.trace.shape[0]

    @property
    def d(self) -> int:
        return self.trace.shape[1]


def run_chain(
    target: AnchoredTarget,
    w_init: np.ndarray,
    constraint: ConstraintConfig,
    sampler_cfg: SamplerConfig,
    projection_cfg: ProjectionConfig | None = None,
) -> ChainOutput:
    """Run one projected non-reversible anchored Langevin chain.

    ``sampler_cfg.step_size`` must be set (``experiment.derive_step_size``
    fills it in from the curvature of ``U0`` when it is ``None``).
    """
    if sampler_cfg.step_size is None:
        raise ValueError("sampler_cfg.step_size must be set before running a chain")
    if projection_cfg is None:
        projection_cfg = ProjectionConfig()

    p = constraint.p_constraint
    eps = constraint.epsilon_constraint
    Lambda = constraint.Lambda_constraint
    alpha = sampler_cfg.alpha
    swirl = sampler_cfg.swirl
    step_size = sampler_cfg.step_size
    n_iterations = sampler_cfg.n_iterations
    triples = coordinate_triples(int(np.asarray(w_init).size))

    rng = np.random.default_rng(sampler_cfg.seed)

    w = np.asarray(w_init, dtype=float).copy()
    # Every initial point is projected onto K before the first step.
    w = project_onto_K(w, constraint, projection_cfg).z
    d = w.size
    identity = np.eye(d)

    trace = np.empty((n_iterations, d), dtype=float)
    record = sampler_cfg.record_diagnostics
    if record:
        diag_U = np.empty(n_iterations)
        diag_U0 = np.empty(n_iterations)
        diag_log_a = np.empty(n_iterations)
        diag_a = np.empty(n_iterations)
        diag_g = np.empty(n_iterations)
        diag_slack = np.empty(n_iterations)
        diag_drift = np.empty(n_iterations)
        diag_drift_rev = np.empty(n_iterations)
        diag_drift_nonrev = np.empty(n_iterations)
        diag_opnorm = np.empty(n_iterations)
        diag_projected = np.empty(n_iterations, dtype=bool)
        diag_proj_dist = np.empty(n_iterations)
        diag_grad_norm = np.empty(n_iterations)
        diag_elapsed = np.empty(n_iterations)

    start = time.perf_counter()
    for k in range(n_iterations):
        grad_k = target.grad_U0(w)
        log_a_k = target.log_a(w)
        a_k = np.exp(log_a_k)

        # Reversible part: -a * grad.  Non-reversible part: -a * alpha * J @ grad.
        reversible = -a_k * grad_k
        if alpha != 0.0:
            J_k = construct_J(w, p, eps, swirl, triples)
            nonreversible = -a_k * alpha * (J_k @ grad_k)
        else:
            # alpha = 0 removes ONLY the non-reversible term.
            nonreversible = np.zeros(d)
        drift_k = reversible + nonreversible

        xi_k = rng.standard_normal(d)
        proposal = w + step_size * drift_k + np.sqrt(2.0 * step_size * a_k) * xi_k

        if not np.all(np.isfinite(proposal)):
            raise FloatingPointError(
                f"non-finite proposal at iteration {k} (alpha={alpha}, "
                f"step_size={step_size}); reduce the step size"
            )

        projection = project_onto_K(proposal, constraint, projection_cfg)
        w_next = projection.z

        # trace[k] is the state w_k at the START of iteration k, so every
        # recorded diagnostic below refers to the same point; the projection
        # entries describe the transition w_k -> w_{k+1}.
        trace[k] = w
        if record:
            diag_U[k] = target.U(w)
            diag_U0[k] = target.U0(w)
            diag_log_a[k] = log_a_k
            diag_a[k] = a_k
            g_k = float(g_value(w, p, eps))
            diag_g[k] = g_k
            diag_slack[k] = Lambda - g_k
            diag_drift[k] = np.linalg.norm(drift_k)
            diag_drift_rev[k] = np.linalg.norm(reversible)
            diag_drift_nonrev[k] = np.linalg.norm(nonreversible)
            diag_opnorm[k] = operator_norm_J(w, p, eps, swirl, triples)
            diag_projected[k] = projection.projected
            diag_proj_dist[k] = projection.distance
            diag_grad_norm[k] = np.linalg.norm(grad_k)
            diag_elapsed[k] = time.perf_counter() - start

        w = w_next

    runtime = time.perf_counter() - start

    stored_index = np.arange(sampler_cfg.burn_in, n_iterations, sampler_cfg.thin)
    samples = trace[stored_index].copy()

    diagnostics: dict[str, np.ndarray] = {}
    if record:
        diagnostics = {
            "U": diag_U,
            "U0": diag_U0,
            "log_a": diag_log_a,
            "a": diag_a,
            "g": diag_g,
            "slack": diag_slack,
            "drift_norm": diag_drift,
            "reversible_drift_norm": diag_drift_rev,
            "nonreversible_drift_norm": diag_drift_nonrev,
            "operator_norm_J": diag_opnorm,
            "projected": diag_projected,
            "projection_distance": diag_proj_dist,
            "grad_norm": diag_grad_norm,
            "elapsed": diag_elapsed,
            "stored_index": stored_index,
        }

    return ChainOutput(
        samples=samples,
        trace=trace,
        alpha=alpha,
        step_size=step_size,
        swirl=swirl,
        seed=sampler_cfg.seed,
        runtime=runtime,
        initial_point=np.asarray(w_init, dtype=float).copy(),
        diagnostics=diagnostics,
    )


def run_chains(
    target: AnchoredTarget,
    initial_points: np.ndarray,
    constraint: ConstraintConfig,
    sampler_cfg: SamplerConfig,
    projection_cfg: ProjectionConfig | None = None,
) -> list[ChainOutput]:
    """Run ``len(initial_points)`` independent chains.

    Chain ``c`` uses seed ``sampler_cfg.seed + c``.  The same seeds and the
    same ``initial_points`` are reused for every value of ``alpha``, so the
    comparison is paired.
    """
    outputs: list[ChainOutput] = []
    for chain_index, w_init in enumerate(np.atleast_2d(initial_points)):
        cfg = SamplerConfig(
            n_iterations=sampler_cfg.n_iterations,
            burn_in=sampler_cfg.burn_in,
            thin=sampler_cfg.thin,
            step_size=sampler_cfg.step_size,
            step_scale=sampler_cfg.step_scale,
            alpha=sampler_cfg.alpha,
            swirl=sampler_cfg.swirl,
            seed=sampler_cfg.seed + chain_index,
            record_diagnostics=sampler_cfg.record_diagnostics,
        )
        outputs.append(run_chain(target, w_init, constraint, cfg, projection_cfg))
    return outputs


def stack_samples(chains: list[ChainOutput]) -> np.ndarray:
    """Stack retained draws into the ArviZ layout ``(n_chains, n_draws, d)``."""
    return np.stack([chain.samples for chain in chains], axis=0)
