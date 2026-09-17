"""Projected Non-Reversible Anchored Langevin (PNRAL) sampler.

One iteration
-------------
At state ``w_k`` compute

    ``anchor_gradient = grad U0(w_k)``,
    ``log_a_k = log a(w_k)``,  ``a_k = exp(log_a_k)``,
    ``J_k = construct_J(w_k)``,

and form

    reversible drift        ``-a_k * anchor_gradient``
    non-reversible drift    ``-alpha * a_k * J_k @ anchor_gradient``
    total drift             ``-a_k (I + alpha J_k) @ anchor_gradient`` ,

so that with ``xi_k ~ N(0, I_d)``

.. math::

    w_{k+1} = \\Pi_K\\Big[ w_k
        - h\\, a_k (I_d + \\alpha J_k)\\,\\nabla U_0(w_k)
        + \\sqrt{2 h a_k}\\, \\xi_k \\Big].

``alpha = 0`` is the constrained *reversible* anchored baseline: it removes
the non-reversible drift and nothing else (same gradient, same anchor, same
noise, same projection, same seed).  Because ``div J = 0`` exactly, no
``div J`` correction appears in the drift.

The scheme is an unadjusted (Euler--Maruyama) discretisation followed by a
metric projection; it therefore carries the usual O(h) discretisation bias,
which is exactly what the step-size sensitivity study of section 12 probes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from .anchor import log_anchor_coefficient, value_and_grad_U0
from .constraint import g_constraint, grad_psi_constraint
from .nonreversible_matrix import make_J_builder, operator_norm_J
from .projection import ProjectionResult, project_K
from .target import LogisticTarget

__all__ = [
    "JSpec",
    "ChainResult",
    "AlphaRun",
    "run_chain",
    "run_chains",
    "smooth_map",
    "constrained_map",
    "initial_points",
    "calibrate_step_size",
]


@dataclass
class JSpec:
    """Everything needed to build ``J(w)`` and its operator norm."""

    builder: Callable[[np.ndarray], np.ndarray]
    triples: List[Tuple[int, int, int]]
    block_scale: float
    swirl_scale: float
    mode: str
    p_constraint: float
    epsilon_constraint: float

    @property
    def is_disjoint(self) -> bool:
        flat = [i for t in self.triples for i in t]
        return len(set(flat)) == len(flat)

    @classmethod
    def create(cls, mode: str, d: int, p_constraint: float,
               epsilon_constraint: float, swirl_scale: float) -> "JSpec":
        builder, triples, block_scale = make_J_builder(
            mode, d, p_constraint, epsilon_constraint, swirl_scale)
        return cls(builder=builder, triples=triples, block_scale=block_scale,
                   swirl_scale=swirl_scale, mode=mode,
                   p_constraint=p_constraint,
                   epsilon_constraint=epsilon_constraint)


@dataclass
class ChainResult:
    """Per-iteration output of a single chain.

    ``states[k]`` is ``w_k`` (the state at the *start* of iteration ``k``) and
    every recorded scalar is evaluated at that same state, so the diagnostic
    traces and the samples are exactly aligned.
    """

    states: np.ndarray                 # (n_iterations, d)
    U: np.ndarray
    U0: np.ndarray
    log_a: np.ndarray
    a: np.ndarray
    g: np.ndarray
    slack: np.ndarray                  # Lambda_constraint - g(w_k)
    grad_norm: np.ndarray
    reversible_drift_norm: np.ndarray
    nonreversible_drift_norm: np.ndarray
    total_drift_norm: np.ndarray
    J_operator_norm: np.ndarray
    projected: np.ndarray              # bool
    projection_distance: np.ndarray
    elapsed: np.ndarray                # cumulative wall-clock seconds
    alpha: float
    step_size: float
    seed: int
    burn_in: int
    thinning: int
    runtime: float
    gradient_evaluations: int
    Lambda_constraint: Optional[float]
    constrained: bool

    @property
    def retained(self) -> np.ndarray:
        """Post-burn-in, thinned samples."""
        return self.states[self.burn_in::self.thinning]

    @property
    def n_retained(self) -> int:
        return int(self.retained.shape[0])

    @property
    def projection_frequency(self) -> float:
        """Fraction of *retained* iterations at which a projection occurred."""
        return float(np.mean(self.projected[self.burn_in::self.thinning]))

    @property
    def mean_projection_distance(self) -> float:
        distances = self.projection_distance[self.burn_in::self.thinning]
        active = distances[distances > 0.0]
        return float(active.mean()) if active.size else 0.0

    @property
    def mean_squared_jumping_distance(self) -> float:
        """MSJD of the retained trace, ``mean_k ||w_{k+1} - w_k||^2``."""
        retained = self.retained
        if retained.shape[0] < 2:
            return float("nan")
        differences = np.diff(retained, axis=0)
        return float(np.mean(np.sum(differences * differences, axis=1)))

    def is_finite(self) -> bool:
        return bool(np.all(np.isfinite(self.states)) and np.all(np.isfinite(self.U0)))


@dataclass
class AlphaRun:
    """All chains for one value of ``alpha``."""

    alpha: float
    chains: List[ChainResult]
    step_size: float
    Lambda_constraint: float
    triples_mode: str
    label: str = ""

    @property
    def samples(self) -> np.ndarray:
        """Retained draws stacked as ``(n_chains, n_draws, d)`` for ArviZ."""
        return np.stack([chain.retained for chain in self.chains], axis=0)

    @property
    def runtime(self) -> float:
        return float(sum(chain.runtime for chain in self.chains))

    @property
    def gradient_evaluations(self) -> int:
        return int(sum(chain.gradient_evaluations for chain in self.chains))


# ----------------------------------------------------------------------
def run_chain(
    target: LogisticTarget,
    w0: np.ndarray,
    alpha: float,
    step_size: float,
    n_iterations: int,
    seed: int,
    j_spec: JSpec,
    Lambda_constraint: Optional[float],
    p_constraint: float,
    epsilon_constraint: float,
    burn_in: int = 0,
    thinning: int = 1,
    constrained: bool = True,
    projection_use_brentq: bool = False,
    projection_tolerance: float = 1e-13,
) -> ChainResult:
    """Run one PNRAL chain.

    Parameters
    ----------
    constrained:
        ``False`` (used by the pilot chain of section 6) skips the projection
        entirely, giving the *unconstrained* reversible anchored sampler.
    Lambda_constraint:
        Required when ``constrained`` is ``True``.  Diagnostics still report
        ``g(w_k)`` when unconstrained (with ``slack = nan``).

    Notes
    -----
    ``alpha == 0`` short-circuits the non-reversible term: the drift is then
    exactly ``-a_k grad U0(w_k)``, bit-for-bit identical to adding
    ``-0 * a_k J_k grad U0(w_k)``.
    """
    if constrained and Lambda_constraint is None:
        raise ValueError("a constrained chain needs Lambda_constraint")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if burn_in >= n_iterations:
        raise ValueError("burn_in must be smaller than n_iterations")

    rng = np.random.default_rng(seed)
    d = target.d
    w = np.array(w0, dtype=np.float64, copy=True)

    states = np.empty((n_iterations, d), dtype=np.float64)
    U_trace = np.empty(n_iterations, dtype=np.float64)
    U0_trace = np.empty(n_iterations, dtype=np.float64)
    log_a_trace = np.empty(n_iterations, dtype=np.float64)
    a_trace = np.empty(n_iterations, dtype=np.float64)
    g_trace = np.empty(n_iterations, dtype=np.float64)
    slack_trace = np.empty(n_iterations, dtype=np.float64)
    grad_norm_trace = np.empty(n_iterations, dtype=np.float64)
    reversible_trace = np.empty(n_iterations, dtype=np.float64)
    nonreversible_trace = np.empty(n_iterations, dtype=np.float64)
    total_trace = np.empty(n_iterations, dtype=np.float64)
    J_norm_trace = np.empty(n_iterations, dtype=np.float64)
    projected_trace = np.zeros(n_iterations, dtype=bool)
    distance_trace = np.zeros(n_iterations, dtype=np.float64)
    elapsed_trace = np.empty(n_iterations, dtype=np.float64)

    use_J = alpha != 0.0
    gradient_evaluations_before = target.gradient_evaluations
    start = time.perf_counter()

    for k in range(n_iterations):
        states[k] = w

        # --- anchor value, gradient and coefficient ---------------------
        U0_k, anchor_gradient = value_and_grad_U0(target, w)
        log_a_k = log_anchor_coefficient(target, w)
        a_k = float(np.exp(log_a_k))
        # U = U0 + log a holds identically (see pnral.anchor).
        U_k = U0_k + log_a_k

        # --- drifts ------------------------------------------------------
        reversible_drift = -a_k * anchor_gradient
        if use_J:
            J_k = j_spec.builder(w)
            nonreversible_drift = -alpha * a_k * (J_k @ anchor_gradient)
            total_drift = reversible_drift + nonreversible_drift
            nonreversible_norm = float(np.linalg.norm(nonreversible_drift))
        else:
            J_k = None
            nonreversible_norm = 0.0
            total_drift = reversible_drift

        # --- operator norm of J (diagnostic; analytic when disjoint) -----
        grad_psi = grad_psi_constraint(w, p_constraint, epsilon_constraint)
        if j_spec.is_disjoint:
            J_norm = operator_norm_J(np.empty((0, 0)), j_spec.triples, grad_psi,
                                     j_spec.block_scale, j_spec.swirl_scale)
        else:
            J_norm = operator_norm_J(J_k if J_k is not None else j_spec.builder(w))

        # --- Euler--Maruyama proposal ------------------------------------
        xi = rng.standard_normal(d)
        proposal = w + step_size * total_drift + np.sqrt(2.0 * step_size * a_k) * xi

        # --- projection ---------------------------------------------------
        if constrained:
            result: ProjectionResult = project_K(
                proposal, p_constraint, epsilon_constraint, float(Lambda_constraint),
                use_brentq=projection_use_brentq, tolerance=projection_tolerance)
            w_next = result.z
            projected_trace[k] = result.projected
            distance_trace[k] = result.distance
        else:
            w_next = proposal

        g_k = g_constraint(w, p_constraint, epsilon_constraint)
        U_trace[k] = U_k
        U0_trace[k] = U0_k
        log_a_trace[k] = log_a_k
        a_trace[k] = a_k
        g_trace[k] = g_k
        slack_trace[k] = (float(Lambda_constraint) - g_k
                          if Lambda_constraint is not None else np.nan)
        grad_norm_trace[k] = float(np.linalg.norm(anchor_gradient))
        reversible_trace[k] = float(np.linalg.norm(reversible_drift))
        nonreversible_trace[k] = nonreversible_norm
        total_trace[k] = float(np.linalg.norm(total_drift))
        J_norm_trace[k] = J_norm
        elapsed_trace[k] = time.perf_counter() - start

        w = w_next
        if not np.all(np.isfinite(w)):
            raise FloatingPointError(
                f"chain diverged at iteration {k} (alpha={alpha}, h={step_size})")

    runtime = time.perf_counter() - start
    return ChainResult(
        states=states, U=U_trace, U0=U0_trace, log_a=log_a_trace, a=a_trace,
        g=g_trace, slack=slack_trace, grad_norm=grad_norm_trace,
        reversible_drift_norm=reversible_trace,
        nonreversible_drift_norm=nonreversible_trace,
        total_drift_norm=total_trace, J_operator_norm=J_norm_trace,
        projected=projected_trace, projection_distance=distance_trace,
        elapsed=elapsed_trace, alpha=float(alpha), step_size=float(step_size),
        seed=int(seed), burn_in=int(burn_in), thinning=int(thinning),
        runtime=float(runtime),
        gradient_evaluations=int(target.gradient_evaluations
                                 - gradient_evaluations_before),
        Lambda_constraint=(None if Lambda_constraint is None
                           else float(Lambda_constraint)),
        constrained=bool(constrained),
    )


def run_chains(
    target: LogisticTarget,
    initial_states: Sequence[np.ndarray],
    alpha: float,
    step_size: float,
    n_iterations: int,
    j_spec: JSpec,
    Lambda_constraint: Optional[float],
    p_constraint: float,
    epsilon_constraint: float,
    burn_in: int,
    thinning: int,
    chain_seed_base: int,
    constrained: bool = True,
    projection_use_brentq: bool = False,
    projection_tolerance: float = 1e-13,
    label: str = "",
) -> AlphaRun:
    """Run ``len(initial_states)`` chains that differ only in their seed.

    Chain ``c`` uses seed ``chain_seed_base + c`` for *every* value of alpha,
    so the driving noise is common across the alpha comparison.
    """
    chains = [
        run_chain(target=target, w0=w0, alpha=alpha, step_size=step_size,
                  n_iterations=n_iterations, seed=chain_seed_base + index,
                  j_spec=j_spec, Lambda_constraint=Lambda_constraint,
                  p_constraint=p_constraint, epsilon_constraint=epsilon_constraint,
                  burn_in=burn_in, thinning=thinning, constrained=constrained,
                  projection_use_brentq=projection_use_brentq,
                  projection_tolerance=projection_tolerance)
        for index, w0 in enumerate(initial_states)
    ]
    return AlphaRun(alpha=float(alpha), chains=chains, step_size=float(step_size),
                    Lambda_constraint=(np.nan if Lambda_constraint is None
                                       else float(Lambda_constraint)),
                    triples_mode=j_spec.mode, label=label)


# ----------------------------------------------------------------------
# Initialisation (section 13)
# ----------------------------------------------------------------------
def smooth_map(target: LogisticTarget,
               w_init: Optional[np.ndarray] = None) -> np.ndarray:
    """Unconstrained smooth MAP: minimise ``U0`` with L-BFGS-B."""
    if w_init is None:
        w_init = np.zeros(target.d, dtype=np.float64)

    def objective(w: np.ndarray) -> Tuple[float, np.ndarray]:
        value, gradient = value_and_grad_U0(target, w)
        return value, gradient

    result = minimize(objective, w_init, jac=True, method="L-BFGS-B",
                      options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-10})
    if not result.success and not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"smooth MAP optimisation failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def constrained_map(target: LogisticTarget, p_constraint: float,
                    epsilon_constraint: float, Lambda_constraint: float,
                    projection_use_brentq: bool = False) -> Dict[str, object]:
    """Smooth MAP subject to ``g(w) <= Lambda_constraint``.

    Uses SLSQP with the non-linear constraint and falls back to *projecting*
    the unconstrained smooth MAP onto ``K`` if the optimiser fails or returns
    an infeasible point.  Both routes are permitted by the specification; the
    route actually taken is reported.
    """
    unconstrained = smooth_map(target)
    projected_fallback = project_K(unconstrained, p_constraint, epsilon_constraint,
                                   Lambda_constraint,
                                   use_brentq=projection_use_brentq).z

    def objective(w: np.ndarray) -> Tuple[float, np.ndarray]:
        return value_and_grad_U0(target, w)

    constraint = NonlinearConstraint(
        lambda w: g_constraint(w, p_constraint, epsilon_constraint),
        -np.inf, Lambda_constraint)
    try:
        result = minimize(objective, projected_fallback, jac=True, method="SLSQP",
                          constraints=[{
                              "type": "ineq",
                              "fun": lambda w: Lambda_constraint - g_constraint(
                                  w, p_constraint, epsilon_constraint),
                              "jac": lambda w: -p_constraint * w * (
                                  w * w + epsilon_constraint ** 2) ** (0.5 * p_constraint - 1.0),
                          }],
                          options={"maxiter": 800, "ftol": 1e-12})
        candidate = np.asarray(result.x, dtype=np.float64)
        feasible = (g_constraint(candidate, p_constraint, epsilon_constraint)
                    <= Lambda_constraint + 1e-9)
        ok = bool(result.success) and feasible and np.all(np.isfinite(candidate))
    except Exception:                                  # pragma: no cover
        ok, candidate = False, projected_fallback

    if ok:
        # Re-project to remove any O(1e-12) constraint overshoot from SLSQP.
        candidate = project_K(candidate, p_constraint, epsilon_constraint,
                              Lambda_constraint,
                              use_brentq=projection_use_brentq).z
        route = "slsqp"
    else:
        candidate = projected_fallback
        route = "projected_unconstrained_map"

    return {
        "w_map_unconstrained": unconstrained,
        "w_map_constrained": candidate,
        "route": route,
        "U0_unconstrained": float(value_and_grad_U0(target, unconstrained)[0]),
        "U0_constrained": float(value_and_grad_U0(target, candidate)[0]),
        "g_unconstrained": float(g_constraint(unconstrained, p_constraint,
                                              epsilon_constraint)),
        "g_constrained": float(g_constraint(candidate, p_constraint,
                                            epsilon_constraint)),
    }


def initial_points(w_map: np.ndarray, number_of_chains: int,
                   perturbation_scale: float, seed: int,
                   p_constraint: float, epsilon_constraint: float,
                   Lambda_constraint: float,
                   projection_use_brentq: bool = False) -> List[np.ndarray]:
    """``w_0 = Pi_K(constrained MAP + small Gaussian perturbation)``.

    The same policy, the same seed and therefore the *same* initial points are
    used for every value of alpha.
    """
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(number_of_chains):
        perturbed = w_map + perturbation_scale * rng.standard_normal(w_map.shape)
        points.append(project_K(perturbed, p_constraint, epsilon_constraint,
                                Lambda_constraint,
                                use_brentq=projection_use_brentq).z)
    return points


# ----------------------------------------------------------------------
# Step-size calibration (section 12)
# ----------------------------------------------------------------------
def calibrate_step_size(
    target: LogisticTarget,
    w0: np.ndarray,
    alpha_max: float,
    j_spec: JSpec,
    Lambda_constraint: float,
    p_constraint: float,
    epsilon_constraint: float,
    safety: float = 0.5,
    max_halvings: int = 12,
    trial_iterations: int = 1500,
    max_projection_frequency: float = 0.25,
    seed: int = 999,
    projection_use_brentq: bool = False,
) -> Dict[str, object]:
    """Pick a common step size that is stable at the **largest** tested alpha.

    The search starts from the smoothness guide

        ``h_0 = safety / L``,   ``L <= 0.25 * ||X||_2^2``   (section 12)

    and halves ``h`` until a short trial chain at ``alpha_max``

    * stays finite,
    * does not drift to a much higher potential than its start, and
    * projects on at most ``max_projection_frequency`` of its iterations.

    The guide is only a starting point; acceptance is decided empirically.
    """
    L_guide = target.likelihood_smoothness_guide()
    step_size = safety / L_guide
    history: List[Dict[str, object]] = []
    U0_start = float(value_and_grad_U0(target, w0)[0])

    for attempt in range(max_halvings + 1):
        record: Dict[str, object] = {"attempt": attempt, "step_size": step_size}
        try:
            trial = run_chain(target=target, w0=w0, alpha=alpha_max,
                              step_size=step_size, n_iterations=trial_iterations,
                              seed=seed, j_spec=j_spec,
                              Lambda_constraint=Lambda_constraint,
                              p_constraint=p_constraint,
                              epsilon_constraint=epsilon_constraint,
                              burn_in=trial_iterations // 2, thinning=1,
                              constrained=True,
                              projection_use_brentq=projection_use_brentq)
        except FloatingPointError as error:
            record.update({"accepted": False, "reason": str(error)})
            history.append(record)
            step_size *= 0.5
            continue

        tail = trial.U0[trial.burn_in:]
        finite = trial.is_finite() and bool(np.all(np.isfinite(tail)))
        projection_frequency = trial.projection_frequency
        # A stable unadjusted chain fluctuates about the MAP by O(d); a
        # runaway chain leaves that band by orders of magnitude.
        drift_ok = bool(np.mean(tail) < U0_start + 50.0 * target.d)
        accepted = finite and drift_ok and projection_frequency <= max_projection_frequency
        record.update({
            "accepted": bool(accepted),
            "finite": bool(finite),
            "mean_U0_tail": float(np.mean(tail)) if finite else float("nan"),
            "U0_start": U0_start,
            "projection_frequency": float(projection_frequency),
            "max_state_norm": float(np.max(np.linalg.norm(trial.states, axis=1))),
        })
        history.append(record)
        if accepted:
            return {"step_size": float(step_size), "L_guide": float(L_guide),
                    "safety": float(safety), "alpha_max": float(alpha_max),
                    "history": history, "accepted": True}
        step_size *= 0.5

    return {"step_size": float(step_size), "L_guide": float(L_guide),
            "safety": float(safety), "alpha_max": float(alpha_max),
            "history": history, "accepted": False}
