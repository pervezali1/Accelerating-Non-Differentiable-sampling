r"""The projected non-reversible anchored Langevin sampler.

One iteration, with ``g_k = \nabla U_0(w_k)``, ``a_k = e^{U(w_k) - U_0(w_k)}``
and ``J_k = J_s(w_k)``:

.. math::
    b_k &= -a_k (I + \alpha J_k) g_k, \qquad \xi_k \sim N(0, I_D), \\
    y_{k+1} &= w_k + h b_k + \sqrt{2 h a_k}\, \xi_k, \\
    w_{k+1} &= P_C(y_{k+1}).

``alpha = 0`` is the reversible anchored baseline.  The clock ``a_k`` multiplies
both the drift and the diffusion, which is what makes this a *time change* of
the smooth dynamics rather than a different dynamics: running
``dX = -a(X) \nabla U_0(X) dt + \sqrt{2 a(X)} dW`` at rate ``a`` leaves
``e^{-U_0}/a = e^{-U}`` invariant.  The skew term adds a divergence-free
circulation that leaves the same measure invariant and needs no ``div J``
correction (see :mod:`geometry`), and because ``q'J_s(w) = 0`` it is tangent to
the boundary of ``C``, so the projection never fights it.

The scheme is an unadjusted Euler discretisation: there is no Metropolis
correction, so the invariant law is ``O(h)`` away from the target.  That bias is
the reason the step size is calibrated conservatively and swept.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from config import ConstraintSpec, FieldConfig, PilotConfig, SamplerConfig
from geometry import (
    construct_J,
    generate_cyclic_triples,
    project_to_centered_ball,
)
from target import LogisticTarget, check_anchor_bounds

__all__ = [
    "SamplerResult",
    "find_smooth_map",
    "calibrate_delta",
    "run_unconstrained_pilot",
    "choose_radius_from_pilot",
    "run_constrained_sampler",
    "calibrate_step_size",
    "initial_point",
]


@dataclass
class SamplerResult:
    """Everything one chain recorded, plus the settings that produced it."""

    # ---- configuration, carried so a result is self-describing
    alpha: float
    h: float
    seed: int
    n_iter: int
    burn_in: int
    thin: int
    s: float
    center: np.ndarray
    radius: float
    projected_flag: bool

    # ---- trajectories
    trajectory: np.ndarray  # (n_iter + 1, D), includes w_0
    samples: np.ndarray  # (n_retained, D), post burn-in and thinning
    log_a: np.ndarray  # (n_iter,)
    a: np.ndarray  # (n_iter,)
    radius_trace: np.ndarray  # (n_iter + 1,)  ||w_k - center||
    drift_norm: np.ndarray  # (n_iter,)  ||b_k||
    reversible_drift_norm: np.ndarray  # (n_iter,)  ||a_k g_k||
    nonreversible_drift_norm: np.ndarray  # (n_iter,)  ||a_k alpha J_k g_k||
    projection_occurred: np.ndarray  # (n_iter,) bool

    # ---- cost and health
    runtime_seconds: float
    n_grad_evaluations: int
    n_numerical_warnings: int

    @property
    def dim(self) -> int:
        return self.trajectory.shape[1]

    @property
    def projection_frequency(self) -> float:
        """Fraction of proposals the constraint actually moved."""
        return float(self.projection_occurred.mean()) if self.n_iter else 0.0

    @property
    def mean_squared_jump(self) -> float:
        """MSJD over the post-burn-in trajectory, ``mean ||w_{k+1} - w_k||^2``."""
        post = self.trajectory[self.burn_in :]
        if post.shape[0] < 2:
            return float("nan")
        return float((np.diff(post, axis=0) ** 2).sum(axis=1).mean())

    def summary(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "h": self.h,
            "seed": self.seed,
            "n_iter": self.n_iter,
            "n_samples": int(self.samples.shape[0]),
            "runtime_seconds": self.runtime_seconds,
            "n_grad_evaluations": self.n_grad_evaluations,
            "projection_frequency": self.projection_frequency,
            "mean_squared_jump": self.mean_squared_jump,
            "a_mean": float(self.a.mean()),
            "a_min": float(self.a.min()),
            "log_a_min": float(self.log_a.min()),
            "radius_mean": float(self.radius_trace.mean()),
            "radius_max": float(self.radius_trace.max()),
            "drift_norm_mean": float(self.drift_norm.mean()),
            "reversible_drift_norm_mean": float(self.reversible_drift_norm.mean()),
            "nonreversible_drift_norm_mean": float(
                self.nonreversible_drift_norm.mean()
            ),
            "n_numerical_warnings": self.n_numerical_warnings,
        }


# ------------------------------------------------------------------- MAP, step
def find_smooth_map(
    target: LogisticTarget,
    w_init: np.ndarray | None = None,
    max_iter: int = 2_000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, object]:
    r"""Minimise the *smooth* anchor ``U0`` with L-BFGS-B.

    The anchor is used rather than the kinked target precisely because L-BFGS-B
    assumes a differentiable objective; ``U0`` is smooth and strictly convex
    here (the logistic Hessian is positive semidefinite, the intercept prior and
    the smoothed penalty are positive definite on their coordinates), so the
    minimiser is unique and a good centre for the constraint.
    """
    D = target.dim
    w0 = np.zeros(D) if w_init is None else np.asarray(w_init, float).ravel()
    result = minimize(
        fun=lambda w: target.U0(w),
        x0=w0,
        jac=lambda w: target.grad_U0(w),
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )
    w_map = np.ascontiguousarray(result.x, dtype=np.float64)
    if not np.isfinite(w_map).all():
        raise RuntimeError("L-BFGS-B returned a non-finite MAP estimate")
    if not result.success:
        warnings.warn(
            f"L-BFGS-B did not report success: {result.message!r}; "
            f"||grad U0|| = {np.linalg.norm(target.grad_U0(w_map)):.3e}",
            RuntimeWarning,
        )
    return w_map, result


def calibrate_delta(
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    sigma0: float,
    candidates: np.ndarray | None = None,
    safety: float = 0.25,
    tempered: bool = False,
) -> tuple[float, list[dict[str, float]]]:
    r"""Pick the anchor smoothing ``delta`` that maximises the effective step.

    ``delta`` is worth calibrating and is *safe* to calibrate, because it enters
    the **anchor only**: ``U`` does not contain ``delta``, so changing it moves
    neither the target posterior nor the constraint, only the algorithm's
    efficiency.  It trades off two ways at once, and the trade is sharp:

    * the clock is bounded by ``a >= exp(-lambda p delta)``, and near a
      coefficient the LASSO has driven to zero the gap per term is about
      ``delta``, so ``a`` falls roughly like ``exp(-lambda k delta)`` with ``k``
      the number of such coefficients -- large ``delta`` stalls the clock;
    * the anchor's curvature at such a coefficient is ``lambda / delta``, and
      the stable step size goes like its reciprocal -- small ``delta`` forces a
      small ``h``.

    The product ``h(delta) * a(delta)`` is what actually advances the dynamics
    per iteration, and it has an interior maximum.  This evaluates it at the
    smooth MAP for each candidate (each needs its own MAP, since ``U0`` depends
    on ``delta``) and returns the best, together with the whole table so the
    choice is auditable rather than asserted.
    """
    if candidates is None:
        candidates = np.geomspace(3e-4, 3e-2, 13)
    rows: list[dict[str, float]] = []
    for delta in np.asarray(candidates, dtype=np.float64):
        probe = LogisticTarget(X, y, lambda_, sigma0, float(delta), tempered)
        w_map, _ = find_smooth_map(probe)
        a_map = probe.a(w_map)
        ev = np.linalg.eigvalsh(probe.hessian_U0(w_map))
        h = safety / float(ev.max())
        rows.append(
            {
                "delta": float(delta),
                "a_at_map": float(a_map),
                "curvature_max": float(ev.max()),
                "curvature_min": float(ev.min()),
                "h": float(h),
                "effective_step": float(h * a_map),
                # iterations for the slowest mode to relax once
                "relaxation_iterations": float(1.0 / (h * a_map * float(ev.min()))),
                "n_near_zero": float(np.sum(np.abs(w_map[1:]) < 2.0 * delta)),
            }
        )
    best = max(rows, key=lambda r: r["effective_step"])
    return best["delta"], rows


def calibrate_step_size(
    target: LogisticTarget, w_center: np.ndarray, safety: float = 0.1
) -> tuple[float, float]:
    r"""``h = safety / lambda_max(Hess U0(w_center))``, and that eigenvalue.

    Explicit Euler on a quadratic with curvature ``L`` is stable only for
    ``h a L < 2``; ``a <= 1`` always, so using the largest eigenvalue of the
    anchor Hessian at the centre and a safety factor well below 2 keeps the
    discretisation away from its stability boundary.  The binding curvature here
    is usually not the likelihood's but the smoothed penalty's, ``lambda /
    delta`` at a coefficient the LASSO has driven onto its kink.
    """
    H = target.hessian_U0(w_center)
    lambda_max = float(np.linalg.eigvalsh(H).max())
    if not np.isfinite(lambda_max) or lambda_max <= 0.0:
        raise RuntimeError(f"invalid anchor curvature {lambda_max!r}")
    return safety / lambda_max, lambda_max


def initial_point(
    center: np.ndarray,
    radius: float | None,
    init_scale: float,
    seed: int,
    project: bool = True,
) -> np.ndarray:
    r"""``w_0 = P_C(center + init_scale * xi)``, ``xi`` standard normal.

    Seeded only by ``seed`` -- never by ``alpha`` -- so chain ``c`` starts from
    the same point for every ``alpha`` being compared.
    """
    center = np.asarray(center, dtype=np.float64).ravel()
    rng = np.random.default_rng([seed, 0xC0FFEE])
    w0 = center + init_scale * rng.normal(size=center.size)
    if project and radius is not None:
        w0, _ = project_to_centered_ball(w0, center, radius)
    return w0


# ----------------------------------------------------------------- the kernel
def _run_chain(
    target: LogisticTarget,
    center: np.ndarray,
    radius: float | None,
    triples: tuple[tuple[int, int, int], ...],
    cfg: SamplerConfig,
    s: float,
) -> SamplerResult:
    r"""The projected anchored Euler loop, shared by the pilot and the final runs.

    ``radius=None`` together with ``cfg.project=False`` gives the unconstrained
    chain used to choose the radius; everything else is identical, so the pilot
    and the final chains cannot drift apart in their dynamics.
    """
    D = target.dim
    center = np.asarray(center, dtype=np.float64).ravel()
    if center.size != D:
        raise ValueError(f"center has length {center.size}, expected D={D}")
    use_projection = bool(cfg.project and radius is not None)
    use_skew = cfg.alpha != 0.0 and s != 0.0

    n = cfg.n_iter
    traj = np.empty((n + 1, D), dtype=np.float64)
    log_a = np.empty(n, dtype=np.float64)
    a_trace = np.empty(n, dtype=np.float64)
    radius_trace = np.empty(n + 1, dtype=np.float64)
    drift_norm = np.empty(n, dtype=np.float64)
    rev_norm = np.empty(n, dtype=np.float64)
    nonrev_norm = np.empty(n, dtype=np.float64)
    projected = np.zeros(n, dtype=bool)

    w = initial_point(center, radius, cfg.init_scale, cfg.seed, use_projection)
    traj[0] = w
    radius_trace[0] = np.linalg.norm(w - center)

    rng = np.random.default_rng([cfg.seed, 0xA9C40])
    n_grad = 0
    n_warn = 0
    sqrt2h = np.sqrt(2.0 * cfg.h)
    started = time.perf_counter()

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for k in range(n):
            q = w - center
            g = target.grad_U0(w)
            n_grad += 1
            la = target.log_a(w)
            ak = np.exp(la)

            # b = -a (I + alpha J) g, split so both halves can be recorded
            rev = ak * g
            if use_skew:
                J = construct_J(w, center, triples, s)
                nonrev = ak * cfg.alpha * (J @ g)
            else:
                nonrev = np.zeros(D)
            b = -(rev + nonrev)

            xi = rng.normal(size=D)
            y = w + cfg.h * b + sqrt2h * np.sqrt(ak) * xi

            if not np.isfinite(y).all():
                n_warn += 1
                raise FloatingPointError(
                    f"non-finite proposal at iteration {k} (alpha={cfg.alpha}, "
                    f"h={cfg.h:.3e}); the step size is too large for this target"
                )

            if use_projection:
                w, hit = project_to_centered_ball(y, center, radius)
            else:
                w, hit = y, False

            log_a[k] = la
            a_trace[k] = ak
            drift_norm[k] = np.linalg.norm(b)
            rev_norm[k] = np.linalg.norm(rev)
            nonrev_norm[k] = np.linalg.norm(nonrev)
            projected[k] = hit
            traj[k + 1] = w
            radius_trace[k + 1] = np.linalg.norm(w - center)

    runtime = time.perf_counter() - started

    keep = np.arange(cfg.burn_in, n, cfg.thin)
    samples = np.ascontiguousarray(traj[1:][keep]) if keep.size else traj[:0]

    check_anchor_bounds(log_a, D, target.lambda_, target.delta)
    if not np.isfinite(traj).all():
        raise AssertionError("the stored trajectory contains non-finite values")

    return SamplerResult(
        alpha=cfg.alpha,
        h=cfg.h,
        seed=cfg.seed,
        n_iter=n,
        burn_in=cfg.burn_in,
        thin=cfg.thin,
        s=s,
        center=center,
        radius=float(radius) if radius is not None else float("inf"),
        projected_flag=use_projection,
        trajectory=traj if cfg.store_trajectory else traj[:: max(1, n // 2_000)],
        samples=samples,
        log_a=log_a,
        a=a_trace,
        radius_trace=radius_trace,
        drift_norm=drift_norm,
        reversible_drift_norm=rev_norm,
        nonreversible_drift_norm=nonrev_norm,
        projection_occurred=projected,
        runtime_seconds=runtime,
        n_grad_evaluations=n_grad,
        n_numerical_warnings=n_warn,
    )


def run_unconstrained_pilot(
    target: LogisticTarget,
    w_center: np.ndarray,
    h: float,
    cfg: PilotConfig | None = None,
) -> SamplerResult:
    r"""A reversible (``alpha = 0``), unprojected anchored chain.

    Its only job is to say how far the posterior actually wanders from the
    smooth MAP, so that the constraint can be placed where it is a mild
    regulariser rather than an active bound.  Using ``alpha = 0`` for this is
    deliberate: the radius must not depend on the quantity being compared.
    """
    cfg = cfg or PilotConfig()
    sampler_cfg = SamplerConfig(
        alpha=0.0,
        h=h,
        n_iter=cfg.n_iter,
        burn_in=cfg.burn_in,
        thin=cfg.thin,
        seed=cfg.seed,
        init_scale=cfg.init_scale,
        project=False,
        store_trajectory=True,
    )
    return _run_chain(
        target, w_center, None, generate_cyclic_triples(target.dim), sampler_cfg, 0.0
    )


def choose_radius_from_pilot(
    pilot: SamplerResult,
    w_center: np.ndarray,
    cfg: PilotConfig | None = None,
) -> ConstraintSpec:
    r"""``R = inflation * quantile(||w_k - center||, q)`` over the pilot samples.

    The exceedance fraction reported back is the fraction of *pilot* states that
    the resulting ball would have rejected.  With ``q = 0.999`` and an inflation
    of ``1.10`` it should be zero or nearly so; a large value means the pilot had
    not settled and the radius is being set by its transient.
    """
    cfg = cfg or PilotConfig()
    center = np.asarray(w_center, dtype=np.float64).ravel()
    post = pilot.trajectory[cfg.burn_in :]
    r = np.linalg.norm(post - center, axis=1)
    if r.size == 0:
        raise ValueError("the pilot produced no post-burn-in states")
    R = float(cfg.inflation * np.quantile(r, cfg.quantile))
    if not np.isfinite(R) or R <= 0.0:
        raise RuntimeError(f"invalid radius {R!r} from the pilot")
    exceed = float((r > R).mean())
    return ConstraintSpec(
        center=tuple(float(v) for v in center),
        radius=R,
        source="pilot",
        pilot_quantile=cfg.quantile,
        pilot_inflation=cfg.inflation,
        pilot_exceedance_fraction=exceed,
    )


def run_constrained_sampler(
    target: LogisticTarget,
    constraint: ConstraintSpec,
    cfg: SamplerConfig,
    field_cfg: FieldConfig | None = None,
) -> SamplerResult:
    r"""One constrained non-reversible anchored chain.

    ``constraint`` is the frozen :class:`~config.ConstraintSpec`; passing it as
    an object rather than as a loose centre and radius is what makes validation
    check 9 -- that the geometry is not recomputed per ``alpha`` -- structural
    rather than a matter of discipline.
    """
    field_cfg = field_cfg or FieldConfig()
    if constraint.dim != target.dim:
        raise ValueError(
            f"constraint is {constraint.dim}-dimensional but the target is "
            f"{target.dim}-dimensional"
        )
    triples = field_cfg.triples or generate_cyclic_triples(target.dim)
    center = np.asarray(constraint.center, dtype=np.float64)
    result = _run_chain(
        target, center, constraint.radius, triples, cfg, field_cfg.s
    )
    # validation check 1: every stored state is inside the constraint
    slack = result.radius_trace.max() - constraint.radius
    if slack > 1e-8 * max(1.0, constraint.radius):
        raise AssertionError(
            f"a stored state left the constraint by {slack:.3e} "
            f"(radius {constraint.radius:.6f})"
        )
    return result
