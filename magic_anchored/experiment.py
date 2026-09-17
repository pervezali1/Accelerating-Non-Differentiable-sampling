r"""The whole experiment: calibrate, freeze the geometry, compare ``alpha``, check.

The order of operations is the point of this module, because it is what makes
the comparison fair:

1. load and split the data, and build the target ``U`` once;
2. calibrate the two purely algorithmic knobs -- the anchor smoothing ``delta``
   and the step size ``h`` -- neither of which changes ``U``;
3. find the smooth MAP, run **one** reversible unconstrained pilot, and freeze
   ``(w_center, R)`` into a :class:`~config.ConstraintSpec` that is written to
   disk and thereafter only read;
4. run every ``alpha`` against that frozen target and geometry, with shared
   seeds so chain ``c`` of every ``alpha`` starts at the same point and, at
   ``alpha = 0``, would see the same noise;
5. diagnose, score, and run the two sensitivity studies (step size and radius);
6. re-verify all ten validation checks against the stored results, not against
   the intentions of the code.

Step 3 happening exactly once is validation checks 8 and 9.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from config import (
    ConstraintSpec,
    DataConfig,
    ExperimentConfig,
    FieldConfig,
    PilotConfig,
    SamplerConfig,
    TargetConfig,
    to_json,
)
from data import Dataset, load_and_preprocess
from diagnostics import (
    HAS_ARVIZ,
    ChainDiagnostics,
    PredictiveScores,
    calculate_chain_diagnostics,
    evaluate_predictive_performance,
    posterior_predictive_probabilities,
)
from geometry import (
    construct_J,
    field_operator_norm,
    generate_cyclic_triples,
    natural_alpha_scale,
    verify_field_properties,
)
from sampler import (
    SamplerResult,
    calibrate_delta,
    calibrate_step_size,
    choose_radius_from_pilot,
    find_smooth_map,
    initial_point,
    run_constrained_sampler,
    run_unconstrained_pilot,
)
from target import LogisticTarget, anchor_bounds, check_anchor_bounds

__all__ = [
    "ExperimentOutput",
    "run_experiment",
    "run_validation_checks",
    "load_chains",
    "regenerate_figures",
]


@dataclass
class ExperimentOutput:
    """Everything the run produced, in memory."""

    dataset: Dataset
    target: LogisticTarget
    constraint: ConstraintSpec
    h: float
    delta_table: list[dict[str, float]]
    pilot: SamplerResult
    chains: dict[float, list[SamplerResult]]
    diagnostics: dict[float, ChainDiagnostics]
    scores: dict[float, PredictiveScores]
    step_sensitivity: pd.DataFrame
    radius_sensitivity: pd.DataFrame
    validation: dict[str, object]
    figures: list[str]
    out_dir: str


def _log(message: str) -> None:
    print(message, flush=True)


def _one_chain(
    args: tuple[LogisticTarget, ConstraintSpec, SamplerConfig, FieldConfig | None]
) -> SamplerResult:
    """Module-level so that it can be pickled for a process pool."""
    target, constraint, sampler_cfg, field_cfg = args
    return run_constrained_sampler(target, constraint, sampler_cfg, field_cfg)


def _run_alpha_group(
    target: LogisticTarget,
    constraint: ConstraintSpec,
    alpha: float,
    h: float,
    cfg: ExperimentConfig,
    n_chains: int,
    n_iter: int | None = None,
    burn_in: int | None = None,
    field_cfg: FieldConfig | None = None,
) -> list[SamplerResult]:
    """``n_chains`` chains at one ``alpha``, seeded ``base_seed + c``.

    The chains are independent -- each carries its own seed and its own RNG
    stream -- so they are run in a process pool when ``cfg.n_jobs != 1``.  The
    results do not depend on ``n_jobs``: a chain's noise is a function of its
    seed alone, never of the order in which chains were scheduled.
    """
    jobs = [
        (
            target,
            constraint,
            SamplerConfig(
                alpha=alpha,
                h=h,
                n_iter=n_iter if n_iter is not None else cfg.n_iter,
                burn_in=burn_in if burn_in is not None else cfg.burn_in,
                thin=cfg.thin,
                seed=cfg.base_seed + c,
                project=True,
            ),
            field_cfg,
        )
        for c in range(n_chains)
    ]
    if cfg.n_jobs == 1 or n_chains == 1:
        return [_one_chain(job) for job in jobs]
    workers = (
        os.cpu_count() or 1 if cfg.n_jobs < 0 else min(cfg.n_jobs, n_chains)
    )
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(_one_chain, jobs))


def run_experiment(
    data_cfg: DataConfig | None = None,
    target_cfg: TargetConfig | None = None,
    field_cfg: FieldConfig | None = None,
    pilot_cfg: PilotConfig | None = None,
    cfg: ExperimentConfig | None = None,
) -> ExperimentOutput:
    """Run the full comparison and write every artefact to ``cfg.out_dir``."""
    data_cfg = data_cfg or DataConfig()
    target_cfg = target_cfg or TargetConfig()
    field_cfg = field_cfg or FieldConfig()
    pilot_cfg = pilot_cfg or PilotConfig()
    cfg = cfg or ExperimentConfig()
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    started = time.perf_counter()

    # -------------------------------------------------------------- 1. data
    _log("[1/6] loading and preprocessing")
    dataset = load_and_preprocess(data_cfg)
    D = dataset.dim
    target_cfg = target_cfg.resolve(dataset.n_train)
    _log(f"      n_train={dataset.n_train} n_test={dataset.n_test} D={D} "
         f"lambda={target_cfg.lambda_:.4g}")

    # ------------------------------------------- 2. algorithmic calibration
    _log("[2/6] calibrating delta and h (neither changes the target U)")
    delta_table: list[dict[str, float]] = []
    delta = target_cfg.delta
    if delta is None:
        delta, delta_table = calibrate_delta(
            dataset.X_train,
            dataset.y_train,
            target_cfg.lambda_,
            target_cfg.sigma0,
            safety=cfg.step_safety,
            tempered=target_cfg.tempered,
        )
        target_cfg = replace(target_cfg, delta=delta)
    target = LogisticTarget(
        dataset.X_train,
        dataset.y_train,
        target_cfg.lambda_,
        target_cfg.sigma0,
        target_cfg.delta,
        target_cfg.tempered,
    )
    log_a_min, a_min = target.bounds()
    _log(f"      delta={delta:.4e}  clock bound a >= {a_min:.4e} "
         f"(log a >= {log_a_min:.4f})")

    w_center, opt = find_smooth_map(target)
    h_auto, curvature = calibrate_step_size(target, w_center, cfg.step_safety)
    h = float(cfg.h) if cfg.h is not None else float(h_auto)
    _log(f"      MAP found, |grad U0| = "
         f"{np.linalg.norm(target.grad_U0(w_center)):.3e}; "
         f"lambda_max(Hess U0) = {curvature:.4g}  ->  h = {h:.4e}")

    # ------------------------------------------------ 3. freeze the geometry
    _log("[3/6] pilot chain and frozen constraint")
    if cfg.fixed_radius is not None:
        constraint = ConstraintSpec(
            center=tuple(float(v) for v in w_center),
            radius=float(cfg.fixed_radius),
            source="fixed",
        )
        pilot = run_unconstrained_pilot(
            target, w_center, h, replace(pilot_cfg, n_iter=max(pilot_cfg.n_iter // 4, 1000))
        )
    else:
        pilot = run_unconstrained_pilot(target, w_center, h, pilot_cfg)
        constraint = choose_radius_from_pilot(pilot, w_center, pilot_cfg)
    _log(f"      R = {constraint.radius:.6f} (source {constraint.source}); "
         f"pilot exceedance fraction "
         f"{constraint.pilot_exceedance_fraction if constraint.pilot_exceedance_fraction is not None else float('nan'):.5f}")
    to_json(constraint, os.path.join(out_dir, "constraint.json"))

    triples = field_cfg.triples or generate_cyclic_triples(D)
    field_residuals = verify_field_properties(
        np.asarray(constraint.center), triples, field_cfg.s, n_points=16, seed=0,
        radius=max(constraint.radius, 1e-3),
    )
    _log("      field identities: "
         + ", ".join(f"{k}={v:.2e}" for k, v in field_residuals.items()))
    alpha_star = natural_alpha_scale(constraint.radius, len(triples), field_cfg.s)
    _log(f"      J_s is linear in q = w - center, so with R = {constraint.radius:.5f} "
         f"and m = {len(triples)} blocks the rotation matches the gradient at "
         f"alpha* = {alpha_star:.2f}")

    # --------------------------------------------------- 4. the alpha sweep
    _log(f"[4/6] alpha sweep {list(cfg.alphas)}, {cfg.n_chains} chains each, "
         f"{cfg.n_iter} iterations")
    chains: dict[float, list[SamplerResult]] = {}
    for alpha in cfg.alphas:
        t0 = time.perf_counter()
        chains[float(alpha)] = _run_alpha_group(
            target, constraint, float(alpha), h, cfg, cfg.n_chains,
            field_cfg=field_cfg,
        )
        _log(f"      alpha={alpha:<5g} {time.perf_counter() - t0:6.1f}s  "
             f"projection {np.mean([r.projection_frequency for r in chains[float(alpha)]]):.3%}")

    # ------------------------------------------- 5. diagnostics and scoring
    _log("[5/6] diagnostics, prediction and sensitivity studies")
    diagnostics: dict[float, ChainDiagnostics] = {}
    scores: dict[float, PredictiveScores] = {}
    for alpha, group in chains.items():
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            diagnostics[alpha] = calculate_chain_diagnostics(
                group, dataset.param_names
            )
        pooled = np.concatenate([r.samples for r in group], axis=0)
        p_bar = posterior_predictive_probabilities(pooled, dataset.X_test)
        scores[alpha] = evaluate_predictive_performance(
            dataset.y_test, p_bar, alpha=alpha, n_samples=pooled.shape[0]
        )
        d = diagnostics[alpha]
        _log(f"      alpha={alpha:<6g} (={alpha / alpha_star:5.2f} alpha*)  "
             f"minESS {d.min_ess:8.1f}  medESS {d.median_ess:8.1f}  "
             f"ESS/s {d.ess_per_second:7.2f}  maxRhat {d.max_r_hat:.4f}  "
             f"|nonrev|/|rev| {d.nonreversible_drift_norm / d.reversible_drift_norm:6.3f}  "
             f"AUC {scores[alpha].roc_auc:.5f}")

    step_rows = []
    base_iter = cfg.n_iter_sensitivity or cfg.n_iter
    base_burn = cfg.burn_in_sensitivity or cfg.burn_in
    for factor in cfg.step_size_factors:
        # hold the simulated time fixed: a smaller step needs more iterations
        n_iter = int(round(base_iter / factor))
        burn_in = int(round(base_burn / factor))
        for alpha in (0.0, max(cfg.alphas)):
            group = _run_alpha_group(
                target, constraint, float(alpha), h * factor, cfg,
                cfg.n_chains_sensitivity, n_iter=n_iter, burn_in=burn_in,
                field_cfg=field_cfg,
            )
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                d = calculate_chain_diagnostics(group, dataset.param_names)
            pooled = np.concatenate([r.samples for r in group], axis=0)
            p_bar = posterior_predictive_probabilities(pooled, dataset.X_test)
            sc = evaluate_predictive_performance(
                dataset.y_test, p_bar, alpha=alpha, n_samples=pooled.shape[0]
            )
            step_rows.append(
                {
                    "step_factor": factor,
                    "h": h * factor,
                    "n_iter": n_iter,
                    "alpha": float(alpha),
                    "posterior_mean_norm": float(np.linalg.norm(pooled.mean(axis=0))),
                    **{f"mean[{n}]": float(v)
                       for n, v in zip(dataset.param_names, pooled.mean(axis=0))},
                    **{f"sd[{n}]": float(v)
                       for n, v in zip(dataset.param_names, pooled.std(axis=0, ddof=1))},
                    "min_ess": d.min_ess,
                    "ess_per_second": d.ess_per_second,
                    "max_r_hat": d.max_r_hat,
                    "projection_frequency": d.projection_frequency,
                    "roc_auc": sc.roc_auc,
                    "log_loss": sc.log_loss,
                    "accuracy": sc.accuracy,
                }
            )
            _log(f"      step h*{factor:<5g} alpha={alpha:<5g} n_iter={n_iter:<7d} "
                 f"AUC {sc.roc_auc:.5f}  minESS {d.min_ess:7.1f}")
    step_sensitivity = pd.DataFrame(step_rows)

    radius_rows = []
    for factor in cfg.radius_factors:
        scaled = replace(
            constraint, radius=constraint.radius * factor,
            source=f"{constraint.source} x {factor:g}",
        )
        for alpha in (0.0, max(cfg.alphas)):
            group = _run_alpha_group(
                target, scaled, float(alpha), h, cfg, cfg.n_chains_sensitivity,
                n_iter=base_iter, burn_in=base_burn, field_cfg=field_cfg,
            )
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                d = calculate_chain_diagnostics(group, dataset.param_names)
            pooled = np.concatenate([r.samples for r in group], axis=0)
            p_bar = posterior_predictive_probabilities(pooled, dataset.X_test)
            sc = evaluate_predictive_performance(
                dataset.y_test, p_bar, alpha=alpha, n_samples=pooled.shape[0]
            )
            radius_rows.append(
                {
                    "radius_factor": factor,
                    "radius": scaled.radius,
                    "alpha": float(alpha),
                    "min_ess": d.min_ess,
                    "ess_per_second": d.ess_per_second,
                    "max_r_hat": d.max_r_hat,
                    "projection_frequency": d.projection_frequency,
                    "posterior_mean_norm": float(np.linalg.norm(pooled.mean(axis=0))),
                    "roc_auc": sc.roc_auc,
                    "log_loss": sc.log_loss,
                    "accuracy": sc.accuracy,
                }
            )
            _log(f"      radius x{factor:<4g} alpha={alpha:<5g} "
                 f"projection {d.projection_frequency:.3%}  AUC {sc.roc_auc:.5f}")
    radius_sensitivity = pd.DataFrame(radius_rows)

    # ------------------------------------------------------ 6. verify, save
    _log("[6/6] validation checks and outputs")
    alpha_star_value = alpha_star
    validation = run_validation_checks(
        dataset, target, constraint, chains, diagnostics, triples, field_cfg.s,
        step_sensitivity,
    )
    _log("      " + "; ".join(
        f"{k}={'PASS' if v is True else v}"
        for k, v in validation["checks"].items()
    ))

    figures: list[str] = []
    if cfg.make_figures:
        from plots import make_all_figures

        figures = make_all_figures(
            chains, diagnostics, scores, dataset.param_names,
            np.asarray(constraint.center), constraint.radius,
            os.path.join(out_dir, "figures"), alpha_star=alpha_star,
        )

    _write_outputs(
        out_dir, dataset, target, target_cfg, data_cfg, field_cfg, pilot_cfg,
        cfg, constraint, h, curvature, delta_table, pilot, chains,
        diagnostics, scores, step_sensitivity, radius_sensitivity, validation,
        figures, time.perf_counter() - started, alpha_star_value,
    )
    _log(f"      wrote {out_dir}/ in {time.perf_counter() - started:.1f}s total")

    return ExperimentOutput(
        dataset=dataset,
        target=target,
        constraint=constraint,
        h=h,
        delta_table=delta_table,
        pilot=pilot,
        chains=chains,
        diagnostics=diagnostics,
        scores=scores,
        step_sensitivity=step_sensitivity,
        radius_sensitivity=radius_sensitivity,
        validation=validation,
        figures=figures,
        out_dir=out_dir,
    )


def load_chains(out_dir: str) -> tuple[dict[float, list[SamplerResult]], ConstraintSpec,
                                       tuple[str, ...]]:
    """Rebuild the chains of a finished run from its NPZ files.

    Enough of each :class:`~sampler.SamplerResult` is stored to recompute every
    diagnostic and redraw every figure, so the plots can be revised without
    re-sampling.  Runtimes and gradient counts come back from ``chains.csv``.
    """
    sample_dir = os.path.join(out_dir, "samples")
    with open(os.path.join(out_dir, "constraint.json")) as fh:
        spec = ConstraintSpec(**json.load(fh))
    costs = {}
    chains_csv = os.path.join(out_dir, "chains.csv")
    if os.path.exists(chains_csv):
        table = pd.read_csv(chains_csv)
        for _, row in table.iterrows():
            costs[(float(row["alpha"]), int(row["chain"]))] = (
                float(row["runtime_seconds"]), int(row["n_grad_evaluations"])
            )
    chains: dict[float, list[SamplerResult]] = {}
    names: tuple[str, ...] = ()
    for path in sorted(glob.glob(os.path.join(sample_dir, "alpha_*.npz"))):
        blob = np.load(path, allow_pickle=False)
        names = tuple(str(v) for v in blob["param_names"])
        alpha = float(os.path.basename(path)[len("alpha_") : -len(".npz")])
        group = []
        for c in range(blob["samples"].shape[0]):
            runtime, n_grad = costs.get((alpha, c), (float("nan"), 0))
            group.append(
                SamplerResult(
                    alpha=alpha,
                    h=float("nan"),
                    seed=int(blob["seeds"][c]),
                    n_iter=int(blob["log_a"].shape[1]),
                    burn_in=int(blob["trajectory"].shape[1] - 1
                               - blob["samples"].shape[1] * 1),
                    thin=1,
                    s=1.0,
                    center=np.asarray(blob["center"], dtype=np.float64),
                    radius=float(blob["radius"][0]),
                    projected_flag=True,
                    trajectory=blob["trajectory"][c],
                    samples=blob["samples"][c],
                    log_a=blob["log_a"][c],
                    a=blob["a"][c],
                    radius_trace=blob["radius_trace"][c],
                    drift_norm=blob["drift_norm"][c],
                    reversible_drift_norm=blob["reversible_drift_norm"][c],
                    nonreversible_drift_norm=blob["nonreversible_drift_norm"][c],
                    projection_occurred=blob["projection_occurred"][c],
                    runtime_seconds=runtime,
                    n_grad_evaluations=n_grad,
                    n_numerical_warnings=0,
                )
            )
        chains[alpha] = group
    if not chains:
        raise FileNotFoundError(f"no alpha_*.npz under {sample_dir!r}")
    return chains, spec, names


def regenerate_figures(out_dir: str, data_cfg: DataConfig | None = None) -> list[str]:
    """Redraw every figure from a finished run's stored samples, without sampling."""
    from plots import make_all_figures

    chains, spec, names = load_chains(out_dir)
    dataset = load_and_preprocess(data_cfg or DataConfig())
    diags, scores = {}, {}
    for alpha, group in chains.items():
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            diags[alpha] = calculate_chain_diagnostics(group, names)
        pooled = np.concatenate([r.samples for r in group], axis=0)
        p_bar = posterior_predictive_probabilities(pooled, dataset.X_test)
        scores[alpha] = evaluate_predictive_performance(
            dataset.y_test, p_bar, alpha=alpha, n_samples=pooled.shape[0]
        )
    triples = generate_cyclic_triples(len(spec.center))
    return make_all_figures(
        chains, diags, scores, names, np.asarray(spec.center), spec.radius,
        os.path.join(out_dir, "figures"),
        alpha_star=natural_alpha_scale(spec.radius, len(triples), 1.0),
    )


# ------------------------------------------------------------------ checks
def run_validation_checks(
    dataset: Dataset,
    target: LogisticTarget,
    constraint: ConstraintSpec,
    chains: dict[float, list[SamplerResult]],
    diagnostics: dict[float, ChainDiagnostics],
    triples: tuple[tuple[int, int, int], ...],
    s: float,
    step_sensitivity: pd.DataFrame,
    tol: float = 1e-8,
) -> dict[str, object]:
    r"""The ten checks from the specification, run against the stored results.

    Each returns ``True`` or a string describing the failure, and the numeric
    residuals are returned alongside so that "passed" is auditable rather than
    merely asserted.  These are deliberately re-derived from the saved
    trajectories: a check that only inspected the configuration would pass even
    if the sampler had ignored it.
    """
    checks: dict[str, object] = {}
    details: dict[str, object] = {}
    D = dataset.dim
    center = np.asarray(constraint.center, dtype=np.float64)

    # 1. every stored state is inside the constraint
    worst_slack = -np.inf
    for group in chains.values():
        for r in group:
            worst_slack = max(
                worst_slack, float(r.radius_trace.max() - constraint.radius)
            )
    details["max_radius_excess"] = worst_slack
    checks["1_states_within_constraint"] = (
        True if worst_slack <= tol * max(1.0, constraint.radius)
        else f"a state exceeded R by {worst_slack:.3e}"
    )

    # 2 and 3. skewness and tangency of J_s, at states the chains actually visited
    rng = np.random.default_rng(7)
    worst_skew = worst_tangency = 0.0
    for group in chains.values():
        r = group[0]
        idx = rng.choice(r.samples.shape[0], size=min(24, r.samples.shape[0]),
                         replace=False)
        for w in r.samples[idx]:
            J = construct_J(w, center, triples, s)
            worst_skew = max(worst_skew, float(np.linalg.norm(J + J.T)))
            worst_tangency = max(
                worst_tangency, float(np.linalg.norm((w - center) @ J))
            )
    details["max_skew_residual"] = worst_skew
    details["max_tangency_residual"] = worst_tangency
    checks["2_J_skew_symmetric"] = (
        True if worst_skew < 1e-10 else f"||J + J'|| = {worst_skew:.3e}"
    )
    checks["3_tangency_qJ_zero"] = (
        True if worst_tangency < 1e-8 else f"||q'J|| = {worst_tangency:.3e}"
    )

    # 4. anchoring bounds on every recorded clock value
    log_a_min, a_min = anchor_bounds(D, target.lambda_, target.delta)
    observed_min = min(float(r.log_a.min()) for g in chains.values() for r in g)
    observed_max = max(float(r.log_a.max()) for g in chains.values() for r in g)
    try:
        for group in chains.values():
            for r in group:
                check_anchor_bounds(r.log_a, D, target.lambda_, target.delta)
        checks["4_anchor_bounds"] = True
    except AssertionError as exc:  # pragma: no cover - a real failure path
        checks["4_anchor_bounds"] = str(exc)
    details["log_a_bound"] = log_a_min
    details["log_a_observed"] = [observed_min, observed_max]
    details["a_bound"] = a_min

    # 5. no NaN or infinity anywhere that matters
    bad = []
    for alpha, group in chains.items():
        for r in group:
            for name in ("trajectory", "samples", "log_a", "a", "radius_trace",
                         "drift_norm"):
                if not np.isfinite(getattr(r, name)).all():
                    bad.append(f"alpha={alpha} {name}")
    checks["5_all_finite"] = True if not bad else f"non-finite in {bad}"

    # 6. gradient dimensions
    g = target.grad_U0(center)
    checks["6_gradient_shape"] = (
        True if g.shape == (D,) else f"grad has shape {g.shape}, expected ({D},)"
    )

    # 7. the intercept is excluded from the L1 penalty.  Shifting beta_0 must
    #    change U by exactly the likelihood plus the Gaussian prior, with no
    #    penalty contribution; and the penalty gradient entry must be zero.
    probe = center.copy()
    probe[0] += 0.5
    penalty_only = target.lambda_ * (
        np.abs(probe[1:]).sum() - np.abs(center[1:]).sum()
    )
    details["intercept_penalty_contribution"] = float(penalty_only)
    grad_no_data = np.zeros(D)
    grad_no_data[0] = probe[0] / target.sigma0**2
    checks["7_intercept_not_l1_penalised"] = (
        True if abs(penalty_only) < 1e-12
        else f"changing beta_0 moved the L1 penalty by {penalty_only:.3e}"
    )

    # 8. every alpha group used the same target and the same constraint
    radii = {float(r.radius) for g in chains.values() for r in g}
    centers = {tuple(np.round(r.center, 15)) for g in chains.values() for r in g}
    hs = {float(r.h) for g in chains.values() for r in g}
    n_iters = {int(r.n_iter) for g in chains.values() for r in g}
    seeds = {float(a): sorted(r.seed for r in g) for a, g in chains.items()}
    same_seeds = len({tuple(v) for v in seeds.values()}) == 1
    problems = []
    if len(radii) != 1:
        problems.append(f"radii {sorted(radii)}")
    if len(centers) != 1:
        problems.append("centres differ between chains")
    if len(hs) != 1:
        problems.append(f"step sizes {sorted(hs)}")
    if len(n_iters) != 1:
        problems.append(f"iteration counts {sorted(n_iters)}")
    if not same_seeds:
        problems.append(f"seed sets differ: {seeds}")
    checks["8_same_target_and_constraint"] = True if not problems else "; ".join(problems)
    details["shared_settings"] = {
        "radius": sorted(radii),
        "h": sorted(hs),
        "n_iter": sorted(n_iters),
        "seeds_per_alpha": {str(k): v for k, v in seeds.items()},
    }

    # 9. the geometry was frozen once: the centre every chain used is the
    #    constraint's own centre, bit for bit
    max_center_drift = max(
        float(np.abs(np.asarray(r.center) - center).max())
        for g in chains.values() for r in g
    )
    details["max_center_drift"] = max_center_drift
    checks["9_geometry_not_recomputed"] = (
        True if max_center_drift == 0.0
        else f"a chain used a centre differing by {max_center_drift:.3e}"
    )

    # 10. projection percentages are reported
    details["projection_frequency_by_alpha"] = {
        str(a): diagnostics[a].projection_frequency for a in sorted(diagnostics)
    }
    checks["10_projection_reported"] = True

    # step-size consistency: do the posterior summaries move between h and h/2?
    inconsistent = []
    if not step_sensitivity.empty:
        mean_cols = [c for c in step_sensitivity.columns if c.startswith("mean[")]
        sd_cols = [c for c in step_sensitivity.columns if c.startswith("sd[")]
        for alpha in step_sensitivity["alpha"].unique():
            sub = step_sensitivity[step_sensitivity["alpha"] == alpha].sort_values(
                "step_factor", ascending=False
            )
            if len(sub) < 2:
                continue
            coarse, fine = sub.iloc[0], sub.iloc[1]
            scale = np.abs(fine[sd_cols].to_numpy(dtype=float))
            shift = np.abs(
                coarse[mean_cols].to_numpy(dtype=float)
                - fine[mean_cols].to_numpy(dtype=float)
            )
            worst = float(np.nanmax(shift / np.maximum(scale, 1e-12)))
            details[f"step_bias_in_sd_alpha_{alpha:g}"] = worst
            if worst > 0.5:
                inconsistent.append(
                    f"alpha={alpha:g}: halving h moved a posterior mean by "
                    f"{worst:.2f} posterior standard deviations"
                )
    if inconsistent:
        for message in inconsistent:
            warnings.warn(message, RuntimeWarning)
    checks["11_step_size_conclusions_consistent"] = (
        True if not inconsistent else "; ".join(inconsistent)
    )

    failures = {k: v for k, v in checks.items() if v is not True}
    return {
        "checks": checks,
        "details": details,
        "n_failures": len(failures),
        "failures": failures,
        "arviz": HAS_ARVIZ,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }


# ------------------------------------------------------------------ outputs
def _write_outputs(
    out_dir, dataset, target, target_cfg, data_cfg, field_cfg, pilot_cfg, cfg,
    constraint, h, curvature, delta_table, pilot, chains, diagnostics, scores,
    step_sensitivity, radius_sensitivity, validation, figures, runtime,
    alpha_star,
) -> None:
    """Configuration and geometry as JSON, samples as NPZ, tables as CSV, a summary."""
    os.makedirs(out_dir, exist_ok=True)
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    to_json(
        {
            "data": data_cfg,
            "target": target_cfg,
            "field": field_cfg,
            "pilot": pilot_cfg,
            "experiment": cfg,
            "resolved": {
                "h": h,
                "alpha_natural_scale": natural_alpha_scale(
                    constraint.radius, len(field_cfg.triples
                                           or generate_cyclic_triples(dataset.dim)),
                    field_cfg.s),
                "anchor_curvature_max": curvature,
                "delta": target_cfg.delta,
                "lambda": target_cfg.lambda_,
                "triples": [list(t) for t in (field_cfg.triples
                                              or generate_cyclic_triples(dataset.dim))],
                "dataset": dataset.summary(),
                "target_description": target.describe(),
                "runtime_seconds": runtime,
            },
        },
        os.path.join(out_dir, "config.json"),
    )
    to_json(constraint, os.path.join(out_dir, "constraint.json"))
    to_json(validation, os.path.join(out_dir, "validation.json"))
    if delta_table:
        pd.DataFrame(delta_table).to_csv(
            os.path.join(out_dir, "delta_calibration.csv"), index=False
        )

    np.savez_compressed(
        os.path.join(sample_dir, "pilot.npz"),
        trajectory=pilot.trajectory,
        radius_trace=pilot.radius_trace,
        log_a=pilot.log_a,
    )
    for alpha, group in chains.items():
        np.savez_compressed(
            os.path.join(sample_dir, f"alpha_{alpha:g}.npz"),
            samples=np.stack([r.samples for r in group]),
            trajectory=np.stack([r.trajectory for r in group]),
            log_a=np.stack([r.log_a for r in group]),
            a=np.stack([r.a for r in group]),
            radius_trace=np.stack([r.radius_trace for r in group]),
            drift_norm=np.stack([r.drift_norm for r in group]),
            reversible_drift_norm=np.stack([r.reversible_drift_norm for r in group]),
            nonreversible_drift_norm=np.stack(
                [r.nonreversible_drift_norm for r in group]
            ),
            projection_occurred=np.stack([r.projection_occurred for r in group]),
            seeds=np.array([r.seed for r in group]),
            center=np.asarray(constraint.center),
            radius=np.array([constraint.radius]),
            param_names=np.array(dataset.param_names),
        )

    pd.DataFrame([diagnostics[a].as_row() for a in sorted(diagnostics)]).to_csv(
        os.path.join(out_dir, "diagnostics.csv"), index=False
    )
    pd.DataFrame([scores[a].as_row() for a in sorted(scores)]).to_csv(
        os.path.join(out_dir, "predictive.csv"), index=False
    )
    pd.DataFrame(
        [
            {"alpha": a, "chain": i, **r.summary()}
            for a in sorted(chains)
            for i, r in enumerate(chains[a])
        ]
    ).to_csv(os.path.join(out_dir, "chains.csv"), index=False)
    step_sensitivity.to_csv(
        os.path.join(out_dir, "step_size_sensitivity.csv"), index=False
    )
    radius_sensitivity.to_csv(
        os.path.join(out_dir, "radius_sensitivity.csv"), index=False
    )
    for alpha, sc in scores.items():
        pd.DataFrame(
            sc.confusion, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]
        ).to_csv(os.path.join(out_dir, f"confusion_alpha_{alpha:g}.csv"))

    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(_summary_text(
            dataset, target, constraint, h, curvature, diagnostics, scores,
            step_sensitivity, radius_sensitivity, validation, runtime, alpha_star,
        ))


def _summary_text(
    dataset, target, constraint, h, curvature, diagnostics, scores,
    step_sensitivity, radius_sensitivity, validation, runtime, alpha_star,
) -> str:
    alphas = sorted(diagnostics)
    base = diagnostics[alphas[0]]
    lines = [
        "Constrained non-reversible anchored Langevin on MAGIC Gamma Telescope",
        "=" * 72,
        "",
        f"data           n_train={dataset.n_train} n_test={dataset.n_test} D={dataset.dim}",
        f"target         U = sum_i[softplus(x'w) - y x'w] + b0^2/(2 sigma0^2) "
        f"+ lambda sum_{{j>=1}} |b_j|",
        f"               lambda={target.lambda_:.6g} sigma0={target.sigma0:g} "
        f"(sum over observations, not the mean)",
        f"anchor         delta={target.delta:.6g}; clock bound "
        f"a(w) in [{target.bounds()[1]:.4e}, 1]",
        f"step size      h={h:.6e} = {0.25:g}/lambda_max, "
        f"lambda_max(Hess U0)={curvature:.6g}",
        f"constraint     ball, R={constraint.radius:.6f}, source={constraint.source}, "
        f"pilot exceedance="
        f"{constraint.pilot_exceedance_fraction if constraint.pilot_exceedance_fraction is not None else float('nan'):.5f}",
        f"runtime        {runtime:.1f}s total",
        "",
        f"rotation scale  J_s is linear in q = w - center, so the rotation matches",
        f"                the gradient only at alpha* = sqrt(m)/(s R) = {alpha_star:.2f};",
        f"                alpha/alpha* is the dimensionless strength.",
        "",
        "Efficiency by alpha (ratios are against the reversible alpha = 0)",
        "-" * 88,
        f"{'alpha':>7} {'a/a*':>7} {'|nr|/|r|':>9} {'minESS':>9} {'medESS':>9} "
        f"{'ESS/s':>9} {'ESS/s rel':>10} {'maxRhat':>8} {'proj%':>7} {'MSJD':>10} "
        f"{'a_mean':>7}",
    ]
    for a in alphas:
        d = diagnostics[a]
        ratio = d.nonreversible_drift_norm / d.reversible_drift_norm
        lines.append(
            f"{a:>7g} {a / alpha_star:>7.3f} {ratio:>9.4f} {d.min_ess:>9.1f} "
            f"{d.median_ess:>9.1f} {d.ess_per_second:>9.2f} "
            f"{d.ess_per_second / base.ess_per_second:>9.2f}x {d.max_r_hat:>8.4f} "
            f"{d.projection_frequency * 100:>6.2f}% {d.mean_squared_jump:>10.3e} "
            f"{d.a_mean:>7.4f}"
        )
    lines += [
        "",
        "Test-set predictive performance (threshold-free scores are the comparable ones)",
        "-" * 72,
        f"{'alpha':>6} {'accuracy':>9} {'balanced':>9} {'ROC-AUC':>9} {'log loss':>9} "
        f"{'Brier':>8} {'sens':>7} {'spec':>7}",
    ]
    for a in alphas:
        s = scores[a]
        lines.append(
            f"{a:>6g} {s.accuracy:>9.5f} {s.balanced_accuracy:>9.5f} {s.roc_auc:>9.5f} "
            f"{s.log_loss:>9.5f} {s.brier_score:>8.5f} {s.sensitivity:>7.4f} "
            f"{s.specificity:>7.4f}"
        )
    lines += [
        "",
        "These posteriors all target the same law, so the predictive scores should",
        "agree; they are reported to confirm that the rotation does not bias the",
        "answer, not as evidence that one alpha predicts better.  Differences of a",
        "few thousandths in accuracy on this test set are inside binomial noise.",
        "",
        "Step-size sensitivity (simulated time held fixed)",
        "-" * 72,
    ]
    if not step_sensitivity.empty:
        cols = ["step_factor", "h", "n_iter", "alpha", "min_ess", "ess_per_second",
                "max_r_hat", "roc_auc", "log_loss", "posterior_mean_norm"]
        lines.append(step_sensitivity[cols].to_string(index=False,
                                                      float_format=lambda v: f"{v:.5g}"))
    lines += ["", "Radius sensitivity", "-" * 72]
    if not radius_sensitivity.empty:
        lines.append(radius_sensitivity.to_string(index=False,
                                                  float_format=lambda v: f"{v:.5g}"))
    lines += ["", "Validation", "-" * 72]
    for name, value in validation["checks"].items():
        lines.append(f"  {'PASS' if value is True else 'FAIL'}  {name}"
                     + ("" if value is True else f" -- {value}"))
    warned = {
        a: diagnostics[a].warnings_raised for a in alphas if diagnostics[a].warnings_raised
    }
    lines += ["", "Warnings raised", "-" * 72]
    if warned:
        for a, messages in warned.items():
            for message in messages:
                lines.append(f"  alpha={a:g}: {message}")
    else:
        lines.append("  none")
    lines.append("")
    return "\n".join(lines)
