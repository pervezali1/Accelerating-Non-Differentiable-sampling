"""End-to-end driver for the MAGIC Gamma Telescope PNRAL experiment.

Run order
---------
1.  load and preprocess ``magic04.data`` (stratified 80/20, training-only scaler);
2.  smooth MAP of ``U0`` by L-BFGS-B;
3.  **unconstrained** reversible (``alpha = 0``) anchored pilot chain at a
    conservative step size;
4.  freeze ``Lambda_constraint`` from the pilot ``g``-quantile rule -- or accept
    a user-supplied value and skip the pilot;
5.  constrained MAP and the shared initial points;
6.  calibrate one common step size at ``max(alphas)``;
7.  the primary alpha comparison (identical data, threshold, initial points,
    seeds, step size, iterations, burn-in and thinning);
8.  automatic checks, diagnostics, figures;
9.  posterior-predictive evaluation on the held-out test split;
10. constraint-radius and step-size sensitivity studies, and the optional
    overlapping-triple extension.

Usage::

    python -m pnral.experiment --data-path data/magic04.data
    python -m pnral.experiment --synthetic --quick      # code smoke-test only
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import textwrap
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import diagnostics as diag
from . import prediction as pred
from .anchor import anchor_bounds
from .config import ExperimentConfig
from .constraint import (ThresholdSelection, choose_threshold_from_pilot,
                         fixed_threshold, g_constraint, g_minimum,
                         validate_threshold)
from .data import ProcessedData, load_dataset, save_preprocessing
from .sampler import (AlphaRun, ChainResult, JSpec, calibrate_step_size,
                      constrained_map, initial_points, run_chain, run_chains,
                      smooth_map)
from .target import LogisticTarget

__all__ = ["run_experiment", "main"]

_SYNTHETIC_BANNER = """
################################################################################
#  SYNTHETIC SURROGATE DATA -- NOT THE MAGIC GAMMA TELESCOPE DATA SET          #
#  This run exercises the code path only.  Every number below is meaningless   #
#  as science.  Re-run with --data-path pointing at the real magic04.data.     #
################################################################################
"""


# ----------------------------------------------------------------------
# persistence helpers
# ----------------------------------------------------------------------
def _chain_arrays(chain: ChainResult, prefix: str) -> Dict[str, np.ndarray]:
    """Flatten one chain's traces into a ``savez``-friendly dictionary."""
    return {
        f"{prefix}_states": chain.states,
        f"{prefix}_U": chain.U,
        f"{prefix}_U0": chain.U0,
        f"{prefix}_log_a": chain.log_a,
        f"{prefix}_a": chain.a,
        f"{prefix}_g": chain.g,
        f"{prefix}_slack": chain.slack,
        f"{prefix}_grad_norm": chain.grad_norm,
        f"{prefix}_reversible_drift_norm": chain.reversible_drift_norm,
        f"{prefix}_nonreversible_drift_norm": chain.nonreversible_drift_norm,
        f"{prefix}_total_drift_norm": chain.total_drift_norm,
        f"{prefix}_J_operator_norm": chain.J_operator_norm,
        f"{prefix}_projected": chain.projected,
        f"{prefix}_projection_distance": chain.projection_distance,
        f"{prefix}_elapsed": chain.elapsed,
    }


def save_run_npz(run: AlphaRun, path: str | Path) -> str:
    """Write retained draws and full per-iteration traces to a compressed NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {
        "retained_samples": run.samples,                 # (chain, draw, d)
        "alpha": np.array(run.alpha),
        "step_size": np.array(run.step_size),
        "Lambda_constraint": np.array(run.Lambda_constraint),
        "seeds": np.array([chain.seed for chain in run.chains]),
        "burn_in": np.array([chain.burn_in for chain in run.chains]),
        "thinning": np.array([chain.thinning for chain in run.chains]),
        "runtime": np.array([chain.runtime for chain in run.chains]),
    }
    for index, chain in enumerate(run.chains):
        payload.update(_chain_arrays(chain, f"chain{index}"))
    np.savez_compressed(path, **payload)
    return str(path)


def trace_dataframe(run: AlphaRun, thin: int) -> pd.DataFrame:
    """Thinned per-iteration diagnostic traces of every chain, long format."""
    frames = []
    for index, chain in enumerate(run.chains):
        step = max(1, int(thin))
        iterations = np.arange(chain.states.shape[0])[::step]
        frames.append(pd.DataFrame({
            "label": run.label,
            "alpha": run.alpha,
            "chain": index,
            "iteration": iterations,
            "U": chain.U[::step],
            "U0": chain.U0[::step],
            "log_a": chain.log_a[::step],
            "a": chain.a[::step],
            "g": chain.g[::step],
            "slack": chain.slack[::step],
            "grad_U0_norm": chain.grad_norm[::step],
            "reversible_drift_norm": chain.reversible_drift_norm[::step],
            "nonreversible_drift_norm": chain.nonreversible_drift_norm[::step],
            "total_drift_norm": chain.total_drift_norm[::step],
            "J_operator_norm": chain.J_operator_norm[::step],
            "projected": chain.projected[::step].astype(int),
            "projection_distance": chain.projection_distance[::step],
            "elapsed_seconds": chain.elapsed[::step],
        }))
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
def run_experiment(config: ExperimentConfig, verbose: bool = True) -> Dict[str, object]:
    """Execute the whole study and write every artefact under ``output_dir``."""
    def log(message: str = "") -> None:
        if verbose:
            print(message, flush=True)

    started = time.time()
    output_dir = Path(config.output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    samples_dir = output_dir / "samples"
    for directory in (output_dir, figures_dir, tables_dir, samples_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # -- 1. data --------------------------------------------------------
    log("=" * 78)
    log("Projected Non-Reversible Anchored Langevin -- MAGIC Gamma Telescope")
    log("=" * 78)
    data: ProcessedData = load_dataset(
        config.data.data_path, test_size=config.data.test_size,
        split_seed=config.data.split_seed, use_synthetic=config.data.use_synthetic,
        synthetic_n_rows=config.data.synthetic_n_rows,
        synthetic_seed=config.data.synthetic_seed)
    if data.is_synthetic:
        log(_SYNTHETIC_BANNER)
    preprocessing = save_preprocessing(data, output_dir)
    d = data.d                                    # read from the design matrix
    log(f"train {data.n_train} x {d} | test {data.n_test} | "
        f"d = {d} (intercept + {len(data.feature_names)} predictors)")
    log(f"class balance: train gamma fraction {data.y_train.mean():.4f}")

    target = LogisticTarget(data.X_train, data.y_train,
                            lambda_lasso=config.target.lambda_lasso,
                            sigma_intercept=config.target.sigma_intercept,
                            delta_anchor=config.target.delta_anchor)
    L_guide = target.likelihood_smoothness_guide()
    log_a_min, log_a_max, a_min, a_max = anchor_bounds(target)
    log(f"smoothness guide  L <= 0.25*||X||_2^2 = {L_guide:.6g}")
    log(f"anchor bounds     log a in [{log_a_min:g}, {log_a_max:g}], "
        f"a in [{a_min:.6g}, {a_max:g}]")

    p_constraint = config.constraint.p_constraint
    epsilon_constraint = config.constraint.epsilon_constraint
    minimum_g = g_minimum(d, p_constraint, epsilon_constraint)
    log(f"constraint        p = {p_constraint}, epsilon = {epsilon_constraint}, "
        f"g(0) = d*eps^p = {minimum_g:.6g}")

    # -- 2. smooth MAP ---------------------------------------------------
    w_map_smooth = smooth_map(target)
    log(f"smooth MAP (L-BFGS-B): g(w_MAP) = "
        f"{g_constraint(w_map_smooth, p_constraint, epsilon_constraint):.6g}")

    # -- 3./4. pilot chain and the frozen threshold -----------------------
    j_spec = JSpec.create(config.sampler.triples_mode, d, p_constraint,
                          epsilon_constraint, config.sampler.swirl_scale)
    pilot_summary: Dict[str, object] = {}
    if config.constraint.Lambda_constraint is None:
        pilot_step = config.constraint.pilot_step_size_factor / L_guide
        log(f"\npilot chain: unconstrained reversible anchored (alpha = 0), "
            f"h_pilot = {pilot_step:.4g}, {config.constraint.pilot_iterations} iterations")
        pilot = run_chain(
            target=target, w0=w_map_smooth, alpha=0.0, step_size=pilot_step,
            n_iterations=config.constraint.pilot_iterations,
            seed=config.constraint.pilot_seed, j_spec=j_spec,
            Lambda_constraint=None, p_constraint=p_constraint,
            epsilon_constraint=epsilon_constraint,
            burn_in=config.constraint.pilot_burn_in,
            thinning=config.constraint.pilot_thinning, constrained=False)
        pilot_g = g_constraint(pilot.retained, p_constraint, epsilon_constraint)
        selection = choose_threshold_from_pilot(
            pilot_g, d, p_constraint, epsilon_constraint,
            quantile=config.constraint.pilot_quantile,
            inflation=config.constraint.pilot_inflation,
            lambda_small_factor=config.constraint.lambda_small_factor,
            lambda_large_factor=config.constraint.lambda_large_factor)
        pilot_summary = {
            "pilot_step_size": float(pilot_step),
            "pilot_iterations": int(config.constraint.pilot_iterations),
            "pilot_retained": int(pilot.n_retained),
            "pilot_g_min": float(pilot_g.min()),
            "pilot_g_median": float(np.median(pilot_g)),
            "pilot_g_max": float(pilot_g.max()),
            "pilot_runtime_seconds": float(pilot.runtime),
        }
        log(f"  pilot g(w): min {pilot_g.min():.6g}, median {np.median(pilot_g):.6g}, "
            f"q{config.constraint.pilot_quantile} "
            f"{selection.pilot_quantile_value:.6g}, max {pilot_g.max():.6g}")
    else:
        selection = fixed_threshold(
            config.constraint.Lambda_constraint, d, p_constraint,
            epsilon_constraint, config.constraint.lambda_small_factor,
            config.constraint.lambda_large_factor)
        log("\nLambda_constraint supplied by the user; the pilot procedure is skipped.")

    Lambda_constraint = selection.Lambda_constraint
    validate_threshold(Lambda_constraint, d, p_constraint, epsilon_constraint)
    log(f"Lambda_constraint = {Lambda_constraint:.6g}  (frozen; identical for every alpha)")
    log(f"  g_min = {selection.g_min:.6g}; pilot exceedance fraction = "
        f"{selection.exceedance_fraction!r}")
    log(f"  sensitivity radii: small {selection.Lambda_small:.6g}, "
        f"large {selection.Lambda_large:.6g}")
    log("  NOTE: the constrained posterior is the TRUNCATION of the unconstrained "
        "posterior to K = {w : g(w) <= Lambda_constraint}.")

    constraint_payload = {
        **selection.to_dict(),
        "p_constraint": p_constraint,
        "epsilon_constraint": epsilon_constraint,
        "d": d,
        "triples_mode": config.sampler.triples_mode,
        "triples": [list(t) for t in j_spec.triples],
        "block_scale": float(j_spec.block_scale),
        "swirl_scale": float(config.sampler.swirl_scale),
        "interpretation": ("the constrained posterior is the unconstrained "
                           "posterior exp(-U) truncated to "
                           "K = {w : g(w) <= Lambda_constraint}"),
        **pilot_summary,
    }
    with (output_dir / "constraint.json").open("w", encoding="utf-8") as handle:
        json.dump(constraint_payload, handle, indent=2)

    # -- 5. initialisation ----------------------------------------------
    map_info = constrained_map(target, p_constraint, epsilon_constraint,
                               Lambda_constraint,
                               config.constraint.projection_use_brentq)
    w_map = map_info["w_map_constrained"]
    log(f"\nconstrained MAP via {map_info['route']}: U0 = {map_info['U0_constrained']:.8g}, "
        f"g = {map_info['g_constrained']:.6g}")
    w0_list = initial_points(w_map, config.sampler.number_of_chains,
                             config.sampler.init_perturbation_scale,
                             config.sampler.init_seed, p_constraint,
                             epsilon_constraint, Lambda_constraint,
                             config.constraint.projection_use_brentq)

    # -- 6. step size ----------------------------------------------------
    alphas = list(config.sampler.alphas)
    alpha_max = max(alphas)
    if config.sampler.step_size is None:
        calibration = calibrate_step_size(
            target, w0_list[0], alpha_max, j_spec, Lambda_constraint,
            p_constraint, epsilon_constraint,
            safety=config.sampler.step_size_safety,
            max_halvings=config.sampler.step_size_max_halvings,
            trial_iterations=config.sampler.calibration_iterations,
            max_projection_frequency=config.sampler.calibration_max_projection_frequency,
            projection_use_brentq=config.constraint.projection_use_brentq)
        step_size = calibration["step_size"]
        log(f"\nstep size calibrated at alpha = {alpha_max:g}: h = {step_size:.6g} "
            f"(from h0 = {config.sampler.step_size_safety}/L, "
            f"{len(calibration['history'])} trial(s), accepted={calibration['accepted']})")
    else:
        step_size = float(config.sampler.step_size)
        calibration = {"step_size": step_size, "L_guide": float(L_guide),
                       "history": [], "accepted": True, "user_supplied": True}
        log(f"\nstep size supplied by the user: h = {step_size:.6g}")

    # -- 7. primary alpha comparison -------------------------------------
    log(f"\nprimary comparison: alphas = {alphas}, "
        f"{config.sampler.number_of_chains} chains x "
        f"{config.sampler.n_iterations} iterations "
        f"(burn-in {config.sampler.burn_in}, thinning {config.sampler.thinning})")
    runs: List[AlphaRun] = []
    for alpha in alphas:
        run = run_chains(
            target=target, initial_states=w0_list, alpha=alpha,
            step_size=step_size, n_iterations=config.sampler.n_iterations,
            j_spec=j_spec, Lambda_constraint=Lambda_constraint,
            p_constraint=p_constraint, epsilon_constraint=epsilon_constraint,
            burn_in=config.sampler.burn_in, thinning=config.sampler.thinning,
            chain_seed_base=config.sampler.chain_seed_base,
            projection_use_brentq=config.constraint.projection_use_brentq,
            label=f"alpha={alpha:g}")
        runs.append(run)
        log(f"  alpha = {alpha:<5g} runtime {run.runtime:7.2f}s  "
            f"projection {np.mean([c.projection_frequency for c in run.chains]):8.4%}  "
            f"MSJD {np.mean([c.mean_squared_jumping_distance for c in run.chains]):.4e}")

    # -- 8. checks and diagnostics ---------------------------------------
    log("\nautomatic checks (section 14)")
    checks = diag.run_automatic_checks(runs, target, p_constraint,
                                       epsilon_constraint, Lambda_constraint,
                                       j_spec, data,
                                       tolerance=config.constraint.projection_tolerance)
    for _, row in checks.iterrows():
        log(f"  [{'PASS' if row['passed'] else 'FAIL'}] {row['check']}")
        log(f"         {row['detail']}")
    checks.to_csv(tables_dir / "automatic_checks.csv", index=False)

    summaries = [diag.summarize_alpha_run(run, data.parameter_names) for run in runs]
    diagnostics_frame = diag.diagnostics_table(summaries)
    coefficients_frame = diag.coefficient_table(summaries)
    diagnostics_frame.to_csv(tables_dir / "diagnostics_summary.csv", index=False)
    coefficients_frame.to_csv(tables_dir / "coefficient_summary.csv", index=False)
    pd.concat([trace_dataframe(run, config.sampler.diagnostic_trace_thin)
               for run in runs], ignore_index=True).to_csv(
        tables_dir / "diagnostics_traces.csv", index=False)

    log("\nper-alpha diagnostics")
    log(diagnostics_frame[["label", "min_ess", "median_ess", "min_ess_per_second",
                           "max_split_rhat", "mean_squared_jumping_distance",
                           "projection_frequency", "runtime_seconds"]]
        .to_string(index=False))

    warnings_list = diag.collect_warnings(summaries, Lambda_constraint,
                                          config.constraint.projection_tolerance)

    for run in runs:
        save_run_npz(run, samples_dir / f"primary_alpha_{run.alpha:g}.npz")

    figure_paths = diag.make_all_diagnostic_figures(
        runs, summaries, data.parameter_names, Lambda_constraint,
        selection.g_min, figures_dir)

    # -- 9. posterior prediction ------------------------------------------
    log("\nposterior-predictive evaluation on the held-out test split")
    reports = []
    for run in runs:
        probabilities = pred.posterior_mean_probabilities(
            run.samples, data.X_test, config.prediction_chunk_size)
        reports.append(pred.predictive_metrics(data.y_test, probabilities,
                                               label=f"alpha = {run.alpha:g}"))
    predictive_frame = pred.predictive_table(reports)
    predictive_frame.to_csv(tables_dir / "predictive_results.csv", index=False)
    log(predictive_frame[["label", "accuracy", "balanced_accuracy", "roc_auc",
                          "log_loss", "brier_score",
                          "sensitivity_recall_positive",
                          "specificity_recall_negative"]].to_string(index=False))
    figure_paths["roc"] = pred.plot_roc_curves(data.y_test, reports, figures_dir)
    figure_paths["calibration"] = pred.plot_calibration_curves(data.y_test, reports,
                                                               figures_dir)
    figure_paths["metrics"] = pred.plot_metric_comparison(reports, figures_dir)

    # -- 10. sensitivity studies ------------------------------------------
    sensitivity_alphas = [a for a in config.sensitivity_alphas if a in alphas] \
        or [alphas[0]]
    constraint_sensitivity: List[Dict[str, object]] = []
    step_sensitivity: List[Dict[str, object]] = []
    extension_summaries: List[Dict[str, object]] = []

    if config.run_constraint_sensitivity:
        log("\nconstraint-radius sensitivity (different thresholds define "
            "DIFFERENT truncated posteriors)")
        radii = [("Lambda_small", selection.Lambda_small),
                 ("Lambda", Lambda_constraint),
                 ("Lambda_large", selection.Lambda_large)]
        for alpha in sensitivity_alphas:
            local_summaries = []
            for name, radius in radii:
                map_local = constrained_map(target, p_constraint,
                                            epsilon_constraint, radius,
                                            config.constraint.projection_use_brentq)
                starts = initial_points(map_local["w_map_constrained"],
                                        config.sampler.number_of_chains,
                                        config.sampler.init_perturbation_scale,
                                        config.sampler.init_seed, p_constraint,
                                        epsilon_constraint, radius,
                                        config.constraint.projection_use_brentq)
                run = run_chains(
                    target=target, initial_states=starts, alpha=alpha,
                    step_size=step_size, n_iterations=config.sensitivity_iterations,
                    j_spec=j_spec, Lambda_constraint=radius,
                    p_constraint=p_constraint, epsilon_constraint=epsilon_constraint,
                    burn_in=config.sensitivity_burn_in,
                    thinning=config.sampler.thinning,
                    chain_seed_base=config.sampler.chain_seed_base,
                    projection_use_brentq=config.constraint.projection_use_brentq,
                    label=f"{name}={radius:.4g} (alpha={alpha:g})")
                summary = diag.summarize_alpha_run(run, data.parameter_names)
                local_summaries.append(summary)
                save_run_npz(run, samples_dir /
                             f"radius_{name}_alpha_{alpha:g}.npz")
                log(f"  {name:<13} Lambda = {radius:.6g}  alpha = {alpha:g}  "
                    f"projection {summary['projection_frequency']:.4%}  "
                    f"min ESS {summary['min_ess']:.1f}")
            constraint_sensitivity.extend(local_summaries)
            reference = local_summaries[1]
            for other in (local_summaries[0], local_summaries[2]):
                changed, worst, message = diag.compare_summaries(reference, other)
                if changed:
                    warnings_list.append("[radius sensitivity] " + message)
            figure_paths[f"radius_alpha_{alpha:g}"] = diag.plot_sensitivity_comparison(
                local_summaries, data.parameter_names, figures_dir,
                f"fig15_radius_sensitivity_alpha_{alpha:g}",
                f"Constraint-radius sensitivity (alpha = {alpha:g}) — "
                f"each radius is a different truncated posterior",
                "constraint radius")

    if config.run_step_size_sensitivity:
        log("\nstep-size sensitivity (h, h/2, h/4 -- same target, different "
            "discretisation bias)")
        for alpha in sensitivity_alphas:
            local_summaries = []
            for divisor in (1, 2, 4):
                trial_step = step_size / divisor
                run = run_chains(
                    target=target, initial_states=w0_list, alpha=alpha,
                    step_size=trial_step, n_iterations=config.sensitivity_iterations,
                    j_spec=j_spec, Lambda_constraint=Lambda_constraint,
                    p_constraint=p_constraint, epsilon_constraint=epsilon_constraint,
                    burn_in=config.sensitivity_burn_in,
                    thinning=config.sampler.thinning,
                    chain_seed_base=config.sampler.chain_seed_base,
                    projection_use_brentq=config.constraint.projection_use_brentq,
                    label=f"h/{divisor} = {trial_step:.3g} (alpha={alpha:g})")
                summary = diag.summarize_alpha_run(run, data.parameter_names)
                local_summaries.append(summary)
                save_run_npz(run, samples_dir /
                             f"stepsize_h_over_{divisor}_alpha_{alpha:g}.npz")
                log(f"  h/{divisor:<2} = {trial_step:.6g}  alpha = {alpha:g}  "
                    f"min ESS {summary['min_ess']:.1f}  "
                    f"MSJD {summary['mean_squared_jumping_distance']:.4e}")
            step_sensitivity.extend(local_summaries)
            changed, worst, message = diag.compare_summaries(local_summaries[0],
                                                             local_summaries[1])
            if changed:
                warnings_list.append("[step-size sensitivity] " + message)
            figure_paths[f"stepsize_alpha_{alpha:g}"] = diag.plot_sensitivity_comparison(
                local_summaries, data.parameter_names, figures_dir,
                f"fig16_step_size_sensitivity_alpha_{alpha:g}",
                f"Step-size sensitivity (alpha = {alpha:g}) — same target, "
                f"different discretisation bias",
                "step size")

    if config.run_overlapping_extension:
        log("\noverlapping-triple EXTENSION (labelled sensitivity experiment; the "
            "disjoint construction remains the primary result)")
        extension_spec = JSpec.create("overlapping", d, p_constraint,
                                      epsilon_constraint, config.sampler.swirl_scale)
        for alpha in sensitivity_alphas:
            run = run_chains(
                target=target, initial_states=w0_list, alpha=alpha,
                step_size=step_size, n_iterations=config.sensitivity_iterations,
                j_spec=extension_spec, Lambda_constraint=Lambda_constraint,
                p_constraint=p_constraint, epsilon_constraint=epsilon_constraint,
                burn_in=config.sensitivity_burn_in, thinning=config.sampler.thinning,
                chain_seed_base=config.sampler.chain_seed_base,
                projection_use_brentq=config.constraint.projection_use_brentq,
                label=f"overlapping (alpha={alpha:g})")
            summary = diag.summarize_alpha_run(run, data.parameter_names)
            extension_summaries.append(summary)
            save_run_npz(run, samples_dir / f"overlapping_alpha_{alpha:g}.npz")
            log(f"  overlapping alpha = {alpha:g}  min ESS {summary['min_ess']:.1f}  "
                f"max R-hat {summary['max_rhat']:.4f}  "
                f"mean ||J|| {summary['mean_J_operator_norm']:.4f}")

    extra_summaries = constraint_sensitivity + step_sensitivity + extension_summaries
    if extra_summaries:
        diag.diagnostics_table(extra_summaries).to_csv(
            tables_dir / "sensitivity_diagnostics.csv", index=False)
        diag.coefficient_table(extra_summaries).to_csv(
            tables_dir / "sensitivity_coefficients.csv", index=False)

    # -- 11. selection, configuration and summary --------------------------
    # alpha is selected from TRAINING diagnostics only; the test split is never
    # consulted for any algorithmic choice.
    best_index = int(np.argmax([s["min_ess_per_second"] for s in summaries]))
    recommended_alpha = summaries[best_index]["alpha"]

    config.to_json(output_dir / "config.json")
    runtime_payload = {
        "total_runtime_seconds": time.time() - started,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "step_size": step_size,
        "step_size_calibration": calibration,
        "L_likelihood_guide": float(L_guide),
        "recommended_alpha_by_training_diagnostics": float(recommended_alpha),
        "selection_rule": "argmax over alpha of minimum ESS per second (training "
                          "diagnostics only; the test split is not used to choose "
                          "any algorithmic parameter)",
        "is_synthetic": bool(data.is_synthetic),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(runtime_payload, handle, indent=2, default=float)

    summary_text = _write_summary(
        output_dir, config, data, preprocessing, target, selection,
        constraint_payload, map_info, calibration, step_size, checks,
        diagnostics_frame, coefficients_frame, predictive_frame, warnings_list,
        figure_paths, runtime_payload, recommended_alpha, summaries)

    log("\nwarnings")
    if warnings_list:
        for message in warnings_list:
            log("  ! " + message)
    else:
        log("  none")
    log(f"\nartefacts written to {output_dir.resolve()}")
    log(f"total runtime {runtime_payload['total_runtime_seconds']:.1f} s")

    return {
        "config": config,
        "data": data,
        "target": target,
        "j_spec": j_spec,
        "threshold": selection,
        "step_size": step_size,
        "runs": runs,
        "summaries": summaries,
        "checks": checks,
        "diagnostics": diagnostics_frame,
        "coefficients": coefficients_frame,
        "predictive": predictive_frame,
        "reports": reports,
        "constraint_sensitivity": constraint_sensitivity,
        "step_sensitivity": step_sensitivity,
        "extension": extension_summaries,
        "warnings": warnings_list,
        "figures": figure_paths,
        "summary_markdown": summary_text,
        "output_dir": str(output_dir),
    }


def _write_summary(output_dir: Path, config, data, preprocessing, target,
                   selection, constraint_payload, map_info, calibration,
                   step_size, checks, diagnostics_frame, coefficients_frame,
                   predictive_frame, warnings_list, figure_paths,
                   runtime_payload, recommended_alpha, summaries) -> str:
    """Compose ``summary.md``, the human-readable record of the run."""
    lines: List[str] = []
    add = lines.append
    add("# Projected Non-Reversible Anchored Langevin on the MAGIC Gamma Telescope data")
    add("")
    if data.is_synthetic:
        add("> **WARNING — SYNTHETIC SURROGATE DATA.** This run did not use the real")
        add("> `magic04.data`; it exercises the code path only and carries no")
        add("> scientific meaning.")
        add("")
    add("## Data")
    add("")
    add(f"- source: `{data.source_path}`")
    add(f"- train / test: {data.n_train} / {data.n_test} (stratified 80/20, "
        f"seed {config.data.split_seed})")
    add(f"- parameter dimension read from the design matrix: **d = {data.d}**")
    add(f"- labels: `g -> 1`, `h -> 0`; training gamma fraction "
        f"{data.y_train.mean():.4f}")
    add("- `StandardScaler` fitted on the training predictors only; intercept "
        "column added after standardisation")
    add("")
    add("## Target and anchor")
    add("")
    add(f"- `lambda_lasso = {target.lambda_lasso}` (slopes only; the intercept is "
        f"not L1-penalised)")
    add(f"- `sigma_intercept = {target.sigma_intercept}`, "
        f"`delta_anchor = {target.delta_anchor}`")
    bounds = anchor_bounds(target)
    add(f"- anchor bounds: `log a in [{bounds[0]:g}, {bounds[1]:g}]`, "
        f"`a in [{bounds[2]:.6g}, {bounds[3]:g}]`")
    add(f"- likelihood is a **sum** over {target.n} training rows; smoothness "
        f"guide `L <= 0.25 ||X||_2^2 = {runtime_payload['L_likelihood_guide']:.6g}`")
    add("")
    add("## Constraint")
    add("")
    add(f"- `g(w) = sum_i (w_i^2 + epsilon^2)^(p/2)` with "
        f"`p_constraint = {config.constraint.p_constraint}`, "
        f"`epsilon_constraint = {config.constraint.epsilon_constraint}`")
    add(f"- `g(0) = d * epsilon^p = {selection.g_min:.6g}`")
    add(f"- **`Lambda_constraint = {selection.Lambda_constraint:.6g}`** "
        f"({'pilot rule' if selection.from_pilot else 'user supplied'}), frozen "
        f"and identical for every alpha")
    if selection.from_pilot:
        add(f"- pilot 0.999 quantile of `g` = {selection.pilot_quantile_value:.6g}; "
            f"inflation {selection.inflation}; unconstrained pilot samples with "
            f"`g > Lambda_constraint`: **{selection.exceedance_fraction:.4%}**")
    add(f"- sensitivity radii: small {selection.Lambda_small:.6g}, "
        f"large {selection.Lambda_large:.6g}")
    add(f"- non-reversible triples ({config.sampler.triples_mode}): "
        f"{constraint_payload['triples']}, constant block scale "
        f"{constraint_payload['block_scale']:.6g}, "
        f"`swirl_scale = {config.sampler.swirl_scale}`")
    add("- **The constrained posterior is the unconstrained posterior "
        "`exp(-U)` truncated to `K = {w : g(w) <= Lambda_constraint}`.** "
        "Runs with different radii target different distributions.")
    add("")
    add("## Sampler")
    add("")
    add(f"- constrained MAP found via `{map_info['route']}`; chains start at "
        f"`Pi_K(MAP + N(0, {config.sampler.init_perturbation_scale}^2 I))`")
    add(f"- common step size **h = {step_size:.6g}**, calibrated at "
        f"alpha = {max(config.sampler.alphas):g}")
    add(f"- {config.sampler.number_of_chains} chains x "
        f"{config.sampler.n_iterations} iterations, burn-in "
        f"{config.sampler.burn_in}, thinning {config.sampler.thinning}")
    add(f"- alphas compared: {list(config.sampler.alphas)} "
        f"(alpha = 0 is the constrained reversible anchored baseline)")
    add("- identical data, threshold, initial points, seeds, step size, "
        "iterations, burn-in and thinning across alphas")
    add("- unadjusted (Euler--Maruyama) discretisation followed by the metric "
        "projection: an O(h) discretisation bias is expected and is what the "
        "step-size study probes")
    add("")
    add("## Automatic checks")
    add("")
    add("| check | result | detail |")
    add("|---|---|---|")
    for _, row in checks.iterrows():
        # escape pipes so residuals such as "max |J + J^T|" do not split cells
        detail = str(row["detail"]).replace("|", r"\|")
        add(f"| {row['check']} | {'PASS' if row['passed'] else '**FAIL**'} | "
            f"{detail} |")
    add("")
    add("## Per-alpha diagnostics")
    add("")
    add(diagnostics_frame[["label", "min_ess", "median_ess", "min_ess_per_second",
                           "max_split_rhat", "max_tau_int",
                           "mean_squared_jumping_distance", "projection_frequency",
                           "mean_projection_distance", "runtime_seconds",
                           "gradient_evaluations"]].to_markdown(index=False))
    add("")
    add(f"Recommended alpha from **training diagnostics only** "
        f"(largest minimum ESS per second): **alpha = {recommended_alpha:g}**. "
        f"The test split below is used for final evaluation only and played no "
        f"part in this choice.")
    add("")
    add("## Posterior predictive performance (held-out test split)")
    add("")
    columns = ["label", "accuracy", "balanced_accuracy", "roc_auc", "log_loss",
               "brier_score", "sensitivity_recall_positive",
               "specificity_recall_negative", "precision", "f1"]
    add(predictive_frame[columns].to_markdown(index=False))
    add("")
    add("Confusion matrices (rows: actual hadron/gamma; columns: predicted):")
    add("")
    add(predictive_frame[["label", "true_negative", "false_positive",
                          "false_negative", "true_positive"]].to_markdown(index=False))
    add("")
    add("Because the classes are imbalanced, read balanced accuracy, ROC-AUC, "
        "sensitivity, specificity and the calibration curve before plain accuracy.")
    add("")
    add("## Warnings")
    add("")
    if warnings_list:
        for message in warnings_list:
            add(f"- {message}")
    else:
        add("- none")
    add("")
    add("## Figures")
    add("")
    for key, paths in figure_paths.items():
        add(f"- `{key}`: `{paths['pdf']}` / `{paths['png']}`")
    add("")
    add("## Reproducibility")
    add("")
    add(f"- python {runtime_payload['python']}, numpy {runtime_payload['numpy']}, "
        f"pandas {runtime_payload['pandas']}")
    add(f"- total runtime {runtime_payload['total_runtime_seconds']:.1f} s")
    add(f"- seeds: split {config.data.split_seed}, pilot "
        f"{config.constraint.pilot_seed}, init {config.sampler.init_seed}, "
        f"chain base {config.sampler.chain_seed_base}")
    add("- full configuration in `config.json`; constraint details in "
        "`constraint.json`; scaler in `standard_scaler.joblib`")
    text = "\n".join(lines) + "\n"
    (output_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


# ----------------------------------------------------------------------
def build_config_from_args(argv: Optional[Sequence[str]] = None) -> ExperimentConfig:
    """Parse the command line into an :class:`ExperimentConfig`."""
    parser = argparse.ArgumentParser(
        prog="python -m pnral.experiment",
        description=textwrap.dedent(__doc__ or ""),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", default=None,
                        help="path to magic04.data")
    parser.add_argument("--synthetic", action="store_true",
                        help="use the clearly-labelled synthetic surrogate "
                             "(code smoke-test only; never real science)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--alphas", default=None,
                        help="comma separated, e.g. 0,0.1,0.25,0.5,1.0")
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--thinning", type=int, default=None)
    parser.add_argument("--step-size", type=float, default=None,
                        help="fix h instead of calibrating it")
    parser.add_argument("--lambda-lasso", type=float, default=None)
    parser.add_argument("--sigma-intercept", type=float, default=None)
    parser.add_argument("--delta-anchor", type=float, default=None)
    parser.add_argument("--p-constraint", type=float, default=None)
    parser.add_argument("--epsilon-constraint", type=float, default=None)
    parser.add_argument("--lambda-constraint", type=float, default=None,
                        help="fix Lambda_constraint and skip the pilot procedure")
    parser.add_argument("--swirl-scale", type=float, default=None)
    parser.add_argument("--triples-mode", choices=("disjoint", "overlapping"),
                        default=None)
    parser.add_argument("--no-sensitivity", action="store_true",
                        help="skip the radius/step-size/overlapping studies")
    parser.add_argument("--quick", action="store_true",
                        help="short chains for a fast end-to-end smoke test")
    args = parser.parse_args(argv)

    config = ExperimentConfig()
    if args.data_path is not None:
        config.data.data_path = args.data_path
    config.data.use_synthetic = bool(args.synthetic)
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.alphas is not None:
        config.sampler.alphas = tuple(float(a) for a in args.alphas.split(","))
    if args.chains is not None:
        config.sampler.number_of_chains = args.chains
    if args.iterations is not None:
        config.sampler.n_iterations = args.iterations
    if args.burn_in is not None:
        config.sampler.burn_in = args.burn_in
    if args.thinning is not None:
        config.sampler.thinning = args.thinning
    if args.step_size is not None:
        config.sampler.step_size = args.step_size
    if args.lambda_lasso is not None:
        config.target.lambda_lasso = args.lambda_lasso
    if args.sigma_intercept is not None:
        config.target.sigma_intercept = args.sigma_intercept
    if args.delta_anchor is not None:
        config.target.delta_anchor = args.delta_anchor
    if args.p_constraint is not None:
        config.constraint.p_constraint = args.p_constraint
    if args.epsilon_constraint is not None:
        config.constraint.epsilon_constraint = args.epsilon_constraint
    if args.lambda_constraint is not None:
        config.constraint.Lambda_constraint = args.lambda_constraint
    if args.swirl_scale is not None:
        config.sampler.swirl_scale = args.swirl_scale
    if args.triples_mode is not None:
        config.sampler.triples_mode = args.triples_mode
    if args.no_sensitivity:
        config.run_constraint_sensitivity = False
        config.run_step_size_sensitivity = False
        config.run_overlapping_extension = False
    if args.quick:
        config = config.smoke_test()
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    config = build_config_from_args(argv)
    run_experiment(config, verbose=True)
    return 0


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(main())
