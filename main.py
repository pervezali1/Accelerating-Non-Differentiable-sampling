"""Run the complete constrained Non-Reversible Anchored Langevin study.

    python main.py                 # full study (~10 minutes)
    python main.py --quick         # short chains, for a smoke test
    python main.py --help          # all options

Stages
------
0.  Report every seed and configuration value.
1.  Structural verification of J, the anchor bounds and the projection (§6, §8).
2.  Exact-reference validation on the weighted-L1 target (§13), run *before*
    the logistic experiment as the specification asks.
3.  Synthetic logistic data (§2) and the smooth MAP.
4.  alpha comparison with 4+ independent chains (§7, §9, §10).
5.  Step-size sensitivity at h, h/2, h/4 (§9).
6.  Tight-constraint variant, where the boundary is actually reached.
7.  Posterior prediction on the test set (§11).
8.  Every figure of §10 plus the validation figures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

import diagnostics as dg
import reference_validation as rv
from config import (
    ConstraintConfig,
    DataConfig,
    ExperimentConfig,
    ProjectionConfig,
    ReferenceConfig,
    SamplerConfig,
    TargetConfig,
    describe_seeds,
    tight_constraint,
)
from constraint import boundary_point, g_value, unit_normal
from experiment import (
    derive_step_size,
    feasibility_audit,
    run_alpha_comparison,
    run_step_size_sensitivity,
    save_artifacts,
)
from nonreversible_matrix import (
    construct_J,
    divergence_J,
    operator_norm_J,
    skew_symmetry_error,
    tangency_residual,
)
from projection import check_kkt, project_onto_K
from synthetic_data import describe_dataset, generate_dataset
from target import LogisticTarget

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


# --------------------------------------------------------------------------
# Stage 1: structural verification
# --------------------------------------------------------------------------
def verify_structure(constraint: ConstraintConfig, target: LogisticTarget) -> dict:
    """Numerically confirm the identities the construction relies on."""
    p, eps = constraint.p_constraint, constraint.epsilon_constraint
    rng = np.random.default_rng(31337)
    states = [np.zeros(constraint.d)] + [
        rng.normal(scale=1.5, size=constraint.d) for _ in range(12)
    ]

    skew = max(skew_symmetry_error(construct_J(w, p, eps)) for w in states)
    divergence = max(float(np.abs(divergence_J(w, p, eps)).max()) for w in states)
    norm_gap = max(
        abs(operator_norm_J(w, p, eps) - float(np.linalg.norm(construct_J(w, p, eps), 2)))
        for w in states
    )

    boundary_states = [
        boundary_point(rng.normal(size=constraint.d), constraint) for _ in range(8)
    ]
    tangency = max(tangency_residual(w, p, eps) for w in boundary_states)
    boundary_error = max(
        abs(float(g_value(w, p, eps)) - constraint.Lambda_constraint)
        for w in boundary_states
    )

    kkt_residual, feasibility = 0.0, 0.0
    for _ in range(15):
        y = rng.normal(scale=3.0, size=constraint.d)
        result = project_onto_K(y, constraint)
        audit = check_kkt(result, y, constraint)
        assert audit["satisfied"], audit
        kkt_residual = max(kkt_residual, result.kkt_residual)
        feasibility = max(feasibility, result.g_value - constraint.Lambda_constraint)

    anchor_ok = all(target.check_anchor_bounds(w) for w in states)
    log_a_range = (
        min(target.log_a(w) for w in states),
        max(target.log_a(w) for w in states),
    )

    report = {
        "g(0)": constraint.g_at_origin,
        "d * epsilon_constraint**p_constraint": constraint.d * eps ** p,
        "Lambda_constraint": constraint.Lambda_constraint,
        "Lambda_constraint > g(0)": constraint.Lambda_constraint > constraint.g_at_origin,
        "max |J + J^T|": skew,
        "max |div J| (centred finite differences)": divergence,
        "max ||J n|| on the boundary": tangency,
        "max |g(boundary point) - Lambda|": boundary_error,
        "max |closed-form ||J|| - SVD|": norm_gap,
        "max projection KKT stationarity residual": kkt_residual,
        "max projection feasibility excess": feasibility,
        "anchor bounds hold": anchor_ok,
        "log_a lower bound (-8*lambda_lasso*delta_anchor)": target.log_a_lower_bound,
        "observed log_a range": log_a_range,
        "a lower bound": target.a_lower_bound,
    }
    for key, value in report.items():
        print(f"  {key:<48} {value}")
    return report


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_config(args: argparse.Namespace) -> ExperimentConfig:
    sampler = SamplerConfig(
        n_iterations=args.iterations,
        burn_in=args.burn_in,
        thin=args.thin,
        step_scale=args.step_scale,
    )
    return ExperimentConfig(
        n_chains=args.chains,
        sampler=sampler,
        sensitivity_iterations=args.sensitivity_iterations,
        sensitivity_burn_in=args.sensitivity_iterations // 4,
        output_dir=args.output_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--burn-in", dest="burn_in", type=int, default=5_000)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--step-scale", dest="step_scale", type=float, default=0.20)
    parser.add_argument(
        "--sensitivity-iterations", dest="sensitivity_iterations", type=int, default=8_000
    )
    parser.add_argument("--output-dir", dest="output_dir", default="results")
    parser.add_argument("--quick", action="store_true", help="short chains for a smoke test")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-tight", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)

    if args.quick:
        args.iterations, args.burn_in, args.thin = 4_000, 1_000, 2
        args.sensitivity_iterations = 2_000

    total_start = time.perf_counter()
    cfg = build_config(args)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---------------- Stage 0: seeds and configuration ----------------
    banner("0. SEEDS AND CONFIGURATION")
    for name, value in describe_seeds().items():
        print(f"  {name:<20} {value}")
    constraint = cfg.constraint
    print(
        f"\n  d = {constraint.d}, p_constraint = {constraint.p_constraint}, "
        f"epsilon_constraint = {constraint.epsilon_constraint}, "
        f"radius_budget = {constraint.radius_budget}"
    )
    print(
        f"  g(0) = d*eps^p = {constraint.g_at_origin:.6f},  "
        f"Lambda_constraint = {constraint.Lambda_constraint:.6f}"
    )
    print(
        f"  lambda_lasso = {cfg.target.lambda_lasso}, "
        f"delta_anchor = {cfg.target.delta_anchor}, "
        f"sigma_intercept = {cfg.target.sigma_intercept}"
    )
    print(
        f"  alphas = {list(cfg.alphas)}, swirl s = {cfg.sampler.swirl} "
        f"(only the product alpha*s matters), chains = {cfg.n_chains}"
    )
    print(
        f"  iterations = {cfg.sampler.n_iterations}, burn_in = {cfg.sampler.burn_in}, "
        f"thin = {cfg.sampler.thin}, stored draws per chain = {cfg.sampler.n_stored}"
    )

    # ---------------- Stage 3 (data first, needed by stage 1) ----------------
    banner("1. SYNTHETIC LOGISTIC-REGRESSION DATA")
    dataset = generate_dataset(cfg.data, constraint)
    description = describe_dataset(dataset, constraint)
    for key, value in description.items():
        print(f"  {key:<24} {value}")
    if dataset.shrink_factor == 1.0:
        print(
            "\n  beta_raw already satisfies g(beta_raw) <= midpoint level, "
            "so no rescaling was needed (t = 1)."
        )
    target = LogisticTarget.from_config(dataset.X_train, dataset.y_train, cfg.target)

    # ---------------- Stage 1: structural verification ----------------
    banner("2. STRUCTURAL VERIFICATION OF J, THE ANCHOR AND THE PROJECTION")
    verification = verify_structure(constraint, target)
    print(
        "\n  div J = 0 exactly, so NO div(J) correction is added to the Langevin drift."
    )

    # ---------------- Stage 2: exact-reference validation ----------------
    reference_output = None
    if not args.skip_reference:
        banner("3. EXACT-REFERENCE VALIDATION (weighted-L1 target on the same K)")
        # These are the h (divisor 1) settings; run_reference_experiment scales
        # them with the step-size divisor so every run is time-matched.
        reference_cfg = (
            ReferenceConfig(n_iterations=40_000, burn_in=8_000, thin=10)
            if not args.quick
            else ReferenceConfig(n_iterations=6_000, burn_in=1_500, thin=3)
        )
        print(f"  omega = {np.round(reference_cfg.omega, 3).tolist()}")
        reference_output = rv.run_reference_experiment(
            reference_cfg, constraint, cfg.projection,
            step_divisors=(1.0, 2.0, 4.0) if not args.quick else (1.0,),
        )
        comparison = reference_output["comparison"]
        comparison.to_csv(
            os.path.join(cfg.output_dir, "reference_comparison.csv"), index=False
        )
        reference_output["summary"].to_csv(
            os.path.join(cfg.output_dir, "reference_summary.csv"), index=False
        )
        print(
            "\n  Aggregated distances to the exact reference (worst coordinate).\n"
            "  'mean_error_in_se' < ~2 means the discrepancy is Monte-Carlo noise,\n"
            "  not discretisation bias; compare 'var_rel_error' against 'var_rel_se'."
        )
        print(
            comparison.groupby("run")[
                ["mean_abs_error", "mc_se_mean", "mean_error_in_se", "var_rel_error",
                 "var_rel_se", "wasserstein_1d", "energy_1d", "energy_multivariate"]
            ].max().to_string()
        )

    # ---------------- Stage 4: alpha comparison ----------------
    banner("4. ALPHA COMPARISON (projected non-reversible anchored Langevin)")
    step_size = derive_step_size(target, cfg.sampler.step_scale)
    print(
        f"  Lipschitz bound L = {target.lipschitz_constant():.2f}  ->  "
        f"step_size h = step_scale / L = {step_size:.3e}"
    )
    print(f"  smooth MAP = {np.round(target.smooth_map(), 4).tolist()}")
    print(f"  starting-point policy = {cfg.start_policy} "
          f"(jitter sd {cfg.start_jitter_sd}), identical for every alpha\n")
    results = run_alpha_comparison(dataset, target, cfg, constraint, step_size)

    print("\n  Summary by alpha:")
    summary_frame = results.summary_frame()
    print(summary_frame.to_string(index=False))

    print("\n  Feasibility audit (every stored state must lie in K):")
    audit = feasibility_audit(results)
    print(audit.to_string(index=False))
    assert bool(audit["all_feasible"].all()), "a sampled state left K"

    print("\n  Per-coefficient ESS / R-hat / IAT (first 18 rows):")
    print(results.coefficient_frame().head(18).to_string(index=False))

    # ---------------- Stage 7: posterior prediction ----------------
    banner("5. POSTERIOR PREDICTION ON THE TEST SET")
    predictive_frame = results.predictive_frame()
    print(predictive_frame.to_string(index=False))
    print("\n  Confusion matrix (alpha = 0):")
    print(results.predictives[0].confusion_matrix)
    print(
        f"\n  ||posterior_mean - beta_true||_2 by alpha: "
        f"{ {p.alpha: round(p.l2_error_vs_beta_true, 4) for p in results.predictives} }"
    )

    # ---------------- Stage 5: step-size sensitivity ----------------
    sensitivity = None
    if not args.skip_sensitivity:
        banner("6. STEP-SIZE SENSITIVITY (h, h/2, h/4)")
        sensitivity = run_step_size_sensitivity(target, cfg, step_size, constraint)
        sensitivity.to_csv(
            os.path.join(cfg.output_dir, "step_size_sensitivity.csv"), index=False
        )
        print("\n" + sensitivity.to_string(index=False))

    # ---------------- Stage 6: tight-constraint variant ----------------
    tight_results = None
    if not args.skip_tight:
        banner("7. TIGHT-CONSTRAINT VARIANT (the boundary is actually reached)")
        tight = tight_constraint(constraint)
        print(
            f"  radius_budget {constraint.radius_budget} -> {tight.radius_budget}, "
            f"Lambda_constraint {constraint.Lambda_constraint:.4f} -> "
            f"{tight.Lambda_constraint:.4f}"
        )
        print(
            "  Everything else (target, step size, seeds, starting policy) is unchanged.\n"
        )
        tight_cfg = replace(
            cfg,
            sampler=replace(
                cfg.sampler,
                n_iterations=max(4_000, cfg.sampler.n_iterations // 2),
                burn_in=max(1_000, cfg.sampler.burn_in // 2),
            ),
            output_dir=os.path.join(cfg.output_dir, "tight"),
        )
        tight_results = run_alpha_comparison(
            dataset, target, tight_cfg, tight, step_size
        )
        tight_frame = tight_results.summary_frame()
        print("\n" + tight_frame.to_string(index=False))
        os.makedirs(tight_cfg.output_dir, exist_ok=True)
        tight_frame.to_csv(
            os.path.join(tight_cfg.output_dir, "summary_by_alpha.csv"), index=False
        )
        tight_audit = feasibility_audit(tight_results)
        print("\n  Feasibility audit under the tight constraint:")
        print(tight_audit.to_string(index=False))
        assert bool(tight_audit["all_feasible"].all())

    # ---------------- Stage 8: figures ----------------
    if not args.skip_plots:
        banner("8. FIGURES")
        paths = []
        paths += dg.plot_traces(
            results.chains_by_alpha, results.burn_in, cfg.output_dir, results.names
        )
        paths.append(dg.plot_autocorrelation(results.samples_by_alpha, cfg.output_dir, results.names))
        paths.append(dg.plot_ess_by_coefficient(results.summaries, cfg.output_dir))
        paths.append(dg.plot_ess_per_second(results.summaries, cfg.output_dir))
        paths.append(dg.plot_rhat(results.summaries, cfg.output_dir))
        paths.append(
            dg.plot_posterior_intervals(
                results.samples_by_alpha, dataset.beta_true, cfg.output_dir, results.names
            )
        )
        paths.append(
            dg.plot_g_histogram(
                results.chains_by_alpha, results.burn_in, constraint.Lambda_constraint,
                constraint.g_at_origin, cfg.output_dir,
            )
        )
        paths.append(
            dg.plot_distance_to_boundary(results.chains_by_alpha, results.burn_in, cfg.output_dir)
        )
        paths.append(
            dg.plot_projection_frequency(
                results.summaries, results.chains_by_alpha, results.burn_in, cfg.output_dir
            )
        )
        paths.append(
            dg.plot_operator_norm(results.chains_by_alpha, results.burn_in, cfg.output_dir)
        )
        paths.append(dg.plot_pairs(results.samples_by_alpha, cfg.output_dir, names=results.names))
        paths.append(dg.plot_calibration(results.predictives, dataset.y_test, cfg.output_dir))
        paths.append(dg.plot_confusion(results.predictives[0], cfg.output_dir))
        if sensitivity is not None:
            paths.append(dg.plot_step_size_sensitivity(sensitivity, cfg.output_dir))
        if tight_results is not None:
            tight_dir = os.path.join(cfg.output_dir, "tight")
            paths.append(
                dg.plot_distance_to_boundary(
                    tight_results.chains_by_alpha, tight_results.burn_in, tight_dir
                )
            )
            paths.append(
                dg.plot_projection_frequency(
                    tight_results.summaries, tight_results.chains_by_alpha,
                    tight_results.burn_in, tight_dir,
                )
            )
            paths.append(
                dg.plot_g_histogram(
                    tight_results.chains_by_alpha, tight_results.burn_in,
                    tight_results.constraint.Lambda_constraint,
                    tight_results.constraint.g_at_origin, tight_dir,
                )
            )
        if reference_output is not None:
            reference_dir = os.path.join(cfg.output_dir, "reference")
            draws = reference_output["reference"].draws
            samples = reference_output["samples_by_key"]
            paths.append(rv.plot_reference_cdfs(samples, draws, reference_dir))
            paths.append(rv.plot_reference_moments(reference_output["comparison"], reference_dir))
            paths.append(rv.plot_reference_pairs(samples, draws, reference_dir))
            paths.append(
                rv.plot_reference_boundary(samples, draws, constraint, reference_dir)
            )
        for path in paths:
            print(f"  wrote {path}")

    # ---------------- Persist ----------------
    extra = {"verification": {k: str(v) for k, v in verification.items()}}
    if reference_output is not None:
        extra["reference_acceptance_probability"] = (
            reference_output["reference"].acceptance_probability
        )
        extra["reference_warning"] = reference_output["reference"].warning
    save_artifacts(results, dataset, cfg, cfg.output_dir, extra)

    banner("DONE")
    print(f"  total wall-clock time: {time.perf_counter() - total_start:.1f} s")
    print(f"  artifacts written to: {os.path.abspath(cfg.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
