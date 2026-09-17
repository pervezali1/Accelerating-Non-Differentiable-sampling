"""Sections 14-18: the automatic checks, the outputs and the CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pnral.config import ExperimentConfig
from pnral.diagnostics import (collect_warnings, coefficient_table,
                               compare_summaries, diagnostics_table,
                               summarize_alpha_run)
from pnral.experiment import build_config_from_args, run_experiment
from pnral.prediction import posterior_mean_probabilities, predictive_metrics


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """A complete (very small) experiment on the synthetic surrogate."""
    config = ExperimentConfig()
    config.data.use_synthetic = True
    config.data.synthetic_n_rows = 1500
    config.output_dir = str(tmp_path_factory.mktemp("pnral_run"))
    config.sampler.alphas = (0.0, 0.5)
    config.sampler.number_of_chains = 2
    config.sampler.n_iterations = 400
    config.sampler.burn_in = 100
    config.sampler.calibration_iterations = 120
    config.constraint.pilot_iterations = 300
    config.constraint.pilot_burn_in = 100
    config.sensitivity_iterations = 200
    config.sensitivity_burn_in = 50
    config.sensitivity_alphas = (0.0,)
    return run_experiment(config, verbose=False)


def test_all_automatic_checks_pass(tiny_run):
    """The fourteen checks of section 14, as run by the driver itself."""
    checks = tiny_run["checks"]
    assert len(checks) == 14
    failed = checks.loc[~checks["passed"], "check"].tolist()
    assert failed == [], f"failing automatic checks: {failed}"


def test_every_required_artefact_is_written(tiny_run):
    root = Path(tiny_run["output_dir"])
    for relative in ["config.json", "constraint.json", "preprocessing.json",
                     "standard_scaler.joblib", "run_metadata.json", "summary.md",
                     "tables/automatic_checks.csv", "tables/diagnostics_summary.csv",
                     "tables/coefficient_summary.csv", "tables/predictive_results.csv",
                     "tables/diagnostics_traces.csv"]:
        assert (root / relative).is_file(), relative
    assert list((root / "samples").glob("*.npz"))
    # every figure exists in both PDF and 300-dpi PNG
    pdfs = {p.stem for p in (root / "figures").glob("*.pdf")}
    pngs = {p.stem for p in (root / "figures").glob("*.png")}
    assert pdfs == pngs and len(pdfs) >= 16


def test_configuration_round_trips(tiny_run):
    root = Path(tiny_run["output_dir"])
    restored = ExperimentConfig.from_json(root / "config.json")
    assert restored.sampler.alphas == tiny_run["config"].sampler.alphas
    assert restored.constraint.p_constraint == 1.5


def test_saved_samples_are_feasible(tiny_run):
    from pnral.constraint import g_constraint
    root = Path(tiny_run["output_dir"])
    threshold = tiny_run["threshold"].Lambda_constraint
    with np.load(root / "samples" / "primary_alpha_0.npz") as archive:
        samples = archive["retained_samples"]
        assert samples.ndim == 3 and samples.shape[2] == 11
        values = g_constraint(samples.reshape(-1, samples.shape[2]), 1.5, 0.05)
        assert np.max(values) <= threshold + 1e-8


def test_threshold_is_frozen_across_alphas(tiny_run):
    thresholds = {run.Lambda_constraint for run in tiny_run["runs"]}
    assert len(thresholds) == 1
    selection = tiny_run["threshold"]
    assert selection.Lambda_constraint > selection.g_min
    assert selection.exceedance_fraction <= 0.002


def test_summary_states_the_truncation_interpretation(tiny_run):
    text = tiny_run["summary_markdown"]
    assert "truncated" in text.lower() or "truncation" in text.lower()
    assert "SYNTHETIC" in text                     # the surrogate is flagged


def test_predictive_probabilities_are_posterior_averages(tiny_run):
    from scipy.special import expit
    data = tiny_run["data"]
    samples = tiny_run["runs"][0].samples.reshape(-1, data.d)[:200]
    fast = posterior_mean_probabilities(samples, data.X_test, chunk_size=37)
    slow = expit(data.X_test @ samples.T).mean(axis=1)
    assert np.allclose(fast, slow)
    assert np.all((fast > 0) & (fast < 1))


def test_predictive_metrics_are_internally_consistent(tiny_run):
    report = tiny_run["reports"][0]
    total = (report["true_negative"] + report["false_positive"]
             + report["false_negative"] + report["true_positive"])
    assert total == report["n_test"]
    assert np.isclose(report["balanced_accuracy"],
                      0.5 * (report["sensitivity_recall_positive"]
                             + report["specificity_recall_negative"]))
    assert np.isclose(report["recall"], report["sensitivity_recall_positive"])
    assert 0.0 <= report["roc_auc"] <= 1.0


def test_diagnostics_tables_cover_every_alpha_and_coefficient(tiny_run):
    diagnostics = tiny_run["diagnostics"]
    coefficients = tiny_run["coefficients"]
    assert len(diagnostics) == len(tiny_run["runs"])
    assert set(coefficients["parameter"]) == set(tiny_run["data"].parameter_names)
    for column in ["min_ess", "median_ess", "min_ess_per_second", "max_split_rhat",
                   "mean_squared_jumping_distance", "projection_frequency",
                   "mean_projection_distance", "runtime_seconds",
                   "gradient_evaluations"]:
        assert column in diagnostics.columns
        assert np.all(np.isfinite(diagnostics[column].to_numpy(dtype=float)))


def test_warning_helper_fires_on_a_bad_run(tiny_run):
    summary = dict(tiny_run["summaries"][0])
    summary["max_rhat"] = 1.5
    summary["projection_frequency"] = 0.4
    summary["min_ess"] = 3.0
    summary["max_J_operator_norm"] = 1e6
    summary["max_g"] = tiny_run["threshold"].Lambda_constraint + 1.0
    messages = collect_warnings([summary], tiny_run["threshold"].Lambda_constraint)
    assert len(messages) == 5
    assert any("R-hat" in m for m in messages)
    assert any("projection frequency" in m for m in messages)
    assert any("constraint violated" in m for m in messages)


def test_sensitivity_comparison_detects_a_shift(tiny_run):
    reference = tiny_run["summaries"][0]
    shifted = dict(reference)
    shifted["posterior_mean"] = (np.asarray(reference["posterior_mean"])
                                 + 5.0 * np.asarray(reference["posterior_sd"]))
    shifted["label"] = "shifted"
    changed, worst, message = compare_summaries(reference, shifted)
    assert changed and worst > 4.0 and "substantial" in message


def test_cli_parses_the_documented_flags():
    config = build_config_from_args([
        "--synthetic", "--alphas", "0,0.25", "--chains", "3", "--iterations",
        "500", "--burn-in", "100", "--step-size", "1e-5", "--lambda-constraint",
        "4.5", "--triples-mode", "overlapping", "--no-sensitivity"])
    assert config.data.use_synthetic is True
    assert config.sampler.alphas == (0.0, 0.25)
    assert config.sampler.number_of_chains == 3
    assert config.sampler.step_size == 1e-5
    assert config.constraint.Lambda_constraint == 4.5
    assert config.sampler.triples_mode == "overlapping"
    assert not any([config.run_constraint_sensitivity,
                    config.run_step_size_sensitivity,
                    config.run_overlapping_extension])


def test_user_supplied_threshold_skips_the_pilot(tmp_path):
    config = ExperimentConfig()
    config.data.use_synthetic = True
    config.data.synthetic_n_rows = 1200
    config.output_dir = str(tmp_path / "fixed")
    config.constraint.Lambda_constraint = 5.0
    config.sampler.alphas = (0.0,)
    config.sampler.number_of_chains = 2
    config.sampler.n_iterations = 200
    config.sampler.burn_in = 50
    config.sampler.calibration_iterations = 80
    config.sensitivity_iterations = 120
    config.sensitivity_burn_in = 20
    config.sensitivity_alphas = (0.0,)
    config.run_constraint_sensitivity = False
    config.run_step_size_sensitivity = False
    config.run_overlapping_extension = False
    result = run_experiment(config, verbose=False)
    assert result["threshold"].from_pilot is False
    assert result["threshold"].Lambda_constraint == 5.0
    payload = json.loads((Path(result["output_dir"]) / "constraint.json").read_text())
    assert payload["Lambda_constraint"] == 5.0
