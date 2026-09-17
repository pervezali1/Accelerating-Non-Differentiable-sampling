"""Convergence/efficiency diagnostics, automatic checks and manuscript figures.

Per-iteration traces come straight from :class:`pnral.sampler.ChainResult`;
effective sample sizes and split-Rhat are computed with ArviZ.

Colour policy
-------------
``alpha`` is an *ordered* quantity, so the alpha series use a single-hue
ordinal blue ramp (light -> dark) with distinct markers as a secondary
encoding; chains use a categorical palette validated for all-pairs
colour-vision separation.  Every multi-series panel carries a legend, so
identity is never colour-alone.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import arviz as az
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .constraint import g_constraint                     # noqa: E402
from .sampler import AlphaRun, ChainResult               # noqa: E402

__all__ = [
    "PALETTE",
    "apply_style",
    "save_figure",
    "to_inference_data",
    "summarize_alpha_run",
    "diagnostics_table",
    "coefficient_table",
    "collect_warnings",
    "run_automatic_checks",
    "make_all_diagnostic_figures",
]

# ----------------------------------------------------------------------
# Design tokens (validated light-mode palette; figures are static PDF/PNG)
# ----------------------------------------------------------------------
PALETTE: Dict[str, object] = {
    "surface": "#fcfcfb",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
    # ordinal blue ramp (steps 250/350/450/550/650) -- for ordered alpha
    "ordinal5": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
    "ordinal3": ["#86b6ef", "#2a78d6", "#104281"],
    # categorical slots, all-pairs validated for 4 series (chains)
    "categorical": ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"],
    "critical": "#d03b3b",
    "warning": "#fab219",
    "good": "#0ca30c",
}

_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
_LINESTYLES = ["-", "--", "-.", ":"]


def apply_style() -> None:
    """Set recessive chrome, thin marks and readable ink for all figures."""
    plt.rcParams.update({
        "figure.facecolor": PALETTE["surface"],
        "axes.facecolor": PALETTE["surface"],
        "savefig.facecolor": PALETTE["surface"],
        "axes.edgecolor": PALETTE["baseline"],
        "axes.labelcolor": PALETTE["text_secondary"],
        "axes.titlecolor": PALETTE["text_primary"],
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.7,
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": PALETTE["text_primary"],
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.4,
        "figure.dpi": 110,
        "font.size": 9.5,
    })


def _group_spread(n_groups: int) -> float:
    """Offset between adjacent series so a group stays centred on its tick.

    The total width of a group never exceeds 0.5 of the category slot, which
    keeps every mark visibly attached to its own axis label.
    """
    total_width = min(0.5, 0.18 * n_groups)
    return total_width / max(n_groups - 1, 1)


def alpha_colors(n: int) -> List[str]:
    """Ordinal ramp with ``n`` steps for ``n`` ordered alpha values."""
    ramp = PALETTE["ordinal5"] if n > 3 else PALETTE["ordinal3"]
    if n == len(ramp):
        return list(ramp)
    index = np.linspace(0, len(ramp) - 1, n).round().astype(int)
    return [ramp[i] for i in index]


def save_figure(fig: plt.Figure, output_dir: str | Path, name: str) -> Dict[str, str]:
    """Save ``fig`` as both a vector PDF and a 300-dpi PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{name}.pdf"
    png_path = output_dir / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf_path), "png": str(png_path)}


# ----------------------------------------------------------------------
# ArviZ summaries
# ----------------------------------------------------------------------
def to_inference_data(run: AlphaRun, parameter_names: Sequence[str]):
    """Wrap the retained draws of an :class:`AlphaRun` in an ``InferenceData``."""
    samples = run.samples                     # (chain, draw, d)
    posterior = {name: samples[:, :, index]
                 for index, name in enumerate(parameter_names)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return az.from_dict(posterior=posterior)


def summarize_alpha_run(run: AlphaRun, parameter_names: Sequence[str]) -> Dict[str, object]:
    """Compute every per-alpha diagnostic required by section 15.

    Returns a dictionary with scalar summaries plus per-coefficient arrays for
    ESS (bulk/tail), split-Rhat, integrated autocorrelation time, posterior
    means, standard deviations and 95% equal-tailed credible intervals.
    """
    idata = to_inference_data(run, parameter_names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ess_bulk = az.ess(idata, method="bulk")
        ess_tail = az.ess(idata, method="tail")
        rhat = az.rhat(idata, method="split")

    names = list(parameter_names)
    ess_bulk_values = np.array([float(ess_bulk[name].values) for name in names])
    ess_tail_values = np.array([float(ess_tail[name].values) for name in names])
    rhat_values = np.array([float(rhat[name].values) for name in names])

    samples = run.samples
    n_chains, n_draws, _ = samples.shape
    total_draws = n_chains * n_draws
    pooled = samples.reshape(total_draws, -1)

    # tau_int = N / ESS is the integrated autocorrelation time of the estimator.
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.where(ess_bulk_values > 0, total_draws / ess_bulk_values, np.inf)

    runtime = run.runtime
    projection_frequency = float(np.mean([c.projection_frequency for c in run.chains]))
    projection_distance = float(np.mean([c.mean_projection_distance for c in run.chains]))
    msjd = float(np.mean([c.mean_squared_jumping_distance for c in run.chains]))

    return {
        "alpha": run.alpha,
        "label": run.label,
        "triples_mode": run.triples_mode,
        "step_size": run.step_size,
        "Lambda_constraint": run.Lambda_constraint,
        "n_chains": int(n_chains),
        "n_draws_per_chain": int(n_draws),
        "total_draws": int(total_draws),
        "parameter_names": names,
        "ess_bulk": ess_bulk_values,
        "ess_tail": ess_tail_values,
        "rhat_split": rhat_values,
        "tau_int": tau,
        "min_ess": float(np.min(ess_bulk_values)),
        "median_ess": float(np.median(ess_bulk_values)),
        "max_rhat": float(np.max(rhat_values)),
        "runtime_seconds": float(runtime),
        "min_ess_per_second": float(np.min(ess_bulk_values) / runtime) if runtime > 0 else np.nan,
        "median_ess_per_second": float(np.median(ess_bulk_values) / runtime) if runtime > 0 else np.nan,
        "gradient_evaluations": int(run.gradient_evaluations),
        "mean_squared_jumping_distance": msjd,
        "projection_frequency": projection_frequency,
        "mean_projection_distance": projection_distance,
        "posterior_mean": pooled.mean(axis=0),
        "posterior_sd": pooled.std(axis=0, ddof=1),
        "credible_low": np.quantile(pooled, 0.025, axis=0),
        "credible_high": np.quantile(pooled, 0.975, axis=0),
        "mean_J_operator_norm": float(np.mean([c.J_operator_norm.mean() for c in run.chains])),
        "max_J_operator_norm": float(np.max([c.J_operator_norm.max() for c in run.chains])),
        "mean_reversible_drift_norm": float(np.mean([c.reversible_drift_norm.mean() for c in run.chains])),
        "mean_nonreversible_drift_norm": float(np.mean([c.nonreversible_drift_norm.mean() for c in run.chains])),
        "mean_anchor_coefficient": float(np.mean([c.a[c.burn_in:].mean() for c in run.chains])),
        "min_anchor_coefficient": float(np.min([c.a.min() for c in run.chains])),
        "max_g": float(np.max([c.g.max() for c in run.chains])),
        "min_slack": float(np.min([np.nanmin(c.slack) for c in run.chains])),
    }


def diagnostics_table(summaries: Sequence[Dict[str, object]]) -> pd.DataFrame:
    """One row of scalar diagnostics per alpha (or per sensitivity setting)."""
    rows = []
    for summary in summaries:
        rows.append({
            "label": summary["label"],
            "alpha": summary["alpha"],
            "triples_mode": summary["triples_mode"],
            "step_size": summary["step_size"],
            "Lambda_constraint": summary["Lambda_constraint"],
            "n_chains": summary["n_chains"],
            "draws_per_chain": summary["n_draws_per_chain"],
            "total_draws": summary["total_draws"],
            "min_ess": summary["min_ess"],
            "median_ess": summary["median_ess"],
            "min_ess_per_second": summary["min_ess_per_second"],
            "median_ess_per_second": summary["median_ess_per_second"],
            "max_split_rhat": summary["max_rhat"],
            "max_tau_int": float(np.max(summary["tau_int"])),
            "mean_squared_jumping_distance": summary["mean_squared_jumping_distance"],
            "projection_frequency": summary["projection_frequency"],
            "mean_projection_distance": summary["mean_projection_distance"],
            "runtime_seconds": summary["runtime_seconds"],
            "gradient_evaluations": summary["gradient_evaluations"],
            "mean_J_operator_norm": summary["mean_J_operator_norm"],
            "max_J_operator_norm": summary["max_J_operator_norm"],
            "mean_reversible_drift_norm": summary["mean_reversible_drift_norm"],
            "mean_nonreversible_drift_norm": summary["mean_nonreversible_drift_norm"],
            "mean_anchor_coefficient": summary["mean_anchor_coefficient"],
            "min_anchor_coefficient": summary["min_anchor_coefficient"],
            "max_g": summary["max_g"],
            "min_slack": summary["min_slack"],
        })
    return pd.DataFrame(rows)


def coefficient_table(summaries: Sequence[Dict[str, object]]) -> pd.DataFrame:
    """Per-coefficient posterior and diagnostic summaries, long format."""
    rows = []
    for summary in summaries:
        for index, name in enumerate(summary["parameter_names"]):
            rows.append({
                "label": summary["label"],
                "alpha": summary["alpha"],
                "parameter": name,
                "posterior_mean": float(summary["posterior_mean"][index]),
                "posterior_sd": float(summary["posterior_sd"][index]),
                "credible_2.5%": float(summary["credible_low"][index]),
                "credible_97.5%": float(summary["credible_high"][index]),
                "ess_bulk": float(summary["ess_bulk"][index]),
                "ess_tail": float(summary["ess_tail"][index]),
                "split_rhat": float(summary["rhat_split"][index]),
                "tau_int": float(summary["tau_int"][index]),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Warnings and automatic checks
# ----------------------------------------------------------------------
def collect_warnings(
    summaries: Sequence[Dict[str, object]],
    Lambda_constraint: float,
    constraint_tolerance: float = 1e-8,
    rhat_threshold: float = 1.01,
    projection_frequency_threshold: float = 0.10,
    min_ess_threshold: float = 100.0,
    J_norm_threshold: float = 1e3,
) -> List[str]:
    """Human-readable warnings for the conditions listed in section 15."""
    messages: List[str] = []
    for summary in summaries:
        tag = summary["label"] or f"alpha={summary['alpha']}"
        if summary["max_rhat"] > rhat_threshold:
            messages.append(
                f"[{tag}] split R-hat = {summary['max_rhat']:.4f} exceeds "
                f"{rhat_threshold}: the chains have not mixed.")
        if summary["projection_frequency"] > projection_frequency_threshold:
            messages.append(
                f"[{tag}] projection frequency = "
                f"{summary['projection_frequency']:.1%} exceeds "
                f"{projection_frequency_threshold:.0%}: the constraint is very "
                f"active and the truncation bias is large.")
        if summary["max_g"] > Lambda_constraint + constraint_tolerance:
            messages.append(
                f"[{tag}] constraint violated: max g(w) = {summary['max_g']:.6g} "
                f"> Lambda_constraint = {Lambda_constraint:.6g}.")
        if summary["min_ess"] < min_ess_threshold:
            messages.append(
                f"[{tag}] minimum ESS = {summary['min_ess']:.1f} is very low "
                f"(< {min_ess_threshold:.0f}): run longer or re-tune the step size.")
        if summary["max_J_operator_norm"] > J_norm_threshold:
            messages.append(
                f"[{tag}] ||J||_2 reaches {summary['max_J_operator_norm']:.3g}: "
                f"the non-reversible drift may be numerically too large.")
    return messages


def compare_summaries(reference: Dict[str, object], other: Dict[str, object],
                      tolerance_in_sd: float = 0.25) -> Tuple[bool, float, str]:
    """Compare two runs' posterior means in units of the reference SD.

    Returns ``(substantially_different, max_shift, message)``.  Used for the
    step-size (h vs h/2 vs h/4) and constraint-radius sensitivity warnings.
    """
    shift = np.abs(np.asarray(reference["posterior_mean"])
                   - np.asarray(other["posterior_mean"]))
    scaled = shift / np.maximum(np.asarray(reference["posterior_sd"]), 1e-12)
    worst = float(np.max(scaled))
    changed = bool(worst > tolerance_in_sd)
    message = (f"posterior means of '{other['label']}' differ from "
               f"'{reference['label']}' by up to {worst:.2f} posterior SD "
               f"({'substantial' if changed else 'within tolerance'}, "
               f"threshold {tolerance_in_sd}).")
    return changed, worst, message


def run_automatic_checks(
    runs: Sequence[AlphaRun],
    target,
    p_constraint: float,
    epsilon_constraint: float,
    Lambda_constraint: float,
    j_spec,
    data,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """The fourteen automatic checks of section 14, as a tidy table.

    Each row is ``(check, passed, detail)``.  The checks are run against the
    *actual* sampler output, not against a re-derivation, so they fail loudly
    if any part of the pipeline drifts from the specification.
    """
    from .anchor import anchor_bounds, log_anchor_coefficient
    from .constraint import g_minimum, outward_normal
    from .nonreversible_matrix import (check_boundary_tangency,
                                       check_divergence_free,
                                       check_skew_symmetry)
    from .projection import kkt_residual, project_K
    from .sampler import run_chain

    d = target.d
    rows: List[Dict[str, object]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        rows.append({"check": name, "passed": bool(passed), "detail": detail})

    # 1 -- g(0) = d * epsilon ** p
    zero_value = g_constraint(np.zeros(d), p_constraint, epsilon_constraint)
    expected = g_minimum(d, p_constraint, epsilon_constraint)
    record("1. g(0) = d * epsilon**p", abs(zero_value - expected) < 1e-12,
           f"g(0)={zero_value:.12g}, d*eps**p={expected:.12g}")

    # 2 -- Lambda_constraint > g(0)
    record("2. Lambda_constraint > g(0)", Lambda_constraint > expected,
           f"Lambda={Lambda_constraint:.6g} vs g(0)={expected:.6g}")

    # 3 -- every sampled state is feasible
    worst_g = max(float(np.max(chain.g)) for run in runs for chain in run.chains)
    record("3. all states satisfy g(w) <= Lambda + tol",
           worst_g <= Lambda_constraint + tolerance,
           f"max over all chains g(w_k) = {worst_g:.12g} <= "
           f"{Lambda_constraint + tolerance:.12g}")

    # 4/5/6 -- structure of J on sampled states and on the boundary
    rng = np.random.default_rng(12345)
    pooled = np.concatenate([run.samples.reshape(-1, d) for run in runs], axis=0)
    probe = pooled[rng.choice(pooled.shape[0], size=min(25, pooled.shape[0]),
                              replace=False)]
    skew_residual = max(check_skew_symmetry(j_spec.builder(w)) for w in probe)
    record("4. J(w)^T = -J(w)", skew_residual <= 1e-12,
           f"max |J + J^T| = {skew_residual:.3g} over {probe.shape[0]} states")

    divergence_residual = max(
        check_divergence_free(j_spec.builder, w, step=1e-5, tolerance=np.inf)
        for w in probe[:10])
    record("5. finite-difference div J = 0", divergence_residual <= 1e-8,
           f"max |div J| = {divergence_residual:.3g} (centred differences, h=1e-5)")

    from scipy.optimize import brentq as _brentq
    boundary_residuals = []
    for _ in range(10):                       # points exactly on {g = Lambda}
        direction = rng.standard_normal(d)
        scale = _brentq(
            lambda t: g_constraint(t * direction, p_constraint, epsilon_constraint)
            - Lambda_constraint, 1e-8, 1e4)
        boundary_point = scale * direction
        normal = outward_normal(boundary_point, p_constraint, epsilon_constraint)
        boundary_residuals.append(
            check_boundary_tangency(j_spec.builder(boundary_point), normal,
                                    tolerance=np.inf))
    worst_boundary = max(boundary_residuals)
    record("6. J(w) n(w) = 0 on the boundary", worst_boundary <= 1e-10,
           f"max ||J n||_inf = {worst_boundary:.3g} over 10 boundary points")

    # 7 -- anchor coefficient bounds
    log_a_min, log_a_max, a_min, a_max = anchor_bounds(target)
    observed_log_a = np.array([log_anchor_coefficient(target, w) for w in probe])
    ok = bool(np.all(observed_log_a >= log_a_min - 1e-10)
              and np.all(observed_log_a <= log_a_max + 1e-10))
    record("7. anchor bounds -10*lambda*delta <= log a <= 0", ok,
           f"observed log a in [{observed_log_a.min():.6g}, "
           f"{observed_log_a.max():.6g}], theory [{log_a_min:.6g}, {log_a_max:.6g}]; "
           f"a in [{np.exp(observed_log_a.min()):.6g}, {np.exp(observed_log_a.max()):.6g}] "
           f"subset of [{a_min:.6g}, {a_max:.6g}]")

    # 8 -- the intercept carries no L1 penalty
    probe_w = np.zeros(d)
    probe_w[0] = 3.0
    penalty_intercept = target.l1_penalty(probe_w)
    probe_w = np.zeros(d)
    probe_w[1] = 3.0
    penalty_slope = target.l1_penalty(probe_w)
    record("8. intercept excluded from the L1 penalty",
           penalty_intercept == 0.0 and np.isclose(penalty_slope,
                                                   3.0 * target.lambda_lasso),
           f"penalty(w0=3)={penalty_intercept:.6g}, "
           f"penalty(w1=3)={penalty_slope:.6g} = 3*lambda_lasso")

    # 9 -- projection KKT conditions
    residuals = []
    for _ in range(20):
        y = pooled[rng.integers(pooled.shape[0])] + 0.6 * rng.standard_normal(d)
        result = project_K(y, p_constraint, epsilon_constraint, Lambda_constraint)
        kkt = kkt_residual(y, result, p_constraint, epsilon_constraint,
                           Lambda_constraint)
        residuals.append(max(kkt["stationarity"], kkt["primal_violation"],
                             abs(kkt["dual_violation"])))
    worst_kkt = float(max(residuals))
    record("9. projection satisfies the KKT conditions", worst_kkt <= 1e-8,
           f"max KKT residual = {worst_kkt:.3g} over 20 random points")

    # 10 -- alpha = 0 removes only the non-reversible drift
    reference = runs[0]
    chain0 = reference.chains[0]
    replay = run_chain(target=target, w0=chain0.states[0], alpha=0.0,
                       step_size=chain0.step_size, n_iterations=min(200, chain0.states.shape[0]),
                       seed=chain0.seed, j_spec=j_spec,
                       Lambda_constraint=Lambda_constraint,
                       p_constraint=p_constraint,
                       epsilon_constraint=epsilon_constraint, burn_in=0)
    zero_alpha_run = next((r for r in runs if r.alpha == 0.0), None)
    if zero_alpha_run is not None:
        reference_states = zero_alpha_run.chains[0].states[:replay.states.shape[0]]
        difference = float(np.max(np.abs(replay.states - reference_states)))
        record("10. alpha=0 removes only the non-reversible drift",
               difference == 0.0,
               f"the alpha=0 chain is reproduced bit-for-bit by the reversible "
               f"anchored update (max difference {difference:.3g})")
    else:
        record("10. alpha=0 removes only the non-reversible drift", True,
               "no alpha=0 run present in this comparison")

    # 11 -- no state-dependent rescaling of J
    w_probe = probe[0]
    J_reference = j_spec.builder(w_probe)
    doubled_spec = type(j_spec).create(j_spec.mode, d, p_constraint,
                                       epsilon_constraint, 2.0 * j_spec.swirl_scale)
    linear = float(np.max(np.abs(doubled_spec.builder(w_probe) - 2.0 * J_reference)))
    record("11. J uses a constant scaling only", linear <= 1e-12,
           f"J is exactly linear in swirl_scale (residual {linear:.3g}); the "
           f"block scale is the constant {j_spec.block_scale:.6g}, independent of w")

    # 12 -- the same constraint for every alpha
    thresholds = {float(run.Lambda_constraint) for run in runs}
    record("12. identical Lambda_constraint for all alpha", len(thresholds) == 1,
           f"Lambda_constraint values across runs: {sorted(thresholds)}")

    # 13 -- no NaN / inf anywhere
    finite = True
    for run in runs:
        for chain in run.chains:
            finite &= bool(np.all(np.isfinite(chain.states))
                           and np.all(np.isfinite(chain.U))
                           and np.all(np.isfinite(chain.U0))
                           and np.all(np.isfinite(chain.a))
                           and np.all(np.isfinite(chain.g)))
    record("13. no NaN or infinite quantities", finite,
           "all states, potentials, anchor coefficients and g-values are finite")

    # 14 -- label encoding
    labels_ok = (set(np.unique(data.y_train)) <= {0.0, 1.0}
                 and set(np.unique(data.y_test)) <= {0.0, 1.0}
                 and 0.0 < data.y_train.mean() < 1.0)
    observed = sorted(float(value) for value in np.unique(data.y_train))
    record("14. labels encoded g->1, h->0", labels_ok,
           f"train positives {data.y_train.mean():.4f}, "
           f"test positives {data.y_test.mean():.4f}, observed label values "
           f"{observed}")

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_traces(run: AlphaRun, parameter_names: Sequence[str],
                output_dir: str | Path, name: str = "fig01_traces") -> Dict[str, str]:
    """Coefficient trace plots, one panel per coefficient, coloured by chain."""
    apply_style()
    samples = run.samples
    n_chains, n_draws, d = samples.shape
    n_columns = 3
    n_rows = int(np.ceil(d / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(11, 2.0 * n_rows),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    colors = PALETTE["categorical"]
    for index in range(d):
        axis = axes[index]
        for chain in range(n_chains):
            axis.plot(samples[chain, :, index], linewidth=0.6,
                      color=colors[chain % len(colors)],
                      linestyle=_LINESTYLES[chain % len(_LINESTYLES)],
                      alpha=0.85, label=f"chain {chain + 1}" if index == 0 else None)
        axis.set_title(parameter_names[index], loc="left")
    for extra in range(d, axes.size):
        axes[extra].axis("off")
    for axis in axes[-n_columns:]:
        axis.set_xlabel("retained iteration")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=n_chains,
               bbox_to_anchor=(0.98, 0.02))
    fig.suptitle(f"Coefficient traces — alpha = {run.alpha:g}"
                 f"{(' (' + run.label + ')') if run.label else ''}",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return save_figure(fig, output_dir, name)


def plot_autocorrelation(run: AlphaRun, parameter_names: Sequence[str],
                         output_dir: str | Path, max_lag: int = 200,
                         name: str = "fig02_autocorrelation") -> Dict[str, str]:
    """Chain-averaged autocorrelation functions for every coefficient."""
    apply_style()
    samples = run.samples
    n_chains, n_draws, d = samples.shape
    lag_limit = int(min(max_lag, n_draws - 1))
    n_columns = 3
    n_rows = int(np.ceil(d / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(11, 2.0 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for index in range(d):
        acf = np.mean([az.autocorr(samples[c, :, index])[:lag_limit]
                       for c in range(n_chains)], axis=0)
        axis = axes[index]
        axis.axhline(0.0, color=PALETTE["baseline"], linewidth=0.8)
        axis.fill_between(np.arange(acf.size), 0.0, acf, color=PALETTE["ordinal5"][2],
                          alpha=0.25, linewidth=0)
        axis.plot(acf, color=PALETTE["ordinal5"][3], linewidth=1.2)
        axis.set_title(parameter_names[index], loc="left")
    for extra in range(d, axes.size):
        axes[extra].axis("off")
    for axis in axes[-n_columns:]:
        axis.set_xlabel("lag")
    axes[0].set_ylabel("autocorrelation")
    fig.suptitle(f"Autocorrelation — alpha = {run.alpha:g}", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, output_dir, name)


def plot_ess_by_coefficient(summaries: Sequence[Dict[str, object]],
                            output_dir: str | Path,
                            name: str = "fig03_ess_by_coefficient") -> Dict[str, str]:
    """Grouped bars: bulk ESS per coefficient for every alpha."""
    apply_style()
    names = summaries[0]["parameter_names"]
    n_groups = len(summaries)
    colors = alpha_colors(n_groups)
    positions = np.arange(len(names))
    width = 0.8 / n_groups
    fig, axis = plt.subplots(figsize=(10, 4.2))
    for index, summary in enumerate(summaries):
        offset = (index - (n_groups - 1) / 2) * width
        axis.bar(positions + offset, summary["ess_bulk"], width=width * 0.92,
                 color=colors[index], label=f"alpha = {summary['alpha']:g}")
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=45, ha="right")
    axis.set_ylabel("bulk ESS")
    axis.set_title("Effective sample size by coefficient", loc="left")
    axis.legend(ncol=n_groups, loc="upper left", bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    return save_figure(fig, output_dir, name)


def plot_ess_per_second_vs_alpha(summaries: Sequence[Dict[str, object]],
                                 output_dir: str | Path,
                                 name: str = "fig04_ess_per_second_vs_alpha"
                                 ) -> Dict[str, str]:
    """Minimum and median ESS per second against alpha (the headline metric)."""
    apply_style()
    alphas = [summary["alpha"] for summary in summaries]
    minimum = [summary["min_ess_per_second"] for summary in summaries]
    median = [summary["median_ess_per_second"] for summary in summaries]
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(alphas, minimum, marker="o", markersize=8,
              color=PALETTE["ordinal5"][4], label="minimum ESS / second")
    axis.plot(alphas, median, marker="s", markersize=8, linestyle="--",
              color=PALETTE["ordinal5"][2], label="median ESS / second")
    for x, y in zip(alphas, minimum):
        axis.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                      xytext=(0, 8), ha="center", fontsize=8,
                      color=PALETTE["text_secondary"])
    axis.set_xlabel("alpha (non-reversible strength)")
    axis.set_ylabel("ESS per second")
    axis.set_title("Sampling efficiency versus alpha", loc="left")
    axis.legend()
    fig.tight_layout()
    return save_figure(fig, output_dir, name)


def plot_rhat_by_coefficient(summaries: Sequence[Dict[str, object]],
                             output_dir: str | Path,
                             threshold: float = 1.01,
                             name: str = "fig05_rhat_by_coefficient") -> Dict[str, str]:
    """Split-Rhat per coefficient with the 1.01 convergence threshold marked."""
    apply_style()
    names = summaries[0]["parameter_names"]
    n_groups = len(summaries)
    colors = alpha_colors(n_groups)
    positions = np.arange(len(names))
    spread = _group_spread(n_groups)
    fig, axis = plt.subplots(figsize=(10, 4.2))
    # R-hat is read against its reference value 1, not against zero, so it is
    # shown as points on a natural (non-zero) baseline rather than as bars,
    # whose length would then encode nothing meaningful.
    axis.axhline(1.0, color=PALETTE["baseline"], linewidth=1.0)
    for index, summary in enumerate(summaries):
        offset = (index - (n_groups - 1) / 2) * spread
        axis.plot(positions + offset, summary["rhat_split"],
                  marker=_MARKERS[index % len(_MARKERS)], linestyle="none",
                  markersize=7, color=colors[index],
                  markeredgecolor=PALETTE["surface"], markeredgewidth=0.8,
                  label=f"alpha = {summary['alpha']:g}")
    axis.axhline(threshold, color=PALETTE["critical"], linewidth=1.2,
                 linestyle="--")
    axis.annotate(f"threshold {threshold}", (len(names) - 0.5, threshold),
                  textcoords="offset points", xytext=(-4, 4), ha="right",
                  fontsize=8, color=PALETTE["critical"])
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=45, ha="right")
    axis.set_ylabel("split R-hat")
    highest = float(np.max([s["rhat_split"].max() for s in summaries]))
    lowest = float(np.min([s["rhat_split"].min() for s in summaries]))
    axis.set_ylim(min(0.999, lowest - 0.002), max(threshold * 1.005, highest * 1.004))
    axis.set_title("Split R-hat by coefficient", loc="left")
    axis.legend(ncol=n_groups, loc="upper left", bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    return save_figure(fig, output_dir, name)


def plot_posterior_intervals(summaries: Sequence[Dict[str, object]],
                             output_dir: str | Path,
                             name: str = "fig06_posterior_intervals") -> Dict[str, str]:
    """Posterior means with 95% equal-tailed credible intervals, by alpha."""
    apply_style()
    names = summaries[0]["parameter_names"]
    n_groups = len(summaries)
    colors = alpha_colors(n_groups)
    base = np.arange(len(names))
    spread = _group_spread(n_groups)
    fig, axis = plt.subplots(figsize=(8.8, 5.4))
    for index, summary in enumerate(summaries):
        offset = (index - (n_groups - 1) / 2) * spread
        y = base + offset
        mean = summary["posterior_mean"]
        low = summary["credible_low"]
        high = summary["credible_high"]
        axis.hlines(y, low, high, color=colors[index], linewidth=2.0)
        axis.plot(mean, y, marker=_MARKERS[index % len(_MARKERS)], linestyle="none",
                  markersize=6, color=colors[index],
                  markeredgecolor=PALETTE["surface"], markeredgewidth=0.8,
                  label=f"alpha = {summary['alpha']:g}")
    axis.axvline(0.0, color=PALETTE["baseline"], linewidth=0.9)
    axis.set_yticks(base)
    axis.set_yticklabels(names)
    axis.invert_yaxis()
    axis.set_xlabel("coefficient value")
    axis.set_title("Posterior means and 95% credible intervals", loc="left")
    # the legend lives outside the data area so it can never cover a row
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                title="non-reversible strength")
    fig.tight_layout()
    return save_figure(fig, output_dir, name)


def plot_g_histogram(runs: Sequence[AlphaRun], Lambda_constraint: float,
                     g_min: float, output_dir: str | Path,
                     name: str = "fig07_g_histogram") -> Dict[str, str]:
    """Histogram of ``g(w)`` over retained draws, with the threshold marked."""
    apply_style()
    colors = alpha_colors(len(runs))
    fig, axis = plt.subplots(figsize=(7.6, 4.2))
    for index, run in enumerate(runs):
        values = np.concatenate([chain.g[chain.burn_in:] for chain in run.chains])
        axis.hist(values, bins=80, histtype="step", linewidth=1.4,
                  color=colors[index], label=f"alpha = {run.alpha:g}", density=True)
    axis.axvline(Lambda_constraint, color=PALETTE["critical"], linewidth=1.4,
                 linestyle="--")
    axis.annotate("Lambda_constraint", (Lambda_constraint, axis.get_ylim()[1]),
                  textcoords="offset points", xytext=(-6, -12), ha="right",
                  fontsize=8, color=PALETTE["critical"])
    axis.axvline(g_min, color=PALETTE["muted"], linewidth=1.0, linestyle=":")
    axis.set_xlabel("g(w)")
    axis.set_ylabel("density")
    axis.set_title("Constraint functional over the retained draws", loc="left")
    axis.legend()
    fig.tight_layout()
    return save_figure(fig, output_dir, name)


def plot_distance_to_boundary(runs: Sequence[AlphaRun], output_dir: str | Path,
                              max_points: int = 4000,
                              name: str = "fig08_distance_to_boundary") -> Dict[str, str]:
    """Traces of the slack ``Lambda_constraint - g(w_k)`` (chain 1 of each alpha)."""
    apply_style()
    colors = alpha_colors(len(runs))
    fig, axis = plt.subplots(figsize=(9.2, 4.2))
    for index, run in enumerate(runs):
        chain = run.chains[0]
        slack = chain.slack
        thin = max(1, slack.size // max_points)
        axis.plot(np.arange(slack.size)[::thin], slack[::thin], linewidth=0.9,
                  color=colors[index], label=f"alpha = {run.alpha:g}")
    axis.axhline(0.0, color=PALETTE["critical"], linewidth=1.2, linestyle="--")
    axis.annotate("boundary", (0, 0), textcoords="offset points", xytext=(4, 4),
                  fontsize=8, color=PALETTE["critical"])
    axis.set_xlabel("iteration")
    axis.set_ylabel("Lambda_constraint - g(w)")
    axis.set_title("Distance to the constraint boundary (chain 1)", loc="left")
    axis.legend(ncol=len(runs))
    fig.tight_layout()
    return save_figure(fig, output_dir, name)


def plot_projection_frequency(summaries: Sequence[Dict[str, object]],
                              output_dir: str | Path,
                              name: str = "fig09_projection_frequency") -> Dict[str, str]:
    """Projection frequency and mean projection distance against alpha."""
    apply_style()
    alphas = [s["alpha"] for s in summaries]
    frequency = [100.0 * s["projection_frequency"] for s in summaries]
    distance = [s["mean_projection_distance"] for s in summaries]
    colors = alpha_colors(len(summaries))
    # Two measures on different scales -> two panels, never a second y-axis.
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    axes[0].bar(range(len(alphas)), frequency, color=colors, width=0.6)
    axes[0].set_xticks(range(len(alphas)))
    axes[0].set_xticklabels([f"{a:g}" for a in alphas])
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("projections (% of retained draws)")
    axes[0].set_title("Projection frequency", loc="left")
    for x, y in zip(range(len(alphas)), frequency):
        axes[0].annotate(f"{y:.2f}%", (x, y), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=8,
                         color=PALETTE["text_secondary"])
    axes[1].bar(range(len(alphas)), distance, color=colors, width=0.6)
    axes[1].set_xticks(range(len(alphas)))
    axes[1].set_xticklabels([f"{a:g}" for a in alphas])
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("mean ||z - y|| when projecting")
    axes[1].set_title("Average projection distance", loc="left")
    fig.tight_layout()
    return save_figure(fig, output_dir, name)


def plot_J_operator_norm(runs: Sequence[AlphaRun], output_dir: str | Path,
                         max_points: int = 4000,
                         name: str = "fig10_J_operator_norm") -> Dict[str, str]:
    """Operator norm of ``J(w_k)`` along the first chain of every alpha."""
    apply_style()
    colors = alpha_colors(len(runs))
    fig, axis = plt.subplots(figsize=(9.2, 4.0))
    for index, run in enumerate(runs):
        values = run.chains[0].J_operator_norm
        thin = max(1, values.size // max_points)
        axis.plot(np.arange(values.size)[::thin], values[::thin], linewidth=0.9,
                  color=colors[index], label=f"alpha = {run.alpha:g}")
    axis.set_xlabel("iteration")
    axis.set_ylabel(r"$\|J(w_k)\|_2$")
    axis.set_title("Operator norm of the non-reversible matrix", loc="left")
    axis.legend(ncol=len(runs))
    fig.tight_layout()
    return save_figure(fig, output_dir, name)


def plot_drift_norms(runs: Sequence[AlphaRun], output_dir: str | Path,
                     max_points: int = 3000,
                     name: str = "fig11_drift_norms") -> Dict[str, str]:
    """Reversible versus non-reversible drift norms for each alpha."""
    apply_style()
    n_runs = len(runs)
    fig, axes = plt.subplots(1, n_runs, figsize=(3.0 * n_runs, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for index, run in enumerate(runs):
        chain = run.chains[0]
        thin = max(1, chain.reversible_drift_norm.size // max_points)
        iterations = np.arange(chain.reversible_drift_norm.size)[::thin]
        axes[index].plot(iterations, chain.reversible_drift_norm[::thin],
                         color=PALETTE["categorical"][0], linewidth=0.8,
                         label="reversible")
        axes[index].plot(iterations, chain.nonreversible_drift_norm[::thin],
                         color=PALETTE["categorical"][1], linewidth=0.8,
                         linestyle="--", label="non-reversible")
        axes[index].set_title(f"alpha = {run.alpha:g}", loc="left")
        axes[index].set_xlabel("iteration")
    axes[0].set_ylabel("drift norm")
    axes[0].legend(loc="upper right")
    fig.suptitle("Reversible versus non-reversible drift magnitude", x=0.01,
                 ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, output_dir, name)


def plot_sensitivity_comparison(summaries: Sequence[Dict[str, object]],
                                parameter_names: Sequence[str],
                                output_dir: str | Path, name: str,
                                title: str, series_label: str) -> Dict[str, str]:
    """Posterior means (with 95% intervals) and ESS across a sensitivity sweep."""
    apply_style()
    colors = alpha_colors(len(summaries))
    base = np.arange(len(parameter_names))
    spread = _group_spread(len(summaries))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    for index, summary in enumerate(summaries):
        offset = (index - (len(summaries) - 1) / 2) * spread
        y = base + offset
        axes[0].hlines(y, summary["credible_low"], summary["credible_high"],
                       color=colors[index], linewidth=2.0)
        axes[0].plot(summary["posterior_mean"], y,
                     marker=_MARKERS[index % len(_MARKERS)], linestyle="none",
                     markersize=6, color=colors[index],
                     markeredgecolor=PALETTE["surface"], markeredgewidth=0.8,
                     label=summary["label"])
    axes[0].set_yticks(base)
    axes[0].set_yticklabels(parameter_names)
    axes[0].invert_yaxis()
    axes[0].axvline(0.0, color=PALETTE["baseline"], linewidth=0.9)
    axes[0].set_xlabel("coefficient value")
    axes[0].set_title("Posterior means and 95% intervals", loc="left")

    width = 0.8 / len(summaries)
    for index, summary in enumerate(summaries):
        offset = (index - (len(summaries) - 1) / 2) * width
        axes[1].bar(base + offset, summary["ess_bulk"], width=width * 0.92,
                    color=colors[index], label=summary["label"])
    axes[1].set_xticks(base)
    axes[1].set_xticklabels(parameter_names, rotation=45, ha="right")
    axes[1].set_ylabel("bulk ESS")
    axes[1].set_title("Effective sample size", loc="left")
    # one figure-level legend below both panels: the same series appear in both,
    # and nothing is then drawn on top of the data
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=series_label, loc="lower center",
               ncol=min(len(summaries), 3), bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return save_figure(fig, output_dir, name)


def make_all_diagnostic_figures(runs: Sequence[AlphaRun],
                                summaries: Sequence[Dict[str, object]],
                                parameter_names: Sequence[str],
                                Lambda_constraint: float, g_min: float,
                                output_dir: str | Path) -> Dict[str, Dict[str, str]]:
    """Produce figures 1-11 of section 17 and return their paths."""
    paths: Dict[str, Dict[str, str]] = {}
    reference = runs[-1] if len(runs) > 1 else runs[0]
    paths["traces"] = plot_traces(reference, parameter_names, output_dir)
    paths["autocorrelation"] = plot_autocorrelation(reference, parameter_names,
                                                    output_dir)
    paths["ess_by_coefficient"] = plot_ess_by_coefficient(summaries, output_dir)
    paths["ess_per_second"] = plot_ess_per_second_vs_alpha(summaries, output_dir)
    paths["rhat"] = plot_rhat_by_coefficient(summaries, output_dir)
    paths["posterior_intervals"] = plot_posterior_intervals(summaries, output_dir)
    paths["g_histogram"] = plot_g_histogram(runs, Lambda_constraint, g_min, output_dir)
    paths["distance_to_boundary"] = plot_distance_to_boundary(runs, output_dir)
    paths["projection_frequency"] = plot_projection_frequency(summaries, output_dir)
    paths["J_operator_norm"] = plot_J_operator_norm(runs, output_dir)
    paths["drift_norms"] = plot_drift_norms(runs, output_dir)
    return paths
