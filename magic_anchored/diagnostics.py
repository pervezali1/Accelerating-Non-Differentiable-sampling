r"""Convergence and efficiency diagnostics, and posterior predictive scoring.

ESS and split-:math:`\hat R` come from ArviZ, which implements the rank-normalised
split-:math:`\hat R` and the Geyer initial-positive-sequence ESS of Vehtari et al.
(2021); a small self-contained fallback is used only if ArviZ is unavailable, and
:data:`HAS_ARVIZ` says which was used.

Two conventions worth stating because they change the numbers:

* the integrated autocorrelation time is reported as ``tau = M / ESS`` with
  ``M`` the number of *retained* samples per chain, so ``tau`` is in units of
  retained draws (a chain thinned by 10 has a ``tau`` ten times smaller than the
  same chain unthinned);
* ESS per second divides the *total* ESS pooled over chains by the *total*
  sampling time of those chains, so it is a throughput and is unaffected by how
  the work was split.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from sampler import SamplerResult
from target import sigmoid

try:  # pragma: no cover - exercised by whichever branch the environment has
    import arviz as az

    HAS_ARVIZ = True
except Exception:  # pragma: no cover
    az = None
    HAS_ARVIZ = False

__all__ = [
    "HAS_ARVIZ",
    "ChainDiagnostics",
    "PredictiveScores",
    "stack_chains",
    "effective_sample_size",
    "split_r_hat",
    "calculate_chain_diagnostics",
    "posterior_predictive_probabilities",
    "evaluate_predictive_performance",
    "calibration_curve_points",
]


# --------------------------------------------------------------- ESS and R-hat
def stack_chains(results: Sequence[SamplerResult]) -> np.ndarray:
    """``(n_chains, n_draws, D)`` from a list of chains, truncated to the shortest."""
    if not results:
        raise ValueError("no chains given")
    n_draws = min(r.samples.shape[0] for r in results)
    if n_draws == 0:
        raise ValueError("at least one chain retained no samples")
    return np.stack([r.samples[:n_draws] for r in results], axis=0)


def _ess_single(x: np.ndarray) -> float:
    r"""Geyer initial-positive-sequence ESS for a ``(n_chains, n_draws)`` array.

    Used only when ArviZ is missing.  Autocovariances are averaged over chains
    and summed over increasing lags while consecutive *pairs* remain positive,
    which is the standard truncation that keeps the estimate from being spoiled
    by the noisy tail of the correlogram.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    n_chains, n = x.shape
    if n < 4:
        return float(n_chains * n)
    means = x.mean(axis=1, keepdims=True)
    centred = x - means
    n_fft = int(2 ** np.ceil(np.log2(2 * n)))
    acov = np.zeros(n)
    for c in range(n_chains):
        f = np.fft.rfft(centred[c], n=n_fft)
        ac = np.fft.irfft(f * np.conjugate(f), n=n_fft)[:n].real
        acov += ac / n
    acov /= n_chains
    var_plus = acov[0]
    if var_plus <= 0:
        return float(n_chains * n)
    rho = acov / var_plus
    # between-chain variance correction, as in the split-R-hat construction
    if n_chains > 1:
        between = means.ravel().var(ddof=1)
        var_plus = acov[0] * (n - 1) / n + between
        rho = (acov[0] - acov) / var_plus
        rho = 1.0 - rho
    total = 0.0
    t = 1
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        total += pair
        t += 2
    tau = 1.0 + 2.0 * total
    return float(n_chains * n / max(tau, 1e-12))


def effective_sample_size(draws: np.ndarray) -> np.ndarray:
    """Per-coefficient ESS from a ``(n_chains, n_draws, D)`` array."""
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 3:
        raise ValueError(f"expected (chains, draws, D), got {draws.shape}")
    if HAS_ARVIZ:
        idata = az.convert_to_dataset({"w": draws})
        return np.asarray(az.ess(idata).w.values, dtype=np.float64)
    return np.array([_ess_single(draws[:, :, j]) for j in range(draws.shape[2])])


def split_r_hat(draws: np.ndarray) -> np.ndarray:
    """Per-coefficient split-:math:`\\hat R` from a ``(n_chains, n_draws, D)`` array."""
    draws = np.asarray(draws, dtype=np.float64)
    if HAS_ARVIZ:
        idata = az.convert_to_dataset({"w": draws})
        return np.asarray(az.rhat(idata).w.values, dtype=np.float64)
    # fallback: split each chain in half, then the usual between/within statistic
    n_chains, n, D = draws.shape
    half = n // 2
    if half < 2:
        return np.full(D, np.nan)
    split = np.concatenate([draws[:, :half], draws[:, half : 2 * half]], axis=0)
    m, n_s, _ = split.shape
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    W = chain_vars.mean(axis=0)
    B = chain_means.var(axis=0, ddof=1)
    var_plus = (n_s - 1) / n_s * W + B
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(var_plus / W)


@dataclass
class ChainDiagnostics:
    """Pooled diagnostics for the group of chains run at one ``alpha``."""

    alpha: float
    h: float
    radius: float
    n_chains: int
    n_draws_per_chain: int
    param_names: tuple[str, ...]

    ess: np.ndarray  # (D,)
    r_hat: np.ndarray  # (D,)
    iat: np.ndarray  # (D,) retained draws per effective draw
    min_ess: float
    median_ess: float
    max_r_hat: float
    ess_per_second: float
    total_runtime_seconds: float
    total_gradient_evaluations: int

    projection_frequency: float
    mean_squared_jump: float
    a_mean: float
    a_min: float
    log_a_min: float
    radius_mean: float
    radius_max: float
    reversible_drift_norm: float
    nonreversible_drift_norm: float
    n_numerical_warnings: int
    warnings_raised: tuple[str, ...] = ()

    def as_row(self) -> dict[str, float | str]:
        row: dict[str, float | str] = {
            "alpha": self.alpha,
            "h": self.h,
            "radius": self.radius,
            "n_chains": self.n_chains,
            "n_draws_per_chain": self.n_draws_per_chain,
            "min_ess": self.min_ess,
            "median_ess": self.median_ess,
            "max_r_hat": self.max_r_hat,
            "ess_per_second": self.ess_per_second,
            "total_runtime_seconds": self.total_runtime_seconds,
            "total_gradient_evaluations": self.total_gradient_evaluations,
            "projection_frequency": self.projection_frequency,
            "mean_squared_jump": self.mean_squared_jump,
            "a_mean": self.a_mean,
            "a_min": self.a_min,
            "log_a_min": self.log_a_min,
            "radius_mean": self.radius_mean,
            "radius_max": self.radius_max,
            "reversible_drift_norm": self.reversible_drift_norm,
            "nonreversible_drift_norm": self.nonreversible_drift_norm,
            "n_numerical_warnings": self.n_numerical_warnings,
            "n_warnings": len(self.warnings_raised),
            "warnings": " | ".join(self.warnings_raised),
        }
        for name, e, r, t in zip(self.param_names, self.ess, self.r_hat, self.iat):
            row[f"ess[{name}]"] = float(e)
            row[f"rhat[{name}]"] = float(r)
            row[f"iat[{name}]"] = float(t)
        return row


def calculate_chain_diagnostics(
    results: Sequence[SamplerResult],
    param_names: Sequence[str],
    r_hat_threshold: float = 1.01,
    min_ess_threshold: float = 100.0,
    projection_threshold: float = 0.10,
) -> ChainDiagnostics:
    """Pool a group of chains and raise the warnings the specification asks for.

    The thresholds are the ones named there: a projection frequency above 10%
    means the constraint is shaping the answer rather than regularising it, an
    :math:`\\hat R` above 1.01 means the chains disagree, and a very low ESS
    means the chains are not usable however well they agree.
    """
    if not results:
        raise ValueError("no chains given")
    draws = stack_chains(results)
    n_chains, n_draws, D = draws.shape
    names = tuple(param_names)
    if len(names) != D:
        raise ValueError(f"{len(names)} names for {D} coefficients")

    ess = effective_sample_size(draws)
    r_hat = split_r_hat(draws)
    with np.errstate(divide="ignore", invalid="ignore"):
        iat = np.where(ess > 0, n_draws / ess, np.inf)

    total_runtime = float(sum(r.runtime_seconds for r in results))
    total_grad = int(sum(r.n_grad_evaluations for r in results))
    min_ess = float(np.nanmin(ess))
    max_r_hat = float(np.nanmax(r_hat))

    messages: list[str] = []
    projection_frequency = float(np.mean([r.projection_frequency for r in results]))
    if projection_frequency > projection_threshold:
        messages.append(
            f"projection frequency {projection_frequency:.1%} exceeds "
            f"{projection_threshold:.0%}: the constraint is active often enough to "
            "shape the posterior, so the radius should be enlarged"
        )
    if np.isfinite(max_r_hat) and max_r_hat > r_hat_threshold:
        worst = names[int(np.nanargmax(r_hat))]
        messages.append(
            f"max split R-hat {max_r_hat:.4f} > {r_hat_threshold} "
            f"(worst coefficient {worst!r}): the chains have not mixed"
        )
    if min_ess < min_ess_threshold:
        worst = names[int(np.nanargmin(ess))]
        messages.append(
            f"minimum ESS {min_ess:.1f} < {min_ess_threshold:.0f} "
            f"(coefficient {worst!r}): too few effective draws to trust summaries"
        )
    radii = np.concatenate([r.radius_trace for r in results])
    if not np.isfinite(radii).all():
        messages.append("parameter norms became non-finite")
    else:
        tail = np.concatenate(
            [r.radius_trace[r.burn_in :] for r in results]
        )
        first, second = np.array_split(tail, 2)
        if second.mean() > 1.5 * max(first.mean(), 1e-12):
            messages.append(
                f"the distance from the centre is still growing "
                f"({first.mean():.4f} -> {second.mean():.4f} across the retained "
                "half): the chain has not stabilised"
            )
    n_warn = int(sum(r.n_numerical_warnings for r in results))
    for message in messages:
        warnings.warn(message, RuntimeWarning)

    return ChainDiagnostics(
        alpha=float(results[0].alpha),
        h=float(results[0].h),
        radius=float(results[0].radius),
        n_chains=n_chains,
        n_draws_per_chain=n_draws,
        param_names=names,
        ess=ess,
        r_hat=r_hat,
        iat=iat,
        min_ess=min_ess,
        median_ess=float(np.nanmedian(ess)),
        max_r_hat=max_r_hat,
        ess_per_second=float(np.nansum(ess) / total_runtime) if total_runtime else float("nan"),
        total_runtime_seconds=total_runtime,
        total_gradient_evaluations=total_grad,
        projection_frequency=projection_frequency,
        mean_squared_jump=float(np.mean([r.mean_squared_jump for r in results])),
        a_mean=float(np.mean([r.a.mean() for r in results])),
        a_min=float(np.min([r.a.min() for r in results])),
        log_a_min=float(np.min([r.log_a.min() for r in results])),
        radius_mean=float(np.mean([r.radius_trace.mean() for r in results])),
        radius_max=float(np.max([r.radius_trace.max() for r in results])),
        reversible_drift_norm=float(
            np.mean([r.reversible_drift_norm.mean() for r in results])
        ),
        nonreversible_drift_norm=float(
            np.mean([r.nonreversible_drift_norm.mean() for r in results])
        ),
        n_numerical_warnings=n_warn,
        warnings_raised=tuple(messages),
    )


# --------------------------------------------------------- predictive scoring
def posterior_predictive_probabilities(
    samples: np.ndarray, X: np.ndarray, batch_size: int = 512
) -> np.ndarray:
    r"""``p_bar_i = mean_m sigmoid(x_i' w^{(m)})``, the posterior-averaged probability.

    This is the Bayesian predictive, an average of probabilities -- *not* the
    probability at the average parameter, which is a different and worse
    quantity because ``sigmoid`` is not affine.  Accumulated in batches of
    samples so memory stays ``O(n)`` rather than ``O(n M)``.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    X = np.asarray(X, dtype=np.float64)
    if samples.shape[1] != X.shape[1]:
        raise ValueError(
            f"samples have dimension {samples.shape[1]} but X has {X.shape[1]} columns"
        )
    total = np.zeros(X.shape[0], dtype=np.float64)
    n_samples = samples.shape[0]
    for start in range(0, n_samples, batch_size):
        block = samples[start : start + batch_size]
        total += sigmoid(X @ block.T).sum(axis=1)
    return total / n_samples


def calibration_curve_points(
    y_true: np.ndarray, p_bar: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-width reliability curve: ``(mean predicted, observed rate, count)``."""
    y_true = np.asarray(y_true, dtype=np.float64)
    p_bar = np.asarray(p_bar, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p_bar, edges[1:-1], right=False), 0, n_bins - 1)
    pred, obs, count = [], [], []
    for b in range(n_bins):
        mask = idx == b
        count.append(int(mask.sum()))
        pred.append(float(p_bar[mask].mean()) if mask.any() else np.nan)
        obs.append(float(y_true[mask].mean()) if mask.any() else np.nan)
    return np.array(pred), np.array(obs), np.array(count)


@dataclass
class PredictiveScores:
    """Test-set performance of one posterior."""

    alpha: float
    n_samples: int
    accuracy: float
    balanced_accuracy: float
    roc_auc: float
    log_loss: float
    brier_score: float
    sensitivity: float
    specificity: float
    confusion: np.ndarray  # 2x2, rows true, columns predicted
    calibration_predicted: np.ndarray
    calibration_observed: np.ndarray
    calibration_counts: np.ndarray
    threshold: float = 0.5

    def as_row(self) -> dict[str, float]:
        tn, fp, fn, tp = self.confusion.ravel()
        return {
            "alpha": self.alpha,
            "n_samples": self.n_samples,
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "roc_auc": self.roc_auc,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
        }


def evaluate_predictive_performance(
    y_true: np.ndarray,
    p_bar: np.ndarray,
    alpha: float = float("nan"),
    n_samples: int = 0,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> PredictiveScores:
    """Accuracy, balanced accuracy, ROC-AUC, log loss, Brier, and calibration.

    ROC-AUC and the Brier and log scores are threshold-free and are the ones to
    compare across ``alpha``; accuracy at 0.5 is reported because it was asked
    for, but a difference in it of a few tenths of a percent on 3804 test points
    is inside binomial noise and should not be read as a ranking.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    p_bar = np.clip(np.asarray(p_bar, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y_pred = (p_bar >= threshold).astype(np.float64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return PredictiveScores(
        alpha=float(alpha),
        n_samples=int(n_samples),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        roc_auc=float(roc_auc_score(y_true, p_bar)),
        log_loss=float(log_loss(y_true, p_bar, labels=[0, 1])),
        brier_score=float(brier_score_loss(y_true, p_bar)),
        sensitivity=float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        specificity=float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        confusion=np.asarray(cm, dtype=np.int64),
        calibration_predicted=calibration_curve_points(y_true, p_bar, n_bins)[0],
        calibration_observed=calibration_curve_points(y_true, p_bar, n_bins)[1],
        calibration_counts=calibration_curve_points(y_true, p_bar, n_bins)[2],
        threshold=threshold,
    )
