"""Posterior predictive evaluation on the held-out MAGIC test split.

For every retained draw ``w^(m)`` the predictive probabilities are
``p_i^(m) = expit(X_test[i] @ w^(m))`` and the reported quantity is the
*posterior-averaged* probability ``p_bar_i = mean_m p_i^(m)`` -- the Bayesian
predictive mean, not the plug-in probability at a point estimate.

MAGIC is imbalanced (about 65% gamma / 35% hadron), so balanced accuracy,
ROC-AUC, sensitivity, specificity and calibration are reported alongside plain
accuracy and are the metrics to read first.

Model-selection discipline
--------------------------
The test split is used **only** for this final evaluation.  ``alpha``, the
step size and the constraint threshold are chosen from training diagnostics
(and the pilot chain); none of them is tuned against these numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from .diagnostics import PALETTE, alpha_colors, apply_style, save_figure

__all__ = [
    "posterior_mean_probabilities",
    "predictive_metrics",
    "predictive_table",
    "plot_roc_curves",
    "plot_calibration_curves",
    "plot_metric_comparison",
]


def posterior_mean_probabilities(samples: np.ndarray, X: np.ndarray,
                                 chunk_size: int = 500) -> np.ndarray:
    """``p_bar_i = mean_m expit(X[i] @ w^(m))`` accumulated in chunks.

    Chunking keeps the ``(n_test, n_draws)`` probability matrix out of memory:
    with 60k draws and 3.8k test rows the full matrix would be ~1.8 GB.
    """
    samples = np.asarray(samples, dtype=np.float64).reshape(-1, X.shape[1])
    total = samples.shape[0]
    accumulator = np.zeros(X.shape[0], dtype=np.float64)
    for start in range(0, total, chunk_size):
        block = samples[start:start + chunk_size]
        accumulator += expit(X @ block.T).sum(axis=1)
    return accumulator / float(total)


def predictive_metrics(y_true: np.ndarray, probabilities: np.ndarray,
                       threshold: float = 0.5,
                       label: str = "") -> Dict[str, object]:
    """Full predictive report for one set of posterior-averaged probabilities.

    ``sensitivity`` (= recall of the positive gamma class) and ``specificity``
    (= recall of the hadron class) are reported separately because the classes
    are imbalanced.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12,
                            1.0 - 1e-12)
    predicted = (probabilities >= threshold).astype(np.float64)

    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    sensitivity = (true_positive / (true_positive + false_negative)
                   if (true_positive + false_negative) else np.nan)
    specificity = (true_negative / (true_negative + false_positive)
                   if (true_negative + false_positive) else np.nan)

    return {
        "label": label,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "sensitivity_recall_positive": float(sensitivity),
        "specificity_recall_negative": float(specificity),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "n_test": int(y_true.size),
        "positive_fraction": float(y_true.mean()),
        "probabilities": probabilities,
    }


def predictive_table(reports: Sequence[Dict[str, object]]) -> pd.DataFrame:
    """Tidy table of predictive metrics (probability vectors dropped)."""
    rows = [{key: value for key, value in report.items() if key != "probabilities"}
            for report in reports]
    return pd.DataFrame(rows)


def plot_roc_curves(y_true: np.ndarray, reports: Sequence[Dict[str, object]],
                    output_dir: str | Path,
                    name: str = "fig12_roc_curves") -> Dict[str, str]:
    """ROC curves for every alpha, with the chance diagonal for reference."""
    apply_style()
    colors = alpha_colors(len(reports))
    fig, axis = plt.subplots(figsize=(5.8, 5.4))
    axis.plot([0, 1], [0, 1], color=PALETTE["baseline"], linewidth=1.0,
              linestyle=":")
    for index, report in enumerate(reports):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true, report["probabilities"])
        axis.plot(false_positive_rate, true_positive_rate, color=colors[index],
                  linewidth=1.6,
                  label=f"{report['label']} (AUC {report['roc_auc']:.4f})")
    axis.set_xlabel("false positive rate (1 - specificity)")
    axis.set_ylabel("true positive rate (sensitivity)")
    axis.set_title("ROC curves — posterior-averaged probabilities", loc="left")
    axis.legend(loc="lower right")
    fig.tight_layout()
    return save_figure(fig, output_dir, name)


def plot_calibration_curves(y_true: np.ndarray,
                            reports: Sequence[Dict[str, object]],
                            output_dir: str | Path, n_bins: int = 15,
                            name: str = "fig13_calibration") -> Dict[str, str]:
    """Reliability diagrams plus the predictive probability histogram."""
    apply_style()
    colors = alpha_colors(len(reports))
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 7.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1.0]})
    axes[0].plot([0, 1], [0, 1], color=PALETTE["baseline"], linewidth=1.0,
                 linestyle=":")
    for index, report in enumerate(reports):
        observed, predicted = calibration_curve(y_true, report["probabilities"],
                                                n_bins=n_bins, strategy="quantile")
        axes[0].plot(predicted, observed, marker="o", markersize=5,
                     color=colors[index], linewidth=1.5,
                     markeredgecolor=PALETTE["surface"], markeredgewidth=0.7,
                     label=f"{report['label']} (Brier {report['brier_score']:.4f})")
        axes[1].hist(report["probabilities"], bins=40, histtype="step",
                     linewidth=1.2, color=colors[index])
    axes[0].set_ylabel("observed gamma frequency")
    axes[0].set_title("Calibration (quantile bins)", loc="left")
    axes[0].legend(loc="upper left")
    axes[1].set_xlabel("posterior-averaged predicted probability")
    axes[1].set_ylabel("count")
    axes[1].set_title("Predictive probability distribution", loc="left")
    fig.tight_layout()
    return save_figure(fig, output_dir, name)


def plot_metric_comparison(reports: Sequence[Dict[str, object]],
                           output_dir: str | Path,
                           name: str = "fig14_predictive_metrics") -> Dict[str, str]:
    """Small multiples of the headline predictive metrics across alpha.

    One panel per metric (never two scales on one axis); the emphasised
    metrics for imbalanced data come first.
    """
    apply_style()
    metrics = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("sensitivity_recall_positive", "Sensitivity"),
        ("specificity_recall_negative", "Specificity"),
        ("accuracy", "Accuracy"),
        ("f1", "F1"),
        ("log_loss", "Log loss (lower better)"),
        ("brier_score", "Brier score (lower better)"),
    ]
    colors = alpha_colors(len(reports))
    labels = [report["label"] for report in reports]
    positions = np.arange(len(reports))
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.0))
    axes = axes.ravel()
    for axis, (key, title) in zip(axes, metrics):
        values = [float(report[key]) for report in reports]
        axis.bar(positions, values, color=colors, width=0.62)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        axis.set_title(title, loc="left", fontsize=10)
        # Bars encode magnitude by length, so the baseline stays at zero: a
        # zoomed baseline would dramatise differences that are, correctly,
        # negligible -- every alpha targets the same truncated posterior.
        # The exact values are carried by the direct labels instead.
        axis.set_ylim(0.0, max(values) * 1.18)
        for x, y in zip(positions, values):
            axis.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                          xytext=(0, 3), ha="center", fontsize=7.5,
                          color=PALETTE["text_secondary"])
    fig.suptitle("Predictive performance on the held-out test split", x=0.01,
                 ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, output_dir, name)


def confusion_matrix_frame(report: Dict[str, object]) -> pd.DataFrame:
    """The 2x2 confusion matrix of one report as a labelled frame."""
    return pd.DataFrame(
        [[report["true_negative"], report["false_positive"]],
         [report["false_negative"], report["true_positive"]]],
        index=["actual hadron (0)", "actual gamma (1)"],
        columns=["predicted hadron (0)", "predicted gamma (1)"])
