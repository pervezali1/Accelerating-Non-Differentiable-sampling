"""Synthetic Bayesian logistic-regression data set.

Design
------
* ``n_train = 2000`` training rows, ``n_test = 1000`` test rows.
* ``8`` predictors drawn from ``Z_i ~ N(0, Sigma_X)`` with an AR(1) correlation
  ``Sigma_X[j, k] = rho ** |j - k|``, ``rho = 0.5``.
* Predictors are standardised using **training-set** means and standard
  deviations (the test set is transformed with the same statistics, never with
  its own).
* An intercept column of ones is appended **after** standardisation, giving
  ``d = 9``.
* ``beta_raw[0]`` is the intercept; the remaining eight entries are a sparse
  slope vector.  ``beta_true`` is ``beta_raw`` shrunk (if necessary) so that it
  lies strictly inside ``K`` with substantial slack -- see
  :func:`constraint.rescale_into_constraint`.
* Responses: ``probability_i = sigmoid(X_i @ beta_true)``,
  ``y_i ~ Bernoulli(probability_i)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from config import ConstraintConfig, DataConfig
from constraint import g_value, rescale_into_constraint


@dataclass
class LogisticDataset:
    """Container for the generated data and the ground truth."""

    X_train: np.ndarray          # (n_train, d), last column is the intercept
    y_train: np.ndarray          # (n_train,) in {0, 1}
    X_test: np.ndarray           # (n_test, d)
    y_test: np.ndarray           # (n_test,)
    beta_true: np.ndarray        # (d,), beta_true[0] is the intercept
    beta_raw: np.ndarray         # (d,), before any rescaling
    shrink_factor: float         # t with beta_true = t * beta_raw
    feature_means: np.ndarray    # (n_features,) training means
    feature_stds: np.ndarray     # (n_features,) training standard deviations
    rho: float                   # AR(1) correlation used for Sigma_X
    seed: int

    @property
    def d(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_train(self) -> int:
        return self.X_train.shape[0]

    @property
    def n_test(self) -> int:
        return self.X_test.shape[0]


def ar1_covariance(n_features: int, rho: float) -> np.ndarray:
    """``Sigma_X[j, k] = rho ** |j - k|``."""
    index = np.arange(n_features)
    return rho ** np.abs(index[:, None] - index[None, :])


def generate_dataset(
    data_cfg: DataConfig | None = None,
    constraint_cfg: ConstraintConfig | None = None,
) -> LogisticDataset:
    """Generate the full synthetic logistic-regression problem.

    The intercept is placed in **column 0** so that ``w[0]`` is the intercept
    everywhere in the project, matching the specification's
    ``w[0]^2 / (2 sigma^2)`` prior and the ``j >= 1`` L1 penalty.
    """
    if data_cfg is None:
        data_cfg = DataConfig()
    if constraint_cfg is None:
        constraint_cfg = ConstraintConfig()

    rng = np.random.default_rng(data_cfg.seed)
    n_total = data_cfg.n_train + data_cfg.n_test

    Sigma_X = ar1_covariance(data_cfg.n_features, data_cfg.rho)
    chol = np.linalg.cholesky(Sigma_X)
    Z = rng.standard_normal((n_total, data_cfg.n_features)) @ chol.T

    Z_train_raw = Z[: data_cfg.n_train]
    Z_test_raw = Z[data_cfg.n_train :]

    # Standardise with training-set statistics only.
    feature_means = Z_train_raw.mean(axis=0)
    feature_stds = Z_train_raw.std(axis=0, ddof=0)
    Z_train = (Z_train_raw - feature_means) / feature_stds
    Z_test = (Z_test_raw - feature_means) / feature_stds

    # Intercept column appended AFTER standardisation, in position 0.
    ones_train = np.ones((data_cfg.n_train, 1))
    ones_test = np.ones((data_cfg.n_test, 1))
    X_train = np.hstack([ones_train, Z_train])
    X_test = np.hstack([ones_test, Z_test])

    beta_raw = np.asarray(data_cfg.beta_raw, dtype=float)
    beta_true, shrink_factor = rescale_into_constraint(beta_raw, constraint_cfg)

    probability_train = expit(X_train @ beta_true)
    probability_test = expit(X_test @ beta_true)
    y_train = rng.binomial(1, probability_train).astype(float)
    y_test = rng.binomial(1, probability_test).astype(float)

    return LogisticDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        beta_true=beta_true,
        beta_raw=beta_raw,
        shrink_factor=shrink_factor,
        feature_means=feature_means,
        feature_stds=feature_stds,
        rho=data_cfg.rho,
        seed=data_cfg.seed,
    )


def describe_dataset(
    dataset: LogisticDataset,
    constraint_cfg: ConstraintConfig,
) -> dict[str, object]:
    """Summary statistics used in the report and the correctness tests."""
    p, eps = constraint_cfg.p_constraint, constraint_cfg.epsilon_constraint
    g_true = float(g_value(dataset.beta_true, p, eps))
    return {
        "n_train": dataset.n_train,
        "n_test": dataset.n_test,
        "n_features": dataset.d - 1,
        "d": dataset.d,
        "rho": dataset.rho,
        "seed": dataset.seed,
        "train_positive_rate": float(dataset.y_train.mean()),
        "test_positive_rate": float(dataset.y_test.mean()),
        "beta_true": dataset.beta_true.tolist(),
        "shrink_factor": dataset.shrink_factor,
        "g_beta_raw": float(g_value(dataset.beta_raw, p, eps)),
        "g_beta_true": g_true,
        "g_at_origin": constraint_cfg.g_at_origin,
        "interior_level": constraint_cfg.interior_level,
        "Lambda_constraint": constraint_cfg.Lambda_constraint,
        "slack_at_beta_true": constraint_cfg.Lambda_constraint - g_true,
    }
