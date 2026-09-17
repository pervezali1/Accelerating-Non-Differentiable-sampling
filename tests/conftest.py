"""Shared fixtures: a small but structurally faithful problem (d = 11)."""

from __future__ import annotations

import numpy as np
import pytest

from pnral.config import ExperimentConfig
from pnral.data import load_dataset
from pnral.sampler import JSpec
from pnral.target import LogisticTarget

P_CONSTRAINT = 1.5
EPSILON_CONSTRAINT = 0.05
LAMBDA_CONSTRAINT = 4.0


@pytest.fixture(scope="session")
def data():
    """Synthetic surrogate with the MAGIC shape (fast; d = 11 exactly)."""
    return load_dataset("", use_synthetic=True, synthetic_n_rows=2000,
                        synthetic_seed=11)


@pytest.fixture(scope="session")
def target(data):
    return LogisticTarget(data.X_train, data.y_train, lambda_lasso=100.0,
                          sigma_intercept=10.0, delta_anchor=0.01)


@pytest.fixture(scope="session")
def j_spec(target):
    return JSpec.create("disjoint", target.d, P_CONSTRAINT, EPSILON_CONSTRAINT, 1.0)


@pytest.fixture(scope="session")
def probe_points(target):
    rng = np.random.default_rng(2024)
    return [0.6 * rng.standard_normal(target.d) for _ in range(12)]
