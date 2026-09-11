"""The three real datasets, checked for the invariants the sampler relies on.

The Gibbs posterior assumes labels in ``{-1, +1}``, an exact intercept column, and
standardised features -- if any of those slip, the anchor gap and the radius ``R``
calibrated in the notebooks stop meaning what they say.
"""

import numpy as np
import pytest

from ands import data as D

KEYS = ["titanic", "magic", "breast_cancer"]


@pytest.fixture(scope="module")
def sets():
    return D.load_all()


def test_load_all_returns_the_three_datasets(sets):
    assert sorted(sets) == sorted(KEYS)


@pytest.mark.parametrize("key", KEYS)
def test_labels_are_plus_minus_one(sets, key):
    y = sets[key].y
    assert set(np.unique(y)) == {-1.0, 1.0}
    # Neither class may vanish, or accuracy against a constant predictor is 1.
    assert 0.05 < (y > 0).mean() < 0.95


@pytest.mark.parametrize("key", KEYS)
def test_intercept_is_exactly_one(sets, key):
    ds = sets[key]
    assert ds.feature_names[0] == "intercept"
    assert np.array_equal(ds.Phi[:, 0], np.ones(ds.n))


@pytest.mark.parametrize("key", KEYS)
def test_features_are_standardised(sets, key):
    X = sets[key].Phi[:, 1:]
    assert np.abs(X.mean(axis=0)).max() < 1e-10
    assert np.abs(X.std(axis=0) - 1.0).max() < 1e-10


@pytest.mark.parametrize("key", KEYS)
def test_shapes_and_names_agree(sets, key):
    ds = sets[key]
    assert ds.Phi.shape == (ds.n, ds.d)
    assert len(ds.feature_names) == ds.d
    assert len(ds.y) == ds.n
    assert np.isfinite(ds.Phi).all()


def test_breast_cancer_is_the_wdbc_table(sets):
    ds = sets["breast_cancer"]
    # 569 biopsies, 30 real features plus the intercept, benign is the +1 majority.
    assert (ds.n, ds.d) == (569, 31)
    assert 0.62 < (ds.y > 0).mean() < 0.64


def test_loaders_are_deterministic():
    a, b = D.load_titanic(seed=0), D.load_titanic(seed=0)
    assert np.array_equal(a.Phi, b.Phi) and np.array_equal(a.y, b.y)
