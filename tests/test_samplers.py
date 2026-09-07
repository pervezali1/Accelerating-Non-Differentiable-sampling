"""End-to-end behaviour on a small, cheap problem."""

import numpy as np
import pytest
import torch

from ands import data as D
from ands import diagnostics as G
from ands import samplers as S
from ands import targets as T

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def tiny():
    ds = D.load_magic(300)
    return T.BallConstrainedGibbsSVM(ds, tau=100.0, lam=1.0, delta=0.02, R=1.9)


def test_samples_stay_in_K(tiny):
    for kind, s in [("none", 0.0), ("const", 4.0), ("axial", 4.0)]:
        x, _ = S.run_anchored_langevin(tiny, kind, s, eta=3e-4, n_steps=60, N=200, seed=0)
        assert np.linalg.norm(x, axis=1).max() <= tiny.R * (1 + 1e-9)


def test_rwm_reference_stays_in_K_and_mixes(tiny):
    x, diag = S.run_rwm_reference(tiny, n_chains=200, n_steps=600, seed=0)
    assert np.linalg.norm(x, axis=1).max() <= tiny.R * (1 + 1e-9)
    assert 0.05 < diag["acc_rate"] < 0.95


def test_zero_strength_reproduces_the_reversible_scheme(tiny):
    """s = 0 must give bit-identical output for all three fields."""
    base, _ = S.run_anchored_langevin(tiny, "none", 0.0, eta=3e-4, n_steps=40, N=150, seed=2)
    for kind in ("const", "axial"):
        x, _ = S.run_anchored_langevin(tiny, kind, 0.0, eta=3e-4, n_steps=40, N=150, seed=2)
        assert np.allclose(base, x)


def test_dropping_the_correction_changes_the_trajectory(tiny):
    """If the ablation were a no-op the d > 3 experiment would be vacuous."""
    a, _ = S.run_anchored_langevin(tiny, "axial", 4.0, eta=3e-4, n_steps=40, N=150, seed=3)
    b, _ = S.run_anchored_langevin(tiny, "axial", 4.0, eta=3e-4, n_steps=40, N=150, seed=3,
                                   drop_correction=True)
    assert not np.allclose(a, b)


def test_diagnostics_are_zero_against_self():
    x = np.random.default_rng(0).normal(size=(500, 4))
    assert G.w1_per_coord(x, x).max() == 0.0
    assert G.max_ks(x, x) == 0.0
