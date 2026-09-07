"""The anchor has to dominate the exact potential and reproduce its gradient."""

import numpy as np
import pytest
import torch

from ands import data as D
from ands import targets as T

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def targets():
    return {k: T.BallConstrainedGibbsSVM(ds, tau=100.0, lam=1.0, delta=0.02, R=1.4)
            for k, ds in [("titanic", D.load_titanic()), ("magic", D.load_magic(400))]}


def _points(tgt, n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tgt.project(torch.randn(n, tgt.d, generator=g))


@pytest.mark.parametrize("key", ["titanic", "magic"])
def test_anchor_dominates(key, targets):
    """``U_0 >= U`` keeps ``e^{Delta} = e^{U - U_0}`` in (0, 1]."""
    tgt = targets[key]
    w = _points(tgt)
    assert bool((tgt.U0(w) >= tgt.U(w) - 1e-12).all())
    assert float(tgt.anchor_gap(w).max()) <= 1e-12


@pytest.mark.parametrize("key", ["titanic", "magic"])
def test_anchor_gap_is_bounded_independently_of_n(key, targets):
    """The hinge term is an average, so the smoothing gap is O(tau*delta), not
    O(n*tau*delta) -- otherwise ``e^{Delta}`` would underflow on real data."""
    tgt = targets[key]
    bound = tgt.tau * tgt.delta / 2 + tgt.lam * tgt.d * tgt.delta
    assert float((-tgt.anchor_gap(_points(tgt))).max()) < bound


@pytest.mark.parametrize("key", ["titanic", "magic"])
def test_grad_U0_matches_autograd(key, targets):
    tgt = targets[key]
    w = _points(tgt, n=8).clone().requires_grad_(True)
    z = 1.0 - w @ tgt.Psi.T
    u0 = (tgt.tau * (0.5 * (z + torch.sqrt(z * z + tgt.delta ** 2))).mean(dim=1)
          + tgt.lam * torch.sqrt(w * w + tgt.delta ** 2).sum(dim=1))
    ga = torch.autograd.grad(u0.sum(), w)[0]
    assert float((ga - tgt.grad_U0(w.detach())).abs().max()) < 1e-10


@pytest.mark.parametrize("key", ["titanic", "magic"])
def test_fused_step_agrees_with_separate_calls(key, targets):
    """The sampler reads Delta and grad U_0 from one pass over the margins."""
    tgt = targets[key]
    w = _points(tgt)
    Delta, grad = tgt.anchored_step(w)
    assert float((Delta - tgt.anchor_gap(w)).abs().max()) < 1e-10
    assert float((grad - tgt.grad_U0(w)).abs().max()) < 1e-12


@pytest.mark.parametrize("key", ["titanic", "magic"])
def test_projection_lands_in_K(key, targets):
    tgt = targets[key]
    g = torch.Generator().manual_seed(1)
    w = tgt.project(torch.randn(500, tgt.d, generator=g) * 3.0)
    assert bool(tgt.inside(w).all())
    assert np.isclose(float(w.norm(dim=1).max()), tgt.R)


@pytest.mark.parametrize("key", ["titanic", "magic"])
def test_accuracy_variants_are_well_formed(key, targets):
    tgt = targets[key]
    w = _points(tgt, n=64, seed=2).numpy()
    per = tgt.accuracy_per_draw(w)
    assert per.shape == (64,)
    assert per.min() >= 0.0 and per.max() <= 1.0
    for a in (tgt.accuracy(w), tgt.accuracy_predictive(w)):
        assert 0.0 <= a <= 1.0


@pytest.mark.parametrize("key", ["titanic", "magic"])
def test_the_three_readings_agree_on_a_single_draw(key, targets):
    """With one draw there is no posterior to average over, so plug-in, majority vote
    and per-draw must coincide."""
    tgt = targets[key]
    w = _points(tgt, n=1, seed=3).numpy()
    assert np.isclose(tgt.accuracy(w), tgt.accuracy_per_draw(w)[0])
    assert np.isclose(tgt.accuracy_predictive(w), tgt.accuracy_per_draw(w)[0])


@pytest.mark.parametrize("key", ["titanic", "magic"])
def test_accuracy_is_blind_to_radial_rescaling(key, targets):
    """Accuracy depends only on the *direction* of w, since sign(w . psi) is invariant
    under w -> c w for c > 0.

    This is the mechanism behind Experiment 5: the dominant distortion an inadmissible
    skew field produces is radial -- mass driven out to the wall -- and a scale-invariant
    statistic cannot see it. Accuracy is therefore not a validity check for a
    constrained sampler.
    """
    tgt = targets[key]
    w = _points(tgt, n=40, seed=4).numpy()
    for c in (0.25, 3.0, 17.0):
        assert np.allclose(tgt.accuracy_per_draw(c * w), tgt.accuracy_per_draw(w))
        assert np.isclose(tgt.accuracy_predictive(c * w), tgt.accuracy_predictive(w))


@pytest.mark.parametrize("key", ["titanic", "magic"])
def test_flipping_every_draw_flips_the_accuracy(key, targets):
    """A sanity anchor that the statistic tracks the labels at all."""
    tgt = targets[key]
    w = _points(tgt, n=40, seed=5).numpy()
    assert np.allclose(tgt.accuracy_per_draw(-w), 1.0 - tgt.accuracy_per_draw(w))
