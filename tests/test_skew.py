"""The two conditions the theory rests on, checked numerically for every field."""

import numpy as np
import pytest
import torch

from ands import skew as sk

torch.set_default_dtype(torch.float64)
DIMS = [3, 4, 7, 10, 11]


def _batch(dim, n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("kind", ["none", "const", "axial"])
def test_antisymmetric(dim, kind):
    field = sk.build_skew(kind, dim, 1.7, radius=1.3)
    assert sk.antisymmetry_error(field, _batch(dim)) < 1e-12


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("kind", ["none", "const", "axial"])
def test_closed_form_divergence_matches_autograd(dim, kind):
    field = sk.build_skew(kind, dim, 1.7, radius=1.3)
    x = _batch(dim)
    err = (sk.autograd_divergence(field, x) - field.divergence(x)).abs().max()
    assert float(err) < 1e-10


@pytest.mark.parametrize("dim", DIMS)
def test_axial_is_tangential_everywhere(dim):
    """``J_s(x) x = 0`` for all x, so in particular ``J_s nu = 0`` on the sphere."""
    field = sk.build_skew("axial", dim, 2.5, radius=1.3)
    x = _batch(dim, n=200, seed=3)
    assert float(field.apply(x, x).abs().max()) < 1e-12


@pytest.mark.parametrize("dim", DIMS)
def test_constant_field_violates_the_boundary_condition(dim):
    """The counterexample the comparison is built on: ``J_a nu`` is O(s), not 0."""
    field = sk.build_skew("const", dim, 2.0)
    x = _batch(dim, n=200, seed=4)
    x = x / x.norm(dim=1, keepdim=True)
    assert float(sk.boundary_flux(field, x).mean()) > 0.5


@pytest.mark.parametrize("dim", [4, 7, 10, 11])
def test_axial_correction_is_not_negligible(dim):
    """In d > 3 the correction term is the same order as the field itself -- the
    reason the state-dependent scheme cannot reuse the constant-J update."""
    field = sk.build_skew("axial", dim, 2.0, radius=1.3)
    x = _batch(dim, n=200, seed=5)
    x = 1.3 * x / x.norm(dim=1, keepdim=True)
    assert float(field.divergence(x).norm(dim=1).mean()) > 1.0


def test_axial_divergence_vanishes_in_two_dimensions():
    """``div J_s = -s(d-2) A x / R^2`` predicts an exactly reversible field at d = 2."""
    field = sk.build_skew("axial", 2, 3.0)
    x = _batch(2, n=50, seed=6)
    assert float(field.divergence(x).abs().max()) == 0.0


@pytest.mark.parametrize("dim", DIMS)
def test_radius_normalisation(dim):
    """``s`` means the same push at the wall whatever the radius: the 1/R^2 factor
    exactly cancels the r^2 growth of the field."""
    norms = []
    for R in (1.0, 1.4, 1.9):
        field = sk.build_skew("axial", dim, 2.0, radius=R)
        x = _batch(dim, n=60, seed=7)
        x = R * x / x.norm(dim=1, keepdim=True)
        norms.append(sk.mean_operator_norm(field, x))
    assert np.allclose(norms, norms[0], rtol=1e-10)


@pytest.mark.parametrize("dim", [7, 10, 11])
def test_axial_and_constant_push_comparably_hard(dim):
    """The head-to-head is only meaningful if the two fields have similar magnitude on
    the wall at the same s."""
    x = _batch(dim, n=100, seed=8)
    x = 1.4 * x / x.norm(dim=1, keepdim=True)
    axial = sk.mean_operator_norm(sk.build_skew("axial", dim, 2.0, radius=1.4), x)
    const = sk.mean_operator_norm(sk.build_skew("const", dim, 2.0), x)
    assert 0.7 < axial / const < 1.4
