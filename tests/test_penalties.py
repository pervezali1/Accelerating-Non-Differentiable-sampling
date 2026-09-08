"""The closed-form smoothing lemma, and the bound on the anchor multiplier.

The lemma replaces a Monte Carlo step with an exact formula, so it is worth pinning
against Monte Carlo.  The bound is worth pinning because the obvious claim about it
(``a <= 1``) is false for two of the three penalties here, and a test is the only thing
that keeps that from quietly coming back.
"""

import math

import numpy as np
import pytest
import torch

from ands import penalties as P

torch.set_default_dtype(torch.float64)

LAM, AA = 1.0, 3.0
ALL = [P.lasso(LAM), P.mcp(LAM, AA), P.scad(LAM, AA)]
CONVEX = [P.lasso(LAM)]
NONCONVEX = [P.mcp(LAM, AA), P.scad(LAM, AA)]


def _ids(pens):
    return [p.name for p in pens]


@pytest.mark.parametrize("pen", ALL, ids=_ids(ALL))
@pytest.mark.parametrize("mu", [0.1, 0.3])
def test_closed_form_matches_monte_carlo(pen, mu):
    """p_0 = E p(x + mu Z), to within Monte Carlo's own error."""
    x = torch.linspace(-4, 4, 33)
    g = torch.Generator().manual_seed(0)
    Z = torch.randn(400_000, generator=g)
    val, _ = pen.smooth(x, mu)
    mc = torch.stack([pen.raw(xi + mu * Z).mean() for xi in x])
    se = float(torch.stack([pen.raw(xi + mu * Z).std() for xi in x]).max() / math.sqrt(len(Z)))
    assert float((val - mc).abs().max()) < 5 * se


@pytest.mark.parametrize("pen", ALL, ids=_ids(ALL))
def test_gradient_matches_finite_differences(pen):
    x = torch.linspace(-4, 4, 81)
    mu, h = 0.3, 1e-6
    _, grad = pen.smooth(x, mu)
    vp, _ = pen.smooth(x + h, mu)
    vm, _ = pen.smooth(x - h, mu)
    assert float((grad - (vp - vm) / (2 * h)).abs().max()) < 1e-7


@pytest.mark.parametrize("pen", ALL, ids=_ids(ALL))
def test_smoothing_converges_to_the_penalty_at_rate_mu(pen):
    """p_0 -> p as mu -> 0, and the rate is O(mu), not O(mu^2).

    The worst point is the kink, where a slope jump of 2*lam costs phi(0)*2*lam*mu; the
    O(mu^2) curvature term is smaller and shows up only away from the kinks.  Asserting
    the rate rather than a bare threshold is what makes this test say something.
    """
    x = torch.linspace(-8, 8, 32_001)
    errs = {}
    for mu in (0.2, 0.1, 0.05, 0.02):
        errs[mu] = float((pen.smooth(x, mu)[0] - pen.raw(x)).abs().max())
    mus = sorted(errs, reverse=True)
    for a, b in zip(mus, mus[1:]):
        assert errs[b] < errs[a]                       # monotone in mu
    slopes = [errs[m] / m for m in mus]
    assert max(slopes) / min(slopes) < 1.1             # error / mu is ~constant
    assert slopes[-1] == pytest.approx(2 * LAM / math.sqrt(2 * math.pi), rel=0.05)


@pytest.mark.parametrize("pen", CONVEX, ids=_ids(CONVEX))
def test_convex_penalty_is_dominated_by_its_anchor(pen):
    """Jensen: for convex p, p_0 >= p, so a = e^{p - p_0} <= 1."""
    x = torch.linspace(-8, 8, 20_001)
    assert pen.is_convex()
    assert float((pen.raw(x) - pen.smooth(x, 0.3)[0]).max()) <= 1e-12
    assert pen.anchor_bound(0.3, dim=3) == 1.0


@pytest.mark.parametrize("pen", NONCONVEX, ids=_ids(NONCONVEX))
def test_nonconvex_penalty_is_not_dominated_by_its_anchor(pen):
    """The correction: MCP and SCAD taper to zero, which needs negative curvature, and
    there the anchor falls BELOW the penalty and a exceeds 1."""
    x = torch.linspace(-8, 8, 20_001)
    assert not pen.is_convex()
    assert float((pen.raw(x) - pen.smooth(x, 0.3)[0]).max()) > 1e-3
    assert pen.anchor_bound(0.3, dim=3) > 1.0


@pytest.mark.parametrize("pen", ALL, ids=_ids(ALL))
@pytest.mark.parametrize("mu", [0.05, 0.1, 0.3, 0.5, 1.0])
def test_anchor_deficit_bound_always_holds(pen, mu):
    """max(p - p_0) <= mu^2 max(0, -c_2^min), for every mu.

    This is the inequality the sampler relies on, and it follows from the heat-equation
    representation: d_xx u >= 2 c_2^min pointwise, so p_0 - p >= c_2^min mu^2.
    """
    x = torch.linspace(-14, 14, 140_001)
    measured = float((pen.raw(x) - pen.smooth(x, mu)[0]).max())
    assert measured <= pen.max_anchor_deficit(mu) + 1e-6


@pytest.mark.parametrize("pen", NONCONVEX, ids=_ids(NONCONVEX))
@pytest.mark.parametrize("mu", [0.05, 0.1, 0.2, 0.3])
def test_anchor_deficit_bound_is_attained_for_small_mu(pen, mu):
    """It is not merely an upper bound: while the kernel fits inside the most concave
    piece the deficit equals mu^2 |c_2^min| to five digits.  (It goes slack for larger
    mu -- at mu = 1 the true deficit is about 0.8 of the bound.)"""
    x = torch.linspace(-14, 14, 140_001)
    measured = float((pen.raw(x) - pen.smooth(x, mu)[0]).max())
    assert measured == pytest.approx(pen.max_anchor_deficit(mu), rel=1e-3)


@pytest.mark.parametrize("pen", ALL, ids=_ids(ALL))
def test_anchor_bound_holds_on_a_random_ensemble(pen):
    """The bound is what the sampler actually relies on: a(x) = e^{g - g_0} over d
    coordinates never exceeds exp(d * mu^2 * max(0, -c2))."""
    d, mu = 3, 0.3
    g = torch.Generator().manual_seed(1)
    x = torch.randn(20_000, d, generator=g) * 2.0
    a = torch.exp(pen.raw(x).sum(1) - pen.smooth(x, mu)[0].sum(1))
    assert float(a.max()) <= pen.anchor_bound(mu, dim=d) + 1e-9
    assert float(a.min()) > 0.0


def test_piece_count_is_validated():
    with pytest.raises(ValueError):
        P.PiecewiseQuadratic([0.0], [(0., 0., 0.)])
