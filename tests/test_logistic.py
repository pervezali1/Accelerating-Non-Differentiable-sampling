"""The unconstrained NALD target and the block-diagonal skew field.

Without a wall the only thing keeping the dynamics honest is that ``J`` is skew-symmetric
with ``div J = 0``.  Both are asserted here against autograd, not against the closed forms
they are supposed to verify.  The kernel dimensions are asserted too: they are what
separates the two fields once the boundary condition stops mattering.
"""

import numpy as np
import pytest
import torch

from ands import data as D
from ands import logistic as L
from ands import skew as S

torch.set_default_dtype(torch.float64)
DIM = 9


@pytest.fixture(scope="module")
def target():
    tr, _ = D.train_test_split(D.load_titanic_d9(), test_frac=0.2, seed=0)
    return L.UnconstrainedLogisticTarget(tr, tau=300.0, lam=0.5, kappa=0.5, mu=0.05)


@pytest.fixture(scope="module")
def pts():
    return L.uniform_ball(64, DIM, 2.0, generator=torch.Generator().manual_seed(0))


FIELDS = [("const", S.ConstantSkew(DIM, 2.0, normalise=False)),
          ("block", S.BlockCrossSkew(DIM, [5.0, 5.0, 5.0]))]


@pytest.mark.parametrize("name,f", FIELDS)
def test_field_is_skew_symmetric(name, f, pts):
    M = torch.stack([f.matrix(pts[i]) for i in range(len(pts))])
    assert (M + M.transpose(1, 2)).abs().max() < 1e-12


@pytest.mark.parametrize("name,f", FIELDS)
def test_field_is_divergence_free_by_autograd(name, f, pts):
    # admissibility for the unconstrained dynamics rests on exactly this
    assert S.autograd_divergence(f, pts).abs().max() < 1e-9
    assert f.divergence(pts).abs().max() < 1e-12


def test_block_field_annihilates_x_and_const_does_not(pts):
    assert S.BlockCrossSkew(DIM, 3.0).apply(pts, pts).abs().max() < 1e-12
    assert S.ConstantSkew(DIM, 3.0, normalise=False).apply(pts, pts).abs().max() > 1e-3


def test_kernel_dimensions(pts):
    # a real skew matrix in odd dimension is singular, so J_a loses one direction;
    # the block field loses one per block
    for f, want in ((S.ConstantSkew(DIM, 2.0, normalise=False), 1),
                    (S.BlockCrossSkew(DIM, [1.0, 2.0, 3.0]), 3)):
        for i in range(8):
            sv = torch.linalg.svdvals(f.matrix(pts[i]))
            assert int((sv < 1e-10 * sv.max()).sum()) == want


def test_constant_field_matches_the_paper_matrix():
    J = S.ConstantSkew(DIM, 2.0, normalise=False).J
    assert torch.allclose(torch.diagonal(J, 1), torch.full((DIM - 1,), 2.0))
    assert torch.allclose(torch.diagonal(J, -1), torch.full((DIM - 1,), -2.0))
    assert torch.allclose(torch.diagonal(J, 0), torch.zeros(DIM))


def test_block_field_rejects_bad_shapes():
    with pytest.raises(ValueError):
        S.BlockCrossSkew(10, 1.0)
    with pytest.raises(ValueError):
        S.BlockCrossSkew(9, [1.0, 2.0])


def test_gradient_matches_autograd(target, pts):
    w = pts.clone().requires_grad_(True)
    target.U0(w).sum().backward()
    _, g = target.anchored(pts)
    assert (g - w.grad).abs().max() < 1e-10


def test_anchor_gap_is_non_positive_for_a_convex_penalty(target, pts):
    # lasso is convex, so U <= U_0 pointwise and a = e^{U - U_0} lands in (0, 1]
    d = target.delta(pts)
    assert d.max() <= 1e-12
    assert float(torch.exp(d).min()) > 0.5


def test_anchor_gap_does_not_scale_with_n():
    # only the penalty is anchored, so Delta carries no data and cannot underflow
    big, _ = D.train_test_split(D.load_magic_d9(n_max=4000), test_frac=0.2, seed=0)
    small, _ = D.train_test_split(D.load_titanic_d9(), test_frac=0.2, seed=0)
    w = L.uniform_ball(32, DIM, 2.0, generator=torch.Generator().manual_seed(3))
    a = L.UnconstrainedLogisticTarget(big, lam=0.5, mu=0.05).delta(w)
    b = L.UnconstrainedLogisticTarget(small, lam=0.5, mu=0.05).delta(w)
    assert (a - b).abs().max() < 1e-12


def test_minibatch_gradient_is_unbiased(target, pts):
    full = target.anchored(pts)[1]
    gen = torch.Generator().manual_seed(0)
    acc = torch.zeros_like(full)
    reps = 400
    for _ in range(reps):
        idx = torch.randint(0, target.n, (64,), generator=gen)
        acc += target.anchored(pts, idx)[1]
    assert (acc / reps - full).abs().max() < 0.05 * full.abs().max()


def test_uniform_ball_start_is_inside_and_at_chance(target):
    w = L.uniform_ball(2000, DIM, 1.0, generator=torch.Generator().manual_seed(1))
    assert float(w.norm(dim=1).max()) <= 1.0 + 1e-12
    assert abs(float(target.accuracy(w).mean()) - 0.5) < 0.05


def test_zero_skew_run_reproduces_and_improves_accuracy(target):
    kw = dict(eta=1e-4, n_steps=300, n_walkers=40, batch=30, track_every=50,
              eval_sets=(target.Psi,), seed=0)
    it, (a1,), w1 = L.run_nald(target, S.ZeroSkew(DIM), **kw)
    _, (a2,), w2 = L.run_nald(target, S.ZeroSkew(DIM), **kw)
    assert np.array_equal(a1, a2) and torch.equal(w1, w2)
    assert a1[-1].mean() > a1[0].mean() + 0.1


def test_train_test_split_is_disjoint_and_stratified():
    ds = D.load_titanic_d9()
    tr, te = D.train_test_split(ds, test_frac=0.2, seed=0)
    assert tr.n + te.n == ds.n
    assert abs((tr.y > 0).mean() - (te.y > 0).mean()) < 0.03
    # standardisation is fitted on train only
    assert np.abs(tr.Phi.mean(axis=0)).max() < 1e-10
