"""Sections 7-9: skew symmetry, divergence freeness, boundary tangency."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from pnral.constraint import (g_constraint, grad_psi_constraint,
                              outward_normal)
from pnral.nonreversible_matrix import (DISJOINT_TRIPLES_D11,
                                        OVERLAPPING_TRIPLES_D11,
                                        construct_J_disjoint,
                                        construct_J_overlapping,
                                        cross_product_block, disjoint_triples,
                                        finite_difference_divergence,
                                        make_J_builder, operator_norm_J,
                                        overlapping_triples)
from tests.conftest import EPSILON_CONSTRAINT, LAMBDA_CONSTRAINT, P_CONSTRAINT


def test_triples_match_the_specification_for_d_eleven():
    assert disjoint_triples(11) == DISJOINT_TRIPLES_D11 == [(0, 1, 2), (3, 4, 5),
                                                            (6, 7, 8)]
    assert overlapping_triples(11) == OVERLAPPING_TRIPLES_D11 == [
        (0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 8), (8, 9, 10), (10, 0, 1)]


def test_cross_product_block_is_the_hat_matrix():
    k = np.array([0.3, -1.2, 2.5])
    block = cross_product_block(*k)
    assert np.allclose(block, np.array([[0, -k[2], k[1]],
                                        [k[2], 0, -k[0]],
                                        [-k[1], k[0], 0]]))
    # [k]_x u = k x u for every u, and [k]_x k = 0
    rng = np.random.default_rng(1)
    for _ in range(5):
        u = rng.standard_normal(3)
        assert np.allclose(block @ u, np.cross(k, u))
    assert np.allclose(block @ k, 0.0)


@pytest.mark.parametrize("mode", ["disjoint", "overlapping"])
def test_J_is_skew_symmetric(mode, probe_points):
    """Automatic check 4."""
    builder, _, _ = make_J_builder(mode, 11, P_CONSTRAINT, EPSILON_CONSTRAINT, 1.0)
    for w in probe_points:
        J = builder(w)
        assert np.max(np.abs(J + J.T)) <= 1e-14
        assert np.allclose(np.diag(J), 0.0)


@pytest.mark.parametrize("mode", ["disjoint", "overlapping"])
def test_J_is_divergence_free(mode, probe_points):
    """Automatic check 5: centred finite differences of sum_i d_i J_ij."""
    builder, _, _ = make_J_builder(mode, 11, P_CONSTRAINT, EPSILON_CONSTRAINT, 1.0)
    for w in probe_points[:5]:
        divergence = finite_difference_divergence(builder, w, step=1e-5)
        assert np.max(np.abs(divergence)) <= 1e-9


@pytest.mark.parametrize("mode", ["disjoint", "overlapping"])
def test_J_is_tangent_to_the_boundary(mode):
    """Automatic check 6: J(w) n(w) = 0 whenever g(w) = Lambda_constraint."""
    builder, _, _ = make_J_builder(mode, 11, P_CONSTRAINT, EPSILON_CONSTRAINT, 1.0)
    rng = np.random.default_rng(7)
    for _ in range(10):
        direction = rng.standard_normal(11)
        scale = brentq(lambda t: g_constraint(t * direction, P_CONSTRAINT,
                                              EPSILON_CONSTRAINT)
                       - LAMBDA_CONSTRAINT, 1e-8, 1e4)
        w = scale * direction
        assert np.isclose(g_constraint(w, P_CONSTRAINT, EPSILON_CONSTRAINT),
                          LAMBDA_CONSTRAINT)
        normal = outward_normal(w, P_CONSTRAINT, EPSILON_CONSTRAINT)
        J = builder(w)
        assert np.max(np.abs(J @ normal)) <= 1e-12
        # hence n^T J grad_U0 = -(J n)^T grad_U0 = 0 for any gradient
        arbitrary = rng.standard_normal(11)
        assert abs(normal @ J @ arbitrary) <= 1e-12


def test_disjoint_construction_leaves_coordinates_nine_and_ten_unswirled(probe_points):
    for w in probe_points[:3]:
        J = construct_J_disjoint(w, P_CONSTRAINT, EPSILON_CONSTRAINT, 1.0)
        assert np.allclose(J[9:, :], 0.0)
        assert np.allclose(J[:, 9:], 0.0)
        assert not np.allclose(J[:9, :9], 0.0)


def test_overlapping_construction_involves_every_coordinate(probe_points):
    for w in probe_points[:3]:
        J = construct_J_overlapping(w, P_CONSTRAINT, EPSILON_CONSTRAINT, 1.0)
        assert np.all(np.abs(J).sum(axis=1) > 0.0)


def test_scaling_is_constant_and_not_state_dependent(probe_points):
    """Automatic check 11: J is exactly linear in swirl_scale."""
    for w in probe_points[:4]:
        base = construct_J_disjoint(w, P_CONSTRAINT, EPSILON_CONSTRAINT, 1.0)
        assert np.allclose(construct_J_disjoint(w, P_CONSTRAINT,
                                                EPSILON_CONSTRAINT, 3.0),
                           3.0 * base, atol=0, rtol=0)
        # the overlapping block scale is the constant 1/sqrt(n_triples)
        _, triples, block_scale = make_J_builder("overlapping", 11, P_CONSTRAINT,
                                                 EPSILON_CONSTRAINT, 1.0)
        assert np.isclose(block_scale, 1.0 / np.sqrt(len(triples)))
    # J is *not* normalised by any function of w: doubling grad_psi doubles J
    w = probe_points[0]
    grad_psi = grad_psi_constraint(w, P_CONSTRAINT, EPSILON_CONSTRAINT)
    J = construct_J_disjoint(w, P_CONSTRAINT, EPSILON_CONSTRAINT, 1.0)
    assert np.isclose(J[0, 1], -grad_psi[2])
    assert np.isclose(J[0, 2], grad_psi[1])
    assert np.isclose(J[1, 2], -grad_psi[0])


def test_analytic_operator_norm_matches_the_svd(probe_points):
    _, triples, block_scale = make_J_builder("disjoint", 11, P_CONSTRAINT,
                                             EPSILON_CONSTRAINT, 1.0)
    for w in probe_points:
        J = construct_J_disjoint(w, P_CONSTRAINT, EPSILON_CONSTRAINT, 1.0)
        grad_psi = grad_psi_constraint(w, P_CONSTRAINT, EPSILON_CONSTRAINT)
        analytic = operator_norm_J(J, triples, grad_psi, block_scale, 1.0)
        assert np.isclose(analytic, np.linalg.norm(J, ord=2))
