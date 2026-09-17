"""Section 11: the Euclidean projection onto K and its KKT conditions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from pnral.constraint import g_constraint, g_minimum
from pnral.projection import kkt_residual, project_K, radial_projection
from tests.conftest import EPSILON_CONSTRAINT, LAMBDA_CONSTRAINT, P_CONSTRAINT


def _random_outside_point(rng, scale=1.6):
    while True:
        y = scale * rng.standard_normal(11)
        if g_constraint(y, P_CONSTRAINT, EPSILON_CONSTRAINT) > LAMBDA_CONSTRAINT:
            return y


def test_feasible_points_are_returned_unchanged():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = 0.1 * rng.standard_normal(11)
        result = project_K(y, P_CONSTRAINT, EPSILON_CONSTRAINT, LAMBDA_CONSTRAINT)
        assert result.projected is False
        assert result.distance == 0.0
        assert np.array_equal(result.z, y)


def test_projection_is_feasible_and_satisfies_kkt():
    """Automatic checks 3 and 9."""
    rng = np.random.default_rng(1)
    for _ in range(60):
        y = _random_outside_point(rng)
        result = project_K(y, P_CONSTRAINT, EPSILON_CONSTRAINT, LAMBDA_CONSTRAINT)
        residual = kkt_residual(y, result, P_CONSTRAINT, EPSILON_CONSTRAINT,
                                LAMBDA_CONSTRAINT)
        assert residual["g_after"] <= LAMBDA_CONSTRAINT + 1e-9
        assert residual["stationarity"] <= 1e-9
        assert residual["primal_violation"] == 0.0
        assert result.eta > 0.0                       # dual feasibility
        assert abs(residual["complementary_slackness"]) <= 1e-7
        assert np.all(np.sign(result.z) == np.sign(y))
        assert np.all(np.abs(result.z) <= np.abs(y) + 1e-12)


def test_projection_matches_a_general_purpose_optimiser():
    """The KKT solution is the true argmin of 0.5||z-y||^2 over K."""
    rng = np.random.default_rng(2)
    for _ in range(8):
        y = _random_outside_point(rng)
        result = project_K(y, P_CONSTRAINT, EPSILON_CONSTRAINT, LAMBDA_CONSTRAINT)
        reference = minimize(
            lambda z: 0.5 * np.sum((z - y) ** 2), 0.4 * y,
            jac=lambda z: z - y, method="SLSQP",
            constraints=[{"type": "ineq",
                          "fun": lambda z: LAMBDA_CONSTRAINT - g_constraint(
                              z, P_CONSTRAINT, EPSILON_CONSTRAINT)}],
            options={"ftol": 1e-14, "maxiter": 800})
        assert reference.success
        objective = 0.5 * np.sum((result.z - y) ** 2)
        assert objective <= reference.fun + 1e-8
        assert np.max(np.abs(result.z - reference.x)) < 1e-6


def test_brentq_and_vectorised_inner_solvers_agree():
    rng = np.random.default_rng(3)
    for _ in range(20):
        y = _random_outside_point(rng)
        fast = project_K(y, P_CONSTRAINT, EPSILON_CONSTRAINT, LAMBDA_CONSTRAINT,
                         use_brentq=False)
        literal = project_K(y, P_CONSTRAINT, EPSILON_CONSTRAINT, LAMBDA_CONSTRAINT,
                            use_brentq=True)
        assert np.max(np.abs(fast.z - literal.z)) < 1e-8


def test_p_equals_two_uses_exact_radial_projection():
    rng = np.random.default_rng(4)
    epsilon, threshold, d = 0.05, 3.0, 11
    radius = np.sqrt(threshold - d * epsilon ** 2)
    for _ in range(20):
        y = 2.0 * rng.standard_normal(d)
        result = project_K(y, 2.0, epsilon, threshold)
        expected = radial_projection(y, epsilon, threshold)
        assert np.allclose(result.z, expected)
        if np.linalg.norm(y) > radius:
            assert np.isclose(np.linalg.norm(result.z), radius)
            # radial: direction preserved exactly
            assert np.allclose(result.z / np.linalg.norm(result.z),
                               y / np.linalg.norm(y))
        assert g_constraint(result.z, 2.0, epsilon) <= threshold + 1e-10


def test_non_convex_exponents_are_rejected():
    with pytest.raises(ValueError, match="p_constraint >= 1"):
        project_K(np.ones(11), 0.5, EPSILON_CONSTRAINT, LAMBDA_CONSTRAINT)


def test_empty_constraint_set_is_rejected():
    minimum = g_minimum(11, P_CONSTRAINT, EPSILON_CONSTRAINT)
    with pytest.raises(ValueError, match="must exceed"):
        project_K(np.ones(11), P_CONSTRAINT, EPSILON_CONSTRAINT, minimum * 0.5)
