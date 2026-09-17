"""Sections 5 and 6: the smoothed l_p functional and the pilot threshold rule."""

from __future__ import annotations

import numpy as np
import pytest

from pnral.constraint import (choose_threshold_from_pilot, fixed_threshold,
                              g_constraint, g_minimum, grad_g_constraint,
                              grad_psi_constraint, outward_normal,
                              psi_constraint, validate_threshold)
from tests.conftest import EPSILON_CONSTRAINT, LAMBDA_CONSTRAINT, P_CONSTRAINT


def test_g_at_origin_equals_d_times_epsilon_to_the_p():
    """Automatic check 1."""
    for d in (3, 11, 25):
        expected = d * EPSILON_CONSTRAINT ** P_CONSTRAINT
        assert np.isclose(g_constraint(np.zeros(d), P_CONSTRAINT,
                                       EPSILON_CONSTRAINT), expected, rtol=0,
                          atol=1e-15)
        assert np.isclose(g_minimum(d, P_CONSTRAINT, EPSILON_CONSTRAINT), expected)


def test_origin_is_the_global_minimum():
    rng = np.random.default_rng(0)
    minimum = g_minimum(11, P_CONSTRAINT, EPSILON_CONSTRAINT)
    for _ in range(50):
        w = rng.standard_normal(11)
        assert g_constraint(w, P_CONSTRAINT, EPSILON_CONSTRAINT) >= minimum


def test_threshold_must_exceed_the_minimum():
    """Automatic check 2."""
    minimum = g_minimum(11, P_CONSTRAINT, EPSILON_CONSTRAINT)
    validate_threshold(minimum + 1e-6, 11, P_CONSTRAINT, EPSILON_CONSTRAINT)
    with pytest.raises(ValueError, match="must exceed"):
        validate_threshold(minimum, 11, P_CONSTRAINT, EPSILON_CONSTRAINT)


def test_grad_g_matches_central_differences():
    rng = np.random.default_rng(4)
    for _ in range(5):
        w = rng.standard_normal(11)
        analytic = grad_g_constraint(w, P_CONSTRAINT, EPSILON_CONSTRAINT)
        numeric = np.empty_like(analytic)
        for i in range(11):
            shift = np.zeros(11)
            shift[i] = 1e-6
            numeric[i] = (g_constraint(w + shift, P_CONSTRAINT, EPSILON_CONSTRAINT)
                          - g_constraint(w - shift, P_CONSTRAINT,
                                         EPSILON_CONSTRAINT)) / 2e-6
        assert np.allclose(numeric, analytic, rtol=1e-6, atol=1e-8)


def test_grad_psi_is_minus_grad_g_and_psi_is_the_slack():
    w = np.linspace(-1, 1, 11)
    assert np.allclose(grad_psi_constraint(w, P_CONSTRAINT, EPSILON_CONSTRAINT),
                       -grad_g_constraint(w, P_CONSTRAINT, EPSILON_CONSTRAINT))
    assert np.isclose(psi_constraint(w, P_CONSTRAINT, EPSILON_CONSTRAINT,
                                     LAMBDA_CONSTRAINT),
                      LAMBDA_CONSTRAINT - g_constraint(w, P_CONSTRAINT,
                                                       EPSILON_CONSTRAINT))


def test_outward_normal_is_a_unit_vector():
    w = np.linspace(-1, 1, 11)
    normal = outward_normal(w, P_CONSTRAINT, EPSILON_CONSTRAINT)
    assert np.isclose(np.linalg.norm(normal), 1.0)


def test_pilot_rule_inflates_the_quantile():
    rng = np.random.default_rng(9)
    pilot = 3.0 + 0.2 * rng.standard_normal(20000)
    selection = choose_threshold_from_pilot(pilot, 11, P_CONSTRAINT,
                                            EPSILON_CONSTRAINT)
    minimum = g_minimum(11, P_CONSTRAINT, EPSILON_CONSTRAINT)
    quantile = np.quantile(pilot, 0.999)
    assert np.isclose(selection.Lambda_constraint,
                      minimum + 1.10 * (quantile - minimum))
    assert selection.Lambda_constraint > quantile          # strictly beyond q999
    assert selection.exceedance_fraction <= 0.001
    assert (selection.Lambda_small < selection.Lambda_constraint
            < selection.Lambda_large)
    assert np.isclose(selection.Lambda_small,
                      minimum + 0.90 * (selection.Lambda_constraint - minimum))
    assert np.isclose(selection.Lambda_large,
                      minimum + 1.10 * (selection.Lambda_constraint - minimum))


def test_user_supplied_threshold_bypasses_the_pilot():
    selection = fixed_threshold(5.0, 11, P_CONSTRAINT, EPSILON_CONSTRAINT)
    assert selection.from_pilot is False
    assert selection.Lambda_constraint == 5.0
    assert selection.n_pilot_samples == 0
