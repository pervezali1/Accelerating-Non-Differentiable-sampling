"""Sections 3 and 4: the potential U, the anchor U0 and the coefficient a."""

from __future__ import annotations

import numpy as np

from pnral.anchor import (anchor_bounds, anchor_coefficient, grad_U0,
                          log_anchor_coefficient, potential_U0,
                          value_and_grad_U0)


def test_U_equals_U0_plus_log_a(target, probe_points):
    for w in probe_points:
        U = target.potential_U(w)
        U0 = potential_U0(target, w)
        assert np.isclose(U, U0 + log_anchor_coefficient(target, w), rtol=0, atol=1e-9)


def test_grad_U0_matches_central_differences(target, probe_points):
    step = 1e-6
    for w in probe_points[:4]:
        analytic = grad_U0(target, w)
        numeric = np.empty_like(analytic)
        for i in range(target.d):
            shift = np.zeros_like(w)
            shift[i] = step
            numeric[i] = (potential_U0(target, w + shift)
                          - potential_U0(target, w - shift)) / (2 * step)
        assert np.max(np.abs(numeric - analytic)) / np.max(np.abs(analytic)) < 1e-7


def test_value_and_grad_agree_with_separate_calls(target, probe_points):
    for w in probe_points[:3]:
        value, gradient = value_and_grad_U0(target, w)
        assert np.isclose(value, potential_U0(target, w))
        assert np.allclose(gradient, grad_U0(target, w))


def test_anchor_coefficient_bounds(target, probe_points):
    """Automatic check 7: exp(-10*lambda*delta) <= a(w) <= 1."""
    log_a_min, log_a_max, a_min, a_max = anchor_bounds(target)
    assert np.isclose(log_a_min, -10 * target.lambda_lasso * target.delta_anchor)
    for w in probe_points + [np.zeros(target.d), 50 * np.ones(target.d)]:
        log_a = log_anchor_coefficient(target, w)
        a = anchor_coefficient(target, w)
        assert log_a_min - 1e-12 <= log_a <= log_a_max + 1e-12
        assert a_min - 1e-12 <= a <= a_max + 1e-12
        assert np.isclose(a, np.exp(log_a))


def test_log_a_is_not_computed_as_a_ratio_of_exponentials(target):
    """A huge |U| would overflow exp(U)/exp(U0); log a stays finite."""
    w = 40.0 * np.ones(target.d)
    assert np.isfinite(log_anchor_coefficient(target, w))
    assert np.isfinite(anchor_coefficient(target, w))


def test_intercept_excluded_from_l1_penalty(target):
    """Automatic check 8."""
    intercept_only = np.zeros(target.d)
    intercept_only[0] = 7.0
    slope_only = np.zeros(target.d)
    slope_only[1] = 7.0
    assert target.l1_penalty(intercept_only) == 0.0
    assert np.isclose(target.l1_penalty(slope_only), 7.0 * target.lambda_lasso)
    # ... and the anchor gradient of the intercept is the Gaussian prior term
    gradient = grad_U0(target, intercept_only)
    likelihood_part = target.X.T @ (target.predict_probabilities(intercept_only)
                                    - target.y)
    assert np.isclose(gradient[0] - likelihood_part[0],
                      7.0 / target.sigma_intercept ** 2)


def test_likelihood_uses_a_sum_not_an_average(target):
    w = np.zeros(target.d)
    # at w = 0 every z_i = 0: softplus(0) - y*0 = log 2 per observation
    assert np.isclose(target.negative_log_likelihood(w), target.n * np.log(2.0))
