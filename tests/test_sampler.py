"""Section 10: the PNRAL update, its alpha = 0 limit and the shared setup."""

from __future__ import annotations

import numpy as np
import pytest

from pnral.anchor import log_anchor_coefficient, value_and_grad_U0
from pnral.constraint import g_constraint
from pnral.projection import project_K
from pnral.sampler import (JSpec, constrained_map, initial_points, run_chain,
                           run_chains, smooth_map)
from tests.conftest import EPSILON_CONSTRAINT, LAMBDA_CONSTRAINT, P_CONSTRAINT

STEP_SIZE = 2e-5


def _start(target):
    return project_K(smooth_map(target), P_CONSTRAINT, EPSILON_CONSTRAINT,
                     LAMBDA_CONSTRAINT).z


def test_every_state_satisfies_the_constraint(target, j_spec):
    """Automatic check 3."""
    chain = run_chain(target, _start(target), alpha=0.5, step_size=STEP_SIZE,
                      n_iterations=400, seed=5, j_spec=j_spec,
                      Lambda_constraint=LAMBDA_CONSTRAINT,
                      p_constraint=P_CONSTRAINT,
                      epsilon_constraint=EPSILON_CONSTRAINT, burn_in=100)
    values = g_constraint(chain.states, P_CONSTRAINT, EPSILON_CONSTRAINT)
    assert np.max(values) <= LAMBDA_CONSTRAINT + 1e-8
    assert np.allclose(values, chain.g)
    assert np.all(chain.slack >= -1e-8)


def test_no_nan_or_infinite_quantities(target, j_spec):
    """Automatic check 13."""
    chain = run_chain(target, _start(target), alpha=1.0, step_size=STEP_SIZE,
                      n_iterations=300, seed=6, j_spec=j_spec,
                      Lambda_constraint=LAMBDA_CONSTRAINT,
                      p_constraint=P_CONSTRAINT,
                      epsilon_constraint=EPSILON_CONSTRAINT, burn_in=50)
    for trace in (chain.states, chain.U, chain.U0, chain.log_a, chain.a,
                  chain.g, chain.slack, chain.grad_norm, chain.total_drift_norm,
                  chain.J_operator_norm, chain.projection_distance):
        assert np.all(np.isfinite(trace))


def test_alpha_zero_is_exactly_the_reversible_anchored_update(target, j_spec):
    """Automatic check 10: alpha = 0 removes the non-reversible drift only."""
    start = _start(target)
    chain = run_chain(target, start, alpha=0.0, step_size=STEP_SIZE,
                      n_iterations=150, seed=17, j_spec=j_spec,
                      Lambda_constraint=LAMBDA_CONSTRAINT,
                      p_constraint=P_CONSTRAINT,
                      epsilon_constraint=EPSILON_CONSTRAINT, burn_in=0)
    # hand-rolled reversible anchored Langevin with the same seed
    rng = np.random.default_rng(17)
    w = start.copy()
    for k in range(150):
        assert np.allclose(w, chain.states[k], atol=0, rtol=0)
        _, gradient = value_and_grad_U0(target, w)
        a = np.exp(log_anchor_coefficient(target, w))
        xi = rng.standard_normal(target.d)
        # the same floating-point association as the sampler, so the two
        # trajectories must agree bit-for-bit and not merely to a tolerance
        reversible_drift = -a * gradient
        proposal = (w + STEP_SIZE * reversible_drift
                    + np.sqrt(2.0 * STEP_SIZE * a) * xi)
        w = project_K(proposal, P_CONSTRAINT, EPSILON_CONSTRAINT,
                      LAMBDA_CONSTRAINT).z
    assert np.all(chain.nonreversible_drift_norm == 0.0)
    assert np.allclose(chain.total_drift_norm, chain.reversible_drift_norm)


def test_drift_decomposition_matches_the_formula(target, j_spec):
    """total drift = -a (I + alpha J) grad U0, recorded norms included."""
    start = _start(target)
    alpha = 0.75
    chain = run_chain(target, start, alpha=alpha, step_size=STEP_SIZE,
                      n_iterations=3, seed=8, j_spec=j_spec,
                      Lambda_constraint=LAMBDA_CONSTRAINT,
                      p_constraint=P_CONSTRAINT,
                      epsilon_constraint=EPSILON_CONSTRAINT, burn_in=0)
    for k in range(3):
        w = chain.states[k]
        _, gradient = value_and_grad_U0(target, w)
        a = np.exp(log_anchor_coefficient(target, w))
        J = j_spec.builder(w)
        reversible = -a * gradient
        nonreversible = -alpha * a * (J @ gradient)
        total = -a * (np.eye(target.d) + alpha * J) @ gradient
        assert np.allclose(total, reversible + nonreversible)
        assert np.isclose(chain.reversible_drift_norm[k],
                          np.linalg.norm(reversible))
        assert np.isclose(chain.nonreversible_drift_norm[k],
                          np.linalg.norm(nonreversible))
        assert np.isclose(chain.total_drift_norm[k], np.linalg.norm(total))


def test_U_trace_equals_U0_plus_log_a(target, j_spec):
    chain = run_chain(target, _start(target), alpha=0.25, step_size=STEP_SIZE,
                      n_iterations=50, seed=9, j_spec=j_spec,
                      Lambda_constraint=LAMBDA_CONSTRAINT,
                      p_constraint=P_CONSTRAINT,
                      epsilon_constraint=EPSILON_CONSTRAINT, burn_in=0)
    assert np.allclose(chain.U, chain.U0 + chain.log_a)
    assert np.allclose(chain.a, np.exp(chain.log_a))


def test_all_alphas_share_constraint_seeds_and_initial_points(target, j_spec):
    """Automatic check 12 plus the common-random-numbers requirement."""
    start_points = initial_points(_start(target), 2, 0.02, 123, P_CONSTRAINT,
                                  EPSILON_CONSTRAINT, LAMBDA_CONSTRAINT)
    runs = [run_chains(target, start_points, alpha=alpha, step_size=STEP_SIZE,
                       n_iterations=60, j_spec=j_spec,
                       Lambda_constraint=LAMBDA_CONSTRAINT,
                       p_constraint=P_CONSTRAINT,
                       epsilon_constraint=EPSILON_CONSTRAINT, burn_in=10,
                       thinning=1, chain_seed_base=1000, label=f"alpha={alpha}")
            for alpha in (0.0, 0.5, 1.0)]
    assert len({run.Lambda_constraint for run in runs}) == 1
    for run in runs:
        assert [chain.seed for chain in run.chains] == [1000, 1001]
        for chain, start in zip(run.chains, start_points):
            assert np.array_equal(chain.states[0], start)


def test_initial_points_are_feasible_and_reproducible(target):
    info = constrained_map(target, P_CONSTRAINT, EPSILON_CONSTRAINT,
                           LAMBDA_CONSTRAINT)
    assert g_constraint(info["w_map_constrained"], P_CONSTRAINT,
                        EPSILON_CONSTRAINT) <= LAMBDA_CONSTRAINT + 1e-9
    first = initial_points(info["w_map_constrained"], 4, 0.02, 77, P_CONSTRAINT,
                           EPSILON_CONSTRAINT, LAMBDA_CONSTRAINT)
    second = initial_points(info["w_map_constrained"], 4, 0.02, 77, P_CONSTRAINT,
                            EPSILON_CONSTRAINT, LAMBDA_CONSTRAINT)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert len({tuple(point) for point in first}) == 4      # genuinely different
    for point in first:
        assert g_constraint(point, P_CONSTRAINT, EPSILON_CONSTRAINT) \
            <= LAMBDA_CONSTRAINT + 1e-9


def test_chain_is_reproducible_from_its_seed(target, j_spec):
    kwargs = dict(w0=_start(target), alpha=0.5, step_size=STEP_SIZE,
                  n_iterations=40, seed=321, j_spec=j_spec,
                  Lambda_constraint=LAMBDA_CONSTRAINT, p_constraint=P_CONSTRAINT,
                  epsilon_constraint=EPSILON_CONSTRAINT, burn_in=0)
    first = run_chain(target, **kwargs)
    second = run_chain(target, **kwargs)
    assert np.array_equal(first.states, second.states)


def test_gradient_evaluations_are_counted(target, j_spec):
    before = target.gradient_evaluations
    chain = run_chain(target, _start(target), alpha=0.0, step_size=STEP_SIZE,
                      n_iterations=25, seed=1, j_spec=j_spec,
                      Lambda_constraint=LAMBDA_CONSTRAINT,
                      p_constraint=P_CONSTRAINT,
                      epsilon_constraint=EPSILON_CONSTRAINT, burn_in=0)
    assert chain.gradient_evaluations == 25
    assert target.gradient_evaluations >= before + 25


def test_unconstrained_pilot_skips_the_projection(target, j_spec):
    chain = run_chain(target, _start(target), alpha=0.0, step_size=STEP_SIZE,
                      n_iterations=80, seed=2, j_spec=j_spec,
                      Lambda_constraint=None, p_constraint=P_CONSTRAINT,
                      epsilon_constraint=EPSILON_CONSTRAINT, burn_in=0,
                      constrained=False)
    assert not np.any(chain.projected)
    assert np.all(np.isnan(chain.slack))
    assert np.all(np.isfinite(chain.g))


def test_overlapping_extension_runs_and_stays_feasible(target):
    spec = JSpec.create("overlapping", target.d, P_CONSTRAINT,
                        EPSILON_CONSTRAINT, 1.0)
    assert spec.is_disjoint is False
    chain = run_chain(target, _start(target), alpha=0.5, step_size=STEP_SIZE,
                      n_iterations=120, seed=3, j_spec=spec,
                      Lambda_constraint=LAMBDA_CONSTRAINT,
                      p_constraint=P_CONSTRAINT,
                      epsilon_constraint=EPSILON_CONSTRAINT, burn_in=20)
    assert np.max(chain.g) <= LAMBDA_CONSTRAINT + 1e-8
    assert np.all(np.isfinite(chain.states))
